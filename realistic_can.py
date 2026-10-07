"""More realistic synthetic CAN traffic for the CARLA co-simulation.

What it adds over the simple model in carla_cl_ids.py:
  * ~30 fixed message IDs (powertrain, chassis, body, gateway, diagnostics), the same on every run like one vehicle
  * per-ECU clock skew and slow drift, timing jitter, rare scheduler hiccups
  * event-driven frames (brake edge, gear change, light change, turn signal, wipers, diagnostic bursts)
  * payloads derived from the vehicle state with sensor noise and quantisation
  * a 500 kbit/s bus with priority arbitration and frame transmission time; one TX mailbox per ID (a new instance
    overwrites an unsent older one), so DoS/Fuzzy delay or displace legitimate frames instead of an assumed 50% drop
The first five IDs keep the same signal layout as the simple model so that the plausibility features still apply.
"""
import math
import random
import struct

BITRATE = 500_000.0
ATTACK_QUEUE = 6
SPOOF_TARGETS = {"speed": 0x0B0, "brake": 0x0C0, "steer": 0x0A0}


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def xor_chk(body):
    x = 0
    for b in body:
        x ^= b
    return x


def encode_primary(cid, st, cnt, noise=None):
    """Same layouts as carla_cl_ids.encode (0A0, 0B0, 0C0, 2F0 with checksum+counter; 1B0 wheel speeds)."""
    if cid == 0x0A0:
        body = struct.pack("<hh2x", int(round(clip(st["steer"], -1, 1) * 1000)), int(round(clip(st["yaw_rate"], -300, 300) * 100)))
    elif cid == 0x0B0:
        body = struct.pack("<Hhh", int(clip(st["speed_kmh"], 0, 650) * 100), int(clip(st["ax"], -300, 300) * 100), int(clip(st["ay"], -300, 300) * 100))
    elif cid == 0x0C0:
        body = struct.pack("<BBbB2x", int(clip(st["throttle"], 0, 1) * 255), int(clip(st["brake"], 0, 1) * 255), int(clip(st["gear"], -1, 8)), int(st["handbrake"]))
    elif cid == 0x2F0:
        body = struct.pack("<BB4x", int(st["lights"]) & 0xFF, (int(st["lights"]) >> 8) & 0xFF)
    elif cid == 0x1B0:
        w = [int(clip(st["speed_kmh"] + d, 0, 650) * 100) for d in st["wheel_noise"]]
        return struct.pack("<4H", *w)
    else:
        raise ValueError(cid)
    return body + bytes([xor_chk(body), cnt & 0xFF])


def tamper(st, target):
    s = dict(st)
    if target == "speed":
        s["speed_kmh"] = st["speed_kmh"] * 0.4
    elif target == "brake":
        s["brake"], s["throttle"] = 1.0, 0.0
    elif target == "steer":
        s["steer"] = clip(st["steer"] + 0.35, -1, 1)
    return s


class Msg:
    def __init__(self, cid, name, period, ecu, counter=False, jitter=None):
        self.id, self.name, self.period, self.ecu, self.counter = cid, name, period, ecu, counter
        self.jitter = jitter if jitter is not None else (40e-6 if period <= 0.02 else 150e-6)
        self.next_ready = 0.0
        self.cnt = 0


# (id, name, period s, ECU, counter?)
TABLE = [
    (0x0A0, "STEER", 0.02, "EPS", True), (0x0B0, "SPEED", 0.02, "ABS", True), (0x0C0, "PEDALS", 0.05, "ECM", True),
    (0x1B0, "WHEELS", 0.02, "ABS", False), (0x2F0, "STATUS", 0.10, "BCM", True),
    (0x120, "ENGINE1", 0.01, "ECM", False), (0x130, "ENGINE2", 0.10, "ECM", False), (0x140, "TRANS", 0.02, "TCM", False),
    (0x150, "BRAKE", 0.02, "ABS", False), (0x160, "STEER_TQ", 0.01, "EPS", False), (0x170, "IMU", 0.01, "ABS", False),
    (0x180, "EPS_STAT", 0.05, "EPS", False), (0x190, "ACC", 0.02, "ADAS", False), (0x1A0, "LKA", 0.05, "ADAS", False),
    (0x1D0, "BATTERY", 0.10, "BMS", False), (0x200, "LIGHTS", 0.10, "BCM", False), (0x210, "TURN", 1.0, "BCM", False),
    (0x220, "DOORS", 0.20, "BCM", False), (0x230, "WIPER", 0.50, "BCM", False), (0x240, "HVAC", 0.20, "HVAC", False),
    (0x250, "TPMS", 1.0, "BCM", False), (0x260, "SAFETY", 0.20, "SRS", False), (0x270, "CLUSTER", 0.10, "IC", False),
    (0x280, "GW_HB", 0.10, "GW", False), (0x290, "NM", 1.0, "GW", False), (0x2A0, "ODO", 0.50, "IC", False),
    (0x300, "TIME", 1.0, "GW", False),
]


class RealisticCan:
    def __init__(self, seed, dos_dt=0.0003, fuzzy_dt=0.0005):
        self.rng = random.Random(seed * 7919 + 13)
        self.dos_dt, self.fuzzy_dt = dos_dt, fuzzy_dt
        r = self.rng
        self.msgs = [Msg(*row) for row in TABLE]
        self.ecu_skew = {e: r.gauss(0, 35.0) for e in {m.ecu for m in self.msgs}}  # ppm
        for m in self.msgs:
            m.next_ready = r.uniform(0, m.period)
        self.pend = []          # frames waiting for the bus: dicts
        self.bus_free = 0.0
        self.tx_count = 0
        self.dropped = 0
        self.prev = None
        self.coolant, self.oil, self.odo, self.fuel = 62.0, 55.0, 15234.0, 0.0
        self.next_diag = r.uniform(15, 40)
        self.diag_left = []
        self.next_wiper = r.uniform(10, 40)
        self.blink_state = 0
        self.next_blink = 0.0
        self.last_t = None

    # ------------------------------------------------------------------ payloads
    def _payload(self, m, st, t):
        r = self.rng
        sp, th, br = st["speed_kmh"], st["throttle"], st["brake"]
        gear = max(int(st["gear"]), 0)
        if m.id in (0x0A0, 0x0B0, 0x0C0, 0x2F0, 0x1B0):
            s2 = dict(st)
            s2["speed_kmh"] = max(0.0, sp + r.gauss(0, 0.03))
            return encode_primary(m.id, s2, m.cnt)
        if m.id == 0x120:
            rpm = clip(800 + sp * 28.0 / max(gear, 1) ** 0.5 + th * 900 + r.gauss(0, 15), 650, 6500)
            return struct.pack("<HhB3x", int(rpm * 4), int(th * 220 + r.gauss(0, 3)), int(th * 255))
        if m.id == 0x130:
            fuel = 1.2 + th * 28.0 + sp * 0.04
            return struct.pack("<BBH2x", int(clip(self.coolant + 40, 0, 255)), int(clip(self.oil + 40, 0, 255)), int(fuel * 10)) + bytes(2)
        if m.id == 0x140:
            return struct.pack("<bHB4x", gear, int(sp * 36.0), 1 if sp > 0.5 else 0)
        if m.id == 0x150:
            return struct.pack("<HBB4x", int(clip(br * 1200 + r.gauss(0, 4), 0, 2000)), 0, 1 if br > 0.05 else 0)
        if m.id == 0x160:
            d = 0.0 if self.prev is None else (st["steer"] - self.prev["steer"]) / max(st.get("dt", 0.05), 1e-3)
            return struct.pack("<hh4x", int(clip(d * 150 + r.gauss(0, 6), -3000, 3000)), int(clip(st["steer"] * 400 + r.gauss(0, 3), -3000, 3000)))
        if m.id == 0x170:
            return struct.pack("<hhh2x", int((st["ax"] + r.gauss(0, 0.02)) * 100), int((st["ay"] + r.gauss(0, 0.02)) * 100), int((st["yaw_rate"] + r.gauss(0, 0.05)) * 100))
        if m.id == 0x180:
            return struct.pack("<BB6x", 1, 0)
        if m.id == 0x190:
            return struct.pack("<HBB4x", int(100 * 100), 3, 1 if sp > 20 else 0)
        if m.id == 0x1A0:
            return struct.pack("<hB5x", int(st["steer"] * 100), 1)
        if m.id == 0x1D0:
            return struct.pack("<HhB3x", int((13.8 + 0.12 * math.sin(t * 0.7) + r.gauss(0, 0.02)) * 100), int((8 + th * 25 + r.gauss(0, 0.3)) * 10), 0)
        if m.id == 0x200:
            return struct.pack("<H6x", int(st["lights"]) & 0xFFFF)
        if m.id == 0x210:
            return struct.pack("<BB6x", self.blink_state, 1 if (int(st["lights"]) & 48) else 0)
        if m.id == 0x220:
            return bytes(8)
        if m.id == 0x230:
            return struct.pack("<BB6x", 0, 0)
        if m.id == 0x240:
            return struct.pack("<BBH4x", int(21 * 2), int(34 + 0.01 * t) & 0xFF, 400)
        if m.id == 0x250:
            return struct.pack("<4B4x", *[int(clip(230 + r.gauss(0, 1), 0, 255)) for _ in range(4)])
        if m.id == 0x260:
            return struct.pack("<BB6x", 1, 0)
        if m.id == 0x270:
            return struct.pack("<HI2x", int(sp * 100), int(self.odo * 10) & 0xFFFFFFFF)
        if m.id == 0x280:
            return struct.pack("<B7x", m.cnt & 0xFF)
        if m.id == 0x290:
            return struct.pack("<BB6x", 0x21, m.cnt & 0xFF)
        if m.id == 0x2A0:
            return struct.pack("<I4x", int(self.odo * 10) & 0xFFFFFFFF)
        if m.id == 0x300:
            return struct.pack("<I4x", int(t) & 0xFFFFFFFF)
        raise ValueError(m.id)

    # ------------------------------------------------------------------ frame generation for one tick
    def _gen(self, t0, t1, st, ep):
        r = self.rng
        kind = ep["can"] if ep else None
        in_win = lambda ts: ep is not None and ep["start"] <= ts < ep["end"]
        out = []
        dt = (t0 - self.last_t) if self.last_t is not None else 0.05
        st = dict(st, dt=dt)
        self.coolant += (90.0 - self.coolant) * dt / 300.0
        self.oil += (95.0 - self.oil) * dt / 420.0
        self.odo += st["speed_kmh"] / 3.6 * dt / 1000.0
        for e in self.ecu_skew:
            self.ecu_skew[e] = clip(self.ecu_skew[e] + r.gauss(0, 0.15) * math.sqrt(dt), -120, 120)
        lights = int(st["lights"])
        prev = self.prev
        events = []
        if prev is not None:
            if (st["brake"] > 0.05) != (prev["brake"] > 0.05):
                events.append(0x150)
            if int(st["gear"]) != int(prev["gear"]):
                events.append(0x140)
            if lights != int(prev["lights"]):
                events.append(0x200)
        if t0 >= self.next_wiper:
            events.append(0x230)
            self.next_wiper = t0 + r.uniform(25, 70)
        blinker = bool(lights & 48)
        if blinker and t0 >= self.next_blink:
            self.blink_state ^= 1
            events.append(0x210)
            self.next_blink = t0 + 0.33
        if not blinker:
            self.blink_state = 0

        for m in self.msgs:
            while m.next_ready < t1:
                ready = max(m.next_ready, t0)
                if r.random() < 0.003:
                    ready += r.uniform(0.0003, 0.002)
                payload = self._payload(m, st, ready)
                label, atype = 0, ""
                spoof_here = kind in ("spoof_plain", "spoof_stealth") and in_win(ready) and m.id == SPOOF_TARGETS[ep["spoof_target"]]
                if spoof_here:
                    bad = encode_primary(m.id, tamper(dict(st), ep["spoof_target"]), m.cnt)
                    if kind == "spoof_stealth":
                        payload, label, atype = bad, 1, kind
                    else:
                        out.append({"ready": ready + r.uniform(0.00008, 0.0004), "id": m.id, "payload": bad, "label": 1, "atype": kind, "atk": True})
                out.append({"ready": ready, "id": m.id, "payload": payload, "label": label, "atype": atype, "atk": False})
                m.cnt += 1
                skew = self.ecu_skew[m.ecu] * 1e-6
                m.next_ready += m.period * (1.0 + skew) + r.gauss(0, m.jitter)
        for cid in events:
            m = next(x for x in self.msgs if x.id == cid)
            out.append({"ready": t0 + r.uniform(0, 0.003), "id": cid, "payload": self._payload(m, st, t0), "label": 0, "atype": "", "atk": False})
            m.cnt += 1
        # diagnostics: request/response bursts
        if t0 >= self.next_diag:
            n = r.randint(1, 4)
            for k in range(n):
                tt = t0 + k * r.uniform(0.008, 0.02)
                svc = r.choice([0x0100, 0x010C, 0x010D, 0x0105, 0x0110])
                out.append({"ready": tt, "id": 0x7DF, "payload": struct.pack("<BBHxxxx", 2, 1, svc & 0xFF), "label": 0, "atype": "", "atk": False})
                out.append({"ready": tt + r.uniform(0.002, 0.006), "id": 0x7E8, "payload": struct.pack("<BBBHx", 4, 0x41, svc & 0xFF, r.randint(0, 65535)) + bytes(1),
                            "label": 0, "atype": "", "atk": False})
            self.next_diag = t0 + r.uniform(20, 60)
        # injection attacks
        if kind in ("dos", "fuzzy"):
            step = self.dos_dt if kind == "dos" else self.fuzzy_dt
            lo, hi = max(t0, ep["start"]), min(t1, ep["end"])
            k0 = math.ceil((lo - ep["start"]) / step)
            ts = ep["start"] + k0 * step
            while ts < hi:
                if kind == "dos":
                    out.append({"ready": ts + r.gauss(0, 8e-6), "id": 0x000, "payload": bytes(8), "label": 1, "atype": kind, "atk": True})
                else:
                    out.append({"ready": ts, "id": r.randint(0, 0x7FF), "payload": bytes(r.randint(0, 255) for _ in range(8)), "label": 1, "atype": kind, "atk": True})
                ts += step
        self.prev = st
        self.last_t = t0
        return out

    # ------------------------------------------------------------------ bus arbitration
    def _tx_time(self, cid, payload):
        """Frame length from the real bit stream (SOF, ID, control, data, random CRC) with bit stuffing, plus the fixed tail."""
        r = self.rng
        bits = [0] + [(cid >> (10 - i)) & 1 for i in range(11)] + [0, 0, 0] + [(len(payload) >> (3 - i)) & 1 for i in range(4)]
        for b in payload:
            bits += [(b >> (7 - i)) & 1 for i in range(8)]
        bits += [r.getrandbits(1) for _ in range(15)]
        n, run, last = 0, 0, None
        for b in bits:
            n += 1
            if b == last:
                run += 1
            else:
                run, last = 1, b
            if run == 5:
                n += 1
                last, run = 1 - b, 1
        return (n + 1 + 2 + 7 + 3) / BITRATE

    def step(self, t0, t1, st, ep):
        self.pend.extend(self._gen(t0, t1, st, ep))
        self.pend.sort(key=lambda f: f["ready"])
        out = []
        while self.pend:
            t = max(self.bus_free, self.pend[0]["ready"])
            if t >= t1:
                break
            elig = [f for f in self.pend if f["ready"] <= t]
            # one TX mailbox per ID for legitimate ECUs: a newer instance overwrites an unsent older one
            latest = {}
            for f in elig:
                if not f["atk"]:
                    latest[f["id"]] = f
            drop = [f for f in elig if not f["atk"] and latest[f["id"]] is not f]
            atk = [f for f in elig if f["atk"]]
            if len(atk) > ATTACK_QUEUE:
                drop += atk[:-ATTACK_QUEUE]
            if drop:
                ids = {id(f) for f in drop}
                self.pend = [f for f in self.pend if id(f) not in ids]
                self.dropped += sum(1 for f in drop if not f["atk"])
                elig = [f for f in elig if id(f) not in ids]
            win = min(elig, key=lambda f: (f["id"], not f["atk"], f["ready"]))
            self.pend.remove(win)
            end = t + self._tx_time(win["id"], win["payload"])
            self.bus_free = end
            self.tx_count += 1
            out.append((end, win["id"], win["payload"], win["label"], win["atype"]))
        return out


if __name__ == "__main__":
    # offline sanity check without CARLA: a 60 s drive with a speed profile and some braking
    import statistics
    sim = RealisticCan(1)
    dt = 0.05
    frames = []
    ep = None
    rr = random.Random(3)
    for k in range(1200):
        t = k * dt
        sp = 35 + 25 * math.sin(t / 9.0)
        st = dict(steer=0.05 * math.sin(t / 4.0), yaw_rate=3 * math.sin(t / 4.0), speed_kmh=sp, ax=0.3 * math.cos(t / 9.0), ay=0.2, throttle=0.3 if math.cos(t / 9.0) > 0 else 0.0,
                  brake=0.4 if math.cos(t / 9.0) < -0.6 else 0.0, gear=3, handbrake=0, lights=0, wheel_noise=[rr.gauss(0, 0.15) for _ in range(4)])
        if 30 <= t < 40:
            ep = {"can": "dos", "start": 30.0, "end": 40.0, "spoof_target": "speed"}
        else:
            ep = None
        frames += [(k, *f) for f in sim.step(t, t + dt, st, ep)]
    by_id = {}
    for k, ts, cid, pl, lab, at in frames:
        by_id.setdefault(cid, []).append(ts)
    normal = [f for f in frames if not (30 <= f[1] < 40) and f[1] < 60]
    print("IDs:", len(by_id), "| frames/s clean:", round(len([f for f in frames if f[1] < 30]) / 30.0), "| dropped legit:", sim.dropped)
    print("bus load clean ~", round(len([f for f in frames if f[1] < 30]) / 30.0 * 0.00024 * 100, 1), "%")
    for cid in (0x0A0, 0x0B0, 0x120, 0x130, 0x7DF):
        ts = [x for x in by_id.get(cid, []) if x < 30]
        d = [b - a for a, b in zip(ts, ts[1:])]
        if d:
            print(f"  0x{cid:03X}: n={len(ts)} mean IAT={statistics.mean(d) * 1000:.3f} ms  std={statistics.pstdev(d) * 1000:.3f} ms")
    dos = [f for f in frames if 30 <= f[1] < 40]
    leg = [f for f in dos if f[2] != 0]
    print("during DoS: frames/s", round(len(dos) / 10.0), "| legit frames/s", round(len(leg) / 10.0), "(clean:", round(len([f for f in frames if f[1] < 30 and f[2] != 0]) / 30.0), ")")
