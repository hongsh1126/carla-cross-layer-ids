#!/usr/bin/env python
"""Cross-layer IDS dataset generator for CARLA 0.9.16 (simulation only).

Co-simulation: CARLA (synchronous, 20 Hz) + a one-directional CAN bridge
(vehicle state -> synthetic CAN frames, written to a log; nothing is sent to a real bus).

Scenarios
  S0 normal            no attacks
  S1 can_only          DoS / Fuzzy / Spoof (plain) / Spoof (stealth, rate-preserving)
  S2 sensor_only       GPS drift / LiDAR phantom / LiDAR hide / camera phantom / camera hide
                       (CAN stays normal -> "stealth" w.r.t. network-only IDS)
  S3 combined          one CAN attack and one sensor attack at the same time

Outputs per run (out/<scenario>_s<seed>/): can_log.csv, sensor_log.csv, meta.json
Conventions: RANGE_CAP (100 m) is the "no obstacle in corridor" sentinel for lidar/camera range.
"""
import argparse
import csv
import json
import math
import os
import queue
import random
import struct
import sys

import numpy as np

import carla

DT = 0.05
CAM_W, CAM_H, CAM_FOV = 320, 180, 90.0
CAM_F = CAM_W / (2.0 * math.tan(math.radians(CAM_FOV) / 2.0))
VEHICLE_TAGS = (14, 15, 16, 18, 19)
CORR_HALF_W = 1.5
RANGE_CAP = 100.0
SENSOR_X = 0.8
M_PER_DEG = 111320.0
CAM_MIN_DEPTH = 3.0  # the ego car's own body is visible to the camera at ~1.1 m

CAN_ATTACKS = ("dos", "fuzzy", "spoof_plain", "spoof_stealth")
SENSOR_ATTACKS = ("gps_drift", "lidar_phantom", "lidar_hide", "cam_phantom", "cam_hide")
CHK_IDS = (0x0A0, 0x0B0, 0x0C0, 0x2F0)
SPOOF_TARGETS = {"speed": 0x0B0, "brake": 0x0C0, "steer": 0x0A0}

_U = np.tile(np.arange(CAM_W, dtype=np.float32), (CAM_H, 1))


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def encode(cid, st, cnt):
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
    x = 0
    for b in body:
        x ^= b
    return body + bytes([x, cnt & 0xFF])


def tamper(st, target):
    s = dict(st)
    if target == "speed":
        s["speed_kmh"] = st["speed_kmh"] * 0.4
    elif target == "brake":
        s["brake"], s["throttle"] = 1.0, 0.0
    elif target == "steer":
        s["steer"] = clip(st["steer"] + 0.35, -1, 1)
    return s


class CanSim:
    PERIODS = {0x0A0: 0.02, 0x0B0: 0.02, 0x0C0: 0.05, 0x1B0: 0.02, 0x2F0: 0.1}

    def __init__(self, rng, n_bg=12, drop_p=0.5, dos_dt=0.0005, fuzzy_dt=0.001):
        self.rng, self.drop_p, self.dos_dt, self.fuzzy_dt = rng, drop_p, dos_dt, fuzzy_dt
        self.cnt = {i: 0 for i in self.PERIODS}
        self.due = {i: rng.uniform(0, p) for i, p in self.PERIODS.items()}
        ids = [i for i in rng.sample(range(0x100, 0x700), n_bg + 8) if i not in self.PERIODS][:n_bg]
        self.bg = []
        for i in ids:
            period = rng.choice([0.01, 0.02, 0.05, 0.1])
            self.bg.append({"id": i, "p": period, "due": rng.uniform(0, period),
                            "b": [rng.randint(0, 255) for _ in range(8)], "act": rng.sample(range(8), 3)})

    def step(self, t0, t1, st, ep):
        """Frames in [t0, t1) built from the vehicle state sampled at t0. ep: active CAN episode or None."""
        rng = self.rng
        kind = ep["can"] if ep else None
        in_win = lambda ts: ep is not None and ep["start"] <= ts < ep["end"]
        out = []
        for cid, per in self.PERIODS.items():
            while self.due[cid] < t1:
                ts = max(t0, self.due[cid] + rng.gauss(0, 1e-4))
                cnt = self.cnt[cid]
                payload = encode(cid, st, cnt)
                self.cnt[cid] += 1
                self.due[cid] += per
                if kind == "dos" and in_win(ts) and rng.random() < self.drop_p:
                    continue
                if kind in ("spoof_plain", "spoof_stealth") and in_win(ts) and cid == SPOOF_TARGETS[ep["spoof_target"]]:
                    bad = encode(cid, tamper(st, ep["spoof_target"]), cnt)
                    if kind == "spoof_stealth":
                        out.append((ts, cid, bad, 1, kind))
                    else:
                        out.append((ts, cid, payload, 0, ""))
                        out.append((ts + 5e-4, cid, bad, 1, kind))
                    continue
                out.append((ts, cid, payload, 0, ""))
        for g in self.bg:
            while g["due"] < t1:
                ts = max(t0, g["due"] + rng.gauss(0, 1e-4))
                for j in g["act"]:
                    g["b"][j] = int(clip(g["b"][j] + rng.randint(-2, 2), 0, 255))
                g["due"] += g["p"]
                if kind == "dos" and in_win(ts) and rng.random() < self.drop_p:
                    continue
                out.append((ts, g["id"], bytes(g["b"]), 0, ""))
        if kind in ("dos", "fuzzy"):
            lo, hi = max(t0, ep["start"]), min(t1, ep["end"])
            dt = self.dos_dt if kind == "dos" else self.fuzzy_dt
            k0 = math.ceil((lo - ep["start"]) / dt)
            ts = ep["start"] + k0 * dt
            while ts < hi:
                if kind == "dos":
                    out.append((ts, 0x000, bytes(8), 1, kind))
                else:
                    out.append((ts, rng.randint(0, 0x7FF), bytes(rng.randint(0, 255) for _ in range(8)), 1, kind))
                ts += dt
        out.sort(key=lambda r: r[0])
        return out


def build_schedule(scenario, rng, warmup, ep_min, ep_max, gap_min, gap_max):
    cans, sens = list(CAN_ATTACKS), list(SENSOR_ATTACKS)
    rng.shuffle(cans)
    rng.shuffle(sens)
    if scenario == "S0":
        plan = []
    elif scenario == "S1":
        plan = [(c, None) for c in cans]
    elif scenario == "S2":
        plan = [(None, s) for s in sens]
    elif scenario == "S3":
        plan = [(cans[i % len(cans)], s) for i, s in enumerate(sens)]
    else:
        raise ValueError(scenario)
    eps, t = [], warmup
    for c, s in plan:
        dur = rng.uniform(ep_min, ep_max)
        ep = {"start": round(t, 3), "end": round(t + dur, 3), "can": c, "sensor": s,
              "spoof_target": rng.choice(list(SPOOF_TARGETS)),
              "gps_theta": rng.uniform(0, 2 * math.pi), "gps_rate": rng.uniform(0.8, 1.5), "gps_cap": 40.0,
              "R_lidar": rng.uniform(8, 20), "R_cam": rng.uniform(8, 20)}
        eps.append(ep)
        t += dur + rng.uniform(gap_min, gap_max)
    return eps, t + 5.0


def active(eps, t, key):
    for e in eps:
        if e[key] and e["start"] <= t < e["end"]:
            return e
    return None


def decode_depth(raw):
    a = np.frombuffer(raw, dtype=np.uint8).reshape(CAM_H, CAM_W, 4).astype(np.float32)
    return (a[:, :, 2] + a[:, :, 1] * 256.0 + a[:, :, 0] * 65536.0) / 16777215.0 * 1000.0


def decode_sem(raw):
    return np.frombuffer(raw, dtype=np.uint8).reshape(CAM_H, CAM_W, 4)[:, :, 2].copy()


def lidar_range(pts):
    m = (pts[:, 0] > 1) & (pts[:, 0] < 60) & (np.abs(pts[:, 1]) < CORR_HALF_W) & (pts[:, 2] > -1.35) & (pts[:, 2] < 1.0)
    n = int(m.sum())
    return (float(pts[m, 0].min()) if n >= 3 else RANGE_CAP), n


def cam_range(depth, sem):
    y = (_U - CAM_W / 2.0) / CAM_F * depth
    m = np.isin(sem, VEHICLE_TAGS) & (np.abs(y) < CORR_HALF_W) & (depth > CAM_MIN_DEPTH) & (depth < 60)
    n = int(m.sum())
    return (float(depth[m].min()) if n >= 5 else RANGE_CAP), n


def phantom_range(R, gt_r):
    """Attacker places the phantom in front of the real obstacle (if any) so it changes the sensor's output."""
    return min(R, max(4.0, 0.7 * gt_r)) if gt_r < RANGE_CAP else R


def attack_lidar(pts, ep, nrng, R):
    if ep["sensor"] == "lidar_phantom":
        n = 150
        ph = np.stack([R + nrng.normal(0, 0.03, n), nrng.uniform(-0.9, 0.9, n), nrng.uniform(-1.4, -0.4, n)], 1)
        return np.vstack([pts, ph.astype(np.float32)])
    if ep["sensor"] == "lidar_hide":
        m = (pts[:, 0] > 0) & (pts[:, 0] < 70) & (np.abs(pts[:, 1]) < 2.0) & (pts[:, 2] > -1.5)
        return pts[~m]
    return pts


def attack_cam(depth, sem, ep, R):
    if ep["sensor"] == "cam_phantom":
        half = max(1, int(0.9 / R * CAM_F))
        r0, r1 = int(0.45 * CAM_H), int(0.75 * CAM_H)
        sem[r0:r1, CAM_W // 2 - half:CAM_W // 2 + half] = 14
        depth[r0:r1, CAM_W // 2 - half:CAM_W // 2 + half] = R
    elif ep["sensor"] == "cam_hide":
        y = (_U - CAM_W / 2.0) / CAM_F * depth
        m = np.isin(sem, VEHICLE_TAGS) & (np.abs(y) < 2.0)
        sem[m] = 1
        depth[m] = 200.0
    return depth, sem


def gt_forward_range(ego, others):
    tf = ego.get_transform()
    yaw = math.radians(tf.rotation.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    best = RANGE_CAP
    for o in others:
        d = o.get_location()
        dx, dy = d.x - tf.location.x, d.y - tf.location.y
        fwd, lat = dx * c + dy * s, -dx * s + dy * c
        if fwd > 0 and abs(lat) < CORR_HALF_W:
            r = fwd - o.bounding_box.extent.x - SENSOR_X
            if 0 < r < best:
                best = r
    return best


def get_frame(q, frame):
    while True:
        d = q.get(timeout=10.0)
        if d.frame >= frame:
            return d


def run(a):
    rng = random.Random(a.seed)
    nrng = np.random.default_rng(a.seed)
    eps, auto_dur = build_schedule(a.scenario, rng, a.warmup, a.ep_min, a.ep_max, a.gap_min, a.gap_max)
    if a.spoof_target:
        for e in eps:
            e["spoof_target"] = a.spoof_target
    duration = a.duration if a.duration else (auto_dur if eps else 90.0)
    outdir = os.path.join(a.out, f"{a.scenario}_s{a.seed}")
    os.makedirs(outdir, exist_ok=True)

    client = carla.Client(a.host, a.port)
    client.set_timeout(120.0)
    world = client.get_world()
    if a.map and not world.get_map().name.endswith(a.map):
        world = client.load_world(a.map)
    orig = world.get_settings()
    tm = client.get_trafficmanager()
    actors, sensors = [], []
    try:
        st = world.get_settings()
        st.synchronous_mode, st.fixed_delta_seconds = True, DT
        world.apply_settings(st)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(a.seed)
        bpl = world.get_blueprint_library()
        sps = world.get_map().get_spawn_points()
        rng.shuffle(sps)

        ego = None
        for sp in sps:
            ego = world.try_spawn_actor(bpl.find("vehicle.tesla.model3"), sp)
            if ego:
                break
        actors.append(ego)
        ego.set_autopilot(True, tm.get_port())
        tm.ignore_lights_percentage(ego, 100.0)
        tm.ignore_signs_percentage(ego, 100.0)
        tm.vehicle_percentage_speed_difference(ego, -20.0)
        cars = [b for b in bpl.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
        npcs = []
        for sp in sps[:a.npcs + 10]:
            if len(npcs) >= a.npcs:
                break
            if sp.location.distance(ego.get_location()) < 10:
                continue
            v = world.try_spawn_actor(rng.choice(cars), sp)
            if v:
                v.set_autopilot(True, tm.get_port())
                npcs.append(v)
        actors += npcs
        if a.lead:
            tm.vehicle_percentage_speed_difference(ego, 0.0)
            tm.distance_to_leading_vehicle(ego, 8.0)
            tm.auto_lane_change(ego, False)
            wp0 = world.get_map().get_waypoint(ego.get_location())
            lead = None
            for dist in (14.0, 17.0, 20.0, 24.0):
                tf = wp0.next(dist)[0].transform
                tf.location.z += 0.3
                lead = world.try_spawn_actor(rng.choice(cars), tf)
                if lead:
                    break
            if lead:
                lead.set_autopilot(True, tm.get_port())
                tm.vehicle_percentage_speed_difference(lead, 25.0)
                tm.ignore_lights_percentage(lead, 100.0)
                tm.auto_lane_change(lead, False)
                npcs.append(lead)
                actors.append(lead)

        def mount(bp_id, x, z, **attrs):
            bp = bpl.find(bp_id)
            for k, v in attrs.items():
                bp.set_attribute(k, str(v))
            s = world.spawn_actor(bp, carla.Transform(carla.Location(x=x, z=z)), attach_to=ego)
            q = queue.Queue()
            s.listen(q.put)
            sensors.append(s)
            return q

        cam = dict(image_size_x=CAM_W, image_size_y=CAM_H, fov=CAM_FOV)
        q_lidar = mount("sensor.lidar.ray_cast", SENSOR_X, 1.8, channels=32, range=70, points_per_second=300000,
                        rotation_frequency=1.0 / DT, upper_fov=10, lower_fov=-30, noise_stddev=0.02)
        q_depth = mount("sensor.camera.depth", SENSOR_X, 1.5, **cam)
        q_sem = mount("sensor.camera.semantic_segmentation", SENSOR_X, 1.5, **cam)
        q_gnss = mount("sensor.other.gnss", 0.0, 1.0, noise_lat_stddev=5e-6, noise_lon_stddev=5e-6, noise_alt_stddev=0.5)

        if a.can_model == "realistic":
            from realistic_can import RealisticCan
            can = RealisticCan(a.seed)
        else:
            can = CanSim(rng, n_bg=a.bg_ids, drop_p=a.dos_drop)
        wheel_off = [0.0, 0.0, 0.0, 0.0]
        fcan = open(os.path.join(outdir, "can_log.csv"), "w", newline="")
        fsen = open(os.path.join(outdir, "sensor_log.csv"), "w", newline="")
        wc, ws = csv.writer(fcan), csv.writer(fsen)
        wc.writerow(["ts", "frame", "id", "dlc", "data", "label", "attack_type"])
        ws.writerow(["frame", "t", "ego_speed", "gt_lat", "gt_lon", "gps_lat", "gps_lon", "gps_err_m", "lidar_range",
                     "lidar_npts", "cam_range", "cam_npix", "gt_range", "lidar_range_clean", "cam_range_clean",
                     "label_can", "label_sensor", "sensor_effective", "attack_can", "attack_sensor", "stealth", "moving"])

        n_ticks = int(duration / DT)
        t_start, n_can = None, 0
        spec = world.get_spectator()
        last_state = None
        for k in range(n_ticks):
            frame = world.tick()
            snap = world.get_snapshot()
            if t_start is None:
                t_start = snap.timestamp.elapsed_seconds
            t = snap.timestamp.elapsed_seconds - t_start
            d_li, d_de, d_se, d_gn = get_frame(q_lidar, frame), get_frame(q_depth, frame), get_frame(q_sem, frame), get_frame(q_gnss, frame)

            ep_c, ep_s = active(eps, t, "can"), active(eps, t, "sensor")
            gt_r = gt_forward_range(ego, list(world.get_actors([v.id for v in npcs])))
            pts = np.frombuffer(d_li.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3].copy()
            depth = decode_depth(d_de.raw_data) * (1.0 + nrng.normal(0, 0.005, (CAM_H, CAM_W)).astype(np.float32))
            sem = decode_sem(d_se.raw_data)
            lat, lon = d_gn.latitude, d_gn.longitude
            pts_clean, depth_clean, sem_clean = pts, depth.copy(), sem.copy()
            if ep_s:
                if ep_s["sensor"] == "gps_drift":
                    off = min(ep_s["gps_rate"] * (t - ep_s["start"]), ep_s["gps_cap"])
                    lat += off * math.cos(ep_s["gps_theta"]) / M_PER_DEG
                    lon += off * math.sin(ep_s["gps_theta"]) / (M_PER_DEG * math.cos(math.radians(lat)))
                elif ep_s["sensor"].startswith("lidar"):
                    pts = attack_lidar(pts, ep_s, nrng, phantom_range(ep_s["R_lidar"], gt_r))
                else:
                    depth, sem = attack_cam(depth, sem, ep_s, phantom_range(ep_s["R_cam"], gt_r))
            r_li_clean, _ = lidar_range(pts_clean)
            r_ca_clean, _ = cam_range(depth_clean, sem_clean)
            r_li, n_li = lidar_range(pts)
            r_ca, n_ca = cam_range(depth, sem)

            gt = world.get_map().transform_to_geolocation(ego.get_location())
            err = math.hypot((lat - gt.latitude) * M_PER_DEG, (lon - gt.longitude) * M_PER_DEG * math.cos(math.radians(gt.latitude)))
            vel, acc, ang, ctl = ego.get_velocity(), ego.get_acceleration(), ego.get_angular_velocity(), ego.get_control()
            speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
            yaw = math.radians(ego.get_transform().rotation.yaw)
            ax, ay = acc.x * math.cos(yaw) + acc.y * math.sin(yaw), -acc.x * math.sin(yaw) + acc.y * math.cos(yaw)
            state = {"steer": ctl.steer, "yaw_rate": ang.z, "speed_kmh": speed * 3.6, "ax": ax, "ay": ay,
                     "throttle": ctl.throttle, "brake": ctl.brake, "gear": ctl.gear, "handbrake": int(ctl.hand_brake),
                     "lights": int(ego.get_light_state()), "wheel_noise": [nrng.normal(0, 0.15) for _ in range(4)]}

            for ts, cid, payload, lab, atype in can.step(t, t + DT, state, ep_c):
                wc.writerow([f"{ts:.6f}", frame, f"{cid:03X}", len(payload), payload.hex(), lab, atype])
                n_can += 1
            if a.viz:
                tf_e = ego.get_transform()
                yw = math.radians(tf_e.rotation.yaw)
                spec.set_transform(carla.Transform(carla.Location(x=tf_e.location.x - 10 * math.cos(yw), y=tf_e.location.y - 10 * math.sin(yw),
                                                                  z=tf_e.location.z + 5.0), carla.Rotation(pitch=-20, yaw=tf_e.rotation.yaw)))
                can_s = ep_c["can"].upper() if ep_c else "NORMAL"
                sen_s = ep_s["sensor"].upper() if ep_s else "NORMAL"
                state = f"CAN={can_s}|SENSOR={sen_s}"
                if state != last_state:
                    with open(os.path.join(outdir, "viz_state.txt"), "w") as f:
                        f.write(state)
                    last_state = state
                if ep_s and ep_s["sensor"] in ("lidar_phantom", "cam_phantom"):
                    R = phantom_range(ep_s["R_lidar"] if ep_s["sensor"] == "lidar_phantom" else ep_s["R_cam"], gt_r)
                    pc = tf_e.location + carla.Location(x=(R + SENSOR_X) * math.cos(yw), y=(R + SENSOR_X) * math.sin(yw), z=0.9)
                    world.debug.draw_box(carla.BoundingBox(pc, carla.Vector3D(2.2, 0.9, 0.8)), tf_e.rotation, 0.08,
                                         carla.Color(255, 0, 255), DT * 1.5)
            lc, ls = int(ep_c is not None), int(ep_s is not None)
            effective = int(ls and (abs(r_li - r_li_clean) > 1.0 or abs(r_ca - r_ca_clean) > 1.0 or
                                    (ep_s["sensor"] == "gps_drift" and err > 2.0)))
            ws.writerow([frame, f"{t:.3f}", f"{speed:.3f}", f"{gt.latitude:.8f}", f"{gt.longitude:.8f}", f"{lat:.8f}",
                         f"{lon:.8f}", f"{err:.3f}", f"{r_li:.3f}", n_li, f"{r_ca:.3f}", n_ca, f"{gt_r:.3f}",
                         f"{r_li_clean:.3f}", f"{r_ca_clean:.3f}", lc, ls, effective,
                         ep_c["can"] if ep_c else "", ep_s["sensor"] if ep_s else "", int(ls and not lc), int(speed > 0.5)])
            if k % 200 == 0:
                print(f"  t={t:6.1f}s / {duration:.0f}s  can_frames={n_can}", flush=True)
        fcan.close()
        fsen.close()
        meta = {"carla": client.get_server_version(), "map": world.get_map().name, "scenario": a.scenario, "seed": a.seed,
                "dt": DT, "duration_s": duration, "npcs": len(npcs), "lead_following": bool(a.lead), "can_model": a.can_model, "spoof_target_forced": a.spoof_target, "episodes": eps, "range_cap_m": RANGE_CAP,
                "corridor_half_width_m": CORR_HALF_W, "dos_legit_drop_prob": a.dos_drop, "background_can_ids": a.bg_ids,
                "notes": "CAN bridge is vehicle->bus only (attacks do not feed back into vehicle control). "
                         "CAN frames in [t,t+dt) are built from the vehicle state sampled at t."}
        with open(os.path.join(outdir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print("done ->", outdir)
    finally:
        for s in sensors:
            try:
                s.stop()
                s.destroy()
            except Exception:
                pass
        for v in actors:
            try:
                v.destroy()
            except Exception:
                pass
        tm.set_synchronous_mode(False)
        world.apply_settings(orig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=["S0", "S1", "S2", "S3"], required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--map", default=None, help="e.g. Town03 (default: whatever the server has loaded)")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    p.add_argument("--npcs", type=int, default=30)
    p.add_argument("--can-model", dest="can_model", choices=["simple", "realistic"], default="simple",
                   help="simple = fixed periodic bus; realistic = arbitration, ECU clock skew, event-driven frames (realistic_can.py)")
    p.add_argument("--viz", action="store_true", help="chase-camera spectator + on-screen attack HUD in the server window; writes viz_state.txt")
    p.add_argument("--lead", action="store_true", help="spawn a slower lead vehicle in the ego lane (lead-following regime)")
    p.add_argument("--spoof-target", dest="spoof_target", choices=["speed", "brake", "steer"], default=None)
    p.add_argument("--bg-ids", dest="bg_ids", type=int, default=12)
    p.add_argument("--dos-drop", dest="dos_drop", type=float, default=0.5)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--warmup", type=float, default=15.0)
    p.add_argument("--ep-min", dest="ep_min", type=float, default=10.0)
    p.add_argument("--ep-max", dest="ep_max", type=float, default=15.0)
    p.add_argument("--gap-min", dest="gap_min", type=float, default=6.0)
    p.add_argument("--gap-max", dest="gap_max", type=float, default=10.0)
    run(p.parse_args())
    sys.stdout.flush()
    os._exit(0)  # the native CARLA client can fault during interpreter teardown (exit 127); data is already on disk
