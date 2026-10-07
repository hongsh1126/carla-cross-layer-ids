"""Targeted supplement: (A) CAN-only runs with each stealth-spoof target forced, (B) lead-following runs (S2 attacks, S0 baseline)."""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
MAPS = ["Town10HD_Opt", "Town03_Opt"]
MODEL = os.environ.get("CL_CAN_MODEL", "simple")
SUFFIX = "_real" if MODEL == "realistic" else ""
CAN_ARGS = ["--can-model", MODEL] if MODEL == "realistic" else []
runs = []
for m in MAPS:
    base = 301 if m == "Town10HD_Opt" else 401
    for ti, tgt in enumerate(["speed", "brake", "steer"]):
        for k in range(2):
            runs.append(("spoof_" + tgt, m, "S1", base + 10 * ti + k, ["--spoof-target", tgt]))
    for k in range(3):
        runs.append(("lead", m, "S2", base + 50 + k, ["--lead"]))
    for k in range(2):
        runs.append(("lead", m, "S0", base + 60 + k, ["--lead"]))
print(f"{len(runs)} supplementary runs", flush=True)
t_all = time.time()
for i, (grp, m, sc, seed, extra) in enumerate(runs, 1):
    out = os.path.join(HERE, "out_supp" + SUFFIX, grp, m)
    if os.path.exists(os.path.join(out, f"{sc}_s{seed}", "meta.json")):
        continue
    t0 = time.time()
    ok = False
    for attempt in (1, 2):
        try:
            r = subprocess.run([PY, os.path.join(HERE, "carla_cl_ids.py"), "--scenario", sc, "--seed", str(seed), "--map", m,
                                "--out", out] + extra + CAN_ARGS, capture_output=True, text=True, timeout=1500)
        except subprocess.TimeoutExpired:
            continue
        ok = os.path.exists(os.path.join(out, f"{sc}_s{seed}", "meta.json"))
        if ok:
            break
        print(f"[{i}/{len(runs)}] FAILED {grp} {m} {sc} s{seed}: {r.stderr[-200:]}", flush=True)
    print(f"[{i}/{len(runs)}] {grp} {m} {sc} s{seed} {'ok' if ok else 'GAVE UP'} in {time.time() - t0:.0f}s (total {(time.time() - t_all) / 60:.1f} min)", flush=True)
print("ALL DONE", flush=True)
