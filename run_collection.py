"""Main collection driver: runs carla_cl_ids.py over maps x seeds x scenarios, skipping finished runs."""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
PLAN = [("Town10HD_Opt", [101, 102, 103, 104]), ("Town03_Opt", [201, 202, 203, 204])]
SCEN = ["S1", "S2", "S0", "S3"]
MODEL = os.environ.get("CL_CAN_MODEL", "simple")
SUFFIX = "_real" if MODEL == "realistic" else ""
EXTRA = ["--can-model", MODEL] if MODEL == "realistic" else []

runs = [(m, s, sc) for m, seeds in PLAN for s in seeds for sc in SCEN]
print(f"{len(runs)} runs planned", flush=True)
t_all = time.time()
for k, (m, seed, sc) in enumerate(runs, 1):
    out = os.path.join(HERE, "out_main" + SUFFIX, m)
    if os.path.exists(os.path.join(out, f"{sc}_s{seed}", "meta.json")):
        print(f"[{k}/{len(runs)}] skip {m} {sc} s{seed} (done)", flush=True)
        continue
    t0 = time.time()
    for attempt in (1, 2):
        try:
            r = subprocess.run([PY, os.path.join(HERE, "carla_cl_ids.py"), "--scenario", sc, "--seed", str(seed),
                                "--map", m, "--out", out] + EXTRA, capture_output=True, text=True, timeout=1500)
        except subprocess.TimeoutExpired:
            print(f"[{k}/{len(runs)}] TIMEOUT {m} {sc} s{seed} attempt {attempt}", flush=True)
            continue
        ok = os.path.exists(os.path.join(out, f"{sc}_s{seed}", "meta.json"))
        if ok:
            break
        print(f"[{k}/{len(runs)}] FAILED {m} {sc} s{seed} attempt {attempt}: {r.stderr[-300:]}", flush=True)
    print(f"[{k}/{len(runs)}] {m} {sc} s{seed} {'ok' if ok else 'GAVE UP'} in {time.time() - t0:.0f}s "
          f"(total {(time.time() - t_all) / 60:.1f} min)", flush=True)
print("ALL DONE", flush=True)
