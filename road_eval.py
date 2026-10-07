"""Validation on the real ROAD CAN dataset (Verma et al.): bus statistics and injection vs masquerade attacks.

Part A  timing/ID/IAT features on 1 s windows of the raw logs; detector = IsolationForest trained on ambient-train windows only
Part B  (road_signals.py) decoded-signal plausibility on the masquerade captures
Windows overlapping the injection interval by >= 0.25 s are positives, no overlap = negatives, the rest dropped.
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

ROOT = r"F:\carla_cl_ids\road_data\road"
CACHE = r"F:\carla_cl_ids\road_data\cache"
os.makedirs(CACHE, exist_ok=True)
TEST_AMBIENT = ["ambient_dyno_drive_extended_short", "ambient_dyno_idle_radio_infotainment",
                "ambient_highway_street_driving_diagnostics", "ambient_dyno_drive_basic_short"]
TIMING = ["n", "uniq", "ent", "unseen", "rate_dev"]
IAT = ["iat_dev", "iat_cv"]


def load_log(path):
    df = pd.read_csv(path, sep=" ", header=None, usecols=[0, 2], dtype=str, names=["ts", "if", "x"])
    ts = pd.to_numeric(df["ts"].str.slice(1, -1)).to_numpy(dtype=np.float64)
    pp = df["x"].str.partition("#")
    cat = pd.Categorical(pp[0])
    idv = np.array([int(c, 16) for c in cat.categories], dtype=np.int64)[cat.codes]
    return ts - ts[0], idv


def build_ref(logs):
    """Per-ID mean rate and period statistics from the ambient-train captures."""
    cnt, tot_t = {}, 0.0
    diffs = {}
    for ts, ids in logs:
        tot_t += ts[-1]
        u, c = np.unique(ids, return_counts=True)
        for a, b in zip(u, c):
            cnt[a] = cnt.get(a, 0) + int(b)
        order = np.lexsort((ts, ids))
        t_s, i_s = ts[order], ids[order]
        starts = np.flatnonzero(np.r_[True, i_s[1:] != i_s[:-1]])
        ends = np.r_[starts[1:], len(i_s)]
        for a, b in zip(starts, ends):
            if b - a >= 10:
                d = np.diff(t_s[a:b])
                diffs.setdefault(int(i_s[a]), []).append(d[:200000])
    ids_sorted = np.array(sorted(cnt))
    rate = np.array([cnt[i] / tot_t for i in ids_sorted])
    per = {}
    for i, lst in diffs.items():
        d = np.concatenate(lst)
        med = np.median(d)
        mad = 1.4826 * np.median(np.abs(d - med))
        if med <= 2.0 and mad / med < 0.25 and cnt[i] / tot_t >= 0.5:
            per[i] = (med, mad)
    return {"ids": ids_sorted, "rate": rate, "periodic": per}


def counts(ts, ids, ref):
    K = len(ref["ids"])
    pos = np.searchsorted(ref["ids"], ids)
    known = (pos < K) & (ref["ids"][np.minimum(pos, K - 1)] == ids)
    col = np.where(known, pos, K)
    W = int(ts[-1])  # drop the last partial window
    w = ts.astype(int)
    keep = w < W
    return np.bincount(w[keep] * (K + 1) + col[keep], minlength=W * (K + 1)).reshape(W, K + 1)


def window_features(ts, ids, ref):
    K = len(ref["ids"])
    W = int(ts[-1])
    cnt = counts(ts, ids, ref)
    n = cnt.sum(1).astype(float)
    p = cnt / np.maximum(n, 1)[:, None]
    ent = -(np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0)).sum(1)
    rate_dev = (np.abs(cnt[:, :K] - ref["rate"][None, :]) / np.maximum(ref["rate"], 1.0)[None, :]).max(1)
    rate_z = (np.abs(cnt[:, :K] - ref["mu"][None, :]) / ref["sd"][None, :]).max(1)
    feats = {"rate_z": rate_z, "w": np.arange(W), "n": n, "uniq": (cnt > 0).sum(1).astype(float), "ent": ent, "unseen": cnt[:, K].astype(float), "rate_dev": rate_dev}
    # inter-arrival features over the periodic IDs
    iat_dev, iat_cv = np.zeros(W), np.zeros(W)
    order = np.lexsort((ts, ids))
    t_s, i_s = ts[order], ids[order]
    starts = np.flatnonzero(np.r_[True, i_s[1:] != i_s[:-1]])
    ends = np.r_[starts[1:], len(i_s)]
    for a, b in zip(starts, ends):
        pid = int(i_s[a])
        if pid not in ref["periodic"] or b - a < 4:
            continue
        med = ref["periodic"][pid][0]
        t = t_s[a:b]
        d = np.diff(t)
        wd = t[1:].astype(int)
        ok = wd < W
        d, wd = d[ok], wd[ok]
        c = np.bincount(wd, minlength=W).astype(float)
        s1 = np.bincount(wd, weights=d, minlength=W)
        s2 = np.bincount(wd, weights=d * d, minlength=W)
        good = c >= 3
        mean = np.where(good, s1 / np.maximum(c, 1), np.nan)
        std = np.sqrt(np.maximum(np.where(good, s2 / np.maximum(c, 1), 0) - np.nan_to_num(mean) ** 2, 0))
        dev = np.abs(mean - med) / med
        cv = std / mean
        iat_dev = np.fmax(iat_dev, np.nan_to_num(dev))
        iat_cv = np.fmax(iat_cv, np.nan_to_num(cv))
    feats["iat_dev"], feats["iat_cv"] = iat_dev, iat_cv
    return pd.DataFrame(feats)


def bus_profile(ts, ids, per=None):
    """Descriptive statistics of a bus (works for ROAD logs and for our simulated can_log)."""
    span = ts[-1] - ts[0]
    out = {"ids": int(len(np.unique(ids))), "fps": len(ts) / span}
    order = np.lexsort((ts, ids))
    t_s, i_s = ts[order], ids[order]
    starts = np.flatnonzero(np.r_[True, i_s[1:] != i_s[:-1]])
    ends = np.r_[starts[1:], len(i_s)]
    rows = []
    for a, b in zip(starts, ends):
        if b - a < 30:
            continue
        d = np.diff(t_s[a:b])
        med = np.median(d)
        mad = 1.4826 * np.median(np.abs(d - med))
        if med > 2.0 or mad / med > 0.25:
            continue
        core = d[(d > 0.5 * med) & (d < 1.5 * med)]
        rows.append((med, core.std() / med if len(core) > 10 else np.nan, np.percentile(d, 99.9) / med))
    r = pd.DataFrame(rows, columns=["period", "rel_std", "p999_ratio"])
    out["periodic_ids"] = len(r)
    for lo, hi, name in ((0.005, 0.015, "10ms"), (0.015, 0.03, "20ms"), (0.04, 0.06, "50ms"), (0.08, 0.12, "100ms")):
        s = r[(r.period >= lo) & (r.period < hi)]
        out[f"{name}_n"] = len(s)
        out[f"{name}_relstd_pct"] = float(100 * s.rel_std.median()) if len(s) else np.nan
        out[f"{name}_p999"] = float(s.p999_ratio.median()) if len(s) else np.nan
    return out


def label_windows(n_w, interval):
    lab = np.full(n_w, -1)  # -1 ambiguous
    a, b = interval
    for w in range(n_w):
        ov = max(0.0, min(b, w + 1) - max(a, w))
        lab[w] = 1 if ov >= 0.25 else (0 if ov == 0 else -1)
    return lab


def main():
    t0 = time.time()
    amb_meta = json.load(open(os.path.join(ROOT, "ambient", "capture_metadata.json")))
    att_meta = json.load(open(os.path.join(ROOT, "attacks", "capture_metadata.json")))
    amb = {}
    for k in amb_meta:
        amb[k] = load_log(os.path.join(ROOT, "ambient", k + ".log"))
        print(f"loaded {k}: {len(amb[k][0]):,} frames ({time.time() - t0:.0f}s)", flush=True)
    train_names = [k for k in amb if k not in TEST_AMBIENT]
    ref = build_ref([amb[k] for k in train_names])
    cs = np.vstack([counts(*amb[k], ref)[:, :len(ref["ids"])] for k in train_names])
    ref["mu"] = cs.mean(0)
    ref["sd"] = np.sqrt(cs.var(0) + ref["mu"] + 1.0)  # Poisson floor
    print(f"reference: {len(ref['ids'])} IDs, {len(ref['periodic'])} periodic ({time.time() - t0:.0f}s)", flush=True)

    # --- bus statistics
    prof = {k: bus_profile(*amb[k]) for k in ("ambient_dyno_drive_basic_long", "ambient_highway_street_driving_long", "ambient_dyno_idle_radio_infotainment")}
    sim = {}
    for name, root in (("simple", "out_main"), ("realistic", "out_main_real")):
        d = os.path.join(r"F:\carla_cl_ids", root, "Town10HD_Opt", "S0_s101")
        c = pd.read_csv(os.path.join(d, "can_log.csv"), dtype={"id": str}, usecols=["ts", "id"])
        sim[name] = bus_profile(c.ts.to_numpy(dtype=float), c.id.map(lambda x: int(x, 16)).to_numpy(dtype=np.int64))
    json.dump({"road": prof, "sim": sim}, open(os.path.join(CACHE, "bus_profile.json"), "w"), indent=1)
    print(json.dumps({"road": prof, "sim": sim}, indent=1), flush=True)

    # --- window features
    rows = []
    for k, (ts, ids) in amb.items():
        f = window_features(ts, ids, ref)
        f["capture"], f["kind"], f["family"], f["label"] = k, "ambient", "ambient", 0
        f["split"] = "test" if k in TEST_AMBIENT else "train"
        rows.append(f)
    for k, m in att_meta.items():
        if not m.get("injection_interval"):
            continue
        ts, ids = load_log(os.path.join(ROOT, "attacks", k + ".log"))
        f = window_features(ts, ids, ref)
        f["label"] = label_windows(len(f), m["injection_interval"])
        base = k.replace("_masquerade", "")
        fam = base.rstrip("0123456789").rstrip("_")
        f["capture"], f["family"] = k, fam
        f["kind"] = "masquerade" if k.endswith("_masquerade") else ("fuzzing" if fam == "fuzzing_attack" else "fabrication")
        f["split"] = "attack"
        rows.append(f)
        print(f"features {k} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.concat(rows, ignore_index=True)
    df.to_pickle(os.path.join(CACHE, "windows.pkl"))
    with open(os.path.join(CACHE, "ref.pkl"), "wb") as fh:
        pickle.dump(ref, fh)
    print("windows:", len(df), f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
