"""ROAD signal-plausibility features per 1 s window (decoded signals from signal_extractions).

Ambient-train captures define, per (ID, signal): value range, 99.9th percentile of |step| between consecutive messages.
Window features: range exceedance, step exceedance, and an IsolationForest score on the 10 Hz forward-filled signal state
(catches inconsistent combinations of signals, e.g. reverse lamp on while the vehicle moves forward).
Only the masquerade captures have decoded signals in ROAD.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ROOT = r"F:\carla_cl_ids\road_data\road\signal_extractions"
CACHE = r"F:\carla_cl_ids\road_data\cache"
TEST_AMBIENT = ["ambient_dyno_drive_extended_short", "ambient_dyno_idle_radio_infotainment",
                "ambient_highway_street_driving_diagnostics", "ambient_dyno_drive_basic_short"]
DT = 0.1


def load_sig(path):
    d = pd.read_csv(path)
    ids = d["ID"].to_numpy(dtype=np.int64)
    t = d["Time"].to_numpy(dtype=np.float64)
    v = d[[c for c in d.columns if c.startswith("Signal_")]].to_numpy(dtype=np.float64)
    lab = d["Label"].to_numpy()
    return t, ids, v, lab


def groups(ids):
    order = np.argsort(ids, kind="stable")
    s = ids[order]
    st = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    en = np.r_[st[1:], len(s)]
    return order, [(int(s[a]), order[a:b]) for a, b in zip(st, en)]


def build_ref(caps):
    lo, hi, steps = {}, {}, {}
    for t, ids, v, _ in caps:
        _, gs = groups(ids)
        for cid, ix in gs:
            x = v[ix]
            lo[cid] = np.fmin(lo.get(cid, np.full(x.shape[1], np.inf)), np.nanmin(np.where(np.isnan(x), np.inf, x), 0))
            hi[cid] = np.fmax(hi.get(cid, np.full(x.shape[1], -np.inf)), np.nanmax(np.where(np.isnan(x), -np.inf, x), 0))
            if len(ix) > 2:
                steps.setdefault(cid, []).append(np.abs(np.diff(x, axis=0)))
    q = {}
    for cid, lst in steps.items():
        s = np.concatenate(lst)
        with np.errstate(all="ignore"):
            q[cid] = np.nanpercentile(s, 99.9, axis=0)
    cols = [(cid, k) for cid in sorted(lo) for k in range(lo[cid].shape[0]) if np.isfinite(lo[cid][k])]
    return {"lo": lo, "hi": hi, "q": q, "cols": cols}


def state_grid(t, ids, v, ref, T):
    """10 Hz forward-filled state matrix over the reference columns."""
    grid = np.arange(0, T, DT)
    out = np.zeros((len(grid), len(ref["cols"])))
    col = {c: i for i, c in enumerate(ref["cols"])}
    _, gs = groups(ids)
    for cid, ix in gs:
        if cid not in ref["lo"]:
            continue
        o = np.argsort(t[ix], kind="stable")
        ix = ix[o]
        pos = np.searchsorted(t[ix], grid, side="right") - 1
        ok = pos >= 0
        for k in range(v.shape[1]):
            if (cid, k) in col:
                x = np.where(ok, v[ix][np.maximum(pos, 0), k], 0.0)
                out[:, col[(cid, k)]] = np.nan_to_num(x)
    return out


def build_pairs(G, mu, sd, thr=0.97, max_pairs=800):
    """Best-correlated partner (from another column) for each non-constant signal; linear fit and residual scale."""
    keep = np.flatnonzero(sd > 1e-3)
    Z = (G[:, keep] - mu[keep]) / sd[keep]
    C = (Z.T @ Z) / len(Z)
    np.fill_diagonal(C, 0)
    pairs = []
    for i in range(len(keep)):
        j = int(np.argmax(np.abs(C[i])))
        if abs(C[i, j]) >= thr:
            x, y = G[:, keep[j]], G[:, keep[i]]
            a, b = np.polyfit(x, y, 1)
            res = y - (a * x + b)
            pairs.append((keep[j], keep[i], a, b, max(res.std(), 0.01 * (y.max() - y.min()), 1e-6)))
    return pairs[:max_pairs]


def pair_score(g, pairs):
    if not pairs:
        return np.zeros(len(g))
    X = np.stack([g[:, p[0]] for p in pairs], 1)
    Y = np.stack([g[:, p[1]] for p in pairs], 1)
    A = np.array([p[2] for p in pairs])
    B = np.array([p[3] for p in pairs])
    S = np.array([p[4] for p in pairs])
    return np.abs(Y - (A * X + B)) / S  # (rows, pairs)


def window_feats(t, ids, v, lab, ref, T):
    W = int(T)
    rng_ex = np.zeros(W)
    step_ex = np.zeros(W)
    _, gs = groups(ids)
    for cid, ix in gs:
        ix = ix[np.argsort(t[ix], kind="stable")]
        x = v[ix]
        w = np.minimum(t[ix].astype(int), W - 1)
        if cid not in ref["lo"]:
            # unseen ID with decodable signals
            np.maximum.at(rng_ex, w, 100.0)
            continue
        lo, hi = ref["lo"][cid], ref["hi"][cid]
        span = np.maximum(hi - lo, 1.0)
        with np.errstate(invalid="ignore"):
            ex = np.maximum(np.maximum(lo - x, x - hi), 0) / span
        ex = np.nan_to_num(ex).max(1)
        np.maximum.at(rng_ex, w, ex)
        if len(ix) > 1 and cid in ref["q"]:
            d = np.abs(np.diff(x, axis=0))
            q = ref["q"][cid]
            with np.errstate(invalid="ignore", divide="ignore"):
                e = np.maximum(d - q, 0) / span
            e = np.nan_to_num(e).max(1)
            np.maximum.at(step_ex, w[1:], e)
    grid_w = (np.arange(0, T, DT) / 1.0).astype(int)
    return W, rng_ex, step_ex, grid_w


def main():
    t0 = time.time()
    amb_meta = json.load(open(os.path.join(ROOT, "ambient", "metadata.json")))
    amb = {k: load_sig(os.path.join(ROOT, "ambient", k + ".csv")) for k in amb_meta if os.path.exists(os.path.join(ROOT, "ambient", k + ".csv"))}
    train = [k for k in amb if k not in TEST_AMBIENT]
    ref = build_ref([amb[k] for k in train])
    print(f"signal reference: {len(ref['cols'])} signals over {len(ref['lo'])} IDs ({time.time() - t0:.0f}s)", flush=True)

    # IsolationForest on the 10 Hz state of the ambient-train captures
    grids = [state_grid(*amb[k][:3], ref, amb[k][0].max()) for k in train]
    G = np.vstack(grids)
    mu, sd = G.mean(0), G.std(0) + 1e-6
    rng = np.random.default_rng(0)
    sub = G[rng.choice(len(G), min(len(G), 60000), replace=False)]
    pairs = build_pairs(G, mu, sd)
    print(f"consistency pairs: {len(pairs)}", flush=True)
    iso = IsolationForest(n_estimators=200, max_samples=4096, random_state=0, n_jobs=-1).fit((sub - mu) / sd)

    def feats(name, c):
        t, ids, v, lab = c
        T = t.max()
        W, rng_ex, step_ex, gw = window_feats(t, ids, v, lab, ref, T)
        g = state_grid(t, ids, v, ref, T)
        sc = -iso.score_samples((g - mu) / sd)
        st = np.full(W, np.nan)
        for w in range(W):
            m = gw == w
            if m.any():
                st[w] = sc[m].max()
        ps = pair_score(g, pairs)
        pr = np.full(W, np.nan)
        for w in range(W):
            m = gw == w
            if m.any():
                pr[w] = np.max(np.median(ps[m], axis=0)) if ps.shape[1] else 0.0
        wl = np.zeros(W)
        np.maximum.at(wl, np.minimum(t.astype(int), W - 1), lab.astype(float))
        return pd.DataFrame({"capture": name, "w": np.arange(W), "rng_ex": rng_ex, "step_ex": step_ex, "state_if": st, "pair_res": np.log1p(pr), "sig_lab": wl})

    rows = [feats(k, amb[k]) for k in amb]
    print(f"ambient signal features ({time.time() - t0:.0f}s)", flush=True)
    att_meta = json.load(open(os.path.join(ROOT, "attacks", "metadata.json")))
    for k in att_meta:
        p = os.path.join(ROOT, "attacks", k + ".csv")
        if k.endswith("_masquerade") and os.path.exists(p):
            rows.append(feats(k, load_sig(p)))
            print(f"signal features {k} ({time.time() - t0:.0f}s)", flush=True)
    pd.concat(rows, ignore_index=True).to_pickle(os.path.join(CACHE, "signal_windows.pkl"))
    print("done", f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
