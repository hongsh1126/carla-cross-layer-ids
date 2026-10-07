"""Window-level analysis of the main collection: dataset stats + network-only vs fused detector baselines.

Windows are 1 s. CAN features are calibrated per run on the clean first seconds (t in [2,12) s, before any attack).
Detector: RandomForest; threshold = 99th percentile of validation-run normal-window scores (about 1% FPR);
evaluation is cross-map (train/val on one map, test on the other).
"""
import glob
import json
import os
import sys
import zlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

SUFFIX = "_real" if os.environ.get("CL_DATA") == "real" else ""
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_main" + SUFFIX)
CAL = (2, 12)
CHK_IDS = (0x0A0, 0x0B0, 0x0C0, 0x2F0)
PRIM = (0x0A0, 0x0B0, 0x0C0, 0x1B0, 0x2F0)
M_PER_DEG = 111320.0


def i16(lo, hi):
    return ((hi.astype(np.int32) << 8) | lo.astype(np.int32)).astype(np.int16).astype(np.float64)


def u16(lo, hi):
    return ((hi.astype(np.int32) << 8) | lo.astype(np.int32)).astype(np.float64)


def run_windows(d, map_name):
    meta = json.load(open(os.path.join(d, "meta.json")))
    s = pd.read_csv(os.path.join(d, "sensor_log.csv"), keep_default_na=False, dtype={"attack_can": str, "attack_sensor": str})
    c = pd.read_csv(os.path.join(d, "can_log.csv"), keep_default_na=False, dtype={"id": str, "data": str, "attack_type": str})
    c["ts"] = c.ts.astype(float)
    idn = c.id.map(lambda x: int(x, 16)).to_numpy()
    raw = np.frombuffer(bytes.fromhex("".join(x.ljust(16, "0") for x in c.data.tolist())), dtype=np.uint8).reshape(-1, 8)
    ts = c.ts.to_numpy()
    w = np.floor(ts).astype(int)
    W = int(s.t.max()) + 1

    def sig(cid, fn):
        m = idn == cid
        return m, fn(raw[m])

    # ---- window-level CAN features
    n = np.bincount(w, minlength=W)[:W].astype(float)
    cal_w = np.arange(CAL[0], CAL[1])
    ref_n = max(n[cal_w].mean(), 1.0)
    cal_mask = (w >= CAL[0]) & (w < CAL[1])
    known = set(np.unique(idn[cal_mask]).tolist())
    unk = ~np.isin(idn, list(known))
    n_unk = np.bincount(w[unk], minlength=W)[:W].astype(float)
    ids_all = sorted(set(np.unique(idn).tolist()))
    cnt_mat = pd.crosstab(pd.Series(w, name="w"), pd.Series(idn, name="id")).reindex(range(W), fill_value=0)
    uniq = (cnt_mat > 0).sum(axis=1).to_numpy().astype(float)
    p = cnt_mat.div(cnt_mat.sum(axis=1).replace(0, 1), axis=0).to_numpy()
    ent = -(np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0)).sum(axis=1)
    ent_ref = ent[cal_w].mean()
    kcols = [i for i in cnt_mat.columns if i in known]
    ref_rate = cnt_mat.loc[cal_w, kcols].mean()
    dev = ((cnt_mat[kcols] - ref_rate).abs() / ref_rate.clip(lower=1.0)).max(axis=1).to_numpy()

    chk_fail = np.zeros(W)
    cnt_anom = np.zeros(W)
    for cid in CHK_IDS:
        m = idn == cid
        if not m.any():
            continue
        r, t_, w_ = raw[m], ts[m], w[m]
        order = np.argsort(t_, kind="stable")
        r, w_ = r[order], w_[order]
        x = np.bitwise_xor.reduce(r[:, :6], axis=1)
        chk_fail += np.bincount(w_[x != r[:, 6]], minlength=W)[:W]
        dcnt = (np.diff(r[:, 7].astype(int)) % 256)
        bad = np.concatenate([[False], dcnt != 1])
        cnt_anom += np.bincount(w_[bad], minlength=W)[:W]

    # inter-arrival-time features for the periodic primary messages (calibrated on the clean start of each run)
    iat_dev = np.zeros(W)
    iat_cv = np.zeros(W)
    for cid in (0x0A0, 0x0B0, 0x0C0, 0x1B0, 0x2F0, 0x120, 0x140, 0x150):
        m_ = idn == cid
        if m_.sum() < 10:
            continue
        t_ = np.sort(ts[m_])
        d_ = np.diff(t_)
        wd = np.floor(t_[1:]).astype(int)
        ref_ = np.median(d_[(wd >= CAL[0]) & (wd < CAL[1])]) if ((wd >= CAL[0]) & (wd < CAL[1])).any() else np.median(d_)
        g_ = pd.Series(d_).groupby(wd).agg(["median", "std", "mean", "count"])
        g_ = g_[g_["count"] >= 3]
        idx_ = g_.index.to_numpy()
        ok_ = idx_ < W
        idx_, g_ = idx_[ok_], g_[ok_]
        iat_dev[idx_] = np.maximum(iat_dev[idx_], (g_["median"] - ref_).abs().to_numpy() / max(ref_, 1e-6))
        iat_cv[idx_] = np.maximum(iat_cv[idx_], (g_["std"] / g_["mean"]).fillna(0).to_numpy())

    def wmean(mask, vals):
        out = np.full(W, np.nan)
        if mask.any():
            sw = np.bincount(w[mask], weights=vals, minlength=W)[:W]
            sc = np.bincount(w[mask], minlength=W)[:W]
            with np.errstate(invalid="ignore", divide="ignore"):
                out = np.where(sc > 0, sw / sc, np.nan)
        return out

    m0b, m0a, m0c, m1b = idn == 0x0B0, idn == 0x0A0, idn == 0x0C0, idn == 0x1B0
    spd = wmean(m0b, u16(raw[m0b, 0], raw[m0b, 1]) / 100.0)
    ax = wmean(m0b, i16(raw[m0b, 2], raw[m0b, 3]) / 100.0)
    steer = wmean(m0a, i16(raw[m0a, 0], raw[m0a, 1]) / 1000.0)
    yaw = wmean(m0a, i16(raw[m0a, 2], raw[m0a, 3]) / 100.0)
    brake = wmean(m0c, raw[m0c, 1] / 255.0)
    wheel = wmean(m1b, np.mean([u16(raw[m1b, 2 * k], raw[m1b, 2 * k + 1]) for k in range(4)], axis=0) / 100.0)

    can_lab = c.label.to_numpy()
    y_can = np.bincount(w[can_lab == 1], minlength=W)[:W] > 0
    atype = np.array([""] * W, dtype=object)
    if (can_lab == 1).any():
        sub = c[c.label == 1].assign(w=w[can_lab == 1])
        top = sub.groupby("w").attack_type.agg(lambda x: x.value_counts().index[0])
        atype[top.index.to_numpy()] = top.to_numpy()

    # ---- window-level sensor features
    sw_ = np.floor(s.t.to_numpy()).astype(int)
    lid, cam = s.lidar_range.to_numpy(), s.cam_range.to_numpy()
    seen = np.minimum(lid, cam) < 60
    dis = np.where(seen, np.abs(lid - cam), 0.0)
    one = ((lid < 60) != (cam < 60)).astype(float)
    lo_ = np.minimum(lid, cam)
    band = (lo_ >= BAND[0]) & (lo_ < BAND[1])
    dis2 = np.where(band, np.abs(lid - cam), 0.0)
    one2 = (band & ((lid >= 99) ^ (cam >= 99))).astype(float)
    rng = np.random.default_rng(zlib.crc32(d.encode()))
    lat0 = s.gt_lat.iloc[0]
    rn = (s.gps_lat - s.gt_lat).to_numpy() * M_PER_DEG + rng.normal(0, 0.5, len(s))
    re = (s.gps_lon - s.gt_lon).to_numpy() * M_PER_DEG * np.cos(np.radians(lat0)) + rng.normal(0, 0.5, len(s))
    k = 20
    sm = lambda a: np.convolve(a, np.ones(k) / k, mode="same")
    rn, re = sm(rn), sm(re)
    lag = 200
    drift = np.zeros(len(s))
    drift[lag:] = np.hypot(rn[lag:] - rn[:-lag], re[lag:] - re[:-lag])

    def tick_agg(vals, fn):
        return pd.Series(vals).groupby(sw_).agg(fn).reindex(range(W))

    f_dis_mean = tick_agg(dis, "mean").to_numpy()
    f_dis_max = tick_agg(dis, "max").to_numpy()
    f_one = tick_agg(one, "mean").to_numpy()
    f_gps = tick_agg(drift, "max").to_numpy()
    inb = ((s.lidar_range_clean >= BAND[0]) & (s.lidar_range_clean < BAND[1]) & (s.cam_range_clean >= BAND[0]) & (s.cam_range_clean < BAND[1])).to_numpy().astype(float)
    f_inb = tick_agg(inb, "mean").to_numpy()
    f_dis2_mean = tick_agg(dis2, "mean").to_numpy()
    f_dis2_max = tick_agg(dis2, "max").to_numpy()
    f_one2 = tick_agg(one2, "mean").to_numpy()
    tgt = np.array([""] * W, dtype=object)
    for e in meta["episodes"]:
        if e["can"] in ("spoof_plain", "spoof_stealth"):
            for wi in range(int(e["start"]), int(np.ceil(e["end"]))):
                if wi < W and atype[wi] == e["can"]:
                    tgt[wi] = e["spoof_target"]
    eff = tick_agg(s.sensor_effective.to_numpy().astype(float), "mean").to_numpy()
    act = tick_agg(s.label_sensor.to_numpy().astype(float), "mean").to_numpy()
    mov = tick_agg(s.moving.to_numpy().astype(float), "mean").to_numpy()
    sname = pd.Series(s.attack_sensor.to_numpy()).groupby(sw_).agg(lambda x: (x[x != ""].value_counts().index[0] if (x != "").any() else "")).reindex(range(W)).fillna("").to_numpy()

    df = pd.DataFrame({
        "map": map_name, "run": f"{map_name}/{os.path.basename(d)}", "scenario": meta["scenario"], "seed": meta["seed"], "w": np.arange(W),
        "n_dev": np.abs(n - ref_n) / ref_n, "uniq": uniq, "ent_dev": np.abs(ent - ent_ref), "unk": n_unk, "rate_dev": dev,
        "chk_fail": chk_fail, "cnt_anom": cnt_anom, "iat_dev": iat_dev, "iat_cv": iat_cv,
        "spd": spd, "wheel": wheel, "brake": brake, "ax": ax, "steer": steer, "yaw": yaw,
        "dis_mean_v1": f_dis_mean, "dis_max_v1": f_dis_max, "one_sided_v1": f_one,
        "dis_mean": f_dis2_mean, "dis_max": f_dis2_max, "one_sided": f_one2, "gps_drift": f_gps,
        "inband": f_inb, "y_can": y_can, "can_type": atype, "spoof_target": tgt, "sens_eff": eff, "sens_act": act, "sens_type": sname, "moving": mov,
    })
    return df[(df.w >= CAL[1]) & (df.w < W - 1)]


def derive(df, a, variant="v2"):
    """v1 = first pass (all ranges, brake mismatch always); v2 = range-banded sensor consistency, brake mismatch only while moving."""
    df = df.copy()
    df["speed_gap"] = (df.spd - df.wheel).abs().fillna(0.0)
    dg = (df.brake - np.clip(-df.ax / 6.0, 0, 1)).clip(lower=0).fillna(0.0)
    df["decel_gap"] = dg if variant == "v1" else dg * (df.spd.fillna(0) > 5)
    df["steer_resid"] = (df.yaw - a * df.spd * df.steer).abs().fillna(0.0)
    sfx = "_v1" if variant == "v1" else ""
    df["sd_mean"], df["sd_max"], df["sd_one"] = df["dis_mean" + sfx], df["dis_max" + sfx], df["one_sided" + sfx]
    return df


BAND = (3.5, 25.0)
CAN_TIMING = ["n_dev", "uniq", "ent_dev", "unk", "rate_dev", "chk_fail", "cnt_anom"]
CAN_FULL = CAN_TIMING + ["speed_gap", "decel_gap", "steer_resid"]
SENSOR = ["sd_mean", "sd_max", "sd_one", "gps_drift"]
FUSED = CAN_FULL + SENSOR
IAT = ["iat_dev", "iat_cv"]
SETS = {"CAN timing/ID only": CAN_TIMING, "CAN + signal plausibility": CAN_FULL, "Sensor only": SENSOR, "Fused (CAN + sensor)": FUSED,
        "CAN timing/ID + IAT": CAN_TIMING + IAT, "Fused + IAT": FUSED + IAT}


def label(df):
    df = df.copy()
    y_s = df.sens_eff >= 0.5
    df["y_sensor"] = y_s
    still_spoof = df.y_can & df.can_type.str.startswith("spoof") & (df.moving < 0.5)
    df["ambiguous"] = ((df.sens_act > 0) & ~y_s) | still_spoof
    df["y_any"] = df.y_can | y_s
    can_name = np.where(df.can_type.str.startswith("spoof_stealth"), df.can_type + "/" + df.spoof_target, df.can_type)
    df["group"] = np.where(~df.y_any, "normal",
                    np.where(df.y_can & ~y_s, "CAN: " + can_name,
                    np.where(~df.y_can & y_s, "Sensor: " + df.sens_type, "Combined")))
    return df[~df.ambiguous]


def experiment(train, val, test, name, variant="v2"):
    c_ = train[(~train.y_any) & (train.spd > 1)]
    a_coef = float((c_.yaw * (c_.spd * c_.steer)).sum() / max(((c_.spd * c_.steer) ** 2).sum(), 1e-9))
    tr, va, te = derive(train, a_coef, variant), derive(val, a_coef, variant), derive(test, a_coef, variant)
    res = {}
    for sname, cols in SETS.items():
        clf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=2, class_weight="balanced", random_state=0, n_jobs=-1)
        clf.fit(tr[cols].fillna(0), tr.y_any)
        thr = np.quantile(clf.predict_proba(va[va.group == "normal"][cols].fillna(0))[:, 1], 0.99)
        sc = clf.predict_proba(te[cols].fillna(0))[:, 1]
        te_ = te.assign(score=sc, det=sc > thr + 1e-9)
        fpr = te_[te_.group == "normal"].det.mean()
        rec = te_[te_.group != "normal"].groupby("group").det.agg(["mean", "size"])
        auc = roc_auc_score(te_.y_any, te_.score)
        norm = te_[te_.group == "normal"]
        gauc = {}
        for g in rec.index:
            sub = pd.concat([norm, te_[te_.group == g]])
            gauc[g] = float(roc_auc_score(sub.group != "normal", sub.score))
        thr_o = np.quantile(norm.score, 0.99)
        te_["det_o"] = te_.score > thr_o + 1e-9
        rec_o = te_[te_.group != "normal"].groupby("group").det_o.mean()
        brng = np.random.default_rng(0)
        runs_ = te_.run.unique()
        by_run = {r: g for r, g in te_.groupby("run")}
        boots = []
        for _ in range(200):
            pick = brng.choice(runs_, len(runs_), replace=True)
            b = pd.concat([by_run[r] for r in pick])
            if b.y_any.nunique() == 2:
                boots.append(roc_auc_score(b.y_any, b.score))
        res[sname] = {"fpr": float(fpr), "auc": float(auc), "auc_ci": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
                      "recall": {g: (float(r["mean"]), int(r["size"])) for g, r in rec.iterrows()},
                      "group_auc": gauc, "recall_oracle": {g: float(v) for g, v in rec_o.items()}}
    return name, res, a_coef


def main():
    runs = []
    for d in sorted(glob.glob(os.path.join(ROOT, "*", "*"))):
        if os.path.exists(os.path.join(d, "meta.json")):
            runs.append(run_windows(d, os.path.basename(os.path.dirname(d))))
    df = label(pd.concat(runs, ignore_index=True))
    df.to_csv(os.path.join(ROOT, "windows.csv"), index=False)
    print(f"runs={df.run.nunique()} windows={len(df)} (ambiguous windows dropped)")

    print("\n== dataset ==")
    print(df.groupby(["map", "scenario"]).agg(runs=("run", "nunique"), windows=("w", "size"), moving=("moving", "mean"),
                                              can_attack_w=("y_can", "sum"), sensor_attack_w=("y_sensor", "sum")).round(2).to_string())
    print("\nwindows per group:")
    print(df.group.value_counts().to_string())

    maps = sorted(df["map"].unique())
    allres = {}
    if len(maps) == 2:
        for variant in ("v1", "v2"):
            allres[variant] = {}
            for tr_map, te_map in ((maps[0], maps[1]), (maps[1], maps[0])):
                trm = df[df["map"] == tr_map]
                seeds = sorted(trm.seed.unique())
                val = trm[trm.seed == seeds[-1]]
                train = trm[trm.seed != seeds[-1]]
                test = df[df["map"] == te_map]
                name, res, a = experiment(train, val, test, f"train {tr_map} -> test {te_map}", variant)
                allres[variant][name] = res
                print(f"\n== [{variant}] {name} (steer coef a={a:.4f}) ==")
                groups = sorted({g for r in res.values() for g in r["recall"]})
                tab = pd.DataFrame({s: {g: f"{r['recall'][g][0] * 100:.0f}% / {r['group_auc'][g]:.2f} (n={r['recall'][g][1]})"
                                        for g in groups if g in r["recall"]} for s, r in res.items()})
                tab.loc["FPR (normal windows)"] = [f"{r['fpr'] * 100:.1f}%" for r in res.values()]
                tab.loc["AUROC (any attack)"] = [f"{r['auc']:.3f}" for r in res.values()]
                print(tab.to_string())
    json.dump(allres, open(os.path.join(ROOT, "baseline_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
