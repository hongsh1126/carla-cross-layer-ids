"""ROAD validation results: per-capture / per-family AUROC for timing, +IAT, plausibility and fused detectors."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

CACHE = r"F:\carla_cl_ids\road_data\cache"
TIMING = ["n", "uniq", "ent", "unseen", "rate_dev"]
IAT = ["iat_dev", "iat_cv"]
PLAUS = ["rng_ex", "step_ex", "state_if"]
PLAUS2 = PLAUS + ["pair_res"]
df = pd.read_pickle(os.path.join(CACHE, "windows.pkl"))
sg = pd.read_pickle(os.path.join(CACHE, "signal_windows.pkl"))
df = df.merge(sg[["capture", "w", "rng_ex", "step_ex", "state_if", "pair_res"]], on=["capture", "w"], how="left")
df["state_if"] = df.state_if.fillna(df.state_if.median())
for c in ("rng_ex", "step_ex"):
    df[c] = np.log1p(df[c].fillna(0) * 100)  # compress heavy tails
df["pair_res"] = df.pair_res.fillna(0)
df.loc[df.kind.isin(["fuzzing", "fabrication"]), PLAUS2] = np.nan

tr = df[df.split == "train"].copy()
T2 = TIMING + ["rate_z"]
SETS = {"timing": TIMING, "timing+IAT": TIMING + IAT, "plausibility": PLAUS, "fused": TIMING + IAT + PLAUS,
        "timing v2": T2, "timing+IAT v2": T2 + IAT, "plausibility v2": PLAUS2, "fused v2": T2 + IAT + PLAUS2}
score = {}
for name, cols in SETS.items():
    s = StandardScaler().fit(tr[cols])
    m = IsolationForest(n_estimators=300, random_state=0, n_jobs=-1).fit(s.transform(tr[cols]))
    ok = df[cols].notna().all(axis=1)
    sc = np.full(len(df), np.nan)
    sc[ok.to_numpy()] = -m.score_samples(s.transform(df.loc[ok, cols]))
    df["s_" + name] = sc
    score[name] = m

te_amb = df[df.split == "test"]
tr = df[df.split == "train"]
att = df[(df.split == "attack") & (df.label >= 0)]
rows = []
rng = np.random.default_rng(0)


def auc(pos, neg):
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return roc_auc_score(y, np.r_[pos, neg]) if len(pos) and len(neg) else np.nan


for (kind, fam), g in att.groupby(["kind", "family"]):
    caps = sorted(g.capture.unique())
    for name in SETS:
        col = "s_" + name
        if g[col].isna().all():
            continue
        pos = g[g.label == 1][col]
        neg_in = g[g.label == 0][col]
        neg_amb = te_amb[col] if te_amb[col].notna().any() else None
        # bootstrap over captures
        bs = []
        for _ in range(500):
            pick = rng.choice(caps, len(caps))
            gg = pd.concat([g[g.capture == c] for c in pick])
            a = auc(gg[gg.label == 1][col], gg[gg.label == 0][col])
            if not np.isnan(a):
                bs.append(a)
        thr = np.percentile(te_amb[col].dropna(), 99) if neg_amb is not None else np.nan
        rows.append(dict(kind=kind, family=fam, detector=name, n_caps=len(caps), n_pos=len(pos), n_neg=len(neg_in),
                         auc_in=auc(pos, neg_in), ci_lo=np.percentile(bs, 2.5) if bs else np.nan, ci_hi=np.percentile(bs, 97.5) if bs else np.nan,
                         auc_amb=auc(pos, neg_amb) if neg_amb is not None else np.nan,
                         recall_at_1pct=float((pos > thr).mean()) if neg_amb is not None else np.nan))
R = pd.DataFrame(rows)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(R.round(3).to_string())
R.to_csv(os.path.join(CACHE, "road_results.csv"), index=False)
# kind-level summary (pool families)
srows = []
for kind, g in att.groupby("kind"):
    for name in SETS:
        col = "s_" + name
        if g[col].isna().all():
            continue
        fams = sorted(g.family.unique())
        per = [auc(g[(g.family == f) & (g.label == 1)][col], g[(g.family == f) & (g.label == 0)][col]) for f in fams]
        srows.append(dict(kind=kind, detector=name, n_fam=len(fams), auc_in_mean=np.nanmean(per), auc_in_min=np.nanmin(per), auc_in_max=np.nanmax(per),
                          auc_amb=auc(g[g.label == 1][col], te_amb[col].dropna()) if te_amb[col].notna().any() else np.nan))
S = pd.DataFrame(srows)
print(S.round(3).to_string())
S.to_csv(os.path.join(CACHE, "road_summary.csv"), index=False)
# FPR on held-out ambient at the train 99th percentile
fp = {n: float((te_amb["s_" + n] > np.nanpercentile(tr["s_" + n], 99)).mean()) for n in ("timing", "timing+IAT", "timing v2", "timing+IAT v2")}
print("held-out ambient FPR at train 99th pct:", fp, "windows:", len(te_amb), "train windows:", len(tr))
print("attack windows: pos", int((att.label == 1).sum()), "neg", int((att.label == 0).sum()))

# recall / FPR at a threshold set on ambient-train windows only (99th pct), as in the simulation study
rec = []
for kind, g in att.groupby("kind"):
    for name in SETS:
        col = "s_" + name
        if g[col].isna().all():
            continue
        thr = np.nanpercentile(tr[col], 99)
        rec.append(dict(kind=kind, detector=name, recall=float((g[g.label == 1][col] > thr).mean()),
                        fpr_in=float((g[g.label == 0][col] > thr).mean()), fpr_amb=float((te_amb[col] > thr).mean())))
RC = pd.DataFrame(rec)
print(RC.round(3).to_string())
RC.to_csv(os.path.join(CACHE, "road_recall.csv"), index=False)
