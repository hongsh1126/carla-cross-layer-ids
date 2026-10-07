"""Final cross-map evaluation on the pooled data (main collection + targeted supplement).

For each direction (train map X -> test map Y): all runs of Y are the test set; runs of X are split into train and validation
(validation = main last seed + one run of each supplement type, used only to set the 1%-FPR threshold).
"""
import glob
import json
import os

import numpy as np
import pandas as pd

import analyze_collection as A

HERE = os.path.dirname(os.path.abspath(__file__))
VAL_SEEDS = {104, 204, 302, 312, 322, 353, 362, 402, 412, 422, 453, 462}


def collect(root, pattern, group_from_path=False):
    out = []
    for d in sorted(glob.glob(os.path.join(root, pattern))):
        if os.path.exists(os.path.join(d, "meta.json")):
            w = A.run_windows(d, os.path.basename(os.path.dirname(d)))
            w["source"] = os.path.basename(os.path.dirname(os.path.dirname(d))) if group_from_path else "main"
            out.append(w)
    return out


runs = collect(A.ROOT, "*/*") + collect(os.path.join(HERE, "out_supp" + A.SUFFIX), "*/*/*", True)
df = A.label(pd.concat(runs, ignore_index=True))
df["group"] = np.where(df.group.str.contains("_hide"), df.group + np.where(df.inband >= 0.5, " [in-band]", " [out-of-band]"), df.group)
df.to_csv(os.path.join(HERE, "windows_all" + A.SUFFIX + ".csv"), index=False)
print(f"runs={df.run.nunique()} windows={len(df)}")
print(df.groupby(["map", "source"]).agg(runs=("run", "nunique"), windows=("w", "size")).to_string())
print("\ngroups:\n" + df.group.value_counts().to_string())

results = {}
for tr_map, te_map in (("Town03_Opt", "Town10HD_Opt"), ("Town10HD_Opt", "Town03_Opt")):
    trm = df[df["map"] == tr_map]
    val = trm[trm.seed.isin(VAL_SEEDS)]
    train = trm[~trm.seed.isin(VAL_SEEDS)]
    test = df[df["map"] == te_map]
    name, res, a = A.experiment(train, val, test, f"{tr_map} -> {te_map}", "v2")
    results[name] = res
    print(f"\n== {name}: train windows={len(train)} (pos {int(train.y_any.sum())}), val={len(val)}, test={len(test)} (pos {int(test.y_any.sum())}) ==")
    groups = sorted({g for r in res.values() for g in r["recall"]})
    print("cell = recall@val-threshold / recall@test-1%FPR / group AUROC")
    tab = pd.DataFrame({s: {g: f"{r['recall'][g][0] * 100:.0f}% / {r['recall_oracle'][g] * 100:.0f}% / {r['group_auc'][g]:.2f} (n={r['recall'][g][1]})"
                            for g in groups if g in r["recall"]} for s, r in res.items()})
    tab.loc["FPR @ val threshold"] = [f"{r['fpr'] * 100:.1f}%" for r in res.values()]
    tab.loc["AUROC any attack [95% CI]"] = [f"{r['auc']:.3f} [{r['auc_ci'][0]:.2f},{r['auc_ci'][1]:.2f}]" for r in res.values()]
    print(tab.to_string())
json.dump(results, open(os.path.join(HERE, "results_final" + A.SUFFIX + ".json"), "w"), indent=2)
