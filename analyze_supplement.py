"""Supplement analysis: detectors trained on the main collection (other map), tested on targeted supplementary runs of this map."""
import glob
import json
import os

import numpy as np
import pandas as pd

import analyze_collection as A

HERE = os.path.dirname(os.path.abspath(__file__))


def load(root, pattern):
    out = []
    for d in sorted(glob.glob(os.path.join(root, pattern))):
        if os.path.exists(os.path.join(d, "meta.json")):
            w = A.run_windows(d, os.path.basename(os.path.dirname(d)))
            w["supp_group"] = os.path.basename(os.path.dirname(os.path.dirname(d)))
            out.append(w)
    return A.label(pd.concat(out, ignore_index=True))


main = A.label(pd.concat([A.run_windows(d, os.path.basename(os.path.dirname(d)))
                          for d in sorted(glob.glob(os.path.join(A.ROOT, "*", "*"))) if os.path.exists(os.path.join(d, "meta.json"))],
                         ignore_index=True))
supp = load(os.path.join(HERE, "out_supp"), "*/*/*")
supp["group"] = np.where(supp.group.str.contains("_hide"), supp.group + np.where(supp.inband >= 0.5, " [in-band]", " [out-of-band]"), supp.group)
print("supplement windows:", len(supp))
print(supp.groupby(["supp_group", "map", "scenario"]).agg(runs=("run", "nunique"), windows=("w", "size"), moving=("moving", "mean")).round(2).to_string())
print("\nsupplement groups:")
print(supp.group.value_counts().to_string())

allres = {}
for tr_map, te_map in (("Town03_Opt", "Town10HD_Opt"), ("Town10HD_Opt", "Town03_Opt")):
    trm = main[main["map"] == tr_map]
    seeds = sorted(trm.seed.unique())
    train, val = trm[trm.seed != seeds[-1]], trm[trm.seed == seeds[-1]]
    test = supp[supp["map"] == te_map]
    name, res, a = A.experiment(train, val, test, f"train main {tr_map} -> test supplement {te_map}", "v2")
    allres[name] = res
    groups = sorted({g for r in res.values() for g in r["recall"]})
    tab = pd.DataFrame({s: {g: f"{r['recall'][g][0] * 100:.0f}% / {r['group_auc'][g]:.2f} (n={r['recall'][g][1]})" for g in groups if g in r["recall"]}
                        for s, r in res.items()})
    tab.loc["FPR (normal windows)"] = [f"{r['fpr'] * 100:.1f}%" for r in res.values()]
    tab.loc["AUROC (any attack)"] = [f"{r['auc']:.3f}" for r in res.values()]
    print(f"\n== {name} ==  (cell = recall at calibrated threshold / per-group AUROC)")
    print(tab.to_string())
json.dump(allres, open(os.path.join(HERE, "out_supp", "supplement_results.json"), "w"), indent=2)
