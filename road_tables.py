"""LaTeX tables for the ROAD validation section."""
import json
import os

import numpy as np
import pandas as pd

CACHE = r"F:\carla_cl_ids\road_data\cache"
TAB = r"F:\carla_cl_ids\paper\tables"
BS = chr(92)
R = pd.read_csv(os.path.join(CACHE, "road_results.csv"))
RC = pd.read_csv(os.path.join(CACHE, "road_recall.csv"))
B = json.load(open(os.path.join(CACHE, "bus_profile.json")))


def rng(key, f="{:.2f}"):
    v = [B["road"][k][key] for k in B["road"]]
    lo, hi = min(v), max(v)
    return f.format(lo) if f.format(lo) == f.format(hi) else f.format(lo) + "--" + f.format(hi)


def sim(name, key, f="{:.2f}"):
    return f.format(B["sim"][name][key])


L = [BS + "begin{tabular}{lccc}", BS + "toprule", "Property & ROAD (real) & Simple & Realistic " + BS + BS, BS + "midrule"]
rows = [("Distinct IDs", "ids", "{:.0f}"), ("Periodic IDs (analysed)", "periodic_ids", "{:.0f}"), ("Frames per second", "fps", "{:.0f}"),
        ("Arrival-interval std., 10" + BS + ",ms IDs (" + BS + "%)", "10ms_relstd_pct", "{:.1f}"), ("Arrival-interval std., 20" + BS + ",ms IDs (" + BS + "%)", "20ms_relstd_pct", "{:.1f}"),
        ("Arrival-interval std., 100" + BS + ",ms IDs (" + BS + "%)", "100ms_relstd_pct", "{:.2f}"), ("99.9th pct. interval / period, 100" + BS + ",ms IDs", "100ms_p999", "{:.1f}")]
for lab, key, f in rows:
    L.append(f"{lab} & {rng(key, f)} & {sim('simple', key, f)} & {sim('realistic', key, f)} " + BS + BS)
L += [BS + "bottomrule", BS + "end{tabular}"]
open(os.path.join(TAB, "road_bus.tex"), "w", encoding="utf-8").write("\n".join(L))

cols = ["timing", "timing v2", "plausibility", "plausibility v2", "fused v2"]


def val(df, det):
    s = df[df.detector == det]
    return np.nan if s.empty else s.auc_in.mean()


def pooled(df):
    return {d: val(df, d) for d in cols}


def fmt(x):
    return "--" if np.isnan(x) else f"{x:.2f}"


L = [BS + "begin{tabular}{lrrccccc}", BS + "toprule",
     "Attack class & Caps. & Pos. & Timing/ID & Timing/ID & Plaus. & Plaus. & Fused " + BS + BS,
     " & & & v1 & v2 & v1 & v2 & v2 " + BS + BS, BS + "midrule"]
npos = lambda df: int(df.drop_duplicates("family").n_pos.sum())


def row(label, df, caps, pos):
    p = pooled(df)
    L.append(f"{label} & {caps} & {pos} & " + " & ".join(fmt(p[c]) for c in cols) + " " + BS + BS)


fu = R[R.kind == "fuzzing"]
row("Fuzzing", fu, int(fu.n_caps.max()), int(fu.n_pos.max()))
fa = R[R.kind == "fabrication"]
row("Fabrication, original ECU still sending", fa, int(fa.drop_duplicates("family").n_caps.sum()), npos(fa))
ma = R[R.kind == "masquerade"]
row("Masquerade, original frames removed", ma, int(ma.drop_duplicates("family").n_caps.sum()), npos(ma))
L.append(BS + "midrule")
names = {"max_speedometer_attack": "~~speedometer maximum", "max_engine_coolant_temp_attack": "~~coolant temperature maximum",
         "reverse_light_on_attack": "~~reverse lamp on", "reverse_light_off_attack": "~~reverse lamp off", "correlated_signal_attack": "~~correlated signals"}
for fam, lab in names.items():
    g = ma[ma.family == fam]
    row(lab, g, int(g.n_caps.max()), int(g.n_pos.max()))
L += [BS + "bottomrule", BS + "end{tabular}"]
open(os.path.join(TAB, "road_det.tex"), "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
print(open(os.path.join(TAB, "road_bus.tex"), encoding="utf-8").read())
