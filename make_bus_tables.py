"""Tables comparing the simple and the realistic CAN bus model (statistics + detector AUROC)."""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, "paper", "tables")
os.makedirs(TAB, exist_ok=True)
BS = chr(92)  # backslash


def write(name, lines):
    with open(os.path.join(TAB, name), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def bus_stats(pattern, ids):
    rows = []
    for d in sorted(glob.glob(pattern)):
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        meta = json.load(open(os.path.join(d, "meta.json")))
        c = pd.read_csv(os.path.join(d, "can_log.csv"), dtype={"id": str}, usecols=["ts", "id", "label"])
        c["ts"] = c.ts.astype(float)
        c["idn"] = c.id.map(lambda x: int(x, 16))
        clean = np.ones(len(c), bool)
        for e in meta["episodes"]:
            if e["can"]:
                clean &= ~((c.ts >= e["start"]) & (c.ts < e["end"]))
        cl = c[clean & (c.ts > 2)]
        dur = (cl.ts.max() - 2) - sum(e["end"] - e["start"] for e in meta["episodes"] if e["can"])
        rec = {"ids": cl.idn.nunique(), "fps": len(cl) / dur}
        for cid in ids:
            t_ = np.sort(cl[cl.idn == cid].ts.to_numpy())
            d_ = np.diff(t_)
            d_ = d_[d_ < 3 * np.median(d_)] if len(d_) else d_
            rec[cid] = d_.std() * 1000 if len(d_) > 20 else np.nan
        dos = [e for e in meta["episodes"] if e["can"] == "dos"]
        if dos:
            e = dos[0]
            w = c[(c.ts >= e["start"]) & (c.ts < e["end"])]
            rec["legit_share"] = (w.label == 0).sum() / (e["end"] - e["start"]) / max(rec["fps"], 1)
            rec["ids_dos"] = w[w.label == 0].idn.nunique()
        rows.append(rec)
    return pd.DataFrame(rows)


simple = bus_stats(os.path.join(HERE, "out_main", "*", "S1_*"), [0x0A0])
real = bus_stats(os.path.join(HERE, "out_main_real", "*", "S1_*"), [0x0A0])
L = [BS + "begin{tabular}{lrr}", BS + "toprule", "Property (clean bus unless noted) & Simple & Realistic " + BS + BS, BS + "midrule",
     f"Distinct IDs & {simple.ids.mean():.0f} & {real.ids.mean():.0f} " + BS + BS,
     f"Frames per second & {simple.fps.mean():.0f} & {real.fps.mean():.0f} " + BS + BS,
     f"Arrival-interval std., 20{BS},ms message (ms) & {simple[0x0A0].mean():.2f} & {real[0x0A0].mean():.2f} " + BS + BS,
     f"Legitimate frames kept during DoS (share of clean rate) & {simple.legit_share.mean():.2f} & {real.legit_share.mean():.2f} " + BS + BS,
     f"Distinct legitimate IDs still seen during DoS & {simple.ids_dos.mean():.0f} & {real.ids_dos.mean():.0f} " + BS + BS,
     BS + "bottomrule", BS + "end{tabular}"]
write("bus_stats.tex", L)
stats = dict(simple={str(k): round(float(v), 3) for k, v in simple.mean().items()}, real={str(k): round(float(v), 3) for k, v in real.mean().items()})
json.dump(stats, open(os.path.join(HERE, "paper", "bus_stats.json"), "w"), indent=1)
print(json.dumps(stats))

rs = json.load(open(os.path.join(HERE, "results_final.json")))
rr = json.load(open(os.path.join(HERE, "results_final_real.json")))
groups = [("CAN: dos", "DoS"), ("CAN: fuzzy", "Fuzzy"), ("CAN: spoof_plain", "Spoofing (plain)"), ("CAN: spoof_stealth/brake", "Stealth, brake"),
          ("CAN: spoof_stealth/speed", "Stealth, speed"), ("CAN: spoof_stealth/steer", "Stealth, steer")]
dets = ["CAN timing/ID only", "CAN + signal plausibility", "Fused (CAN + sensor)"]


def mean_auc(r, det, g):
    return np.mean([r[k][det]["group_auc"][g] for k in r if g in r[k][det]["group_auc"]])


L = [BS + "begin{tabular}{lcccccc}", BS + "toprule",
     " & " + BS + "multicolumn{2}{c}{CAN timing/ID} & " + BS + "multicolumn{2}{c}{CAN + plausibility} & " + BS + "multicolumn{2}{c}{Fused} " + BS + BS,
     "CAN attack & Simple & Real. & Simple & Real. & Simple & Real. " + BS + BS, BS + "midrule"]
for g, label in groups:
    cells = []
    for d in dets:
        cells += [f"{mean_auc(rs, d, g):.2f}", f"{mean_auc(rr, d, g):.2f}"]
    L.append(label + " & " + " & ".join(cells) + " " + BS + BS)
L.append(BS + "midrule")
ov = []
for d in dets:
    for r in (rs, rr):
        k = list(r.keys())
        ov.append(f"{r[k[0]][d]['auc']:.2f}/{r[k[1]][d]['auc']:.2f}")
L.append("Overall AUROC (dir. 1/dir. 2) & " + " & ".join(ov) + " " + BS + BS)
L += [BS + "bottomrule", BS + "end{tabular}"]
write("bus_compare.tex", L)
print("\n".join(L))
