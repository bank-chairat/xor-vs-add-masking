"""Rebuild every results table from the saved files. Run: python3 summarize.py
Reads (if present): results_mlp.json, a03/*.csv, overnight/*.csv, overnight/A-08_parity_tracking.csv.
Primary metric: excess cross entropy in bits = (exact information ceiling) - (network PI); lower is better."""
import csv, glob, json, os
from collections import defaultdict
import numpy as np

def stats(x):
    x = np.array(x, float)
    return f"n={len(x):2d} mean={x.mean():.4f} sd={x.std(ddof=1) if len(x) > 1 else float('nan'):.4f} min={x.min():.4f} max={x.max():.4f}"

if os.path.exists("results_mlp.json"):
    print("== Experiment 2 (A-02): excess CE (bits), 1,000,000 training traces, 4000 steps ==")
    R = json.load(open("results_mlp.json"))
    for sg in (0.5, 1.0):
        for k, N in [("xor", 256), ("add", 256), ("add", 251)]:
            x = [r["ceiling"] - r["pi"] for r in R if r["kind"] == k and r["N"] == N and r["sigma"] == sg and r["n_train"] == 1_000_000]
            if x: print(f"  sigma={sg} {k}{N}: {stats(x)}")

files = sorted(glob.glob("a03/*.csv") + glob.glob("overnight/A-0[3-7].csv"))
groups = defaultdict(list)
best = defaultdict(list)
for f in files:
    for r in csv.DictReader(open(f)):
        shape = r["notes"].split("device")[0].strip() if "width=" in r["notes"] else "width=256 depth=3"
        key = (r["exp_id"], r["masking"], float(r["sigma"]), int(r["n_train"]), int(r["steps"]), float(r["lr"]), shape)
        (groups if r["checkpoint_rule"] == "final" else best)[key].append(float(r["excess_CE_bits"]))
if groups:
    print("\n== Experiments 3 to 7: final-step excess CE (bits) by setting ==")
    for k in sorted(groups):
        e, m, sg, n, st, lr, sh = k
        print(f"  {e} {m:3s} sigma={sg} n_train={n} steps={st} lr={lr} {sh}: {stats(groups[k])}")
    print("\n== Same settings, best-validation checkpoint instead of final step ==")
    for k in sorted(best):
        e, m, sg, n, st, lr, sh = k
        print(f"  {e} {m:3s} sigma={sg} n_train={n} steps={st} lr={lr} {sh}: {stats(best[k])}")

p = "overnight/A-08_parity_tracking.csv"
if os.path.exists(p):
    last = {}
    for r in csv.DictReader(open(p)): last[(r["masking"], int(r["seed"]))] = r
    print("\n== A-08 (noiseless exact table): final step, per masking ==")
    for k in ("xor", "add"):
        rs = [v for (m, s), v in last.items() if m == k]
        if rs:
            print(f"  {k}: excess CE {stats([float(v['excess_CE_bits']) for v in rs])}")
            print(f"  {k}: parity prob {stats([float(v['parity_prob']) for v in rs])}")
