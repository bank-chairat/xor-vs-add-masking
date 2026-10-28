"""Figure for README Chapter 2: distance from the information ceiling (excess cross entropy, bits)
after 4,000, 16,000 and 64,000 training steps, XOR vs addition mod 256, sigma 0.5, 1,000,000 traces,
final-step checkpoint. One dot per seed, line through the means. Reads the saved Experiment 3 results
(a03/*.csv and overnight/A-03.csv, the same files summarize.py uses). Run: python3 plot_ch2.py"""
import csv, glob, os
from collections import defaultdict
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

vals = defaultdict(list)
for f in sorted(glob.glob("a03/*.csv") + ["overnight/A-03.csv"]):
    for r in csv.DictReader(open(f)):
        if r["exp_id"] == "A-03" and r["checkpoint_rule"] == "final":
            vals[(r["masking"], int(r["steps"]))].append(float(r["excess_CE_bits"]))

steps = [4000, 16000, 64000]
fig, ax = plt.subplots(figsize=(6, 4))
for k, (name, color) in enumerate([("xor", "#c0392b"), ("add", "#2471a3")]):
    means = []
    for i, s in enumerate(steps):
        v = vals[(name, s)]
        x = i + (k - 0.5) * 0.12
        ax.scatter([x] * len(v), v, color=color, alpha=0.5, s=18)
        means.append(np.mean(v))
        print(f"{name} {s}: n={len(v)} mean={np.mean(v):.4f}")
    ax.plot([i + (k - 0.5) * 0.12 for i in range(3)], means, color=color, marker="o",
            label="XOR" if name == "xor" else "Addition mod 256")
ax.set_xticks(range(3), ["4,000", "16,000", "64,000"])
ax.set_xlabel("training steps")
ax.set_ylabel("distance from ceiling (excess cross entropy, bits)")
ax.set_ylim(bottom=0)
ax.set_title("Networks vs. the information ceiling (σ = 0.5)")
ax.legend(frameon=False)
fig.tight_layout()
os.makedirs("results", exist_ok=True)
fig.savefig("results/ch2_gap_vs_steps.png", dpi=150)
