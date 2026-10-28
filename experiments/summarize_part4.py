"""README Part 4: rebuilds the numbers of Experiments 12-27 from the saved results and draws Figure 3.
Flat start = first evaluated step (a multiple of 100) at which the validation parity part A is below
90% of its value at step 25. Run: python3 summarize_part4.py"""
import csv, glob, os
from collections import defaultdict
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def curves(pattern, key="seed"):
    out = defaultdict(list)
    for f in sorted(glob.glob(pattern)):
        for r in csv.DictReader(open(f)):
            out[(r["run_id"].rsplit("-seed", 1)[0], int(r[key]))].append((int(r["step"]), float(r["val_A_parity_group"])))
    return {k: sorted(v) for k, v in out.items()}

def flat_start(c):
    ref = dict(c)[25]
    return next((s for s, a in c if s >= 100 and s % 100 == 0 and a < 0.9 * ref), None)

def report(title, pattern, filt=lambda run: True):
    print(title)
    g = defaultdict(list)
    for (run, seed), c in curves(pattern).items():
        if filt(run):
            g[run].append((seed, flat_start(c)))
    for run in sorted(g):
        fs = [x for _, x in sorted(g[run])]
        med = np.median([x if x is not None else 4000 for x in fs])
        print(f"  {run}: flat start per seed {['never' if x is None else x for x in fs]}  median {med:.0f}")

report("Signal strength (Experiment 14, XOR, noise sigma)", "a12/A-14_curves.csv", lambda r: "-xor-" in r)
report("Learning rate (Experiment 15, XOR, constant Adam, batch 1,024)", "a15/*_curves.csv")
report("Batch size and width (Experiment 16, XOR, lr 2e-3)", "a16/*_curves.csv")
report("Soft labels from the exact posterior (Experiment 17)", "a17/*_curves.csv")
report("Share of one-hot label in the target (Experiment 18)", "a18/*_curves.csv")
report("Rule-of-thumb checks (Experiments 19, 22 = R-7, 27)", "a19*/*_curves.csv")
report("", "a22*/*_curves.csv")
report("", "a27*/*_curves.csv")

print("Long runs, fixed 1M traces vs a fresh batch every step (Experiments 20 and 25, 64,000 steps, test gap)")
for d in ["a20f", "a20r", "a25f", "a25r"]:
    for r in csv.DictReader(open(glob.glob(d + "/*_final.csv")[0])):
        print(f"  {r['masking']:3s} {'fresh' if r['fresh'] == '1' else 'fixed'} seed {r['seed']}: gap {float(r['test_total']):.4f}"
              f"  (secret-in-class part {float(r['test_C']):.4f})")

print("Gradient noise, one-hot vs exact-posterior labels (Experiment 26)")
rows = list(csv.DictReader(open("a26/A-26_snr.csv")))
by = defaultdict(dict)
for r in rows:
    by[(int(r["seed"]), int(r["step"]))][r["label"]] = r
for (seed, step) in sorted(by):
    d = by[(seed, step)]
    if "H" in d and "S" in d and step in (0, 1000, 2000):
        nr = float(d["H"]["N256"]) / float(d["S"]["N256"])
        snr = float(d["H"]["S"]) / (float(d["H"]["N256"]) / 4)
        print(f"  seed {seed} step {step}: noise ratio hard/soft {nr:.0f}" + (f", hard-label signal-to-noise at batch 1,024 {snr:.2f}" if step == 0 else ""))

# Figure 3: parity part during training, one-hot labels at three batch sizes vs exact-posterior labels
fig, ax = plt.subplots(figsize=(6.4, 4.8))
sets = [("a16/A-16-bs256_curves.csv", "#7b241c", "one-hot labels, batch 256"),
        ("a15/A-15-lr0.002_curves.csv", "#c0392b", "one-hot labels, batch 1,024"),
        ("a16/A-16-bs4096_curves.csv", "#e59866", "one-hot labels, batch 4,096"),
        ("a17/A-17_curves.csv", "#1e8449", "exact-posterior (soft) labels, batch 1,024")]
for path, color, label in sets:
    for i, ((run, seed), c) in enumerate(sorted(curves(path).items())):
        ax.plot([s for s, _ in c], [a for _, a in c], color=color, lw=1.3, alpha=0.9, label=label if i == 0 else None)
ax.set_xlabel("training step")
ax.set_ylabel("parity part of the gap (bits)")
ax.set_ylim(bottom=0)
ax.set_title("XOR flat start: batch size and label noise (σ = 0.5)")
ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
fig.tight_layout()
os.makedirs("results", exist_ok=True)
fig.savefig("results/part4_flat_start.png", dpi=150)
print("figure: results/part4_flat_start.png")
