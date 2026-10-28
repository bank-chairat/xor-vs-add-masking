"""README Part 3: rebuilds the tables of Experiments 9-11 from the saved results and draws Figure 2.
Reads a09/A-09.csv, a10/A-10.csv, a11/A-11_curves.csv, a11/A-11_final.csv. Run: python3 summarize_part3.py
'Parity part' = extra cost of getting the parity group of the secret (even/odd Hamming weight) right,
'within part' = extra cost of picking the secret inside its group; both relative to the exact Bayes posterior."""
import csv, os
from collections import defaultdict
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def table(path, steps=4000):
    g = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if int(r["steps"]) == steps:
            g[(r["masking"], r["feature"])].append((float(r["excess_final"]), float(r["between_excess_final"]),
                                                     float(r["within_excess_final"])))
    return {k: np.mean(v, axis=0) for k, v in g.items()}, {k: len(v) for k, v in g.items()}

print("Experiment 9, where the shortfall lives (gap, parity part, within part; bits; mean of seeds)")
for st in (4000, 16000):
    m, n = table("a09/A-09.csv", st)
    for key in [("xor", "F0"), ("add", "F0")]:
        print(f"  {key[0]:3s} {st:>6} steps: gap {m[key][0]:.4f}  parity {m[key][1]:.4f}  within {m[key][2]:.4f}  (n={n[key]})")
print("Experiments 9-10, extra inputs at 4,000 steps (XOR; addition in brackets)")
m9, _ = table("a09/A-09.csv"); m10, _ = table("a10/A-10.csv")
m = {**m9, **m10}
names = {"F0": "plain inputs", "F1": "cos(pi(r1+r2))", "F2": "cos, sin of pi r1 and pi r2", "F3": "r1+r2 (linear control)",
         "G1": "cos(pi/2 (r1+r2)) half frequency", "G2": "cos(2pi (r1+r2)) double frequency",
         "G3": "cos(pi(r1+r2)+pi/4) shifted phase", "G4": "16 random Fourier features"}
for f, name in names.items():
    x, a = m[("xor", f)], m[("add", f)]
    print(f"  {f} {name:38s} XOR gap {x[0]:.4f} parity {x[1]:.4f} within {x[2]:.4f}   [add gap {a[0]:.4f}]")

print("Experiment 11, parity part on validation traces during training (plain XOR)")
curves = defaultdict(list)
for r in csv.DictReader(open("a11/A-11_curves.csv")):
    curves[(r["masking"], r["feature"], int(r["seed"]))].append((int(r["step"]), float(r["val_A_parity_group"])))
for key in sorted(curves):
    c = sorted(curves[key]); start = c[0][1]
    half = next((s for s, a in c if a < start / 2), None)
    if key[0] == "xor":
        print(f"  {key[0]} {key[1]} seed {key[2]}: start {start:.4f}, at 1000 {dict(c)[1000]:.4f}, "
              f"at 4000 {dict(c)[4000]:.4f}, first below half: {half if half else 'never'}")

fig, ax = plt.subplots(figsize=(6.4, 4))
style = {("xor", "F0"): ("#c0392b", "XOR, plain inputs"), ("xor", "F1"): ("#e59866", "XOR, plus cos(π(r1+r2))"),
         ("add", "F0"): ("#2471a3", "Addition, plain inputs")}
for key in sorted(curves):
    color, label = style[key[:2]]
    c = sorted(curves[key])
    ax.plot([s for s, _ in c], [a for _, a in c], color=color, lw=1.4, alpha=0.9,
            label=label if key[2] == 0 else None)
ax.set_xlabel("training step")
ax.set_ylabel("parity part of the gap (bits)")
ax.set_ylim(bottom=0)
ax.set_title("When is the parity clue learned? (σ = 0.5, one line per seed)")
ax.legend(frameon=False)
fig.tight_layout()
os.makedirs("results", exist_ok=True)
fig.savefig("results/part3_parity_curves.png", dpi=150)
print("figure: results/part3_parity_curves.png")
