"""README Part 5: rebuilds the numbers of the direct tests (Experiments A-28 to A-35, A-39, A-41b, A-42)
by running each experiment's own scoring script on the saved results, and draws Figure 4.
Run: python3 summarize_part5.py"""
import csv, os, subprocess, sys
from collections import defaultdict
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def run(title, args):
    print(title)
    out = subprocess.run([sys.executable] + args, capture_output=True, text=True).stdout.strip()
    print("  " + out.replace("\n", "\n  "))

run("Label noise added to clean labels (A-30)", ["a30_score.py"])
run("Same test with momentum SGD instead of Adam (A-33)", ["a33_score.py"])
for d, steps, title in [("a34", "2500", "Sine first layer, default and 4x initial frequencies (A-34)"),
                        ("a39", "2500", "Control: ReLU first layer with the same 4x weights (A-39)"),
                        ("a42", "2500", "Sine 4x with a fresh batch every step (A-42)"),
                        ("a28_c2", "2500", "Every initial weight x2 (A-28)"), ("a28_c4", "2500", "x4 (A-28)"),
                        ("a29_c0.5", "2500", "x0.5 (A-29)"), ("a29_c0.25", "2500", "x0.25 (A-29)"),
                        ("a31", "2500", "Hidden layers vs output layer scaled separately (A-31)"),
                        ("a31b", "6000", "Every weight x2, run to 6,000 steps (A-31b)"),
                        ("a31c", "6000", "Output layer x16 only, run to 6,000 steps (A-31c)")]:
    run(title, ["a31_score.py", d, steps])
run("Sine result at a weaker signal, sigma 0.6 (A-41b)", ["a41b_score.py"])

print("Lazy start: first-layer weight movement at step 1,000, output layer x16 vs default (A-35, A-35b)")
mv = defaultdict(dict)
for r in csv.DictReader(open("a35/move.csv")):
    if int(r["step"]) == 1000:
        mv[int(r["seed"])][float(r["CO"])] = float(r["rel_L1"])
for s in sorted(mv):
    print(f"  seed {s}: default {mv[s][1.0]:.3f}, output x16 {mv[s][16.0]:.3f}, ratio {mv[s][16.0] / mv[s][1.0]:.2f}")

# Figure 4: parity part during training, ReLU vs sine first layer (and the ReLU control at the same scale)
def curves(path):
    out = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if int(r["step"]) <= 2500:
            out[int(r["seed"])].append((int(r["step"]), float(r["val_A_parity_group"])))
    return {k: sorted(v) for k, v in out.items()}
fig, ax = plt.subplots(figsize=(6.4, 4.8))
sets = [("a15/A-15-lr0.002_curves.csv", "#c0392b", "ReLU first layer (default)"),
        ("a39/A-39-relu-w4.0_curves.csv", "#7b241c", "ReLU first layer, weights x4 (control)"),
        ("a34/A-34-sin-w1.0_curves.csv", "#85c1e9", "sine first layer, default weights"),
        ("a34/A-34-sin-w4.0_curves.csv", "#1f618d", "sine first layer, weights x4")]
for path, color, label in sets:
    for i, (seed, c) in enumerate(sorted(curves(path).items())):
        ax.plot([s for s, _ in c], [a for _, a in c], color=color, lw=1.3, alpha=0.9, label=label if i == 0 else None)
ax.set_xlabel("training step")
ax.set_ylabel("parity part of the gap (bits)")
ax.set_ylim(bottom=0)
ax.set_title("XOR with one-hot labels: ReLU vs. sine first layer (σ = 0.5)")
ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
fig.tight_layout()
os.makedirs("results", exist_ok=True)
fig.savefig("results/part5_sine_layer.png", dpi=150)
print("figure: results/part5_sine_layer.png")
