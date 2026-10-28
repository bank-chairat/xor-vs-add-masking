"""README Part 6: other leakage models. Prints the information results (A-32 weighted bits, A-37 transition
leak), reruns the learning scorers (A-32, A-38, A-40) on the saved results, and draws Figure 5.
Run: python3 summarize_part6.py"""
import os, re, subprocess, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ceiling import weight_table, gaussian_mi

def run(title, args):
    print(title)
    out = subprocess.run([sys.executable] + args, capture_output=True, text=True).stdout.strip()
    print("  " + out.replace("\n", "\n  "))

hw = {k: gaussian_mi(weight_table(k, 256), 0.5)[0] for k in ("xor", "add")}   # returns (bits, density check)
print(f"Hamming-weight leakage (Part 1, exact): XOR {hw['xor']:.4f}, addition {hw['add']:.4f} bits")
print("Weighted-bit leakage (A-32, Monte Carlo):\n  " + open("a32_info.txt").read().strip().replace("\n", "\n  "))
print("Transition leak (A-37, Monte Carlo):\n  " + open("a37_info.txt").read().strip().replace("\n", "\n  "))
run("Learning under weighted-bit leakage (A-32)", ["a32_score.py"])
run("Learning under the transition leak, one-hot labels (A-38)", ["a38_score.py"])
run("Same with exact-posterior labels (A-40)", ["a40_score.py"])

wb = {k: float(re.search(rf"{k}: MI estimate ([0-9.]+)", open("a32_info.txt").read()).group(1)) for k in ("xor", "add")}
tr = {k: float(re.search(rf"{k} with transition reading: I = ([0-9.]+)", open("a37_info.txt").read()).group(1)) for k in ("xor", "add")}
models = ["Hamming weight\n(Parts 1-5)", "weighted bits\n(Exp. 32)", "Hamming weight +\ntransition leak (Exp. 37)"]
x = np.arange(3)
fig, ax = plt.subplots(figsize=(6.4, 4))
for i, (k, name, color) in enumerate([("xor", "XOR", "#c0392b"), ("add", "Addition mod 256", "#2471a3")]):
    vals = [hw[k], wb[k], tr[k]]
    bars = ax.bar(x + (i - 0.5) * 0.36, vals, 0.36, color=color, label=name)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.3f}", ha="center", fontsize=8.5)
ax.set_yscale("log")
ax.set_ylim(0.05, 4)
ax.set_xticks(x, models)
ax.set_ylabel("information about the secret (bits, log scale)")
ax.set_title("Information ceiling under three leakage models (σ = 0.5)")
ax.legend(frameon=False)
fig.tight_layout()
os.makedirs("results", exist_ok=True)
fig.savefig("results/part6_leakage_models.png", dpi=150)
print("figure: results/part6_leakage_models.png")
