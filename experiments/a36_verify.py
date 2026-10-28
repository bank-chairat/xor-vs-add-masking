"""A-36 (check): re-run the scoring scripts of A-30 to A-35 on the raw curve files and compare with the saved score files; recompute the A-32 means quoted in PROGRESS.md."""
import subprocess, sys, csv, os, numpy as np
def run(args): return subprocess.run([sys.executable] + args, capture_output=True, text=True).stdout.strip()
checks = [(["a30_score.py"], "a30_scores.txt"), (["a31_score.py", "a31", "2500"], "a31_scores.txt"), (["a31_score.py", "a31b", "6000"], "a31b_scores.txt"),
          (["a31_score.py", "a31c", "6000"], "a31c_scores.txt"), (["a32_score.py"], "a32_scores.txt"), (["a33_score.py"], "a33_scores.txt"), (["a31_score.py", "a34", "2500"], "a34_scores.txt")]
for args, saved in checks:
    if os.path.exists(saved): print(f"{' '.join(args)} -> identical to {saved}: {run(args) == open(saved).read().strip()}")
R = list(csv.DictReader(open("a32/A-32_curves.csv")))
for kind, lab in [("xor", "hard"), ("xor", "soft"), ("add", "hard")]:
    m = [np.mean([float(r["val_total_excess_bits"]) for r in R if r["kind"] == kind and r["labels"] == lab and int(r["step"]) == t]) for t in (100, 600, 4000)]
    print(f"A-32 {kind} {lab} means at 100/600/4000: " + " / ".join(f"{x:.4f}" for x in m))
if os.path.exists("a35/move.csv"):
    M = list(csv.DictReader(open("a35/move.csv")))
    for r in M:
        if int(r["step"]) in (1000, 2000): print("A-35", r["CO"], r["seed"], r["step"], "A", r["valA"], "rel L1/L2/L3/out", r["rel_L1"], r["rel_L2"], r["rel_L3"], r["rel_out"])
