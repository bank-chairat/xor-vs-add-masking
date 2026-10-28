"""A-33 scoring: flat start = first evaluated step (multiple of 100, >= 100) with validation parity part A below 0.9 x 0.0353; censored at 6,000."""
import csv, numpy as np
R = list(csv.DictReader(open("a33/A-33_curves.csv"))); acol = [k for k in R[0] if "A" in k and "val" in k][0]
med = {}
for lab in ["soft", "hard"]:
    out, aend = [], []
    for seed in range(4):
        rr = sorted([r for r in R if r["run_id"].startswith(f"A-33-{lab}-") and r["run_id"].endswith(f"seed{seed}")], key=lambda r: int(r["step"]))
        fs = next((int(r["step"]) for r in rr if int(r["step"]) >= 100 and int(r["step"]) % 100 == 0 and float(r[acol]) < 0.9 * 0.0353), -1)
        out.append(fs); aend.append(float(rr[-1][acol]))
        a_at = {int(r["step"]): float(r[acol]) for r in rr}
        print(f"{lab} seed {seed}: flat start {fs}; A at 1000/2000/3000/6000: " + " ".join(f"{a_at.get(s, float('nan')):.4f}" for s in (1000, 2000, 3000, 6000)))
    cens = sum(x < 0 for x in out); m = np.median([x if x > 0 else 6000 for x in out]); med[lab] = m
    print(f"{lab}: flat starts {out}, median {m} (censored counted as 6,000), censored {cens}/4, A at end mean {np.mean(aend):.4f}")
print(f"(1) soft median <= 2000: {'HIT' if med['soft'] <= 2000 else 'MISSED'} ({med['soft']})")
hard = med['hard']; print(f"(2) hard median >= 2 x soft or censored in >= 3/4: ratio {hard / med['soft']:.2f}")
