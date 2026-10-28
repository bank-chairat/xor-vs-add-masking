"""A-41b scoring (sigma 0.6): flat start = first evaluated step (multiple of 100, >= 200) with validation A below 0.9 x A_ref, A_ref = median over the ReLU seeds of A at step 100."""
import csv, os, numpy as np
def runs(f):
    R = list(csv.DictReader(open(f))); out = {}
    for s in range(4):
        rr = sorted([r for r in R if r["run_id"].endswith(f"seed{s}")], key=lambda r: int(r["step"]))
        if rr: out[s] = {int(r["step"]): float(r["val_A_parity_group"]) for r in rr}
    return out
relu = runs("a41b/A-39-relu-w1.0-sig0.6_curves.csv"); aref = float(np.median([relu[s][100] for s in relu])); print(f"A_ref (median ReLU A at step 100) = {aref:.5f}; threshold {0.9 * aref:.5f}")
def fs(c): return next((t for t in sorted(c) if t >= 200 and t % 100 == 0 and c[t] < 0.9 * aref), -1)
res = {}
for name, f in [("ReLU", "a41b/A-39-relu-w1.0-sig0.6_curves.csv"), ("sine x4", "a41b/A-34-sin-w4.0-sig0.6_curves.csv")]:
    if not os.path.exists(f): continue
    r = runs(f); flat = [fs(r[s]) for s in sorted(r)]; med = np.median([x if x > 0 else 4000 for x in flat]); res[name] = (flat, med, [r[s][4000] for s in sorted(r)])
    print(f"{name}: flat start per seed {flat} (median {med:.0f}, censored counted as 4,000); A at 4,000 {[round(r[s][4000], 4) for s in sorted(r)]}")
fl, m, _ = res["ReLU"]; print(f"(1) ReLU median >= 2,000 or censored in >= 3 of 4: {'HIT' if m >= 2000 or sum(x < 0 for x in fl) >= 3 else 'MISSED'}")
if "sine x4" in res and len(res["sine x4"][0]) == 4:
    fs_, ms, a_s = res["sine x4"]; a_r = res["ReLU"][2]
    print(f"(2) sine median <= half ReLU median ({ms:.0f} vs {m:.0f}) and A at 4,000 below ReLU in all 4 paired seeds ({sum(x < y for x, y in zip(a_s, a_r))} of 4): {'HIT' if ms <= m / 2 and all(x < y for x, y in zip(a_s, a_r)) else 'MISSED'}")
