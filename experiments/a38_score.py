"""A-38 scoring (transition-leak learning)."""
import csv, numpy as np
R = [r for r in csv.DictReader(open("a38/A-38_curves.csv")) if r["kind"] in ("xor", "add")]
def cur(kind, s): return {int(r["step"]): float(r["val_total_excess_bits"]) for r in R if r["kind"] == kind and int(r["seed"]) == s}
MI = {"xor": 1.6335, "add": 0.8095}
for kind in ("xor", "add"):
    for s in range(4):
        c = cur(kind, s)
        if c: print(f"{kind} seed {s}: excess at 0/100/600/1000/2000/4000 " + " ".join(f"{c.get(t, float('nan')):.4f}" for t in (0, 100, 600, 1000, 2000, 4000)) + f"; fall 100->600 {100 * (1 - c[600] / c[100]):.0f}%; captured {100 * (1 - c[4000] / MI[kind]):.1f}% of the information")
f = [1 - cur("xor", s)[600] / cur("xor", s)[100] for s in range(4) if cur("xor", s)]
print(f"(1) XOR excess falls > 50% from 100 to 600 in >= 3 of 4: {sum(x > 0.5 for x in f)} of {len(f)} -> {'HIT' if sum(x > 0.5 for x in f) >= 3 else 'MISSED'}")
pairs = [(cur("xor", s)[4000], cur("add", s)[4000]) for s in range(4) if cur("xor", s) and cur("add", s)]
k = sum(x <= a for x, a in pairs); print(f"(2) XOR excess at 4000 <= addition's in >= 3 of 4 paired seeds: {k} of {len(pairs)} -> {'HIT' if k >= 3 else 'MISSED'}; pairs " + ", ".join(f"{x:.4f}/{a:.4f}" for x, a in pairs))
