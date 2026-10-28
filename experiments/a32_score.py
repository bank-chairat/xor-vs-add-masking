"""A-32 scoring against the four predictions in PROGRESS.md (weighted-bit leakage, total excess in bits on 100,000 validation traces)."""
import csv, numpy as np
R = list(csv.DictReader(open("a32/A-32_curves.csv")))
def cur(kind, lab, seed): return {int(r["step"]): float(r["val_total_excess_bits"]) for r in R if r["kind"] == kind and r["labels"] == lab and int(r["seed"]) == seed}
have = {(r["kind"], r["labels"]) for r in R}
for kind, lab in sorted(have):
    for s in range(4):
        c = cur(kind, lab, s)
        if c: print(f"{kind} {lab} seed {s}: excess at 0/100/600/1000/2000/4000: " + " ".join(f"{c.get(t, float('nan')):.4f}" for t in (0, 100, 600, 1000, 2000, 4000)) + (f"; fall 100->600 {100 * (1 - c[600] / c[100]):.0f}%" if 600 in c else ""))
def fall(kind, lab): return [1 - cur(kind, lab, s)[600] / cur(kind, lab, s)[100] for s in range(4) if 600 in cur(kind, lab, s)]
if ("xor", "hard") in have:
    f = fall("xor", "hard"); print(f"(1) XOR hard plateau (fall < 10% in >= 3 of 4): {sum(x < 0.10 for x in f)} of {len(f)} -> {'HIT' if sum(x < 0.10 for x in f) >= 3 else 'MISSED'}")
if ("xor", "soft") in have:
    f = fall("xor", "soft"); print(f"(2) XOR soft falls > 50% in >= 3 of 4: {sum(x > 0.5 for x in f)} of {len(f)} -> {'HIT' if sum(x > 0.5 for x in f) >= 3 else 'MISSED'}")
    h = np.mean([cur("xor", "hard", s)[4000] for s in range(4)]); so = np.mean([cur("xor", "soft", s)[4000] for s in range(4) if 4000 in cur("xor", "soft", s)])
    print(f"(3) XOR hard excess at 4000 >= 3 x soft: means {h:.4f} vs {so:.4f} (ratio {h / so:.0f}) -> {'HIT' if h >= 3 * so else 'MISSED'}")
if ("add", "hard") in have:
    pairs = [(cur("xor", "hard", s)[4000], cur("add", "hard", s)[4000]) for s in range(4) if 4000 in cur("add", "hard", s)]
    k = sum(x > a for x, a in pairs); print(f"(4) XOR hard > addition hard at 4000 in >= 3 of 4 paired seeds: {k} of {len(pairs)} -> {'HIT' if k >= 3 else 'MISSED'}; pairs " + ", ".join(f"{x:.4f}/{a:.4f}" for x, a in pairs))
