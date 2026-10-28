"""A-40 scoring: transition-leak learning with exact-posterior (soft) labels (a40/), against A-38 one-hot labels (a38/)."""
import csv, numpy as np
def load(d): return [r for r in csv.DictReader(open(f"{d}/A-38_curves.csv")) if r["kind"] in ("xor", "add")]
S, H = load("a40"), load("a38")
def end(R, kind): return {int(r["seed"]): float(r["val_total_excess_bits"]) for r in R if r["kind"] == kind and int(r["step"]) == 4000}
for kind in ("xor", "add"):
    s, h = end(S, kind), end(H, kind)
    print(f"{kind}: soft excess at 4000 per seed {[round(s[k], 4) for k in sorted(s)]} mean {np.mean(list(s.values())):.4f} | one-hot (A-38) mean {np.mean(list(h.values())):.4f}")
sx, sa = end(S, "xor"), end(S, "add")
if len(sx) == 4 and len(sa) == 4:
    mx, ma = np.mean(list(sx.values())), np.mean(list(sa.values()))
    print(f"(1) both soft means at most 0.010: {'HIT' if max(mx, ma) <= 0.010 else 'MISSED'} ({mx:.4f}, {ma:.4f})")
    print(f"(2) XOR minus addition gap at most 0.0046 (a third of A-38's 0.0137): {'HIT' if mx - ma <= 0.0046 else 'MISSED'} ({mx - ma:+.4f})")
