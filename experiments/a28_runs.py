"""A-28: initial weight scale and the XOR flat start. Usage: C=2 SEEDS="0 1 2 3" python3 a28_runs.py OUTDIR [--smoke]
Same configuration as A-15's lr 2e-3 row (a12_runs.one with BS 1024, LR 2e-3, 1M fixed traces) with every initial weight and bias multiplied by C."""
import os, torch
import a12_runs as A
C = float(os.environ.get("C", "1")); STEPS = int(os.environ.get("STEPS", "2500"))
_orig = A.make_net2
def make_scaled(*a, **k):
    net = _orig(*a, **k)
    with torch.no_grad():
        for p in net.parameters(): p.mul_(C)
    return net
A.make_net2 = make_scaled
A.BS = 1024; A.LR = 2e-3
for seed in ([int(s) for s in os.environ["SEEDS"].split()] if os.environ.get("SEEDS") else range(1 if A.SMOKE else 4)):
    A.one(f"A-28-c{C}", "xor", 0.5, STEPS, 10**9, False, seed)
