"""A-31: which part of the initial scale stalls the XOR flat start (A-28)? Hidden layers (weights and biases) times CH, output layer times CO.
Same configuration as A-28 (A-15 lr 2e-3 row, batch 1,024, 1M fixed traces, 2,500 steps). Usage: CH=2 CO=0.125 SEEDS="0 1 2 3" python3 a31_runs.py OUTDIR"""
import os, torch
import a12_runs as A
CH = float(os.environ.get("CH", "1")); CO = float(os.environ.get("CO", "1")); STEPS = int(os.environ.get("STEPS", "2500"))
_orig = A.make_net2
def make_scaled(*a, **k):
    net = _orig(*a, **k); lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for m in lin[:-1]:
            for p in m.parameters(): p.mul_(CH)
        for p in lin[-1].parameters(): p.mul_(CO)
    return net
A.make_net2 = make_scaled
A.BS = 1024; A.LR = 2e-3
for seed in [int(s) for s in os.environ["SEEDS"].split()]:
    A.one(f"A-31-h{CH}-o{CO}", "xor", 0.5, STEPS, 10**9, False, seed)
