"""A-34: does a learnable periodic (sine) first layer remove the XOR flat start under one-hot labels (hypothesis H3)?
A-15 configuration (XOR, sigma 0.5, 3 x 256, Adam constant 2e-3, batch 1,024, 1M fixed traces, one-hot labels), 2,500 steps,
first hidden activation ReLU -> sin; first-layer weights (not bias) times WS at initialisation (WS=1 default init, WS=4 wider).
SIG sets the noise level (A-41); FRESH=1 draws a new batch from the simulator every step (A-42). ACT=relu gives the A-39 control (first layer ReLU, only its weights x WS). Usage: [ACT=relu] WS=1 SEEDS="0 1 2 3" python3 a34_periodic.py OUTDIR"""
import os, torch
import a12_runs as A
WS = float(os.environ.get("WS", "1")); ACT = os.environ.get("ACT", "sin"); SIGMA = float(os.environ.get("SIG", "0.5")); FRESH = os.environ.get("FRESH", "0") == "1"; STEPS = int(os.environ.get("STEPS", "2500"))
class Sin(torch.nn.Module):
    def forward(self, x): return torch.sin(x)
_orig = A.make_net2
def make_sin(*a, **k):
    net = _orig(*a, **k)
    if ACT == "sin": net[1] = Sin()
    with torch.no_grad(): net[0].weight.mul_(WS)
    return net
A.make_net2 = make_sin
A.BS = 1024; A.LR = 2e-3
for seed in [int(s) for s in os.environ["SEEDS"].split()]:
    A.one((f"A-34-sin-w{WS}" if ACT == "sin" else f"A-39-relu-w{WS}") + ("" if SIGMA == 0.5 else f"-sig{SIGMA}") + ("-fresh" if FRESH else ""), "xor", SIGMA, STEPS, 10**9, FRESH, seed)
