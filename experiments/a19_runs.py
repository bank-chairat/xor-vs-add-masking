"""A-19: is the flat start set by a noise-scale combination of learning rate and batch size? XOR, sigma 0.5, constant Adam, 4000 steps, width 256, 4 seeds.
Configs (lr, batch): ratio family lr/B = 2e-3/1024: (1e-3,512), (4e-3,2048); sqrt family lr/sqrt(B) = 2e-3/32: (1e-3,256), (4e-3,4096).
Usage: python3 a19_runs.py OUTDIR [--smoke]  (resumable via a12_runs.one)"""
import os
import a12_runs as A
CONF = [("lr1e-3_bs512", 1e-3, 512), ("lr4e-3_bs2048", 4e-3, 2048), ("lr1e-3_bs256", 1e-3, 256), ("lr4e-3_bs4096", 4e-3, 4096)]
only = os.environ.get("CONF")
for tag, lr, bs in CONF:
    if only and tag not in only.split(","): continue
    A.BS = bs; A.LR = lr
    for seed in range(1 if A.SMOKE else 4):
        A.one(f"A-19-{tag}", "xor", 0.5, 4000, 10**9, False, seed)
