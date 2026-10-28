"""A-15: constant learning rate sweep (tests whether the length of XOR's flat start is set by the learning rate or by the number of steps).
XOR only, sigma 0.5, plain inputs, Adam, constant lr (cosine length 1e9 steps is flat to 1e-9), 4000 steps, 4 seeds.
Usage: python3 a15_runs.py OUTDIR [--smoke]  (same conventions as a12_runs.py; OUTDIR is a12_runs's OUT)"""
import a12_runs as A
for lr in (5e-4, 1e-3, 2e-3, 4e-3):
    A.LR = lr
    for seed in range(1 if A.SMOKE else 4):
        A.one(f"A-15-lr{lr}", "xor", 0.5, 4000, 10**9, False, seed)
