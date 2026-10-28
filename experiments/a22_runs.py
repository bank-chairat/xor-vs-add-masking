"""A-22 (R-7): prospective test of the flat-start formula. XOR, sigma 0.5, constant Adam, 4000 steps, width 256, 4 seeds. Usage: CONF=tag,tag python3 a22_runs.py OUTDIR [--smoke]"""
import os
import a12_runs as A
CONF = [("lr2e-3_bs512", 2e-3, 512), ("lr1e-3_bs2048", 1e-3, 2048), ("lr3e-3_bs1024", 3e-3, 1024), ("lr5e-4_bs4096", 5e-4, 4096)]
only = os.environ.get("CONF")
for tag, lr, bs in CONF:
    if only and tag not in only.split(","): continue
    A.BS = bs; A.LR = lr
    for seed in ([int(s) for s in os.environ["SEEDS"].split()] if os.environ.get("SEEDS") else range(1 if A.SMOKE else 4)):
        A.one(f"A-22-{tag}", "xor", 0.5, 4000, 10**9, False, seed)
