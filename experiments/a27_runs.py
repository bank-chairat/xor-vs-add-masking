"""A-27: flat-start law outside the fitted range. XOR, sigma 0.5, constant Adam lr 2.5e-4, fixed 1M traces. Usage: CONF=tag python3 a27_runs.py OUTDIR [--smoke]"""
import os
import a12_runs as A
CONF = [("lr2.5e-4_bs2048", 2.5e-4, 2048, 4000), ("lr2.5e-4_bs8192", 2.5e-4, 8192, 3000), ("lr2.5e-4_bs16384", 2.5e-4, 16384, 3000)]
only = os.environ.get("CONF")
for tag, lr, bs, steps in CONF:
    if only and tag not in only.split(","): continue
    A.BS = bs; A.LR = lr
    for seed in ([int(s) for s in os.environ["SEEDS"].split()] if os.environ.get("SEEDS") else range(1 if A.SMOKE else 4)):
        A.one(f"A-27-{tag}", "xor", 0.5, steps, 10**9, False, seed)
