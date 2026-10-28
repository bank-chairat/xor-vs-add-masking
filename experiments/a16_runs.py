"""A-16: what sets the ~1,000-step floor of the flat start? Batch size and network width at constant learning rate 2e-3.
XOR, sigma 0.5, plain inputs, Adam constant lr 2e-3, 4000 steps, 4 seeds. Configs: batch 256, batch 4096, width 64, width 1024
(the default batch 1024, width 256 is A-15's lr 2e-3 run). Usage: python3 a16_runs.py OUTDIR [--smoke]"""
import a12_runs as A
orig = A.make_net2
for tag, bs, width in (("bs256", 256, 256), ("bs4096", 4096, 256), ("w64", 1024, 64), ("w1024", 1024, 1024)):
    A.BS = bs; A.make_net2 = lambda d_in, width=width: orig(d_in, width=width); A.LR = 2e-3
    for seed in range(1 if A.SMOKE else 4):
        A.one(f"A-16-{tag}", "xor", 0.5, 4000, 10**9, False, seed)
