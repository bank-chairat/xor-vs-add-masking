"""A-43: does a LayerNorm before each ReLU change XOR's flat start? A-15 configuration (XOR, sigma 0.5, 3 x 256, Adam constant 2e-3, batch 1,024,
1M fixed traces, one-hot labels), 2,500 steps. Usage: SEEDS="0 1 2 3" python3 a43_layernorm.py OUTDIR"""
import os, torch, torch.nn as nn
import a12_runs as A
def make_ln(d_in, width=256, depth=3):
    layers, d = [], d_in
    for _ in range(depth): layers += [nn.Linear(d, width), nn.LayerNorm(width), nn.ReLU()]; d = width
    layers.append(nn.Linear(d, A.N)); return nn.Sequential(*layers)
A.make_net2 = make_ln
A.BS = 1024; A.LR = 2e-3
for seed in [int(s) for s in os.environ["SEEDS"].split()]:
    A.one("A-43-ln", "xor", 0.5, 2500, 10**9, False, seed)
