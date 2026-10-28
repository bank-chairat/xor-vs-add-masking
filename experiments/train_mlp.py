"""Train MLPs on simulated masked HW leakage and compare PI to the exact ceiling.

PI (perceived information, bits) = log2(N) - cross_entropy_in_bits, measured on
a large fresh test set. By Jensen it cannot exceed the true mutual information
(up to sampling error), so PI / ceiling tells us how much of the available
information the network extracted.
"""
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from ceiling import HW, CASES, weight_table, gaussian_mi

torch.set_num_threads(2)


def make_data(kind, N, sigma, n, rng):
    s = rng.integers(0, N, n)
    m = rng.integers(0, N, n)
    sh2 = (s ^ m) if kind == "xor" else (s + m) % N
    x = np.stack([HW[m], HW[sh2]], axis=1).astype(np.float32)
    x += rng.normal(0, sigma, x.shape).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(s.astype(np.int64))


def make_net(N, width=256, depth=3):
    layers, d = [], 2
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.ReLU()]
        d = width
    layers.append(nn.Linear(d, N))
    return nn.Sequential(*layers)


@torch.no_grad()
def pi_bits(net, x, y, N, bs=50000):
    tot = 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")
    losses = []
    for i in range(0, len(x), bs):
        lg = net(x[i:i + bs])
        losses.append(nn.functional.cross_entropy(lg, y[i:i + bs], reduction="none"))
    l = torch.cat(losses) / np.log(2)
    pi = np.log2(N) - l.mean().item()
    se = l.std().item() / np.sqrt(len(l))
    return pi, se


def run(kind, N, sigma, n_train, seed, steps=6000, bs=1024, lr=2e-3, eval_every=250,
        val=None, test=None):
    rng = np.random.default_rng(1000 + seed)
    torch.manual_seed(seed)
    xtr, ytr = make_data(kind, N, sigma, n_train, rng)
    net = make_net(N)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    best_val, best_state = -1e9, None
    for step in range(1, steps + 1):
        idx = torch.randint(0, n_train, (min(bs, n_train),))
        loss = nn.functional.cross_entropy(net(xtr[idx]), ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % eval_every == 0:
            v, _ = pi_bits(net, *val, N)
            if v > best_val:
                best_val = v
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    t_pi, t_se = pi_bits(net, *test, N)
    return t_pi, t_se, best_val


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results_mlp.json"
    sigmas = [float(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.5, 1.0]
    sizes = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [2000, 10000, 50000, 250000, 1000000]
    seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 6000

    results = []
    for kind, N in CASES:
        T = weight_table(kind, N)
        for sigma in sigmas:
            ceil, _ = gaussian_mi(T, sigma)
            drng = np.random.default_rng(7)
            val = make_data(kind, N, sigma, 100000, drng)
            test = make_data(kind, N, sigma, 400000, drng)
            for n_train in sizes:
                for seed in range(seeds):
                    t0 = time.time()
                    pi, se, v = run(kind, N, sigma, n_train, seed, steps=steps, val=val, test=test)
                    rec = dict(kind=kind, N=N, sigma=sigma, n_train=n_train, seed=seed,
                               pi=pi, se=se, val_pi=v, ceiling=ceil)
                    results.append(rec)
                    print(f"{kind}{N} s={sigma} n={n_train:>8} seed={seed} PI={pi:.4f}+-{se:.4f} "
                          f"ceil={ceil:.4f} frac={pi/ceil:.2f} ({time.time()-t0:.0f}s)", flush=True)
                    json.dump(results, open(out, "w"), indent=1)
