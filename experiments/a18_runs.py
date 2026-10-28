"""A-18: mixture lambda * one-hot + (1 - lambda) * exact posterior (derived from a17_runs.py). XOR, sigma 0.5, constant Adam 2e-3, batch 1024, 4000 steps, seeds 0-3.
Usage: python3 a17_runs.py OUTDIR [--smoke]"""
import os, csv, time
import numpy as np, torch, torch.nn as nn
import a12_runs as A
from ceiling import weight_table

def one(lam, seed, kind="xor", sigma=0.5, steps=4000):
    if A.SMOKE: steps = 250
    rid = f"A-18-lam{lam}-{kind}-sig{sigma}-s{steps}-seed{seed}"; cpath, spath = f"{A.OUT}/A-18_curves.csv", f"{A.OUT}/A-18_final.csv"
    if os.path.exists(spath) and rid in {r["run_id"] for r in csv.DictReader(open(spath))}: return
    t0 = time.time(); val, test, bv, bt, dev = A.sets(kind, sigma)
    Tm = torch.tensor(weight_table(kind, A.N).reshape(A.N, 81), dtype=torch.float64)
    drng = np.random.default_rng(9000 + seed); xtr, ytr = A.make_data(kind, A.N, sigma, 20000 if A.SMOKE else A.NTR, drng)
    torch.manual_seed(5000 + seed); net = A.make_net2(2); opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    every = 10 if A.SMOKE else 100; rows = []
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(xtr), (A.BS,)); xb = xtr[idx]
        with torch.no_grad(): target = (1 - lam) * A.bayes_logp(Tm, xb).exp() + lam * torch.nn.functional.one_hot(ytr[idx], A.N).float()
        loss = -(target * torch.log_softmax(net(xb), 1)).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step in (1, 25, 50, 100) or step % every == 0:
            e = (A.net_levels(net, *val, "F0").mean(0) - bv).tolist(); rows.append([rid, kind, sigma, seed, step] + e)
    el = (A.net_levels(net, *test, "F0").mean(0) - bt).tolist()
    new = not os.path.exists(cpath)
    with open(cpath, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["run_id", "masking", "sigma", "seed", "step", "val_A_parity_group", "val_B_class_in_group", "val_C_secret_in_class", "val_total"])
        w.writerows(rows)
    new = not os.path.exists(spath)
    with open(spath, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["run_id", "masking", "sigma", "steps", "seed", "test_A", "test_B", "test_C", "test_total", "bayes_total_bits"])
        w.writerow([rid, kind, sigma, steps, seed] + el + [bt[3].item()])
    print(f"{rid} A={el[0]:.4f} B={el[1]:.4f} C={el[2]:.4f} total={el[3]:.4f} ({time.time() - t0:.0f}s)", flush=True)
for lam in (0.03, 0.1, 0.3):
    for seed in range(1 if A.SMOKE else 2): one(lam, seed)
