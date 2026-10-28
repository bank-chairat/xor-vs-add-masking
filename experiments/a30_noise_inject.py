"""A-30: causal test of the label-noise explanation. Ordinary Adam (hand-written as A-24, both moments from the same gradient) on
g = g_soft(B) + K * (g_hard(B') - g_soft(B')), where B is the training minibatch and B' an independent minibatch of the same size:
the mean is the exact-posterior (soft-label) gradient and the added term is pure one-hot label noise (zero mean given the inputs), scaled by K.
XOR, sigma 0.5, 3x256, batch 1,024, constant lr 2e-3, 4,000 steps. Env: K, SEEDS. Usage: K=1 python3 a30_noise_inject.py OUTDIR"""
import os, csv, time
import numpy as np, torch, torch.nn as nn
import a12_runs as A
from ceiling import weight_table
torch.set_num_threads(1)
K = float(os.environ.get("K", "1")); STEPS = int(os.environ.get("STEPS", "4000")); SEEDS = [int(s) for s in os.environ.get("SEEDS", "0 1 2 3").split()]
kind, sigma = "xor", 0.5
val, test, bv, bt, dev = A.sets(kind, sigma)
Tm = torch.tensor(weight_table(kind, A.N).reshape(A.N, 81), dtype=torch.float64)
cpath, spath = f"{A.OUT}/A-30_curves.csv", f"{A.OUT}/A-30_final.csv"
for seed in SEEDS:
    rid = f"A-30-K{K}-st{STEPS}-seed{seed}"
    if os.path.exists(spath) and rid in {r["run_id"] for r in csv.DictReader(open(spath))}: continue
    t0 = time.time(); drng = np.random.default_rng(9000 + seed); xtr, ytr = A.make_data(kind, A.N, sigma, A.NTR, drng)
    torch.manual_seed(5000 + seed); net = A.make_net2(2); params = list(net.parameters()); g2 = torch.Generator().manual_seed(77000 + seed)
    m = [torch.zeros_like(p) for p in params]; v = [torch.zeros_like(p) for p in params]; rows = []
    for t in range(1, STEPS + 1):
        idx = torch.randint(0, len(xtr), (A.BS,)); xb = xtr[idx]; z = net(xb)
        with torch.no_grad(): soft = A.bayes_logp(Tm, xb).exp()
        gs = torch.autograd.grad(-(soft * torch.log_softmax(z, 1)).sum(1).mean(), params)
        idx2 = torch.randint(0, len(xtr), (A.BS,), generator=g2); xb2 = xtr[idx2]; z2 = net(xb2)
        with torch.no_grad(): soft2 = A.bayes_logp(Tm, xb2).exp()
        lh = nn.functional.cross_entropy(z2, ytr[idx2]); ls = -(soft2 * torch.log_softmax(z2, 1)).sum(1).mean()
        gn = torch.autograd.grad(lh - ls, params)
        with torch.no_grad():
            for p, mi, vi, a, b in zip(params, m, v, gs, gn):
                g = a + K * b
                mi.mul_(0.9).add_(g, alpha=0.1); vi.mul_(0.999).addcmul_(g, g, value=0.001)
                p.addcdiv_(mi / (1 - 0.9 ** t), torch.sqrt(vi / (1 - 0.999 ** t)) + 1e-8, value=-A.LR)
        if t in (1, 25, 50, 100) or t % 100 == 0:
            e = (A.net_levels(net, *val, "F0").mean(0) - bv).tolist(); rows.append([rid, K, seed, t] + e)
    el = (A.net_levels(net, *test, "F0").mean(0) - bt).tolist(); new = not os.path.exists(cpath)
    with open(cpath, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["run_id","K","seed","step","val_A_parity_group","val_B_class_in_group","val_C_secret_in_class","val_total"])
        w.writerows(rows)
    new = not os.path.exists(spath)
    with open(spath, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["run_id","K","steps","seed","test_A","test_B","test_C","test_total","bayes_total_bits"])
        w.writerow([rid, K, STEPS, seed] + el + [bt[3].item()])
    print(f"{rid} A={el[0]:.4f} total={el[3]:.4f} ({time.time() - t0:.0f}s)", flush=True)
