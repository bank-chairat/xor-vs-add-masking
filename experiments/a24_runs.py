"""A-24: hand-written Adam with separate gradient sources for the first and second moments. XOR, equal noise.
Env: MODE=HH|SH|HS|SS (first letter: source of m, second: source of v; H = hard one-hot, S = exact-posterior soft), SEEDS, STEPS. Usage: MODE=SH python3 a24_runs.py OUTDIR"""
import os, csv, time
import numpy as np, torch, torch.nn as nn
import a12_runs as A
from ceiling import weight_table
torch.set_num_threads(1)
MODE = os.environ.get("MODE", "HH"); STEPS = int(os.environ.get("STEPS", "4000")); SEEDS = [int(s) for s in os.environ.get("SEEDS", "0 1 2 3").split()]
kind, sigma = "xor", 0.5
val, test, bv, bt, dev = A.sets(kind, sigma)
Tm = torch.tensor(weight_table(kind, A.N).reshape(A.N, 81), dtype=torch.float64)
cpath, spath = f"{A.OUT}/A-24_curves.csv", f"{A.OUT}/A-24_final.csv"
for seed in SEEDS:
    rid = f"A-24-{MODE}-st{STEPS}-seed{seed}"
    if os.path.exists(spath) and rid in {r["run_id"] for r in csv.DictReader(open(spath))}: continue
    t0 = time.time(); drng = np.random.default_rng(9000 + seed); xtr, ytr = A.make_data(kind, A.N, sigma, A.NTR, drng)
    torch.manual_seed(5000 + seed); net = A.make_net2(2); params = list(net.parameters())
    m = [torch.zeros_like(p) for p in params]; v = [torch.zeros_like(p) for p in params]; rows = []
    for t in range(1, STEPS + 1):
        idx = torch.randint(0, len(xtr), (A.BS,)); xb = xtr[idx]; z = net(xb)
        with torch.no_grad(): soft = A.bayes_logp(Tm, xb).exp()
        def loss(src): return -(soft * torch.log_softmax(z, 1)).sum(1).mean() if src == "S" else nn.functional.cross_entropy(z, ytr[idx])
        gm = torch.autograd.grad(loss(MODE[0]), params, retain_graph=(MODE[0] != MODE[1]))
        gv = gm if MODE[0] == MODE[1] else torch.autograd.grad(loss(MODE[1]), params)
        with torch.no_grad():
            for p, mi, vi, g1, g2 in zip(params, m, v, gm, gv):
                mi.mul_(0.9).add_(g1, alpha=0.1); vi.mul_(0.999).addcmul_(g2, g2, value=0.001)
                p.addcdiv_(mi / (1 - 0.9 ** t), torch.sqrt(vi / (1 - 0.999 ** t)) + 1e-8, value=-A.LR)
        if t in (1, 25, 50, 100) or t % 100 == 0:
            e = (A.net_levels(net, *val, "F0").mean(0) - bv).tolist(); rows.append([rid, MODE, seed, t] + e)
    el = (A.net_levels(net, *test, "F0").mean(0) - bt).tolist(); new = not os.path.exists(cpath)
    with open(cpath, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["run_id","mode","seed","step","val_A_parity_group","val_B_class_in_group","val_C_secret_in_class","val_total"])
        w.writerows(rows)
    new = not os.path.exists(spath)
    with open(spath, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["run_id","mode","steps","seed","test_A","test_B","test_C","test_total","bayes_total_bits"])
        w.writerow([rid, MODE, STEPS, seed] + el + [bt[3].item()])
    print(f"{rid} A={el[0]:.4f} B={el[1]:.4f} C={el[2]:.4f} total={el[3]:.4f} ({time.time() - t0:.0f}s)", flush=True)
