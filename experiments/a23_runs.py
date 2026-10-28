"""A-23: unequal share noise, hard vs soft labels (XOR). Usage: SIG="0.25 0.75" python3 a23_runs.py OUTDIR (env LABEL=hard|soft, SEEDS="0 1 2 3", STEPS)
Per-share noise: monkeypatches data generation and the exact posterior of a12_runs; one thread; resumable per run."""
import os, csv, time
import numpy as np, torch, torch.nn as nn
import a12_runs as A
from ceiling import HW
torch.set_num_threads(1)
S1, S2 = [float(x) for x in os.environ.get("SIG", "0.25 0.75").split()]; LABEL = os.environ.get("LABEL", "hard"); STEPS = int(os.environ.get("STEPS", "4000"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0 1 2 3").split()]
SV = torch.tensor([S1, S2], dtype=torch.float64)
def make_data(kind, N, sigma, n, rng):
    s = rng.integers(0, N, n); m = rng.integers(0, N, n); sh2 = (s ^ m) if kind == "xor" else (s + m) % N
    x = np.stack([HW[m], HW[sh2]], axis=1).astype(np.float32); x += rng.normal(0, 1, x.shape).astype(np.float32) * np.array([S1, S2], dtype=np.float32)
    return torch.from_numpy(x), torch.from_numpy(s.astype(np.int64))
def bayes_logp(Tm, xb):
    xb = xb.double(); g = torch.exp(-0.5 * ((xb[:, :, None] - torch.arange(9.0, dtype=torch.float64)) / SV[None, :, None]) ** 2)
    lik = (g[:, 0, :, None] * g[:, 1, None, :]).reshape(len(xb), 81) @ Tm.T
    return torch.log(lik / lik.sum(1, keepdim=True)).float()
A.make_data = make_data; A.bayes_logp = bayes_logp
from ceiling import weight_table
kind = "xor"; key = (S1, S2)
val, test, bv, bt, dev = A.sets(kind, key)
Tm = torch.tensor(weight_table(kind, A.N).reshape(A.N, 81), dtype=torch.float64)
cpath, spath = f"{A.OUT}/A-23_curves.csv", f"{A.OUT}/A-23_final.csv"
for seed in SEEDS:
    rid = f"A-23-{LABEL}-s1_{S1}-s2_{S2}-st{STEPS}-seed{seed}"
    if os.path.exists(spath) and rid in {r["run_id"] for r in csv.DictReader(open(spath))}: continue
    t0 = time.time(); drng = np.random.default_rng(9000 + seed); xtr, ytr = make_data(kind, A.N, key, A.NTR, drng)
    torch.manual_seed(5000 + seed); net = A.make_net2(2); opt = torch.optim.Adam(net.parameters(), lr=2e-3); rows = []
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(xtr), (A.BS,)); xb = xtr[idx]
        if LABEL == "hard": loss = nn.functional.cross_entropy(net(xb), ytr[idx])
        else:
            with torch.no_grad(): target = A.bayes_logp(Tm, xb).exp()
            loss = -(target * torch.log_softmax(net(xb), 1)).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step in (1, 25, 50, 100) or step % 100 == 0:
            e = (A.net_levels(net, *val, "F0").mean(0) - bv).tolist(); rows.append([rid, kind, S1, S2, LABEL, seed, step] + e)
    el = (A.net_levels(net, *test, "F0").mean(0) - bt).tolist(); new = not os.path.exists(cpath)
    with open(cpath, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["run_id","masking","sigma1","sigma2","label","seed","step","val_A_parity_group","val_B_class_in_group","val_C_secret_in_class","val_total"])
        w.writerows(rows)
    new = not os.path.exists(spath)
    with open(spath, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["run_id","sigma1","sigma2","label","steps","seed","test_A","test_B","test_C","test_total","bayes_total_bits"])
        w.writerow([rid, S1, S2, LABEL, STEPS, seed] + el + [bt[3].item()])
    print(f"{rid} A={el[0]:.4f} B={el[1]:.4f} C={el[2]:.4f} total={el[3]:.4f} ({time.time() - t0:.0f}s)", flush=True)
