"""A-33: is the XOR flat start an Adam phenomenon? Momentum SGD (momentum 0.9), hard or soft labels, XOR, sigma 0.5, 3x256, batch 1,024, 1M fixed traces.
Env: LABELS=hard|soft, LR, SEEDS, STEPS. Usage: LABELS=hard LR=0.05 SEEDS="0 1 2 3" python3 a33_sgd.py OUTDIR"""
import os, csv, time
import numpy as np, torch, torch.nn as nn
import a12_runs as A
from ceiling import weight_table
torch.set_num_threads(1)
LABELS = os.environ.get("LABELS", "hard"); LR = float(os.environ.get("LR", "0.05")); STEPS = int(os.environ.get("STEPS", "4000")); SEEDS = [int(s) for s in os.environ.get("SEEDS", "0 1 2 3").split()]
kind, sigma = "xor", 0.5
val, test, bv, bt, dev = A.sets(kind, sigma)
Tm = torch.tensor(weight_table(kind, A.N).reshape(A.N, 81), dtype=torch.float64)
cpath = f"{A.OUT}/A-33_curves.csv"
for seed in SEEDS:
    rid = f"A-33-{LABELS}-lr{LR}-st{STEPS}-seed{seed}"
    t0 = time.time(); drng = np.random.default_rng(9000 + seed); xtr, ytr = A.make_data(kind, A.N, sigma, A.NTR, drng)
    torch.manual_seed(5000 + seed); net = A.make_net2(2); opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9); rows = []
    for t in range(1, STEPS + 1):
        idx = torch.randint(0, len(xtr), (A.BS,)); xb = xtr[idx]; z = net(xb)
        if LABELS == "soft":
            with torch.no_grad(): soft = A.bayes_logp(Tm, xb).exp()
            loss = -(soft * torch.log_softmax(z, 1)).sum(1).mean()
        else: loss = nn.functional.cross_entropy(z, ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if t in (1, 25, 50, 100) or t % 100 == 0:
            e = (A.net_levels(net, *val, "F0").mean(0) - bv).tolist(); rows.append([rid, LABELS, LR, seed, t] + e)
    new = not os.path.exists(cpath)
    with open(cpath, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["run_id","labels","lr","seed","step","val_A_parity_group","val_B_class_in_group","val_C_secret_in_class","val_total"])
        w.writerows(rows)
    print(f"{rid} A(val) at end {rows[-1][5]:.4f} total {rows[-1][8]:.4f} ({time.time() - t0:.0f}s)", flush=True)
