"""A-26: gradient signal/noise of hard vs soft (exact posterior) labels along a hard-label run. XOR, sigma 0.5. Usage: python3 a26_snr.py OUTDIR"""
import os, csv, numpy as np, torch, torch.nn as nn
import a12_runs as A
from ceiling import weight_table
torch.set_num_threads(1)
kind, sigma = "xor", 0.5; K, BB = 64, 256; MEAS = (0, 100, 300, 1000, 2000)
Tm = torch.tensor(weight_table(kind, A.N).reshape(A.N, 81), dtype=torch.float64)
fp = f"{A.OUT}/A-26_snr.csv"; new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["seed", "step", "label", "N256", "S", "gbar_sq"])
for seed in range(4):
    drng = np.random.default_rng(9000 + seed); torch.manual_seed(5000 + seed); net = A.make_net2(2); params = list(net.parameters())
    opt = torch.optim.Adam(params, lr=A.LR)
    def grad(src, xb, yb):
        z = net(xb)
        if src == "S":
            with torch.no_grad(): soft = A.bayes_logp(Tm, xb).exp()
            l = -(soft * torch.log_softmax(z, 1)).sum(1).mean()
        else: l = nn.functional.cross_entropy(z, yb)
        return torch.cat([g.flatten() for g in torch.autograd.grad(l, params)])
    def measure(step):
        G = {"H": [], "S": []}
        for _ in range(K):
            xb, yb = A.make_data(kind, A.N, sigma, BB, drng)
            for s in "HS": G[s].append(grad(s, xb, yb))
        for s in "HS":
            g = torch.stack(G[s]); gbar = g.mean(0); N = ((g - gbar) ** 2).sum(1).mean().item(); gs = (gbar ** 2).sum().item()
            w.writerow([seed, step, s, N, gs - N / K, gs]); fh.flush()
    for step in range(0, max(MEAS) + 1):
        if step in MEAS: measure(step)
        xb, yb = A.make_data(kind, A.N, sigma, A.BS, drng)
        loss = nn.functional.cross_entropy(net(xb), yb); opt.zero_grad(); loss.backward(); opt.step()
    print("seed", seed, flush=True)
