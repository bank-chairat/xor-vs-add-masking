"""A-38: learning when the Hamming distance between the two shares also leaks (A-37 model). Env KIND=xor|add, SEEDS, STEPS.
Readings: HW(m), HW(share2), HW(m ^ share2), each + N(0, 0.5^2); share2 = s ^ m or (s + m) mod 256 (project convention).
Hard labels (LABELS=soft: exact posterior, used by A-40), 3x256 ReLU, Adam 2e-3 constant, batch 1,024, 1M fixed traces. Metric: total excess (bits) on 100,000 validation traces.
Usage: KIND=xor SEEDS="0 1 2 3" python3 a38_transition.py OUTDIR"""
import os, sys, csv, time, numpy as np, torch, torch.nn as nn
torch.set_num_threads(2)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); N = 256; SIG = 0.5; BS = 1024; LR = 2e-3; STEPS = int(os.environ.get("STEPS", "4000")); NTR = 1_000_000
KIND = os.environ.get("KIND", "xor"); LABELS = os.environ.get("LABELS", "hard"); SEEDS = [int(s) for s in os.environ.get("SEEDS", "0 1 2 3").split()]
HW = np.array([bin(v).count("1") for v in range(N)])
def sh2(s, m): return (s ^ m) if KIND == "xor" else (s + m) % N
def make(n, rng):
    s = rng.integers(0, N, n); m = rng.integers(0, N, n); t = sh2(s, m)
    x = np.c_[HW[m], HW[t], HW[m ^ t]] + rng.normal(0, SIG, (n, 3))
    return torch.tensor(x, dtype=torch.float32), torch.tensor(s)
S_, M_ = np.meshgrid(np.arange(N), np.arange(N), indexing="ij"); T_ = sh2(S_, M_)
C = np.zeros((N, 9, 9, 9)); np.add.at(C, (S_.ravel(), HW[M_].ravel(), HW[T_].ravel(), HW[M_ ^ T_].ravel()), 1.0)
Ct = torch.tensor(C.reshape(N, 729), dtype=torch.float64)
def post(x):
    x = x.double(); lv = torch.arange(9, dtype=torch.float64)
    L = torch.exp(-0.5 * ((x[:, :, None] - lv[None, None, :]) / SIG) ** 2)          # (n, 3, 9)
    outer = (L[:, 0, :, None, None] * L[:, 1, None, :, None] * L[:, 2, None, None, :]).reshape(len(x), 729)
    lik = (outer @ Ct.T).clamp_min(1e-300); return lik / lik.sum(1, keepdim=True)
_r = np.random.default_rng(1); _x, _ = make(3, _r); _p = post(_x)
for k in range(3):
    g = lambda o, h: np.exp(-0.5 * ((o - h) / SIG) ** 2)
    bf = np.array([sum(g(_x[k, 0].item(), HW[m]) * g(_x[k, 1].item(), HW[sh2(s, m)]) * g(_x[k, 2].item(), HW[m ^ sh2(s, m)]) for m in range(N)) for s in range(N)]); bf /= bf.sum()
    assert np.abs(bf - _p[k].numpy()).max() < 1e-9, "posterior check failed"
vrng = np.random.default_rng(424242); xv, yv = make(100000, vrng)
pv = torch.cat([post(xv[i:i + 10000]) for i in range(0, len(xv), 10000)]); bayes_ce = float(-torch.log2(pv[torch.arange(len(yv)), yv]).mean())
print(f"{KIND}: posterior check passed; exact Bayes CE {bayes_ce:.4f} bits, MI {8 - bayes_ce:.4f} bits", flush=True)
def make_net():
    L, d = [], 3
    for _ in range(3): L += [nn.Linear(d, 256), nn.ReLU()]; d = 256
    return nn.Sequential(*L, nn.Linear(d, N))
path = f"{OUT}/A-38_curves.csv"
for seed in SEEDS:
    t0 = time.time(); drng = np.random.default_rng(9000 + seed); xtr, ytr = make(NTR, drng); torch.manual_seed(5000 + seed); net = make_net(); opt = torch.optim.Adam(net.parameters(), lr=LR); rows = []
    for t in range(0, STEPS + 1):
        if t > 0:
            idx = torch.randint(0, NTR, (BS,)); z = net(xtr[idx])
            if LABELS == "soft":
                with torch.no_grad(): q = post(xtr[idx]).float()
                loss = -(q * torch.log_softmax(z, 1)).sum(1).mean()
            else: loss = nn.functional.cross_entropy(z, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if t in (0, 25, 50) or t % 100 == 0:
            with torch.no_grad(): ce = float(torch.cat([nn.functional.cross_entropy(net(xv[i:i + 50000]), yv[i:i + 50000], reduction="none") for i in range(0, len(xv), 50000)]).mean() / np.log(2))
            rows.append([KIND, seed, t, ce - bayes_ce])
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        wr = csv.writer(f)
        if new: wr.writerow(["kind", "seed", "step", "val_total_excess_bits"])
        wr.writerows(rows)
    e = {r[2]: r[3] for r in rows}
    print(f"A-38 {KIND} {LABELS} seed {seed}: excess at 100 {e[100]:.4f}, 600 {e.get(600, float("nan")):.4f}, 4000 {e[STEPS]:.4f} ({time.time() - t0:.0f}s)", flush=True)
