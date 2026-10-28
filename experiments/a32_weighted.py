"""A-32: another leakage model. Each share leaks a WEIGHTED sum of its bits, L(v) = sum_i w_i bit_i(v) (fixed weights w ~ U(0.5, 1.5), seed 2026),
plus Gaussian noise sigma 0.5; XOR masking (share2 = s ^ m) with N = 256. Exact posterior P(s | x1, x2) by an XOR convolution (Walsh-Hadamard).
Hard (one-hot) or soft (exact posterior) labels; 3x256 ReLU, Adam 2e-3, batch 1,024, 1M fixed traces, 4,000 steps.
Metric: total excess = network cross-entropy minus exact-Bayes cross-entropy, bits, on 100,000 validation traces. Env: LABELS=hard|soft, SEEDS, KIND=xor|add.
Usage: LABELS=hard SEEDS="0 1 2 3" python3 a32_weighted.py OUTDIR"""
import os, sys, csv, time, numpy as np, torch, torch.nn as nn
torch.set_num_threads(2)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); N = 256; SIG = 0.5; BS = 1024; LR = 2e-3; STEPS = int(os.environ.get("STEPS", "4000")); NTR = 1_000_000
LABELS = os.environ.get("LABELS", "hard"); KIND = os.environ.get("KIND", "xor"); SEEDS = [int(s) for s in os.environ.get("SEEDS", "0 1 2 3").split()]
w = np.random.default_rng(2026).uniform(0.5, 1.5, 8); Lv = torch.tensor(((np.arange(N)[:, None] >> np.arange(8)) & 1) @ w, dtype=torch.float64)   # leakage value of each byte
H = torch.tensor([[(-1) ** bin(i & j).count("1") for j in range(N)] for i in range(N)], dtype=torch.float64)
def make(n, rng):
    s = rng.integers(0, N, n); m = rng.integers(0, N, n); sh2 = (s ^ m) if KIND == "xor" else (s + m) % N
    x = np.c_[Lv.numpy()[m] + rng.normal(0, SIG, n), Lv.numpy()[sh2] + rng.normal(0, SIG, n)]
    return torch.tensor(x, dtype=torch.float32), torch.tensor(s)
def post(x):
    x = x.double(); g1 = torch.exp(-0.5 * ((x[:, :1] - Lv[None, :]) / SIG) ** 2); g2 = torch.exp(-0.5 * ((x[:, 1:] - Lv[None, :]) / SIG) ** 2)
    if KIND == "xor": lik = ((g1 @ H) * (g2 @ H)) @ H / N          # lik[s] = sum_m g1[m] g2[s ^ m]
    else: lik = torch.real(torch.fft.ifft(torch.conj(torch.fft.fft(g1, dim=1)) * torch.fft.fft(g2, dim=1), dim=1))   # lik[s] = sum_m g1[m] g2[(s + m) mod N]
    lik = lik.clamp_min(1e-300); return lik / lik.sum(1, keepdim=True)
# brute-force check of the posterior on a few traces
_r = np.random.default_rng(1); _x, _ = make(5, _r); _p = post(_x)
for k in range(5):
    g1 = np.exp(-0.5 * ((_x[k, 0].item() - Lv.numpy()) / SIG) ** 2); g2 = np.exp(-0.5 * ((_x[k, 1].item() - Lv.numpy()) / SIG) ** 2)
    bf = np.array([sum(g1[m] * g2[(s ^ m) if KIND == "xor" else (s + m) % N] for m in range(N)) for s in range(N)]); bf /= bf.sum()
    assert np.abs(bf - _p[k].numpy()).max() < 1e-9, "posterior check failed"
vrng = np.random.default_rng(424242); xv, yv = make(100000, vrng); pv = post(xv); bayes_ce = float(-torch.log2(pv[torch.arange(len(yv)), yv]).mean())
print(f"posterior check passed; exact Bayes CE {bayes_ce:.4f} bits, MI {8 - bayes_ce:.4f} bits", flush=True)
def make_net():
    L, d = [], 2
    for _ in range(3): L += [nn.Linear(d, 256), nn.ReLU()]; d = 256
    return nn.Sequential(*L, nn.Linear(d, N))
path = f"{OUT}/A-32_curves.csv"
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
            rows.append([KIND, LABELS, seed, t, ce - bayes_ce])
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        wr = csv.writer(f)
        if new: wr.writerow(["kind", "labels", "seed", "step", "val_total_excess_bits"])
        wr.writerows(rows)
    print(f"A-32 {KIND} {LABELS} seed {seed}: excess at 25 {rows[1][4]:.4f}, 100 {rows[3][4]:.4f}, 600 {[r for r in rows if r[3] == 600][0][4]:.4f}, 4000 {rows[-1][4]:.4f} ({time.time() - t0:.0f}s)", flush=True)
