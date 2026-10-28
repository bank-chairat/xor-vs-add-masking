"""A-12/13/14 (generalised from A-11): per-step parity curves (H4) and within-class symmetry split (H8). Predictions are in PROGRESS.md.
Usage: python3 a11_curves.py OUTDIR [--smoke]     Resumable.
Same data, seeds, network, optimizer and schedule as A-09 (parity_tests.py), so final numbers should reproduce A-09.
Cross entropy is split by the chain rule in three levels (all in bits, network minus exact Bayes on the same traces):
  A  parity group of the secret (parity of HW(s))          -log Qgroup(g(s))
  B  Hamming-weight class inside the group                 -log [Qclass(c(s)) / Qgroup(g(s))]
  C  the secret inside its Hamming-weight class            -log [q(s) / Qclass(c(s))]
A + B + C = total excess. For XOR the exact posterior is uniform inside a class, so C is the cost of the network
treating same-weight secrets differently (H8). For addition C is just the remaining part."""
import sys, os, csv, json, time
import numpy as np, torch, torch.nn as nn
from ceiling import weight_table, HW
from train_mlp import make_data

torch.set_num_threads(2)
OUT = sys.argv[1]; SMOKE = "--smoke" in sys.argv
os.makedirs(OUT, exist_ok=True)
N, BS, LR, NTR = 256, 1024, 2e-3, 1_000_000
SIGMA = 0.5
CLS = torch.tensor([HW[s] for s in range(N)])                  # Hamming-weight class of each secret
GRP = CLS % 2                                                  # parity group
CMASK = (CLS[:, None] == CLS[None, :])                         # CMASK[r, s]: same class
GMASK = (GRP[:, None] == GRP[None, :])                         # GMASK[r, s]: same parity group
LN2 = np.log(2)
OH = torch.nn.functional.one_hot(CLS, 9).float()          # (256, 9)

def feats(x, mode):
    if mode == "F0": return x
    if mode == "F1": return torch.cat([x, torch.cos(np.pi * (x[:, :1] + x[:, 1:2]))], 1)
    raise ValueError(mode)
DIM = {"F0": 2, "F1": 3}

def make_net2(d_in, width=256, depth=3):
    layers, d = [], d_in
    for _ in range(depth): layers += [nn.Linear(d, width), nn.ReLU()]; d = width
    layers.append(nn.Linear(d, N)); return nn.Sequential(*layers)

def levels(logp, y):
    """per-sample (A, B, C, total) in bits from log-probabilities logp (n,N) in nats."""
    lq = logp.gather(1, y[:, None])[:, 0]
    lc = torch.logsumexp(logp.masked_fill(~CMASK[:, y].T, -1e30), 1)
    lg = torch.logsumexp(logp.masked_fill(~GMASK[:, y].T, -1e30), 1)
    return torch.stack([-lg, -(lc - lg), -(lq - lc), -lq], 1) / LN2

@torch.no_grad()
def net_levels(net, x, y, mode, bs=50000):
    return torch.cat([levels(torch.log_softmax(net(feats(x[i:i+bs], mode)), 1), y[i:i+bs]) for i in range(0, len(x), bs)])

def bayes_logp(Tm, xb):
    xb = xb.double()
    g = torch.exp(-0.5 * ((xb[:, :, None] - torch.arange(9.0, dtype=torch.float64)) / SIGMA) ** 2)
    lik = (g[:, 0, :, None] * g[:, 1, None, :]).reshape(len(xb), 81) @ Tm.T
    return torch.log(lik / lik.sum(1, keepdim=True)).float()

def bayes_levels(kind, x, y, bs=50000):
    Tm = torch.tensor(weight_table(kind, N).reshape(N, 81), dtype=torch.float64)
    out, dev = [], 0.0
    for i in range(0, len(x), bs):
        lp = bayes_logp(Tm, x[i:i+bs]); out.append(levels(lp, y[i:i+bs]))
        if kind == "xor":      # check: exact posterior uniform inside a class (Hiltenbrand Prop. 1)
            p = lp.exp(); m = ((p @ OH) / OH.sum(0))[:, CLS]
            dev = max(dev, (p - m).abs().max().item())
    return torch.cat(out), dev

_c = {}
def sets(kind, sigma):
    global SIGMA
    SIGMA = sigma
    if (kind, sigma) not in _c:
        d = np.random.default_rng(7)
        val = make_data(kind, N, sigma, 100000, d); test = make_data(kind, N, sigma, 400000, d)
        bl_v, _ = bayes_levels(kind, *val); bl_t, dev = bayes_levels(kind, *test)
        _c[(kind, sigma)] = (val, test, bl_v.mean(0), bl_t.mean(0), dev)
    return _c[(kind, sigma)]

def one(exp, kind, sigma, steps, sched, fresh, seed):
    """exp tag; steps = run length; sched = cosine length (>= steps); fresh = new batch from the simulator every step."""
    if SMOKE: steps, sched = 250, 250
    rid = f"{exp}-{kind}-sig{sigma}-s{steps}-sched{sched}-{'fresh' if fresh else 'fixed'}-seed{seed}"
    cpath, spath = f"{OUT}/{exp}_curves.csv", f"{OUT}/{exp}_final.csv"
    if os.path.exists(spath) and rid in {r["run_id"] for r in csv.DictReader(open(spath))}: return
    t0 = time.time()
    val, test, bv, bt, dev = sets(kind, sigma)
    drng = np.random.default_rng(9000 + seed)
    if not fresh: xtr, ytr = make_data(kind, N, sigma, 20000 if SMOKE else NTR, drng)
    torch.manual_seed(5000 + seed)
    net = make_net2(2); opt = torch.optim.Adam(net.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, sched)
    every = (10 if SMOKE else (100 if steps <= 4000 else 200))
    rows = []
    for step in range(1, steps + 1):
        if fresh: xb, yb = make_data(kind, N, sigma, BS, drng)
        else:
            idx = torch.randint(0, len(xtr), (BS,)); xb, yb = xtr[idx], ytr[idx]
        loss = nn.functional.cross_entropy(net(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if step in (1, 25, 50, 100) or step % every == 0:
            e = (net_levels(net, *val, "F0").mean(0) - bv).tolist()
            rows.append([rid, kind, sigma, seed, step] + e)
    el = (net_levels(net, *test, "F0").mean(0) - bt).tolist()
    new = not os.path.exists(cpath)
    with open(cpath, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["run_id","masking","sigma","seed","step","val_A_parity_group","val_B_class_in_group","val_C_secret_in_class","val_total"])
        w.writerows(rows)
    new = not os.path.exists(spath)
    with open(spath, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["run_id","masking","sigma","fresh","steps","sched","seed","test_A","test_B","test_C","test_total","bayes_total_bits"])
        w.writerow([rid, kind, sigma, int(fresh), steps, sched, seed] + el + [bt[3].item()])
    print(f"{rid} A={el[0]:.4f} B={el[1]:.4f} C={el[2]:.4f} total={el[3]:.4f} ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    S = 1 if SMOKE else 4
    for seed in range(S):                                   # A-12: fresh data every step
        for kind in ("xor", "add"): one("A-12", kind, 0.5, 4000, 4000, True, seed)
    print("A-12 COMPLETE", flush=True)
    for seed in range(S):                                   # A-14: noise levels
        for sg in (0.25, 0.35, 0.75):
            for kind in ("xor", "add"): one("A-14", kind, sg, 4000, 4000, False, seed)
    print("A-14 COMPLETE", flush=True)
    for seed in range(S):                                   # A-13: cosine over 16,000 steps (longest, last)
        for kind in ("xor", "add"): one("A-13", kind, 0.5, 16000, 16000, False, seed)
    print("A-13 COMPLETE", flush=True)
