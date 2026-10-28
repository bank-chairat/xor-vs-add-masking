"""A-09: why does XOR need more training? Test D (where the shortfall lives) and Test F (extra input features).
Usage: python3 parity_tests.py OUTDIR [--smoke]      Resumable. Predictions are written in PROGRESS.md before running.
Cross entropy is split by the chain rule into  -log q(parity group of s)  [between]  and  -log q(s | group)  [within],
both for the network and for the exact Bayes posterior on the same test traces."""
import sys, os, csv, json, time
import numpy as np, torch, torch.nn as nn
from ceiling import weight_table, gaussian_mi, HW
from train_mlp import make_data

torch.set_num_threads(2)
OUT = sys.argv[1]; SMOKE = "--smoke" in sys.argv
os.makedirs(OUT, exist_ok=True)
N, BS, LR, SIGMA, NTR = 256, 1024, 2e-3, 0.5, 1_000_000
PAR = torch.tensor([HW[s] % 2 for s in range(N)])
SAME = (PAR[:, None] == PAR[None, :])            # SAME[r, s]: classes r and s share parity
COLS = ["exp_id","run_id","masking","feature","seed","steps","excess_final","excess_bestval","between_excess_final",
        "within_excess_final","between_excess_bestval","within_excess_bestval","ideal_between","ideal_within","notes"]

def feats(x, mode):
    a, b = x[:, :1], x[:, 1:2]
    pi = np.pi
    if mode == "F0": return x
    if mode == "F1": return torch.cat([x, torch.cos(pi * (a + b))], 1)
    if mode == "F2": return torch.cat([x, torch.cos(pi * a), torch.sin(pi * a), torch.cos(pi * b), torch.sin(pi * b)], 1)
    if mode == "F3": return torch.cat([x, a + b], 1)
    raise ValueError(mode)
DIM = {"F0": 2, "F1": 3, "F2": 6, "F3": 3}

def make_net2(d_in, width=256, depth=3):
    layers, d = [], d_in
    for _ in range(depth): layers += [nn.Linear(d, width), nn.ReLU()]; d = width
    layers.append(nn.Linear(d, N)); return nn.Sequential(*layers)

def parts_from_logp(logp, y):
    """logp (n,N) log-probabilities in nats. Returns per-sample (total, between) in bits."""
    tot = -logp.gather(1, y[:, None])[:, 0] / np.log(2)
    mass = torch.logsumexp(logp.masked_fill(~SAME[:, y].T, -1e30), 1)   # log prob of the true parity group
    return tot, -mass / np.log(2)

@torch.no_grad()
def net_parts(net, x, y, mode, bs=50000):
    T, B = [], []
    for i in range(0, len(x), bs):
        lp = torch.log_softmax(net(feats(x[i:i+bs], mode)), 1)
        t, b = parts_from_logp(lp, y[i:i+bs]); T.append(t); B.append(b)
    return torch.cat(T), torch.cat(B)

def bayes_parts(kind, x, y, bs=50000):
    Tm = torch.tensor(weight_table(kind, N).reshape(N, 81), dtype=torch.float64)
    T, B = [], []
    for i in range(0, len(x), bs):
        xb = x[i:i+bs].double()
        g = torch.exp(-0.5 * ((xb[:, :, None] - torch.arange(9.0, dtype=torch.float64)) / SIGMA) ** 2)  # (n,2,9)
        outer = (g[:, 0, :, None] * g[:, 1, None, :]).reshape(len(xb), 81)
        lik = outer @ Tm.T
        lp = torch.log(lik / lik.sum(1, keepdim=True)).float()
        t, b = parts_from_logp(lp, y[i:i+bs]); T.append(t); B.append(b)
    return torch.cat(T), torch.cat(B)

_c = {}
def sets(kind):
    if kind not in _c:
        d = np.random.default_rng(7)
        val = make_data(kind, N, SIGMA, 100000, d); test = make_data(kind, N, SIGMA, 400000, d)
        bt, bb = bayes_parts(kind, *test)
        _c[kind] = (val, test, bt.mean().item(), bb.mean().item())
    return _c[kind]

def one(kind, mode, seed, steps):
    if SMOKE: steps = 250
    rid = f"A-09-{kind}-{mode}-s{steps}-seed{seed}"; path = f"{OUT}/A-09.csv"
    if os.path.exists(path) and rid in {r["run_id"] for r in csv.DictReader(open(path))}: return
    t0 = time.time()
    val, test, id_tot, id_bet = sets(kind)
    xtr, ytr = make_data(kind, N, SIGMA, 20000 if SMOKE else NTR, np.random.default_rng(9000 + seed))
    torch.manual_seed(5000 + seed)
    net = make_net2(DIM[mode]); opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    best, state = -1e9, None
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(xtr), (BS,))
        loss = nn.functional.cross_entropy(net(feats(xtr[idx], mode)), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 250 == 0:
            v = N and (8.0 - net_parts(net, *val, mode)[0].mean().item())
            if v > best: best, state = v, {k: t.clone() for k, t in net.state_dict().items()}
    res = {}
    for tag in ("final", "bestval"):
        if tag == "bestval": net.load_state_dict(state)
        t, b = net_parts(net, *test, mode)
        res[tag] = (t.mean().item() - id_tot, b.mean().item() - id_bet)       # (excess total, excess between)
    row = dict(exp_id="A-09", run_id=rid, masking=kind, feature=mode, seed=seed, steps=steps,
               excess_final=res["final"][0], excess_bestval=res["bestval"][0],
               between_excess_final=res["final"][1], within_excess_final=res["final"][0] - res["final"][1],
               between_excess_bestval=res["bestval"][1], within_excess_bestval=res["bestval"][0] - res["bestval"][1],
               ideal_between=id_bet, ideal_within=id_tot - id_bet, notes="excess = network minus exact Bayes on the same test traces; cpu")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new: w.writeheader()
        w.writerow(row)
    print(f"{rid} excess={row['excess_final']:.4f} between={row['between_excess_final']:.4f} within={row['within_excess_final']:.4f} ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    nseed = 1 if SMOKE else 4
    for seed in range(nseed):
        for kind in ("xor", "add"):
            for mode in ("F0", "F1", "F2", "F3"): one(kind, mode, seed, 4000)
    for seed in range(nseed):                       # Test D: baseline at 16,000 steps
        for kind in ("xor", "add"): one(kind, "F0", seed, 16000)
    print("A-09 COMPLETE", flush=True)
