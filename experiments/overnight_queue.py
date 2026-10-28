"""Overnight queue (A-04 to A-08). Resumable: finished run_ids are skipped on restart.
Usage: python3 overnight_queue.py OUTDIR [--smoke]
Experiments (all sigma-0.5 unless stated, N=256, same Adam+cosine setup as A-03):
 A-04 learning-rate sweep, 4k steps       A-05 noise sweep, 16k steps
 A-06 training-set-size sweep, 16k steps  A-07 network size sweep, 4k steps
 A-08 noiseless exact-table training, parity tracking (20 seeds, small net)
"""
import sys, os, csv, json, time
import numpy as np, torch, torch.nn as nn
from ceiling import weight_table, gaussian_mi, HW
from train_mlp import make_data, make_net, pi_bits

torch.set_num_threads(2)
OUT = sys.argv[1]; SMOKE = "--smoke" in sys.argv
T_START = time.time()
LIMIT = float(os.environ.get("QUEUE_LIMIT_S", "1e9"))
class OutOfTime(Exception): pass
os.makedirs(OUT + "/weights", exist_ok=True)
N, BS = 256, 1024
COLS = ["exp_id","run_id","owner","masking","N","sigma","n_train","seed_init","seed_data","steps","batch",
        "optimizer","lr","params","H_S","MI_ceiling_bits","H_S_given_Y","test_CE_bits","test_CE_se","PI_bits",
        "excess_CE_bits","fraction_PI_over_MI","accuracy","bayes_accuracy","parity_prob","checkpoint_rule","status","notes"]

def done_ids(path):
    if not os.path.exists(path): return set()
    return {r["run_id"] for r in csv.DictReader(open(path))}

def append(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new: w.writeheader()
        w.writerow(row)

_cache = {}
def ceil_and_sets(kind, sigma):
    k = (kind, sigma)
    if k not in _cache:
        T = weight_table(kind, N)
        c, _ = gaussian_mi(T, sigma)
        d = np.random.default_rng(7)
        _cache[k] = (c, make_data(kind, N, sigma, 100000, d), make_data(kind, N, sigma, 400000, d))
    return _cache[k]

def one_run(exp, kind, sigma, n_train, steps, lr, width, depth, seed):
    if SMOKE: steps, n_train = 250, min(n_train, 20000)
    rid = f"{exp}-{kind}-sig{sigma}-n{n_train}-s{steps}-lr{lr}-w{width}d{depth}-seed{seed}"
    path = f"{OUT}/{exp}.csv"
    if rid in done_ids(path): return
    if time.time() - T_START > LIMIT: raise OutOfTime
    t0 = time.time()
    ceil, val, test = ceil_and_sets(kind, sigma)
    xtr, ytr = make_data(kind, N, sigma, n_train, np.random.default_rng(9000 + seed))
    torch.manual_seed(5000 + seed)
    net = make_net(N, width, depth)
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    best, state, curve = -1e9, None, []
    ev = 250 if steps >= 1000 else 50
    for step in range(1, steps + 1):
        idx = torch.randint(0, n_train, (min(BS, n_train),))
        loss = nn.functional.cross_entropy(net(xtr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % ev == 0:
            v, _ = pi_bits(net, *val, N)
            curve.append((step, v))
            if v > best: best, state = v, {k: t.clone() for k, t in net.state_dict().items()}
    res = {}
    for tag in ["final", "bestval"]:
        if tag == "bestval": net.load_state_dict(state)
        pi, se = pi_bits(net, *test, N)
        with torch.no_grad(): acc = (net(test[0]).argmax(1) == test[1]).float().mean().item()
        res[tag] = (pi, se, acc)
    torch.save(state, f"{OUT}/weights/{rid}.pt")
    json.dump(curve, open(f"{OUT}/weights/{rid}_curve.json", "w"))
    for tag in ["final", "bestval"]:
        pi, se, acc = res[tag]
        append(path, dict(exp_id=exp, run_id=rid, owner="A", masking=kind, N=N, sigma=sigma, n_train=n_train,
            seed_init=5000+seed, seed_data=9000+seed, steps=steps, batch=BS, optimizer="adam", lr=lr, params=npar,
            H_S=8.0, MI_ceiling_bits=ceil, H_S_given_Y=8.0-ceil, test_CE_bits=8.0-pi, test_CE_se=se, PI_bits=pi,
            excess_CE_bits=ceil-pi, fraction_PI_over_MI=pi/ceil, accuracy=acc, bayes_accuracy="not computed",
            parity_prob="n/a", checkpoint_rule=tag, status="completed",
            notes=f"width={width} depth={depth} device=cpu val/test fixed seed 7"))
    print(f"{rid} excess final={ceil-res['final'][0]:.4f} bestval={ceil-res['bestval'][0]:.4f} ({time.time()-t0:.0f}s)", flush=True)

def sweeps():
    M = ["xor", "add"]
    for seed in range(6):                                      # A-03 stage 2 (16k); xor seeds 0,1 finished earlier
        for k in M:
            if not (k == "xor" and seed < 2): yield ("A-03", k, 0.5, 1_000_000, 16000, 2e-3, 256, 3, seed)
    for seed in range(3):                                      # A-03 stage 3 (64k)
        for k in M: yield ("A-03", k, 0.5, 1_000_000, 64000, 2e-3, 256, 3, seed)
    for lr in [5e-4, 1e-3, 4e-3, 8e-3]:                       # A-04
        for seed in range(4):
            for k in M: yield ("A-04", k, 0.5, 1_000_000, 4000, lr, 256, 3, seed)
    for n in [50_000, 200_000]:                                # A-06
        for seed in range(3):
            for k in M: yield ("A-06", k, 0.5, n, 16000, 2e-3, 256, 3, seed)
    for sg in [0.25, 0.75, 1.0, 1.5]:                          # A-05
        for seed in range(3):
            for k in M: yield ("A-05", k, sg, 1_000_000, 16000, 2e-3, 256, 3, seed)
    for (w, d) in [(64, 3), (512, 3), (256, 2), (256, 5)]:    # A-07
        for seed in range(3):
            for k in M: yield ("A-07", k, 0.5, 1_000_000, 4000, 2e-3, w, d, seed)

def a08():
    """Noiseless: train on the exact table of (h1,h2) cells weighted by their true probability."""
    path = f"{OUT}/A-08_parity_tracking.csv"
    par_s = torch.tensor([HW[s] % 2 for s in range(N)])
    cells = torch.tensor([[a, b] for a in range(9) for b in range(9)], dtype=torch.float32)
    for kind in ["xor", "add"]:
        T = torch.tensor(weight_table(kind, N), dtype=torch.float64) / N         # P(s,h1,h2)
        P = T.reshape(N, 81).T                                                    # (cell, s)
        Py = P.sum(1, keepdim=True)                                               # P(cell)
        post = torch.where(Py > 0, P / Py.clamp_min(1e-300), torch.zeros_like(P))
        Hs_y = -(P[P > 0] * torch.log2(post[P > 0])).sum().item()                # H(S|Y), bits
        for seed in range(2 if SMOKE else 20):
            rid = f"A-08-{kind}-seed{seed}"
            if rid in done_ids(path): continue
            torch.manual_seed(7000 + seed)
            net = make_net(N, 64, 2).double()
            opt = torch.optim.Adam(net.parameters(), lr=3e-3)
            rows, steps = [], (400 if SMOKE else 20000)
            for step in range(1, steps + 1):
                lq = torch.log_softmax(net(cells.double()), 1)
                loss = -(P * lq).sum() / np.log(2)                                # CE in bits
                opt.zero_grad(); loss.backward(); opt.step()
                if step % (50 if SMOKE else 200) == 0 or step == 1:
                    with torch.no_grad():
                        q = torch.softmax(net(cells.double()), 1)
                        ce = loss.item()
                        same = (par_s[None, :] == par_s[:, None]).double()        # r,s same parity
                        pmass = ((P * (q @ same)).sum() ).item()                  # E sum_{r same parity as s} q(r|y)
                        rows.append(dict(exp_id="A-08", run_id=rid, masking=kind, seed=seed, step=step,
                                         test_CE_bits=ce, excess_CE_bits=ce - Hs_y, parity_prob=pmass))
            new = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                if new: w.writeheader()
                w.writerows(rows)
            print(f"{rid} final excess={rows[-1]['excess_CE_bits']:.4f} parity_prob={rows[-1]['parity_prob']:.3f}", flush=True)

if __name__ == "__main__":
    try:
        for cfg in sweeps():
            one_run(*cfg)
        a08()
        print("QUEUE COMPLETE", flush=True)
    except OutOfTime:
        print("TIME LIMIT REACHED, restart to continue", flush=True)
