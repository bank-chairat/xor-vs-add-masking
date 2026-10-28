"""A-35: how far do the weights move during the flat start, default output scale versus output x16 (A-31c)?
Same data, initial weights and batch draws as a31_runs.py (a12_runs.one with sched 1e9, i.e. constant Adam 2e-3, batch 1,024, 1M fixed traces).
Records relative weight change ||W(t) - W(0)|| / ||W(0)|| per layer and validation parity part A at steps 0, 100, 250, 500, 1000, 1500, 2000.
Usage: CO=16 SEEDS="0 1" python3 a35_move.py OUTDIR"""
import os, csv, numpy as np, torch, torch.nn as nn
import a12_runs as A
CO = float(os.environ.get("CO", "1")); STEPS = 2000; REC = (0, 100, 250, 500, 1000, 1500, 2000)
out = f"{A.OUT}/move.csv"; new = not os.path.exists(out); fh = open(out, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["CO", "seed", "step", "valA", "rel_L1", "rel_L2", "rel_L3", "rel_out"])
for seed in [int(s) for s in os.environ["SEEDS"].split()]:
    val, test, bv, bt, dev = A.sets("xor", 0.5)
    drng = np.random.default_rng(9000 + seed); xtr, ytr = A.make_data("xor", A.N, 0.5, A.NTR, drng)
    torch.manual_seed(5000 + seed); net = A.make_net2(2)
    lin = [m for m in net if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for p in lin[-1].parameters(): p.mul_(CO)
    W0 = [m.weight.detach().clone() for m in lin]
    opt = torch.optim.Adam(net.parameters(), lr=2e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10 ** 9)
    for step in range(0, STEPS + 1):
        if step > 0:
            idx = torch.randint(0, len(xtr), (A.BS,)); loss = nn.functional.cross_entropy(net(xtr[idx]), ytr[idx]); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if step in REC:
            with torch.no_grad(): a_ = (A.net_levels(net, *val, "F0").mean(0) - bv)[0].item(); rel = [float((m.weight - w0).norm() / w0.norm()) for m, w0 in zip(lin, W0)]
            w.writerow([CO, seed, step, f"{a_:.5f}"] + [f"{r:.4f}" for r in rel]); fh.flush(); print(CO, seed, step, f"A {a_:.4f}", " ".join(f"{r:.3f}" for r in rel), flush=True)
