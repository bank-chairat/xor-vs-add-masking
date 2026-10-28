"""A-20: is the worsening at 64,000 steps a data-reuse effect? Addition, sigma 0.5, cosine schedule over the run length, batch 1024, width 256.
Env: FRESH=1 (new batch from the simulator every step) or 0 (fixed 1M traces); SEEDS="0 1"; STEPS (default 64000); one thread per job.
Checkpoints (net, optimizer, scheduler, RNG states, curve rows) every 2,000 steps, so a stopped job is resumed by running the same command again.
Usage: FRESH=0 SEEDS="0" python3 a20_runs.py OUTDIR"""
import os, csv, time
import numpy as np, torch, torch.nn as nn
import a12_runs as A
torch.set_num_threads(1)
fresh = os.environ.get("FRESH", "0") == "1"; STEPS = int(os.environ.get("STEPS", "64000")); CK = 2000
kind, sigma = os.environ.get("KIND", "xor"), 0.5
def run(seed):
    steps = STEPS; exp = "A-25"
    rid = f"{exp}-{kind}-sig{sigma}-s{steps}-sched{steps}-{'fresh' if fresh else 'fixed'}-seed{seed}"
    cpath, spath = f"{A.OUT}/{exp}_curves.csv", f"{A.OUT}/{exp}_final.csv"
    if os.path.exists(spath) and rid in {r["run_id"] for r in csv.DictReader(open(spath))}: return
    ckp = f"{A.OUT}/ckpt_{rid}.pt"; t0 = time.time()
    val, test, bv, bt, dev = A.sets(kind, sigma)
    drng = np.random.default_rng(9000 + seed)
    if not fresh: xtr, ytr = A.make_data(kind, A.N, sigma, A.NTR, drng)
    torch.manual_seed(5000 + seed); net = A.make_net2(2); opt = torch.optim.Adam(net.parameters(), lr=A.LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); rows = []; start = 1
    if os.path.exists(ckp):
        s = torch.load(ckp, weights_only=False); net.load_state_dict(s["net"]); opt.load_state_dict(s["opt"]); sch.load_state_dict(s["sch"])
        rows = s["rows"]; start = s["step"] + 1; torch.set_rng_state(s["trng"])
        if fresh: drng.bit_generator.state = s["drng"]
        print(f"resumed {rid} at step {start}", flush=True)
    every = 200
    for step in range(start, steps + 1):
        if fresh: xb, yb = A.make_data(kind, A.N, sigma, A.BS, drng)
        else:
            idx = torch.randint(0, len(xtr), (A.BS,)); xb, yb = xtr[idx], ytr[idx]
        loss = nn.functional.cross_entropy(net(xb), yb); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if step in (1, 25, 50, 100) or step % every == 0:
            e = (A.net_levels(net, *val, "F0").mean(0) - bv).tolist(); rows.append([rid, kind, sigma, seed, step] + e)
        if step % CK == 0 and step < steps:
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(), "rows": rows, "step": step,
                        "trng": torch.get_rng_state(), "drng": drng.bit_generator.state if fresh else None}, ckp + ".tmp"); os.replace(ckp + ".tmp", ckp)
            print(f"{rid} step {step} ({time.time() - t0:.0f}s)", flush=True)
    el = (A.net_levels(net, *test, "F0").mean(0) - bt).tolist()
    new = not os.path.exists(cpath)
    with open(cpath, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["run_id","masking","sigma","seed","step","val_A_parity_group","val_B_class_in_group","val_C_secret_in_class","val_total"])
        w.writerows(rows)
    new = not os.path.exists(spath)
    with open(spath, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["run_id","masking","sigma","fresh","steps","sched","seed","test_A","test_B","test_C","test_total","bayes_total_bits"])
        w.writerow([rid, kind, sigma, int(fresh), steps, steps, seed] + el + [bt[3].item()])
    if os.path.exists(ckp): os.remove(ckp)
    print(f"{rid} A={el[0]:.4f} B={el[1]:.4f} C={el[2]:.4f} total={el[3]:.4f} ({time.time() - t0:.0f}s)", flush=True)
for seed in ([int(s) for s in os.environ["SEEDS"].split()] if os.environ.get("SEEDS") else [0, 1]): run(seed)
