"""A-03: is the sigma=0.5 XOR gap budget, optimization (seed), or rule difficulty?
Fixed: N=256, sigma=0.5, n_train=1M, 3x256 ReLU MLP, Adam lr 2e-3 cosine, batch 1024.
Varies: masking (xor, add), step budget, seed. Separate init and data seeds.
Saves curves (val PI every 250 steps), final-step and best-val checkpoint test PI.
Usage: python3 a03_budget.py out_prefix steps seed_list
"""
import sys, csv, json, time
import numpy as np, torch, torch.nn as nn
from ceiling import weight_table, gaussian_mi
from train_mlp import make_data, make_net, pi_bits

torch.set_num_threads(2)
prefix, steps = sys.argv[1], int(sys.argv[2])
seeds = [int(s) for s in sys.argv[3].split(",")]
N, SIGMA, NTR, BS, LR = 256, 0.5, 1_000_000, 1024, 2e-3
COLS = ["exp_id","run_id","owner","masking","N","sigma","n_train","seed_init","seed_data","steps","batch",
        "optimizer","lr","params","H_S","MI_ceiling_bits","H_S_given_Y","test_CE_bits","test_CE_se","PI_bits",
        "excess_CE_bits","fraction_PI_over_MI","accuracy","bayes_accuracy","parity_prob","checkpoint_rule","status","notes"]
rows, curves = [], {}
for kind in ["xor", "add"]:
    T = weight_table(kind, N)
    ceil, _ = gaussian_mi(T, SIGMA)
    drng = np.random.default_rng(7)
    val = make_data(kind, N, SIGMA, 100000, drng)
    test = make_data(kind, N, SIGMA, 400000, drng)
    for seed in seeds:
        t0 = time.time()
        seed_init, seed_data = 5000 + seed, 9000 + seed
        xtr, ytr = make_data(kind, N, SIGMA, NTR, np.random.default_rng(seed_data))
        torch.manual_seed(seed_init)
        net = make_net(N)
        npar = sum(p.numel() for p in net.parameters())
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        best, state, curve = -1e9, None, []
        for step in range(1, steps + 1):
            idx = torch.randint(0, NTR, (BS,))
            loss = nn.functional.cross_entropy(net(xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if step % 250 == 0:
                v, _ = pi_bits(net, *val, N)
                curve.append((step, v, loss.item() / np.log(2)))
                if v > best:
                    best, state = v, {k: t.clone() for k, t in net.state_dict().items()}
        res = {}
        for tag in ["final", "bestval"]:
            if tag == "bestval":
                net.load_state_dict(state)
            pi, se = pi_bits(net, *test, N)
            with torch.no_grad():
                acc = (net(test[0]).argmax(1) == test[1]).float().mean().item()
            res[tag] = (pi, se, acc)
        rid = f"A-03-{kind}-s{steps}-seed{seed}"
        curves[rid] = curve
        for tag in ["final", "bestval"]:
            pi, se, acc = res[tag]
            rows.append(dict(exp_id="A-03", run_id=rid, owner="A", masking=kind, N=N, sigma=SIGMA, n_train=NTR,
                seed_init=seed_init, seed_data=seed_data, steps=steps, batch=BS, optimizer="adam", lr=LR,
                params=npar, H_S=8.0, MI_ceiling_bits=ceil, H_S_given_Y=8.0 - ceil, test_CE_bits=8.0 - pi,
                test_CE_se=se, PI_bits=pi, excess_CE_bits=ceil - pi, fraction_PI_over_MI=pi / ceil,
                accuracy=acc, bayes_accuracy="not computed", parity_prob="n/a", checkpoint_rule=tag,
                status="completed", notes="val/test sets fixed seed 7, shared across seeds"))
        print(f"{rid} final excess={rows[-2]['excess_CE_bits']:.4f} bestval excess={rows[-1]['excess_CE_bits']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        with open(prefix + ".csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
        json.dump(curves, open(prefix + "_curves.json", "w"))
