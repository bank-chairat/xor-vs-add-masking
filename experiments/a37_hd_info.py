"""A-37: mutual information between the secret and the readings when, besides HW(share1) and HW(share2), the Hamming distance HW(share1 xor share2) leaks (sigma 0.5).
Exact posterior by enumerating (s, m); MI by Monte Carlo. Usage: python3 a37_hd_info.py"""
import numpy as np
import os
N, SIG, T = 256, 0.5, 200000; CONV = os.environ.get('CONV', 'plus')
HW = np.array([bin(v).count("1") for v in range(N)])
s_, m_ = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
def tables(kind):
    t = (s_ ^ m_) if kind == "xor" else ((s_ + m_) % N) if CONV == "plus" else ((s_ - m_) % N)   # project convention: share2 = s + m (train_mlp.make_data)
    return HW[m_], HW[t], HW[m_ ^ t]                          # (s, m) -> three Hamming weights
def counts(kind, use3):
    h1, h2, h3 = tables(kind); C = np.zeros((N, 9, 9, 9))
    np.add.at(C, (s_.ravel(), h1.ravel(), h2.ravel(), (h3 if use3 else np.zeros_like(h3)).ravel()), 1.0)
    return C / N                                                # p(h1, h2, h3 | s) with m uniform
def mi(kind, use3, rng):
    C = counts(kind, use3); h1, h2, h3 = tables(kind)
    s = rng.integers(0, N, T); m = rng.integers(0, N, T)
    o = np.stack([h1[s, m], h2[s, m], h3[s, m]], 1) + SIG * rng.standard_normal((T, 3))
    lv = np.arange(9)
    L = np.exp(-(o[:, :, None] - lv[None, None, :]) ** 2 / (2 * SIG ** 2))   # (T, 3, 9)
    if not use3: L[:, 2, :] = 0; L[:, 2, 0] = 1
    ce = []
    for i in range(0, T, 5000):
        Lb = L[i:i + 5000]
        lik = np.einsum("shij,th,ti,tj->ts", C, Lb[:, 0], Lb[:, 1], Lb[:, 2], optimize=True)
        p = lik / lik.sum(1, keepdims=True); ce.append(-np.log2(p[np.arange(len(p)), s[i:i + 5000]]))
    ce = np.concatenate(ce); return 8 - ce.mean(), ce.std() / np.sqrt(T)
rng = np.random.default_rng(37)
for kind in ["xor", "add"]:
    for use3 in [False, True]:
        v, se = mi(kind, use3, rng); print(f"{kind} {'with' if use3 else 'without'} transition reading: I = {v:.4f} +- {se:.4f} bits", flush=True)
