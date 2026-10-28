"""Exact information ceiling for masked Hamming weight leakage.

Setup (one secret value s, uniform on Z_N):
    mask m uniform on Z_N
    share1 = m
    share2 = s XOR m           (Boolean masking)
           = (s + m) mod N     (arithmetic masking, N = 256 or 251)
    leak1 = HW(share1) + noise, leak2 = HW(share2) + noise, noise ~ N(0, sigma^2)

Because the noise only touches the Hamming weights, the whole joint
distribution of (leak1, leak2) given s is a mixture of at most 9 x 9
Gaussians with weights T[s, h1, h2]. We compute T by exact enumeration
and the mutual information I(S; L1, L2) by numerical quadrature.
"""
import numpy as np

HW = np.array([bin(i).count("1") for i in range(512)])


def weight_table(kind, N):
    """T[s, h1, h2] = P(HW(share1)=h1, HW(share2)=h2 | s)."""
    T = np.zeros((N, 9, 9))
    m = np.arange(N)
    for s in range(N):
        if kind == "xor":
            sh2 = s ^ m
        else:
            sh2 = (s + m) % N
        np.add.at(T[s], (HW[m], HW[sh2]), 1.0 / N)
    return T


def discrete_mi(T):
    """Noiseless limit: I(S; HW1, HW2) in bits, S uniform."""
    N = T.shape[0]
    pj = T.reshape(N, -1)
    pm = pj.mean(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(pj > 0, pj * np.log2(pj / pm), 0.0)
    return r.sum(axis=1).mean()


def gaussian_mi(T, sigma, step_frac=0.25, span=5.0, chunk=8):
    """I(S; L1, L2) in bits for Gaussian noise of std sigma, by 2D quadrature.

    Grid spacing is step_frac * sigma, so quadrature stays accurate at every
    noise level. Secrets are processed in chunks to keep memory small.
    """
    N = T.shape[0]
    lo, hi = -span * sigma, 8 + span * sigma
    grid = int((hi - lo) / (step_frac * sigma)) + 1
    x = np.linspace(lo, hi, grid)
    dx = x[1] - x[0]
    K = np.exp(-0.5 * ((x[None, :] - np.arange(9)[:, None]) / sigma) ** 2)
    K /= sigma * np.sqrt(2 * np.pi)

    def dens(s0, s1):
        return np.einsum("sab,ax,by->sxy", T[s0:s1], K, K, optimize=True)

    pm = np.zeros((grid, grid))
    for s0 in range(0, N, chunk):
        pm += dens(s0, min(s0 + chunk, N)).sum(axis=0)
    pm /= N
    tot = pm.sum() * dx * dx
    acc = 0.0
    for s0 in range(0, N, chunk):
        p = dens(s0, min(s0 + chunk, N))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(p > 1e-300, p * np.log2(p / pm[None]), 0.0)
        acc += r.sum()
    mi = acc / N * dx * dx
    return mi, tot


def single_share_mi(T, which):
    """MI between s and the HW of just one share (should be 0)."""
    N = T.shape[0]
    marg = T.sum(axis=2 - which)  # (N, 9)
    pm = marg.mean(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(marg > 0, marg * np.log2(marg / pm), 0.0)
    return r.sum(axis=1).mean()


CASES = [("xor", 256), ("add", 256), ("add", 251)]

if __name__ == "__main__":
    print("Sanity 1: each share alone reveals nothing (bits, expect ~0)")
    for kind, N in CASES:
        T = weight_table(kind, N)
        assert np.allclose(T.sum(axis=(1, 2)), 1.0)
        print(f"  {kind} mod {N}: share1 {single_share_mi(T,0):.2e}  share2 {single_share_mi(T,1):.2e}")

    print("\nNoiseless limit I(S; HW1, HW2) in bits (max possible 8)")
    for kind, N in CASES:
        T = weight_table(kind, N)
        print(f"  {kind} mod {N}: {discrete_mi(T):.4f}")

    print("\nQuadrature check: density integrates to 1, and sigma->small approaches noiseless limit")
    for kind, N in CASES:
        T = weight_table(kind, N)
        mi, tot = gaussian_mi(T, 0.15)
        print(f"  {kind} mod {N}: sigma=0.15 MI={mi:.4f} bits, integral={tot:.5f}, noiseless={discrete_mi(T):.4f}")

    print("\nMI (bits) vs noise sigma")
    sigmas = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    print("sigma   " + "  ".join(f"{k}{N}".rjust(9) for k, N in CASES))
    for sg in sigmas:
        row = []
        for kind, N in CASES:
            T = weight_table(kind, N)
            row.append(gaussian_mi(T, sg)[0])
        print(f"{sg:5.2f}   " + "  ".join(f"{v:9.5f}" for v in row))
