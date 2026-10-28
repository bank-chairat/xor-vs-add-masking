"""Fast checks that the core numbers and identities hold. Run: python3 tests_sanity.py
Every assert states a fact the write-up relies on."""
import numpy as np
from ceiling import weight_table, discrete_mi, gaussian_mi, single_share_mi, HW

# 1. Parity identity: HW(s xor m) + HW(m) has the same parity as HW(s), for all pairs.
assert all((HW[m] + HW[s ^ m]) % 2 == HW[s] % 2 for s in range(256) for m in range(256))
# 2. Exactly half of the 256 secrets have even Hamming weight (so parity is worth exactly 1 bit).
assert sum(HW[s] % 2 == 0 for s in range(256)) == 128

for kind, N in [("xor", 256), ("add", 256), ("add", 251)]:
    T = weight_table(kind, N)
    # 3. For every secret the table is a probability distribution over (h1, h2).
    assert np.allclose(T.sum(axis=(1, 2)), 1.0)
    # 4. Each share alone carries no information about the secret.
    assert abs(single_share_mi(T, 0)) < 1e-12 and abs(single_share_mi(T, 1)) < 1e-12
    # 5. Quadrature at small noise approaches the exact noiseless value (within 0.01 bits).
    assert abs(gaussian_mi(T, 0.15)[0] - discrete_mi(T)) < 0.01
    # 6. Information decreases as noise grows, and never exceeds 8 bits.
    mis = [gaussian_mi(T, s)[0] for s in (0.25, 0.5, 1.0, 2.0)]
    assert all(a > b for a, b in zip(mis, mis[1:])) and mis[0] < 8

# 7. Reference values reported in PROGRESS.md (tolerance covers the quadrature step).
ref = {("xor", 256, 0.5): 0.1435, ("add", 256, 0.5): 0.1077, ("xor", 256, 1.0): 0.0466,
       ("add", 256, 1.0): 0.0365, ("add", 251, 0.5): 0.1049, ("add", 251, 1.0): 0.0313}
for (k, N, s), v in ref.items():
    assert abs(gaussian_mi(weight_table(k, N), s)[0] - v) < 6e-4, (k, N, s)
assert abs(discrete_mi(weight_table("xor", 256)) - 1.1499) < 5e-4
assert abs(discrete_mi(weight_table("add", 256)) - 0.2750) < 5e-4
assert abs(discrete_mi(weight_table("add", 251)) - 0.2604) < 5e-4
print("all sanity checks passed")
