# XOR vs. Addition Masking: Less Information, or Harder to Learn?

A simulated study of why a neural network attacker learns one kind of masked secret more
slowly than another, written up in the order the work was done. Each chapter says what
question we asked, where the idea came from, which files answer it, what we found, and what
it led to next.

**Scope, up front:** everything here is simulated. The leakage model is mostly one simple
model (Hamming weight plus Gaussian noise), the networks are small MLPs, and nothing is
measured on a real device. Results are claims about this model only.


## Motivation and question

**[Masking](#masking)** is a standard defence against [side-channel attacks](#side-channel-attack), where an attacker learns a
secret from a device's power use or electromagnetic emissions. The device never handles the
secret `s` directly. It splits it into two random-looking pieces ([shares](#shares)), so that neither
piece alone says anything about `s`. Two common ways to split a byte:

- **Boolean masking ([XOR](#xor)):** shares `m` and `s XOR m`
- **Arithmetic masking (addition):** shares `m` and `(s + m) mod N` (see [mod N](#mod-n))

Deep-learning attacks train a neural network to read the secret from the leakage of both
shares together (for example, [MLPs](#mlp) and CNNs on a masked [AES](#aes) implementation in the ASCAD
study, [Prouff et al. 2018](https://eprint.iacr.org/2018/053)).

Suppose a network finds one masking harder to attack. There are two very different reasons
that could be:

1. **Information:** the measurements of that masking reveal *less* about the secret.
   Then it is genuinely better protection.
2. **Learnability:** the measurements carry the same or more information, but the rule
   linking them to the secret is *harder for a network to learn*. Then a network that fails
   says more about its training than about the protection.

These need different evidence, so the project separates them: first compute exactly how much
information is available, a ceiling no attacker can beat, then measure how close trained
networks get to it.

**Setup (used throughout).** The secret `s` is a byte (0–255, uniform), the mask `m` is a
fresh random byte. Each share leaks its **[Hamming weight](#hamming-weight)** (number of 1 bits) plus [Gaussian
noise](#gaussian-noise) with [standard deviation](#standard-deviation) σ. The attacker sees two noisy numbers and must guess `s`. We
chose this model because it is small enough to compute everything exactly. A published paper
uses the same kind of model for Boolean masking ([Cheng et al. 2021](https://arxiv.org/abs/2105.07436)), and Hamming-weight models
fit some real 8-bit chips ([Brier et al. 2004](https://www.iacr.org/archive/ches2004/31560016/31560016.pdf)), but that does not make it a model of any
particular device.

**Environment:** Python 3.13, CPU only, versions pinned in
[requirements.txt](requirements.txt).

## 1. Is there less information?

**Question.** Before asking why a network struggles, check whether the information is even
there. How many bits about `s` can *any* attacker get from one noisy reading of both shares?

**Where the idea came from.** Our own starting question (the information-versus-learnability
split). We later read [Masure et al. 2023](https://eprint.iacr.org/2022/863) and [Hiltenbrand et al. 2025](https://eprint.iacr.org/2025/604), which report the same direction for Boolean versus arithmetic masking, so this chapter
re-derives a published result as a check on our code, not a new finding.

**Files.**
- [ceiling.py](ceiling.py) – computes the information ceiling ([mutual information](#mutual-information) between
  the secret and the two noisy readings). For each secret it counts, over all 256 masks, the
  exact table of (weight of share 1, weight of share 2); adds the Gaussian noise; and
  integrates numerically on a grid (spacing σ/4).
- [tests_sanity.py](tests_sanity.py) – checks every fact the write-up relies on, for example
  that each share alone carries 0 bits, and the parity identity below.

```
python3 tests_sanity.py   # ~3 s, prints: all sanity checks passed
python3 ceiling.py        # ~3 s, prints the checks and the table below
```

**Result** (bits of information about the 8-bit secret):

| Noise σ | XOR | Addition mod 256 | Addition mod 251 |
|---|---|---|---|
| none | 1.150 | 0.275 | 0.260 |
| 0.5 | 0.144 | 0.108 | 0.105 |
| 1.0 | 0.047 | 0.037 | 0.031 |

Checks: each share alone gives 0 bits (to 1e-15), the noise density integrates to 1.00000, and
at small noise (σ = 0.15) the grid answer matches the exact noise-free value (1.1432 vs 1.1499 for XOR).

**What it answered.** XOR leaks **more** than addition at every noise level tested, so in this
model "XOR is harder because it reveals less" is ruled out. One bit of XOR's noiseless 1.150 has
a simple explanation: under XOR, the [parity](#parity) of the two readings' sum equals the parity of the
secret's own Hamming weight (checked for all 65,536 secret–mask pairs), and that parity splits
the 256 secrets exactly in half. [Masure et al. 2023](https://eprint.iacr.org/2022/863) state the same identity; we found it first
and read it there afterwards.

**What it raised.** If XOR has more information, then any difficulty a network has with XOR is
about *learning*. Next: do trained networks actually reach these limits?


## Related work

Prouff, Strullu, Benadjila, Cagli, and Dumas (2018), "Study of Deep Learning Techniques for
Side-Channel Analysis and Introduction to ASCAD Database" ([ePrint 2018/053](https://eprint.iacr.org/2018/053)).
Applies multilayer perceptrons and convolutional networks to a masked AES implementation and
releases an open dataset (ASCAD). Cited here as background for why training a neural network on
masked leakage is a realistic attack to study.

Prouff and Rivain (2013), "Masking against Side-Channel Attacks: A Formal Security Proof"
([EUROCRYPT 2013](https://www.iacr.org/archive/eurocrypt2013/78810139/78810139.pdf)). A standard
reference on masking as a countermeasure, giving a formal security proof for masked
implementations. Cited for the definition of masking used throughout.

Brier, Clavier, and Olivier (2004), "Correlation Power Analysis with a Leakage Model"
([CHES 2004](https://www.iacr.org/archive/ches2004/31560016/31560016.pdf)). Proposes the Hamming
distance leakage model (Hamming weight when the reference state is zero) and reports that it fits
power measurements of various 8-bit chips well, and a 32-bit device less well. Cited as evidence
that Hamming-weight leakage is a reasonable, though simplified, model.

Cheng, Liu, Guilley, and Rioul (2021), "Attacking Masked Cryptographic Implementations:
Information-Theoretic Bounds" ([arXiv:2105.07436](https://arxiv.org/abs/2105.07436)). Analyses
Boolean masking with Hamming-weight leakage plus Gaussian noise, the same kind of model as ours,
and derives information-theoretic bounds on attacking it. Boolean masking only, so it does not
compare the two maskings.

Masure, Meaux, Moos, and Standaert (2023), "Effective and Efficient Masking with Low Noise using
Small-Mersenne-Prime Ciphers" ([ePrint 2022/863](https://eprint.iacr.org/2022/863), EUROCRYPT 2023).
Computes the mutual information between a secret and the Hamming-weight leakage of its shares for
Boolean and arithmetic masking, and states the parity identity that explains part of Boolean
masking's leakage. Our Chapter 1 re-derives the same direction (Boolean leaks more) for two
shares, which we used as a check on our code rather than as a new finding.

Hiltenbrand, Eynard, and Poussier (2025), "On the success rate of simple side-channel attacks
against masking with unlimited attack traces" ([ePrint 2025/604](https://eprint.iacr.org/2025/604),
CASCADE 2025). Uses the same measurement model as ours and confirms that two-share arithmetic
masking carries less mutual information than Boolean masking, while showing that in their setting
arithmetic masking can still be easier to attack with unlimited traces. A reminder that more
information does not by itself mean an easier attack; this project measures information and
learning, not attack success.
## Glossary

Short definitions of the technical terms used above. Each links to a longer explanation.

### AES
The Advanced Encryption Standard, the most widely used block cipher. Side-channel research
often attacks one byte of its internal state at a time, which is why our secret is one byte.
[More](https://en.wikipedia.org/wiki/Advanced_Encryption_Standard)

### Gaussian noise
Random error drawn from a bell-shaped (normal) distribution. Here it models the measurement
noise added to each leaked Hamming weight.
[More](https://en.wikipedia.org/wiki/Normal_distribution)

### Hamming weight
The number of 1 bits in a binary number. Example: `10110000` has Hamming weight 3. A simple
and common model of how much power a chip uses when it handles a value.
[More](https://en.wikipedia.org/wiki/Hamming_weight)

### Masking
A defence that never handles a secret directly: the secret is split into random
[shares](#shares), and the device only ever works on the shares. An attacker then has to
combine the leakage of all shares, which takes far more measurements when there is noise.
[More (research paper)](https://www.iacr.org/archive/eurocrypt2013/78810139/78810139.pdf)

### MLP
Multilayer perceptron: a basic neural network made of stacked fully connected layers. All
networks in this project are small MLPs.
[More](https://en.wikipedia.org/wiki/Multilayer_perceptron)

### mod N
Modular arithmetic: the remainder after dividing by N, so numbers wrap around.
Example: `(200 + 100) mod 256 = 44`.
[More](https://en.wikipedia.org/wiki/Modular_arithmetic)

### Mutual information
How many bits a measurement tells you, on average, about a secret. For an 8-bit secret,
8 bits would mean the measurement reveals it completely and 0 bits that it reveals nothing.
No attacker can extract more than this, which is why we use it as the ceiling.
[More](https://en.wikipedia.org/wiki/Mutual_information)

### Parity
Whether a whole number is even or odd. "The parity of the Hamming weight" asks whether a
byte has an even or odd number of 1 bits.
[More](https://en.wikipedia.org/wiki/Parity_(mathematics))

### Shares
The random-looking pieces a masked secret is split into. Each share alone is uniformly
random and reveals nothing; only the shares together determine the secret.
[More](https://en.wikipedia.org/wiki/Secret_sharing)

### Side-channel attack
An attack that learns a secret from physical effects of a computation, such as power use,
electromagnetic emissions or timing, rather than from the algorithm's inputs and outputs.
[More](https://en.wikipedia.org/wiki/Side-channel_attack)

### Standard deviation
A measure of how spread out random values are. Here σ (sigma) sets how noisy each reading
is: larger σ means noisier readings and less information.
[More](https://en.wikipedia.org/wiki/Standard_deviation)

### XOR
Exclusive or: a bitwise operation that outputs 1 where two bits differ. Example:
`1100 XOR 1010 = 0110`. XOR-ing with a random mask hides a value completely.
[More](https://en.wikipedia.org/wiki/Exclusive_or)
