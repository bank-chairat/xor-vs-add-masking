# XOR vs. Addition Masking: Less Information, or Harder to Learn?

A simulated study of why a neural network attacker learns one kind of masked secret more
slowly than another, written up in the order the work was done. Each part says what
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

**Sources.** A claim goes into this write-up only if it is our own measurement or is supported
by a paper listed in [Related work](#related-work).

**Environment:** Python 3.13, CPU only, versions pinned in
[requirements.txt](requirements.txt).


## 1. Is there less information?

**Question.** Before asking why a network struggles, check whether the information is even
there. How many bits about `s` can *any* attacker get from one noisy reading of both shares?

**Where the idea came from.** Our own starting question (the information-versus-learnability
split). We later read [Masure et al. 2023](https://eprint.iacr.org/2022/863) and [Hiltenbrand et al. 2025](https://eprint.iacr.org/2025/604), which report the same direction for Boolean versus arithmetic masking, so this part
re-derives a published result as a check on our code, not a new finding.

**Files.**
- [ceiling.py](experiments/ceiling.py) – computes the information ceiling ([mutual information](#mutual-information) between
  the secret and the two noisy readings). For each secret it counts, over all 256 masks, the
  exact table of (weight of share 1, weight of share 2); adds the Gaussian noise; and
  integrates numerically on a grid (spacing σ/4).
- [tests_sanity.py](experiments/tests_sanity.py) – checks every fact the write-up relies on, for example
  that each share alone carries 0 bits, and the parity identity below.

```
cd experiments
python3 tests_sanity.py   # ~3 s, prints: all sanity checks passed
python3 ceiling.py        # ~3 s, prints the checks and the table below
```

**Result** (bits of information about the 8-bit secret):

| Noise σ | XOR | Addition mod 256 |
|---|---|---|
| none | 1.150 | 0.275 |
| 0.5 | 0.144 | 0.108 |
| 1.0 | 0.047 | 0.037 |

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


## 2. Do trained networks reach the ceiling?

**Question.** Part 1 says XOR leaks more. So does a trained network actually get close to
that limit, and does it get as close for XOR as for addition?

**Where the idea came from.** Our own follow-up to Part 1. To measure how much a network has
extracted we use [perceived information](#perceived-information), as in
[Masure, Dumas & Prouff 2019](https://eprint.iacr.org/2019/439), and we rely on
[Bronchain et al. 2019](https://eprint.iacr.org/2019/132) for why it can never exceed the true
information. The network (an [MLP](#mlp)) and optimizer ([Adam](#adam)) were default choices,
not taken from a paper.

**How we measure the gap.** The network sees the two noisy readings and outputs a probability
for each of the 256 secrets. Its [cross entropy](#cross-entropy) on fresh test traces gives its
perceived information (PI = 8 bits − cross entropy). PI can't exceed the ceiling, so the
**distance from the ceiling** (ceiling − PI, in bits) shows how much information the network
left on the table. Lower is better, and 0 would mean a perfect attacker.

**Files.**
- [train_mlp.py](experiments/train_mlp.py) – simulates traces, trains the network, measures PI.
  Experiment 2: 60 runs over noise levels and training-set sizes. Results in
  [results_mlp.json](experiments/results_mlp.json).
- [a03_budget.py](experiments/a03_budget.py) and [overnight_queue.py](experiments/overnight_queue.py) –
  Experiment 3 (training length, 6 [seeds](#seed) per masking) and Experiments 4–8 (learning rate,
  noise, data size, network size, noise-free data). Results in [a03/](experiments/a03) and
  [overnight/](experiments/overnight). The queue is resumable: it skips runs already in its result files.
- [summarize.py](experiments/summarize.py) – rebuilds every number in this part from the saved results.
- [plot_ch2.py](experiments/plot_ch2.py) – draws the figure below.

```
cd experiments
python3 summarize.py      # seconds: all tables in this part, from saved results
python3 plot_ch2.py       # seconds: the figure below
# to rerun the training itself (CPU, slow):
python3 train_mlp.py results_mlp.json 0.5,1.0 2000,10000,50000,250000,1000000 2 4000   # ~1 hour
python3 a03_budget.py a03/stage1_4k 4000 0,1,2,3,4,5                                   # ~12 min
QUEUE_LIMIT_S=6300 python3 overnight_queue.py overnight                                # many hours
```

**Result 1: networks get close, but XOR less so (Experiment 2).** With 1,000,000 training traces
the networks reach 80–95% of the ceiling, and none exceed it, which is the check we wanted. With
only 2,000–10,000 traces every network [overfits](#overfitting). At σ = 0.5, though, XOR networks
are about three times further from their ceiling than addition networks (0.022–0.031 vs
0.008 bits, 2 seeds).

**Result 2: the gap is about training time (Experiment 3, σ = 0.5, 6 seeds).**

![Distance from the ceiling vs. training steps](experiments/results/ch2_gap_vs_steps.png)

| Training steps | XOR | Addition |
|---|---|---|
| 4,000 | 0.0260 | 0.0085 |
| 16,000 | 0.0086 | 0.0083 |
| 64,000 (3 seeds) | 0.0143 | 0.0160 |

At 4,000 steps every addition seed beats every XOR seed, and XOR's seeds vary about 13 times more.
At 16,000 steps the gap is gone: addition was already at its level by 4,000 steps, while XOR
needed four times longer to get there. At 64,000 steps both got *worse*, a puzzle we come back
to in Part 4.

**Result 3: what the 4,000-step gap depends on (Experiments 4–8, 3–4 seeds each).**
- [Learning rate](#learning-rate): no value from 0.0005 to 0.008 closes it.
- Network shape: a deeper network closes most of it (depth 5: 0.0101 vs 0.0070), a shallower one
  widens it (depth 2: 0.0439 vs 0.0155), and width alone does not help.
- Training data: with less data XOR falls further behind (200,000 traces, best checkpoint: 0.049 vs 0.028).
- Noise: at 16,000 steps XOR is never clearly further from its ceiling than addition (about equal at
  σ = 0.5 and 0.75), and at low noise (σ = 0.25) it is clearly closer (0.013 vs 0.030).
- Noise-free data: trained on the exact probabilities, all 20 XOR networks learn the parity
  clue from Part 1 almost perfectly (0.999 of their probability on answers with the right parity).

**What it answered.** XOR isn't harder because the network can't reach its limit. Given enough
steps, depth or data, it gets there. XOR simply needs *more training* to get as close as addition.

**What it raised.** Why does XOR need more training? Experiment 8 hints at an answer: the parity
clue. Next we look at where exactly XOR's networks fall short, and when during training they
learn the parity.


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
masking's leakage. Our Part 1 re-derives the same direction (Boolean leaks more) for two
shares, which we used as a check on our code rather than as a new finding.

Hiltenbrand, Eynard, and Poussier (2025), "On the success rate of simple side-channel attacks
against masking with unlimited attack traces" ([ePrint 2025/604](https://eprint.iacr.org/2025/604),
CASCADE 2025). Uses the same measurement model as ours and confirms that two-share arithmetic
masking carries less mutual information than Boolean masking, while showing that in their setting
arithmetic masking can still be easier to attack with unlimited traces. A reminder that more
information does not by itself mean an easier attack; this project measures information and
learning, not attack success.

Masure, Dumas, and Prouff (2019), "A Comprehensive Study of Deep Learning for Side-Channel
Analysis" ([ePrint 2019/439](https://eprint.iacr.org/2019/439)). Argues that training a network by
minimizing its negative log-likelihood is equivalent, in the limit, to maximizing a lower bound on
mutual information, and links training to perceived information. Cited as the source for using
perceived information to measure how much a trained network extracts.

Bronchain, Hendrickx, Massart, Olshevsky, and Standaert (2019), "Leakage Certification Revisited:
Bounding Model Errors in Side-Channel Security Evaluations" ([ePrint 2019/132](https://eprint.iacr.org/2019/132),
CRYPTO 2019). Defines perceived information as the information an estimated model can extract and
shows it cannot exceed the true mutual information when the secret is uniform. Cited for why our
networks can never beat the ceiling from Part 1.

## Glossary

Short definitions of the technical terms used above. Each links to a longer explanation.

### Adam
A popular training method (optimizer) that adjusts how big a step each weight takes, based on
the recent size of that weight's gradients.
[More](https://arxiv.org/abs/1412.6980)

### AES
The Advanced Encryption Standard, the most widely used block cipher. Side-channel research
often attacks one byte of its internal state at a time, which is why our secret is one byte.
[More](https://en.wikipedia.org/wiki/Advanced_Encryption_Standard)

### Cross entropy
How surprised a model is, on average, by the true answer, measured in bits. Lower is better.
A model that guesses uniformly among 256 secrets has a cross entropy of 8 bits.
[More](https://en.wikipedia.org/wiki/Cross-entropy)

### Gaussian noise
Random error drawn from a bell-shaped (normal) distribution. Here it models the measurement
noise added to each leaked Hamming weight.
[More](https://en.wikipedia.org/wiki/Normal_distribution)

### Hamming weight
The number of 1 bits in a binary number. Example: `10110000` has Hamming weight 3. A simple
and common model of how much power a chip uses when it handles a value.
[More](https://en.wikipedia.org/wiki/Hamming_weight)

### Learning rate
How big a step the optimizer takes at each training update. Too small learns slowly; too large
is unstable.
[More](https://en.wikipedia.org/wiki/Learning_rate)

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

### Overfitting
When a model memorizes its training data instead of learning the general rule, so it does
worse on new data.
[More](https://en.wikipedia.org/wiki/Overfitting)

### Parity
Whether a whole number is even or odd. "The parity of the Hamming weight" asks whether a
byte has an even or odd number of 1 bits.
[More](https://en.wikipedia.org/wiki/Parity_(mathematics))

### Perceived information
How many bits a trained model actually extracts: 8 bits minus its cross entropy on fresh test
data. It can never exceed the mutual information, so the gap between the two shows what the
model missed.
[More](https://eprint.iacr.org/2019/132)

### Seed
The number that fixes a run's random choices (starting weights, data order) so it can be
repeated exactly. Running several seeds shows how much results vary by chance.
[More](https://en.wikipedia.org/wiki/Random_seed)

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
