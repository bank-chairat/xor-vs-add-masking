# XOR vs. Addition Masking: Less Information, or Harder to Learn?

A simulated study of why a neural network attacker learns one kind of masked secret more
slowly than another, written up in the order the work was done. Each chapter says what
question we asked, where the idea came from, which files answer it, what we found, and what
it led to next.

**Scope, up front:** everything here is simulated. The leakage model is mostly one simple
model (Hamming weight plus Gaussian noise), the networks are small MLPs, and nothing is
measured on a real device. Results are claims about this model only.

## 0. Motivation and question

**[Masking](#masking)** is a standard defence against [side-channel attacks](#side-channel-attack), where an attacker learns a
secret from a device's power use or electromagnetic emissions. The device never handles the
secret `s` directly. It splits it into two random-looking pieces ([shares](#shares)), so that neither
piece alone says anything about `s`. Two common ways to split a byte:

- **Boolean masking ([XOR](#xor)):** shares `m` and `s XOR m`
- **Arithmetic masking (addition):** shares `m` and `(s + m) mod N` (see [mod N](#mod-n))

Deep-learning attacks train a neural network to read the secret from the leakage of both
shares together (for example, [MLPs](#mlp) and CNNs on a masked [AES](#aes) implementation in the ASCAD
study, [Prouff et al. 2018](https://eprint.iacr.org/2018/053); see [literature.md](literature.md)).

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

**Reading status.** [literature.md](literature.md) records every paper we used, with how much
of it we actually read (many at abstract level only) and what it does and does not support.
A claim goes into this write-up only if it is our own measurement or is supported there.

**Environment:** Python 3.13, CPU only, versions pinned in
[requirements.txt](requirements.txt).

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
[shares](#shares), and the device only ever works on the shares.

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
