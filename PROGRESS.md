# XOR vs Addition Masking: Progress Log

> **About this file.** This is the full lab notebook, kept as it was written during the project, with
> every plan, prediction and result in order (newest at the bottom). The [README](README.md) tells the
> same story in shorter form. Notes for reading it here:
> - Script and result paths (for example `a30/`, `parity_tests.py`) are inside the [experiments](experiments) folder.
> - `literature.md` was the private reading log; the papers we relied on are listed under
>   [Related work](README.md#related-work) in the README.
> - "Outside chat" or "outside bundle" means suggestions from a separate discussion. They were treated
>   as unverified until reproduced here (Experiments 23 and 24).
> - Later corrections are marked in place, for example "[corrected ...]".


Each step says what we did, why we did it, what came out, and what we took from it.
Claims are limited to what we measured, proved, or can cite. Anything else is marked as
not tested or not sourced. Newest step at the bottom.

Every experiment has a line **Where the idea came from**. It says whether the experiment came
from our own question, from an earlier result of ours (named), or from a published paper
(named, with the reading status in literature.md). If a paper was found only after we designed
an experiment, the line says so. We do not claim a paper motivated an experiment unless it did.

## Question

Masking protects a secret by splitting it into two random looking pieces, so that neither
piece alone reveals it. Two common ways to split a secret are XOR (Boolean masking) and
addition mod N (arithmetic masking). Suppose a neural network finds one of them harder to
attack than the other. There are two different possible reasons:

1. **Information:** the measurements of one reveal less about the secret.
2. **Learnability:** the measurements carry the same or more information, but the
   rule connecting them to the secret is harder for a network to learn.

These need different evidence, so we separate them. First we compute exactly how much
information is available (a ceiling no attacker can beat). Then we measure how close
trained networks get to that ceiling. Which of the two maskings is harder to attack in
practice is not something we have sourced. Related published work is recorded in
literature.md with the reading status of each paper (abstracts only so far). The one paper
found that compares the information of the two encodings, Hiltenbrand, Eynard and Poussier
(ePrint 2025/604), states in its abstract that the average information of arithmetic
encoding is "proven less informative than the Boolean one", in a different setting (simple
power analysis with a fixed known target). That is the same direction as Experiment 1 below,
but it is not the same setting, so it does not confirm our numbers.

## Setup

Everything is simulated. A secret s is a number from 0 to 255 (chosen uniformly), and a
mask m is a fresh random number from 0 to 255. The device holds two pieces: piece 1 is m,
and piece 2 is s XOR m (Boolean masking) or (s + m) mod N (arithmetic masking, with
N = 256 or 251). Each piece leaks its Hamming weight (the number of 1 bits in it) plus
Gaussian noise of standard deviation sigma. The attacker sees two noisy numbers and must
guess s.

We chose Hamming weight plus noise because it is small enough to compute everything
exactly instead of estimating it from samples. We have not cited a source that it matches
real devices, so all results hold for this model only.

Environment: Linux, 2 CPU cores, Python 3.13, numpy, scipy, matplotlib, torch 2.14 (CPU).

## Experiment 1 (A-01): the information ceiling

**Where the idea came from.** Our own question (the information versus learnability split, chosen
at the start). The paper we later checked for why mutual information is a valid yardstick is
Bronchain et al. 2019 (Theorem 6: for a uniform secret, perceived information cannot exceed mutual
information; tool summary). Hiltenbrand et al. 2025 (p. 11, read in full) and Masure et al. 2023
(Sec. 2.2 and Fig. 1, pages 1 to 10 read) were read after this experiment was run; both report the same
direction (Boolean leaks more than arithmetic), so this experiment re-derives a published result for
2 shares, as a check on our code and not as a new finding. The parity identity below is stated in
Masure et al. 2023, Sec. 2.4, which we also read after finding it ourselves.

**1a. Check that each piece alone reveals nothing.** m is random, and s XOR m is also
uniformly random whatever s is. The code computes the information in each piece alone and
gets 0 bits (to 1e-15). Why: this is the whole point of masking, so a nonzero value would
mean the code is wrong.

**1b. Build the exact table.** For each secret, loop over all 256 masks, compute both
Hamming weights, and count each pair. This gives a 9 by 9 table per secret (weights run
0 to 8). Why: with only 256 masks the counting is exact, so there is no sampling error.

**1c. Add the noise.** Given a secret, the measurement is a cloud of bell curves, one per
cell of the table, centered on (h1, h2) with the table's counts as weights. Different
secrets give different clouds. Why: how different the clouds are decides how well any
attacker can tell the secrets apart.

**1d. Convert to bits.** Mutual information says how many bits of the secret are learned
from one measurement (8 bits would be everything, 0 nothing). It averages, over secrets
and measurements, how much likelier the measurement is under the true secret than under
a random one. The integral has no closed form, so we lay a grid over the measurement plane
(spacing a quarter of sigma) and sum. Checks: the density integrates to 1.00000, and at
tiny noise the grid answer matches the exact noise free sum (1.1432 vs 1.1499 for XOR).

**Result.**

| Noise | XOR | Add mod 256 | Add mod 251 |
|---|---|---|---|
| none | 1.150 bits | 0.275 | 0.260 |
| sigma 0.5 | 0.144 | 0.108 | 0.105 |
| sigma 1.0 | 0.047 | 0.037 | 0.031 |

**What we learned.** Measured: XOR leaks more than addition in this model, at every noise
level tested. Provable: under XOR the parity of h1 + h2 equals the parity of the secret's
own Hamming weight (checked for all 65,536 secret and mask pairs), and that parity splits
the 256 secrets into two equal halves, so with no noise it accounts for exactly 1 bit of
the 1.150. The remaining 0.150 bits are not explained, and we have not tested why addition
lacks a similar shortcut. So "XOR is harder because it reveals less" is contradicted for
this model. Code: ceiling.py.

## Experiment 2 (A-02): networks against the ceiling

**Where the idea came from.** Our own question: does a trained network reach the ceiling from
Experiment 1? The measure (8 bits minus test cross entropy) is the perceived information of
Masure et al. 2019 and Bronchain et al. 2019 (both read at abstract or summary level; see
literature.md). The choice of an MLP and of Adam was a default, not taken from a paper.

**What we did.** Simulated traces under both maskings at noise 0.5 and 1.0. Trained a
3 layer MLP (256 wide, ReLU, Adam, learning rate 2e-3 with cosine decay, 4,000 steps,
batch 1,024) to predict s from the two readings, using 2,000 to 1,000,000 training traces,
2 seeds each, 60 runs. The best checkpoint was chosen on validation data. We measured
perceived information (PI = 8 bits minus test cross entropy) on 400,000 fresh traces.

**Why.** PI cannot exceed the true information (up to sampling error), so comparing it to
the ceiling shows how much a network extracted. No run exceeded the ceiling (highest was
95% of it), which is the check we wanted.

**Result.** With 1,000,000 traces the networks reached 80% to 95% of their ceiling.
With 2,000 and 10,000 traces every network overfit and did worse than a uniform guess.

**Metric choice.** "Fraction of ceiling" hides a gap because XOR's ceiling is larger, so
the primary metric is excess cross entropy = ceiling minus PI, in bits. At 1,000,000 traces
and sigma 0.5 it is 0.022 to 0.031 bits for XOR and 0.008 to 0.009 for addition (2 seeds).
At sigma 1.0 the order flips slightly (XOR about 0.0025, addition about 0.004). The cause
of the sigma 0.5 gap is not established.

**Limitations of this experiment.** The initialization seed and data seed were the same
number. No training curves were saved. One architecture, one training budget, two seeds.
Experiment 3 addresses these.

## Experiment 3 (A-03): is the gap training budget, seed, or the rule?

**Where the idea came from.** An earlier result of ours: Experiment 2 showed XOR farther from its
ceiling than addition at sigma 0.5 (2 seeds, with a coupled initialization and data seed), and we
could not tell whether that was a real effect. No paper suggested this; it is the minimum follow-up
(more seeds, separate seeds, longer training) before believing a 2-seed difference.

**What we are doing.** Sigma 0.5, N = 256, 1,000,000 training traces, same network and
optimizer for both maskings, separate initialization and data seeds, training curves saved,
and both the final and the best validation checkpoint recorded. Three stages by training
length: 4,000 steps (6 seeds each), 16,000 steps (6 seeds each), 64,000 steps (3 seeds each).

**Why.** If more steps close the gap, training budget was the cause. If seeds disagree
widely, it is an optimization issue. If the gap stays, the rule may genuinely be harder
to learn (this still would not prove it).

**Stage 1 result (4,000 steps, 6 seeds each).** Excess cross entropy in bits:

| Masking | Per seed | Mean | Spread |
|---|---|---|---|
| XOR | 0.0342, 0.0216, 0.0233, 0.0212, 0.0248, 0.0307 | 0.0260 | 0.0053 |
| Addition | 0.0079, 0.0084, 0.0090, 0.0087, 0.0082, 0.0085 | 0.0085 | 0.0004 |

XOR is about 3 times farther from its ceiling. The worst addition seed beats the best XOR
seed, and XOR varies about 13 times more between seeds. This holds only at 4,000 steps.

**Stage 2 and 3 (16,000 and 64,000 steps).** The first job was stopped early by the
environment after two XOR runs at 16,000 steps, so the remaining runs were redone with the same
seeds and settings in a resumable queue (overnight_queue.py). Final-step excess cross entropy
in bits, mean over seeds (sd in brackets), as read from the saved files:

| Training steps | XOR | Addition |
|---|---|---|
| 4,000 | 0.0260 (0.0053), 6 seeds | 0.0085 (0.0004), 6 seeds |
| 16,000 | 0.0086 (0.0008), 6 seeds | 0.0083 (0.0004), 6 seeds |
| 64,000 | 0.0143 (0.0013), 3 seeds | 0.0160 (0.0016) [corrected 28 Oct from 0.0022; recomputed from overnight/A-03.csv], 3 seeds |

What this shows (measured): at 16,000 steps XOR is as close to its ceiling as addition (0.0086
vs 0.0083), so the 4,000 step gap goes away with more training. Addition was already at its
level by 4,000 steps, so XOR needed more steps, not a different outcome. At 64,000 steps both
maskings are further from the ceiling than at 16,000 steps, and the final checkpoint is worse
than the best validation checkpoint (for example addition seed 0: 0.0176 final vs 0.0117 best
validation). Even the best validation checkpoint at 64,000 steps (XOR 0.0124, addition 0.0117) is
worse than at 16,000 steps (0.0087, 0.0083). Overfitting to the 1,000,000 training traces would fit
the final vs best validation difference, but we have not tested it, and the cause of the
64,000 step result is unknown (the cosine learning rate schedule is stretched to the run length,
so these runs differ in more than length).

What this does not show: that the gap is "only" training time in general. It was measured for
one network, one learning rate, one noise level (sigma 0.5), one data size. Experiments 4 to 7
vary those.

## Experiments 4 to 8 (A-04 to A-08): what the XOR gap depends on

**Where the ideas came from.** A-04 to A-07 (learning rate, data size, noise, network size) came from
Experiment 3: once the XOR gap was shown to close with more steps, we varied the usual training
settings one at a time to see which of them it depends on. None came from a paper; the choice of
what to vary was ours. (Perin and Picek 2020, an optimizer study, was found afterward and is
background only; we did not change optimizers.) A-08 (the noiseless exact table) came from the
parity identity of Experiment 1 and from a small earlier pilot report we were shown, which described
parity being learned slowly or not at all by some XOR networks; we did not reproduce that.

All use sigma 0.5, N = 256, 1,000,000 training traces unless stated, the same network (3 hidden
layers of 256) and Adam with a cosine schedule. Values are final-step excess cross entropy in bits,
mean over seeds (3 or 4 per cell; Experiment 3 values for the baseline row). Lower is better.

**A-04, learning rate (4,000 steps).** No learning rate closes the gap at 4,000 steps.

| Learning rate | XOR | Addition |
|---|---|---|
| 0.0005 | 0.0389 | 0.0117 |
| 0.001 | 0.0312 | 0.0093 |
| 0.002 (Experiment 3) | 0.0260 | 0.0085 |
| 0.004 | 0.0315 | 0.0087 |
| 0.008 | 0.0360 | 0.0104 |

**A-06, training-set size (16,000 steps, best validation checkpoint).** With less data XOR falls
further behind. At 50,000 traces both overfit badly by the final step (1.4 to 1.6 bits), so only the
best validation checkpoint is meaningful.

| Training traces | XOR | Addition |
|---|---|---|
| 50,000 | 0.0959 | 0.0687 |
| 200,000 | 0.0493 | 0.0275 |
| 1,000,000 (Experiment 3) | 0.0087 | 0.0083 |

**A-05, noise level (16,000 steps).** At 16,000 steps XOR is not further from its ceiling than
addition at any noise level tested: about equal at sigma 0.5 and 0.75, closer at 0.25, 1.0 and 1.5.
Shorter training was not tested at these noise levels (Experiment 2 had 4,000 steps at sigma 0.5 and 1.0).

| Sigma | XOR | Addition |
|---|---|---|
| 0.25 | 0.0133 | 0.0298 |
| 0.5 (Experiment 3) | 0.0086 | 0.0083 |
| 0.75 | 0.0056 | 0.0061 |
| 1.0 | 0.0052 | 0.0062 |
| 1.5 | 0.0040 | 0.0070 |

**A-07, network size (4,000 steps).** A deeper network closes most of the XOR gap at 4,000 steps; a
shallower one widens it. Width alone does not.

| Network | XOR | Addition |
|---|---|---|
| width 256, depth 2 | 0.0439 | 0.0155 |
| width 64, depth 3 | 0.0374 | 0.0118 |
| width 256, depth 3 (Experiment 3) | 0.0260 | 0.0085 |
| width 512, depth 3 | 0.0276 (sd 0.0091) | 0.0081 |
| width 256, depth 5 | 0.0101 | 0.0070 |

**A-08, noiseless, exact table (20 seeds each, small network of 20,992 parameters, Adam lr 0.003).**
Trained on the exact probabilities of all 81 pairs of Hamming weights, no sampling. At 20,000
steps the XOR networks put 0.9992 of their probability (mean; min 0.9960) on answers with the right
parity and excess cross entropy is 0.0027 for both maskings (XOR sd 0.0032, addition sd 0.0020).
For addition the parity probability stays at 0.513, as expected since addition has no such link.
At 2,000 steps all 20 XOR seeds were already at 0.980 to 0.997. We did not reproduce a weak XOR
start (a pilot we were shown reported one at 0.50 at 2,000 steps); we have not seen that pilot's
code, so differences in initialization or setup could explain it, and this does not show it was wrong.

**What these show together (measured).** The XOR shortfall is a training-length, depth and data
effect in this model: it shrinks with 4x more steps (Experiment 3), with a deeper network (A-07), and
with more data (A-06), and it was not closed by any learning rate from 0.0005 to 0.008 at 4,000 steps (A-04). At 16,000 steps XOR is closer to its
ceiling than addition at sigma 0.25 (A-05). None of this identifies a harder rule; it identifies that the XOR networks needed
more training resources to reach the same distance from their ceiling.

**What it does not show.** Other leakage models, other architectures (only MLPs of these shapes),
other optimizers, longer trace models with hidden positions, real measurements. Cell sizes are 3 to
6 seeds, so small differences should not be interpreted. The reason XOR needs more training resources
is unexplained; one candidate (the parity structure being harder to represent) was not tested directly.

## Experiment 9 (A-09): why does XOR need more training? (predictions written before running)

**Where the idea came from.** Our own reasoning from earlier results, before the search for papers
on this question: (1) the parity identity from Experiment 1; (2) A-07, where a deeper network
shrinks the XOR gap, suggesting the XOR rule needs more internal structure; (3) A-08, where the
parity of the secret's Hamming weight is the quantity XOR networks learn. The periodic features
in Test F came from the idea that the parity of a sum is a zigzag function of the readings. No paper
suggested the design. Papers found afterward that bear on it: Rahaman et al. 2019 (spectral bias)
and Tancik et al. 2020 (Fourier features); arXiv 2502.00384 reports a late parity phase on real
Boolean-masked traces. See Experiment 10.

Candidate explanation to test: under XOR the useful clue is the parity of the sum of the two
readings, a function that flips every time the sum changes by one. A network with plain inputs
must build that zigzag itself, which may take many training steps; addition has no such clue.
Two tests, both at sigma 0.5, 1,000,000 traces, 4,000 steps unless stated, 4 seeds per cell:

**Test D (where does the shortfall live?).** Split each network's cross entropy into two parts
by the chain rule: (1) the cost of getting the parity group of the secret (even or odd Hamming
weight) right, and (2) the cost of picking the secret within its group. Compare each part with the
exact best possible values, computed from the exact posterior on the same test traces. Prediction
if parity is the bottleneck: for XOR networks at 4,000 steps most of the excess is in part (1), and
at 16,000 steps that part has shrunk. If instead the excess is mostly in part (2) or does not move
with training, the parity explanation is wrong.

**Test F (give the network the clue).** Train with extra input features, same training otherwise:
F0 plain inputs (baseline); F1 plain plus cos(pi*(r1+r2)) (a smooth stand-in for parity of the
sum, from the noisy readings only, no labels); F2 plain plus cos and sin of pi*r1 and pi*r2 (periodic
features per reading; parity of the sum then needs a product); F3 plain plus r1+r2 (linear control,
which the network could compute itself). Prediction if parity is the bottleneck: F1 brings XOR at
4,000 steps close to addition (excess near 0.01 or below), F3 changes nothing, and addition is not
helped much by F1. If F1 does not help XOR, the explanation is wrong or incomplete.

## Experiment 10 (A-10): is it the frequency of the parity clue? (done; predictions were written before running)

**Where the idea came from.** Experiment 9 (a parity-like input feature brought XOR close to addition at
4,000 steps) combined with two papers found after A-09 was designed: Rahaman et al. 2019 (ICML), which
reports that ReLU networks learn low-frequency functions before high-frequency ones, and Tancik et al.
2020 (NeurIPS), which reports that feeding low-dimensional inputs through Fourier features lets an MLP
learn high-frequency functions it otherwise learns very slowly. Both were read only through a fetch tool
summary (literature.md). Neither studies masking or parity of noisy readings, so this is a
hypothesis, not something the papers show for our setting.

**Hypothesis.** XOR is slow at 4,000 steps because the useful clue, the parity of r1 + r2, changes sign
every time r1 + r2 moves by 1, which is a high-frequency function of the inputs that a plain MLP learns late.

**Design.** Sigma 0.5, 1,000,000 traces, 4,000 steps, same network and training as A-09, 4 seeds, XOR and
addition. Plain inputs and the A-09 feature cos(pi * (r1 + r2)) are reused from A-09 (same seeds, same code).
New extra inputs: G1 cos(pi/2 * (r1 + r2)) (half the frequency); G2 cos(2 pi * (r1 + r2)) (double the
frequency, equal to 1 at every integer sum, so uninformative at noise-free readings); G3
cos(pi * (r1 + r2) + pi/4) (correct frequency, shifted phase); G4 sixteen random Fourier features
(cosine and sine of random projections of the two readings, projection weights from a Gaussian with
standard deviation 1), which do not use any knowledge of parity.

**Predictions if the frequency hypothesis is right.**
- G3 helps XOR about as much as the A-09 feature (excess near 0.010 bits at 4,000 steps).
- G1 and G2 do not help XOR (excess stays near the plain-input value, 0.02 to 0.03).
- G4 helps XOR to some degree. How much depends on whether random frequencies land near pi, so a weak
  result there would not contradict the hypothesis.
- None of them changes addition much.
**If instead** G1 or G2 helps XOR as much as the correct frequency, the hypothesis is wrong or the
effect of A-09 came from something other than frequency, for example just a larger input
dimension or a cheap non-linearity. If G3 does not help, phase matters more than we thought.

### A-09 results (final: 40 runs, 4 seeds per cell)

Sigma 0.5, 1,000,000 traces, N = 256, 3 x 256 MLP, final-step excess cross entropy in bits,
mean over 4 seeds (sd in brackets). "Parity part" is the cost of getting the secret's parity group
(even or odd Hamming weight) right; "within part" is the cost of picking the secret inside its group.
Both are measured against the exact best possible on the same test traces. Script: parity_tests.py.

**Test D, where the shortfall lives.**

| Masking, steps | Total excess | Parity part | Within part |
|---|---|---|---|
| XOR, 4,000 | 0.0248 (0.0062) | 0.0180 | 0.0068 |
| XOR, 16,000 | 0.0079 (0.0006) | 0.0014 | 0.0065 |
| Addition, 4,000 | 0.0090 (0.0005) | 0.0007 | 0.0083 |
| Addition, 16,000 | 0.0086 (0.0003) | 0.0005 | 0.0081 |

About 72 percent of XOR's excess at 4,000 steps is in the parity part (0.0180 of 0.0248); for addition it is
about 8 percent. With four times the steps XOR's parity part falls from 0.0180 to 0.0014. The within
part is about 0.007 to 0.008 for both maskings and does not depend on masking type in these runs.

**Test F, giving the network the clue (4,000 steps).** F0 plain inputs; F1 plus cos(pi (r1 + r2));
F2 plus cosine and sine of pi r1 and of pi r2; F3 plus r1 + r2 (linear control). Extra inputs use only the
noisy readings, never the label.

| Masking, extra input | Total excess | Parity part | Within part | Per-seed total |
|---|---|---|---|---|
| XOR, F0 | 0.0248 | 0.0180 | 0.0068 | 0.0339, 0.0213, 0.0230, 0.0209 |
| XOR, F1 | 0.0106 (0.0002) | 0.0039 | 0.0066 | 0.0105, 0.0103, 0.0108, 0.0106 |
| XOR, F2 | 0.0097 (0.0006) | 0.0014 | 0.0084 | 0.0089, 0.0103, 0.0101, 0.0095 |
| XOR, F3 | 0.0322 (0.0088) | 0.0245 | 0.0076 | 0.0399, 0.0274, 0.0392, 0.0222 |
| Addition, F0 | 0.0090 (0.0005) | 0.0007 | 0.0083 | |
| Addition, F1 | 0.0088 (0.0003) | 0.0006 | 0.0082 | |
| Addition, F2 | 0.0092 (0.0003) | 0.0007 | 0.0086 | |
| Addition, F3 | 0.0121 (0.0028) | 0.0008 | 0.0113 | 0.0162, 0.0107, 0.0112, 0.0102 |

**What this shows (measured).** The XOR shortfall at 4,000 steps is mostly in the parity part. Adding
F1 removes about 90 percent of the XOR gap to addition (0.0248 to 0.0106, against addition's 0.0090) and
F2 about 96 percent (0.0097). The linear control F3 does not help (0.0322, within seed spread of the plain
value or slightly worse), so the benefit is not from any extra input. F1 and F2 do not change addition. This
fits the prediction we wrote down before running, that parity is the bottleneck.

**What it does not show.** It does not show that parity is the only cause: F1 leaves a parity part of 0.0039
against 0.0007 for addition, and F2 lowers the parity part to 0.0014 while raising the within part
(0.0084 against 0.0068); we do not know why. Per seed, the within part is 0.0061 to 0.0072 for plain
XOR, 0.0065 to 0.0067 with F1, and 0.0074 to 0.0091 with F2 (every F2 seed above every plain seed, only 4
seeds each); for addition it is 0.0077 to 0.0088 plain and 0.0081 to 0.0090 with F2 (overlapping). So F2
brings XOR's within part up to addition's level (0.0084 vs 0.0083), and plain XOR's within part is lower
than addition's. Untested explanations: the extra input dimensions themselves, or periodic inputs adding
noise to the smooth within-group part. Experiment 10 records both parts for wrong-frequency inputs, which
separates these. It does not show why parity is slow (the frequency hypothesis is
Experiment 10). It does not show anything about the network's internal representation, because we did not
probe it. One noise level, one network type, 4 seeds.

### A-10 results (final: 32 runs, 4 seeds per cell, 4,000 steps, sigma 0.5)

Excess cross entropy over the exact ceiling (bits), mean of 4 seeds; "between" is the parity part, "within" the rest.
Plain input and the A-09 feature are from A-09 (same seeds).

| XOR input | excess | parity part | within part |
|---|---|---|---|
| plain (F0) | 0.0248 | 0.0180 | 0.0068 |
| cos(pi (r1+r2)) (F1, A-09) | 0.0106 | 0.0039 | 0.0066 |
| G1 cos(pi/2 (r1+r2)) | 0.0119 | 0.0053 | 0.0067 |
| G2 cos(2 pi (r1+r2)) | 0.0306 | 0.0234 | 0.0072 |
| G3 cos(pi (r1+r2) + pi/4) | 0.0130 | 0.0058 | 0.0071 |
| G4 16 random Fourier features | 0.0088 | 0.0009 | 0.0079 |

Addition is unchanged by every feature: excess 0.0080 to 0.0094 (plain 0.0090), parity part 0.0005 to 0.0007.
Per-seed XOR excess: G1 0.0185, 0.0086, 0.0089, 0.0117; G2 0.0340, 0.0307, 0.0315, 0.0263;
G3 0.0134, 0.0125, 0.0140, 0.0121; G4 0.0086, 0.0090, 0.0089, 0.0086.

**Against the predictions.**
- G3 (right frequency, shifted phase) helps XOR, as predicted: 0.0130 versus plain 0.0248 (the unshifted feature gave 0.0106).
- G2 (double frequency) does not help, as predicted; its mean is even above the plain mean, but the plain runs vary a lot
  (0.0209 to 0.0339), so only "no help" is supported, not "hurts".
- G1 (half frequency) was predicted not to help and did help (0.0119). This contradicts the strict form of the frequency
  hypothesis, in which only the exact frequency of the parity pattern would help.
- G4 (random features, no parity knowledge) was predicted to help partially and helped about as much as the exact feature
  (0.0088, parity part 0.0009, all four seeds within 0.0004 of each other).
- Addition unaffected, as predicted.

**What this shows.** Giving the network a few sinusoidal views of the readings, even ones that do not match the parity
pattern, removes most of the XOR gap at 4,000 steps; a feature at twice the frequency does not. So the parity delay behaves
like a problem of building the right oscillating inputs from plain readings, and a fixed set of random sinusoids is enough
to avoid it. This is the direction of Rahaman et al. 2019 and Tancik et al. 2020 (spectral bias, Fourier features),
but those papers are not about this setting and I read them only through a fetch tool summary.

**What it does not show.** It does not show that spectral bias is the cause: H1 (weak signal), H4 (plateau and escape) and
the simple "more input dimension" idea are not separated, since G1 and G4 also add inputs and the one control with
extra dimension and no help (G2) is a single feature. The sigma sweep with input rescaling and per-step parity curves
(see candidate explanations) are what would separate them. The within part is about 0.0067 to 0.0072 for the single
features and 0.0079 for G4, so the earlier rise of the within part with more inputs (F2) is not reproduced for the
single features; G4 is slightly higher but with 4 seeds I cannot say that is more than noise. Four seeds per cell, one noise
level, one network.

## Experiment 11 (A-11): when is parity learned, and is the within part a symmetry cost? (done; predictions were written before running)

**Where the idea came from.** Two gaps left by Experiments 9 and 10. (1) A-09 measured the parity part only at the end of
training; the plateau-then-escape idea (H4, from Barak et al. 2022, Abbe et al. 2023 and the Scoop/Don't Learn papers, which
describe a plateau then a drop for masked leakage) says it should stay flat and then fall at a seed-dependent step, while
the weak-signal idea (H1) says it should shrink gradually. Only per-step curves can tell these apart. (2) For XOR the exact
posterior is uniform over secrets with the same Hamming weight (verified numerically, Hiltenbrand et al. Prop. 1), so any
difference a network makes between same-weight secrets is pure waste (H8). We never measured how much of the XOR within part
is that.

**Design.** Same data, seeds, network (3x256 ReLU), Adam 2e-3, cosine over 4,000 steps, 1M traces, sigma 0.5 as A-09, 4 seeds;
XOR plain, addition plain, XOR with the extra input cos(pi (r1+r2)). The code re-trains rather than loading earlier weights
because the default 4,000-step networks were not saved (only the best-validation state of other runs was). Per-step curves
(steps 1, 25, 50, 100, then every 100) on the 100k validation traces; final split on the 400k test traces. Excess over the exact
Bayes posterior, in three chain-rule levels: A parity group of the secret, B Hamming-weight class inside the group,
C the secret inside its class. A + B + C = total excess; A + (B + C) = the between/within split of A-09. The cosine schedule
is stretched to the run length, so onset steps depend on that schedule too.

**Predictions (made before the run).**
- Reproduction: final total excess for XOR plain, addition plain and XOR+feature should equal the A-09 values per seed (same
  seeds, deterministic CPU code). If not, something differs and the run is not used.
- Start value: A for XOR at step 1 is about 0.0364 (the whole parity-group information, since a fresh network knows nothing).
- H4 (plateau and escape): for plain XOR, A stays within about 10% of its start value for the first part of training and then falls
  fast, with the step where it first drops below half its start differing between seeds by more than 20%. Under H1 (gradual) A would
  decline smoothly from the start and the onset step would be similar across seeds.
- XOR with the cosine feature: A below half its start value by step 500 in all seeds, and little seed spread.
- Addition: A is already small (under 0.005) from the first evaluation.
- H8 (symmetry cost): for plain XOR at 4,000 steps, C is the largest part of the within part (C greater than B and more than half
  of B + C). If C is small, H8 is not what keeps the within floor above zero.
**What would change my mind.** Smooth, seed-independent A curves for plain XOR argue against H4. C smaller than B argues against H8.

### A-11 results (final: 12 runs, 4 seeds per cell, 4,000 steps, sigma 0.5)

Reproduction check passed: the total excess of every run equals the A-09 value for the same seed to within 0.000005 bits.
The exact XOR posterior is uniform inside each Hamming-weight class to within 1.5e-8 (numerical check of Hiltenbrand Prop. 1).

Final test excess (bits), mean of 4 seeds:

| Input | A parity group | B class inside group | C secret inside class | total |
|---|---|---|---|---|
| XOR plain | 0.0180 | 0.0039 | 0.0028 | 0.0248 |
| XOR + cos(pi (r1+r2)) | 0.0039 | 0.0031 | 0.0035 | 0.0106 |
| addition plain | 0.0007 | 0.0018 | 0.0065 | 0.0090 |

Parity part A on the 100k validation traces, plain XOR (start value 0.0353 to 0.0356, i.e. the whole parity-group information):

| seed | step 500 | 1000 | 2000 | 3000 | 4000 | first step below half the start |
|---|---|---|---|---|---|---|
| 0 | 0.0351 | 0.0347 | 0.0313 | 0.0281 | 0.0274 | never |
| 1 | 0.0353 | 0.0325 | 0.0217 | 0.0163 | 0.0148 | 2800 |
| 2 | 0.0351 | 0.0341 | 0.0243 | 0.0180 | 0.0165 | 3200 |
| 3 | 0.0351 | 0.0315 | 0.0201 | 0.0151 | 0.0141 | 2300 |

With the cosine feature A starts to fall at once and first drops below half at steps 1000, 900, 1200 and 900.
For addition A starts at only 0.0014, so there is little parity information to learn there.

**Against the predictions.**
- Reproduction and the start value (about 0.0364 predicted, 0.0353 to 0.0356 measured on validation traces; the small difference is a sampling difference between trace sets, not checked further): held.
- Plateau (H4): partly held. For plain XOR the parity part stays within about 10% of its start for the first ~1,000 steps, and the step where it
  halves differs by more than 20% between seeds (2300 to 3200, one seed never). But the fall after the plateau is gradual over
  about 2,000 steps, not abrupt, so "plateau then sharp escape" is not what we see.
- With the feature: the prediction (below half by step 500) failed; it takes about 900 to 1,200 steps, though with little seed spread as predicted.
- Symmetry cost (H8): failed. For plain XOR, C (cost of treating same-weight secrets differently) is 0.0028, less than B (0.0039) and
  about 42% of B + C. H8 is not the main thing behind the XOR within part. It is also not zero.
- Addition: the prediction held trivially (little parity information exists to be learned).

**What this shows.** For plain XOR the network first does nothing about parity for roughly a quarter of the run, then improves slowly with
seed-dependent timing, and it is still far from done at step 4,000 in the weakest seed. The feature removes the long flat start.
Of the within part of plain XOR (0.0067), about 58% is the wrong weight class distribution inside the parity group and 42% is unequal
treatment of same-weight secrets.

**What it does not show.** The learning-rate schedule is cosine over the run length, so the late fall of A overlaps with the rate decaying;
a run with the same length but a constant or longer schedule is needed to separate time from schedule. Which of H1 (weak signal) or H2
(frequency) produces the flat start is not separated. Four seeds, one noise level, one network.

## Experiments 12 to 14 (A-12, A-13, A-14): is it data reuse, the schedule, or the signal strength? (plans; predictions written before running)

All three use the A-11 code (three-level excess: A parity group, B class inside the group, C secret inside the class, each
against the exact Bayes posterior), plain inputs, the same 3x256 ReLU network, Adam 2e-3, batch 1024, 4 seeds, init seed
5000+seed, data seed 9000+seed. Per-step curves on the 100k validation traces; final numbers on the 400k test traces.

**A-12: fresh data every step (tests H7, finite-data reuse).**
*Where the idea came from.* The Deep Bootstrap paper (Nakkiran et al. 2021, read only through a tool summary) compares training on
a fixed set with training on fresh samples; and our own signs of reuse: the shared within floor of about 0.007 for XOR and addition, the
within part rising in F2, and the worsening at 64,000 steps (about 62 passes over the 1M traces).
*Design.* XOR and addition, sigma 0.5, 4,000 steps, every batch drawn fresh from the simulator (4.1M distinct traces, no repeats),
schedule unchanged.
*Predictions.* If reuse matters, the within part (B + C) falls clearly below the A-11 values (XOR 0.0067, addition 0.0083), and the
total gets close to the sum of A and a small remainder. The parity part A for XOR stays close to 0.0180 (the optimization delay should not
depend on the data being fresh). With 4.1 passes over 1M traces the effect of reuse should be modest, so I expect the within part to fall
by at most about 30%. If the within part does not change at all, reuse is not the cause of the floor; then it is an optimization floor
that more steps or a different optimizer would address.

**A-13: schedule length (separates "needs time" from "needs learning rate").**
*Where the idea came from.* In A-11 the fall of the parity part after the flat start coincides with the cosine schedule decaying, and the
schedule is stretched to the run length, so time and learning rate are tied together. Experiment 5 showed XOR catches up by 16,000
steps, but with a different schedule.
*Design.* Cosine schedule over 16,000 steps, same code, XOR and addition, curves from the same run; compared with the 4,000-step
schedule (A-11) at the same step numbers. Evaluation every 200 steps.
*Predictions.* (a) The flat start of the XOR parity part lasts about as long as in A-11 (parity part above 90% of its start at step 500,
above 80% at step 1,000), because the schedule has barely changed by then. (b) At step 4,000 the parity part is lower than the A-11 value
at step 4,000 (mean 0.0180) because the learning rate is still high. (c) By step 16,000 it is below 0.002, matching A-09's 16,000-step runs.
If (a) fails the flat start depends on the schedule, so it is not purely a time effect.

**A-14: noise level (tests H1, weak signal).**
*Where the idea came from.* The parity-group information shrinks very fast with noise (0.7128, 0.2815, 0.0364, 0.0073, 0.0006 bits at sigma
0.25, 0.35, 0.5, 0.6, 0.75; computed from the exact table). If the flat start comes from a weak parity signal, it should be much shorter
at low noise, and at high noise there is almost nothing to learn.
*Design.* XOR and addition, plain inputs, sigma 0.25, 0.35 and 0.75, 4,000 steps, 4 seeds.
*Predictions.* The parity-part flat start (steps until A falls below 90% of its start) is shorter at sigma 0.25 and 0.35 than at 0.5,
and the fraction of the parity information captured at 4,000 steps is higher: above 90% at sigma 0.25. At sigma 0.75 the parity part is
tiny in absolute terms (start about 0.0006), so the XOR-minus-addition gap should be about the same as the within part. If the flat start does
not shorten at low noise, weak signal (H1) is not what causes it, and the frequency or plateau ideas gain weight.

### A-12 results (done: 8 runs, 4 seeds per cell; A-13 and A-14 still running when this was written)

Fresh batch from the simulator at every step (4.1M distinct traces), everything else as in A-11. Test excess, bits, mean of 4 seeds
(fixed 1M-trace set from A-11 in brackets):

| | A parity group | B class in group | C secret in class | total |
|---|---|---|---|---|
| XOR plain | 0.0157 (0.0180) | 0.0041 (0.0039) | 0.0010 (0.0028) | 0.0208 (0.0248) |
| addition plain | 0.0007 (0.0007) | 0.0018 (0.0018) | 0.0041 (0.0065) | 0.0066 (0.0090) |

Per-seed totals, XOR fresh: 0.0224, 0.0241, 0.0169, 0.0198 (fixed: 0.0339, 0.0213, 0.0230, 0.0209). Addition fresh: 0.0067, 0.0065, 0.0068, 0.0063.

**Against the prediction.** The within part (B + C) fell by about 24 percent for XOR (0.0067 to 0.0051) and 29 percent for addition
(0.0083 to 0.0059), inside the predicted "at most about 30 percent". Nearly all of the fall is in C: for XOR 0.0028 to 0.0010, for addition
0.0065 to 0.0041. B did not change. The parity part for XOR stayed in the same range (mean 0.0157 versus 0.0180, but seeds move in both
directions: 0.0173, 0.0190, 0.0118, 0.0146 against 0.0278, 0.0143, 0.0158, 0.0141), as predicted.

**What this shows.** Reusing the same 1M traces about four times accounts for part of the within floor, mostly the part where the network
treats same-weight secrets differently (C). So the symmetry cost found in A-11 behaves like a finite-data effect: the network fits noise that
differs between secrets of equal weight. It does not account for the class error B, nor for the parity part, so the XOR-specific delay
does not come from data reuse.

**What it does not show.** Whether reuse also explains the worsening at 64,000 steps (not tested here). With 4 seeds the change in the
parity part is within the seed spread, so I cannot say fresh data helps or hurts it.

### A-13 results (done: 8 runs, 4 seeds per cell; data a12/A-13_curves.csv, A-13_final.csv)

Parity part A of XOR (validation excess in bits, mean of 4 seeds), cosine schedule over 16,000 steps against the A-11 schedule over 4,000 steps (start 0.0354 in both):

| step | A-11 (4k schedule) | A-13 (16k schedule) |
|---|---|---|
| 1,000 | 0.0332 (94% of start) | 0.0345 (97%) |
| 2,000 | 0.0244 (69%) | 0.0247 (70%) |
| 4,000 | 0.0182 (51%) | 0.0109 (31%) |
| 8,000 | n/a | 0.0057 (16%) |
| 16,000 | n/a | 0.0014 (4%) |

Final test totals at 16,000 steps (mean of 4 seeds, std): XOR 0.0079 (0.0006), addition 0.0086 (0.0003); XOR parity part 0.0014, addition 0.0005.

**Against the predictions.** (a) Flat start similar to A-11: held (97% of start at step 1,000; I did not check step 500 separately). (b) Parity part at step 4,000 below the A-11 value: held (0.0109 against 0.0182). (c) Below 0.002 by 16,000: held (0.0014).
**What this shows.** The fall of the parity part up to step 2,000 is the same in both runs (0.0247 against 0.0244) although the learning rate at that step is very different (nearly full in the 16k schedule, about half in the 4k schedule). Stretching the schedule only helps after step 2,000. So the early part of the delay does not look like it is controlled by the learning rate; it tracks the number of steps. With a longer run XOR's total (0.0079) ends at the same level as addition's (0.0086): XOR catches up completely at this noise level.
**What it does not show.** Why the flat start lasts about 1,000 steps. The test compares two schedules at one noise level; other optimizers and learning-rate values are untested.

### A-14 results (done: 24 runs; data a12/A-14_curves.csv, A-14_final.csv)

XOR only; "flat start" = first evaluated step where the parity part falls below 90% of its start value; "captured" = 1 minus the parity part at 4,000 steps (test) divided by its start value. Mean of 4 seeds.

| sigma | parity information at start (bits) | flat start (steps, per seed) | captured at 4,000 steps | parity part left (bits) | XOR total | addition total |
|---|---|---|---|---|---|---|
| 0.25 | 0.710 | 300, 400, 200, 300 | 0.85 (0.80 to 0.89) | 0.1075 | 0.1212 | 0.0403 |
| 0.35 | 0.279 | 600, 500, 700, 500 | 0.92 (0.91 to 0.95) | 0.0216 | 0.0329 | 0.0248 |
| 0.5 (A-11) | 0.035 | about 1,100 | 0.49 | 0.0182 | 0.0248 | 0.0090 |
| 0.75 | 0.0005 | 25 or 2,200 (noise) | not meaningful | 0.0004 | 0.0034 | 0.0046 |

**Against the predictions.** Flat start shorter at 0.25 and 0.35 than at 0.5: held (200 to 700 steps against about 1,100). More than 90% captured at sigma 0.25: MISSED (0.85; the seed range 0.80 to 0.89 does not reach 0.9). At sigma 0.75 the XOR-minus-addition gap about equal to the within part: not supported as written; XOR total is below addition's (0.0034 against 0.0046), the parity part is 0.0004 and the within part is what is left, so the experiment says only that nothing XOR-specific is visible at that noise.
**What this shows.** The flat start is shorter when the parity signal is stronger, which is what a weak-signal cause (H1) predicts, and it does not need any other cause to explain this much. But strong signal does not remove the delay: at sigma 0.25 the XOR parity part left after 4,000 steps (0.1075 bits) is 17 times addition's total parity part (0.0062), and the XOR total is three times the addition total. I did not test whether more steps would close that gap at sigma 0.25.
**What it does not show.** That weak signal is the only cause: the captured fraction at 0.25 (0.85) is lower than at 0.35 (0.92), so signal strength alone does not order the outcomes. Four seeds per cell.

## Experiment 15 (A-15): is the flat start set by the learning rate or by the number of steps? (plan; predictions written before running, 2026-10-24)
*Where the idea came from.* A-13 showed the XOR parity part falls identically up to step 2,000 under a 4,000-step and a 16,000-step cosine schedule, although the learning rate differs a lot there. That suggests the early fall tracks steps, not learning rate, but both runs used the same peak learning rate (2e-3), so the test is indirect.
*Design.* XOR only, sigma 0.5, plain inputs, same 3x256 ReLU network, Adam, constant learning rate (no decay) of 5e-4, 1e-3, 2e-3 and 4e-3, 4,000 steps, 4 seeds. Measure the first evaluated step where the parity part A falls below 90% of its start value, and the parity part at step 4,000.
*Predictions.* (1) If the flat start is set by the number of steps, the step at which A first falls below 90% at 1e-3 and at 4e-3 is within a factor 1.5 of the 2e-3 value (more than 0.67 times and less than 1.5 times). (2) At 5e-4 the fall comes later than at 2e-3 but by less than a factor 2. (3) The parity part at step 4,000 is lowest for the largest learning rate and highest for the smallest. If instead the step of the 90% crossing scales roughly as 1/lr (factor 2 earlier at 4e-3, factor 2 later at 1e-3), the flat start is limited by how far the weights must move, and a learning-rate warm schedule or larger steps would shorten it. I have low confidence in (1) and (2); (3) is a guess from ordinary optimizer behavior.

### A-15 results (done: 16 runs, 4 seeds per learning rate; data a15/A-15-lr*_curves.csv, _final.csv)

XOR, sigma 0.5, Adam with constant learning rate, 4,000 steps. Flat start = first evaluated step where the parity part A is below 90% of its start; mean of 4 seeds, per-seed values in brackets.

| lr | flat start (steps) | step where A is below 50% of start | A at 4,000 (test, bits) | total at 4,000 |
|---|---|---|---|---|
| 5e-4 | 1,925 (2600, 1400, 2100, 1600) | reached in 1 of 4 seeds (3,900) | 0.0193 | 0.0301 |
| 1e-3 | 1,700 (1600, 2200, 1900, 1100) | 3,200 / 3,700 / 3,500 / 2,700 | 0.0129 | 0.0236 |
| 2e-3 | 1,125 (900, 1200, 1300, 1100) | 2,100 / 2,500 / 2,200 / 2,300 | 0.0105 | 0.0218 |
| 4e-3 | 1,250 (1600, 1400, 1000, 1000) | reached in 2 of 4 seeds | 0.0178 | 0.0298 |

**Against the predictions.** (1) Flat start at 1e-3 and 4e-3 within a factor 1.5 of the 2e-3 value: half held. 4e-3 gives 1.11 times; 1e-3 gives 1.51 times, just outside the stated bound, so I count it as missed. (2) 5e-4 later than 2e-3 but by less than a factor 2: held (1.71 times). (3) A at 4,000 lowest for the largest learning rate and highest for the smallest: MISSED; it is lowest at 2e-3 and highest at 5e-4, with 4e-3 close to 5e-4.
**What this shows.** The flat start is not set simply by the learning rate (an 8-fold range of learning rate, 5e-4 to 4e-3, moves it by a factor 1.7 and not in one direction) and not simply by the number of steps either (it is longer at small learning rates). Above 2e-3 it does not get shorter, which looks like a floor of about 1,000 steps at this width and noise level; below it, learning is slower. Separate result: at constant 2e-3 the parity part at 4,000 steps (0.0105) is below the cosine-schedule value from A-11 (0.0182) and the total is lower too (0.0218 against 0.0248), so decaying the learning rate within 4,000 steps costs parity learning; this is consistent with A-13.
**What it does not show.** Why there is a floor of about 1,000 steps (the candidates, plateau escape or the signal-to-noise of the gradient, are untested), and whether the floor depends on width, depth or batch size. 4 seeds per cell, and the seeds overlap between neighbouring learning rates (for example 1e-3 and 2e-3 flat starts: 1,100 to 2,200 against 900 to 1,300).

## Experiment 16 (A-16): what sets the floor of about 1,000 steps? (plan; predictions written before running, 2026-10-24)
*Where the idea came from.* A-15: above a learning rate of 2e-3 the flat start does not shorten (1,125 steps at 2e-3, 1,250 at 4e-3), below it it lengthens. A floor that does not depend on learning rate suggests something other than step size: either gradient noise (the parity signal is small compared with the noise in a batch gradient, so more data per step should help), or the size of the network (how fast useful features form).
*Design.* XOR, sigma 0.5, plain inputs, Adam with constant 2e-3, 4,000 steps, 4 seeds. Four configurations against A-15's 2e-3 run (batch 1,024, width 256): batch 256, batch 4,096, width 64, width 1,024. Same measure as A-15: first evaluated step where the parity part is below 90% of its start.
*Predictions.* (1) If the floor is not limited by gradient noise, the flat start at batch 256 and at batch 4,096 stays within 0.77 to 1.3 times 1,125 steps. If noise limits it, 4,096 gives fewer than 900 steps and 256 more than 1,500. (2) Width: wider is faster, so width 1,024 gives fewer than 1,000 steps and width 64 more than 1,500. I have low confidence in both (1) and (2) and no source for them.

## Experiment 17 (A-17): soft labels from the exact posterior (plan; predictions written before running, 2026-10-24)
*Where the idea came from.* A-16 (first runs): batch 256 barely learns the parity part in 4,000 steps (0.031 bits left) and batch 4,096 learns almost all of it (about 0.003), which points to gradient noise: the parity signal is small compared with the noise of a one-hot secret label. The cleanest test of that is to remove the label noise: train the network on the exact Bayes posterior over secrets (a soft label) instead of the one-hot secret. This is ordinary knowledge distillation with the exact posterior as teacher; I have no source for its outcome here.
*Design.* XOR, sigma 0.5, plain inputs, same 3x256 network, Adam, constant 2e-3 (the A-15 configuration), batch 1,024, the same 1M fixed training traces and seeds 0 to 3, 4,000 steps. Loss: cross entropy against the exact posterior of each training trace (computed from the same Hamming-weight table used for the ceiling). The comparison run with one-hot labels is A-15's lr 2e-3 run (parity part 0.0105 at 4,000 steps; flat start 1,125 steps).
*Predictions.* (1) The flat start disappears: the parity part falls below 90% of its start within 300 steps in all 4 seeds. (2) At 4,000 steps the parity part on the test set is below 0.003 bits (mean over 4 seeds). (3) The within-class part C also shrinks, since the soft posterior is uniform inside a class and removes the label noise that makes the network treat same-weight secrets differently: C below 0.003 (A-15 hard-label C was about 0.007). If the flat start remains with soft labels, the delay is not label noise and the batch-size effect needs another explanation (for instance the effective step size).

### A-16 results (done: batch 256, batch 4,096 and width 64 with 4 seeds each; width 1,024 first had only 2 seeds because those runs take 40 minutes each and the job hit its 2-hour limit; seeds 2 and 3 were run later in a separate job; data a16/A-16-*_curves.csv, _final.csv)

XOR, sigma 0.5, constant Adam 2e-3, 4,000 steps; the reference is A-15's lr 2e-3 run (batch 1,024, width 256: flat start 1,125 steps, parity part 0.0105 at 4,000). Flat start = first evaluated step where the parity part is below 90% of its value at step 25 (the width-1,024 network starts above that value at step 1, so step 25 is used for it).

| configuration | flat start (steps, per seed) | parity part at 4,000 (test) | total at 4,000 |
|---|---|---|---|
| batch 256 | 3,300 / 3,500 / 3,400 / not reached | 0.0310 | 0.0477 |
| batch 1,024 (A-15) | 1,125 | 0.0105 | 0.0218 |
| batch 4,096 | 400 / 800 / 400 / 300 | 0.0029 | 0.0136 |
| width 64 | 1,300 / 2,500 / 1,100 / 2,200 | 0.0192 | 0.0303 |
| width 256 (A-15) | 1,125 | 0.0105 | 0.0218 |
| width 1,024 | 1,700 / 1,300 / 1,200 / 1,200 | 0.0151 (0.0121, 0.0095, 0.0168, 0.0223) | 0.0279 |

**Against the predictions.** (1) "If the floor is not limited by gradient noise, batch 256 and 4,096 both stay within 0.77 to 1.3 times 1,125 steps; if noise limits it, 4,096 gives fewer than 900 steps and 256 more than 1,500": the first branch is MISSED, the second held (475 and 3,400). (2) "Width 1,024 gives fewer than 1,000 steps and width 64 more than 1,500": MISSED for width 1,024 (1,200 to 1,700, not faster than width 256), held for width 64 (1,775).
**What this shows.** The flat start depends strongly on batch size (0.4 times at 4x the batch, 3 times at one quarter of it), which fits gradient noise as the limiting factor; it depends weakly on width: narrow networks are somewhat slower and a 4x wider one is not faster. Together with A-17 (no label noise, no delay) this makes label or gradient noise the leading explanation.
**What it does not show.** Width 1,024 gives a larger seed spread (parity part 0.0095 to 0.0223 at 4,000 steps); batch size and learning rate were varied one at a time, so the combined "noise scale" relationship (for example learning rate over batch size) is untested.

### A-18 results (XOR, sigma 0.5, constant Adam 2e-3, batch 1,024, 4,000 steps; target = (1-lambda) x exact posterior + lambda x one-hot label; all 6 runs finished (lambda 0.3 seed 1 was rerun in a separate job after the first job hit its time limit); data a18/A-18_*.csv)

| lambda | flat start (steps, seeds 0 / 1) | parity part at 4,000 | total at 4,000 |
|---|---|---|---|
| 0 (A-17 soft labels) | about 100 | 0.00024 | - |
| 0.03 | 100 / 100 | 0.00020 / 0.00021 | 0.00074 / 0.00091 |
| 0.1 | 200 / 100 | 0.00074 / 0.00073 | 0.0020 / 0.0023 |
| 0.3 | 200 / 300 | 0.00155 / 0.00199 | 0.0046 / 0.0053 |
| 1 (one-hot, A-15) | 1,125 | 0.0105 | 0.0218 |

Flat start is the first evaluated step (every 100 steps) where the parity part is below 90% of its step-25 value (0.035).
**Against the predictions.** Flat start below 300, 600 and 1,000 steps at lambda 0.03, 0.1, 0.3: all held (100, 100 to 200, 200). Parity part below 0.003 at lambda 0.03 and 0.1: held.
**What this shows.** The parity part and the delay grow smoothly with the share of one-hot label in the target; even lambda 0.3 stays far from the one-hot result (0.0016 vs 0.0105). This supports label noise as a driver of the delay (with A-16: batch noise too).
**What it does not show.** Only 2 seeds per lambda; the shape between lambda 0.3 and 1 is untested; the evaluation grid of 100 steps limits the resolution of the flat start.

### A-17 results (done: 4 seeds; data a17/A-17_curves.csv, A-17_final.csv)

Training on the exact Bayes posterior (soft labels) instead of the one-hot secret; XOR, sigma 0.5, constant Adam 2e-3, batch 1,024, the same 1M fixed traces, 4,000 steps. Test excess in bits, mean of 4 seeds, with the one-hot run of A-15 (same settings) for comparison:

| | A parity group | B class in group | C secret in class | total |
|---|---|---|---|---|
| soft labels (A-17) | 0.00024 | 0.00025 | 0.00001 | 0.00049 |
| one-hot labels (A-15, lr 2e-3) | 0.0105 | 0.0044 | 0.0069 | 0.0218 |

Per-seed totals with soft labels: 0.0003, 0.0011, 0.0001, 0.0005. Parity part falls below 90 percent of its start at step 100 in all 4 seeds (one-hot: about 1,125 steps) and to 61 to 67 percent of its start by step 100, to 25 to 32 percent by step 300.
**Against the predictions.** (1) Flat start gone, parity part below 90 percent within 300 steps in all 4 seeds: HELD (at the first evaluation after step 50, which is step 100). (2) Parity part below 0.003 at 4,000 steps: HELD (0.00024). (3) Within-class part C below 0.003: HELD (0.00001).
**What this shows.** With the label noise removed, the XOR parity delay and the within-class floor essentially vanish: the total excess is about 45 times smaller (0.0005 against 0.0218 bits). Together with A-16 (batch size 4,096 leaves a parity part of 0.0029, while batch size 256 leaves 0.031, about 3 times the 0.0105 at batch 1,024) this supports the explanation that the delay comes from the noise of one-hot secret labels in the gradient: the parity signal is small and the one-hot label drowns it, so the network needs many steps (or a big batch) to average the noise away.
**What it does not show.** That label noise is the only cause (other settings were not varied here, for example width: A-16's width runs are still running), or anything about practical training: the exact posterior is the answer key and is not available in real attacks. Soft labels remove the delay in this setting; whether a partially noisy teacher gives a partial effect is the question of A-18. Only sigma 0.5 and XOR were run with soft labels.

## Experiment 18 (A-18): dose-response in label noise (plan; predictions written before running, 2026-10-24)
*Where the idea came from.* A-17 removes label noise entirely (soft labels) and the delay disappears; A-15 has full label noise and a flat start of about 1,125 steps. A dose-response would make the claim "label noise causes the delay" much stronger than two endpoints.
*Design.* Same setting as A-17. Target = lambda times the one-hot secret + (1 - lambda) times the exact posterior, for lambda = 0.03, 0.1 and 0.3; 2 seeds (0 and 1) per lambda; 4,000 steps. (lambda = 0 is A-17, lambda = 1 is A-15's 2e-3 run.)
*Predictions.* (1) The flat start (first evaluated step where the parity part is below 90 percent of its start) increases with lambda: below 300 steps at 0.03, below 600 at 0.1, below 1,000 at 0.3 (A-17: 100; A-15: 1,125), in both seeds. (2) Parity part at 4,000 steps stays below 0.003 for lambda 0.03 and 0.1. If the flat start is the same at all lambda below 1 and jumps only at 1, the effect is not a smooth noise effect and I will say so.

## Candidate explanations for the parity delay (thinking only; nothing below has been run)

Written after Experiment 9 and before Experiment 10 finishes, so that the list exists before the results.
H marks a hypothesis, not a finding.

Facts these build on (measured): the parity group of the secret is predictable from the readings only weakly
at sigma 0.5. We computed the exact information in the parity group, 1 minus its ideal cost: 0.7128 bits at
sigma 0.25, 0.2815 at 0.35, 0.0364 at 0.5, 0.0073 at 0.6, 0.0006 at 0.75. A network that ignored parity
would have a parity-part excess of exactly 0.0364 at sigma 0.5; the XOR networks at 4,000 steps have
0.0141 to 0.0278 (mean 0.0180), so they have captured about half of it (24 to 61 percent by seed), and at
16,000 steps about 96 percent. In the noiseless case the parity clue is learned by all 20 seeds quickly (A-08).
Derivation (ours, not yet checked beyond the numbers above): the parity ripple's amplitude in the readings
is about exp(-pi^2 sigma^2) (0.085 at sigma 0.5), because Gaussian noise attenuates a period-2 oscillation.

| H | Idea | Where it comes from | What it predicts |
|---|---|---|---|
| H1 | The parity signal is weak, so gradient training needs many steps to find it (signal strength, not frequency) | Our numbers above; Barak et al. 2022 and Abbe et al. 2023 for plateaus with weak signals; the Scoop abstract (gradients shrink and a saddle appears as masking order grows) and, as an unverified lead, Masure et al. 2023 (plateau lengthens with masking order) | Changing sigma at fixed frequency changes the delay; parity learned late at sigma 0.5, early at 0.35, nothing to learn at 0.75 |
| H2 | The parity ripple is a high-frequency function and networks learn low frequencies first | Rahaman et al. 2019, Xu et al. 2018, Tancik et al. 2020 | Rescaling the inputs by a factor c changes the frequency in network units without changing signal strength: larger c should help, smaller c hurt. Wrong-frequency extra inputs (Experiment 10) should not help |
| H3 | The ripple needs depth because it oscillates | Telgarsky 2016; our A-07 (depth helped, width did not) | A first layer with a periodic activation (Ziyin et al. 2020) should remove the gap without hand-built features |
| H4 | Plateau then abrupt escape: parity is found at a seed-dependent step | Barak et al. 2022, Abbe et al. 2023; our seed spread (XOR sd 0.0062 vs 0.0005; with the parity feature sd 0.0002) | Parity part per step stays flat then drops sharply, at different steps in different seeds. H2 would predict a smooth gradual decrease |
| H5 | The parity signal is diluted in the 256-way loss (0.036 of 8 bits) | Our numbers; hierarchical labels help in Marquet and Oswald 2023 (masked AES) | An auxiliary parity-group loss removes the gap even without an extra input; if not, representing the ripple is the hard part |
| H6 | Gradient noise: batch 1024 is too small to see a weak signal | Barak et al. 2022 (small batches less robust) | Larger batch at equal steps closes the gap |
| H7 | Finite-data effects: the same 1,000,000 traces are reused (4 passes at 4,000 steps, 65 at 64,000) | Deep Bootstrap (Nakkiran et al. 2020); our A-06 (excess falls about as 1 over N) | Fresh data every step (ideal world): the 4,000-step parity gap persists (it is optimization speed); the within-group floor (about 0.007) shrinks for both maskings; 64,000 steps is no longer worse than 16,000; the F2 within-group rise disappears |
| H8 | XOR's within-group cost is lower because its posterior depends on the secret only through Hamming weight (verified: 0 of 247 same-weight pairs differ), so there are fewer distinct values to learn | Hiltenbrand et al. 2025 Proposition 1, checked numerically | Same-weight secrets get nearly identical logits in trained XOR networks (checkable on saved networks, no training); the XOR and addition within-group floors move together under fresh data |

Distinguishing H1 from H2: the sigma sweep changes strength at fixed frequency; the input rescaling changes
frequency at fixed strength. Distinguishing H2 from H4: shape of the per-step curve. H7 is the most
likely explanation of the within-group floor, the F2 within-group rise and the 64,000-step result, and it is
cheap because the data are simulated.

Not caught by any of these: a change of masking scheme, other leakage models, real traces.

## Reproduction check

Rerunning saved runs on the same machine and software reproduces them exactly: Experiment 2
(XOR, sigma 0.5, 50,000 traces, seed 0) perceived information 0.051009 saved vs 0.051009 rerun;
Experiment 3 (XOR, 16,000 steps, seeds 0 and 1) excess cross entropy 0.008633 and 0.007404, saved
vs rerun, difference zero. This also shows the two scripts that produced Experiment 3 (a03_budget.py
and overnight_queue.py) agree. Not tested: other machines or library versions.

## Status

Done: Experiments 1 to 18 and the reproduction check (A-16 and A-18 complete).
Not done: full text reading of most papers in literature.md, separating the causes of the parity delay (Experiment 10 only narrows it), other leakage models,
a trace model where the leakage positions are hidden, other optimizers on sampled data, GitHub write up.

## What we know, what we do not know

Know (in this simulated model): XOR has the higher information ceiling at every noise level tested.
At short training and low noise XOR networks end up further from their ceiling than addition
networks, but this gap closes with 4x more training steps, with a deeper network, or with more
training data. At 16,000 steps it is reversed at the lowest noise (sigma 0.25). No learning rate
from 0.0005 to 0.008 closed it at 4,000 steps.
Also know (Experiment 9): most of the XOR shortfall at 4,000 steps is in getting the parity group right,
and giving the network a parity-like input removes about 90 percent of it.
Do not know: why parity is slow to learn, whether the parity clue is the only cause, whether this holds for other leakage models,
architectures or optimizers, whether it holds on real measurements, and how it relates to published
work (abstracts only read).
Added by Experiments 12 to 18 (all in this simulated model, XOR, sigma 0.5 unless stated): with a 16,000-step schedule XOR
catches up with addition (A-13); a stronger parity signal (smaller sigma) shortens the flat start (A-14); at constant learning
rate the flat start has a floor of about 1,100 steps (A-15); it depends strongly on batch size (3,400 / 1,125 / 475 steps at
batch 256 / 1,024 / 4,096) and weakly on width (A-16); removing label noise with soft labels from the exact posterior removes
the delay (A-17); and the delay grows smoothly with the share of one-hot label in the target (A-18). The leading explanation
is therefore noise from one-hot labels in the gradient hiding a weak parity signal. Not tested: a combined noise-scale rule
(learning rate over batch size), other leakage models, other optimizers, real traces; the 64,000-step worsening is explained for addition by A-20 (data reuse, 2 seeds per condition; XOR not rerun).

## Experiment 19 (A-19): is the flat start a function of a noise scale built from learning rate and batch size? (plan; predictions written before running, 2026-10-25)
*Where the idea came from.* A-15 (learning rate alone) and A-16 (batch size alone) each moved the flat start, and A-17/A-18 point to label noise. In the optimization literature the effective noise of SGD is often summarized as learning rate divided by batch size (a lead I have not opened); for Adam a square-root rule is sometimes used instead (also a lead). I do not know which, if either, applies here, so I test both.
*Design.* XOR, sigma 0.5, constant Adam, 4,000 steps, width 256, 4 seeds. Baseline from A-15/A-16: lr 2e-3, batch 1,024, flat start 1,125. Ratio family (same lr / batch): (lr 1e-3, batch 512) and (4e-3, 2,048). Square-root family (same lr / sqrt(batch)): (1e-3, 256) and (4e-3, 4,096). Flat start = first evaluated step where the parity part is below 90% of its step-25 value; judged by the median over seeds.
*Predictions.* A family "holds" if both of its configurations have median flat start within 0.77 to 1.3 times 1,125. (1) The ratio family does not hold: at least one configuration is outside the band (existing data: lr alone is non-monotonic, batch alone is strong). (2) The square-root family does not hold either. (3) Therefore a single noise-scale variable does not fully describe the flat start. If a family holds, I will say that the delay is consistent with that noise scale in this model; one family holding would be a result I did not expect.

## Experiment 20 (A-20): is the worsening at 64,000 steps a data-reuse effect? (plan; predictions written before running, 2026-10-25)
*Where the idea came from.* A-03: both maskings are further from the ceiling at 64,000 steps than at 16,000 (addition 0.0160, XOR 0.0143 vs 0.0083 and 0.0086), and the final checkpoint is worse than the best validation checkpoint, which fits overfitting to the 1,000,000 fixed traces (about 65 passes). A-12 showed fresh data lowers the within-class floor at 4,000 steps. Deep Bootstrap (Nakkiran et al. 2020, lead) frames this as a gap between a finite-data and an ideal world.
*Design.* Addition, sigma 0.5, cosine schedule over 64,000 steps, batch 1024, width 256 (same code path as A-12, `a12_runs.one`), seeds 0 and 1. Condition F: fixed 1M traces. Condition R: fresh batch from the simulator at every step. Both re-run here so the code path is identical. Metric: test excess cross-entropy in bits at 64,000 steps (A-03 value for fixed: 0.0160, 3 seeds).
*Predictions.* (1) The fixed condition reproduces the A-03 level: both seeds between 0.012 and 0.020. (2) The fresh condition does not worsen: both seeds at or below 0.010 (the 16,000-step fixed level is 0.0083). (3) The fixed run's final test excess is above its own best validation checkpoint's value. If (2) fails, reuse does not explain the worsening and the cause is optimization (for example the long constant-ish learning rate stretch), and I will say that.
Note: 64,000-step runs are long; they are launched only when the CPUs are free so that the 2-hour task limit is not hit.

### A-19 results (XOR, sigma 0.5, constant Adam, 4,000 steps, width 256; 14 of 16 runs: (4e-3, batch 4,096) has seeds 0 and 1 only, the job hit its time limit; data a19/, a19b/)
Flat start (first evaluated step where the parity part is below 90% of its step-25 value), per seed; median in brackets. Baseline (2e-3, batch 1,024) is 1,125 steps; the 0.77 to 1.3 band is 866 to 1,463.

| family | lr, batch | flat start per seed | median | parity part at 4,000 (mean) |
|---|---|---|---|---|
| ratio lr / B | 1e-3, 512 | 2,500 / 3,400 / 2,400 / 3,400 | 2,950 | 0.0239 |
| ratio lr / B | 4e-3, 2,048 | 800 / 900 / 1,300 / 600 | 850 | 0.0075 |
| sqrt lr / sqrt(B) | 1e-3, 256 | not reached / 3,800 / 3,300 / 3,500 | 3,500 | 0.0292 |
| sqrt lr / sqrt(B) | 4e-3, 4,096 (2 seeds) | 600 / 700 | 650 | 0.0037 |
Earlier points for comparison: lr 2e-3 with batch 256 / 1,024 / 4,096: 3,400 / 1,125 / 475 (A-16); batch 1,024 with lr 5e-4 / 1e-3 / 4e-3: 1,925 / 1,700 / 1,250 (A-15).
**Against the predictions.** (1) Ratio family does not hold: HELD (both configurations outside the band: 2.6 and 0.76 times). (2) Square-root family does not hold: HELD (3.1 and 0.58 times). (3) A single noise-scale variable does not fully describe the flat start: HELD.
**What this shows.** Smaller learning rate with smaller batch is much slower, and larger learning rate with larger batch is faster, rather than the product staying fixed: in both families the flat start moves in the direction of the batch change but by more than a pure noise-scale rule would allow. At every learning rate tried, a larger batch gives a shorter flat start (lr 4e-3: 1,250 / 850 / 650 for batch 1,024 / 2,048 / 4,096; lr 2e-3: 3,400 / 1,125 / 475 for batch 256 / 1,024 / 4,096). Learning rate alone is not monotonic (A-15). The label-noise explanation (A-17, A-18) stands; the specific "lr over batch" form is not supported.
**What it does not show.** The functional form (I did not fit one), the effect of width and depth at these settings, other optimizers. 4 seeds per cell (2 for one cell).

# Phase 2 for this project (protocol: see carry_chain/PROGRESS.md, written 2026-10-25)

## R-6 (CONFIRMATORY, existing data): does a simple formula for the flat start predict held-out configurations?
Question B-Q1: the delay is not a function of lr/B or lr/sqrt(B) alone (A-19). Is there any simple formula that predicts it out of sample?
Model (fixed now): log T = b0 + b1 log(lr) + b2 (log lr)^2 + b3 log(B), where T = median flat start (first evaluated step where the parity part is below 90% of its step-25 value, 4,000-step runs, width 256, XOR, sigma 0.5), fitted by least squares on the A-15 cells (batch 1,024; lr 5e-4, 1e-3, 2e-3, 4e-3) and A-16 cells (lr 2e-3; batch 256 and 4,096), 6 cells. The squared term allows the non-monotonic effect of lr seen in A-15. Held-out cells: the four A-19 cells (lr, batch): (1e-3, 512), (4e-3, 2,048), (1e-3, 256), (4e-3, 4,096).
Pass rule: predicted T within a factor 1.3 of the observed median in at least 3 of 4 held-out cells. Caveat fixed now: 6 cells for 4 parameters leaves 2 degrees of freedom; a pass is suggestive only. Prediction (mine): fail on at least one cell (lr x batch interaction, since the data above are not additive in the logs).

### R-6 result (existing data; the formula was fitted before looking at the A-19 cells)
Fitted on 6 cells (flat start medians, step-25 plateau criterion): log T = 17.227 + 1.660 log(lr) + 0.144 (log lr)^2 - 0.772 log(B). The batch exponent -0.77 means T falls roughly as B^-0.77 at fixed lr.
| held-out cell (lr, batch) | observed median | predicted | ratio |
|---|---|---|---|
| 1e-3, 512 | 2,950 | 2,534 | 0.86 |
| 4e-3, 2,048 | 850 | 721 | 0.85 |
| 1e-3, 256 | 3,500 | 4,327 | 1.24 |
| 4e-3, 4,096 | 650 | 422 | 0.65 |
**Against the rule.** Within a factor 1.3 in 3 of 4 cells: pass rule MET. My prediction "fails on at least one cell" also HELD (the 4e-3, batch 4,096 cell is off by 35 percent and has only 2 seeds). **What this shows.** A smooth formula with a strong batch term (about B^-0.77) and a weak, non-monotonic lr term predicts held-out configurations to within about 25 percent in 3 of 4 cases. **Limits.** The model has 4 parameters fitted to 6 cells (2 degrees of freedom); the quadratic term was added after seeing the non-monotonic lr effect in A-15, so this is not a preregistered form; one held-out cell rests on 2 seeds. It is a description, not a mechanism.

## R-7 (CONFIRMATORY, prospective): predict new configurations before running them (written 2026-10-25, coefficients fixed above; not refitted)
Cells (lr, batch) and the predicted median flat start: (2e-3, 512): 2,114; (1e-3, 2,048): 850; (3e-3, 1,024): 1,203; (5e-4, 4,096): 670. XOR, sigma 0.5, width 256, constant Adam, 4,000 steps, 4 seeds per cell, same criterion.
Pass rule: observed median within a factor 1.3 of the prediction in at least 3 of 4 cells. If it fails, the formula is rejected as a description outside the range already measured. Run with `a22_runs.py`.

Note on A-20 (2026-10-25): the fixed-data condition was stopped at the 2-hour task limit without finishing a single 64,000-step run (four jobs shared 2 CPUs), and the fresh-data job (started later, slower per step) was stopped by me for the same reason. No A-20 result exists. It will be rerun, one condition at a time, when the CPUs are free.

### R-7 result (prospective test of the flat-start formula; coefficients fixed before the runs, not refitted; 4 seeds per cell)
| cell (lr, batch) | predicted | observed median (per seed) | ratio predicted/observed |
|---|---|---|---|
| 2e-3, 512 | 2,114 | 1,850 (1,800 / 2,300 / 1,900 / 1,700) | 1.14 |
| 1e-3, 2,048 | 850 | 800 (800 / 1,300 / 800 / 800) | 1.06 |
| 3e-3, 1,024 | 1,203 | 1,450 (1,400 / 1,500 / 2,200 / 1,000) | 0.83 |
| 5e-4, 4,096 | 670 | 700 (800 / 600 / 700 / 700) | 0.96 |
**Against the rule.** Within a factor 1.3 in at least 3 of 4 cells: MET (4 of 4). **What this shows.** The 4-parameter formula, fitted only on A-15 and A-16 cells, predicted four new configurations (inside the range already measured: lr 5e-4 to 3e-3, batch 512 to 4,096) to within 17 percent. As a description of the flat start in this model it survives a prospective test. **What it does not show.** A mechanism (the bundle's toy gives a batch slope of -0.446 against the -0.77 here), or anything outside this range, other widths, depths or noise levels.

A-20 relaunch (2026-10-25): timing test on idle CPUs, 500 steps including setup and evaluations took 24 s for both the fixed and the fresh condition; extrapolated 64,000 steps is at most about 51 minutes per run (upper bound, since setup is included), under the 72-minute (60 percent of the limit) rule. One run per job, two jobs at the same time (fixed seed 0, fresh seed 0), then seed 1 of each.

A-20 second loss (2026-10-25): both relaunched runs were stopped after 90 minutes with no output, because I had set the task timeout (90 minutes) below the real run time and the script saved nothing before the end. Cause of the slowness: each job used PyTorch's default multi-threading, so two jobs on 2 CPUs oversubscribed them (load average 4). Fix: one thread per job and checkpoints every 2,000 steps (resumable). Timing with the fixed script, two jobs at once, 2,200 steps: 76 s each, so 64,000 steps is about 37 minutes per run.

### A-20 results (addition, sigma 0.5, 64,000 steps, cosine schedule over the run, batch 1,024, width 256; 2 seeds per condition; data a20f/, a20r/)
| condition | seed | test total | A (parity) | B (class) | C (secret in class) | validation: last / best (step) |
|---|---|---|---|---|---|---|
| fixed 1M traces | 0 | 0.0182 | 0.0004 | 0.0010 | 0.0168 | 0.0186 / 0.0112 (10,800) |
| fixed 1M traces | 1 | 0.0139 | 0.0004 | 0.0016 | 0.0118 | 0.0143 / 0.0110 (11,800) |
| fresh batch each step | 0 | 0.0022 | 0.0004 | 0.0005 | 0.0013 | 0.0025 / 0.0025 (60,600) |
| fresh batch each step | 1 | 0.0019 | 0.0004 | 0.0004 | 0.0011 | 0.0024 / 0.0024 (60,000) |
**Against the predictions.** (1) Fixed condition reproduces the A-03 level (both seeds between 0.012 and 0.020): HELD (0.0182, 0.0139). (2) Fresh condition does not worsen (both at or below 0.010): HELD (0.0022, 0.0019; also below the 16,000-step fixed value of 0.0083). (3) Fixed run's final validation excess above its own best: HELD (0.0186 against 0.0112; 0.0143 against 0.0110).
**What this shows.** With the same schedule, the same network and the same number of updates, drawing a fresh batch every step removes the late worsening; with the 1M fixed traces the validation error is lowest around step 11,000 to 12,000 and then rises. The excess is almost entirely in the within-class part C (0.012 to 0.017 against 0.001), the part where the network treats secrets of equal Hamming weight differently, which is what overfitting to the finite sample would produce (consistent with A-12). The fresh run's last checkpoint is also its best, which answers the selection-effect objection in the outside bundle (the best of K checkpoints always looks better than the last): the same procedure on the fresh run finds no gap. So for addition at this size, the worsening between 16,000 and 64,000 steps is a data-reuse (finite-sample) effect, not a schedule effect.
**What it does not show.** XOR (not run here; the A-03 XOR runs also worsened); other data sizes or noise levels; 2 seeds per condition (the two fixed seeds differ, 0.0182 against 0.0139); that nothing else contributes.

## Experiment 23 (A-23): replicate, in my own code, the outside claim that unequal share noise keeps the hard-label delay and soft labels remove it (plan; predictions written before running, 2026-10-25)
*Where the idea came from.* The outside evidence bundle (unverified, see literature.md) reports that with noise (0.25, 0.75) on the two shares the hard-label parity delay persists and exact-posterior soft labels remove it, and gives B3 as an answer for one alternative leakage model. I do not take it as a result until my pipeline reproduces it.
*Design.* XOR masking, Hamming-weight leakage, independent Gaussian noise sigma1 = 0.25 on the mask share and sigma2 = 0.75 on the masked share (the order of the two shares is my assumption; the bundle does not state which share has which noise, so I also run the swapped assignment (0.75, 0.25)). Exact posterior and information ceiling recomputed with per-share sigmas (counts of masks by Hamming weight pair times the two Gaussian factors). Same network as A-17 (3x256 ReLU, Adam constant 2e-3, batch 1,024, 4,000 steps), hard one-hot labels versus soft labels from the exact posterior, 4 seeds each, 2 noise assignments, so 16 runs. Metrics as before (parity part A, flat start = first evaluated step where A is below 90 percent of its step-25 value).
*Predictions.* (1) Hard labels: flat start of at least 800 steps in at least 3 of 4 seeds for each assignment, and A at 4,000 steps at least 5 times the soft-label value. (2) Soft labels: flat start at most 300 steps in all seeds. (3) The (0.25, 0.75) delay is not shorter than the equal-noise 0.5 baseline (1,125 steps) by more than a factor 1.5 (the bundle reports 1,700 against 1,100 on its metric). If (1) fails the outside claim does not replicate in my pipeline.

### A-23 results (XOR, Hamming-weight leakage, unequal share noise, 4,000 steps, constant Adam 2e-3, batch 1,024, width 256; 4 seeds per cell; data a23a/, a23b/)
| noise (mask share, masked share) | labels | flat start per seed (steps) | parity part A at 4,000 | total at 4,000 |
|---|---|---|---|---|
| (0.25, 0.75) | hard | 1,600 / 1,500 / 1,900 / 2,000 | 0.00683 | 0.0197 |
| (0.25, 0.75) | exact-posterior soft | 100 / 100 / 100 / 100 | 0.00058 | 0.0021 |
| (0.75, 0.25) | hard | 1,500 / 2,000 / 2,300 / 1,400 | 0.00761 | 0.0203 |
| (0.75, 0.25) | exact-posterior soft | 100 / 100 / 100 / 100 | 0.00066 | 0.0020 |
**Against the predictions.** (1) Hard labels: flat start at least 800 in at least 3 of 4 seeds for each assignment (4 of 4 in both) and A at 4,000 steps at least 5 times the soft value (11.8 and 11.5 times): HELD. (2) Soft labels: flat start at most 300 in all seeds (100 in all 8): HELD. (3) The hard-label delay is not shorter than the equal-noise baseline by more than 1.5 times: HELD (median about 1,750 against 1,125, i.e. longer, not shorter).
**What this shows.** In my own pipeline, with unequal independent noise on the two shares (either way round), the XOR parity delay with one-hot labels is present (1,400 to 2,300 steps) and exact-posterior soft labels remove it (100 steps). This replicates the outside bundle's B3 claim in direction and approximate size (its (0.25, 0.75) hard-label value was 1,700 on a different metric). So the label-noise explanation survives one alternative leakage model. **What it does not show.** Hamming-distance or weighted-bit leakage (not trained), other noise levels, 4 seeds per cell, and a mechanism. Grid caveat: 100 steps is the first evaluated checkpoint after step 25, so "removes the delay" means the fall had started by step 100, not that the onset is exactly zero; a finer grid was not run (the outside chat made the same caveat about its own result).

## Experiment 24 (A-24): which part of Adam carries the label noise, the update (first moment) or the denominator (second moment)? (plan; predictions written before running, 2026-10-25)
*Where the idea came from.* The outside chat reported (its data, unverified) that giving Adam a soft-label first moment with a hard-label second moment (SH) rescues the parity delay, while the reverse (HS) does not. If true, the noise that delays parity learning is in the update direction (the numerator), not in the denominator, which would also explain why the batch-size effect is strong. I replicate it in my own pipeline.
*Design.* XOR, sigma (0.5, 0.5), 3x256 ReLU, batch 1,024, 4,000 steps, constant lr 2e-3, hand-written Adam (beta1 0.9, beta2 0.999, eps 1e-8, bias correction): m is updated with gradient g_m and v with gradient g_v, where each of g_m and g_v is the gradient of the cross-entropy against either the hard one-hot label or the exact-posterior soft label. Four conditions HH, SH, HS, SS, 4 seeds each, same initialization and data stream across conditions. Metrics as before (flat start = first step where the parity part A is below 90 percent of its step-25 value, every 100 steps; A at 4,000).
*Predictions.* (1) HH reproduces the A-15 level (flat start between 800 and 1,600 in at least 3 of 4 seeds). (2) SS has flat start at most 300 in all seeds. (3) SH has flat start at most 800 in at least 3 of 4 seeds (the rescue). (4) HS has flat start at least 1,000 in at least 3 of 4 seeds (no rescue). If (3) fails the outside claim does not replicate; if (4) fails the denominator also matters. Caveat fixed now: the hybrids are not ordinary Adam and change the update noise as well as the scale, so a result is about this construction only.

### A-24 results (hand-written Adam, separate gradient sources for the two moments; XOR, sigma 0.5, 4,000 steps, constant lr 2e-3, batch 1,024; 4 seeds per mode; data a24a/, a24b/)
| mode (m source / v source) | flat start per seed (steps) | A at 4,000 (mean) | total at 4,000 (mean) |
|---|---|---|---|
| HH hard / hard | 1,100 / 1,200 / 900 / 1,700 | 0.01048 | 0.0213 |
| SH soft / hard | 400 / 300 / 1,100 / 700 | 0.00271 | 0.0056 |
| HS hard / soft | 50 / 50 / 50 / 50 (criterion artefact, see below) | 0.03555 | 0.0797 |
| SS soft / soft | 100 / 100 / 100 / 100 | 0.00039 | 0.0008 |
**Against the predictions.** (1) HH between 800 and 1,600 in at least 3 of 4 seeds: HELD (1,100, 1,200, 900; seed 3 is 1,700). (2) SS at most 300 in all seeds: HELD (100). (3) SH at most 800 in at least 3 of 4 seeds: HELD (400, 300, 700; seed 2 is 1,100); A at 4,000 steps is 0.0027 against 0.0105 for HH. (4) HS at least 1,000 in at least 3 of 4 seeds: MISSED on the literal criterion (50 steps), but the 50 is an artefact: in HS the parity part starts at 120 to 420 (nats-scale blow-up in the first steps, total excess 850 to 2,560) and falls to the no-parity plateau (0.0353) by about step 50 to 500 and stays there, never learning parity (A at 4,000 is 0.0351 to 0.0354; total 0.07 to 0.11 against 0.021 for HH). The hybrid is unstable at the start (hard-label first moment divided by a second moment built from the much smaller soft-label gradients), so the HS run is not a clean "no rescue" and I do not use it as evidence about the denominator.
**What this shows.** Taking the first moment (the update direction) from the exact-posterior soft label and only the second moment from the one-hot label (SH) recovers most of the benefit of soft labels (A 4 times lower than HH at 4,000 steps; flat start 300 to 700 in 3 of 4 seeds), which is consistent with the delay being caused by noise in the update direction rather than by the size of the denominator. This replicates the outside chat's SH result in direction. **What it does not show.** That the denominator plays no role: the HS control failed for a trivial instability reason, so it says nothing either way; SH is not ordinary Adam and also changes update noise; seed 2 of SH (1,100) is not rescued; 4 seeds; no coordinate-level gradient mean or variance was measured.

## A-25 (CONFIRMATORY, follows A-20): is the worsening with a long run also a data-reuse effect for XOR masking?
Idea source: A-20 (addition: fixed 1M traces ended at 0.0182 and 0.0139, fresh batches 0.0022 and 0.0019).
Design: identical to A-20 (sigma 0.5, cosine schedule over 64,000 steps, batch 1024, width 256, 1M fixed traces or fresh batches each step, 2 seeds per condition, checkpointed every 2,000 steps), with XOR masking in place of addition. Script a25_runs.py (a20_runs.py with KIND=xor and run-id prefix A-25), output a25f (fixed) and a25r (fresh).
Predictions. (1) fixed data final test excess total at least 3 times the fresh-batch value in both seeds (A-20 addition ratio was 6 to 9). (2) fresh-batch final total below 0.005. (3) the best validation step of the fixed runs occurs before 20,000 (overfitting), while the fresh runs' best validation step is the last. If XOR's fixed-data runs do not worsen, data reuse is not a general explanation and the A-20 result is specific to addition. Misses reported.

### A-25 result (XOR masking, sigma 0.5, 64,000 steps, 2 seeds per condition; a25_runs.py, a25f/, a25r/, logs a25f.log, a25r.log). Final test excess total (bits): fixed 1M traces 0.0131 and 0.0132; fresh batches 0.0005 and 0.0008.
Predictions. (1) fixed at least 3 times fresh in both seeds: PASS (26 times and 16 times). (2) fresh total below 0.005: PASS. (3) fixed best-validation step before 20,000 and fresh best step is the last: PARTLY MISSED. Fixed seed 0 best at 10,400 (val 0.0114, final 0.0143) but fixed seed 1 best at 46,000 (val 0.0129, final 0.0134), i.e. it plateaus rather than rising; fresh best steps 61,200 and 56,000 equal the last value at four decimals, so "last" holds only to that precision.
Reading: for XOR, as for addition (A-20), long runs on a fixed 1M-trace set end far above fresh-data runs, so the worsening comes from reusing a finite training set; in the fixed runs the leftover excess sits almost all in the secret-in-class part C (0.0114 to 0.0117) while A and B are 0.0006 to 0.0011. Caveats: 2 seeds per condition, one sigma, one schedule, one set size (1M); the A/B/C split of the leftover is described, not explained.

## A-26 (CONFIRMATORY): gradient signal and noise, hard versus soft labels, along a hard-label training run (XOR masking, sigma 0.5)
Idea source: the plan item B1, A-24 (the rescue by soft-label second moments points at gradient noise) and the outside chat's Adam escape toy (unverified). Question: is the one-hot label gradient noisier than the exact-posterior soft-label gradient while having the same mean, and how large is the difference over the early phase?
Design. Train the standard network (A.make_net2(2), Adam, batch A.BS, fresh data, hard labels, seeds 0-3). At steps 0, 100, 300, 1000, 2000 compute the gradient (all parameters) of the hard loss and of the soft loss (exact posterior) on K = 64 independent fresh minibatches of size 256. For each loss: mean gradient gbar, per-minibatch noise N256 = mean_k ||g_k - gbar||^2, signal estimate S = ||gbar||^2 - N256 / K. Report N256_hard / N256_soft, S_hard / S_soft and the SNR at batch 1024 = S / (N256 / 4). Script a26_snr.py, output a26/.
Predictions. (1) N256_hard / N256_soft is above 1.5 at every measured step in at least 3 of 4 seeds. (2) S_hard and S_soft agree within 30 percent at step 0 (same expected gradient). (3) SNR at batch 1024 is lower for hard than for soft at every measured step up to 1000 in all 4 seeds. A miss on (1) would mean the soft-label rescue of A-24 does not come from lower gradient noise.

### A-26 result (XOR masking, sigma 0.5, 4 seeds, hard-label training run; a26_snr.py, a26/A-26_snr.csv; K = 64 minibatches of 256 per measurement). Noise ratio N256_hard / N256_soft: step 0: 878, 885, 983, 922; step 100: 108, 102, 145, 98; step 300: 99, 82, 123, 116; step 1000: 147, 68, 140, 62; step 2000: 54, 182, 146, 79. Signal ratio S_hard / S_soft at step 0: 1.04, 1.10, 1.00, 1.00 (later steps 0.07 to 2.1, the signal estimate is noisy once it is small). SNR at batch 1024 (S / (N256 / 4)): hard 0.34, 0.24, 0.31, 0.29 at step 0 and 0.03 to 0.29 at step 100; soft 196 to 306 at step 0 and 6 to 24 at step 100.
Predictions. (1) noise ratio above 1.5 at every measured step in at least 3 of 4 seeds: PASS (the ratio is 54 to 983 everywhere). (2) S_hard and S_soft within 30 percent at step 0: PASS (0 to 10 percent). (3) hard SNR at batch 1024 lower than soft at every measured step up to 1000, all 4 seeds: PASS.
Reading: the one-hot-label gradient has the same mean as the exact-posterior gradient at initialisation but about 900 times more noise (per-minibatch squared deviation), which puts the batch-1024 signal-to-noise ratio of the hard-label gradient near 0.3 (noise-dominated) while that of the soft-label gradient is 200 to 300. It stays about 100 times noisier through step 2000. This is a measurement of the gradient, consistent with A-23 and A-24 (soft labels remove the flat start, soft second moments alone rescue it); it does not by itself show that this noise causes the flat start. Caveats: 4 seeds; the signal estimate S = ||gbar||^2 - N/K is itself noisy when small (some later-step values are close to zero, one is 0.07 times the soft one); the whole-parameter gradient is used, not an identified parity coordinate; the training run is hard-label Adam with constant lr and fresh data, not the A-24 hand-written variant.
Next: a prospective check that the SNR scale predicts where the batch-size dependence of the flat start changes (SNR = 1 at about batch 3,400 at step 0 if SNR scales linearly with batch size; the fitted law T proportional to B^-0.77 is from batches 128 to 1024 only).

## A-27 (CONFIRMATORY, follows A-26): does the flat-start time stop falling with batch size above the batch where the initial SNR reaches 1?
Idea source: A-26 (SNR_hard = 0.3 at batch 1024; SNR scales with B if the minibatches are independent, so SNR = 1 near B = 3,400) and R-7 (fitted law log T = 17.227 + 1.660 log(lr) + 0.144 (log lr)^2 - 0.772 log(B), from batches up to 1024).
Design. Same flat-start measurement as R-7 (first evaluated step where A < 90 percent of its step-25 value, 100-step grid; XOR masking, one-hot labels, sigma 0.5, width 256, constant lr 2e-3), batch 4096 and 8192, 4 seeds each, 3,000-step cap, using a12_runs.one with its BS global. Compare with the law's extrapolation.
Predictions. (1) the observed flat start at batch 4096 is within 25 percent of the extrapolated law (law at lr 2e-3, B=1024 is 850; the law gives 850 x 4^-0.772 = 291 steps, below the 100-step grid resolution times 3). (2) the ratio T(8192) / T(4096) is above 0.6 (the power law would give 0.59; a saturation would give a larger ratio). Because the grid is 100 steps, a flat start of 100 to 300 can only be resolved coarsely, so a pass on (2) will be reported with that caveat. If everything is at the first grid point (100), this test cannot discriminate and I will say so.

Revision of A-27 (written before any run; the draft above is dropped because its extrapolated times (about 290 steps) lie at the 100-step grid floor, so it could not discriminate).
## A-27 (CONFIRMATORY, prospective, outside the fitted range): flat-start law at low learning rate and large batch
Design. The R-7 pipeline (a22_runs.py style: XOR, sigma 0.5, width 256, constant Adam, fixed 1M traces, one-hot labels, flat start = first evaluated step with A below 90 percent of its step-25 value, 4 seeds per cell), cells at lr 2.5e-4 (below the fitted range 5e-4 to 3e-3) and batch 2,048, 8,192, 16,384 (the last two above the range used in the fit). Steps: 4,000 for batch 2,048 and 3,000 for the others; if no flat start is seen by then it is reported as censored. Coefficients fixed as in R-7 (log T = 17.227 + 1.660 log(lr) + 0.144 (log lr)^2 - 0.772 log(B)); predictions 1,768 / 606 / 356 steps.
Pass rule. Observed median within a factor 1.3 of the prediction in at least 2 of the 3 cells. Direction of failure is part of the result: observed above prediction at large batch would mean the batch dependence flattens (saturation); observed below would mean it steepens. The 100-step grid limits resolution of the 356 prediction (a factor 1.3 is 274 to 463, only about 2 grid steps).
Script a27_runs.py (CONF tags), output a27/. A timing test comes first.

### A-27 result (XOR masking, sigma 0.5, lr 2.5e-4 constant Adam, fixed 1M traces, one-hot labels, 4 seeds per cell; a27/ and a27b/; flat start = first evaluated step with A below 90 percent of its step-25 value, 100-step grid). Note on execution: three jobs hit the 2-hour task limit (the batch-16,384 job at its 4th seed, the batch-2,048/8,192 job during batch 8,192), because two jobs on 2 CPUs ran about twice slower than my timing test (about 30 minutes per seed at batch 16,384, not 15); the missing seeds were rerun one by one, identical code and seeds, so no result depends on a partial run.
| batch | predicted (law, fixed before the runs) | observed per seed | median | observed / predicted |
|---|---|---|---|---|
| 2,048 | 1,770 | 2100, 2000, 1700, 1600 | 1,850 | 1.05 |
| 8,192 | 607 | 1000, 1000, 800, 700 | 900 | 1.48 |
| 16,384 | 355 | 700, 600, 700, 600 | 650 | 1.83 |
Pass rule: within a factor 1.3 in at least 2 of 3 cells. FAILED (1 of 3: only batch 2,048). Direction: observed above the law at large batch, i.e. the batch dependence flattens; the fitted exponent over batches 2,048 to 16,384 is -0.51 (law: -0.77). Every seed of the two large-batch cells is above the law's value (the gap at batch 16,384, 250 to 350 steps, is three grid steps).
Reading: the formula of R-7 (fitted on batches up to 1,024 and checked up to 4,096 at 5e-4) holds at batch 2,048 at a lower learning rate (2.5e-4) but overestimates the benefit of larger batches beyond about 4,000: the speed-up with batch size saturates. A slope of about -0.5 over 2,048 to 16,384 is compatible with the outside toy's -0.446 (unverified lead) but I have not tested that model. It also fits the A-26 estimate that the hard-label gradient SNR reaches 1 near batch 3,400 (an expected change of regime), but that connection is a coincidence of scale until tested with SNR measured at the batch sizes used here. Caveats: 4 seeds per cell; one learning rate; the law is a fit, not a mechanism; the measured SNR is at lr 2e-3 and step 0 only.

## A-28 (CONFIRMATORY, link to the adder project C-66/C-72): does the initial weight scale change the length of XOR's flat start?
Idea source: in the adder project, multiplying all initial weights by a factor c shortened the long plateau of the joint net 5 times (C-66), and the bit-4 time at large batch became independent of noise (C-65), as the XOR flat start saturates above batch of about 4,000 (A-27). Question: is the XOR flat start also shortened by a larger initial scale?
Design. The A-15 configuration at constant lr 2e-3 (XOR masking, sigma 0.5, plain inputs, 3x256 ReLU, Adam, batch 1,024, 1M fixed traces, one-hot labels, 4 seeds 0-3; flat start = first evaluated step (100-step grid) where the parity part A (validation) is below 90 percent of its value at step 25), with every initial weight and bias of the network multiplied by c in {2, 4} (the network builder is wrapped; nothing else changes). c = 1 is the A-15 lr 2e-3 row (per seed 900, 1200, 1300, 1100; mean 1,125). 2,500 steps per run; a run without a flat start before step 2,500 is censored at 2,500. Script a28_runs.py, output a28_c<c>/ (A-28-c<c> tags).
Predictions. (1) check: a c = 1 rerun of seed 0 gives flat start 900 as in A-15. (2) the median flat start at c = 2 is at most 0.6 times the c = 1 median (that is at most 675 steps; the grid is 100 steps). (3) at c = 4 the median is at most the c = 2 median, or the run is visibly unstable (parity part A at the end of the run larger than at step 25 in at least one seed; reported either way). I have no strong prior: the XOR network outputs 256 logits and the starting point is a different problem (information, not bit-wise accuracy), so a null result (ratio between 0.8 and 1.25) is quite possible.

### A-28 RESULT (XOR masking, sigma 0.5, Adam constant lr 2e-3, batch 1,024, 1M fixed traces, one-hot labels, 4 seeds, 2,500 steps; a28_c2, a28_c4, a28_c1check; every initial weight and bias times c). Validation parity part A (bits):
c = 1 (A-15, same seeds): flat start 900, 1300, 1300, 1200 with my step-25 rule from the curves (A-15's own table lists 900, 1200, 1300, 1100; the 1-step difference of the rule is not investigated); A at 2,500 steps (test): 0.0155 0.0172 0.0138 0.0181. The c = 1 rerun of seed 0 reproduces 900 (check passes).
c = 2: A at step 1: 0.041 0.039 0.041 0.249; A stays at the plateau 0.034 to 0.035 for the whole run; no seed leaves the plateau (A below 0.9 x 0.0353 = 0.0318 never reached; minimum 0.0338 to 0.0350; A at 2,500: 0.0350 0.0343 0.0349 0.0338).
c = 4: A at step 1: 5.9, 2.1, 9.0, 18.5 bits (huge initial logits); at step 25 0.07 to 0.09; reaches the plateau 0.0352 by step 100 and stays there; no seed leaves the plateau by 2,500 steps (minimum A 0.0349 to 0.0352). The step-25 rule would give a false flat start at step 50 because A at step 25 is still in the start-up transient; the plateau-referenced rule (below 0.9 x 0.0353) is used here.
Scoring against the predictions written before the run:
(1) PASS: the c = 1 rerun of seed 0 gives 900, as in A-15.
(2) MISS, in the opposite direction: no seed at c = 2 has a flat start before step 2,500 (censored), against 900 to 1,300 at c = 1; the prediction was a median at most 0.6 times the c = 1 median.
(3) Not evaluable: c = 2 and c = 4 are both censored at 2,500 steps; no visible instability in the sense of A larger at the end than at step 25 (the c = 4 start-up transient is large but recovered by step 100).
Reading. In this XOR masking setting a larger initial weight scale does not shorten the flat start; with the scale doubled none of four seeds has left the plateau after 2,500 steps (at c = 1 the four seeds leave at 900 to 1,300), and at 4 times the scale the network starts with huge logits and then sits on the same plateau. This is the opposite direction to the adder (C-66, C-72), where a larger scale shortened the plateau. So 'larger initial scale is faster' is not a general property of plateaus in my two problems. A hypothesis (not tested): the adder's bits can be partly read from random hidden features (C-58: p_4 is 0.86 and c_4 0.96 readable at step 0), so a more 'lazy' regime helps it; the XOR secret needs a parity feature that is not in the random features, so the lazy regime is slower. Caveats: 4 seeds; 2,500 steps only (c = 2 may leave the plateau later; not tested beyond); one noise level, one learning rate and batch; the default-scale baseline is A-15.

## A-29 (CONFIRMATORY, follows A-28): is the initial-scale dependence of the XOR flat start monotone, with a smaller scale shortening it?
Idea source: A-28 (scale 2 and 4: no escape in 2,500 steps; default scale: 900 to 1,300) and the adder results (C-66, C-67, C-71: larger scale shortens its plateau and shifts the net toward a smaller relative weight change). If the initial scale controls how 'lazy' training is, and the XOR secret needs a feature that random hidden features do not contain, a smaller scale should shorten the XOR flat start; a vanishing-gradient effect of small weights (the gradient to the first layers falls with the scale) could instead lengthen it.
Design. A-28 configuration with c in {0.5, 0.25} (4 seeds 0-3, 2,500 steps, same driver a28_runs.py with C). Flat start = first evaluated step where the validation parity part A is below 0.9 x 0.0353 (plateau-referenced, as in A-28) and censored at 2,500.
Predictions. (1) at c = 0.5 the median flat start is at most 0.85 times the c = 1 median (c = 1 per-seed values 900, 1300, 1300, 1200; median 1,250; so at most 1,060). (2) at c = 0.25 the median is at most the c = 0.5 median, or the run is visibly stalled (A stays at the plateau for the whole run in at least 2 seeds; reported either way). A miss on (1) with the median above the c = 1 value would mean that small scales are also slower, i.e. the default scale is a minimum in both directions.

### A-29 RESULT (XOR masking, Adam constant lr 2e-3, batch 1,024, 4 seeds, 2,500 steps; a29_c0.5, a29_c0.25; flat start = first evaluated step where the validation parity part A is below 0.9 x 0.0353; step-25 value equals the plateau value for these scales). Flat start per seed:
c = 0.25: 2200, 2000, 1900, 1900 (median 1,950) | c = 0.5: 1900, 1500, 1500, 1600 (median 1,550) | c = 1 (A-15): 900, 1300, 1300, 1200 (median 1,250) | c = 2 and c = 4 (A-28): none by 2,500 (censored).
A at 2,500 (test): c = 0.5 0.0253 0.0193 0.0232 0.0222; c = 0.25 0.0287 0.0290 0.0276 0.0281 (c = 1: 0.0138 to 0.0181).
Scoring against the predictions written before the run:
(1) MISS: the c = 0.5 median (1,550) is above the c = 1 median (1,250), not at most 1,060.
(2) MISS: the c = 0.25 median (1,950) is above the c = 0.5 median; no run is stalled.
Reading. The alternative I wrote down in advance is what happened: the flat start of this XOR setting is shortest at the default scale among the five scales tried (c = 0.25, 0.5, 1, 2, 4: 1,950, 1,550, 1,250, then censored twice), i.e. a minimum in both directions. The idea that a smaller scale would shorten it because larger scales are 'lazier' is not supported. In the adder (C-66, C-72) the trend over c = 0.5, 1, 2, 4 is monotone (slower, default, faster, faster). Whatever the reason the XOR secret behaves differently, I do not have a tested explanation. Caveats: 4 seeds; 2,500 steps; one noise level, learning rate and batch; the default scale is not claimed to be optimal in other settings.

# Night of 26-27 Oct 2026: filling gaps
## A-30 (CONFIRMATORY): does one-hot label noise, injected into clean soft-label training, bring the flat start back?
Idea source: A-17/A-18 (soft labels remove the flat start, mixing one-hot labels brings it back), A-24 (soft first moment rescues), A-26 (the one-hot gradient has the same mean but about 900 times more minibatch noise). Missing so far: a test where ONLY the noise is added, with the soft-label mean kept.
Design. Hand-written ordinary Adam (as A-24, both moments from one gradient) on g = g_soft(B) + K (g_hard(B') - g_soft(B')), B the training minibatch and B' an independent minibatch of the same size (1,024): the mean of the added term is zero given the inputs (the posterior is the mean of the one-hot label), so the expected gradient is the soft-label one and K scales pure label noise. K = 1 (noise about equal to that of one-hot training at batch 1,024) and K = 0.3; XOR, sigma 0.5, 3x256, constant lr 2e-3, 4,000 steps, seeds 0-3 (same initialisations and data as A-24). References from A-24: hard labels (HH) flat start 1,100 1,200 900 1,700 (median 1,150); soft labels (SS) 100 in all seeds. Flat start as in A-24 (first evaluated step where the validation parity part A is below 90 percent of its step-25 value). Script a30_noise_inject.py, output a30/.
Predictions (written before running). (1) K = 1: median flat start at least 800 (the noise alone recreates most of the delay). (2) K = 0.3: median between 200 and the K = 1 median. (3) K = 1: A at 4,000 steps at least 0.005 bits (soft labels: 0.0004). If (1) fails, the delay is not caused by label noise alone.

## A-31 (CONFIRMATORY; why a larger initial scale stalls the XOR flat start, A-28)
Idea. With every initial weight and bias doubled (A-28, c = 2), no seed left the plateau in 2,500 steps. Two candidate causes: (a) large initial output logits (with all four layers doubled the logits grow roughly 16-fold), which put the 256-way softmax far from uniform and change the early gradient; (b) a larger scale of the hidden features themselves (a 'lazier' start). Separating them: hidden layers x2 with the output layer /8 (logits back to about their default size while the hidden features stay large), and hidden layers x1 with the output layer x2 (only the logits larger, twice).
Design. A-28 configuration (XOR, sigma 0.5, 3 x 256, Adam 2e-3, batch 1,024, 1M fixed traces, 2,500 steps, seeds 0-3); flat start as in A-28 (first evaluated step where validation parity part A is below 0.9 x 0.0353). Script a31_runs.py, output a31/.
Predictions (written before running). Logit account: hidden x2 / output /8 escapes like the default (median flat start at most 1.5 x 1,250 = 1,875). Hidden-scale account: it stalls (no escape by 2,500 in at least 3 of 4 seeds). Output x2 alone: no prediction (it is the milder case); reported.

### A-31 result (4 seeds per arm; a31/*_curves.csv, a31_score.py -> a31_scores.txt)
| arm | flat start per seed | median | test A at 2,500 (bits) |
|---|---|---|---|
| default (A-15/A-28, c = 1) | 900 1,300 1,300 1,200 | 1,250 | 0.0138-0.0181 |
| hidden x2, output /8 (logits about default size) | 2,000 2,300 1,500 1,900 | 1,950 | 0.0305 0.0314 0.0230 0.0308 |
| hidden x1, output x2 (logits about twice) | 1,600 1,400 2,300 1,500 | 1,550 | 0.0187 0.0122 0.0310 0.0183 |
| all layers x2 (A-28, c = 2) | none by 2,500 | censored | 0.0338-0.0350 |
Predictions. Logit account (hidden x2 / output /8 escapes with median at most 1,875): MISSED narrowly (1,950). Hidden-scale account (it stalls, no escape in at least 3 of 4): MISSED (all 4 escape). Output x2 alone: no prediction; it slows the escape by about 1.24 times.
Reading. Neither single account holds. Doubling the hidden layers with the logits kept at their default size slows the escape about 1.56 times; doubling only the output layer slows it about 1.24 times; neither stalls. Their product (about 1.9 times, i.e. about 2,400 steps) is close to the 2,500-step censoring of A-28, so the c = 2 'stall' may be the two slowdowns together rather than a true stall. A-31b tests that by running c = 2 longer.

## A-31b (CONFIRMATORY, follows A-31): does the c = 2 network stall, or only escape later than 2,500 steps?
Design. A-28 c = 2 (every initial weight and bias doubled: CH = 2, CO = 2), seeds 0-3, 6,000 steps, otherwise as A-31; flat start plateau-referenced. Script a31_runs.py with STEPS = 6000, output a31b/.
Predictions (written before running). (1) If the A-31 slowdowns multiply, c = 2 escapes in at least 3 of 4 seeds by 6,000 steps, with median flat start between 2,500 and 4,000. (2) If it does not escape in at least 3 of 4 seeds by 6,000, there is an interaction (the combination stalls) and I will say so.

## A-32 (another leakage model: weighted bits instead of Hamming weight)
Information first (EXPLORATORY, computed before any training; a32_weighted.py, no predictions written): each share leaks L(v) = sum_i w_i bit_i(v) with fixed weights w ~ U(0.5, 1.5) (seed 2026) plus Gaussian noise sigma 0.5. Exact posterior by XOR convolution (Walsh-Hadamard) or circular convolution (addition), checked against brute force on 5 traces (max difference below 1e-9). Monte Carlo estimate of the mutual information on 100,000 traces: XOR 0.1545 bits, addition 0.1480 bits (Hamming-weight leakage at the same sigma: XOR clearly above addition, Experiment 1). So under this leakage model the two maskings leak almost the same; the large XOR advantage of Experiment 1 is a property of Hamming-weight leakage. Check with 400,000 traces per masking (a32_info.txt): XOR 0.1550 +- 0.0009 bits, addition 0.1483 +- 0.0009; difference 0.0067 +- 0.0013 bits (about 5 standard errors): XOR still leaks more, but only about 4.5 percent more, against several times more under Hamming-weight leakage.
Learning (CONFIRMATORY). Design: XOR, hard (one-hot) versus soft (exact-posterior) labels, 3x256 ReLU, Adam 2e-3, batch 1,024, 1M fixed traces, 4,000 steps, seeds 0-3; total excess (network cross-entropy minus exact-Bayes cross-entropy, bits) on 100,000 validation traces at steps 0, 25, 50 and every 100. Plus addition with hard labels, seeds 0-3 (is XOR harder to learn here too?).
Predictions (written before running). (1) XOR, hard labels: a plateau, i.e. the excess falls by less than 10 percent between step 100 and step 600 in at least 3 of 4 seeds. (2) XOR, soft labels: it falls by more than 50 percent over the same window in at least 3 of 4 seeds. (3) XOR hard-label excess at 4,000 at least 3 times the soft-label excess. (4) XOR hard-label excess at 4,000 larger than addition's in at least 3 of 4 paired seeds. If (1) fails, the flat start is specific to Hamming-weight leakage and the label-noise account cannot be tested in this model.

### A-30 result (4 seeds per K; a30/A-30_curves.csv, A-30_final.csv, a30_scores.txt)
| gradient | flat start per seed | median | A at 4,000 (mean, bits) |
|---|---|---|---|
| soft labels (A-24 SS) | 100 100 100 100 | 100 | 0.0004 |
| soft + 0.3 x one-hot label noise | 300 400 200 200 | 250 | 0.0019 |
| soft + 1 x one-hot label noise | 1,800 1,500 1,400 1,000 | 1,450 | 0.0101 |
| hard labels (A-24 HH) | 1,100 1,200 900 1,700 | 1,150 | 0.0105 |
Predictions. (1) K = 1 median at least 800: HIT (1,450). (2) K = 0.3 between 200 and the K = 1 median: HIT (250). (3) K = 1 A at 4,000 at least 0.005: HIT (0.0101).
Reading. With the expected gradient held at the clean soft-label gradient, adding only the zero-mean noise that one-hot labels carry brings back the whole flat start (1,450 against 1,150 for hard labels and 100 for soft labels) and the whole final parity error (0.0101 against 0.0105), and a third of the noise gives a short flat start (250). This is the causal test that was missing: in this setting the flat start is caused by the label noise in the gradient, not by anything else that differs between one-hot and soft-label training. Caveats: 4 seeds; one sigma; the injected noise comes from an independent minibatch, so its correlation with the training minibatch's input noise differs from real one-hot training.

## A-33 (CONFIRMATORY; is the XOR flat start and its label-noise cause an Adam phenomenon?)
Pilot disclosure (seed 0 only, 2,000 steps, before writing the predictions; scratch output not used as data): momentum SGD with hard labels left the parity plateau neither at lr 0.02 nor at 0.1 within 2,000 steps (A 0.0351 and 0.0346); with soft labels at lr 0.1 the parity part A left the plateau at about step 1,500 (0.0312) and was 0.0241 at 2,000.
Design. Momentum SGD (momentum 0.9, lr 0.1), XOR, sigma 0.5, 3 x 256, batch 1,024, 1M fixed traces, 6,000 steps, hard versus soft (exact-posterior) labels, seeds 0-3; flat start = first evaluated step (100-step grid) where the validation parity part A is below 0.9 x 0.0353 (plateau-referenced, as A-28), censored at 6,000. Script a33_sgd.py, output a33/.
Predictions (written after the pilot, before the runs). (1) soft labels: median flat start at most 2,000. (2) hard labels: median at least 2 times the soft median, or censored in at least 3 of 4 seeds: label noise also delays the parity under SGD.

## A-34 (CONFIRMATORY; hypothesis H3: is the flat start a difficulty of building oscillating features, removable by a periodic first layer?)
Idea source: hypothesis table H3 (Telgarsky 2016; periodic activations, Ziyin et al. 2020, abstract only, see literature.md; correction 27 Oct: I first wrote 'not opened', but the abstract had been fetched on 23 Oct) and Experiment 10 (fixed sinusoidal inputs, even at the wrong frequency, remove most of the XOR gap; "the exact cause is still untested"). A-30 showed that one-hot label noise causes the flat start when the parity signal in the gradient is weak; this tests whether a network whose first layer already oscillates escapes despite the same label noise.
Design. A-15 configuration as in A-28/A-31 (XOR, sigma 0.5, 3 x 256, Adam constant 2e-3, batch 1,024, 1M fixed traces, one-hot labels, seeds 0-3, 2,500 steps); the first hidden layer's ReLU replaced by sin (learnable weights and biases, everything else unchanged). Two initial scales of the first-layer weights: WS = 1 (default init, |w| up to 0.71 per reading unit, i.e. oscillation periods of at least 8.9 Hamming-weight units) and WS = 4 (|w| up to 2.83, periods down to 2.2; the parity ripple has period 2, w = pi). Flat start = first evaluated step where the validation parity part A is below 0.9 x 0.0353 (plateau-referenced, as A-28). ReLU reference at the same configuration: 900, 1,300, 1,300, 1,200 (median 1,250; A-29 c = 1 row). Script a34_periodic.py, output a34/.
Predictions (written before running). (1) WS = 4: median flat start at most 625 (half the ReLU median). (2) WS = 1: no clear help; median at least 1,000 (Adam moves a weight about 0.002 per step, so growing |w| from 0.7 to about pi takes on the order of 1,000 steps). If (1) holds, the delay is about the gradient's parity signal being weak in a ReLU network (a representation problem) and label noise only hides that weak signal; if (1) fails, oscillating first-layer features are not enough.

### A-33 result (momentum SGD 0.9, lr 0.1, 6,000 steps, 4 seeds per arm; a33/A-33_curves.csv, a33_score.py -> a33_scores.txt)
| labels | flat start per seed | median | validation A at 2,000 / 6,000 (mean, bits) |
|---|---|---|---|
| soft (exact posterior) | 1,500 900 1,100 1,000 | 1,050 | 0.0207 / 0.0047 |
| hard (one-hot) | 3,600 3,700 2,300 4,200 | 3,650 | 0.0345 / 0.0165 |
Predictions. (1) soft median at most 2,000: HIT (1,050). (2) hard median at least 2 times the soft median, or censored in at least 3 of 4: HIT (3.5 times; no run censored).
Reading. The label-noise delay is not specific to Adam: with momentum SGD, one-hot labels delay the parity escape 3.5 times relative to exact-posterior labels (3,650 against 1,050 steps), about the same mechanism as under Adam. One difference: under SGD the soft-label run also has a flat start (about 1,000 steps, against 100 with Adam, A-17/A-24), so with SGD part of the delay is the weak parity gradient itself, which Adam's per-parameter normalisation removes when labels are clean. Caveats: one learning rate (0.1), chosen after a one-seed pilot (disclosed in the plan); 4 seeds.

### A-31b result (c = 2, every initial weight and bias doubled, 6,000 steps, 4 seeds; a31b/, a31_score.py a31b 6000 -> a31b_scores.txt; evaluation every 200 steps)
Flat start per seed: none by 6,000, 4,400, none by 6,000, 4,400 (median 5,200 with censored runs counted at 6,000). Validation A at 6,000: 0.0331, 0.0285, 0.0333, 0.0292 (default c = 1 at 2,500: 0.0138-0.0181). All four creep down slowly from about step 3,000.
Predictions. (1) escape in at least 3 of 4 by 6,000 with median 2,500-4,000 (the two A-31 slowdowns multiplying, about 2,400): MISSED (2 of 4 escape, at 4,400). (2) no escape in at least 3 of 4 (a true stall): MISSED (2 of 4).
Reading. c = 2 is not a permanent stall but a very long one: at least 3.5 times the default (4,400 or more against 1,250), much longer than the product of the two separate slowdowns (about 1.9). So doubling the hidden layers and doubling the output layer compound more than multiplicatively. A-31c tests whether the overall output scale (product of the layer scales) is what matters.

## A-31c (CONFIRMATORY, follows A-31/A-31b): is the c = 2 stall set by the overall output scale (all layers multiply), i.e. a 'lazy' start?
Disclosure: written after seeing A-31b seed 0 only (c = 2, 6,000 steps: A still 0.0331 at step 6,000, barely below the plateau), before the other A-31b seeds finished.
Idea. With ReLU layers and small biases, the network output at initialisation scales roughly as the product of the layer scales: alpha = CH^3 x CO. A-31: alpha about 1 (hidden x2, output /8) escapes at 1,950; alpha 2 (output x2) at 1,550; A-28/A-31b c = 2 has alpha about 16 and stalls far longer. Chizat, Oyallon and Bach 2019 (arXiv:1812.07956; read through a page-reader summary for the adder project, carry_chain PROGRESS.md) describe a large output scale as pushing training toward the lazy regime, where features barely change; the XOR parity needs a new feature (A-28 reading), so a lazy start should stall. Test: output layer x16 with default hidden layers (alpha about 16, hidden scale default).
Design. a31_runs.py with CH = 1, CO = 16, STEPS = 6,000, seeds 0-3, output a31c/; flat start plateau-referenced (evaluation every 200 steps beyond 4,000-step runs).
Predictions (written before running). (1) Output-scale account: CO = 16 stalls like c = 2 (no escape by 6,000 in at least 3 of 4 seeds, or median flat start at least 2 times the c = 2 hidden x2 / output /8 median, i.e. at least 3,900). (2) If instead CO = 16 escapes with median at most 2,500, the stall needs the hidden scale too (an interaction), and I will say so.

### A-32 learning result (weighted-bit leakage, 4 seeds per arm, 4,000 steps; a32/A-32_curves.csv, a32_score.py -> a32_scores.txt)
Total excess over the exact-Bayes cross-entropy (bits, 100,000 validation traces), mean over seeds at steps 100 / 600 / 4,000: XOR hard 0.0935 / 0.0349 / 0.0158; XOR soft 0.0245 / 0.0013 / 0.0003; addition hard 0.1060 / 0.0314 / 0.0134.
Predictions. (1) XOR hard-label plateau (excess falls by less than 10 percent from step 100 to 600 in at least 3 of 4 seeds): MISSED (it falls 61-65 percent in every seed). (2) XOR soft falls by more than 50 percent: HIT (92-96 percent). (3) XOR hard excess at 4,000 at least 3 times the soft one: HIT (55 times). (4) XOR hard above addition hard at 4,000 in at least 3 of 4 paired seeds: HIT (3 of 4; 0.0169/0.0137, 0.0142/0.0143, 0.0172/0.0120, 0.0149/0.0135; means 0.0158 against 0.0134, 1.18 times).
Reading. With weighted-bit leakage there is no flat start: the XOR network's excess falls from the beginning, like addition's. The parity plateau of the main experiments is therefore a property of Hamming-weight leakage, where the XOR posterior depends on a single weak parity clue (the parity of the secret's Hamming weight); with unequal bit weights that clue is not isolated. Two things carry over: one-hot labels still cost most of the remaining excess (hard 0.0158 against soft 0.0003 bits at 4,000 steps), and XOR is still slightly further from its limit than addition (1.18 times, against about 3 times under Hamming-weight leakage at 4,000 steps in Experiment 3, which used a cosine schedule rather than a constant rate, so the comparison is indicative). Information under this model: XOR 0.1550, addition 0.1483 bits (A-32 information part). Caveats: one weight draw (seed 2026), one sigma, 4 seeds.

### A-31c result (hidden layers default, output layer x16, 6,000 steps, 4 seeds; a31c/, a31_score.py a31c 6000 -> a31c_scores.txt)
Flat start per seed: 4,800, 4,200, 4,200, 4,000 (median 4,200; all four escape). Validation A at 6,000: 0.0254, 0.0168, 0.0279, 0.0216.
Predictions. (1) Output-scale account (no escape in at least 3 of 4, or median at least 3,900): HIT (median 4,200). (2) Interaction (median at most 2,500): not the case.
Reading (A-28 to A-31c together). The length of XOR's flat start follows the overall output scale at initialisation, roughly the product of the layer scales: about 1 (hidden x2, output /8): 1,950 steps; 1 (default): 1,250; 2 (output x2): 1,550; 16 (output x16): 4,200; about 16 (every layer x2): 4,400 or more, two of four not escaped by 6,000. (Smaller scales are also slower, A-29: 1,550 at c = 0.5 and 1,950 at c = 0.25, so the dependence is not monotone and the default is near the fastest scale tried.) A large output scale alone reproduces most of the c = 2 delay, while doubling the hidden layers with the output scale kept near 1 costs only about 1.5 times. This fits the lazy-training picture (Chizat, Oyallon and Bach 2019, read through a summary). One reading, not measured here: with a large output scale the network fits the easy part of the labels with small weight changes, so the new parity feature, which needs the first layers to change, forms much later. It is also the opposite of the adder, where a larger initial scale shortened the plateau (C-66, C-72); the adder's bits are partly readable from random features (C-58), the XOR parity is not. Caveats: 4 seeds per arm; the 'product of layer scales' is approximate (biases are scaled too); I did not measure weight movement here.

## A-35 (EXPLORATORY-WITH-PREDICTION, follows A-31c): do the weights move less during the flat start when the output scale is large?
Idea source: my lazy-training reading of A-31c (not measured there). Chizat et al. state the lazy argument for gradient descent; under Adam every parameter moves by about the learning rate per step whatever the gradient size, so it is not obvious that a large output scale makes the hidden weights move less.
Design. Default (CO = 1) and output x16 (CO = 16), seeds 0-1, the same data, initial weights and batch draws as a31_runs.py (check: validation A at step 2,000 equals the A-31/A-31c curves), 2,000 steps; relative change ||W(t) - W(0)|| / ||W(0)|| of each layer's weights at steps 100, 250, 500, 1,000, 1,500, 2,000. Script a35_move.py, output a35/.
Predictions (written before running). (1) At step 1,000 the relative change of the first-layer weights with CO = 16 is at most 0.67 times the default's in both seeds (the hidden layers move less: lazy). (2) Exploratory: the output layer's relative change is smaller with CO = 16 (its weights start 16 times larger). If (1) fails, the A-31c delay is not laziness in the sense of less hidden-weight movement, and I will say so.

### A-34 result (sine first layer, one-hot labels, 2,500 steps, 4 seeds per arm; a34/, a31_score.py a34 2500 -> a34_scores.txt)
| first layer | flat start per seed | median | validation A / C / total at 2,500 (range, bits) |
|---|---|---|---|
| ReLU (A-15 lr 2e-3, same seeds) | 900 1,300 1,300 1,200 | 1,250 | 0.0138-0.0181 / 0.0063-0.0082 / 0.0280-0.0335 |
| sine, default init (WS = 1) | 800 700 900 600 | 750 | 0.0039-0.0063 / 0.0078-0.0093 / 0.0156-0.0190 |
| sine, weights x4 (WS = 4) | 100 100 100 100 | 100 | 0.0021-0.0025 / 0.0130-0.0148 / 0.0202-0.0207 |
Predictions. (1) WS = 4 median at most 625: HIT (100; the parity part falls from step 25 on, 0.0058-0.0115 by step 100). (2) WS = 1 no clear help, median at least 1,000: MISSED (750, and its parity part at 2,500 is 3 times lower than ReLU's).
Reading. A first layer that can oscillate removes XOR's flat start even with one-hot labels when its initial frequencies reach near the parity ripple (period 2), and shortens it and leaves far less parity error when it starts at low frequencies. Together with A-17 (soft labels remove it) and A-30 (one-hot label noise alone brings it back), this gives one picture: the parity signal in the gradient of a ReLU network is weak because the network has to build an oscillating feature from plain readings, and one-hot label noise hides that weak signal; removing either the noise or the representational difficulty removes the flat start. Side effect: the WS = 4 sine net is worse on the secret-inside-class part C (0.013-0.015 against 0.006-0.008), i.e. it treats same-weight secrets differently, so its total is between the ReLU and the WS = 1 values. The WS = 1 sine net has the lowest total of the three (0.016-0.019). Caveats: 4 seeds, one noise level, 2,500 steps; Ziyin et al. 2020 (abstract only) is about extrapolating periodic functions, not about how fast a network fits a periodic clue, so it motivates but does not support this result; I used plain sin, not their x + sin^2(x).

### A-35 result (seeds 0-1; a35/move.csv, a35.log)
Reproduction check: validation A at step 2,000 equals the A-31c curves for CO = 16 (0.0350, 0.0347) and the A-15 lr 2e-3 curves for CO = 1 (0.0186, 0.0225).
Relative weight change ||W(t) - W(0)|| / ||W(0)|| at step 1,000 (layer 1 / layer 2 / layer 3 / output), seeds 0 and 1:
- CO = 1: 0.226 / 1.060 / 0.593 / 1.538 and 0.234 / 1.079 / 0.553 / 1.527.
- CO = 16: 0.132 / 0.668 / 0.371 / 0.137 and 0.134 / 0.606 / 0.382 / 0.162.
At step 2,000: layer 1 0.317, 0.318 (CO = 1) against 0.186, 0.186 (CO = 16).
Predictions. (1) layer-1 change with CO = 16 at most 0.67 times the default at step 1,000 in both seeds: HIT (0.59, 0.57). (2) output layer's relative change smaller with CO = 16: HIT (0.14-0.16 against 1.5; relative to weights that start 16 times larger, so in absolute terms the output layer moves about as much or more).
Reading. Even under Adam, a 16 times larger output layer makes all three hidden layers move about 0.6 times as far in the first 1,000-2,000 steps, while the parity part stays on its plateau. That is the lazy-training signature (smaller relative change of the feature layers), measured, and it goes with the 3.4 times longer flat start of A-31c. Why Adam moves less here is not measured; a possible reason is that once the large output layer has fitted the easy part of the labels, the hidden-layer gradients are dominated by label noise and Adam's steps partly cancel. Caveats: 2 seeds; movement measured, not its direction.

## A-36 (check): numbers for A-30 to A-35 recomputed from the raw curve files (a36_verify.py -> a36_verify.txt)
Every scoring script (A-30, A-31, A-31b, A-31c, A-32, A-33, A-34) reproduces its saved score file exactly. Rounding corrections: A-32 means at step 100 for XOR hard 0.0935 (not 0.0936) and at step 600 for addition hard 0.0314 (not 0.0315); A-35 layer-1 ratio for seed 0 0.59 (not 0.58). Reproduction: A-34 WS = 4 seed 3 rerun from scratch gives an identical curve (28 of 28 rows).

## A-37 (EXPLORATORY-WITH-PREDICTION, information only, no training): a transition (Hamming-distance) leak between the two shares
Idea source: the README's untested 'Hamming-distance leakage'. Model (my choice, stated as an assumption): besides HW(share 1) + noise and HW(share 2) + noise, the device leaks the Hamming distance between the two shares, HW(share1 XOR share2) + noise, as when one register is overwritten by the other. For XOR masking share1 XOR share2 = s, so this third reading is an unmasked first-order leak of HW(s); for addition it is HW(m XOR (s - m mod 256)), which still depends on the mask.
Computation: exact posterior by enumerating all (s, m) (likelihood factorises over the three Hamming weights), mutual information I(S; readings) = 8 - E[-log2 p(s | readings)] by Monte Carlo on 200,000 traces per masking at sigma 0.5; check: without the third reading it must reproduce Experiment 1 (XOR 0.144, addition 0.108 bits). Script a37_hd_info.py, output a37_info.txt.
Predictions (written before computing). (1) The two-reading check reproduces Experiment 1 within 0.003 bits. (2) With the transition reading, XOR's information rises by at least 0.5 bits (a direct first-order leak of HW(s): a single noisy HW reading of an 8-bit uniform value at sigma 0.5 gives about 1.5 to 2 bits on its own). (3) Addition's rises by less than half of XOR's increase.

### A-37 result (200,000 traces per cell; a37_hd_info.py -> a37_info.txt)
| masking | two readings (shares only) | plus transition reading |
|---|---|---|
| XOR | 0.1447 +- 0.0012 bits | 1.6334 +- 0.0030 |
| addition | 0.1089 +- 0.0011 | 0.8057 +- 0.0027 |
Predictions. (1) two-reading check within 0.003 of Experiment 1 (0.144, 0.108): HIT (0.1447, 0.1089). (2) XOR rises by at least 0.5 bits: HIT (+1.49). (3) addition rises by less than half of XOR's increase: HIT, narrowly (+0.70 against a bound of 0.74).
Execution note: my first computation used share2 = s - m for addition; the project's convention (train_mlp.make_data) is share2 = s + m. Rerun with the project convention (the table above); the first run's addition values were 0.1097 and 0.8063 (a37_info_sminusm.txt), the same within the Monte Carlo error.
Reading. If the two shares ever meet in a way that leaks their Hamming distance, XOR masking loses its protection at first order (the third reading is HW(s) itself) and the information jumps about elevenfold, to 1.63 bits; addition masking also leaks much more (0.81 bits, about sevenfold) because HW(m XOR (s - m)) still depends strongly on s, but about half as much as XOR. In this model the ranking of Experiment 1 (XOR leaks more) holds and the gap widens. This is an information calculation under my own transition model, not a statement about a real device; no networks were trained on it.

## A-38 (CONFIRMATORY, follows A-37): learning under the transition-leak model
Design. Readings HW(m) + noise, HW(share2) + noise, HW(m XOR share2) + noise (sigma 0.5; share2 = s XOR m or s + m mod 256), one-hot labels, 3 inputs, 3 x 256 ReLU, Adam constant 2e-3, batch 1,024, 1M fixed traces, 4,000 steps, seeds 0-3, XOR and addition; total excess over the exact-Bayes cross-entropy on 100,000 validation traces (exact posterior by enumerating (s, m), checked by brute force on 3 traces). Script a38_transition.py, output a38/.
Predictions (written before running). (1) No flat start for XOR: its excess falls by more than 50 percent from step 100 to 600 in at least 3 of 4 seeds (with HW(s) read directly, the parity clue is no longer the only route). (2) XOR's excess at 4,000 is at most addition's in at least 3 of 4 paired seeds (XOR has a first-order leak; addition does not).

### A-38 result (4 seeds per masking; a38/A-38_curves.csv, a38_score.py -> a38_scores.txt)
Exact information under this model (100,000 validation traces): XOR 1.6335, addition 0.8095 bits. Total excess (bits) at steps 100 / 600 / 4,000, per seed: XOR 0.174/0.078/0.063, 0.237/0.079/0.043, 0.215/0.125/0.042, 0.257/0.094/0.047; addition 0.232/0.072/0.034, 0.210/0.079/0.040, 0.225/0.079/0.032, 0.211/0.072/0.035. Means at 4,000: XOR 0.0488, addition 0.0351; the networks capture 96-97 percent (XOR) and 95-96 percent (addition) of the information.
Predictions. (1) XOR excess falls by more than 50 percent from step 100 to 600 in at least 3 of 4 seeds: HIT (3 of 4: 55, 67, 42, 64 percent). (2) XOR excess at 4,000 at most addition's in at least 3 of 4 paired seeds: MISSED (0 of 4).
Reading. With a direct reading of HW(s) available, XOR has no parity plateau, but its networks end further from the exact limit than addition's in absolute bits (0.049 against 0.035), while capturing a slightly larger share of a twice larger amount of information. XOR seed 0 is worse at 4,000 than at 2,000 (0.063 against 0.053), the data-reuse pattern of A-20/A-25. My prediction assumed the first-order leak would make XOR easier in absolute terms; it does not. Caveats: 4 seeds, one sigma, my own transition model.

Check addendum (A-36): a38_score.py reproduces a38_scores.txt and a37_hd_info.py reproduces a37_info.txt exactly; the A-38 per-seed values quoted above were checked against the score file.

# Status after the night of 26-27 Oct 2026 (A-30 to A-38)
Closed tonight: the cause of the flat start (one-hot label noise, causal, A-30) and why the parity signal is weak (a ReLU network must build an oscillating feature; a sine first layer removes the flat start, A-34); not an Adam artefact (A-33); the initial-scale effect follows the overall output scale and goes with less hidden-weight movement (A-31 to A-31c, A-35); weighted-bit leakage (A-32) and a transition (Hamming-distance) leak (A-37, A-38).
Not possible here: real traces. Missed predictions tonight: A-31 (both accounts), A-31b (both), A-32 (1), A-34 (2), A-38 (2). Checks: A-36 (rounding corrections only) and one rerun identical.

## A-39 (CONFIRMATORY control for A-34): is it the oscillation or the first-layer scale?
Why: the A-34 sine net with WS = 4 also had its first-layer weights 4 times larger, and A-28/A-31 showed that initial scale alone changes the flat start. Control: ReLU first layer (as in A-15) with only the first-layer weights x4 (biases unchanged), everything else as A-34 (seeds 0-3, 2,500 steps, one-hot labels). Script a34_periodic.py with ACT=relu WS=4, output a39/.
Predictions (written before running). (1) If the oscillation is what matters, the ReLU x4 control does not remove the flat start: median at least 1,000 (A-34 sine x4: 100; ReLU default: 1,250). (2) Its validation parity part A at 2,500 is at least 3 times the sine x4 value (0.0021-0.0025), i.e. at least 0.0063.

## A-40 (CONFIRMATORY control for A-38): is XOR's larger excess under the transition leak label noise?
Why: A-38 (one-hot labels) left XOR 0.049 bits from its limit against 0.035 for addition, the reverse of my prediction. Label noise is the main source of excess in every other setting (A-17, A-32). Control: the same runs with exact-posterior (soft) labels. Script a38_transition.py with LABELS=soft, output a40/.
Predictions (written before running). (1) With soft labels both maskings end at most 0.010 bits from their limit at 4,000 steps (mean over 4 seeds). (2) The XOR-minus-addition gap of means is at most a third of A-38's (0.0137 bits), i.e. at most 0.0046.

### A-39 result (ReLU, first-layer weights x4, 4 seeds, 2,500 steps; a39/, a31_score.py a39 2500 -> a39_scores.txt)
Flat start per seed: none by 2,500, none, 1,800, none (median at least 2,500). Validation A at 2,500: 0.0325, 0.0329, 0.0268, 0.0331 (sine x4: 0.0021-0.0025; ReLU default: 0.0138-0.0181).
Predictions. (1) the ReLU x4 control does not remove the flat start (median at least 1,000): HIT (at least 2,500; three of four seeds still on the plateau). (2) its A at 2,500 at least 0.0063: HIT (0.027-0.033).
Reading. A larger first-layer scale by itself makes the ReLU network slower, not faster (consistent with A-28/A-31: larger scales lengthen the flat start), so the A-34 effect of the sine x4 layer (flat start 100) comes from the oscillation, not from the scale. The A-34 reading stands.

## A-41 (CONFIRMATORY; does the sine-layer result hold at a higher noise level?)
Why: A-34 used sigma 0.5 only. At larger noise the parity ripple in the readings is much weaker (amplitude about exp(-pi^2 sigma^2): 0.085 at 0.5, 0.004 at 0.75), so the flat start of a ReLU network should be much longer.
Design. sigma 0.75; ReLU default (ACT=relu, WS=1) and sine x4 (WS=4); XOR, one-hot labels, constant Adam 2e-3, batch 1,024, 1M fixed traces, 4,000 steps, seeds 0-3. Flat start = first evaluated step (multiple of 100, at least 200) where validation A is below 0.9 x A_ref, with A_ref = median over the ReLU seeds of A at step 100 (fixed before looking at the sine runs). Script a34_periodic.py with SIG=0.75 STEPS=4000, output a41/.
Predictions (written before running). (1) ReLU: median flat start at least 2,000 or censored at 4,000 in at least 3 of 4 seeds. (2) sine x4: median flat start at most half of the ReLU median (censored counted as 4,000), and its A at 4,000 below the ReLU A in all 4 paired seeds.

### A-40 result (transition leak, exact-posterior labels, 4 seeds per masking, 4,000 steps; a40/A-38_curves.csv, a40_score.py -> a40_scores.txt)
Total excess at 4,000 (bits): XOR 0.0091, 0.0093, 0.0096, 0.0142 (mean 0.0105; one-hot A-38: 0.0488); addition 0.0068, 0.0057, 0.0061, 0.0083 (mean 0.0067; one-hot: 0.0351).
Predictions. (1) both soft-label means at most 0.010: MISSED narrowly (XOR 0.0105). (2) XOR-minus-addition gap at most 0.0046: HIT (0.0038, against 0.0137 with one-hot labels).
Reading. Exact-posterior labels cut the excess about fivefold for both maskings and shrink XOR's extra gap from 0.0137 to 0.0038 bits, so most of A-38's 'XOR further from its limit' was label noise again, which costs more where there is more information to learn. A smaller gap remains with clean labels. Unlike A-32 (soft-label excess 0.0003), a residual of 0.006-0.014 bits remains here with clean labels; not investigated (three readings, a posterior that is harder to represent, finite data are candidates).

## A-35b (CONFIRMATORY; A-35 on two more seeds)
Design. a35_move.py, CO = 1 and CO = 16, seeds 2-3 (appended to a35/move.csv). Prediction (written before running): layer-1 relative change at step 1,000 with CO = 16 at most 0.67 times the CO = 1 value in both new seeds.

### A-41 aborted after one run (design error, 27 Oct 07:25)
The first ReLU run at sigma 0.75 showed a parity part A of only 0.0004 bits at every step. My own A-14 notes give the parity-group information at sigma 0.75 as 0.0006 bits, so there is almost nothing to learn and a flat start cannot be measured; I should have checked this before planning. The run is kept in a41_aborted_sig0.75/ and not used. About 20 minutes of compute were lost. Replaced by A-41b at sigma 0.6. (While stopping it, the A-35b queue started early and was stopped before writing anything; it is rescheduled after A-41b.)

## A-41b (CONFIRMATORY; replaces A-41): the sine-layer result at a weaker parity signal, sigma 0.6
Why: as A-41. At sigma 0.6 the parity-group information is 0.0073 bits (A-14 table) and the ripple amplitude about exp(-pi^2 x 0.36) = 0.029 (0.085 at sigma 0.5), so the signal is weaker but still measurable.
Design. As A-41 with SIG = 0.6: ReLU default and sine x4, 4 seeds, 4,000 steps; flat start = first evaluated step (multiple of 100, at least 200) where validation A is below 0.9 x A_ref, A_ref = median over the ReLU seeds of A at step 100 (fixed before looking at the sine runs). Output a41b/.
Predictions (written before running). (1) ReLU: median flat start at least 2,000, or censored at 4,000 in at least 3 of 4 seeds. (2) sine x4: median flat start at most half of the ReLU median (censored counted as 4,000), and its A at 4,000 below the ReLU A in all 4 paired seeds.

## A-42 (EXPLORATORY-WITH-PREDICTION; the side effect of the sine x4 layer: is its larger part C a data-reuse effect?)
Idea source: A-34 (sine x4: part C, same-weight secrets treated differently, 0.013-0.015 bits against 0.006-0.008 for ReLU) and A-12/A-20/A-25 (with the 1M fixed traces, C grows by data reuse; fresh data removes most of it).
Design. A-34 sine x4 configuration (sigma 0.5, 2,500 steps, seeds 0-3) with a fresh batch from the simulator at every step (FRESH=1), output a42/.
Predictions (written before running). (1) With fresh data the sine x4 network's validation C at 2,500 is at most 0.008 bits in at least 3 of 4 seeds (the extra C was overfitting the fixed traces). (2) Its flat start stays at most 200 steps (fresh data should not slow the parity escape).

### A-42 result (sine x4, fresh data every step, 4 seeds; a42/, a42_scores.txt)
Flat start 100 in all four seeds. Validation A / C / total at 2,500: 0.0012-0.0017 / 0.0038-0.0060 / 0.0084-0.0109 bits (fixed 1M traces, A-34: 0.0021-0.0025 / 0.0130-0.0148 / 0.0202-0.0207).
Predictions. (1) C at most 0.008 in at least 3 of 4: HIT (4 of 4). (2) flat start at most 200: HIT (100).
Reading. The sine x4 network's larger same-weight asymmetry (part C) in A-34 was overfitting the fixed training traces: with fresh data its C falls to 0.004-0.006 bits, below the fixed-data ReLU values, and its total excess halves (about 0.010 bits). The fast parity escape does not depend on the data regime.

### A-41b result (sigma 0.6, 4 seeds per arm, 4,000 steps; a41b/, a41b_score.py -> a41b_scores.txt; note: the ReLU curves file is named A-39-relu-w1.0-sig0.6 by the script's naming rule)
A_ref (median ReLU A at step 100) = 0.00666 bits. ReLU: flat start 2,300, 2,000, none by 4,000, 2,100 (median 2,200); validation A at 4,000 0.0048, 0.0041, 0.0061, 0.0042. Sine x4: flat start 200 in all four (the earliest the rule allows); A at 4,000 0.0011, 0.0012, 0.0006, 0.0008.
Predictions. (1) ReLU median at least 2,000: HIT (2,200). (2) sine x4 median at most half of ReLU's and lower A in all 4 paired seeds: HIT (200; 4 of 4).
Reading. At a weaker parity signal (ripple amplitude about a third of that at sigma 0.5) the ReLU flat start roughly doubles (2,200 against 1,250 steps), while the sine x4 network still escapes almost at once and ends with 4-7 times less parity error. The A-34 result is not specific to sigma 0.5.

## A-43 (CONFIRMATORY; link to the adder project C-169/C-170): does LayerNorm change XOR's flat start?
Why: in the adder, per-example normalisation of the hidden layers removes the joint delay, because part of the competing code lives in activation magnitude (C-169, C-170). The XOR flat start has a different cause (one-hot label noise hiding a weak parity signal that a ReLU network must build as an oscillation; A-30, A-34). If the two delays are different mechanisms, LayerNorm should not remove the XOR one.
Design. A-15 configuration, LayerNorm (with gain and bias) before each ReLU, seeds 0-3, 2,500 steps; flat start plateau-referenced (A below 0.9 x 0.0353); output a43/.
Prediction (written before running). Median flat start at least 900 (the default-ReLU range is 900-1,300; LayerNorm does not remove the delay).

### A-35b result (seeds 2-3; appended to a35/move.csv, a35b.log)
Layer-1 relative change at step 1,000: CO = 1: 0.239, 0.229; CO = 16: 0.152, 0.122 (ratios 0.64, 0.53). Prediction (at most 0.67 in both new seeds): HIT. Over four seeds the ratio is 0.53-0.64; layers 2 and 3 move 0.56-0.69 times as far (step 1,000).

### A-43 result (XOR with LayerNorm, 4 seeds, 2,500 steps; a43/, a31_score.py a43 2500 -> a43_scores.txt)
Flat start 2,400, 2,000, 2,000, 2,100 (median 2,050; default ReLU 1,250). Validation A at 2,500: 0.0267-0.0309 (default 0.0138-0.0181).
Prediction (median at least 900, LayerNorm does not remove the delay): HIT; it is in fact 1.6 times longer.
Reading. The fix that removes the adder's delay does nothing for XOR's flat start (it makes it longer), as expected if the two delays have different causes: the adder's comes from the other outputs taking over shared layers through a code partly carried by activation magnitude, which LayerNorm removes; XOR's comes from label noise hiding a weak parity signal that the ReLU network must build as an oscillation (A-30, A-34), which LayerNorm does not supply. Why LayerNorm lengthens the XOR flat start is not tested.

## A-44 (check): A-39 to A-43 scores recomputed from the raw curve files (a44_verify.py -> a44_verify.txt)
All four scoring outputs match the saved score files (A-39 and A-43 files have per-seed curve lines appended by hand after the scorer's lines; the scorer's lines match). The A-41b ReLU/sine values and the A-42 numbers in PROGRESS.md were taken from these outputs. No corrections.
