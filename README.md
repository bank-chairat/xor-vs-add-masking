### XOR vs. Addition Masking: Less Information, or Harder to Learn? 

A simulated study of why a neural network attacker learns one kind of masked secret more slowly than another, written up in the order the work was done. Each chapter says what question we asked, where the idea came from, which files answer it, what we found, and what it led to next.

**Scope, up front:** everything here is simulated. The leakage model is mostly one simple model (Hamming weight plus Gaussian noise), the networks are small MLPs, and nothing is measured on a real device. Results are claims about this model only.

---

### Motivation and question

Masking is a standard defence against side-channel attacks, where an attacker learns a secret from a device's power use or electromagnetic emissions. The device never handles the secret s directly. It splits it into two random-looking pieces (shares), so that neither piece alone says anything about `s`. Two common ways to split a byte:
- **Boolean masking (XOR):** shares `m` and `s XOR m`
- **Arithmetic masking (addition):** shares `m` and `(s + m) mod N`

Deep-learning attacks train a neural network to read the secret from the leakage of both shares together (for example, MLPs and CNNs on a masked AES implementation in the ASCAD study, Prouff et al. 2018; see literature.md).

Suppose a network finds one masking harder to attack. There are two very different reasons that could be:
1. **Information:** the measurements of that masking reveal less about the secret. Then it is genuinely better protection.
2. **Learnability:** the measurements carry the same or more information, but the rule linking them to the secret is harder for a network to learn. Then a network that fails says more about its training than about the protection.

These need different evidence, so the project separates them: first compute exactly how much information is available, a ceiling no attacker can beat, then measure how close trained networks get to it.
