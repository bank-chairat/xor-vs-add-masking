### XOR vs. Addition Masking: Less Information, or Harder to Learn? 

A simulated study of why a neural network attacker learns one kind of masked secret more slowly than another, written up in the order the work was done. Each chapter says what question we asked, where the idea came from, which files answer it, what we found, and what it led to next.

**Scope, up front:** everything here is simulated. The leakage model is mostly one simple model (Hamming weight plus Gaussian noise), the networks are small MLPs, and nothing is measured on a real device. Results are claims about this model only.

---

### Motivation and question

Masking is a standard defence against side-channel attacks, where an attacker learns a secret from a device's power use or electromagnetic emissions. The device never handles the secret s directly. It splits it into two random-looking pieces (shares), so that neither piece alone says anything about `s`. Two common ways to split a byte:
- Boolean masking (XOR): shares m and s XOR m
- Arithmetic masking (addition): shares m and (s + m) mod N
