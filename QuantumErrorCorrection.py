# Unit-VII Introduction to quantum error Correction: Classical Error correction, Fault tolerance,
Quantum Error Correction, Fault Tolerance Quantum computation.

import numpy as np
import random

# ------------------------------
# 1. Classical Error Correction (3-bit repetition code)
# ------------------------------
print("=== Classical Error Correction (3-bit Repetition) ===")

def encode_classical(bit):
    return [bit] * 3

def introduce_noise(bits, flip_prob=0.2):
    return [b ^ (1 if random.random() < flip_prob else 0) for b in bits]

def decode_classical(bits):
    return int(sum(bits) > 1)

# Encode, corrupt, decode
original_bit = 1
encoded = encode_classical(original_bit)
noisy = introduce_noise(encoded)
decoded = decode_classical(noisy)

print(f"Original bit: {original_bit}")
print(f"Encoded: {encoded}")
print(f"Noisy:   {noisy}")
print(f"Decoded: {decoded}\n")

# ------------------------------
# 2. Quantum Error Correction (3-qubit bit-flip code)
# ------------------------------
print("=== Quantum Error Correction (3-Qubit Bit-Flip Code) ===")

def apply_bit_flip(qubit, flip=False):
    return qubit if not flip else np.array([qubit[1], qubit[0]])

# |0> = [1, 0], |1> = [0, 1]
zero = np.array([1, 0])
one = np.array([0, 1])

def encode_quantum_logical_0():
    return np.kron(np.kron(zero, zero), zero)

def encode_quantum_logical_1():
    return np.kron(np.kron(one, one), one)

def flip_qubit(state, qubit_index):
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    ops = [I, I, I]
    ops[qubit_index] = X
    return tensor(*ops) @ state

# Simulate a bit-flip on one qubit
logical_0 = encode_quantum_logical_0()
flipped = flip_qubit(logical_0, qubit_index=1)  # flip 2nd qubit

print("Encoded logical |0⟩ (3 qubits):", logical_0)
print("State after bit-flip on qubit 1:", flipped)

print("\n(Decoding/measurement is not done here; conceptually handled by syndrome detection.)")

# ------------------------------
# 3. Fault Tolerance Conceptual Illustration
# ------------------------------
print("\n=== Fault Tolerance Concept ===")
print("""
- Fault tolerance ensures that errors do not propagate uncontrollably.
- Gates and measurements are designed so single errors don't corrupt logical information.
- Quantum circuits use ancilla qubits and verification steps.
- Example: Majority voting, stabilizer codes, and syndrome measurement.
""")

# ------------------------------
# Summary
# ------------------------------
print("=== Summary ===")
print("""
1. Classical error correction uses redundancy (e.g., 3-bit repetition code).
2. Quantum error correction must preserve coherence and uses entanglement (e.g., 3-qubit bit-flip code).
3. Fault-tolerant quantum computing protects logical qubits from physical errors with layered codes and careful circuit design.
""")
