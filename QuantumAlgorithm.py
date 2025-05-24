# Unit-IV Introduction to Quantum Algorithm: Probabilistic versus Quantum Algorithm, Phase Kick- Back Algorithm, The Deutsch Algorithm, The Deutsch-Jozsa Algorithm, Simon’s Algorithm.

import numpy as np
import math

# Utility: Hadamard gate for 1 qubit
H = (1/ math.sqrt(2)) * np.array([[1, 1],
                                 [1, -1]])

# Pauli-X gate
X = np.array([[0,1],
              [1,0]])

# Tensor product of multiple matrices
def tensor(*matrices):
    result = np.array([[1]])
    for m in matrices:
        result = np.kron(result, m)
    return result

# Measure computational basis
def measure(state):
    probabilities = np.abs(state) ** 2
    outcome = np.random.choice(len(state), p=probabilities)
    return outcome

# ------------------------------
# 1. Probabilistic vs Quantum Algorithm (simple example)
# ------------------------------
print("== Probabilistic vs Quantum Algorithm ==")
def probabilistic_oracle(input_bit):
    # Returns 0 or 1 with 50% probability regardless of input (probabilistic)
    return np.random.choice([0,1])

def quantum_oracle(bit):
    # Deterministic oracle returning bit itself
    return bit

print("Probabilistic oracle output on input 0:", probabilistic_oracle(0))
print("Quantum oracle output on input 0:", quantum_oracle(0))
print()

# ------------------------------
# 2. Phase Kickback Concept
# ------------------------------

def phase_kickback_example():
    # Simple phase kickback using controlled-Z operation
    # Control qubit in |+> = H|0>
    control = H @ np.array([1,0])
    # Target qubit in |1>
    target = np.array([0,1])
    
    # Controlled-Z matrix (4x4)
    CZ = np.eye(4)
    CZ[3,3] = -1  # phase flip on |11>

    # Combine control and target into joint state
    joint_state = np.kron(control, target)
    
    # Apply CZ
    new_state = CZ @ joint_state
    print("Before CZ:", joint_state)
    print("After CZ (phase kickback):", new_state)

phase_kickback_example()
print()

# ------------------------------
# 3. Deutsch Algorithm
# ------------------------------

def deutsch_algorithm(oracle):
    # 1 qubit input + 1 qubit output
    # Initialize |0>|1>
    state = np.array([0,1,0,0])  # |0>|1>

    # Apply H to both qubits
    H2 = tensor(H, H)
    state = H2 @ state

    # Apply oracle U_f
    state = oracle @ state

    # Apply H to first qubit
    H1_I = tensor(H, np.eye(2))
    state = H1_I @ state

    # Measure first qubit (indices 0,1 and 2,3 grouped)
    p0 = np.abs(state[0])**2 + np.abs(state[1])**2
    p1 = np.abs(state[2])**2 + np.abs(state[3])**2
    print(f"Measurement probabilities for first qubit: |0>={p0:.2f}, |1>={p1:.2f}")
    # If first qubit is 0 → function is constant, else balanced
    if p0 > p1:
        print("Function is constant")
    else:
        print("Function is balanced")

# Define oracle matrices for f(x)=0 (constant) and f(x)=x (balanced)

# U_f for f(x) = 0: flips second qubit if f(x)=1 (never here)
Uf_const = np.eye(4)

# U_f for f(x) = x: flips second qubit if x=1
Uf_balanced = np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,0,1],
                        [0,0,1,0]])

print("== Deutsch Algorithm with constant function f(x)=0 ==")
deutsch_algorithm(Uf_const)
print("\n== Deutsch Algorithm with balanced function f(x)=x ==")
deutsch_algorithm(Uf_balanced)
print()

# ------------------------------
# 4. Deutsch-Jozsa Algorithm
# ------------------------------

def deutsch_joza_algorithm(oracle, n):
    """
    n input qubits, 1 output qubit.
    Oracle U_f acts on (n+1) qubits.
    """
    # Initialize |0>^n |1>
    dim = 2 ** (n+1)
    state = np.zeros(dim)
    state[2**0 - 1] = 1  # The last qubit is |1>, but index is tricky

    # Actually the state vector with last qubit |1>:
    # The last qubit is least significant bit in index.
    # So state index with last qubit 1 and first n qubits 0 is index 1.
    state = np.zeros(dim)
    state[1] = 1  # |0...0>|1>

    # Apply Hadamard on first n+1 qubits
    Hn = tensor(*([H]*(n+1)))
    state = Hn @ state

    # Apply oracle
    state = oracle @ state

    # Apply Hadamard on first n qubits
    Hn_I = tensor(*([H]*n), np.eye(2))
    state = Hn_I @ state

    # Measure first n qubits
    probs = np.zeros(2**n)
    for i in range(2**(n+1)):
        # Extract first n qubits bits:
        prefix = i >> 1
        probs[prefix] += abs(state[i])**2

    print("Measurement probabilities of first n qubits:")
    for i, p in enumerate(probs):
        print(f"|{format(i,'0'+str(n)+'b')}>: {p:.3f}")

    # If measurement outcome is |0...0>, function is constant; else balanced
    if probs[0] > 0.9:
        print("Function is constant")
    else:
        print("Function is balanced")

# Example for n=2 qubits

# Oracle for constant f=0
dim = 2**3
Uf_const = np.eye(dim)

# Oracle for balanced f(x) = parity(x)
Uf_balanced = np.eye(dim)
for i in range(dim):
    x = i >> 1
    y = i & 1
    f = bin(x).count("1") % 2  # parity
    if f == 1:
        # flip output qubit (least significant bit)
        flipped_i = i ^ 1
        Uf_balanced[i, i] = 0
        Uf_balanced[flipped_i, i] = 1

print("== Deutsch-Jozsa Algorithm with constant function ==")
deutsch_joza_algorithm(Uf_const, n=2)
print("\n== Deutsch-Jozsa Algorithm with balanced function ==")
deutsch_joza_algorithm(Uf_balanced, n=2)
print()

# ------------------------------
# 5. Simon's Algorithm (Simplified Demo)
# ------------------------------

def simon_algorithm(f, n):
    """
    f: function from n-bit to n-bit strings with period s
    n: number of bits
    We do not simulate full quantum states here due to complexity.
    Instead, show conceptual steps.
    """

    print("Simon's Algorithm Conceptual Demo")
    print(f"Function f with secret string s exists.")

    # 1. Prepare n qubit input in superposition |+>^n
    print("Prepare |+>^n state")

    # 2. Query oracle to get f(x)
    print("Query oracle in superposition")

    # 3. Measure second register -> collapses input superposition to subset satisfying f(x) = y
    print("Measure second register")

    # 4. Apply Hadamard on first register and measure to get bitstring z s.t. z.s = 0 (mod 2)")
    print("Apply Hadamard and measure to get z")

    # 5. Repeat O(n) times to get enough equations to find s classically.

    print("Classical step: solve linear system modulo 2 for s.")

# Example use:
def f_example(x):
    # f(x) = f(x xor s)
    # Let's say s= '11' (2 bits)
    s = 0b11
    return x ^ s

print("== Simon's Algorithm ==")
simon_algorithm(f_example, n=2)
