# Unit-V Fast Factorization & Search Algorithms: Quantum Fourier Transform, Shor’s Algorithm, The correctness Probability. Grover’s Search Algorithm.

import numpy as np
import math

# --- Tensor product utility ---
def tensor(*matrices):
    result = np.array([[1]])
    for m in matrices:
        result = np.kron(result, m)
    return result

# --- 1. Quantum Fourier Transform (QFT) ---
def qft_matrix(n):
    N = 2 ** n
    omega = np.exp(2j * np.pi / N)
    Q = np.zeros((N, N), dtype=complex)
    for j in range(N):
        for k in range(N):
            Q[j, k] = omega ** (j * k)
    return Q / np.sqrt(N)

def apply_qft(state):
    n = int(np.log2(len(state)))
    Q = qft_matrix(n)
    return Q @ state

print("=== Quantum Fourier Transform ===")
n_qubits = 3
N = 2 ** n_qubits
state = np.zeros(N)
state[5] = 1  # Basis state |5>
print("Input state |5>:", state)
qft_state = apply_qft(state)
print("State after QFT:\n", np.round(qft_state, 3))
print()

# --- 2. Shor's Algorithm (Order Finding) Conceptual Demo ---
def shor_order_finding(a, N):
    print(f"Shor's Algorithm (Order Finding) for a={a}, N={N}:")
    print("- Prepare superposition over |x>")
    print("- Compute |x>|a^x mod N>")
    print("- Apply QFT to 1st register")
    print("- Measure 1st register to obtain info on order r")
    print("- Use classical post-processing to find factors of N\n")

print("=== Shor's Algorithm Outline ===")
shor_order_finding(a=7, N=15)

# --- 3. Correctness Probability ---
def calc_correctness_probability(trials, success_prob_per_trial):
    p_fail_all = (1 - success_prob_per_trial) ** trials
    p_success = 1 - p_fail_all
    print(f"After {trials} trials with success probability {success_prob_per_trial} per trial:")
    print(f"Probability of at least one success = {p_success:.4f}\n")

print("=== Correctness Probability ===")
calc_correctness_probability(trials=5, success_prob_per_trial=0.4)

# --- 4. Grover's Search Algorithm ---
def grover_iteration(state, oracle, diffusion):
    state = oracle @ state
    state = diffusion @ state
    return state

def grover_search(n_qubits, marked_index):
    N = 2 ** n_qubits
    print(f"Grover's Search on {N} items, marked index = {marked_index}")

    # Initial equal superposition state |s>
    state = np.ones(N) / np.sqrt(N)

    # Oracle: flip phase of marked element
    oracle = np.eye(N)
    oracle[marked_index, marked_index] = -1

    # Diffusion operator = 2|s><s| - I
    s_matrix = np.ones((N, N)) / N
    diffusion = 2 * s_matrix - np.eye(N)

    iterations = int(np.floor((np.pi / 4) * np.sqrt(N)))
    print(f"Number of Grover iterations: {iterations}")

    for _ in range(iterations):
        state = grover_iteration(state, oracle, diffusion)

    probs = np.abs(state) ** 2
    for i, p in enumerate(probs):
        print(f"State |{format(i, '0' + str(n_qubits) + 'b')}>: probability = {p:.3f}")
    print(f"Max probability at index {np.argmax(probs)} (expected {marked_index})\n")

print("=== Grover's Search Algorithm ===")
grover_search(n_qubits=3, marked_index=5)
