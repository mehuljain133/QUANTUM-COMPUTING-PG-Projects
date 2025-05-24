# Unit-III Quantum Circuits: Boolean Circuits, Reversible Circuits, Quantum Circuit Model, Quantum Gates, Universal Sets of Quantum Gates, Efficiency in approximating unitary transformation, Implementing measurements with Quantum Gates.

import numpy as np
import math

# Utility: Tensor product of matrices
def tensor(*matrices):
    result = np.array([[1]])
    for m in matrices:
        result = np.kron(result, m)
    return result

# ------------------------------
# 1. Boolean Circuits (classical gates)
# ------------------------------

def AND(a, b):
    return a & b

def OR(a, b):
    return a | b

def NOT(a):
    return 1 - a

print("== Boolean Circuits ==")
print("AND(1,0) =", AND(1,0))
print("OR(1,0) =", OR(1,0))
print("NOT(1) =", NOT(1))
print()

# ------------------------------
# 2. Reversible Circuits: Toffoli Gate (CCNOT)
# ------------------------------

# 3-qubit Toffoli gate matrix (8x8)
def toffoli_gate():
    T = np.eye(8)
    T[6,6] = 0
    T[7,7] = 0
    T[6,7] = 1
    T[7,6] = 1
    return T

print("== Reversible Circuits ==")
T = toffoli_gate()
print("Toffoli gate matrix shape:", T.shape)
print()

# ------------------------------
# 3. Quantum Gates
# ------------------------------

# Pauli-X (NOT) gate
X = np.array([[0, 1],
              [1, 0]])

# Hadamard gate
H = (1 / math.sqrt(2)) * np.array([[1, 1],
                                   [1, -1]])

# Phase gate (T gate)
T = np.array([[1, 0],
              [0, complex(math.cos(math.pi/4), math.sin(math.pi/4))]])

# CNOT gate (2 qubits)
CNOT = np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,0,1],
                 [0,0,1,0]])

print("== Quantum Gates ==")
print("Pauli-X gate:\n", X)
print("Hadamard gate:\n", H)
print("T gate:\n", T)
print("CNOT gate:\n", CNOT)
print()

# ------------------------------
# 4. Universal Sets of Quantum Gates
# ------------------------------

# The set {H, T, CNOT} is universal for quantum computation.

# ------------------------------
# 5. Efficiency in Approximating Unitary Transformations (conceptual demo)
# ------------------------------

def approximate_rotation(theta, epsilon=1e-3):
    """
    Approximate rotation around Z-axis by theta using T-gates.
    For demo: returns the number of T-gates needed roughly.
    """
    # Solovay-Kitaev theorem guarantees polylog(1/epsilon) complexity
    # This is just a dummy function representing efficiency.
    steps = int(math.ceil(math.log2(1/epsilon)))
    print(f"Approximate rotation by {theta:.2f} radians with accuracy {epsilon} using ~{steps} T-gates.")
    return steps

approximate_rotation(math.pi/8)
print()

# ------------------------------
# 6. Implementing Measurements with Quantum Gates (conceptual)
# ------------------------------

def measure_in_computational_basis(state_vector):
    """
    Simulate measurement of a single qubit state vector in computational basis.
    """
    probabilities = np.abs(state_vector) ** 2
    outcome = np.random.choice(len(state_vector), p=probabilities)
    print(f"Measurement outcome: |{outcome}> with probability {probabilities[outcome]:.2f}")
    # Post-measurement state collapses to outcome state
    collapsed_state = np.zeros_like(state_vector)
    collapsed_state[outcome] = 1
    return outcome, collapsed_state

# Example: apply H gate to |0> then measure
initial_state = np.array([1,0])  # |0>
after_H = H @ initial_state

print("== Measurement Implementation with Quantum Gates ==")
print("State after Hadamard on |0>:", after_H)
outcome, collapsed = measure_in_computational_basis(after_H)
print("State after measurement collapse:", collapsed)
print()

# ------------------------------
# Demo: Building a small quantum circuit combining gates
# ------------------------------

def apply_cnot(control_qubit, target_qubit, num_qubits, state):
    """
    Apply CNOT to a quantum state vector.
    control_qubit, target_qubit: indices (0-based) of qubits in register (num_qubits)
    """
    dim = 2 ** num_qubits
    new_state = np.zeros_like(state, dtype=complex)
    for i in range(dim):
        # Check if control bit is 1
        control_bit = (i >> (num_qubits - control_qubit -1)) & 1
        if control_bit == 1:
            # Flip target bit
            flipped_i = i ^ (1 << (num_qubits - target_qubit -1))
            new_state[flipped_i] += state[i]
        else:
            new_state[i] += state[i]
    return new_state

print("== Quantum Circuit Example ==")
num_qubits = 2
state = np.zeros(2**num_qubits)
state[0] = 1  # |00>
print("Initial state |00>:", state)

# Apply Hadamard on qubit 0
H1 = tensor(H, np.eye(2))
state = H1 @ state
print("After Hadamard on qubit 0:", state)

# Apply CNOT with qubit 0 control, qubit 1 target
state = apply_cnot(control_qubit=0, target_qubit=1, num_qubits=num_qubits, state=state)
print("After CNOT(0->1):", state)

# Measure qubits (simulate)
probabilities = np.abs(state) ** 2
print("Probabilities of states:")
for i, p in enumerate(probabilities):
    print(f"|{format(i, '02b')}>: {p:.2f}")
