# Unit-II QUBITS and Framework of Quantum Systems: The State of a Quantum System, Quantum Bits, Quantum Registers, Quantum information, Quantum Turing Machine.

import math
import random
import numpy as np

# ------------------------------
# 1. Quantum State (Ket Vector)
# ------------------------------

class QuantumState:
    def __init__(self, amplitudes):
        # amplitudes: list or np.array of complex numbers representing the state vector
        self.amplitudes = np.array(amplitudes, dtype=complex)
        self.normalize()

    def normalize(self):
        norm = np.linalg.norm(self.amplitudes)
        if norm == 0:
            raise ValueError("Zero vector is not a valid quantum state")
        self.amplitudes /= norm

    def measure(self):
        # Probabilities are square of amplitudes magnitude
        probabilities = np.abs(self.amplitudes) ** 2
        # Choose index based on probability distribution
        return np.random.choice(len(self.amplitudes), p=probabilities)

    def __str__(self):
        terms = []
        for i, amp in enumerate(self.amplitudes):
            if abs(amp) > 1e-10:
                terms.append(f"({amp:.2f})|{format(i, f'0{int(math.log2(len(self.amplitudes)))}b')}>")
        return " + ".join(terms)


# ------------------------------
# 2. Qubit Class
# ------------------------------

class Qubit:
    def __init__(self, alpha=1, beta=0):
        # alpha|0> + beta|1>, normalized
        self.state = QuantumState([alpha, beta])

    def measure(self):
        return self.state.measure()

    def __str__(self):
        return str(self.state)

# ------------------------------
# 3. Quantum Register (multiple qubits)
# ------------------------------

class QuantumRegister:
    def __init__(self, num_qubits):
        # Start in |0...0> state
        dim = 2 ** num_qubits
        amplitudes = np.zeros(dim, dtype=complex)
        amplitudes[0] = 1.0  # |00..0> initial state
        self.state = QuantumState(amplitudes)
        self.num_qubits = num_qubits

    def apply_hadamard_to_qubit(self, qubit_index):
        # Apply Hadamard gate to one qubit in the register
        H = (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]])
        new_amplitudes = np.zeros_like(self.state.amplitudes)
        for i in range(len(self.state.amplitudes)):
            # Bit value of target qubit
            bit = (i >> (self.num_qubits - qubit_index - 1)) & 1
            for b in [0, 1]:
                # Flip target bit to b
                j = (i & ~(1 << (self.num_qubits - qubit_index - 1))) | (b << (self.num_qubits - qubit_index - 1))
                new_amplitudes[j] += H[b, bit] * self.state.amplitudes[i]
        self.state.amplitudes = new_amplitudes
        self.state.normalize()

    def measure(self):
        return self.state.measure()

    def __str__(self):
        return str(self.state)

# ------------------------------
# 4. Quantum Turing Machine (Conceptual and simplified)
# ------------------------------

class QuantumTuringMachine:
    """
    Simplified conceptual model: 
    Tape is a quantum register of qubits, operations are unitary gates applied on qubits.
    Head moves and measurement define halting.
    """

    def __init__(self, num_qubits):
        self.register = QuantumRegister(num_qubits)
        self.head = 0
        self.halted = False

    def step(self):
        if self.halted:
            return
        # Example operation: apply Hadamard on current head qubit
        print(f"Applying Hadamard to qubit at position {self.head}")
        self.register.apply_hadamard_to_qubit(self.head)

        # Measure qubit at head position, if result is 1 halt (simulate)
        outcome = self.measure_qubit(self.head)
        print(f"Measurement at head: {outcome}")
        if outcome == 1:
            print("Halting condition met.")
            self.halted = True
        else:
            # Move head to next qubit, wrap around
            self.head = (self.head + 1) % self.register.num_qubits

    def measure_qubit(self, qubit_index):
        # Partial measurement of single qubit from full register state (simplified)
        # We measure entire state and extract bit from result
        full_measurement = self.register.measure()
        # Extract the bit corresponding to qubit_index from full_measurement
        bit = (full_measurement >> (self.register.num_qubits - qubit_index - 1)) & 1
        return bit

    def run(self, max_steps=10):
        step_count = 0
        while not self.halted and step_count < max_steps:
            self.step()
            step_count += 1
        print("Final quantum register state:")
        print(self.register)

# ------------------------------
# Demo: Put it all together
# ------------------------------

print("=== Qubit Example ===")
q = Qubit(1/math.sqrt(2), 1/math.sqrt(2))  # |+> state
print("Qubit state:", q)
print("Measurement outcomes:", [q.measure() for _ in range(5)])

print("\n=== Quantum Register Example (3 qubits) ===")
qr = QuantumRegister(3)
print("Initial state:", qr)
qr.apply_hadamard_to_qubit(0)
qr.apply_hadamard_to_qubit(1)
print("After Hadamard on qubits 0 and 1:", qr)
print("Measurement of register:", qr.measure())

print("\n=== Quantum Turing Machine Example ===")
qtm = QuantumTuringMachine(num_qubits=3)
qtm.run(max_steps=6)
