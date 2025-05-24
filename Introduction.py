# Unit-I Introduction & Back ground: A brief history of computing including ‘Turing Machines’, Probabilistic systems, Quantum Systems.

import random
import math
import cmath

# ------------------------------
# 1. Deterministic Turing Machine (very simple)
# ------------------------------

class SimpleTuringMachine:
    def __init__(self, tape="", blank_symbol="_", initial_state="q0", final_states=None):
        self.tape = list(tape)
        self.blank_symbol = blank_symbol
        self.head = 0
        self.state = initial_state
        self.final_states = final_states if final_states else {"qf"}

        # Example: increment binary number by 1
        self.transitions = {
            ("q0", "1"): ("q0", "1", 1),   # move right on 1
            ("q0", "0"): ("q0", "0", 1),   # move right on 0
            ("q0", "_"): ("q1", "_", -1),  # hit blank, move left
            ("q1", "1"): ("q1", "0", -1),  # change 1->0, move left
            ("q1", "0"): ("qf", "1", 0),   # change 0->1, halt
            ("q1", "_"): ("qf", "1", 0),   # all ones case, add 1 at front
        }

    def step(self):
        symbol = self.tape[self.head] if self.head < len(self.tape) else self.blank_symbol
        if (self.state, symbol) not in self.transitions:
            # Halt if no rule
            self.state = "qf"
            return
        new_state, write_symbol, direction = self.transitions[(self.state, symbol)]

        # Write
        if self.head < len(self.tape):
            self.tape[self.head] = write_symbol
        else:
            self.tape.append(write_symbol)

        # Move head
        self.head += direction
        if self.head < 0:
            self.tape.insert(0, self.blank_symbol)
            self.head = 0
        self.state = new_state

    def run(self):
        while self.state not in self.final_states:
            self.step()

        return "".join(self.tape).strip(self.blank_symbol)

print("== Deterministic Turing Machine: Increment Binary ==")
tm = SimpleTuringMachine(tape="1101")
print("Initial tape:", "1101")
result = tm.run()
print("Final tape after increment:", result)
print()


# ------------------------------
# 2. Probabilistic System (Coin toss)
# ------------------------------

def coin_toss():
    # 50-50 probabilistic system
    return "Heads" if random.random() < 0.5 else "Tails"

print("== Probabilistic System: Coin Toss ==")
for i in range(5):
    print(f"Toss {i+1}:", coin_toss())
print()


# ------------------------------
# 3. Quantum System (Qubit)
# ------------------------------

class Qubit:
    def __init__(self, alpha=1, beta=0):
        # alpha and beta are complex amplitudes for |0> and |1>
        # Normalization: |alpha|^2 + |beta|^2 = 1
        norm = math.sqrt(abs(alpha)**2 + abs(beta)**2)
        self.alpha = alpha / norm
        self.beta = beta / norm

    def measure(self):
        # Probability of 0 and 1
        p0 = abs(self.alpha)**2
        p1 = abs(self.beta)**2
        r = random.random()
        return 0 if r < p0 else 1

    def __str__(self):
        return f"{self.alpha:.2f}|0> + {self.beta:.2f}|1>"

# Example: create a qubit in superposition state |+> = (|0> + |1>) / sqrt(2)
alpha = 1 / math.sqrt(2)
beta = 1 / math.sqrt(2)
q = Qubit(alpha, beta)

print("== Quantum System: Qubit Superposition and Measurement ==")
print("Qubit state:", q)
measurements = [q.measure() for _ in range(10)]
print("Measurements (0 means |0>, 1 means |1>):", measurements)
