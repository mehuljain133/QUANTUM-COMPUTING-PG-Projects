# Unit-VI Computational lower bound complexity for quantum circuits: General ideas, Polynomial representations, Quantum Circuit Lower Bound.

import numpy as np

# ------------------------------
# 1. General Ideas: Complexity & Lower Bounds
# ------------------------------
print("=== General Ideas on Quantum Circuit Complexity ===")
print("""
- Quantum circuit complexity measures minimum resources (gates, depth) needed to compute functions.
- Lower bounds prove some functions require at least a certain complexity, even on quantum computers.
- Tools include polynomial degree of Boolean functions and adversary arguments.
""")

# ------------------------------
# 2. Polynomial Representation of Boolean Functions
# ------------------------------
print("\n=== Polynomial Representation of Boolean Functions ===")

def boolean_to_polynomial(f_values):
    """
    Given truth table of f(x) for n bits, compute multilinear polynomial coefficients.
    Using Fourier expansion over {0,1}^n.
    For simplicity, demonstrate for n=1 or n=2.

    Returns polynomial coefficients indexed by subsets of variables.
    """
    n = int(np.log2(len(f_values)))
    # For n=1:
    if n == 1:
        # f(0), f(1)
        a0 = f_values[0]
        a1 = f_values[1] - f_values[0]
        print(f"Polynomial representation: f(x) = {a0} + {a1} * x")
    elif n == 2:
        # f(00), f(01), f(10), f(11)
        f00, f01, f10, f11 = f_values
        a0 = f00
        a1 = f10 - f00
        a2 = f01 - f00
        a12 = f11 - f10 - f01 + f00
        print(f"Polynomial representation:")
        print(f"f(x1,x2) = {a0} + {a1}*x1 + {a2}*x2 + {a12}*x1*x2")
    else:
        print("Polynomial representation for n>2 is not implemented here.")

# Example: AND function on 2 bits
and_truth = [0, 0, 0, 1]
boolean_to_polynomial(and_truth)

# ------------------------------
# 3. Quantum Circuit Lower Bound (Conceptual)
# ------------------------------
print("\n=== Quantum Circuit Lower Bound (Conceptual) ===")
print("""
- Some functions like parity require linear depth quantum circuits.
- Polynomial degree of Boolean function lower bounds quantum query complexity.
- For example, parity has degree n, so quantum algorithms need at least n queries.
- Demonstrating by comparing polynomial degree and circuit depth.
""")

def parity_function(x_bits):
    return sum(x_bits) % 2

def parity_polynomial_degree(n):
    """Degree of parity polynomial is n."""
    return n

def simulate_simple_circuit_depth(f_degree):
    """
    Simplified assumption:
    Circuit depth >= polynomial degree of function.
    """
    return f_degree

n = 4
degree = parity_polynomial_degree(n)
depth = simulate_simple_circuit_depth(degree)
print(f"For parity function on {n} bits:")
print(f"Polynomial degree = {degree}")
print(f"Minimum quantum circuit depth (lower bound) >= {depth}")

# ------------------------------
# Summary
# ------------------------------
print("\nSummary:")
print("""
- Polynomial degree provides lower bounds on quantum query complexity.
- Quantum circuits computing certain functions need at least as many gates/depth as degree.
- Lower bounds guide quantum algorithm design and limitations.
""")
