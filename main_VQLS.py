# Pennylane
import pennylane as qml
from pennylane import numpy as np

# Plotting
import matplotlib.pyplot as plt

#
# Setting of the main hyper-parameters of the model
#

n_qubits = 3  # Number of system qubits.
n_shots = 10 ** 6  # Number of quantum measurements.
tot_qubits = n_qubits + 1  # Addition of an ancillary qubit.
ancilla_idx = n_qubits  # Index of the ancillary qubit (last position).
steps = 100  # Number of optimization steps (increased)
eta = 0.8  # Learning rate (reduced for better convergence)
q_delta = 0.01  # Initial spread of random quantum weights (increased)
rng_seed = 0  # Seed for random number generator


#
# Definition of the matrix components and coefficients
#

# Define the base matrices A_0, A_1, A_2
def get_base_matrices():
    """Returns the three base matrices A_0, A_1, A_2 for the decomposition A = c_0*A_0 + c_1*A_1 + c_2*A_2"""
    Id = np.identity(2)
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])

    A_0 = np.identity(2 ** n_qubits)  # Identity matrix
    A_1 = np.kron(np.kron(X, Z), Id)  # X ⊗ Z ⊗ I
    A_2 = np.kron(np.kron(X, Id), Id)  # X ⊗ I ⊗ I

    return A_0, A_1, A_2


def construct_matrix_A(coefficients):
    """Constructs the matrix A from the linear combination A = c_0*A_0 + c_1*A_1 + c_2*A_2"""
    A_0, A_1, A_2 = get_base_matrices()
    A = coefficients[0] * A_0 + coefficients[1] * A_1 + coefficients[2] * A_2
    return A


# Coefficients of the linear combination A = c_0 A_0 + c_1 A_1 + c_2 A_2
# You can change these values to test different matrices
c = np.array([1.0, 0.5, 0.3])  # Changed from [1.0, 0.2, 0.2] to test different coefficients

# Define the b vector - you can change this to any vector
b = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
b = b / np.linalg.norm(b)  # Normalize b

# Construct the matrix A using the coefficients
A_matrix = construct_matrix_A(c)

print("Coefficients c =", c)
print("Matrix A =")
print(A_matrix)
print("Vector b =", b)
print()


#
# Circuits of the quantum linear problem
#

def U_b():
    """Unitary matrix rotating the ground state to the problem vector |b> = U_b |0>."""
    # Use state preparation to create the quantum state corresponding to vector b
    qml.StatePrep(b, wires=range(n_qubits))


def CA(idx):
    """Controlled versions of the unitary components A_l of the problem matrix A."""
    if idx == 0:
        # Identity operation (A_0 = I)
        pass

    elif idx == 1:
        # Controlled version of A_1 = X ⊗ Z ⊗ I
        qml.CNOT(wires=[ancilla_idx, 0])  # Controlled-X on qubit 0
        qml.CZ(wires=[ancilla_idx, 1])  # Controlled-Z on qubit 1

    elif idx == 2:
        # Controlled version of A_2 = X ⊗ I ⊗ I
        qml.CNOT(wires=[ancilla_idx, 0])  # Controlled-X on qubit 0


#
# Variational quantum circuit
#

def variational_block(weights):
    """Variational circuit mapping the ground state |0> to the ansatz state |x>."""
    # We first prepare an equal superposition of all the states of the computational basis.
    for idx in range(n_qubits):
        qml.Hadamard(wires=idx)

    # Enhanced variational circuit with more parameters
    for idx, element in enumerate(weights[:n_qubits]):
        qml.RY(element, wires=idx)

    # Add entangling gates for more expressivity
    for idx in range(n_qubits - 1):
        qml.CNOT(wires=[idx, idx + 1])

    # Second layer of rotations if we have enough parameters
    if len(weights) > n_qubits:
        for idx in range(n_qubits):
            if idx + n_qubits < len(weights):
                qml.RY(weights[idx + n_qubits], wires=idx)


#
# Hadamard test
#

dev_mu = qml.device("lightning.qubit", wires=tot_qubits, shots=None)


@qml.qnode(dev_mu, interface="autograd")
def local_hadamard_test(weights, l=None, lp=None, j=None, part=None):
    # First Hadamard gate applied to the ancillary qubit.
    qml.Hadamard(wires=ancilla_idx)

    # For estimating the imaginary part of the coefficient "mu", we must add a "-i"
    # phase gate.
    if part == "Im" or part == "im":
        qml.PhaseShift(-np.pi / 2, wires=ancilla_idx)

    # Variational circuit generating a guess for the solution vector |x>
    variational_block(weights)

    # Controlled application of the unitary component A_l of the problem matrix A.
    CA(l)

    # Adjoint of the unitary U_b associated to the problem vector |b>.
    qml.adjoint(U_b)()

    # Controlled Z operator at position j. If j = -1, apply the identity.
    if j != -1:
        qml.CZ(wires=[ancilla_idx, j])

    # Unitary U_b associated to the problem vector |b>.
    U_b()

    # Controlled application of Adjoint(A_lp).
    # In this specific example Adjoint(A_lp) = A_lp.
    CA(lp)

    # Second Hadamard gate applied to the ancillary qubit.
    qml.Hadamard(wires=ancilla_idx)

    # Expectation value of Z for the ancillary qubit.
    return qml.expval(qml.PauliZ(wires=ancilla_idx))


def mu(weights, l=None, lp=None, j=None):
    """Generates the coefficients to compute the "local" cost function C_L."""

    mu_real = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Re")
    mu_imag = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Im")

    return mu_real + 1.0j * mu_imag


#
# Local cost function
#

def psi_norm(weights):
    """Returns the normalization constant <psi|psi>, where |psi> = A |x>."""
    norm = 0.0

    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            norm = norm + c[l] * np.conj(c[lp]) * mu(weights, l, lp, -1)

    return abs(norm)


def cost_loc(weights):
    """Local version of the cost function. Tends to zero when A|x> is proportional to |b>."""
    mu_sum = 0.0

    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            for j in range(0, n_qubits):
                mu_sum = mu_sum + c[l] * np.conj(c[lp]) * mu(weights, l, lp, j)

    mu_sum = abs(mu_sum)

    # Cost function C_L
    return 0.5 - 0.5 * mu_sum / (n_qubits * psi_norm(weights))


#
# Variational optimization
#

print("Starting quantum optimization...")
np.random.seed(rng_seed)
# Use more parameters for enhanced variational circuit
n_params = 2 * n_qubits  # Double the parameters for two layers
w = q_delta * np.random.randn(n_params, requires_grad=True)

opt = qml.GradientDescentOptimizer(eta)

cost_history = []
for it in range(steps):
    w, cost = opt.step_and_cost(cost_loc, w)
    if it % 10 == 0 or it < 10:  # Print every 10 steps after the first 10
        print("Step {:3d}       Cost_L = {:9.7f}".format(it, cost))
    cost_history.append(cost)

# Check final convergence
final_cost = cost_history[-1]
print(f"\nFinal cost: {final_cost:.7f}")
if final_cost > 0.1:
    print("WARNING: Cost function did not converge well. Consider:")
    print("- Increasing number of steps")
    print("- Adjusting learning rate")
    print("- Using a more complex variational circuit")

plt.figure(figsize=(10, 6))
plt.plot(cost_history, "g", linewidth=2)
plt.ylabel("Cost function")
plt.xlabel("Optimization steps")
plt.title("VQLS Optimization Progress")
plt.grid(True, alpha=0.3)
plt.show()

#
# Comparison of quantum and classical results
#

print("\n" + "=" * 50)
print("COMPARISON OF RESULTS")
print("=" * 50)

# Classical algorithm - using the constructed matrix A
print("Solving classical linear system Ax = b...")

# Check if matrix is invertible
det_A = np.linalg.det(A_matrix)
print(f"Determinant of A: {det_A:.6f}")

if abs(det_A) < 1e-10:
    print("Warning: Matrix A is nearly singular! Using pseudo-inverse.")
    A_inv = np.linalg.pinv(A_matrix)
else:
    A_inv = np.linalg.inv(A_matrix)

x_classical = np.dot(A_inv, b)
c_probs = (x_classical / np.linalg.norm(x_classical)) ** 2

#
# Preparation of the quantum solution
#
print("Preparing quantum solution...")
dev_x = qml.device("lightning.qubit", wires=n_qubits, shots=None)


@qml.qnode(dev_x, interface="autograd")
def prepare_and_sample(weights):
    variational_block(weights)
    return qml.probs(wires=range(n_qubits))


q_probs = prepare_and_sample(w)

print("\nRESULTS:")
print("-" * 30)
print("Classical probabilities |x_n|^2:")
for i, prob in enumerate(c_probs):
    print(f"  State |{i:03b}>: {prob:.6f}")

print("\nQuantum probabilities |<x|n>|^2:")
for i, prob in enumerate(q_probs):
    print(f"  State |{i:03b}>: {prob:.6f}")

# Calculate the difference between classical and quantum results
diff = np.abs(c_probs - q_probs)
max_diff = np.max(diff)
mean_diff = np.mean(diff)
mse = np.mean(diff ** 2)
fidelity = np.sum(np.sqrt(c_probs * q_probs))

print(f"\nDifference analysis:")
print(f"  Maximum difference: {max_diff:.6f}")
print(f"  Mean difference: {mean_diff:.6f}")
print(f"  Mean squared error: {mse:.6f}")
print(f"  Fidelity: {fidelity:.6f}")

# Interpretation
if max_diff < 0.05:
    print("  → Excellent agreement!")
elif max_diff < 0.15:
    print("  → Good agreement (typical for VQLS)")
elif max_diff < 0.25:
    print("  → Reasonable agreement (could be improved)")
else:
    print("  → Poor agreement (needs optimization)")

# Visualization
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Classical probabilities
ax1.bar(np.arange(0, 2 ** n_qubits), c_probs, color="blue", alpha=0.7)
ax1.set_xlim(-0.5, 2 ** n_qubits - 0.5)
ax1.set_xlabel("Basis states")
ax1.set_ylabel("Probability")
ax1.set_title("Classical probabilities")
ax1.set_xticks(range(2 ** n_qubits))
ax1.set_xticklabels([f"|{i:03b}>" for i in range(2 ** n_qubits)], rotation=45)

# Quantum probabilities
ax2.bar(np.arange(0, 2 ** n_qubits), q_probs, color="green", alpha=0.7)
ax2.set_xlim(-0.5, 2 ** n_qubits - 0.5)
ax2.set_xlabel("Basis states")
ax2.set_ylabel("Probability")
ax2.set_title("Quantum probabilities")
ax2.set_xticks(range(2 ** n_qubits))
ax2.set_xticklabels([f"|{i:03b}>" for i in range(2 ** n_qubits)], rotation=45)

# Difference
ax3.bar(np.arange(0, 2 ** n_qubits), diff, color="red", alpha=0.7)
ax3.set_xlim(-0.5, 2 ** n_qubits - 0.5)
ax3.set_xlabel("Basis states")
ax3.set_ylabel("Absolute difference")
ax3.set_title("Difference |Classical - Quantum|")
ax3.set_xticks(range(2 ** n_qubits))
ax3.set_xticklabels([f"|{i:03b}>" for i in range(2 ** n_qubits)], rotation=45)

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("VQLS RESULTS INTERPRETATION")
print("=" * 70)
print("Based on the PennyLane VQLS tutorial:")
print("• Differences < 0.05: Excellent")
print("• Differences 0.05-0.15: Good (typical for VQLS)")
print("• Differences 0.15-0.25: Reasonable")
print("• Differences > 0.25: May need more optimization")
print()
print("To improve results further, try:")
print("1. More optimization steps (current: {})".format(steps))
print("2. Better variational circuit (more layers/parameters)")
print("3. Different optimizers (Adam, RMSprop)")
print("4. Multiple random initializations")
print("=" * 70)
