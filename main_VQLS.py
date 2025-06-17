# Pennylane
import pennylane as qml
from pennylane import numpy as np

# Plotting
import matplotlib.pyplot as plt

#
# Setting of the main hyper-parameters of the model
#

n_qubits = 3  # Number of system qubits.
tot_qubits = n_qubits + 1  # Addition of an ancillary qubit.
ancilla_idx = n_qubits  # Index of the ancillary qubit (last position).
steps = 100  # Number of optimization steps
eta = 0.4  # Learning rate
q_delta = 0.01  # Initial spread of random quantum weights
rng_seed = 0  # Seed for random number generator

# --------------------------------------------------------------------------
# --- MODIFICACIÓN PRINCIPAL ---
# Opción para elegir el método de obtención de resultados.
# True: Usa el vector de estado (rápido, sin shots, ideal para desarrollo).
# False: Usa muestreo (lento, simula un computador cuántico real).
MODO_ANALITICO = True
# El número de shots solo se usará si MODO_ANALITICO es False.
n_shots = 10 ** 6


# --------------------------------------------------------------------------


#
# Definition of the matrix components and coefficients
#

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
c = np.array([1.0, 0.5, 0.3])
# Define the b vector
b = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
b = b / np.linalg.norm(b)  # Normalize b

# Construct the matrix A using the coefficients
A_matrix = construct_matrix_A(c)

print("Coefficients c =", c)
print("Matrix A =\n", A_matrix)
print("Vector b =", b, "\n")


#
# Circuits of the quantum linear problem
#

def U_b():
    """Unitary matrix rotating the ground state to the problem vector |b> = U_b |0>."""
    qml.StatePrep(b, wires=range(n_qubits))


def CA(idx):
    """Controlled versions of the unitary components A_l of the problem matrix A."""
    if idx == 0:  # Identity
        pass
    elif idx == 1:  # Controlled-A_1
        qml.CNOT(wires=[ancilla_idx, 0])
        qml.CZ(wires=[ancilla_idx, 1])
    elif idx == 2:  # Controlled-A_2
        qml.CNOT(wires=[ancilla_idx, 0])


#
# Variational quantum circuit
#

def variational_block(weights):
    """Variational circuit mapping the ground state |0> to the ansatz state |x>."""
    for idx in range(n_qubits):
        qml.Hadamard(wires=idx)

    for idx, element in enumerate(weights[:n_qubits]):
        qml.RY(element, wires=idx)

    for idx in range(n_qubits - 1):
        qml.CNOT(wires=[idx, idx + 1])

    if len(weights) > n_qubits:
        for idx in range(n_qubits):
            if idx + n_qubits < len(weights):
                qml.RY(weights[idx + n_qubits], wires=idx)


#
# Hadamard test & Cost function
#

# The device for the cost function is already analytic (no shots specified), so it's fast.
dev_mu = qml.device("lightning.qubit", wires=tot_qubits)


@qml.qnode(dev_mu, interface="autograd")
def local_hadamard_test(weights, l=None, lp=None, j=None, part=None):
    qml.Hadamard(wires=ancilla_idx)
    if part == "Im" or part == "im":
        qml.PhaseShift(-np.pi / 2, wires=ancilla_idx)
    variational_block(weights)
    CA(l)
    qml.adjoint(U_b)()
    if j != -1:
        qml.CZ(wires=[ancilla_idx, j])
    U_b()
    CA(lp)
    qml.Hadamard(wires=ancilla_idx)
    return qml.expval(qml.PauliZ(wires=ancilla_idx))


def mu(weights, l=None, lp=None, j=None):
    mu_real = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Re")
    mu_imag = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Im")
    return mu_real + 1.0j * mu_imag


def psi_norm(weights):
    norm = 0.0
    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            norm = norm + c[l] * np.conj(c[lp]) * mu(weights, l, lp, -1)
    return abs(norm)


def cost_loc(weights):
    mu_sum = 0.0
    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            for j in range(0, n_qubits):
                mu_sum = mu_sum + c[l] * np.conj(c[lp]) * mu(weights, l, lp, j)
    mu_sum = abs(mu_sum)
    return 0.5 - 0.5 * mu_sum / (n_qubits * psi_norm(weights))


#
# Variational optimization
#

print("Starting quantum optimization...")
np.random.seed(rng_seed)
n_params = 2 * n_qubits
w = q_delta * np.random.randn(n_params, requires_grad=True)
opt = qml.GradientDescentOptimizer(eta)
cost_history = []

for it in range(steps):
    w, cost = opt.step_and_cost(cost_loc, w)
    if it % 10 == 0 or it < 10:
        print(f"Step {it:3d}       Cost_L = {cost:9.7f}")
    cost_history.append(cost)

final_cost = cost_history[-1]
print(f"\nFinal cost: {final_cost:.7f}")
if final_cost > 0.1:
    print("WARNING: Cost function did not converge well.")

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

# Classical algorithm
print("Solving classical linear system Ax = b...")
if abs(np.linalg.det(A_matrix)) < 1e-10:
    print("Warning: Matrix A is nearly singular! Using pseudo-inverse.")
    A_inv = np.linalg.pinv(A_matrix)
else:
    A_inv = np.linalg.inv(A_matrix)

x_classical = np.dot(A_inv, b)
# For a fair comparison, we normalize the classical solution vector
# as the quantum state |x> is inherently normalized.
norm_x_classical = np.linalg.norm(x_classical)
x_classical_normalized = x_classical / norm_x_classical
c_probs = np.abs(x_classical_normalized) ** 2

#
# Preparation of the quantum solution (with the new analytic option)
#
print("Preparing quantum solution...")

if MODO_ANALITICO:
    # MÉTODO ANALÍTICO: Obtiene el vector de estado exacto (rápido, sin error de muestreo).
    print("Using analytical mode (statevector)...")

    # Define a device without shots to get the state vector.
    dev_x = qml.device("lightning.qubit", wires=n_qubits)


    @qml.qnode(dev_x)
    def get_solution_statevector(weights):
        """Prepares the state |x> and returns the full statevector."""
        variational_block(weights)
        return qml.state()


    # Execute the circuit once to get the final statevector.
    state_vector = get_solution_statevector(w)

    # The probabilities are the squared absolute values of the amplitudes.
    q_probs = np.abs(state_vector) ** 2

    # Adicional: Imprimir los coeficientes complejos, como se pedía.
    print("\nComplex amplitudes of the quantum solution vector |x>:")
    for i, amp in enumerate(state_vector):
        print(f"  State |{i:03b}>: {amp.real:+.4f} {amp.imag:+.4f}j")

else:
    # MÉTODO DE MUESTREO: Simula mediciones (lento, como el original).
    print(f"Using sampling mode with {n_shots} shots...")

    # Define a device with shots.
    dev_x = qml.device("lightning.qubit", wires=n_qubits, shots=n_shots)


    @qml.qnode(dev_x)
    def prepare_and_sample(weights):
        """Prepares the state |x> and returns samples."""
        variational_block(weights)
        return qml.sample()


    # Get the raw samples (this is the slow part).
    raw_samples = prepare_and_sample(w)

    # Process samples to get probabilities.
    samples = [int("".join(str(bs) for bs in sam), base=2) for sam in raw_samples]
    q_probs = np.bincount(samples, minlength=2 ** n_qubits) / n_shots

#
# Analysis and Visualization
#

print("\nRESULTS:")
print("-" * 30)
print("Classical probabilities |x_n|^2:")
for i, prob in enumerate(c_probs):
    print(f"  State |{i:03b}>: {prob:.6f}")

print("\nQuantum probabilities |<x|n>|^2:")
for i, prob in enumerate(q_probs):
    print(f"  State |{i:03b}>: {prob:.6f}")

# Calculate metrics
diff = np.abs(c_probs - q_probs)
max_diff = np.max(diff)
mean_diff = np.mean(diff)
fidelity = np.sum(np.sqrt(c_probs * q_probs))  # Quantum fidelity

print(f"\nDifference analysis:")
print(f"  Maximum difference: {max_diff:.6f}")
print(f"  Mean difference:    {mean_diff:.6f}")
print(f"  Fidelity:           {fidelity:.6f}")

# Interpretation
if fidelity > 0.99:
    print("  → Excellent agreement!")
elif fidelity > 0.95:
    print("  → Good agreement.")
elif fidelity > 0.90:
    print("  → Reasonable agreement.")
else:
    print("  → Poor agreement (needs optimization).")

# Visualization
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Classical probabilities
ax1.bar(np.arange(0, 2 ** n_qubits), c_probs, color="blue", alpha=0.7)
ax1.set_xlim(-0.5, 2 ** n_qubits - 0.5)
ax1.set_xlabel("Basis states")
ax1.set_ylabel("Probability")
ax1.set_title("Classical Probabilities")
ax1.set_xticks(range(2 ** n_qubits))
ax1.set_xticklabels([f"|{i:03b}>" for i in range(2 ** n_qubits)], rotation=45)

# Quantum probabilities
ax2.bar(np.arange(0, 2 ** n_qubits), q_probs, color="green", alpha=0.7)
ax2.set_xlim(-0.5, 2 ** n_qubits - 0.5)
ax2.set_xlabel("Basis states")
ax2.set_ylabel("Probability")
ax2.set_title("Quantum Probabilities")
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
