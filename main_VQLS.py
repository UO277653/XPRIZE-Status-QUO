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
eta = 0.8  # Learning rate
q_delta = 0.01  # Initial spread of random quantum weights
rng_seed = 0  # Seed for random number generator
n_layers = 2

# --------------------------------------------------------------------------
# --- NUEVA OPTIMIZACIÓN DE VELOCIDAD ---
# True: Usa la función de coste "Global". Es mucho más rápida (menos simulaciones por paso).
# False: Usa la función de coste "Local" original (muy lenta pero teóricamente robusta).
USAR_COSTE_GLOBAL = False
# --------------------------------------------------------------------------

# Opción para modo analítico/muestreo en el resultado final (ya implementada)
MODO_ANALITICO = True
n_shots = 10 ** 6


#
# Definiciones (Matrices, U_b, CA, variational_block) - Sin cambios
#
def get_base_matrices():
    Id = np.identity(2);
    Z = np.array([[1, 0], [0, -1]]);
    X = np.array([[0, 1], [1, 0]])
    A_0 = np.identity(2 ** n_qubits);
    A_1 = np.kron(np.kron(X, Z), Id);
    A_2 = np.kron(np.kron(X, Id), Id)
    return A_0, A_1, A_2


def construct_matrix_A(coefficients):
    A_0, A_1, A_2 = get_base_matrices()
    return coefficients[0] * A_0 + coefficients[1] * A_1 + coefficients[2] * A_2


c = np.array([1.0, 0.5, 0.3])

# Ejemplo de b complejo
raw_b = np.array([1 + 1j, 0, 1 - 1j, 0, 0, 0, 0, 0], dtype=complex)
b = raw_b / np.linalg.norm(raw_b)
A_matrix = construct_matrix_A(c)


def U_b():
    """Prepara el estado |b> con amplitudes complejas."""
    qml.MottonenStatePreparation(b, wires=range(n_qubits))


def CA(idx):
    if idx == 1:
        qml.CNOT(wires=[ancilla_idx, 0]);
        qml.CZ(wires=[ancilla_idx, 1])
    elif idx == 2:
        qml.CNOT(wires=[ancilla_idx, 0])


def variational_block(weights):
    for i in range(n_qubits): qml.Hadamard(wires=i)
    for i in range(n_qubits - 1): qml.CNOT(wires=[i, i + 1])
    if len(weights) > n_qubits:
        for i in range(n_qubits):
            if i + n_qubits < len(weights): qml.RY(weights[i + n_qubits], wires=i)


def ansatz_complex(params, wires):
    n = len(wires)
    assert len(params) == 3 * n  # ahora necesitarás 3 parámetros por qubit

    idx = 0
    for w in wires:
        qml.Hadamard(wires=w)
    # una capa entera de Rot:
    for w in wires:
        theta, phi, lam = params[idx], params[idx + 1], params[idx + 2]
        qml.Rot(theta, phi, lam, wires=w)
        idx += 3
    for i in range(n - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])  # podrías repetir Rot otra vez si quieres dos capas


def variational_block_complex(weights):
    # Asumimos que tu VQLS usa n_qubits global
    ansatz_complex(weights, wires=list(range(n_qubits)))


#
# Definiciones para las funciones de coste (Local y Global)
#
dev_cost = qml.device("lightning.qubit", wires=tot_qubits)


# --- Subrutinas para el Coste LOCAL (original) ---
@qml.qnode(dev_cost, interface="autograd")
def local_hadamard_test(weights, l, lp, j, part):
    qml.Hadamard(wires=ancilla_idx)
    if part == "Im": qml.PhaseShift(-np.pi / 2, wires=ancilla_idx)
    variational_block_complex(weights)
    CA(l)
    qml.adjoint(U_b)()
    if j != -1: qml.CZ(wires=[ancilla_idx, j])
    U_b()
    CA(lp)
    qml.Hadamard(wires=ancilla_idx)
    return qml.expval(qml.PauliZ(wires=ancilla_idx))


def mu(weights, l, lp, j=-1):
    real = local_hadamard_test(weights, l, lp, j, "Re")
    imag = local_hadamard_test(weights, l, lp, j, "Im")
    return real + 1.0j * imag


def cost_loc(weights):
    mu_sum = 0.0
    for l in range(len(c)):
        for lp in range(len(c)):
            for j in range(n_qubits):
                mu_sum += c[l] * np.conj(c[lp]) * mu(weights, l, lp, j)
    norm = psi_norm(weights)
    return 0.5 - 0.5 * abs(mu_sum) / (n_qubits * norm)


# --- Subrutinas para el Coste GLOBAL (NUEVO y RÁPIDO) ---
@qml.qnode(dev_cost, interface="autograd")
def global_hadamard_test(weights, l, part):
    qml.Hadamard(wires=ancilla_idx)
    if part == "Im": qml.PhaseShift(-np.pi / 2, wires=ancilla_idx)
    variational_block_complex(weights)
    CA(l)
    qml.adjoint(U_b)()
    qml.Hadamard(wires=ancilla_idx)
    return qml.expval(qml.PauliZ(ancilla_idx))


def cost_glob(weights):
    """Función de coste global, computacionalmente más barata."""
    # 1. Calcular el numerador: |<b|A|x>|^2
    bAx_overlap = 0.0
    for l in range(len(c)):
        real = global_hadamard_test(weights, l, "Re")
        imag = global_hadamard_test(weights, l, "Im")
        bAx_overlap += c[l] * (real + 1.0j * imag)

    numerator = np.abs(bAx_overlap) ** 2

    # 2. Calcular el denominador: <x|A†A|x> (reutilizamos psi_norm)
    denominator = psi_norm(weights)

    return 1 - (numerator / denominator)


# --- Función compartida por ambas funciones de coste ---
def psi_norm(weights):
    """Calcula <x|A†A|x>. Es necesario para ambas funciones de coste."""
    norm = 0.0
    for l in range(len(c)):
        for lp in range(len(c)):
            norm += c[l] * np.conj(c[lp]) * mu(weights, l, lp)
    return abs(norm)


#
# Variational optimization
#

print("Starting quantum optimization...")
np.random.seed(rng_seed)
n_params = 3 * n_qubits
w = q_delta * np.random.randn(n_params, requires_grad=True)

# --- Elegimos el optimizador y la función de coste a usar ---
# El optimizador Adam suele converger más rápido que GradientDescent
opt = qml.AdamOptimizer(eta)

# opt = qml.GradientDescentOptimizer(eta)

if USAR_COSTE_GLOBAL:
    cost_function_to_use = cost_glob
    cost_name = "Global"
    print("Using **FAST** Global Cost Function.")
else:
    cost_function_to_use = cost_loc
    cost_name = "Local"
    print("Using **SLOW** Local Cost Function.")

cost_history = []
for it in range(steps):
    w, cost = opt.step_and_cost(cost_function_to_use, w)
    if it % 10 == 0 or it < 10:
        print(f"Step {it:3d}       Cost_{cost_name} = {cost:9.7f}")
    cost_history.append(cost)

final_cost = cost_history[-1]
print(f"\nFinal cost: {final_cost:.7f}")
plt.figure(figsize=(10, 6))
plt.plot(cost_history, "g", linewidth=2)
plt.ylabel(f"Cost Function ({cost_name})")
plt.xlabel("Optimization steps")
plt.title(f"VQLS Optimization Progress ({cost_name} Cost)")
plt.grid(True, alpha=0.3)
plt.show()

#
# Comparison of quantum and classical results (código sin cambios, se adapta a MODO_ANALITICO)
#
print("\n" + "=" * 50 + "\nCOMPARISON OF RESULTS\n" + "=" * 50)

# Solución clásica
A_inv = np.linalg.pinv(A_matrix)
x_classical = np.dot(A_inv, b)
c_probs = (np.abs(x_classical / np.linalg.norm(x_classical))) ** 2

# Solución cuántica (sección ya optimizada con MODO_ANALITICO)
print("\nPreparing quantum solution...")
if MODO_ANALITICO:
    print("Using analytical mode (statevector)...")
    dev_x = qml.device("lightning.qubit", wires=n_qubits)


    @qml.qnode(dev_x)
    def get_solution_statevector(weights):
        variational_block_complex(weights)
        return qml.state()


    state_vector = get_solution_statevector(w)
    q_probs = np.abs(state_vector) ** 2
else:
    print(f"Using sampling mode with {n_shots} shots...")
    dev_x = qml.device("lightning.qubit", wires=n_qubits, shots=n_shots)


    @qml.qnode(dev_x)
    def prepare_and_sample(weights):
        variational_block_complex(weights)
        return qml.sample()


    raw_samples = prepare_and_sample(w)
    samples = [int("".join(str(bs) for bs in sam), base=2) for sam in raw_samples]
    q_probs = np.bincount(samples, minlength=2 ** n_qubits) / n_shots

# Impresión de resultados y gráficos (sin cambios)
print("\nRESULTS:")
print("-" * 30)
print("Classical probabilities |x_n|^2:")
for i, prob in enumerate(c_probs): print(f"  State |{i:03b}>: {prob:.6f}")
print("\nQuantum probabilities |<x|n>|^2:")
for i, prob in enumerate(q_probs): print(f"  State |{i:03b}>: {prob:.6f}")
fidelity = np.sum(np.sqrt(c_probs * q_probs))
print(f"\nFidelity: {fidelity:.6f}")

# Visualización
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
ax1.bar(np.arange(2 ** n_qubits), c_probs, color="blue", alpha=0.7)
ax1.set_title("Classical Probabilities")
ax1.set_xlabel("Basis states")
ax1.set_ylabel("Probability")
ax2.bar(np.arange(2 ** n_qubits), q_probs, color="green", alpha=0.7)
ax2.set_title("Quantum Probabilities")
ax2.set_xlabel("Basis states")
plt.tight_layout()
plt.show()
