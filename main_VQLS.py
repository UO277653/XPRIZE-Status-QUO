# Pennylane
import pennylane as qml
from pennylane import numpy as np

# Plotting
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

#
# Setting of the main hyper-parameters of the model
#

n_qubits = 3  # Number of system qubits.
tot_qubits = n_qubits + 1  # Addition of an ancillary qubit.
ancilla_idx = n_qubits  # Index of the ancillary qubit (last position).
steps = 2000  # Number of optimization steps
eta = 0.4  # Learning rate
q_delta = 0.01  # Initial spread of random quantum weights
rng_seed = 2  # Seed for random number generator
n_layers = 2

# --------------------------------------------------------------------------
# MODOS DE EJECUCIÓN
MULTIPLE_RUNS = False  # Para comparación sistemática
SINGLE_MODE = True  # Para experimento individual
REFINEMENT_MODE = False  # NUEVO: Para refinamiento de la mejor configuración
# --------------------------------------------------------------------------

# True: Usa la función de coste "Global". Es mucho más rápida (menos simulaciones por paso).
# False: Usa la función de coste "Local" original (muy lenta pero teóricamente robusta).
USAR_COSTE_GLOBAL = False

# Opción para modo analítico/muestreo en el resultado final
MODO_ANALITICO = True
n_shots = 10 ** 6

# Configuración para múltiples experimentos
N_RANDOM_SEEDS = 3  # Número de semillas aleatorias por configuración


#
# Definiciones (Matrices, U_b, CA, variational_block) - Sin cambios
#
def get_base_matrices():
    Id = np.identity(2)
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    A_0 = np.identity(2 ** n_qubits)
    A_1 = np.kron(np.kron(X, Z), Id)
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
        qml.CNOT(wires=[ancilla_idx, 0])
        qml.CZ(wires=[ancilla_idx, 1])
    elif idx == 2:
        qml.CNOT(wires=[ancilla_idx, 0])


def variational_block(weights):
    """Ansatz básico original."""
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    if len(weights) > n_qubits:
        for i in range(n_qubits):
            if i + n_qubits < len(weights):
                qml.RY(weights[i + n_qubits], wires=i)


def ansatz_complex(params, wires):
    """Ansatz más expresivo para números complejos."""
    n = len(wires)
    assert len(params) == 3 * n  # 3 parámetros por qubit

    idx = 0
    for w in wires:
        qml.Hadamard(wires=w)

    # Una capa entera de rotaciones Rot (más expresivo que solo RY)
    for w in wires:
        theta, phi, lam = params[idx], params[idx + 1], params[idx + 2]
        qml.Rot(theta, phi, lam, wires=w)
        idx += 3

    # Entanglement
    for i in range(n - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])


def variational_block_complex(weights):
    """Wrapper para el ansatz complejo."""
    ansatz_complex(weights, wires=list(range(n_qubits)))


def ansatz_layered(params, wires, n_layers=2):
    """Ansatz con múltiples capas para mayor expresividad."""
    n = len(wires)
    params_per_layer = 3 * n
    assert len(params) == params_per_layer * n_layers

    idx = 0
    for layer in range(n_layers):
        # Capa de rotaciones
        for w in wires:
            theta, phi, lam = params[idx], params[idx + 1], params[idx + 2]
            qml.Rot(theta, phi, lam, wires=w)
            idx += 3

        # Capa de entanglement (varía según la capa)
        if layer % 2 == 0:
            # Entanglement lineal
            for i in range(n - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
        else:
            # Entanglement circular
            for i in range(n):
                qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])


def variational_block_layered(weights):
    """Wrapper para el ansatz multicapa."""
    ansatz_layered(weights, wires=list(range(n_qubits)), n_layers=n_layers)


def ansatz_hardware_efficient(params, wires):
    """Ansatz eficiente para hardware cuántico real."""
    n = len(wires)
    assert len(params) == 2 * n  # 2 parámetros por qubit

    idx = 0
    # Capa inicial de RY
    for w in wires:
        qml.RY(params[idx], wires=w)
        idx += 1

    # Entanglement
    for i in range(n - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])

    # Segunda capa de RY
    for w in wires:
        qml.RY(params[idx], wires=w)
        idx += 1


def variational_block_hardware_efficient(weights):
    """Wrapper para el ansatz hardware-efficient."""
    ansatz_hardware_efficient(weights, wires=list(range(n_qubits)))


# Diccionario de optimizadores disponibles (ampliado)
optimizers = {"adam": lambda lr: qml.AdamOptimizer(stepsize=lr), "sgd": lambda lr: qml.GradientDescentOptimizer(stepsize=lr),
              "rmsprop": lambda lr: qml.RMSPropOptimizer(stepsize=lr), "nesterov": lambda lr: qml.NesterovMomentumOptimizer(stepsize=lr),
              "adagrad": lambda lr: qml.AdagradOptimizer(stepsize=lr), "momentum": lambda lr: qml.MomentumOptimizer(stepsize=lr),
              "spsa": lambda lr: qml.SPSAOptimizer(maxiter=steps, blocking=False)  # Para ruido
              }

# Diccionario de bloques variacionales (ampliado)
ansatzes = {"default": (variational_block, lambda: 2 * n_qubits),  # (función, parámetros)
            "complex": (variational_block_complex, lambda: 3 * n_qubits), "layered": (variational_block_layered, lambda: 3 * n_qubits * n_layers),
            "hardware_efficient": (variational_block_hardware_efficient, lambda: 2 * n_qubits), }

#
# Definiciones para las funciones de coste (Local y Global)
#
dev_cost = qml.device("lightning.qubit", wires=tot_qubits)


# --- Subrutinas para el Coste LOCAL (original) ---
@qml.qnode(dev_cost, interface="autograd")
def local_hadamard_test(weights, l, lp, j, part, variational_block_fn):
    qml.Hadamard(wires=ancilla_idx)
    if part == "Im":
        qml.PhaseShift(-np.pi / 2, wires=ancilla_idx)
    variational_block_fn(weights)
    CA(l)
    qml.adjoint(U_b)()
    if j != -1:
        qml.CZ(wires=[ancilla_idx, j])
    U_b()
    CA(lp)
    qml.Hadamard(wires=ancilla_idx)
    return qml.expval(qml.PauliZ(wires=ancilla_idx))


def mu(weights, l, lp, j=-1, variational_block_fn=variational_block):
    real = local_hadamard_test(weights, l, lp, j, "Re", variational_block_fn)
    imag = local_hadamard_test(weights, l, lp, j, "Im", variational_block_fn)
    return real + 1.0j * imag


def cost_loc(weights, variational_block_fn=variational_block):
    mu_sum = 0.0
    for l in range(len(c)):
        for lp in range(len(c)):
            for j in range(n_qubits):
                mu_sum += c[l] * np.conj(c[lp]) * mu(weights, l, lp, j, variational_block_fn)
    norm = psi_norm(weights, variational_block_fn)
    return 0.5 - 0.5 * abs(mu_sum) / (n_qubits * norm)


# --- Subrutinas para el Coste GLOBAL (NUEVO y RÁPIDO) ---
@qml.qnode(dev_cost, interface="autograd")
def global_hadamard_test(weights, l, part, variational_block_fn):
    qml.Hadamard(wires=ancilla_idx)
    if part == "Im":
        qml.PhaseShift(-np.pi / 2, wires=ancilla_idx)
    variational_block_fn(weights)
    CA(l)
    qml.adjoint(U_b)()
    qml.Hadamard(wires=ancilla_idx)
    return qml.expval(qml.PauliZ(ancilla_idx))


def cost_glob(weights, variational_block_fn=variational_block):
    """Función de coste global, computacionalmente más barata."""
    # 1. Calcular el numerador: |<b|A|x>|^2
    bAx_overlap = 0.0
    for l in range(len(c)):
        real = global_hadamard_test(weights, l, "Re", variational_block_fn)
        imag = global_hadamard_test(weights, l, "Im", variational_block_fn)
        bAx_overlap += c[l] * (real + 1.0j * imag)

    numerator = np.abs(bAx_overlap) ** 2

    # 2. Calcular el denominador: <x|A†A|x>
    denominator = psi_norm(weights, variational_block_fn)

    return 1 - (numerator / denominator)


# --- Función compartida por ambas funciones de coste ---
def psi_norm(weights, variational_block_fn=variational_block):
    """Calcula <x|A†A|x>. Es necesario para ambas funciones de coste."""
    norm = 0.0
    for l in range(len(c)):
        for lp in range(len(c)):
            norm += c[l] * np.conj(c[lp]) * mu(weights, l, lp, variational_block_fn=variational_block_fn)
    return abs(norm)


def run_single_optimization(optimizer_name, ansatz_name, seed, learning_rate=eta, max_steps=None, q_delta_local=None):
    """Ejecuta una optimización individual con parámetros opcionales."""
    if max_steps is None:
        max_steps = steps
    if q_delta_local is None:
        q_delta_local = q_delta

    print(f"\n--- Running: {optimizer_name} + {ansatz_name} (seed={seed}, lr={learning_rate}, steps={max_steps}) ---")

    # Configurar optimizador
    if optimizer_name == "spsa":
        opt = optimizers[optimizer_name](learning_rate)
    else:
        opt = optimizers[optimizer_name](learning_rate)

    # Configurar ansatz
    variational_block_fn, n_params_fn = ansatzes[ansatz_name]
    n_params = n_params_fn()

    # Inicializar parámetros
    np.random.seed(seed)
    w = q_delta_local * np.random.randn(n_params, requires_grad=True)

    # Elegir función de coste
    if USAR_COSTE_GLOBAL:
        cost_function = lambda weights: cost_glob(weights, variational_block_fn)
        cost_name = "Global"
    else:
        cost_function = lambda weights: cost_loc(weights, variational_block_fn)
        cost_name = "Local"

    # Optimización
    cost_history = []
    for it in range(max_steps):
        if optimizer_name == "spsa":
            # SPSA tiene una interfaz ligeramente diferente
            w = opt.step(cost_function, w)
            cost = cost_function(w)
        else:
            w, cost = opt.step_and_cost(cost_function, w)

        cost_history.append(float(cost))  # Asegurar que es float

        if it % 100 == 0 or it <= 5:
            print(f"  Step {it:3d}: Cost = {cost:9.20f}")

    final_cost = cost_history[-1]
    print(f"  Final cost: {final_cost:.7f}")

    return {"optimizer": optimizer_name, "ansatz": ansatz_name, "seed": seed, "learning_rate": learning_rate, "final_cost": final_cost,
            "cost_history": cost_history, "n_params": n_params, "final_weights": w.tolist() if hasattr(w, 'tolist') else list(w),
            "variational_block_fn": variational_block_fn, "max_steps": max_steps, "q_delta_used": q_delta_local}


def calculate_solution_quality(result):
    """Calcula métricas de calidad de la solución cuántica."""
    w = np.array(result["final_weights"])
    variational_block_fn = result["variational_block_fn"]

    # Obtener vector de estado cuántico
    dev_x = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev_x)
    def get_quantum_state(weights):
        variational_block_fn(weights)
        return qml.state()

    quantum_state = get_quantum_state(w)
    q_probs = np.abs(quantum_state) ** 2

    # Solución clásica
    A_inv = np.linalg.pinv(A_matrix)
    x_classical = np.dot(A_inv, b)
    x_classical_normalized = x_classical / np.linalg.norm(x_classical)
    c_probs = np.abs(x_classical_normalized) ** 2

    # Métricas de calidad
    fidelity = np.sum(np.sqrt(c_probs * q_probs))

    # Error residual: ||Ax - b||
    quantum_amplitudes = quantum_state / np.linalg.norm(quantum_state)
    residual_error = np.linalg.norm(A_matrix @ quantum_amplitudes - b)

    # Distancia L2 entre distribuciones de probabilidad
    prob_distance = np.linalg.norm(c_probs - q_probs)

    return {'fidelity': float(fidelity), 'residual_error': float(residual_error), 'prob_distance': float(prob_distance),
            'classical_probs': c_probs.tolist(), 'quantum_probs': q_probs.tolist()}


def compare_classical_quantum_solution(result):
    """Compara la solución clásica con la cuántica y calcula la fidelidad."""
    print("\n" + "=" * 50)
    print("COMPARISON OF CLASSICAL AND QUANTUM RESULTS")
    print("=" * 50)

    # Extraer parámetros optimizados
    w = np.array(result["final_weights"])
    variational_block_fn = result["variational_block_fn"]

    # Solución clásica
    print("Computing classical solution...")
    A_inv = np.linalg.pinv(A_matrix)
    x_classical = np.dot(A_inv, b)
    c_probs = (np.abs(x_classical / np.linalg.norm(x_classical))) ** 2

    # Solución cuántica
    print("Computing quantum solution...")
    if MODO_ANALITICO:
        print("Using analytical mode (statevector)...")
        dev_x = qml.device("lightning.qubit", wires=n_qubits, shots=None)

        @qml.qnode(dev_x, interface="autograd")
        def get_solution_statevector(weights):
            variational_block_fn(weights)
            return qml.state()

        state_vector = get_solution_statevector(w)
        state_vector = state_vector / np.linalg.norm(state_vector)  # ¡AÑADE ESTA LÍNEA!
        q_probs = np.abs(state_vector) ** 2
    else:
        print(f"Using sampling mode with {n_shots} shots...")
        dev_x = qml.device("lightning.qubit", wires=n_qubits, shots=n_shots)

        @qml.qnode(dev_x)
        def prepare_and_sample(weights):
            variational_block_fn(weights)
            return qml.sample()

        raw_samples = prepare_and_sample(w)
        samples = [int("".join(str(bs) for bs in sam), base=2) for sam in raw_samples]
        q_probs = np.bincount(samples, minlength=2 ** n_qubits) / n_shots

    # Mostrar resultados
    print("\nRESULTS:")
    print("-" * 30)
    print("Classical probabilities |x_n|^2:")
    for i, prob in enumerate(c_probs):
        print(f"  State |{i:03b}>: {prob:.6f}")

    print("\nQuantum probabilities |<x|n>|^2:")
    for i, prob in enumerate(q_probs):
        print(f"  State |{i:03b}>: {prob:.6f}")

    # Calcular fidelidad
    fidelity = np.sum(np.sqrt(c_probs * q_probs))
    print(f"\nFidelity between classical and quantum solutions: {fidelity:.6f}")

    # Visualización CORREGIDA
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Gráfico clásico
    ax1.bar(np.arange(2 ** n_qubits), c_probs, color="blue", alpha=0.7)
    ax1.set_title("Classical Probabilities")
    ax1.set_xlabel("Basis states")
    ax1.set_ylabel("Probability")
    ax1.set_xticks(np.arange(2 ** n_qubits))
    ax1.set_xticklabels([f"|{i:03b}>" for i in range(2 ** n_qubits)], rotation=45)

    # Gráfico cuántico
    ax2.bar(np.arange(2 ** n_qubits), q_probs, color="green", alpha=0.7)
    ax2.set_title("Quantum Probabilities")
    ax2.set_xlabel("Basis states")
    ax2.set_xticks(np.arange(2 ** n_qubits))
    ax2.set_xticklabels([f"|{i:03b}>" for i in range(2 ** n_qubits)], rotation=45)

    plt.tight_layout()
    plt.show()

    return {"classical_probs": c_probs.tolist(), "quantum_probs": q_probs.tolist(), "fidelity": float(fidelity)}


def save_complete_results(result, comparison_data, mode="single"):
    """Guarda los resultados completos incluyendo la comparación."""

    # Helper to convert complex numbers
    def convert_complex(obj):
        if isinstance(obj, complex):
            return {"real": float(obj.real), "imag": float(obj.imag)}
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj

    # Crear el diccionario de resultados completo
    complete_results = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "mode": mode,
                        "optimization": {"optimizer": result["optimizer"], "ansatz": result["ansatz"], "seed": result["seed"],
                                         "learning_rate": result["learning_rate"], "final_cost": result["final_cost"],
                                         "cost_history": result["cost_history"], "n_params": result["n_params"]}, "comparison": comparison_data,
                        "configuration": {"n_qubits": n_qubits, "steps": steps, "eta": eta, "coefficients": c.tolist(),
                                          "b_vector": [convert_complex(x) for x in b], "usar_coste_global": USAR_COSTE_GLOBAL,
                                          "modo_analitico": MODO_ANALITICO}}

    # Determinar filename según el modo
    if mode == "refinement":
        filename = "vqls_refinement_results.json"
    else:
        filename = "vqls_single_results.json"

    all_experiments = []
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                all_experiments = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            all_experiments = []

    # Agregar el nuevo experimento
    complete_results["experiment_id"] = len(all_experiments) + 1
    all_experiments.append(complete_results)

    # Guardar todos los experimentos
    with open(filename, "w") as f:
        json.dump(all_experiments, f, indent=2, default=convert_complex)

    print(f"\nComplete results saved to: {filename} (Experiment #{complete_results['experiment_id']})")
    return filename


# ============================================================================
# REFINEMENT MODE
# ============================================================================

def refine_best_configuration():
    """Refina la mejor configuración encontrada con diferentes parámetros."""
    print("=" * 70)
    print("🔬 REFINEMENT MODE: Optimizing Best Configuration")
    print("=" * 70)

    # Mejor configuración base (basada en tus resultados anteriores)
    base_optimizer = "nesterov"
    base_ansatz = "hardware_efficient"

    # Parámetros a probar para refinamiento
    refinement_configs = [(base_optimizer, base_ansatz, 0.2, "Medium LR", rng_seed, steps, q_delta),
                          (base_optimizer, base_ansatz, 0.4, "Medium LR", rng_seed, steps, q_delta),
                          (base_optimizer, base_ansatz, 0.6, "Medium LR", rng_seed, steps, q_delta),
                          (base_optimizer, base_ansatz, 0.8, "Base LR", rng_seed, steps, q_delta),
                          (base_optimizer, base_ansatz, 0.95, "Higher LR", rng_seed, steps, q_delta),
                          (base_optimizer, base_ansatz, 0.4, "Fine-tuned", rng_seed, 1500, q_delta),
                          (base_optimizer, base_ansatz, 0.4, "Very Fine", rng_seed, 2000, q_delta),
                          ("adam", base_ansatz, 0.6, "Adam Fine", rng_seed, steps, q_delta),
                          ("adagrad", base_ansatz, 0.6, "Adagrad", rng_seed, steps, q_delta), ]

    best_fidelity = 0
    best_residual = float('inf')
    best_overall = None
    all_refined_results = []

    for opt, ans, lr, desc, seed, max_steps, q_delta_local in refinement_configs:
        print(f"\n🧪 Refining: {desc} (lr={lr}, seed={seed}, steps={max_steps}, q_delta={q_delta_local})")

        try:
            result = run_single_optimization(opt, ans, seed, lr, max_steps, q_delta_local)

            # Calcular métricas de calidad
            quality = calculate_solution_quality(result)
            result['quality'] = quality
            result['description'] = desc
            result['config_details'] = f"lr={lr}, seed={seed}, steps={max_steps}, q_delta={q_delta_local}"

            all_refined_results.append(result)

            fidelity = quality['fidelity']
            residual = quality['residual_error']

            print(f"  📊 Fidelity: {fidelity:.6f}, Residual: {residual:.6f}")

            # Criterio de mejor solución: priorizar fidelidad alta, luego residual bajo
            is_better = (fidelity > best_fidelity) or (fidelity > 0.95 and residual < best_residual)

            if is_better:
                best_fidelity = fidelity
                best_residual = residual
                best_overall = result
                print(f"  🌟 NEW BEST!")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    return best_overall, all_refined_results


def analyze_refinement_results(best_result, all_results):
    """Analiza y visualiza los resultados del refinamiento."""
    print(f"\n" + "=" * 70)
    print("📈 REFINEMENT ANALYSIS")
    print("=" * 70)

    # Tabla de resultados
    print(f"\n{'Configuration':<25} {'Fidelity':<10} {'Residual':<10} {'Cost':<12}")
    print("-" * 60)

    for result in sorted(all_results, key=lambda x: x['quality']['fidelity'], reverse=True):
        desc = result['description']
        fidelity = result['quality']['fidelity']
        residual = result['quality']['residual_error']
        cost = result['final_cost']

        is_good = fidelity > 0.95 and residual < 1.0
        marker = "🏆" if result == best_result else "✅" if is_good else "❌"
        print(f"{desc:<25} {fidelity:<10.6f} {residual:<10.6f} {cost:<12.6f} {marker}")

    # Visualizaciones
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Fidelidad vs Residual scatter plot
    fidelities = [r['quality']['fidelity'] for r in all_results]
    residuals = [r['quality']['residual_error'] for r in all_results]

    scatter = ax1.scatter(residuals, fidelities, c=range(len(all_results)), cmap='viridis', s=100, alpha=0.7)
    ax1.set_xlabel('Residual Error')
    ax1.set_ylabel('Fidelity')
    ax1.set_title('Fidelity vs Residual Error')
    ax1.grid(True, alpha=0.3)

    # Marcar el mejor punto
    if best_result:
        best_fid = best_result['quality']['fidelity']
        best_res = best_result['quality']['residual_error']
        ax1.scatter([best_res], [best_fid], c='red', s=200, marker='*', label='Best Overall', edgecolor='black', linewidth=2)
        ax1.legend()

    # 2. Convergencia de las mejores configuraciones
    top_3 = sorted(all_results, key=lambda x: x['quality']['fidelity'], reverse=True)[:3]

    for i, result in enumerate(top_3):
        ax2.plot(result['cost_history'], linewidth=2, label=f"{result['description']} (F={result['quality']['fidelity']:.3f})")

    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Cost')
    ax2.set_title('Convergence of Top 3 Configurations')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Distribución de fidelidades
    ax3.hist(fidelities, bins=10, alpha=0.7, color='green', edgecolor='black')
    ax3.axvline(x=np.mean(fidelities), color='red', linestyle='--', label=f'Mean: {np.mean(fidelities):.3f}')
    if best_result:
        ax3.axvline(x=best_result['quality']['fidelity'], color='gold', linestyle='-', linewidth=3, label='Best')
    ax3.set_xlabel('Fidelity')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Fidelities')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Mejores probabilidades cuánticas vs clásicas
    if best_result:
        c_probs = best_result['quality']['classical_probs']
        q_probs = best_result['quality']['quantum_probs']

        x_pos = np.arange(len(c_probs))
        width = 0.35

        ax4.bar(x_pos - width / 2, c_probs, width, label='Classical', color='blue', alpha=0.7)
        ax4.bar(x_pos + width / 2, q_probs, width, label='Quantum', color='green', alpha=0.7)

        ax4.set_xlabel('Basis States')
        ax4.set_ylabel('Probability')
        ax4.set_title(f'Best Solution: {best_result["description"]}')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f'|{i:03b}⟩' for i in range(len(c_probs))])
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig


def run_refinement_mode():
    """Ejecuta el modo de refinamiento completo."""
    best_refined, all_refined = refine_best_configuration()

    if best_refined:
        print(f"\n🏆 BEST REFINED SOLUTION:")
        print(f"  Configuration: {best_refined['description']}")
        print(f"  Details: {best_refined['config_details']}")
        print(f"  Fidelity: {best_refined['quality']['fidelity']:.8f}")
        print(f"  Residual Error: {best_refined['quality']['residual_error']:.8f}")
        print(f"  Final Cost: {best_refined['final_cost']:.8f}")

        # Análisis detallado
        analyze_refinement_results(best_refined, all_refined)

        # Comparación final detallada
        print(f"\n🔬 DETAILED ANALYSIS OF BEST REFINED SOLUTION:")
        comparison_data = compare_classical_quantum_solution(best_refined)

        # Guardar resultados refinados
        filename = save_complete_results(best_refined, comparison_data, mode="refinement")

        return best_refined, filename
    else:
        print("❌ No refined solutions found")
        return None, None


def compare_optimizers_and_ansatzes():
    """Compara diferentes combinaciones de optimizadores y ansätze."""

    # Configuraciones a probar
    optimizer_list = ["adam", "rmsprop", "nesterov", "adagrad"]  # Mejores para VQLS
    ansatz_list = ["layered", "hardware_efficient"]  # Más expresivos

    all_results = []

    print("=" * 60)
    print("SYSTEMATIC COMPARISON OF OPTIMIZERS AND ANSÄTZE")
    print("=" * 60)

    for optimizer_name in optimizer_list:
        for ansatz_name in ansatz_list:
            best_result = None
            results_for_config = []

            # Probar múltiples semillas para esta configuración
            for seed in range(N_RANDOM_SEEDS):
                try:
                    result = run_single_optimization(optimizer_name, ansatz_name, seed)
                    results_for_config.append(result)

                    if best_result is None or result["final_cost"] < best_result["final_cost"]:
                        best_result = result

                except Exception as e:
                    print(f"  ERROR with {optimizer_name}+{ansatz_name} (seed={seed}): {e}")
                    continue

            if best_result:
                best_result["all_seeds_results"] = results_for_config
                all_results.append(best_result)
                print(f"  BEST for {optimizer_name}+{ansatz_name}: {best_result['final_cost']:.7f}")

    return all_results


def analyze_and_save_results(results):
    """Analiza y guarda los resultados."""

    # Encontrar el mejor resultado global
    best_overall = min(results, key=lambda x: x["final_cost"])

    print("\n" + "=" * 60)
    print("ANALYSIS OF RESULTS")
    print("=" * 60)

    print(f"\nBEST OVERALL CONFIGURATION:")
    print(f"  Optimizer: {best_overall['optimizer']}")
    print(f"  Ansatz: {best_overall['ansatz']}")
    print(f"  Final Cost: {best_overall['final_cost']:.7f}")
    print(f"  Seed: {best_overall['seed']}")
    print(f"  Parameters: {best_overall['n_params']}")

    # Ranking por optimizador
    print(f"\nRANKING BY OPTIMIZER:")
    optimizer_avg = {}
    for result in results:
        opt = result["optimizer"]
        if opt not in optimizer_avg:
            optimizer_avg[opt] = []
        optimizer_avg[opt].append(result["final_cost"])

    for opt, costs in sorted(optimizer_avg.items(), key=lambda x: np.mean(x[1])):
        avg_cost = np.mean(costs)
        std_cost = np.std(costs)
        print(f"  {opt:12s}: {avg_cost:.6f} ± {std_cost:.6f} (n={len(costs)})")

    # Ranking por ansatz
    print(f"\nRANKING BY ANSATZ:")
    ansatz_avg = {}
    for result in results:
        ans = result["ansatz"]
        if ans not in ansatz_avg:
            ansatz_avg[ans] = []
        ansatz_avg[ans].append(result["final_cost"])

    for ans, costs in sorted(ansatz_avg.items(), key=lambda x: np.mean(x[1])):
        avg_cost = np.mean(costs)
        std_cost = np.std(costs)
        print(f"  {ans:18s}: {avg_cost:.6f} ± {std_cost:.6f} (n={len(costs)})")

    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"vqls_comparison_{timestamp}.json"

    # Helper mejorado para convertir números complejos y evitar referencias circulares
    def convert_complex(obj):
        if isinstance(obj, complex):
            return {"real": float(obj.real), "imag": float(obj.imag)}
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj

    # Crear versión limpia de los resultados sin referencias circulares
    clean_results = []
    for result in results:
        clean_result = {"optimizer": result["optimizer"], "ansatz": result["ansatz"], "seed": result["seed"],
                        "learning_rate": result["learning_rate"], "final_cost": float(result["final_cost"]), "n_params": result["n_params"],
                        "cost_history": [float(c) for c in result["cost_history"][-50:]]}
        # Solo incluir final_weights si no es muy grande
        if result["n_params"] <= 20:
            clean_result["final_weights"] = [float(w) for w in result["final_weights"]]
        clean_results.append(clean_result)

    # Crear best_overall limpio
    clean_best_overall = {"optimizer": best_overall["optimizer"], "ansatz": best_overall["ansatz"], "seed": best_overall["seed"],
                          "learning_rate": best_overall["learning_rate"], "final_cost": float(best_overall["final_cost"]),
                          "n_params": best_overall["n_params"], "cost_history": [float(c) for c in best_overall["cost_history"]]}
    if best_overall["n_params"] <= 20:
        clean_best_overall["final_weights"] = [float(w) for w in best_overall["final_weights"]]

    experiment_data = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "best_overall": clean_best_overall, "all_results": clean_results,
                       "summary": {"optimizer_ranking": {opt: {"mean": float(np.mean(costs)), "std": float(np.std(costs))} for opt, costs in
                                                         optimizer_avg.items()},
                                   "ansatz_ranking": {ans: {"mean": float(np.mean(costs)), "std": float(np.std(costs))} for ans, costs in
                                                      ansatz_avg.items()}},
                       "configuration": {"n_qubits": n_qubits, "steps": steps, "eta": eta, "coefficients": c.tolist(),
                                         "b_vector": [convert_complex(x) for x in b], "usar_coste_global": USAR_COSTE_GLOBAL}}

    try:
        with open(filename, "w") as f:
            json.dump(experiment_data, f, indent=2, default=convert_complex)
        print(f"\nResults saved to: {filename}")
    except Exception as e:
        print(f"\nError saving JSON: {e}")
        # Guardar versión mínima si falla
        minimal_data = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "best_config": f"{best_overall['optimizer']} + {best_overall['ansatz']}", "best_cost": float(best_overall["final_cost"]),
                        "optimizer_ranking": {opt: float(np.mean(costs)) for opt, costs in optimizer_avg.items()},
                        "ansatz_ranking": {ans: float(np.mean(costs)) for ans, costs in ansatz_avg.items()}}
        backup_filename = f"vqls_minimal_{timestamp}.json"
        with open(backup_filename, "w") as f:
            json.dump(minimal_data, f, indent=2)
        print(f"Minimal results saved to: {backup_filename}")
        filename = backup_filename

    return best_overall, filename


def plot_comparison_results(results):
    """Visualiza los resultados de la comparación."""

    # Crear figura con subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Boxplot por optimizador
    optimizer_data = {}
    for result in results:
        opt = result["optimizer"]
        if opt not in optimizer_data:
            optimizer_data[opt] = []
        optimizer_data[opt].append(result["final_cost"])

    ax1.boxplot(optimizer_data.values(), tick_labels=optimizer_data.keys())
    ax1.set_title("Final Cost by Optimizer")
    ax1.set_ylabel("Final Cost")
    ax1.tick_params(axis='x', rotation=45)

    # 2. Boxplot por ansatz
    ansatz_data = {}
    for result in results:
        ans = result["ansatz"]
        if ans not in ansatz_data:
            ansatz_data[ans] = []
        ansatz_data[ans].append(result["final_cost"])

    ax2.boxplot(ansatz_data.values(), tick_labels=ansatz_data.keys())
    ax2.set_title("Final Cost by Ansatz")
    ax2.set_ylabel("Final Cost")
    ax2.tick_params(axis='x', rotation=45)

    # 3. Convergencia del mejor resultado
    best_result = min(results, key=lambda x: x["final_cost"])
    ax3.plot(best_result["cost_history"], 'b-', linewidth=2)
    ax3.set_title(f"Best Convergence: {best_result['optimizer']} + {best_result['ansatz']}")
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Cost")
    ax3.grid(True, alpha=0.3)

    # 4. Heatmap de costes finales
    optimizers = sorted(set(r["optimizer"] for r in results))
    ansatzes = sorted(set(r["ansatz"] for r in results))

    cost_matrix = np.full((len(optimizers), len(ansatzes)), np.nan)
    for result in results:
        i = optimizers.index(result["optimizer"])
        j = ansatzes.index(result["ansatz"])
        cost_matrix[i, j] = result["final_cost"]

    im = ax4.imshow(cost_matrix, cmap='viridis', aspect='auto')
    ax4.set_xticks(range(len(ansatzes)))
    ax4.set_yticks(range(len(optimizers)))
    ax4.set_xticklabels(ansatzes, rotation=45)
    ax4.set_yticklabels(optimizers)
    ax4.set_title("Cost Heatmap (Optimizer vs Ansatz)")

    # Agregar colorbar
    plt.colorbar(im, ax=ax4)

    # Agregar valores en el heatmap
    for i in range(len(optimizers)):
        for j in range(len(ansatzes)):
            if not np.isnan(cost_matrix[i, j]):
                ax4.text(j, i, f'{cost_matrix[i, j]:.4f}', ha='center', va='center', color='white', fontsize=8)

    plt.tight_layout()
    plt.show()

    return fig


#
# EJECUCIÓN PRINCIPAL CON SELECCIÓN DE MODO
#

if __name__ == "__main__":
    print("=" * 70)
    print("🔬 VQLS OPTIMIZATION SUITE")
    print("=" * 70)

    # Verificar qué modo está activado
    active_modes = []
    if MULTIPLE_RUNS:
        active_modes.append("MULTIPLE_RUNS")
    if SINGLE_MODE:
        active_modes.append("SINGLE_MODE")
    if REFINEMENT_MODE:
        active_modes.append("REFINEMENT_MODE")

    if len(active_modes) != 1:
        print("❌ ERROR: Exactly one mode must be True!")
        print("Current modes:", active_modes)
        print("Please set only one of: MULTIPLE_RUNS, SINGLE_MODE, REFINEMENT_MODE to True")
        exit(1)

    active_mode = active_modes[0]
    print(f"🎯 Running in {active_mode}")
    print("=" * 70)

    if MULTIPLE_RUNS:
        # Comparación sistemática
        print("🔄 SYSTEMATIC COMPARISON MODE")
        results = compare_optimizers_and_ansatzes()

        if results:
            best_result, filename = analyze_and_save_results(results)
            plot_comparison_results(results)

            print(f"\n🎉 Experiment completed!")
            print(f"Best configuration: {best_result['optimizer']} + {best_result['ansatz']}")
            print(f"Final cost: {best_result['final_cost']:.7f}")
        else:
            print("No successful results obtained.")

    elif SINGLE_MODE:
        # Ejecución individual (modo original)
        print("🎯 SINGLE EXPERIMENT MODE")

        optimizer_name = "nesterov"
        ansatz_name = "hardware_efficient"

        print(f"Running optimization with {optimizer_name} + {ansatz_name}...")
        result = run_single_optimization(optimizer_name, ansatz_name, rng_seed)

        # Mostrar convergencia de la optimización
        plt.figure(figsize=(10, 6))
        plt.plot(result["cost_history"], "g", linewidth=2)
        plt.ylabel("Cost Function")
        plt.xlabel("Optimization steps")
        plt.title(f"VQLS Optimization: {optimizer_name} + {ansatz_name}")
        plt.grid(True, alpha=0.3)
        plt.show()

        print(f"Final optimization cost: {result['final_cost']:.7f}")

        # Comparación clásica vs cuántica
        comparison_data = compare_classical_quantum_solution(result)

        # Guardar resultados completos
        filename = save_complete_results(result, comparison_data, mode="single")

        print(f"\n🎉 Single experiment completed successfully!")
        print(f"Optimization cost: {result['final_cost']:.7f}")
        print(f"Classical-Quantum fidelity: {comparison_data['fidelity']:.6f}")
        print(f"Results saved to: {filename}")

    elif REFINEMENT_MODE:
        # Modo de refinamiento
        print("🔬 REFINEMENT MODE")
        print("This mode will systematically refine the best known configuration:")
        print("  - Base: nesterov + hardware_efficient")
        print("  - Test different learning rates, seeds, steps, and q_delta values")
        print("  - Find optimal parameters for maximum fidelity and minimum residual error")
        print()

        best_refined, refined_filename = run_refinement_mode()

        if best_refined:
            print(f"\n🎉 REFINEMENT COMPLETED SUCCESSFULLY!")
            print(f"📈 Best configuration found: {best_refined['description']}")
            print(f"📊 Final fidelity: {best_refined['quality']['fidelity']:.6f}")
            print(f"📉 Final residual error: {best_refined['quality']['residual_error']:.6f}")
            print(f"💾 Detailed results saved to: {refined_filename}")

            # Mostrar recomendaciones finales
            fidelity = best_refined['quality']['fidelity']
            residual = best_refined['quality']['residual_error']

            print(f"\n📋 QUALITY ASSESSMENT:")
            if fidelity > 0.98:
                print("✅ Excellent fidelity (>0.98)")
            elif fidelity > 0.95:
                print("✅ Good fidelity (>0.95)")
            else:
                print("⚠️  Moderate fidelity (<0.95)")

            if residual < 0.1:
                print("✅ Excellent residual error (<0.1)")
            elif residual < 0.5:
                print("✅ Good residual error (<0.5)")
            else:
                print("⚠️  High residual error (>0.5)")

        else:
            print("❌ Refinement failed to find improved solutions")
            print("Consider:")
            print("  - Different base configurations")
            print("  - Modified problem parameters")
            print("  - Alternative ansätze designs")

    print(f"\n" + "=" * 70)
    print("🏁 VQLS SUITE EXECUTION COMPLETED")
    print("=" * 70)
