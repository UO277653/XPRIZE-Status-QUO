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
steps = 300  # Number of optimization steps
eta = 0.8  # Learning rate
q_delta = 0.01  # Initial spread of random quantum weights
rng_seed = 0  # Seed for random number generator
n_layers = 2

# --------------------------------------------------------------------------
# True: Usa la función de coste "Global". Es mucho más rápida (menos simulaciones por paso).
# False: Usa la función de coste "Local" original (muy lenta pero teóricamente robusta).
USAR_COSTE_GLOBAL = False  # Cambiado a True por defecto para rapidez
# --------------------------------------------------------------------------

# Opción para modo analítico/muestreo en el resultado final (ya implementada)
MODO_ANALITICO = False
n_shots = 10 ** 6

# NUEVO: Configuración para múltiples experimentos
MULTIPLE_RUNS = True  # Para probar múltiples configuraciones
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


def run_single_optimization(optimizer_name, ansatz_name, seed, learning_rate=eta):
    """Ejecuta una optimización individual."""
    print(f"\n--- Running: {optimizer_name} + {ansatz_name} (seed={seed}) ---")

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
    w = q_delta * np.random.randn(n_params, requires_grad=True)

    # Elegir función de coste
    if USAR_COSTE_GLOBAL:
        cost_function = lambda weights: cost_glob(weights, variational_block_fn)
        cost_name = "Global"
    else:
        cost_function = lambda weights: cost_loc(weights, variational_block_fn)
        cost_name = "Local"

    # Optimización
    cost_history = []
    for it in range(steps):
        if optimizer_name == "spsa":
            # SPSA tiene una interfaz ligeramente diferente
            w = opt.step(cost_function, w)
            cost = cost_function(w)
        else:
            w, cost = opt.step_and_cost(cost_function, w)

        cost_history.append(float(cost))  # Asegurar que es float

        if it % 20 == 0 or it < 5:
            print(f"  Step {it:3d}: Cost = {cost:9.15f}")

    final_cost = cost_history[-1]
    print(f"  Final cost: {final_cost:.7f}")

    return {"optimizer": optimizer_name, "ansatz": ansatz_name, "seed": seed, "learning_rate": learning_rate, "final_cost": final_cost,
            "cost_history": cost_history, "n_params": n_params, "final_weights": w.tolist() if hasattr(w, 'tolist') else list(w),
            "variational_block_fn": variational_block_fn}


def compare_classical_quantum_solution(result):
    """
    Compara la solución clásica con la cuántica y calcula la fidelidad.
    """
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
        dev_x = qml.device("lightning.qubit", wires=n_qubits)

        @qml.qnode(dev_x)
        def get_solution_statevector(weights):
            variational_block_fn(weights)
            return qml.state()

        state_vector = get_solution_statevector(w)
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

    # Visualización
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


def save_complete_results(result, comparison_data):
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
    complete_results = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "optimization": {"optimizer": result["optimizer"], "ansatz": result["ansatz"], "seed": result["seed"],
                                         "learning_rate": result["learning_rate"], "final_cost": result["final_cost"],
                                         "cost_history": result["cost_history"], "n_params": result["n_params"]}, "comparison": comparison_data,
                        "configuration": {"n_qubits": n_qubits, "steps": steps, "eta": eta, "coefficients": c.tolist(),
                                          "b_vector": [convert_complex(x) for x in b], "usar_coste_global": USAR_COSTE_GLOBAL,
                                          "modo_analitico": MODO_ANALITICO}}

    # Leer experimentos existentes
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


def compare_optimizers_and_ansatzes():
    """Compara diferentes combinaciones de optimizadores y ansätze."""

    # Configuraciones a probar
    optimizer_list = ["adam", "rmsprop", "nesterov", "adagrad"]  # Mejores para VQLS
    ansatz_list = ["complex", "layered", "hardware_efficient"]  # Más expresivos

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
                        "cost_history": [float(c) for c in result["cost_history"][-50:]]  # Solo últimos 50 para ahorrar espacio
                        }
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

    ax1.boxplot(optimizer_data.values(), labels=optimizer_data.keys())
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

    ax2.boxplot(ansatz_data.values(), labels=ansatz_data.keys())
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
# EJECUCIÓN PRINCIPAL
#

if __name__ == "__main__":
    if MULTIPLE_RUNS:
        # Comparación sistemática
        results = compare_optimizers_and_ansatzes()

        if results:
            best_result, filename = analyze_and_save_results(results)
            plot_comparison_results(results)

            print(f"\n🎉 Experiment completed!")
            print(f"Best configuration: {best_result['optimizer']} + {best_result['ansatz']}")
            print(f"Final cost: {best_result['final_cost']:.7f}")
        else:
            print("No successful results obtained.")

    else:
        # Ejecución individual (modo original) - AHORA CON COMPARACIÓN COMPLETA
        print("=" * 60)
        print("SINGLE EXPERIMENT MODE WITH FULL COMPARISON")
        print("=" * 60)

        optimizer_name = "adam"
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
        filename = save_complete_results(result, comparison_data)

        print(f"\n🎉 Single experiment completed successfully!")
        print(f"Optimization cost: {result['final_cost']:.7f}")
        print(f"Classical-Quantum fidelity: {comparison_data['fidelity']:.6f}")
        print(f"Results saved to: {filename}")
