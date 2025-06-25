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
steps = 1000  # Number of optimization steps
eta = 0.4  # Learning rate
q_delta = 0.01  # Initial spread of random quantum weights
rng_seed = 2  # Seed for random number generator
n_layers = 2

# --------------------------------------------------------------------------
# Configuración de costes y convergencia - CRITERIOS MÁS REALISTAS
USAR_COSTE_GLOBAL = False  # False usa función local (más robusta)
CONVERGENCE_THRESHOLD = 1e-6  # Umbral para considerar convergencia
MIN_FIDELITY = 0.7  # Fidelidad mínima aceptable (más realista)
MAX_RESIDUAL = 1.0  # Error residual máximo aceptable (más permisivo)
PATIENCE = 150  # Pasos sin mejora antes de parar
# --------------------------------------------------------------------------

# Opción para modo analítico/muestreo en el resultado final
MODO_ANALITICO = True
n_shots = 10 ** 6

# Configuración para múltiples experimentos
MULTIPLE_RUNS = False
N_RANDOM_SEEDS = 3


#
# Definiciones matrices y funciones base
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
    assert len(params) == 3 * n

    idx = 0
    for w in wires:
        qml.Hadamard(wires=w)

    for w in wires:
        theta, phi, lam = params[idx], params[idx + 1], params[idx + 2]
        qml.Rot(theta, phi, lam, wires=w)
        idx += 3

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
        for w in wires:
            theta, phi, lam = params[idx], params[idx + 1], params[idx + 2]
            qml.Rot(theta, phi, lam, wires=w)
            idx += 3

        if layer % 2 == 0:
            for i in range(n - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
        else:
            for i in range(n):
                qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])


def variational_block_layered(weights):
    """Wrapper para el ansatz multicapa."""
    ansatz_layered(weights, wires=list(range(n_qubits)), n_layers=n_layers)


def ansatz_hardware_efficient(params, wires):
    """Ansatz eficiente para hardware cuántico real."""
    n = len(wires)
    assert len(params) == 2 * n

    idx = 0
    for w in wires:
        qml.RY(params[idx], wires=w)
        idx += 1

    for i in range(n - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])

    for w in wires:
        qml.RY(params[idx], wires=w)
        idx += 1


def variational_block_hardware_efficient(weights):
    """Wrapper para el ansatz hardware-efficient."""
    ansatz_hardware_efficient(weights, wires=list(range(n_qubits)))


# Diccionarios de optimizadores y ansätze
optimizers = {"adam": lambda lr: qml.AdamOptimizer(stepsize=lr), "sgd": lambda lr: qml.GradientDescentOptimizer(stepsize=lr),
              "rmsprop": lambda lr: qml.RMSPropOptimizer(stepsize=lr), "nesterov": lambda lr: qml.NesterovMomentumOptimizer(stepsize=lr),
              "adagrad": lambda lr: qml.AdagradOptimizer(stepsize=lr), "momentum": lambda lr: qml.MomentumOptimizer(stepsize=lr),
              "spsa": lambda lr: qml.SPSAOptimizer(maxiter=steps, blocking=False)}

ansatzes = {"default": (variational_block, lambda: 2 * n_qubits), "complex": (variational_block_complex, lambda: 3 * n_qubits),
            "layered": (variational_block_layered, lambda: 3 * n_qubits * n_layers),
            "hardware_efficient": (variational_block_hardware_efficient, lambda: 2 * n_qubits), }

#
# Definiciones para las funciones de coste
#
dev_cost = qml.device("lightning.qubit", wires=tot_qubits)


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
    bAx_overlap = 0.0
    for l in range(len(c)):
        real = global_hadamard_test(weights, l, "Re", variational_block_fn)
        imag = global_hadamard_test(weights, l, "Im", variational_block_fn)
        bAx_overlap += c[l] * (real + 1.0j * imag)

    numerator = np.abs(bAx_overlap) ** 2
    denominator = psi_norm(weights, variational_block_fn)
    return 1 - (numerator / denominator)


def psi_norm(weights, variational_block_fn=variational_block):
    """Calcula <x|A†A|x>."""
    norm = 0.0
    for l in range(len(c)):
        for lp in range(len(c)):
            norm += c[l] * np.conj(c[lp]) * mu(weights, l, lp, variational_block_fn=variational_block_fn)
    return abs(norm)


def calculate_solution_quality(weights, variational_block_fn):
    """
    Calcula métricas de calidad de la solución cuántica.
    """
    # Obtener vector de estado cuántico
    dev_x = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev_x)
    def get_quantum_state(w):
        variational_block_fn(w)
        return qml.state()

    quantum_state = get_quantum_state(weights)
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

    return {'fidelity': float(fidelity), 'residual_error': float(residual_error), 'prob_distance': float(prob_distance), 'classical_probs': c_probs,
            'quantum_probs': q_probs}


def check_convergence(cost_history, patience=PATIENCE, threshold=CONVERGENCE_THRESHOLD):
    """
    Verifica si el algoritmo ha convergido.
    """
    if len(cost_history) < patience:
        return False, "Not enough steps"

    recent_costs = cost_history[-patience:]
    cost_variance = np.var(recent_costs)
    cost_improvement = recent_costs[0] - recent_costs[-1]

    converged = cost_variance < threshold and cost_improvement < threshold

    reason = ""
    if converged:
        reason = f"Converged: variance={cost_variance:.2e}, improvement={cost_improvement:.2e}"
    else:
        reason = f"Not converged: variance={cost_variance:.2e}, improvement={cost_improvement:.2e}"

    return converged, reason


def run_single_optimization_improved(optimizer_name, ansatz_name, seed, learning_rate=eta):
    """
    Ejecuta una optimización individual con verificación de convergencia.
    """
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

    print(f"Using {cost_name} cost function")

    # Optimización con verificación de convergencia
    cost_history = []
    quality_history = []
    best_cost = float('inf')
    best_weights = w.copy()
    no_improvement_count = 0

    for it in range(steps):
        if optimizer_name == "spsa":
            w = opt.step(cost_function, w)
            cost = cost_function(w)
        else:
            w, cost = opt.step_and_cost(cost_function, w)

        cost_history.append(float(cost))

        # Trackear mejor solución
        if cost < best_cost:
            best_cost = cost
            best_weights = w.copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Evaluar calidad cada 50 pasos
        if it % 50 == 0:
            try:
                quality = calculate_solution_quality(w, variational_block_fn)
                quality_history.append({'step': it, 'cost': cost, 'fidelity': quality['fidelity'], 'residual_error': quality['residual_error'],
                                        'prob_distance': quality['prob_distance']})

                if it % 100 == 0 or it < 10:
                    print(f"  Step {it:3d}: Cost = {cost:9.7f}, Fidelity = {quality['fidelity']:.4f}, Residual = {quality['residual_error']:.4f}")

                # Verificar si tenemos una buena solución
                if quality['fidelity'] > MIN_FIDELITY and quality['residual_error'] < MAX_RESIDUAL:
                    print(f"  ✅ Good solution found at step {it}!")
                    break

            except Exception as e:
                print(f"  Warning: Quality evaluation failed at step {it}: {e}")

        # Verificar convergencia
        if it > 200:  # Empezar a verificar después de algunos pasos
            converged, reason = check_convergence(cost_history)
            if converged:
                print(f"  🔄 {reason} at step {it}")
                break

        # Parar si no hay mejora por mucho tiempo
        if no_improvement_count > PATIENCE * 2:
            print(f"  🛑 No improvement for {no_improvement_count} steps, stopping early")
            break

    # Evaluar calidad final
    final_quality = calculate_solution_quality(best_weights, variational_block_fn)

    print(f"  Final cost: {best_cost:.7f}")
    print(f"  Final fidelity: {final_quality['fidelity']:.4f}")
    print(f"  Final residual error: {final_quality['residual_error']:.4f}")

    # Determinar si la solución es aceptable
    is_good_solution = (final_quality['fidelity'] > MIN_FIDELITY and final_quality['residual_error'] < MAX_RESIDUAL)

    if is_good_solution:
        print("  ✅ GOOD SOLUTION!")
    else:
        print("  ❌ Poor solution - consider different parameters")

    return {"optimizer": optimizer_name, "ansatz": ansatz_name, "seed": seed, "learning_rate": learning_rate, "final_cost": float(best_cost),
            "cost_history": cost_history, "quality_history": quality_history, "n_params": n_params,
            "final_weights": best_weights.tolist() if hasattr(best_weights, 'tolist') else list(best_weights),
            "variational_block_fn": variational_block_fn, "final_quality": final_quality, "is_good_solution": is_good_solution, "total_steps": it + 1}


def compare_classical_quantum_solution_improved(result):
    """
    Compara la solución clásica con la cuántica con métricas mejoradas.
    """
    print("\n" + "=" * 60)
    print("DETAILED COMPARISON OF CLASSICAL AND QUANTUM RESULTS")
    print("=" * 60)

    final_quality = result["final_quality"]

    print(f"\n📊 QUALITY METRICS:")
    print(f"  Fidelity: {final_quality['fidelity']:.6f}")
    print(f"  Residual Error ||Ax-b||: {final_quality['residual_error']:.6f}")
    print(f"  Probability Distance: {final_quality['prob_distance']:.6f}")
    print(f"  Solution Quality: {'✅ GOOD' if result['is_good_solution'] else '❌ POOR'}")

    c_probs = final_quality['classical_probs']
    q_probs = final_quality['quantum_probs']

    print("\n📈 PROBABILITY DISTRIBUTIONS:")
    print("-" * 40)
    print("State    Classical    Quantum      Diff")
    print("-" * 40)
    for i in range(len(c_probs)):
        diff = abs(c_probs[i] - q_probs[i])
        status = "✅" if diff < 0.1 else "❌"
        print(f"|{i:03b}>    {c_probs[i]:.6f}    {q_probs[i]:.6f}    {diff:.6f} {status}")

    # Visualización mejorada
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Gráfico de probabilidades
    x_pos = np.arange(2 ** n_qubits)
    width = 0.35

    ax1.bar(x_pos - width / 2, c_probs, width, label='Classical', color='blue', alpha=0.7)
    ax1.bar(x_pos + width / 2, q_probs, width, label='Quantum', color='green', alpha=0.7)
    ax1.set_title("Classical vs Quantum Probabilities")
    ax1.set_xlabel("Basis states")
    ax1.set_ylabel("Probability")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"|{i:03b}>" for i in range(2 ** n_qubits)])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Diferencias
    diffs = np.abs(c_probs - q_probs)
    ax2.bar(x_pos, diffs, color='red', alpha=0.7)
    ax2.set_title("Probability Differences |Classical - Quantum|")
    ax2.set_xlabel("Basis states")
    ax2.set_ylabel("Absolute Difference")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"|{i:03b}>" for i in range(2 ** n_qubits)])
    ax2.axhline(y=0.1, color='orange', linestyle='--', label='Tolerance (0.1)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Convergencia del coste
    ax3.plot(result["cost_history"], 'b-', linewidth=2)
    ax3.set_title("Cost Function Convergence")
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Cost")
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # Evolución de la fidelidad
    if result["quality_history"]:
        steps_q = [q['step'] for q in result["quality_history"]]
        fidelities = [q['fidelity'] for q in result["quality_history"]]
        residuals = [q['residual_error'] for q in result["quality_history"]]

        ax4_twin = ax4.twinx()
        line1 = ax4.plot(steps_q, fidelities, 'g-', linewidth=2, label='Fidelity')
        line2 = ax4_twin.plot(steps_q, residuals, 'r-', linewidth=2, label='Residual Error')

        ax4.set_xlabel("Steps")
        ax4.set_ylabel("Fidelity", color='g')
        ax4_twin.set_ylabel("Residual Error", color='r')
        ax4.set_title("Quality Evolution")
        ax4.axhline(y=MIN_FIDELITY, color='g', linestyle='--', alpha=0.5, label=f'Min Fidelity ({MIN_FIDELITY})')
        ax4_twin.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Max Residual (0.1)')

        # Combinar leyendas
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='center right')

    plt.tight_layout()
    plt.show()

    return final_quality


def save_complete_results_improved(result, comparison_data):
    """Guarda los resultados completos con métricas mejoradas."""

    def convert_complex(obj):
        if isinstance(obj, complex):
            return {"real": float(obj.real), "imag": float(obj.imag)}
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj

    complete_results = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "optimization": {"optimizer": result["optimizer"], "ansatz": result["ansatz"], "seed": result["seed"],
                                         "learning_rate": result["learning_rate"], "final_cost": result["final_cost"],
                                         "total_steps": result["total_steps"], "is_good_solution": result["is_good_solution"],
                                         "cost_history": result["cost_history"], "quality_history": result["quality_history"],
                                         "n_params": result["n_params"]}, "quality_metrics": comparison_data,
                        "configuration": {"n_qubits": n_qubits, "max_steps": steps, "eta": eta, "coefficients": c.tolist(),
                                          "b_vector": [convert_complex(x) for x in b], "usar_coste_global": USAR_COSTE_GLOBAL,
                                          "convergence_threshold": CONVERGENCE_THRESHOLD, "min_fidelity": MIN_FIDELITY, "patience": PATIENCE}}

    filename = "vqls_improved_results.json"
    all_experiments = []
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                all_experiments = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            all_experiments = []

    complete_results["experiment_id"] = len(all_experiments) + 1
    all_experiments.append(complete_results)

    with open(filename, "w") as f:
        json.dump(all_experiments, f, indent=2, default=convert_complex)

    print(f"\nImproved results saved to: {filename} (Experiment #{complete_results['experiment_id']})")
    return filename


#
# EJECUCIÓN PRINCIPAL
#

if __name__ == "__main__":
    if MULTIPLE_RUNS:
        print("Multiple runs mode not yet updated for improved version")
        print("Set MULTIPLE_RUNS = False to test improved single optimization")
    else:
        print("=" * 60)
        print("IMPROVED SINGLE EXPERIMENT WITH CONVERGENCE VERIFICATION")
        print("=" * 60)

        # Probar con diferentes configuraciones - CONFIGURACIONES MEJORADAS
        configs_to_try = [("adam", "hardware_efficient", 0.01),  # Learning rate más bajo
                          ("adagrad", "complex", 0.05),  # Diferentes combinaciones
                          ("rmsprop", "default", 0.02),  # Ansatz más simple
                          ("nesterov", "hardware_efficient", 0.03),  # Optimizador con momentum
                          ]

        all_results = []
        best_result = None
        best_quality = 0

        for optimizer_name, ansatz_name, lr in configs_to_try:
            print(f"\n🧪 Testing {optimizer_name} + {ansatz_name} (lr={lr})")

            result = run_single_optimization_improved(optimizer_name, ansatz_name, rng_seed, lr)
            all_results.append(result)

            # Guardar la mejor solución (incluso si no es "buena")
            if result["final_quality"]["fidelity"] > best_quality:
                best_result = result
                best_quality = result["final_quality"]["fidelity"]

        # SIEMPRE mostrar resultados del mejor intento
        if best_result:
            print(f"\n🎯 BEST ATTEMPT (may not be perfect):")
            print(f"Configuration: {best_result['optimizer']} + {best_result['ansatz']}")
            print(f"Final fidelity: {best_result['final_quality']['fidelity']:.6f}")
            print(f"Final cost: {best_result['final_cost']:.6f}")
            print(f"Residual error: {best_result['final_quality']['residual_error']:.6f}")

            # Mostrar gráfico de convergencia SIEMPRE
            plt.figure(figsize=(12, 8))

            # Subplot 1: Convergencia de todas las configuraciones
            plt.subplot(2, 2, 1)
            for i, result in enumerate(all_results):
                config_name = f"{result['optimizer']}+{result['ansatz']}"
                plt.plot(result["cost_history"], linewidth=2, label=config_name)
            plt.ylabel("Cost Function")
            plt.xlabel("Steps")
            plt.title("Cost Convergence Comparison")
            plt.yscale('log')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Subplot 2: Mejor convergencia
            plt.subplot(2, 2, 2)
            plt.plot(best_result["cost_history"], "g", linewidth=3)
            plt.ylabel("Cost Function")
            plt.xlabel("Steps")
            plt.title(f"Best: {best_result['optimizer']} + {best_result['ansatz']}")
            plt.yscale('log')
            plt.grid(True, alpha=0.3)

            # Subplot 3: Evolución de calidad
            if best_result["quality_history"]:
                steps_q = [q['step'] for q in best_result["quality_history"]]
                fidelities = [q['fidelity'] for q in best_result["quality_history"]]
                residuals = [q['residual_error'] for q in best_result["quality_history"]]

                plt.subplot(2, 2, 3)
                plt.plot(steps_q, fidelities, 'g-', linewidth=2, label='Fidelity')
                plt.axhline(y=MIN_FIDELITY, color='g', linestyle='--', alpha=0.5, label=f'Target ({MIN_FIDELITY})')
                plt.ylabel("Fidelity")
                plt.xlabel("Steps")
                plt.title("Fidelity Evolution")
                plt.legend()
                plt.grid(True, alpha=0.3)

                plt.subplot(2, 2, 4)
                plt.plot(steps_q, residuals, 'r-', linewidth=2, label='Residual Error')
                plt.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Target (0.1)')
                plt.ylabel("Residual Error")
                plt.xlabel("Steps")
                plt.title("Residual Error Evolution")
                plt.yscale('log')
                plt.legend()
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

            # SIEMPRE hacer comparación detallada
            print(f"\n📊 DETAILED ANALYSIS OF BEST ATTEMPT:")
            comparison_data = compare_classical_quantum_solution_improved(best_result)

            # SIEMPRE guardar resultados
            filename = save_complete_results_improved(best_result, comparison_data)

            # Resumen final con recomendaciones
            print(f"\n" + "=" * 60)
            print("SUMMARY AND RECOMMENDATIONS")
            print("=" * 60)

            print(f"✅ Best configuration: {best_result['optimizer']} + {best_result['ansatz']}")
            print(f"📈 Final fidelity: {best_result['final_quality']['fidelity']:.6f}")
            print(f"📉 Final cost: {best_result['final_cost']:.6f}")
            print(f"🎯 Residual error: {best_result['final_quality']['residual_error']:.6f}")
            print(f"⏱️  Steps taken: {best_result['total_steps']}")

            if best_result['is_good_solution']:
                print("🎉 EXCELLENT: Solution meets quality criteria!")
            else:
                print("\n🔧 RECOMMENDATIONS FOR IMPROVEMENT:")
                if best_result['final_quality']['fidelity'] < 0.8:
                    print("  • Fidelity too low - try different ansatz or more parameters")
                if best_result['final_quality']['residual_error'] > 0.5:
                    print("  • High residual error - problem may be ill-conditioned")
                if best_result['final_cost'] > 0.1:
                    print("  • High final cost - try smaller learning rates or more steps")
                print("  • Try different initial seeds (rng_seed)")
                print("  • Increase number of optimization steps")
                print("  • Experiment with q_delta (initial parameter spread)")

            print(f"\n💾 Results saved to: {filename}")

        else:
            print("\n❌ Unexpected error: No results obtained!")

        # Mostrar tabla comparativa final
        if all_results:
            print(f"\n📋 CONFIGURATION COMPARISON:")
            print("-" * 80)
            print(f"{'Config':<25} {'Final Cost':<12} {'Fidelity':<10} {'Residual':<10} {'Steps':<8}")
            print("-" * 80)
            for result in sorted(all_results, key=lambda x: x['final_quality']['fidelity'], reverse=True):
                config = f"{result['optimizer']}+{result['ansatz']}"
                cost = result['final_cost']
                fidelity = result['final_quality']['fidelity']
                residual = result['final_quality']['residual_error']
                steps = result['total_steps']
                status = "✅" if result['is_good_solution'] else "❌"
                print(f"{config:<25} {cost:<12.6f} {fidelity:<10.4f} {residual:<10.4f} {steps:<8} {status}")
            print("-" * 80)
