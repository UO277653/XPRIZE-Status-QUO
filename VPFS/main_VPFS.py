import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pennylane as qml
from pennylane import numpy as np
from scipy.linalg import sqrtm, polar
from scipy.optimize import minimize
import torch
import matplotlib.pyplot as plt
import json
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
import warnings

# Suppress complex casting warnings for cleaner output
warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")

#
# Setting of the main hyper-parameters of the model
#

# VPFS Problem parameters
print("🚀 USING WELL-CONDITIONED Y MATRICES")

# Opción 1: Matriz regularizada
Y_real = np.array([[1, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 1]], dtype=complex)
regularization = 0.1  # Añadir regularización para mejorar condicionamiento
Y_real = Y_real + regularization * np.eye(4)  # Condition number ~35 (BUENO)

# Opción 2: Matriz óptima (matemáticamente bien condicionada)
eigenvals = [4.0, 3.0, 2.0, 1.5]  # Eigenvalues bien separados
Q = np.array([  # Matriz ortogonal
    [0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, -0.5], [0.5, 0.5, -0.5, -0.5], [0.5, -0.5, -0.5, 0.5]])
D = np.diag(eigenvals)
Y_complex = (Q @ D @ Q.T).astype(complex)  # Condition number ~2.67 (EXCELENTE)

# Añadir pequeña parte imaginaria para versión compleja
Y_complex = Y_complex + 1j * Y_complex * 0.05  # 5% de complejidad

print(f"📊 Y_real condition number: {np.linalg.cond(Y_real):.1f}")
print(f"📊 Y_complex condition number: {np.linalg.cond(Y_complex):.1f}")

# 🎯 TESTING BOTH CASES
TEST_COMPLEX_Y = True  # Set to True to test complex Y matrix

if TEST_COMPLEX_Y:
    Y = Y_complex
    print("🔬 USING WELL-CONDITIONED COMPLEX Y MATRIX - General electrical networks")
else:
    Y = Y_real
    print("🔬 USING WELL-CONDITIONED REAL Y MATRIX - Urban/rural networks")

V = np.array([1, 1.1, 0.95, 0.9])  # Real power reference (can be complex in future)
num_qubits = 2  # Number of qubits per register
total_wires = 2 * num_qubits + 1

# Optimization parameters
max_iters = 2000
tolerance = 1e-9  # Convergence tolerance
learning_rate = 0.4
radius = 0.1  # Parameter for amplitude ansatz
PEN_COEF_SCALE = 0.0  # Penalty coefficient scale
loss_option = 4

# Experimental configuration
rng_seed = 2
N_RANDOM_SEEDS = 5  # Increased for better statistics

# --------------------------------------------------------------------------
# EXECUTION MODES
MULTIPLE_RUNS = False  # For systematic comparison
SINGLE_MODE = False  # For individual experiment with multiple seeds
REFINEMENT_MODE = True  # For refining best configuration with robustness analysis
TURBO_MODE = False  # For high-speed parallel experiments
# --------------------------------------------------------------------------

# Speed optimization
USAR_OPTIMIZACION_RAPIDA = True  # Use fast optimizations when possible

global MODO_SILENCIOSO
MODO_SILENCIOSO = False  # Silent mode for batch experiments


#
# VPFS Algorithm Core Functions
#

def create_unitaries(Y, B):
    """Creates the unitary matrices Y_extended and U_b† (calculated from B)."""
    # Normalization and Y_extended calculation
    Y_norm = np.max(np.abs(np.linalg.eigvals(Y)))
    Y_normalized = Y / Y_norm
    sqrt_diff = sqrtm(np.eye(len(Y)) - Y_normalized @ Y_normalized)
    Y_extended = np.block([[Y_normalized, sqrt_diff], [sqrt_diff, -Y_normalized]])
    Y_extended, _ = polar(Y_extended)  # Take unitary part

    # Build U_b† based on B
    b = B / np.linalg.norm(B)
    U_b = np.eye(len(b), dtype=complex)
    U_b[:, 0] = b
    for i in range(1, len(b) - 1):
        v_vec = U_b[1:, i]
        for j in range(i):
            v_vec -= np.dot(U_b[1:, j].conj(), v_vec) * U_b[1:, j]
        v_vec = v_vec / np.linalg.norm(v_vec)
        U_b[1:, i] = v_vec
    U_b[0, -1] = 1
    U_b[-1, -1] = 0
    U_b_dagger = U_b.conj().T

    return Y_extended, U_b_dagger, Y_norm


#
# Ansatz Functions
#

def ansatz_amplitude(params):
    """Calculates vector v from parameter vector params (amplitude embedding)."""
    v = [1]  # First component fixed
    for i in range(len(params)):
        if radius > 0:
            v.append(np.sin(params[i]) * radius + 1)
        else:
            v.append(params[i] + 1)
    v = np.array(v)
    v = v / np.linalg.norm(v)
    return v


def ansatz_variational_block(weights, n_qubits):
    """Ansatz with entanglement and exactly 2^n - 1 parameters."""
    num_params = len(weights)
    assert num_params == 2 ** n_qubits - 1, f"Expected {2 ** n_qubits - 1} parameters, got {num_params}."

    # 1. Initialize in uniform superposition with Hadamard
    for idx in range(n_qubits):
        qml.Hadamard(wires=idx)

    # 2. Apply RY rotations with first n_qubits parameters
    for idx in range(n_qubits):
        qml.RY(weights[idx], wires=idx)

    # 3. Apply entanglement (CNOT in linear topology)
    for idx in range(n_qubits - 1):
        qml.CNOT(wires=[idx, idx + 1])

    # 4. Apply additional rotations to reach 2^n - 1 parameters
    extra_params = weights[n_qubits:]
    for idx, param in enumerate(extra_params):
        qml.RY(param, wires=idx % n_qubits)


def ansatz_complex_z(params, wires):
    """Complex ansatz using RZ rotations."""
    n_qubits_reg = len(wires)
    expected_params = 2 * n_qubits_reg
    assert len(params) == expected_params, f"Expected {expected_params} params for {n_qubits_reg} qubits, got {len(params)}"

    for wire in wires:  # Initial state preparation
        qml.Hadamard(wires=wire)

    param_idx = 0
    for wire in wires:  # First RZ layer
        qml.RZ(params[param_idx], wires=wire)
        param_idx += 1

    for i in range(n_qubits_reg - 1):  # Entanglement
        qml.CNOT(wires=[wires[i], wires[i + 1]])

    for wire in wires:  # Second RZ layer
        qml.RZ(params[param_idx], wires=wire)
        param_idx += 1


def ansatz_hardware_efficient(params, wires):
    """
    A standard hardware-efficient ansatz with layered rotations and entanglement.
    Each layer consists of RY rotations on each qubit, followed by a chain of CNOTs.
    """
    n_qubits = len(wires)
    # The number of parameters is expected to be n_layers * n_qubits
    params_per_layer = n_qubits

    for l in range(n_layers):
        # Rotation layer
        for i in range(n_qubits):
            qml.RY(params[l * params_per_layer + i], wires=wires[i])

        # Entanglement layer (linear chain)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])


def ansatz_strongly_entangling(params, wires):
    """A strongly entangling layered ansatz from PennyLane templates."""
    qml.StronglyEntanglingLayers(weights=params, wires=wires)


# Dictionary of available ansätze - CLEANED VERSION
n_layers = 2  # For standard layered ansatz

ansatzes = {# Original VPFS ansätze
    "amplitude": {"n_params": lambda: 2 ** num_qubits - 1, "description": "Amplitude embedding ansatz (NOT hardware realistic)"},
    "variational_block": {"n_params": lambda: 2 ** num_qubits - 1, "description": "Variational block with entanglement"},
    "complex_z": {"n_params": lambda: 2 * num_qubits, "description": "Complex ansatz with RZ rotations"},

    # Hardware-realistic ansätze
    "hardware_efficient": {"n_params": lambda: n_layers * num_qubits, "description": "Layered hardware-efficient ansatz (RY + CNOT)"},
    "strongly_entangling": {"n_params": lambda: qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=num_qubits)[0] *
                                                qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=num_qubits)[1] *
                                                qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=num_qubits)[2],
                            "description": "Strongly Entangling Layers ansatz"}}

# Dictionary of available optimizers (VQLS + VPFS combined)
optimizers_vpfs = {"basic": "Basic finite difference gradient descent", "analytic": "Analytic gradient with PennyLane autodiff",
                   "sequential": "Sequential parameter optimization", "cobyla": "COBYLA constrained optimization",
                   "adam": "Adam optimizer with PyTorch",  # VQLS optimizers added
                   "nesterov": "Nesterov Momentum Optimizer", "rmsprop": "RMSProp Optimizer", "adagrad": "Adagrad Optimizer",
                   "momentum": "Momentum Optimizer", "sgd": "Stochastic Gradient Descent", "spsa": "SPSA Optimizer"}

# PennyLane optimizers (from VQLS)
pennylane_optimizers = {"nesterov": lambda lr: qml.NesterovMomentumOptimizer(stepsize=lr), "rmsprop": lambda lr: qml.RMSPropOptimizer(stepsize=lr),
                        "adagrad": lambda lr: qml.AdagradOptimizer(stepsize=lr), "momentum": lambda lr: qml.MomentumOptimizer(stepsize=lr),
                        "sgd": lambda lr: qml.GradientDescentOptimizer(stepsize=lr),
                        "spsa": lambda lr: qml.SPSAOptimizer(maxiter=max_iters, blocking=False)}


#
# VPFS Quantum Optimization Core
#

def run_vpfs_optimization(ansatz_name, optimizer_name, seed, lr=None, max_iterations=None, verbose=True):
    """
    Runs a single VPFS optimization with specified parameters.

    Args:
        ansatz_name (str): Name of ansatz to use
        optimizer_name (str): Name of optimizer to use
        seed (int): Random seed for reproducibility
        lr (float): Learning rate (optional)
        max_iterations (int): Maximum iterations (optional)
        verbose (bool): Whether to print progress

    Returns:
        dict: Results dictionary with optimization metrics
    """
    global learning_rate, max_iters

    # Set parameters
    if lr is not None:
        learning_rate = lr
    if max_iterations is not None:
        max_iters = max_iterations

    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    if verbose and not MODO_SILENCIOSO:
        print(f"\n--- Running VPFS: {optimizer_name} + {ansatz_name} (seed={seed}, lr={learning_rate}) ---")

    # Initialize parameters based on ansatz
    n_params = ansatzes[ansatz_name]["n_params"]()
    initial_params = np.random.uniform(0, 2 * np.pi, size=n_params)

    # VPFS setup
    B = V * (Y @ V)
    B[0] = 0
    B_norm = np.linalg.norm(B)
    Y_extended, U_b_dagger, Y_norm = create_unitaries(Y, B)

    dev = qml.device("default.qubit", wires=total_wires)
    dev_v = qml.device("default.qubit", wires=num_qubits)

    # QNode for getting vector 'v' in unified way
    @qml.qnode(dev_v)
    def get_v_qnode(params, option_ansatz):
        if option_ansatz == 'amplitude':
            v_temp = ansatz_amplitude(params)
            qml.AmplitudeEmbedding(v_temp, wires=range(num_qubits), normalize=False)
        elif option_ansatz == 'complex_z':
            ansatz_complex_z(params, range(num_qubits))
        elif option_ansatz == 'variational_block':
            ansatz_variational_block(params, num_qubits)
        elif option_ansatz == 'hardware_efficient':
            ansatz_hardware_efficient(params, range(num_qubits))
        elif option_ansatz == 'strongly_entangling':
            # Reshape is needed because the template expects weights in a specific shape
            shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=num_qubits)
            ansatz_strongly_entangling(params.reshape(shape), wires=range(num_qubits))

        return qml.state()

    # Main quantum circuits
    @qml.qnode(dev)
    def circuit1(params, option_ansatz):
        state_v = get_v_qnode(params, option_ansatz)
        qml.StatePrep(state_v, wires=range(1, num_qubits + 1))
        qml.StatePrep(state_v, wires=range(num_qubits + 1, 2 * num_qubits + 1))
        qml.QubitUnitary(Y_extended, wires=range(num_qubits + 1))
        for i in range(1, num_qubits + 1):
            qml.CNOT(wires=[i + num_qubits, i])
        return qml.state()

    @qml.qnode(dev)
    def circuit2(params, option_ansatz):
        state_v = get_v_qnode(params, option_ansatz)
        qml.StatePrep(state_v, wires=range(1, num_qubits + 1))
        qml.StatePrep(state_v, wires=range(num_qubits + 1, 2 * num_qubits + 1))
        qml.QubitUnitary(Y_extended, wires=range(num_qubits + 1))
        for i in range(1, num_qubits + 1):
            qml.CNOT(wires=[i + num_qubits, i])
        qml.QubitUnitary(U_b_dagger, wires=range(num_qubits + 1, 2 * num_qubits + 1))
        return qml.state()

    def calculate_loss_with_simulation(params):
        """VPFS loss function - ensures real output."""
        try:
            # Ensure params are real for gradient calculations
            params_real = np.real(params) if np.iscomplexobj(params) else params

            v = get_v_qnode(params_real, ansatz_name)

            if abs(v[0]) < 1e-9:
                return 1e6  # Avoid division by zero
            V_norm = abs(1 / v[0])

            dim = 2 ** num_qubits
            statevector1 = circuit1(params_real, ansatz_name)
            statevector2 = circuit2(params_real, ansatz_name)

            shots_array = np.abs(statevector1[1:dim]) ** 2
            shots_total = np.sum(shots_array)
            if shots_total < 1e-12:
                return 1e6
            norm_yv_cnot = np.sqrt(shots_total)

            shots_array2 = np.abs(statevector2[0]) ** 2
            norm_after_ub = np.sqrt(shots_array2)

            norm_YV_cnot = norm_yv_cnot * Y_norm * V_norm * V_norm
            pen_coef = PEN_COEF_SCALE / B_norm ** 2

            a2 = norm_YV_cnot ** 2
            b2 = B_norm ** 2
            ab = norm_after_ub * Y_norm * B_norm * V_norm * V_norm
            loss = a2 + b2 - 2 * ab

            # Ensure loss is real
            selected_loss = float(np.real(loss))

            if np.isnan(selected_loss) or np.isinf(selected_loss):
                return 1e6

            if selected_loss < -1e6:
                return 1e6

            return max(selected_loss, 0)

        except Exception as e:
            if verbose:
                print(f"Error in loss calculation: {e}")
            return 1e6

    def finite_difference_gradient(params, delta=1e-4):
        grad = np.zeros_like(params, dtype=float)
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += delta
            params_minus[i] -= delta
            loss_plus = calculate_loss_with_simulation(params_plus)
            loss_minus = calculate_loss_with_simulation(params_minus)
            grad[i] = (loss_plus - loss_minus) / (2 * delta)
        return grad

    # Get analytic gradient function with PennyLane
    grad_fn = qml.grad(calculate_loss_with_simulation)

    # Optimization loop
    current_params = initial_params.copy()
    loss_history = []
    start_time = time.time()

    # Run optimization based on optimizer type
    if optimizer_name in ["nesterov", "rmsprop", "adagrad", "momentum", "sgd", "spsa"]:
        # PennyLane optimizers from VQLS - Convert params to real to avoid ComplexWarning
        if optimizer_name == "spsa":
            opt = pennylane_optimizers[optimizer_name](learning_rate)
        else:
            opt = pennylane_optimizers[optimizer_name](learning_rate)

        # Ensure parameters are real-valued for PennyLane optimizers
        current_params = np.real(current_params.astype(np.float64))

        for iter_count in range(max_iters):
            loss = calculate_loss_with_simulation(current_params)
            loss_history.append(float(loss))

            if loss is None or np.isnan(loss):
                if verbose:
                    print(f"Iteration {iter_count + 1}: Invalid loss encountered. Stopping.")
                break
            if loss < tolerance:
                if verbose:
                    print(f"Iteration {iter_count + 1}: Converged using {optimizer_name}.")
                break

            try:
                if optimizer_name == "spsa":
                    # SPSA has slightly different interface
                    current_params = opt.step(calculate_loss_with_simulation, current_params)
                    current_params = np.real(current_params.astype(np.float64))  # Keep real
                else:
                    current_params, loss = opt.step_and_cost(calculate_loss_with_simulation, current_params)
                    current_params = np.real(current_params.astype(np.float64))  # Keep real

            except Exception as e:
                if verbose:
                    print(f"Iteration {iter_count + 1}: Error during optimization step: {e}")
                break

            if verbose and not MODO_SILENCIOSO and (iter_count % 200 == 0 or iter_count <= 5):
                print(f"  Step {iter_count:3d}: Loss = {loss:.8e}")

    elif optimizer_name == "analytic":
        for iter_count in range(max_iters):
            loss = calculate_loss_with_simulation(current_params)
            loss_history.append(float(loss))

            if loss is None or np.isnan(loss):
                if verbose:
                    print(f"Iteration {iter_count + 1}: Invalid loss encountered. Stopping.")
                break
            if loss < tolerance:
                if verbose:
                    print(f"Iteration {iter_count + 1}: Converged using analytic gradient.")
                break

            try:
                grad = grad_fn(current_params)
                if np.any(np.isnan(grad)):
                    if verbose:
                        print(f"Iteration {iter_count + 1}: NaN gradient encountered. Stopping.")
                    break
                current_params = current_params - learning_rate * grad
            except Exception as e:
                if verbose:
                    print(f"Iteration {iter_count + 1}: Error during gradient calculation: {e}")
                break

            if verbose and not MODO_SILENCIOSO and (iter_count % 100 == 0 or iter_count <= 5):
                print(f"  Step {iter_count:3d}: Loss = {loss:.8e}")

    elif optimizer_name == "basic":
        for iter_count in range(max_iters):
            loss = calculate_loss_with_simulation(current_params)
            loss_history.append(float(loss))

            if loss is None or np.isnan(loss):
                break
            if loss < tolerance:
                if verbose:
                    print(f"Iteration {iter_count + 1}: Converged using basic finite difference.")
                break

            grad = finite_difference_gradient(current_params)
            if np.any(np.isnan(grad)):
                break
            current_params = current_params - learning_rate * grad

            if verbose and not MODO_SILENCIOSO and (iter_count % 100 == 0 or iter_count <= 5):
                print(f"  Step {iter_count:3d}: Loss = {loss:.8e}")

    elif optimizer_name == "cobyla":
        def loss_func_cobyla(params):
            l = calculate_loss_with_simulation(params)
            return float(l) if l is not None and not np.isnan(l) else 1e6

        result = minimize(loss_func_cobyla, current_params, tol=tolerance, method="COBYLA", options={"maxiter": max_iters, "disp": False})
        current_params = result.x
        iter_count = result.nfev
        loss = result.fun
        loss_history = [loss]  # COBYLA doesn't provide history

        if verbose:
            print(f"COBYLA finished after {result.nfev} evaluations. Final loss: {result.fun:.6e}")

    elif optimizer_name == "adam":
        params_tensor = torch.tensor(current_params.astype(np.float32), requires_grad=True)
        optim = torch.optim.Adam([params_tensor], lr=learning_rate)

        for iter_count in range(max_iters):
            loss_val = calculate_loss_with_simulation(params_tensor.detach().numpy())
            loss_history.append(float(loss_val))

            if loss_val is None or np.isnan(loss_val):
                break
            if loss_val < tolerance:
                if verbose:
                    print(f"Iteration {iter_count + 1}: Converged using Adam.")
                break

            optim.zero_grad()
            grad_np = finite_difference_gradient(params_tensor.detach().numpy())
            if np.any(np.isnan(grad_np)):
                break

            with torch.no_grad():
                params_tensor.grad = torch.tensor(grad_np.astype(np.float32))

            optim.step()

            if verbose and not MODO_SILENCIOSO and (iter_count % 100 == 0 or iter_count <= 5):
                print(f"  Step {iter_count:3d}: Loss = {loss_val:.8e}")

        current_params = params_tensor.detach().numpy()
        loss = calculate_loss_with_simulation(current_params)

    else:
        raise ValueError(f"Optimizer '{optimizer_name}' not recognized.")

    # Calculate final metrics
    final_loss = loss_history[-1] if loss_history else 1e6
    optimization_time = time.time() - start_time

    # Get final vector v and calculate solution quality
    v_final = get_v_qnode(current_params, ansatz_name)

    if abs(v_final[0]) < 1e-9:
        Vsol = [np.nan] * len(V)
        max_err_V = np.inf
        solution_quality = 0.0
    else:
        v0 = v_final[0]
        Vsol = v_final / v0
        err_V = np.abs(V - Vsol)
        max_err_V = np.max(err_V)
        solution_quality = 1.0 / (1.0 + max_err_V)  # Quality metric

    if verbose and not MODO_SILENCIOSO:
        print(f"  Final loss: {final_loss:.7e}")
        print(f"  Max error in V: {max_err_V:.6f}")
        print(f"  Solution quality: {solution_quality:.6f}")

    return {"optimizer": optimizer_name, "ansatz": ansatz_name, "seed": seed, "learning_rate": learning_rate, "max_iterations": max_iters,
            "final_loss": final_loss, "loss_history": loss_history, "final_params": current_params.tolist(), "n_params": len(current_params),
            "optimization_time": optimization_time, "converged_at": len(loss_history), "solution_quality": solution_quality, "max_error_V": max_err_V,
            "V_target": V.tolist(), "V_solution": Vsol.tolist() if not np.any(np.isnan(Vsol)) else None, "converged": final_loss < tolerance}


#
# Multi-seed analysis functions
#

def run_multiseed_analysis(ansatz_name, optimizer_name, num_seeds=5, lr=0.1, max_iterations=1000, verbose=True):
    """Run multiple seeds and return statistics."""

    print(f"\n🎲 MULTI-SEED ANALYSIS: {ansatz_name} + {optimizer_name}")
    print(f"Running {num_seeds} seeds with lr={lr}, max_iters={max_iterations}")
    print("=" * 60)

    results = []
    qualities = []

    for seed in range(num_seeds):
        if verbose:
            print(f"\n--- Seed {seed + 1}/{num_seeds} ---")

        try:
            result = run_vpfs_optimization(ansatz_name, optimizer_name, seed, lr=lr, max_iterations=max_iterations, verbose=verbose)
            results.append(result)
            qualities.append(result['solution_quality'])

            if verbose:
                print(f"Seed {seed + 1}: Quality = {result['solution_quality']:.4f}, Loss = {result['final_loss']:.2e}")

        except Exception as e:
            print(f"Seed {seed + 1}: ERROR - {e}")
            qualities.append(0.0)

    # Calculate statistics
    if qualities:
        mean_quality = np.mean(qualities)
        std_quality = np.std(qualities)
        min_quality = np.min(qualities)
        max_quality = np.max(qualities)
        success_rate = sum(1 for q in qualities if q > 0.6) / len(qualities)  # >60% quality
        coefficient_of_variation = std_quality / mean_quality if mean_quality > 0 else float('inf')

        stats = {'mean_quality': mean_quality, 'std_quality': std_quality, 'min_quality': min_quality, 'max_quality': max_quality,
            'range_quality': max_quality - min_quality, 'success_rate': success_rate, 'coefficient_of_variation': coefficient_of_variation,
            'num_seeds': len(qualities), 'all_qualities': qualities}

        print(f"\n📊 STATISTICAL SUMMARY:")
        print(f"  Mean Quality: {mean_quality:.4f} ± {std_quality:.4f}")
        print(f"  Range: [{min_quality:.4f}, {max_quality:.4f}]")
        print(f"  Success Rate (>60%): {success_rate * 100:.1f}%")
        print(f"  Coefficient of Variation: {coefficient_of_variation:.3f}")

        # Assess reliability
        if coefficient_of_variation < 0.15:
            reliability = "🟢 HIGHLY RELIABLE"
        elif coefficient_of_variation < 0.3:
            reliability = "🟡 MODERATELY RELIABLE"
        else:
            reliability = "🔴 UNRELIABLE"

        print(f"  Reliability Assessment: {reliability}")

        return results, stats
    else:
        return [], {}


def calculate_solution_metrics(result):
    """Calculate detailed solution metrics for VPFS result."""
    if result["V_solution"] is None:
        return {"fidelity": 0.0, "relative_error": np.inf, "component_errors": [np.inf] * len(V), "solution_quality": 0.0}

    V_target = np.array(result["V_target"])
    V_solution = np.array(result["V_solution"])

    # Calculate component-wise errors
    component_errors = np.abs(V_target - V_solution)
    relative_error = np.linalg.norm(component_errors) / np.linalg.norm(V_target)

    # Calculate fidelity-like metric (higher is better, 1.0 is perfect)
    fidelity = np.exp(-relative_error)

    return {"fidelity": float(fidelity), "relative_error": float(relative_error), "component_errors": component_errors.tolist(),
            "solution_quality": result["solution_quality"]}


def compare_vpfs_solutions(result):
    """Compare target and optimized solutions with visualization."""
    print("\n" + "=" * 50)
    print("VPFS SOLUTION COMPARISON")
    print("=" * 50)

    if result["V_solution"] is None:
        print("❌ Solution failed to converge properly")
        return {"comparison_successful": False}

    V_target = np.array(result["V_target"])
    V_solution = np.array(result["V_solution"])

    print("Target vector V:")
    for i, val in enumerate(V_target):
        print(f"  V[{i}]: {val:.6f}")

    print("\nOptimized vector V:")
    for i, val in enumerate(V_solution):
        print(f"  V[{i}]: {val:.6f}")

    errors = np.abs(V_target - V_solution)
    print("\nComponent-wise errors:")
    for i, err in enumerate(errors):
        print(f"  |V[{i}] - V_opt[{i}]|: {err:.6f}")

    max_error = np.max(errors)
    relative_error = np.linalg.norm(errors) / np.linalg.norm(V_target)

    metrics = calculate_solution_metrics(result)
    fidelity = metrics['fidelity']

    print(f"\nMaximum component error: {max_error:.6f}")
    print(f"Relative error (L2 norm): {relative_error:.6f}")
    print(f"Fidelity: {fidelity:.6f}")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Solution comparison
    x_pos = np.arange(len(V_target))
    width = 0.35

    ax1.bar(x_pos - width / 2, V_target, width, label='Target V', color='blue', alpha=0.7)
    ax1.bar(x_pos + width / 2, V_solution, width, label='Optimized V', color='green', alpha=0.7)
    ax1.set_xlabel('Component Index')
    ax1.set_ylabel('Value')
    ax1.set_title('Target vs Optimized Solution')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'V[{i}]' for i in range(len(V_target))])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Error plot
    ax2.bar(x_pos, errors, color='red', alpha=0.7)
    ax2.set_xlabel('Component Index')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Component-wise Errors')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'V[{i}]' for i in range(len(V_target))])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {"comparison_successful": True, "max_error": float(max_error), "relative_error": float(relative_error), "V_target": V_target.tolist(),
            "V_solution": V_solution.tolist(), "component_errors": errors.tolist()}


def run_systematic_comparison():
    """Systematic comparison of optimizers and ansätze for VPFS."""

    # Include VQLS optimizers + new advanced ansätze in systematic comparison
    optimizer_list = ["basic", "analytic", "cobyla", "adam", "nesterov", "rmsprop", "adagrad"]
    ansatz_list = ["amplitude", "complex_z"]

    all_results = []

    print("=" * 60)
    print("SYSTEMATIC COMPARISON OF VPFS OPTIMIZERS AND ANSÄTZE")
    print("=" * 60)

    total_experiments = len(optimizer_list) * len(ansatz_list) * N_RANDOM_SEEDS
    experiment_count = 0

    for optimizer_name in optimizer_list:
        for ansatz_name in ansatz_list:
            best_result = None
            results_for_config = []

            # Test multiple seeds for this configuration
            for seed in range(N_RANDOM_SEEDS):
                experiment_count += 1
                print(f"\n[{experiment_count}/{total_experiments}] Testing {optimizer_name} + {ansatz_name} (seed={seed})")

                try:
                    result = run_vpfs_optimization(ansatz_name, optimizer_name, seed, verbose=False)
                    result["config_id"] = f"{optimizer_name}_{ansatz_name}"
                    results_for_config.append(result)

                    if best_result is None or result["solution_quality"] > best_result["solution_quality"]:
                        best_result = result

                except Exception as e:
                    print(f"  ERROR with {optimizer_name}+{ansatz_name} (seed={seed}): {e}")
                    continue

            if best_result:
                best_result["all_seeds_results"] = results_for_config
                all_results.append(best_result)
                print(
                    f"  BEST for {optimizer_name}+{ansatz_name}: Quality={best_result['solution_quality']:.6f}, Loss={best_result['final_loss']:.6e}")

    return all_results


def run_vpfs_refinement():
    """
    Hardware-realistic ansatz testing with robust multi-seed analysis.
    """
    print("\n" + "=" * 60)
    print("🔬 VPFS HARDWARE-REALISTIC ANSÄTZE WITH ROBUSTNESS ANALYSIS")
    print("=" * 60)

    # Focus on hardware-realistic ansätze
    hardware_ansatze = ["hardware_efficient", "strongly_entangling"]
    proven_optimizers = ["adam", "cobyla"]

    print(f"🎯 Goal: Find reliable hardware-realistic ansatz with good average performance")
    print(f"🧪 Testing {len(hardware_ansatze)} ansätze with {len(proven_optimizers)} optimizers")
    print(f"📊 Using {N_RANDOM_SEEDS} seeds per configuration for statistical robustness")

    # First, run amplitude baseline for comparison
    print(f"\n" + "=" * 60)
    print("📊 BASELINE: AMPLITUDE ANSATZ (NOT HARDWARE REALISTIC)")
    print("=" * 60)

    amplitude_results, amplitude_stats = run_multiseed_analysis("amplitude", "adam", num_seeds=N_RANDOM_SEEDS, lr=0.1, max_iterations=1000,
                                                                verbose=False)

    print(f"\nAmplitude Baseline Statistics:")
    print(f"  Mean Quality: {amplitude_stats['mean_quality']:.4f} ± {amplitude_stats['std_quality']:.4f}")
    print(f"  Success Rate: {amplitude_stats['success_rate'] * 100:.1f}%")
    print(f"  Reliability: CV = {amplitude_stats['coefficient_of_variation']:.3f}")

    # Test hardware ansätze
    all_hardware_results = {}

    for ansatz in hardware_ansatze:
        print(f"\n" + "=" * 60)
        print(f"🔧 TESTING HARDWARE ANSATZ: {ansatz.upper()}")
        print("=" * 60)

        best_config = None
        best_stats = None
        best_reliability_score = -1

        for optimizer in proven_optimizers:
            for lr in [0.05, 0.1, 0.15]:  # Test different learning rates

                print(f"\n--- Configuration: {optimizer} + lr={lr} ---")

                try:
                    results, stats = run_multiseed_analysis(ansatz, optimizer, num_seeds=N_RANDOM_SEEDS, lr=lr, max_iterations=1500, verbose=False)

                    if stats:
                        # Calculate reliability score: balance of performance and consistency
                        reliability_score = (stats['mean_quality'] * 0.6 +  # 60% weight on performance
                                             (1 - stats['coefficient_of_variation']) * 0.4)  # 40% weight on consistency

                        print(
                            f"  Results: Mean={stats['mean_quality']:.4f}, CV={stats['coefficient_of_variation']:.3f}, Score={reliability_score:.4f}")

                        if reliability_score > best_reliability_score:
                            best_reliability_score = reliability_score
                            best_config = (optimizer, lr)
                            best_stats = stats

                except Exception as e:
                    print(f"  ERROR: {e}")

        if best_config:
            all_hardware_results[ansatz] = {'optimizer': best_config[0], 'lr': best_config[1], 'stats': best_stats,
                'reliability_score': best_reliability_score}

    # Analysis and comparison
    print(f"\n" + "=" * 70)
    print("🏆 HARDWARE ANSÄTZE FINAL RESULTS")
    print("=" * 70)

    if not all_hardware_results:
        print("❌ No hardware ansätze produced reliable results")
        return

    # Compare with amplitude baseline
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"{'Ansatz':<20} | {'Mean Quality':<12} | {'Reliability':<12} | {'vs Amplitude':<12}")
    print("-" * 65)

    # Amplitude baseline
    print(
        f"{'Amplitude (baseline)':<20} | {amplitude_stats['mean_quality']:.4f}       | {'CV=' + f'{amplitude_stats['coefficient_of_variation']:.3f}':<12} | {'100.0%':<12}")

    # Hardware results
    best_hardware_ansatz = None
    best_hardware_score = -1

    for ansatz, data in all_hardware_results.items():
        stats = data['stats']
        vs_amplitude = (stats['mean_quality'] / amplitude_stats['mean_quality']) * 100
        reliability = f"CV={stats['coefficient_of_variation']:.3f}"

        print(f"{ansatz:<20} | {stats['mean_quality']:.4f}       | {reliability:<12} | {vs_amplitude:.1f}%")

        if data['reliability_score'] > best_hardware_score:
            best_hardware_score = data['reliability_score']
            best_hardware_ansatz = ansatz

    # Final recommendation
    print(f"\n💡 FINAL RECOMMENDATIONS:")

    if best_hardware_ansatz:
        best_data = all_hardware_results[best_hardware_ansatz]
        best_stats = best_data['stats']

        print(f"\n🏆 BEST HARDWARE-REALISTIC ANSATZ:")
        print(f"  Ansatz: {best_hardware_ansatz}")
        print(f"  Optimizer: {best_data['optimizer']}")
        print(f"  Learning Rate: {best_data['lr']}")
        print(f"  Mean Quality: {best_stats['mean_quality']:.4f} ± {best_stats['std_quality']:.4f}")
        print(f"  Success Rate (>60%): {best_stats['success_rate'] * 100:.1f}%")
        print(f"  Reliability Score: {best_data['reliability_score']:.4f}")

        # Performance vs amplitude
        performance_ratio = best_stats['mean_quality'] / amplitude_stats['mean_quality']
        print(f"  Performance vs Amplitude: {performance_ratio * 100:.1f}%")

        if performance_ratio > 0.8:
            print(f"  ✅ EXCELLENT: Achieves >80% of amplitude performance!")
        elif performance_ratio > 0.6:
            print(f"  👍 GOOD: Achieves >60% of amplitude performance")
        elif performance_ratio > 0.4:
            print(f"  ⚠️  MODERATE: Achieves >40% of amplitude performance")
        else:
            print(f"  ❌ POOR: <40% of amplitude performance")

        # Reliability assessment
        if best_stats['coefficient_of_variation'] < 0.2:
            print(f"  ✅ RELIABLE: Low variability across seeds")
        elif best_stats['coefficient_of_variation'] < 0.4:
            print(f"  ⚠️  MODERATE RELIABILITY: Some variability")
        else:
            print(f"  ❌ UNRELIABLE: High variability - not suitable for production")

        # Final verdict
        if performance_ratio > 0.6 and best_stats['coefficient_of_variation'] < 0.3:
            print(f"\n🎉 CONCLUSION: {best_hardware_ansatz} is VIABLE for hardware implementation!")
        else:
            print(f"\n⚠️  CONCLUSION: Hardware ansätze have significant limitations")
            print(f"   Consider ensemble methods or hybrid approaches")

        # Run best configuration one more time for detailed analysis
        print(f"\n🔬 DETAILED ANALYSIS OF BEST CONFIGURATION:")
        best_result = run_vpfs_optimization(best_hardware_ansatz, best_data['optimizer'], rng_seed, lr=best_data['lr'], max_iterations=2000,
                                            verbose=True)

        if best_result and best_result["V_solution"] is not None:
            compare_vpfs_solutions(best_result)

        return best_result, all_hardware_results

    else:
        print(f"❌ No hardware ansätze achieved acceptable performance")
        return None, all_hardware_results


def run_turbo_comparison():
    """High-speed parallel comparison of optimizers and ansätze."""
    optimizer_list = ["adam", "nesterov", "rmsprop"]
    ansatz_list = ["amplitude", "complex_z", "hardware_efficient"]
    all_results = []

    print("=" * 60)
    print("🚀 TURBO MODE: HIGH-SPEED PARALLEL COMPARISON")
    print("=" * 60)

    tasks = []
    for optimizer_name in optimizer_list:
        for ansatz_name in ansatz_list:
            for seed in range(N_RANDOM_SEEDS):
                tasks.append((ansatz_name, optimizer_name, seed, 0.1, 800, False))

    global MODO_SILENCIOSO
    MODO_SILENCIOSO = True

    start_time = time.time()
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [executor.submit(run_vpfs_optimization, *task) for task in tasks]
        all_results = [f.result() for f in futures]
    end_time = time.time()

    MODO_SILENCIOSO = False
    print(f"\nTurbo mode finished {len(tasks)} experiments in {end_time - start_time:.2f} seconds.")

    # Process results to find best per config
    processed_results = {}
    for r in all_results:
        key = (r['optimizer'], r['ansatz'])
        if key not in processed_results or r['solution_quality'] > processed_results[key]['solution_quality']:
            processed_results[key] = r

    final_results = list(processed_results.values())
    return final_results


#
# Main execution block
#
if __name__ == '__main__':

    # Suppress warnings in main execution for clarity
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Casting complex values to real.*")

    if MULTIPLE_RUNS:
        results = run_systematic_comparison()
        if results:
            print("Systematic comparison completed.")

    if SINGLE_MODE:
        print("\n" + "=" * 60)
        print("🎯 SINGLE MODE: MULTI-SEED STATISTICAL ANALYSIS")
        print("=" * 60)

        # Test both amplitude (baseline) and hardware_efficient
        test_configs = [("amplitude", "adam", "Baseline - NOT hardware realistic"), ("hardware_efficient", "adam", "Hardware-realistic ansatz"),
            ("hardware_efficient", "cobyla", "Hardware-realistic with COBYLA")]

        for ansatz, optimizer, description in test_configs:
            print(f"\n🧪 TESTING: {description}")
            print("=" * 50)

            results, stats = run_multiseed_analysis(ansatz, optimizer, num_seeds=N_RANDOM_SEEDS, lr=0.1, max_iterations=1500, verbose=True)

            if results and stats['mean_quality'] > 0.6:
                # Show detailed analysis for best result
                best_result = max(results, key=lambda x: x['solution_quality'])
                compare_vpfs_solutions(best_result)

    if REFINEMENT_MODE:
        best_result, all_hardware_results = run_vpfs_refinement()

    if TURBO_MODE:
        results = run_turbo_comparison()
        print(f"Turbo comparison completed with {len(results)} configurations.")

    print(f"\n" + "=" * 70)
    print("🏁 VPFS HARDWARE-REALISTIC ANALYSIS COMPLETED")
    print("=" * 70)
    print("✅ Multi-seed robustness analysis performed")
    print("✅ Hardware ansätze statistical performance evaluated")
    print("✅ Realistic expectations for NISQ hardware established")