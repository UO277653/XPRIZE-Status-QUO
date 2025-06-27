# VPFS (Variational Parametric Feedback System) Experimental Suite
# Inspirado en la estructura del VQLS pero usando el algoritmo VPFS
# 🎯 INCLUDES VQLS OPTIMIZERS & ANSÄTZE - Try the winning combo: nesterov + hardware_efficient!
# 🏆 Best VQLS config: nesterov + hardware_efficient (seed=2, lr=0.4, steps=2000)

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

#
# Setting of the main hyper-parameters of the model
#

# VPFS Problem parameters
Y = np.array([[1, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 1]], dtype=complex) * 5
V = np.array([1, 1.1, 0.95, 0.9])
num_qubits = 2  # Number of qubits per register
total_wires = 2 * num_qubits + 1

# Optimization parameters
max_iters = 2000  # Updated to match VQLS best config
tolerance = 1e-9  # Convergence tolerance
learning_rate = 0.4  # Updated to match VQLS best config (was 0.05)
radius = 0.1  # Parameter for amplitude ansatz
PEN_COEF_SCALE = 0.0  # Penalty coefficient scale
loss_option = 4  # Loss option (from original VPFS)

# Experimental configuration
rng_seed = 2  # Updated to match VQLS best config (was 42)
N_RANDOM_SEEDS = 3  # Number of random seeds per configuration

# --------------------------------------------------------------------------
# EXECUTION MODES
MULTIPLE_RUNS = False  # For systematic comparison
SINGLE_MODE = True  # For individual experiment
REFINEMENT_MODE = False  # For refining best configuration
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


# VQLS Ansätze adapted for VPFS
def ansatz_hardware_efficient(params, wires):
    """Hardware-efficient ansatz from VQLS (adapted for VPFS)."""
    n = len(wires)
    assert len(params) == 2 * n, f"Expected {2 * n} parameters for hardware_efficient, got {len(params)}"

    idx = 0
    # First RY layer
    for w in wires:
        qml.RY(params[idx], wires=w)
        idx += 1

    # Entanglement layer
    for i in range(n - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])

    # Second RY layer
    for w in wires:
        qml.RY(params[idx], wires=w)
        idx += 1


def ansatz_layered(params, wires, n_layers=2):
    """Layered ansatz from VQLS (adapted for VPFS)."""
    n = len(wires)
    params_per_layer = 3 * n
    assert len(params) == params_per_layer * n_layers, f"Expected {params_per_layer * n_layers} params for layered ansatz, got {len(params)}"

    idx = 0
    for layer in range(n_layers):
        # Rotation layer
        for w in wires:
            theta, phi, lam = params[idx], params[idx + 1], params[idx + 2]
            qml.Rot(theta, phi, lam, wires=w)
            idx += 3

        # Entanglement layer (varies by layer)
        if layer % 2 == 0:
            # Linear entanglement
            for i in range(n - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
        else:
            # Circular entanglement
            for i in range(n):
                qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])


def ansatz_complex_vqls(params, wires):
    """Complex ansatz from VQLS (adapted for VPFS)."""
    n = len(wires)
    assert len(params) == 3 * n, f"Expected {3 * n} parameters for complex ansatz, got {len(params)}"

    idx = 0
    # Initial Hadamard layer
    for w in wires:
        qml.Hadamard(wires=w)

    # Full Rot layer (more expressive than just RY)
    for w in wires:
        theta, phi, lam = params[idx], params[idx + 1], params[idx + 2]
        qml.Rot(theta, phi, lam, wires=w)
        idx += 3

    # Entanglement
    for i in range(n - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])


# Dictionary of available ansätze (VQLS + VPFS combined)
n_layers = 2  # For layered ansatz
ansatzes = {  # Original VPFS ansätze
    "amplitude": {"n_params": lambda: 2 ** num_qubits - 1, "description": "Amplitude embedding ansatz"},
    "variational_block": {"n_params": lambda: 2 ** num_qubits - 1, "description": "Variational block with entanglement"},
    "complex_z": {"n_params": lambda: 2 * num_qubits, "description": "Complex ansatz with RZ rotations"},  # VQLS ansätze added
    "hardware_efficient": {"n_params": lambda: 2 * num_qubits, "description": "Hardware-efficient ansatz from VQLS"},
    "layered": {"n_params": lambda: 3 * num_qubits * n_layers, "description": "Layered ansatz with multiple entanglement patterns"},
    "complex": {"n_params": lambda: 3 * num_qubits, "description": "Complex ansatz from VQLS with Rot gates"}}

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
        elif option_ansatz == 'layered':
            ansatz_layered(params, range(num_qubits), n_layers)
        elif option_ansatz == 'complex':
            ansatz_complex_vqls(params, range(num_qubits))
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
        """VPFS loss function."""
        try:
            v = get_v_qnode(params, ansatz_name)

            if abs(v[0]) < 1e-9:
                return 1e6  # Avoid division by zero
            V_norm = abs(1 / v[0])

            dim = 2 ** num_qubits
            statevector1 = circuit1(params, ansatz_name)
            statevector2 = circuit2(params, ansatz_name)

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

            selected_loss = np.real(loss)

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
        # PennyLane optimizers from VQLS
        if optimizer_name == "spsa":
            opt = pennylane_optimizers[optimizer_name](learning_rate)
        else:
            opt = pennylane_optimizers[optimizer_name](learning_rate)

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
                    loss = calculate_loss_with_simulation(current_params)
                else:
                    current_params, loss = opt.step_and_cost(calculate_loss_with_simulation, current_params)

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
# Experimental Suite Functions (adapted from VQLS)
#

def calculate_solution_metrics(result):
    """Calculate detailed solution metrics for VPFS result."""
    if result["V_solution"] is None:
        return {"fidelity": 0.0, "relative_error": np.inf, "component_errors": [np.inf] * len(V), "solution_quality": 0.0}

    V_target = np.array(result["V_target"])
    V_solution = np.array(result["V_solution"])

    # Calculate component-wise errors
    component_errors = np.abs(V_target - V_solution)
    relative_error = np.linalg.norm(component_errors) / np.linalg.norm(V_target)

    # Calculate fidelity-like metric
    fidelity = np.exp(-relative_error)  # Exponential decay with error

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

    print(f"\nMaximum component error: {max_error:.6f}")
    print(f"Relative error (L2 norm): {relative_error:.6f}")

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

    # Include VQLS optimizers in systematic comparison
    optimizer_list = ["basic", "analytic", "cobyla", "adam", "nesterov", "rmsprop", "adagrad"]
    ansatz_list = ["amplitude", "complex_z", "variational_block", "hardware_efficient", "layered", "complex"]

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


def analyze_and_save_vpfs_results(results):
    """Analyze and save VPFS experimental results."""

    if not results:
        print("No results to analyze!")
        return None, None

    # Find best result overall
    best_overall = max(results, key=lambda x: x["solution_quality"])

    print("\n" + "=" * 60)
    print("VPFS EXPERIMENTAL RESULTS ANALYSIS")
    print("=" * 60)

    print(f"\nBEST OVERALL CONFIGURATION:")
    print(f"  Optimizer: {best_overall['optimizer']}")
    print(f"  Ansatz: {best_overall['ansatz']}")
    print(f"  Solution Quality: {best_overall['solution_quality']:.6f}")
    print(f"  Final Loss: {best_overall['final_loss']:.6e}")
    print(f"  Max Error in V: {best_overall['max_error_V']:.6f}")
    print(f"  Converged: {best_overall['converged']}")

    # Ranking by optimizer
    print(f"\nRANKING BY OPTIMIZER:")
    optimizer_stats = {}
    for result in results:
        opt = result["optimizer"]
        if opt not in optimizer_stats:
            optimizer_stats[opt] = []
        optimizer_stats[opt].append(result["solution_quality"])

    for opt, qualities in sorted(optimizer_stats.items(), key=lambda x: np.mean(x[1]), reverse=True):
        avg_quality = np.mean(qualities)
        std_quality = np.std(qualities)
        print(f"  {opt:12s}: {avg_quality:.6f} ± {std_quality:.6f} (n={len(qualities)})")

    # Ranking by ansatz
    print(f"\nRANKING BY ANSATZ:")
    ansatz_stats = {}
    for result in results:
        ans = result["ansatz"]
        if ans not in ansatz_stats:
            ansatz_stats[ans] = []
        ansatz_stats[ans].append(result["solution_quality"])

    for ans, qualities in sorted(ansatz_stats.items(), key=lambda x: np.mean(x[1]), reverse=True):
        avg_quality = np.mean(qualities)
        std_quality = np.std(qualities)
        print(f"  {ans:18s}: {avg_quality:.6f} ± {std_quality:.6f} (n={len(qualities)})")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"vpfs_comparison_{timestamp}.json"

    # Clean results for JSON serialization
    clean_results = []
    for result in results:
        clean_result = {"optimizer": result["optimizer"], "ansatz": result["ansatz"], "seed": result["seed"],
                        "learning_rate": result["learning_rate"], "solution_quality": result["solution_quality"], "final_loss": result["final_loss"],
                        "max_error_V": result["max_error_V"], "converged": result["converged"], "optimization_time": result["optimization_time"],
                        "n_params": result["n_params"]}
        clean_results.append(clean_result)

    experiment_data = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       "best_overall": {"optimizer": best_overall["optimizer"], "ansatz": best_overall["ansatz"],
                                        "solution_quality": best_overall["solution_quality"], "final_loss": best_overall["final_loss"],
                                        "max_error_V": best_overall["max_error_V"], "converged": best_overall["converged"]},
                       "all_results": clean_results, "summary": {
            "optimizer_ranking": {opt: {"mean": float(np.mean(qualities)), "std": float(np.std(qualities))} for opt, qualities in
                                  optimizer_stats.items()},
            "ansatz_ranking": {ans: {"mean": float(np.mean(qualities)), "std": float(np.std(qualities))} for ans, qualities in ansatz_stats.items()}},
                       "configuration": {"num_qubits": num_qubits, "max_iters": max_iters, "tolerance": tolerance, "Y_matrix": Y.tolist(),
                                         "V_target": V.tolist(), "N_RANDOM_SEEDS": N_RANDOM_SEEDS}}

    try:
        with open(filename, "w") as f:
            json.dump(experiment_data, f, indent=2, default=str)
        print(f"\nResults saved to: {filename}")
    except Exception as e:
        print(f"\nError saving JSON: {e}")
        filename = None

    return best_overall, filename


def plot_vpfs_comparison_results(results):
    """Visualize VPFS comparison results."""

    if not results:
        print("No results to plot!")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Solution quality by optimizer
    optimizer_data = {}
    for result in results:
        opt = result["optimizer"]
        if opt not in optimizer_data:
            optimizer_data[opt] = []
        optimizer_data[opt].append(result["solution_quality"])

    ax1.boxplot(optimizer_data.values(), tick_labels=optimizer_data.keys())
    ax1.set_title("Solution Quality by Optimizer")
    ax1.set_ylabel("Solution Quality")
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # 2. Solution quality by ansatz
    ansatz_data = {}
    for result in results:
        ans = result["ansatz"]
        if ans not in ansatz_data:
            ansatz_data[ans] = []
        ansatz_data[ans].append(result["solution_quality"])

    ax2.boxplot(ansatz_data.values(), tick_labels=ansatz_data.keys())
    ax2.set_title("Solution Quality by Ansatz")
    ax2.set_ylabel("Solution Quality")
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # 3. Convergence of best result
    best_result = max(results, key=lambda x: x["solution_quality"])
    if len(best_result["loss_history"]) > 1:
        ax3.plot(best_result["loss_history"], 'b-', linewidth=2)
        ax3.set_title(f"Best Convergence: {best_result['optimizer']} + {best_result['ansatz']}")
        ax3.set_xlabel("Iterations")
        ax3.set_ylabel("Loss")
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No convergence history available", ha='center', va='center', transform=ax3.transAxes)

    # 4. Solution quality vs final loss scatter
    qualities = [r["solution_quality"] for r in results]
    losses = [r["final_loss"] for r in results]
    colors = [hash(r["optimizer"]) % 10 for r in results]

    scatter = ax4.scatter(losses, qualities, c=colors, cmap='tab10', alpha=0.7, s=50)
    ax4.set_xlabel("Final Loss")
    ax4.set_ylabel("Solution Quality")
    ax4.set_title("Solution Quality vs Final Loss")
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)

    # Add legend for optimizers
    unique_optimizers = list(set(r["optimizer"] for r in results))
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab10(hash(opt) % 10 / 10), markersize=8, label=opt) for opt
                       in unique_optimizers]
    ax4.legend(handles=legend_elements, title="Optimizers", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    return fig


def run_vpfs_refinement():
    """Refinement mode for VPFS - find optimal parameters for best configuration."""

    print("=" * 70)
    print("🔬 VPFS REFINEMENT MODE: Optimizing Best Configuration")
    print("=" * 70)

    # Use VQLS winning configuration as base
    base_optimizer = "nesterov"  # VQLS winner
    base_ansatz = "hardware_efficient"  # VQLS winner

    print(f"🎯 Base configuration from VQLS: {base_optimizer} + {base_ansatz}")
    print(f"   Parameters: seed={rng_seed}, lr={learning_rate}, steps={max_iters}")

    # Parameters to test for refinement
    refinement_configs = [  # Test around winning VQLS configuration
        (base_optimizer, base_ansatz, 0.4, "VQLS Winner", 2), (base_optimizer, base_ansatz, 0.3, "Lower LR", 2),
        (base_optimizer, base_ansatz, 0.5, "Higher LR", 2), (base_optimizer, base_ansatz, 0.4, "More Iters", 2, 2500),
        (base_optimizer, base_ansatz, 0.4, "Seed 1", 1), (base_optimizer, base_ansatz, 0.4, "Seed 3", 3),  # Test other VQLS-inspired combinations
        ("rmsprop", base_ansatz, 0.4, "RMSProp", 2), ("adagrad", base_ansatz, 0.4, "Adagrad", 2), (base_optimizer, "layered", 0.4, "Layered", 2),
        (base_optimizer, "complex", 0.4, "Complex", 2),  # Original VPFS best performers
        ("cobyla", "complex_z", 0.05, "VPFS Classic", 2), ("adam", "complex_z", 0.05, "Adam Classic", 2), ]

    best_quality = 0
    best_overall = None
    all_refined_results = []

    for config in refinement_configs:
        if len(config) == 5:
            opt, ans, lr, desc, seed = config
            max_iters_local = max_iters
        else:
            opt, ans, lr, desc, seed, max_iters_local = config

        print(f"\n🧪 Refining: {desc} (lr={lr}, seed={seed}, max_iters={max_iters_local})")

        try:
            result = run_vpfs_optimization(ans, opt, seed, lr=lr, max_iterations=max_iters_local, verbose=False)

            # Calculate detailed metrics
            metrics = calculate_solution_metrics(result)
            result.update(metrics)
            result['description'] = desc
            result['config_details'] = f"lr={lr}, seed={seed}, max_iters={max_iters_local}"

            all_refined_results.append(result)

            quality = result['solution_quality']
            print(f"  📊 Quality: {quality:.6f}, Loss: {result['final_loss']:.6e}, Error: {result['max_error_V']:.6f}")

            if quality > best_quality:
                best_quality = quality
                best_overall = result
                print(f"  🌟 NEW BEST!")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    return best_overall, all_refined_results


def save_vpfs_results(result, comparison_data, mode="single"):
    """Save complete VPFS results including comparison."""

    complete_results = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "mode": mode,
                        "optimization": {"optimizer": result["optimizer"], "ansatz": result["ansatz"], "seed": result["seed"],
                                         "learning_rate": result["learning_rate"], "final_loss": result["final_loss"],
                                         "solution_quality": result["solution_quality"], "max_error_V": result["max_error_V"],
                                         "converged": result["converged"], "optimization_time": result["optimization_time"]},
                        "comparison": comparison_data,
                        "configuration": {"num_qubits": num_qubits, "max_iters": max_iters, "tolerance": tolerance, "Y_matrix": Y.tolist(),
                                          "V_target": V.tolist()}}

    # Determine filename based on mode
    if mode == "refinement":
        filename = "vpfs_refinement_results.json"
    else:
        filename = "vpfs_single_results.json"

    all_experiments = []
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                all_experiments = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            all_experiments = []

    # Add new experiment
    complete_results["experiment_id"] = len(all_experiments) + 1
    all_experiments.append(complete_results)

    # Save all experiments
    with open(filename, "w") as f:
        json.dump(all_experiments, f, indent=2, default=str)

    print(f"\nComplete results saved to: {filename} (Experiment #{complete_results['experiment_id']})")
    return filename


#
# Main Execution
#

if __name__ == "__main__":
    print("=" * 70)
    print("🔬 VPFS EXPERIMENTAL OPTIMIZATION SUITE")
    print("=" * 70)

    # Check which mode is active
    active_modes = []
    if MULTIPLE_RUNS:
        active_modes.append("MULTIPLE_RUNS")
    if SINGLE_MODE:
        active_modes.append("SINGLE_MODE")
    if REFINEMENT_MODE:
        active_modes.append("REFINEMENT_MODE")
    if TURBO_MODE:
        active_modes.append("TURBO_MODE")

    if len(active_modes) != 1:
        print("❌ ERROR: Exactly one mode must be True!")
        print("Current modes:", active_modes)
        print("Please set only one of: MULTIPLE_RUNS, SINGLE_MODE, REFINEMENT_MODE, TURBO_MODE to True")
        exit(1)

    active_mode = active_modes[0]
    print(f"🎯 Running in {active_mode}")
    print("=" * 70)

    if MULTIPLE_RUNS:
        # Systematic comparison
        print("🔄 SYSTEMATIC COMPARISON MODE")
        results = run_systematic_comparison()

        if results:
            best_result, filename = analyze_and_save_vpfs_results(results)
            plot_vpfs_comparison_results(results)

            print(f"\n🎉 Experiment completed!")
            print(f"Best configuration: {best_result['optimizer']} + {best_result['ansatz']}")
            print(f"Best solution quality: {best_result['solution_quality']:.6f}")
        else:
            print("No successful results obtained.")

    elif SINGLE_MODE:
        # Individual experiment
        print("🎯 SINGLE EXPERIMENT MODE")

        # Test different configurations
        print("🔬 Testing multiple configurations for VPFS:")

        configs_to_test = [("cobyla", "complex_z", 0.05, "VPFS Classic"), ("nesterov", "hardware_efficient", 0.1, "VQLS Style (Lower LR)"),
                           ("adam", "complex_z", 0.01, "Adam + Complex"), ("nesterov", "complex_z", 0.05, "Nesterov + Complex"),
                           ("cobyla", "hardware_efficient", 0.05, "COBYLA + HW"), ]

        best_result = None
        best_quality = 0

        for opt, ans, lr, desc in configs_to_test:
            print(f"\n🧪 Testing: {desc} ({opt} + {ans}, lr={lr})")

            try:
                result = run_vpfs_optimization(ans, opt, rng_seed, lr=lr, verbose=False)
                quality = result['solution_quality']
                loss = result['final_loss']

                print(f"   📊 Quality: {quality:.6f}, Loss: {loss:.6e}")

                if quality > best_quality:
                    best_quality = quality
                    best_result = result
                    print(f"   🌟 NEW BEST!")

            except Exception as e:
                print(f"   ❌ Error: {e}")

        if best_result:
            print(f"\n🏆 BEST CONFIGURATION FOUND:")
            print(f"   {best_result['optimizer']} + {best_result['ansatz']}")
            print(f"   Quality: {best_result['solution_quality']:.6f}")
            print(f"   Learning Rate: {best_result['learning_rate']}")
            result = best_result
        else:
            print("\n❌ All configurations failed, using fallback...")
            result = run_vpfs_optimization("complex_z", "cobyla", rng_seed, lr=0.05)

        # Show convergence
        if len(result["loss_history"]) > 1:
            plt.figure(figsize=(10, 6))
            plt.plot(result["loss_history"], "g", linewidth=2)
            plt.ylabel("Loss Function")
            plt.xlabel("Optimization steps")
            plt.title(f"VPFS Optimization: {optimizer_name} + {ansatz_name}")
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.show()

        print(f"Final loss: {result['final_loss']:.6e}")
        print(f"Solution quality: {result['solution_quality']:.6f}")

        # Compare solutions
        comparison_data = compare_vpfs_solutions(result)

        # Save complete results
        filename = save_vpfs_results(result, comparison_data, mode="single")

        print(f"\n🎉 Single experiment completed successfully!")
        print(f"Solution quality: {result['solution_quality']:.6f}")
        print(f"Max error in V: {result['max_error_V']:.6f}")
        print(f"Results saved to: {filename}")

    elif REFINEMENT_MODE:
        # Refinement mode
        print("🔬 REFINEMENT MODE")
        print("This mode will systematically refine the best known configuration for VPFS")
        print()

        best_refined, refined_results = run_vpfs_refinement()

        if best_refined:
            print(f"\n🏆 BEST REFINED SOLUTION:")
            print(f"  Configuration: {best_refined['description']}")
            print(f"  Details: {best_refined['config_details']}")
            print(f"  Solution Quality: {best_refined['solution_quality']:.6f}")
            print(f"  Max Error in V: {best_refined['max_error_V']:.6f}")
            print(f"  Final Loss: {best_refined['final_loss']:.6e}")

            # Detailed analysis of best refined solution
            print(f"\n🔬 DETAILED ANALYSIS OF BEST REFINED SOLUTION:")
            comparison_data = compare_vpfs_solutions(best_refined)

            # Save refined results
            filename = save_vpfs_results(best_refined, comparison_data, mode="refinement")

            print(f"\n🎉 REFINEMENT COMPLETED SUCCESSFULLY!")
            print(f"📈 Best configuration found: {best_refined['description']}")
            print(f"📊 Final solution quality: {best_refined['solution_quality']:.6f}")
            print(f"💾 Detailed results saved to: {filename}")
        else:
            print("❌ No refined solutions found")

    elif TURBO_MODE:
        # High-speed parallel experiments
        print("🚀 TURBO MODE - High-Speed Parallel Experiments")

        # Configure for speed
        MODO_SILENCIOSO = True

        # Quick parameter sweep (including VQLS winners)
        optimizers_to_test = ["nesterov", "cobyla", "adam", "rmsprop"]
        ansatzes_to_test = ["hardware_efficient", "complex_z", "layered"]
        learning_rates = [0.3, 0.4, 0.5]

        all_configs = []
        for opt in optimizers_to_test:
            for ans in ansatzes_to_test:
                for lr in learning_rates:
                    for seed in [1, 2, 3]:
                        all_configs.append((ans, opt, seed, lr, 500))  # Reduced iterations for speed

        print(f"🚀 Testing {len(all_configs)} configurations in parallel...")


        # Use multiprocessing for parallel execution
        def run_single_config(config):
            ansatz_name, optimizer_name, seed, lr, max_iters_local = config
            try:
                return run_vpfs_optimization(ansatz_name, optimizer_name, seed, lr=lr, max_iterations=max_iters_local, verbose=False)
            except Exception as e:
                print(f"Error with config {config}: {e}")
                return None


        start_time = time.time()

        # Sequential execution (can be made parallel with ProcessPoolExecutor)
        results = []
        for i, config in enumerate(all_configs):
            print(f"Progress: {i + 1}/{len(all_configs)}", end='\r')
            result = run_single_config(config)
            if result:
                results.append(result)

        total_time = time.time() - start_time

        print(f"\n🎉 Turbo mode completed in {total_time:.1f}s!")
        print(f"Successfully completed {len(results)}/{len(all_configs)} experiments")

        if results:
            best_turbo = max(results, key=lambda x: x["solution_quality"])
            print(f"🏆 Best turbo result: {best_turbo['optimizer']} + {best_turbo['ansatz']}")
            print(f"   Quality: {best_turbo['solution_quality']:.6f}")
            print(f"   Loss: {best_turbo['final_loss']:.6e}")

    print(f"\n" + "=" * 70)
    print("🏁 VPFS EXPERIMENTAL SUITE EXECUTION COMPLETED")
    print("=" * 70)
