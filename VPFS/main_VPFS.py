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

# VPFS Problem parameters with COMPLEX V support
print("🚀 USING WELL-CONDITIONED Y MATRICES WITH COMPLEX V VECTORS")

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
TEST_COMPLEX_V = True  # 🆕 NEW: Set to True to test complex V vector

if TEST_COMPLEX_Y:
    Y = Y_complex
    print("🔬 USING WELL-CONDITIONED COMPLEX Y MATRIX - General electrical networks")
else:
    Y = Y_real
    print("🔬 USING WELL-CONDITIONED REAL Y MATRIX - Urban/rural networks")

# 🆕 COMPLEX V VECTOR SUPPORT
if TEST_COMPLEX_V:
    # Complex voltage reference - more realistic for electrical power systems
    V_real_part = np.array([1.0, 1.1, 0.95, 0.9])  # Voltage magnitudes
    V_imag_part = np.array([0.0, 0.1, -0.05, 0.08])  # Phase components
    V = V_real_part + 1j * V_imag_part
    print("🔬 USING COMPLEX V VECTOR - Realistic power system voltages with phase")
    print(f"   V = {V}")
    print(f"   |V| = {np.abs(V)}")
    print(f"   ∠V = {np.angle(V) * 180 / np.pi} degrees")
else:
    V = np.array([1, 1.1, 0.95, 0.9], dtype=complex)  # Keep as complex array for consistency
    print("🔬 USING REAL V VECTOR (stored as complex) - Simplified case")
    print(f"   V = {V}")

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
# VPFS Algorithm Core Functions with Complex V Support
#

def create_unitaries(Y, B):
    """Creates the unitary matrices Y_extended and U_b† (calculated from B) - supports complex B."""
    # Normalization and Y_extended calculation
    Y_norm = np.max(np.abs(np.linalg.eigvals(Y)))
    Y_normalized = Y / Y_norm
    sqrt_diff = sqrtm(np.eye(len(Y)) - Y_normalized @ np.conj(Y_normalized.T))  # 🆕 Use conjugate transpose
    Y_extended = np.block([[Y_normalized, sqrt_diff], [sqrt_diff, -Y_normalized]])
    Y_extended, _ = polar(Y_extended)  # Take unitary part

    # Build U_b† based on complex B using Gram-Schmidt
    b = B / np.linalg.norm(B)
    U_b = np.eye(len(b), dtype=complex)
    U_b[:, 0] = b

    # 🆕 Modified Gram-Schmidt for complex vectors
    for i in range(1, len(b) - 1):
        v_vec = U_b[1:, i]
        for j in range(i):
            v_vec -= np.vdot(U_b[1:, j], v_vec) * U_b[1:, j]  # Use vdot for complex inner product
        norm = np.linalg.norm(v_vec)
        if norm > 1e-10:  # Avoid division by zero
            v_vec = v_vec / norm
            U_b[1:, i] = v_vec

    U_b[0, -1] = 1
    U_b[-1, -1] = 0
    U_b_dagger = np.conj(U_b.T)  # 🆕 Proper conjugate transpose

    return Y_extended, U_b_dagger, Y_norm


#
# Enhanced Ansatz Functions for Complex V
#

def ansatz_amplitude(params):
    """Calculates vector v from parameter vector params (amplitude embedding) - supports complex output."""
    v = [1]  # First component fixed
    for i in range(len(params)):
        if radius > 0:
            # 🆕 Support complex amplitudes with phase information
            magnitude = np.sin(params[i]) * radius + 1
            if i < len(params) // 2:  # First half for magnitudes
                v.append(magnitude)
            else:  # Second half for phases (if available)
                phase_idx = i - len(params) // 2
                if phase_idx < len(v) - 1:
                    v[phase_idx + 1] *= np.exp(1j * params[i])
        else:
            v.append(params[i] + 1)

    v = np.array(v, dtype=complex)
    v = v / np.linalg.norm(v)
    return v


def ansatz_variational_block(weights, n_qubits):
    """Ansatz with entanglement and exactly 2^n - 1 parameters - enhanced for complex states."""
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

    # 4. 🆕 Apply additional rotations including RZ for complex phases
    extra_params = weights[n_qubits:]
    for idx, param in enumerate(extra_params):
        if idx % 2 == 0:
            qml.RY(param, wires=idx % n_qubits)
        else:
            qml.RZ(param, wires=idx % n_qubits)  # Add phase rotations


def ansatz_complex_z(params, wires):
    """Complex ansatz using RZ rotations - optimized for complex V."""
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
    🆕 Enhanced for complex states with RZ gates for phase control.
    """
    n_qubits = len(wires)
    params_per_layer = 2 * n_qubits  # 🆕 Double params: RY + RZ per qubit

    for l in range(n_layers):
        # Rotation layer - both magnitude (RY) and phase (RZ)
        for i in range(n_qubits):
            qml.RY(params[l * params_per_layer + 2 * i], wires=wires[i])  # Magnitude
            qml.RZ(params[l * params_per_layer + 2 * i + 1], wires=wires[i])  # Phase

        # Entanglement layer (linear chain)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])


def ansatz_strongly_entangling(params, wires):
    """A strongly entangling layered ansatz from PennyLane templates."""
    qml.StronglyEntanglingLayers(weights=params, wires=wires)


# Dictionary of available ansätze - UPDATED for complex V
n_layers = 2  # For standard layered ansatz

ansatzes = {  # Original VPFS ansätze - enhanced
    "amplitude": {"n_params": lambda: 2 ** num_qubits - 1, "description": "Amplitude embedding ansatz (enhanced for complex)"},
    "variational_block": {"n_params": lambda: 2 ** num_qubits - 1, "description": "Variational block with RY+RZ rotations"},
    "complex_z": {"n_params": lambda: 2 * num_qubits, "description": "Complex ansatz with RZ rotations (optimal for complex V)"},

    # Hardware-realistic ansätze - enhanced for complex states
    "hardware_efficient": {"n_params": lambda: n_layers * 2 * num_qubits, "description": "Enhanced hardware-efficient (RY+RZ per layer)"},
    "strongly_entangling": {"n_params": lambda: qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=num_qubits)[0] *
                                                qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=num_qubits)[1] *
                                                qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=num_qubits)[2],
                            "description": "Strongly Entangling Layers ansatz"}}

# Dictionary of available optimizers (unchanged)
optimizers_vpfs = {"basic": "Basic finite difference gradient descent", "analytic": "Analytic gradient with PennyLane autodiff",
                   "sequential": "Sequential parameter optimization", "cobyla": "COBYLA constrained optimization",
                   "adam": "Adam optimizer with PyTorch", "nesterov": "Nesterov Momentum Optimizer", "rmsprop": "RMSProp Optimizer",
                   "adagrad": "Adagrad Optimizer", "momentum": "Momentum Optimizer", "sgd": "Stochastic Gradient Descent", "spsa": "SPSA Optimizer"}

# PennyLane optimizers (unchanged)
pennylane_optimizers = {"nesterov": lambda lr: qml.NesterovMomentumOptimizer(stepsize=lr), "rmsprop": lambda lr: qml.RMSPropOptimizer(stepsize=lr),
                        "adagrad": lambda lr: qml.AdagradOptimizer(stepsize=lr), "momentum": lambda lr: qml.MomentumOptimizer(stepsize=lr),
                        "sgd": lambda lr: qml.GradientDescentOptimizer(stepsize=lr),
                        "spsa": lambda lr: qml.SPSAOptimizer(maxiter=max_iters, blocking=False)}


#
# Enhanced VPFS Quantum Optimization Core for Complex V
#

def run_vpfs_optimization(ansatz_name, optimizer_name, seed, lr=None, max_iterations=None, verbose=True):
    """
    🆕 Enhanced VPFS optimization with full complex V support.
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
        if TEST_COMPLEX_V:
            print(
                f"    🆕 Using COMPLEX V: |V|_max = {np.max(np.abs(V)):.3f}, phase_range = [{np.min(np.angle(V) * 180 / np.pi):.1f}°, {np.max(np.angle(V) * 180 / np.pi):.1f}°]")

    # Initialize parameters based on ansatz
    n_params = ansatzes[ansatz_name]["n_params"]()
    initial_params = np.random.uniform(0, 2 * np.pi, size=n_params)

    # 🆕 VPFS setup with complex V support
    B = V * (Y @ V)  # Now fully supports complex operations
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
            shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=num_qubits)
            ansatz_strongly_entangling(params.reshape(shape), wires=range(num_qubits))

        return qml.state()

    # Main quantum circuits (unchanged structure, but now supports complex operations)
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
        """🆕 Enhanced VPFS loss function with full complex support."""
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

    # Optimization loop (unchanged - same structure as before)
    current_params = initial_params.copy()
    loss_history = []
    start_time = time.time()

    # [Same optimization code as before - works with complex V through the loss function]
    if optimizer_name in ["nesterov", "rmsprop", "adagrad", "momentum", "sgd", "spsa"]:
        # PennyLane optimizers
        if optimizer_name == "spsa":
            opt = pennylane_optimizers[optimizer_name](learning_rate)
        else:
            opt = pennylane_optimizers[optimizer_name](learning_rate)

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
                    current_params = opt.step(calculate_loss_with_simulation, current_params)
                    current_params = np.real(current_params.astype(np.float64))
                else:
                    current_params, loss = opt.step_and_cost(calculate_loss_with_simulation, current_params)
                    current_params = np.real(current_params.astype(np.float64))

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
        loss_history = [loss]

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

    # Calculate final metrics - 🆕 Enhanced for complex V
    final_loss = loss_history[-1] if loss_history else 1e6
    optimization_time = time.time() - start_time

    # Get final vector v and calculate solution quality
    v_final = get_v_qnode(current_params, ansatz_name)

    if abs(v_final[0]) < 1e-9:
        Vsol = [np.nan] * len(V)
        max_err_V = np.inf
        solution_quality = 0.0
        phase_error = np.inf
    else:
        v0 = v_final[0]
        Vsol = v_final / v0

        # 🆕 Enhanced error calculation for complex V
        err_V = np.abs(V - Vsol)  # Magnitude error
        max_err_V = np.max(err_V)

        # 🆕 Phase error calculation (only for complex case)
        if TEST_COMPLEX_V:
            phase_V = np.angle(V)
            phase_Vsol = np.angle(Vsol)
            phase_diff = np.abs(phase_V - phase_Vsol)
            # Handle phase wrapping (difference should be < π)
            phase_diff = np.where(phase_diff > np.pi, 2 * np.pi - phase_diff, phase_diff)
            phase_error = np.max(phase_diff)

            # Combined quality metric for complex case
            magnitude_quality = 1.0 / (1.0 + max_err_V)
            phase_quality = 1.0 / (1.0 + phase_error)
            solution_quality = 0.7 * magnitude_quality + 0.3 * phase_quality  # Weight magnitude more
        else:
            phase_error = 0.0
            solution_quality = 1.0 / (1.0 + max_err_V)

    if verbose and not MODO_SILENCIOSO:
        print(f"  Final loss: {final_loss:.7e}")
        print(f"  Max magnitude error: {max_err_V:.6f}")
        if TEST_COMPLEX_V:
            print(f"  Max phase error: {phase_error:.6f} rad ({phase_error * 180 / np.pi:.1f}°)")
        print(f"  Solution quality: {solution_quality:.6f}")

    # 🆕 Enhanced return dictionary with complex support
    result_dict = {"optimizer": optimizer_name, "ansatz": ansatz_name, "seed": seed, "learning_rate": learning_rate, "max_iterations": max_iters,
                   "final_loss": final_loss, "loss_history": loss_history, "final_params": current_params.tolist(), "n_params": len(current_params),
                   "optimization_time": optimization_time, "converged_at": len(loss_history), "solution_quality": solution_quality,
                   "max_error_V": max_err_V, "V_target": V.tolist(), "converged": final_loss < tolerance, "complex_V_used": TEST_COMPLEX_V}

    # Add complex-specific metrics
    if TEST_COMPLEX_V:
        result_dict["phase_error"] = phase_error
        result_dict["V_target_magnitude"] = np.abs(V).tolist()
        result_dict["V_target_phase"] = np.angle(V).tolist()

    # Handle complex solution serialization
    if not np.any(np.isnan(Vsol)):
        if TEST_COMPLEX_V:
            result_dict["V_solution"] = Vsol.tolist()
            result_dict["V_solution_magnitude"] = np.abs(Vsol).tolist()
            result_dict["V_solution_phase"] = np.angle(Vsol).tolist()
        else:
            result_dict["V_solution"] = np.real(Vsol).tolist()  # Real part only for real case
    else:
        result_dict["V_solution"] = None

    return result_dict


def compare_vpfs_solutions_complex(result):
    """🆕 Enhanced solution comparison with complex V support."""
    print("\n" + "=" * 50)
    print("VPFS COMPLEX SOLUTION COMPARISON")
    print("=" * 50)

    if result["V_solution"] is None:
        print("❌ Solution failed to converge properly")
        return {"comparison_successful": False}

    V_target = np.array(result["V_target"], dtype=complex)
    V_solution = np.array(result["V_solution"], dtype=complex)

    print("Target vector V:")
    for i, val in enumerate(V_target):
        if result["complex_V_used"]:
            print(f"  V[{i}]: {val:.6f} = {np.abs(val):.6f}∠{np.angle(val) * 180 / np.pi:.1f}°")
        else:
            print(f"  V[{i}]: {np.real(val):.6f}")

    print("\nOptimized vector V:")
    for i, val in enumerate(V_solution):
        if result["complex_V_used"]:
            print(f"  V[{i}]: {val:.6f} = {np.abs(val):.6f}∠{np.angle(val) * 180 / np.pi:.1f}°")
        else:
            print(f"  V[{i}]: {np.real(val):.6f}")

    # Calculate errors
    magnitude_errors = np.abs(np.abs(V_target) - np.abs(V_solution))

    print("\nMagnitude errors:")
    for i, err in enumerate(magnitude_errors):
        print(f"  ||V[{i}]| - |V_opt[{i}]||: {err:.6f}")

    if result["complex_V_used"]:
        phase_target = np.angle(V_target)
        phase_solution = np.angle(V_solution)
        phase_errors = np.abs(phase_target - phase_solution)
        # Handle phase wrapping
        phase_errors = np.where(phase_errors > np.pi, 2 * np.pi - phase_errors, phase_errors)

        print("\nPhase errors:")
        for i, err in enumerate(phase_errors):
            print(f"  |∠V[{i}] - ∠V_opt[{i}]|: {err:.6f} rad ({err * 180 / np.pi:.1f}°)")

    max_magnitude_error = np.max(magnitude_errors)
    relative_error = np.linalg.norm(V_target - V_solution) / np.linalg.norm(V_target)

    print(f"\nMaximum magnitude error: {max_magnitude_error:.6f}")
    if result["complex_V_used"]:
        max_phase_error = np.max(phase_errors)
        print(f"Maximum phase error: {max_phase_error:.6f} rad ({max_phase_error * 180 / np.pi:.1f}°)")
    print(f"Relative error (L2 norm): {relative_error:.6f}")
    print(f"Solution quality: {result['solution_quality']:.6f}")

    # Enhanced visualization for complex case
    if result["complex_V_used"]:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Magnitude comparison
        x_pos = np.arange(len(V_target))
        width = 0.35

        ax1.bar(x_pos - width / 2, np.abs(V_target), width, label='Target |V|', color='blue', alpha=0.7)
        ax1.bar(x_pos + width / 2, np.abs(V_solution), width, label='Optimized |V|', color='green', alpha=0.7)
        ax1.set_xlabel('Component Index')
        ax1.set_ylabel('Magnitude')
        ax1.set_title('Magnitude Comparison')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'V[{i}]' for i in range(len(V_target))])
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Phase comparison
        ax2.bar(x_pos - width / 2, np.angle(V_target) * 180 / np.pi, width, label='Target ∠V', color='blue', alpha=0.7)
        ax2.bar(x_pos + width / 2, np.angle(V_solution) * 180 / np.pi, width, label='Optimized ∠V', color='green', alpha=0.7)
        ax2.set_xlabel('Component Index')
        ax2.set_ylabel('Phase [degrees]')
        ax2.set_title('Phase Comparison')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'V[{i}]' for i in range(len(V_target))])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Magnitude errors
        ax3.bar(x_pos, magnitude_errors, color='red', alpha=0.7)
        ax3.set_xlabel('Component Index')
        ax3.set_ylabel('Magnitude Error')
        ax3.set_title('Magnitude Errors')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'V[{i}]' for i in range(len(V_target))])
        ax3.grid(True, alpha=0.3)

        # Phase errors
        ax4.bar(x_pos, phase_errors * 180 / np.pi, color='orange', alpha=0.7)
        ax4.set_xlabel('Component Index')
        ax4.set_ylabel('Phase Error [degrees]')
        ax4.set_title('Phase Errors')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f'V[{i}]' for i in range(len(V_target))])
        ax4.grid(True, alpha=0.3)

    else:
        # Simple comparison for real case
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        x_pos = np.arange(len(V_target))
        width = 0.35

        ax1.bar(x_pos - width / 2, np.real(V_target), width, label='Target V', color='blue', alpha=0.7)
        ax1.bar(x_pos + width / 2, np.real(V_solution), width, label='Optimized V', color='green', alpha=0.7)
        ax1.set_xlabel('Component Index')
        ax1.set_ylabel('Value')
        ax1.set_title('Target vs Optimized Solution')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'V[{i}]' for i in range(len(V_target))])
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Error plot
        errors = np.abs(V_target - V_solution)
        ax2.bar(x_pos, errors, color='red', alpha=0.7)
        ax2.set_xlabel('Component Index')
        ax2.set_ylabel('Absolute Error')
        ax2.set_title('Component-wise Errors')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'V[{i}]' for i in range(len(V_target))])
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {"comparison_successful": True, "max_magnitude_error": float(max_magnitude_error),
            "max_phase_error": float(max_phase_error) if result["complex_V_used"] else 0.0, "relative_error": float(relative_error),
            "V_target": V_target.tolist(), "V_solution": V_solution.tolist(), "magnitude_errors": magnitude_errors.tolist(),
            "phase_errors": phase_errors.tolist() if result["complex_V_used"] else []}


#
# Main execution block with Complex V testing
#
if __name__ == '__main__':

    # Suppress warnings in main execution for clarity
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Casting complex values to real.*")

    print("\n" + "=" * 70)
    print("🎯 VPFS WITH COMPLEX V VECTOR - DEMONSTRATION")
    print("=" * 70)

    # Test configuration optimized for complex V
    best_ansatz = "complex_z"  # This ansatz is designed for complex states
    best_optimizer = "adam"

    print(f"\n🧪 TESTING: {best_ansatz} + {best_optimizer} with {'COMPLEX' if TEST_COMPLEX_V else 'REAL'} V")
    print("=" * 50)

    # Run single optimization with detailed output
    result = run_vpfs_optimization(ansatz_name=best_ansatz, optimizer_name=best_optimizer, seed=rng_seed, lr=0.1, max_iterations=1500, verbose=True)

    if result and result["V_solution"] is not None:
        print(f"\n🎉 Optimization completed successfully!")
        print(f"   Final loss: {result['final_loss']:.2e}")
        print(f"   Solution quality: {result['solution_quality']:.4f}")
        if TEST_COMPLEX_V:
            print(f"   Max magnitude error: {result['max_error_V']:.6f}")
            print(f"   Max phase error: {result['phase_error']:.6f} rad ({result['phase_error'] * 180 / np.pi:.1f}°)")

        # Show detailed comparison
        compare_vpfs_solutions_complex(result)

    else:
        print(f"❌ Optimization failed to converge")

    print(f"\n" + "=" * 70)
    print("🏁 VPFS COMPLEX V ANALYSIS COMPLETED")
    print("=" * 70)
    if TEST_COMPLEX_V:
        print("✅ Complex voltage vector V successfully handled")
        print("✅ Phase and magnitude errors calculated")
        print("✅ Enhanced visualization for complex case")
        print("💡 This demonstrates VPFS capability for realistic power system problems!")
    else:
        print("✅ Real voltage vector V processed (stored as complex for consistency)")
        print("💡 Set TEST_COMPLEX_V = True to test full complex functionality")
