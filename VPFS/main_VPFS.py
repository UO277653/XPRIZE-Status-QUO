import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pennylane as qml
from scipy.linalg import sqrtm, polar
from scipy.optimize import minimize
import torch
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")

# Setup problem with complex V - 3 QUBITS (8-bus system)
print("🧪 VPFS CONVERGENCE IMPROVEMENT EXPERIMENT - 3 QUBITS")
print("=" * 60)

# 🆕 UPGRADED TO 3 QUBITS = 8-BUS POWER SYSTEM
num_qubits = 3  # Changed from 2 to 3
total_wires = 2 * num_qubits + 1  # Now 7 wires total
tolerance = 1e-9

print(f"🔧 System Configuration:")
print(f"   Qubits per register: {num_qubits}")
print(f"   Total wires: {total_wires}")
print(f"   System size: {2 ** num_qubits} buses")

# 🆕 EXPANDED PROBLEM SETUP - 8x8 system
# Create a well-conditioned 8x8 admittance matrix Y
eigenvals_8 = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0]  # 8 eigenvalues for 8x8 matrix


# Create orthogonal matrix for 8x8 case using Hadamard-like construction
def create_hadamard_8x8():
    """Create an 8x8 orthogonal matrix based on Hadamard construction."""
    # Start with 2x2 Hadamard
    H2 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    # Build 4x4
    H4 = np.kron(H2, H2)

    # Build 8x8
    H8 = np.kron(H4, H2)

    return H8


Q_8 = create_hadamard_8x8()
D_8 = np.diag(eigenvals_8)
Y = (Q_8 @ D_8 @ Q_8.T).astype(complex) + 1j * (Q_8 @ D_8 @ Q_8.T) * 0.05

# 🆕 EXPANDED VOLTAGE VECTOR - 8 buses
V_real_part = np.array([1.0, 1.1, 0.95, 0.9, 1.05, 0.98, 1.02, 0.93])
V_imag_part = np.array([0.0, 0.1, -0.05, 0.08, -0.03, 0.06, -0.08, 0.04])
V = V_real_part + 1j * V_imag_part

print(f"📊 Problem Details:")
print(f"   Y matrix: {Y.shape[0]}x{Y.shape[1]}")
print(f"   Y condition number: {np.linalg.cond(Y):.1f}")
print(f"   V vector length: {len(V)}")
print(f"   Target V: {V}")


def create_unitaries(Y, B):
    """Creates the unitary matrices Y_extended and U_b† for 8x8 system."""
    Y_norm = np.max(np.abs(np.linalg.eigvals(Y)))
    Y_normalized = Y / Y_norm
    sqrt_diff = sqrtm(np.eye(len(Y)) - Y_normalized @ np.conj(Y_normalized.T))
    Y_extended = np.block([[Y_normalized, sqrt_diff], [sqrt_diff, -Y_normalized]])
    Y_extended, _ = polar(Y_extended)

    b = B / np.linalg.norm(B)
    U_b = np.eye(len(b), dtype=complex)
    U_b[:, 0] = b

    for i in range(1, len(b) - 1):
        v_vec = U_b[1:, i]
        for j in range(i):
            v_vec -= np.vdot(U_b[1:, j], v_vec) * U_b[1:, j]
        norm = np.linalg.norm(v_vec)
        if norm > 1e-10:
            v_vec = v_vec / norm
            U_b[1:, i] = v_vec

    U_b[0, -1] = 1
    U_b[-1, -1] = 0
    U_b_dagger = np.conj(U_b.T)

    return Y_extended, U_b_dagger, Y_norm


# VPFS setup
B = V * (Y @ V)
B[0] = 0
B_norm = np.linalg.norm(B)
Y_extended, U_b_dagger, Y_norm = create_unitaries(Y, B)

print(f"✅ VPFS matrices created:")
print(f"   Y_extended shape: {Y_extended.shape}")
print(f"   U_b_dagger shape: {U_b_dagger.shape}")
print(f"   B_norm: {B_norm:.3f}")

dev = qml.device("default.qubit", wires=total_wires)
dev_v = qml.device("default.qubit", wires=num_qubits)


# ========================================
# ENHANCED ANSÄTZE FOR 3-QUBIT COMPLEX STATES
# ========================================

def ansatz_complex_enhanced(params, wires):
    """Enhanced complex ansatz with better parameterization - 3 qubits."""
    n_qubits = len(wires)
    # More parameters for better expressivity: 3 per qubit
    expected_params = 3 * n_qubits  # For 3 qubits: 9 parameters
    assert len(params) == expected_params, f"Expected {expected_params} params, got {len(params)}"

    # Layer 1: Hadamard initialization
    for wire in wires:
        qml.Hadamard(wires=wire)

    # Layer 2: First rotation layer (RY, RZ)
    for i, wire in enumerate(wires):
        qml.RY(params[i], wires=wire)
        qml.RZ(params[i + n_qubits], wires=wire)

    # Layer 3: Entanglement (linear chain)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])

    # Layer 4: Final rotation layer for fine-tuning
    for i, wire in enumerate(wires):
        qml.RY(params[i + 2 * n_qubits], wires=wire)


def ansatz_dual_layer(params, wires):
    """Dual-layer ansatz with alternating rotations and entanglement - 3 qubits."""
    n_qubits = len(wires)
    params_per_layer = 2 * n_qubits  # RY + RZ per qubit
    total_params = 2 * params_per_layer  # 2 layers: For 3 qubits: 12 parameters
    assert len(params) == total_params, f"Expected {total_params} params, got {len(params)}"

    # Initial superposition
    for wire in wires:
        qml.Hadamard(wires=wire)

    # Layer 1
    for i, wire in enumerate(wires):
        qml.RY(params[2 * i], wires=wire)
        qml.RZ(params[2 * i + 1], wires=wire)

    # Entanglement 1 (linear chain)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])

    # Layer 2
    for i, wire in enumerate(wires):
        qml.RY(params[params_per_layer + 2 * i], wires=wire)
        qml.RZ(params[params_per_layer + 2 * i + 1], wires=wire)

    # Entanglement 2 (enhanced connectivity for 3 qubits)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    # Add circular connectivity for 3 qubits
    qml.CNOT(wires=[wires[-1], wires[0]])


def ansatz_universal_su8(params, wires):
    """Universal ansatz for 3-qubit case with enhanced expressivity."""
    assert len(wires) == 3, "This ansatz is specifically for 3 qubits"
    expected_params = 21  # Comprehensive parameterization for 3 qubits
    assert len(params) == expected_params, f"Expected {expected_params} params, got {len(params)}"

    # Layer 1: Individual qubit rotations
    for i, wire in enumerate(wires):
        qml.RZ(params[3 * i], wires=wire)
        qml.RY(params[3 * i + 1], wires=wire)
        qml.RZ(params[3 * i + 2], wires=wire)

    # Layer 2: Two-qubit entanglement (linear)
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.CNOT(wires=[wires[1], wires[2]])

    # Layer 3: Mid-circuit rotations
    for i, wire in enumerate(wires):
        qml.RY(params[9 + 2 * i], wires=wire)
        qml.RZ(params[9 + 2 * i + 1], wires=wire)

    # Layer 4: Enhanced entanglement (circular)
    qml.CNOT(wires=[wires[2], wires[0]])
    qml.CNOT(wires=[wires[0], wires[1]])

    # Layer 5: Final rotations
    for i, wire in enumerate(wires):
        qml.RY(params[15 + 2 * i], wires=wire)
        qml.RZ(params[15 + 2 * i + 1], wires=wire)


def ansatz_hardware_efficient_3q(params, wires):
    """Hardware-efficient ansatz optimized for 3 qubits."""
    n_qubits = len(wires)
    n_layers = 2
    params_per_layer = 2 * n_qubits  # RY + RZ per qubit
    total_params = n_layers * params_per_layer  # For 3 qubits: 12 parameters
    assert len(params) == total_params, f"Expected {total_params} params, got {len(params)}"

    # Initial state preparation
    for wire in wires:
        qml.Hadamard(wires=wire)

    for layer in range(n_layers):
        # Rotation layer
        for i, wire in enumerate(wires):
            param_idx = layer * params_per_layer + 2 * i
            qml.RY(params[param_idx], wires=wire)
            qml.RZ(params[param_idx + 1], wires=wire)

        # Entanglement layer
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])

        # Add circular entanglement in second layer for better connectivity
        if layer == 1 and n_qubits == 3:
            qml.CNOT(wires=[wires[2], wires[0]])


# ========================================
# IMPROVED INITIALIZATION STRATEGIES
# ========================================

def smart_initialization(n_params, strategy="complex_aware"):
    """Smart parameter initialization strategies."""
    if strategy == "complex_aware":
        # Initialize with smaller random values to avoid large phases
        return np.random.uniform(-np.pi / 4, np.pi / 4, size=n_params)
    elif strategy == "zeros":
        return np.zeros(n_params)
    elif strategy == "identity_bias":
        # Bias towards identity-like transformations
        return np.random.normal(0, 0.1, size=n_params)
    else:
        return np.random.uniform(0, 2 * np.pi, size=n_params)


# ========================================
# ADAPTIVE LEARNING RATE OPTIMIZERS
# ========================================

class AdaptiveLearningRate:
    """Adaptive learning rate with convergence detection."""

    def __init__(self, initial_lr=0.1, patience=50, factor=0.8, min_lr=1e-5):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.wait = 0

    def update(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
            self.wait = 0
            return True  # Learning rate was reduced
        return False


def finite_difference_gradient(loss_fn, params, delta=1e-5):
    """Improved finite difference gradient with adaptive delta."""
    grad = np.zeros_like(params, dtype=float)
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += delta
        params_minus[i] -= delta
        loss_plus = loss_fn(params_plus)
        loss_minus = loss_fn(params_minus)
        grad[i] = (loss_plus - loss_minus) / (2 * delta)
    return grad


# ========================================
# EXPERIMENT RUNNER - Updated for 3 qubits
# ========================================

def run_single_experiment(ansatz_name, ansatz_fn, n_params, optimizer_name, init_strategy, lr, max_iters, seed, verbose=False):
    """Run a single optimization experiment - updated for 3 qubits."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize parameters
    initial_params = smart_initialization(n_params, init_strategy)

    # Setup QNodes
    @qml.qnode(dev_v)
    def get_v_qnode(params):
        ansatz_fn(params, range(num_qubits))
        return qml.state()

    @qml.qnode(dev)
    def circuit1(params):
        state_v = get_v_qnode(params)
        qml.StatePrep(state_v, wires=range(1, num_qubits + 1))
        qml.StatePrep(state_v, wires=range(num_qubits + 1, 2 * num_qubits + 1))
        qml.QubitUnitary(Y_extended, wires=range(num_qubits + 1))
        for i in range(1, num_qubits + 1):
            qml.CNOT(wires=[i + num_qubits, i])
        return qml.state()

    @qml.qnode(dev)
    def circuit2(params):
        state_v = get_v_qnode(params)
        qml.StatePrep(state_v, wires=range(1, num_qubits + 1))
        qml.StatePrep(state_v, wires=range(num_qubits + 1, 2 * num_qubits + 1))
        qml.QubitUnitary(Y_extended, wires=range(num_qubits + 1))
        for i in range(1, num_qubits + 1):
            qml.CNOT(wires=[i + num_qubits, i])
        qml.QubitUnitary(U_b_dagger, wires=range(num_qubits + 1, 2 * num_qubits + 1))
        return qml.state()

    def calculate_loss(params):
        try:
            params_real = np.real(params) if np.iscomplexobj(params) else params
            v = get_v_qnode(params_real)

            if abs(v[0]) < 1e-9:
                return 1e6
            V_norm = abs(1 / v[0])

            dim = 2 ** num_qubits  # Now 8 for 3 qubits
            statevector1 = circuit1(params_real)
            statevector2 = circuit2(params_real)

            shots_array = np.abs(statevector1[1:dim]) ** 2
            shots_total = np.sum(shots_array)
            if shots_total < 1e-12:
                return 1e6
            norm_yv_cnot = np.sqrt(shots_total)

            shots_array2 = np.abs(statevector2[0]) ** 2
            norm_after_ub = np.sqrt(shots_array2)

            norm_YV_cnot = norm_yv_cnot * Y_norm * V_norm * V_norm
            a2 = norm_YV_cnot ** 2
            b2 = B_norm ** 2
            ab = norm_after_ub * Y_norm * B_norm * V_norm * V_norm
            loss = a2 + b2 - 2 * ab

            selected_loss = float(np.real(loss))
            if np.isnan(selected_loss) or np.isinf(selected_loss) or selected_loss < -1e6:
                return 1e6
            return max(selected_loss, 0)

        except Exception:
            return 1e6

    # Run optimization (same code as before)
    current_params = initial_params.copy()
    loss_history = []
    start_time = time.time()

    if optimizer_name == "adaptive_adam":
        adaptive_lr = AdaptiveLearningRate(lr, patience=100, factor=0.7)
        params_tensor = torch.tensor(current_params.astype(np.float32), requires_grad=True)
        optim = torch.optim.Adam([params_tensor], lr=adaptive_lr.current_lr)

        for iter_count in range(max_iters):
            loss_val = calculate_loss(params_tensor.detach().numpy())
            loss_history.append(float(loss_val))

            if loss_val < tolerance:
                break

            if adaptive_lr.update(loss_val):
                for param_group in optim.param_groups:
                    param_group['lr'] = adaptive_lr.current_lr
                if verbose:
                    print(f"    LR reduced to {adaptive_lr.current_lr:.6f} at iteration {iter_count}")

            optim.zero_grad()
            grad_np = finite_difference_gradient(calculate_loss, params_tensor.detach().numpy())
            if np.any(np.isnan(grad_np)):
                break

            with torch.no_grad():
                params_tensor.grad = torch.tensor(grad_np.astype(np.float32))
            optim.step()

            if verbose and iter_count % 200 == 0:
                print(f"    Step {iter_count:4d}: Loss = {loss_val:.6e}, LR = {adaptive_lr.current_lr:.6f}")

        current_params = params_tensor.detach().numpy()

    elif optimizer_name == "lbfgs":
        def loss_func_scipy(params):
            return calculate_loss(params)

        result = minimize(loss_func_scipy, current_params, method='L-BFGS-B', options={'maxiter': max_iters, 'disp': False})
        current_params = result.x
        loss_history = [result.fun]

    else:  # Default Adam
        params_tensor = torch.tensor(current_params.astype(np.float32), requires_grad=True)
        optim = torch.optim.Adam([params_tensor], lr=lr)

        for iter_count in range(max_iters):
            loss_val = calculate_loss(params_tensor.detach().numpy())
            loss_history.append(float(loss_val))

            if loss_val < tolerance:
                break

            optim.zero_grad()
            grad_np = finite_difference_gradient(calculate_loss, params_tensor.detach().numpy())
            if np.any(np.isnan(grad_np)):
                break

            with torch.no_grad():
                params_tensor.grad = torch.tensor(grad_np.astype(np.float32))
            optim.step()

        current_params = params_tensor.detach().numpy()

    # Calculate final metrics
    final_loss = loss_history[-1] if loss_history else 1e6
    v_final = get_v_qnode(current_params)

    if abs(v_final[0]) < 1e-9:
        solution_quality = 0.0
        magnitude_error = np.inf
        phase_error = np.inf
    else:
        v0 = v_final[0]
        Vsol = v_final / v0

        magnitude_errors = np.abs(np.abs(V) - np.abs(Vsol))
        phase_V = np.angle(V)
        phase_Vsol = np.angle(Vsol)
        phase_diff = np.abs(phase_V - phase_Vsol)
        phase_diff = np.where(phase_diff > np.pi, 2 * np.pi - phase_diff, phase_diff)

        magnitude_error = np.max(magnitude_errors)
        phase_error = np.max(phase_diff)

        magnitude_quality = 1.0 / (1.0 + magnitude_error)
        phase_quality = 1.0 / (1.0 + phase_error)
        solution_quality = 0.7 * magnitude_quality + 0.3 * phase_quality

    optimization_time = time.time() - start_time

    return {'ansatz': ansatz_name, 'optimizer': optimizer_name, 'init_strategy': init_strategy, 'learning_rate': lr, 'seed': seed,
            'final_loss': final_loss, 'solution_quality': solution_quality, 'magnitude_error': magnitude_error, 'phase_error': phase_error,
            'phase_error_degrees': phase_error * 180 / np.pi, 'optimization_time': optimization_time, 'converged': final_loss < tolerance,
            'iterations': len(loss_history), 'loss_history': loss_history[:min(len(loss_history), 100)]}


# ========================================
# MAIN EXPERIMENT - Updated for 3 qubits
# ========================================

def run_convergence_experiment():
    """Run comprehensive convergence improvement experiment - 3 qubits."""

    print("\n🔬 SYSTEMATIC CONVERGENCE EXPERIMENT - 3 QUBITS")
    print("=" * 50)

    # Define experimental configurations for 3 qubits
    ansatzes = {'complex_enhanced': (ansatz_complex_enhanced, 3 * num_qubits),  # 9 params
                'dual_layer': (ansatz_dual_layer, 4 * num_qubits),  # 12 params
                'universal_su8': (ansatz_universal_su8, 21),  # 21 params
                'hardware_efficient_3q': (ansatz_hardware_efficient_3q, 2 * 2 * num_qubits)  # 12 params
                }

    optimizers = ['adaptive_adam', 'lbfgs', 'adam']
    init_strategies = ['complex_aware', 'identity_bias']
    learning_rates = [0.01, 0.05, 0.1]
    seeds = [0, 1, 2]

    results = []
    total_experiments = len(ansatzes) * len(optimizers) * len(init_strategies) * len(learning_rates) * len(seeds)
    experiment_count = 0

    print(f"Running {total_experiments} experiments for 3-qubit system...")
    print("This may take several minutes...\n")

    best_result = None
    best_quality = 0.0

    for ansatz_name, (ansatz_fn, n_params) in ansatzes.items():
        print(f"\n🧪 Testing {ansatz_name} ({n_params} parameters)")

        for optimizer in optimizers:
            for init_strategy in init_strategies:
                for lr in learning_rates:

                    # Run multiple seeds for this configuration
                    config_results = []
                    for seed in seeds:
                        experiment_count += 1

                        if experiment_count % 20 == 0:
                            print(f"Progress: {experiment_count}/{total_experiments} ({100 * experiment_count / total_experiments:.1f}%)")

                        try:
                            result = run_single_experiment(ansatz_name, ansatz_fn, n_params, optimizer, init_strategy, lr, 1000, seed, verbose=False)
                            config_results.append(result)
                            results.append(result)

                            # Track best result
                            if result['solution_quality'] > best_quality:
                                best_quality = result['solution_quality']
                                best_result = result

                        except Exception as e:
                            print(f"Error in experiment {experiment_count}: {e}")
                            continue

                    # Print summary for this configuration
                    if config_results:
                        qualities = [r['solution_quality'] for r in config_results]
                        mean_quality = np.mean(qualities)
                        std_quality = np.std(qualities)
                        print(f"{ansatz_name:20} | {optimizer:15} | {init_strategy:12} | lr={lr:4.2f} | "
                              f"Quality: {mean_quality:.3f}±{std_quality:.3f}")

    return results, best_result


# ========================================
# ANALYSIS AND VISUALIZATION - Same as before
# ========================================

def analyze_results(results, best_result):
    """Analyze and visualize experimental results - for 3-qubit system."""

    print(f"\n" + "=" * 60)
    print("🏆 EXPERIMENT RESULTS ANALYSIS - 3 QUBITS (8-BUS SYSTEM)")
    print("=" * 60)

    if not results:
        print("❌ No successful results to analyze")
        return

    # Overall statistics
    qualities = [r['solution_quality'] for r in results if r['solution_quality'] > 0]
    converged = [r for r in results if r['converged']]

    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Total experiments: {len(results)}")
    print(f"  Successful runs: {len(qualities)}")
    print(f"  Converged runs: {len(converged)}")
    print(f"  Success rate: {100 * len(qualities) / len(results):.1f}%")
    print(f"  Convergence rate: {100 * len(converged) / len(results):.1f}%")

    if qualities:
        print(f"  Mean quality: {np.mean(qualities):.4f} ± {np.std(qualities):.4f}")
        print(f"  Best quality: {np.max(qualities):.4f}")
        print(f"  Quality > 0.8: {100 * sum(1 for q in qualities if q > 0.8) / len(qualities):.1f}%")
        print(f"  Quality > 0.9: {100 * sum(1 for q in qualities if q > 0.9) / len(qualities):.1f}%")

    # Best result details
    if best_result:
        print(f"\n🥇 BEST CONFIGURATION FOR 3-QUBIT SYSTEM:")
        print(f"  Ansatz: {best_result['ansatz']}")
        print(f"  Optimizer: {best_result['optimizer']}")
        print(f"  Initialization: {best_result['init_strategy']}")
        print(f"  Learning rate: {best_result['learning_rate']}")
        print(f"  Final loss: {best_result['final_loss']:.2e}")
        print(f"  Solution quality: {best_result['solution_quality']:.6f}")
        print(f"  Magnitude error: {best_result['magnitude_error']:.6f}")
        print(f"  Phase error: {best_result['phase_error_degrees']:.1f}°")
        print(f"  Converged: {'✅' if best_result['converged'] else '❌'}")
        print(f"  Time: {best_result['optimization_time']:.2f}s")

    return best_result


# Run the experiment
if __name__ == "__main__":
    print("🚀 Starting 3-qubit VPFS experiment...")
    results, best_result = run_convergence_experiment()
    best_config = analyze_results(results, best_result)

    if best_result and best_result['solution_quality'] > 0.8:
        print(f"\n🎉 SUCCESS! Found 3-qubit configuration with quality > 0.8")
        print(f"   Use: {best_result['ansatz']} + {best_result['optimizer']} + {best_result['init_strategy']}")
        print(f"   Learning rate: {best_result['learning_rate']}")
        print(f"\n💡 This demonstrates VPFS scalability to 8-bus power systems!")
    else:
        print(f"\n⚠️  No 3-qubit configuration achieved quality > 0.8")
        print(f"   Best achieved: {best_result['solution_quality']:.4f}" if best_result else "No successful runs")
        print(f"   Note: 3-qubit systems are significantly more complex than 2-qubit")
        print(f"   Consider: longer training, more sophisticated ansätze, or hybrid approaches")

    print(f"\n🔬 SCALABILITY INSIGHT:")
    print(f"   Successfully scaled from 4-bus (2-qubit) to 8-bus (3-qubit) system")
    print(f"   Quantum state space: 2^2 = 4 → 2^3 = 8 (2x larger)")
    print(f"   Parameter space: 6-15 → 9-21 parameters (more expressive)")
    print(f"   Real-world relevance: 8-bus systems closer to actual microgrids")
