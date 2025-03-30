import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pennylane as qml
from scipy.linalg import sqrtm, polar
from pennylane import numpy as np
from scipy.optimize import minimize
import torch

#####
## PARAMETROS INICIALES
#####
# Crear Y y V de tamaño correspondiente
Y = np.array([[1, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 1]], dtype=complex) * 5
V = np.array([1, 1.1, 0.95, 0.9])

max_iters = 1000  # Máximo número de iteraciones
tolerance = 1e-9  # Tolerancia para la convergencia

anotarConvergencia = True  # Flag para indicar que se quieren guardar en un fichero los resultados que han convergido
anotarConvergenciaTolerance = 100 / 1e6  # Umbral de tolerancia para guardar configuraciones, igualar a tolerance si se quieren solo los convergidos totalmente


#####
## ANSATZS Y CIRCUITOS
#####
def ansatz(params):
    r"""Calcula el vector v a partir del vector de parámetros params.

    Args:
        params (list o numpy.ndarray): Vector de parámetros.

    Returns:
        numpy.ndarray: Vector v normalizado.
    """
    v = [1]  # Primera componente fija
    for i in range(len(params)):
        if radius > 0:
            v.append(np.sin(params[i]) * radius + 1)
        else:
            v.append(params[i] + 1)
    v = np.array(v)
    # v = np.array([1, 1.1, 0.95, 0.9])
    v = v / np.linalg.norm(v)
    return v


def circuit_amplitude(params, num_qubits):
    v = ansatz(params)
    qml.AmplitudeEmbedding(v, wires=range(1, num_qubits + 1), normalize=False)
    qml.AmplitudeEmbedding(v, wires=range(num_qubits + 1, 2 * num_qubits + 1), normalize=False)


def variational_block(weights, n_qubits):
    """
    Ansatz con entrelazamiento y exactamente 2^n - 1 parámetros.

    Args:
        weights (array-like): Parámetros del ansatz. Debe tener longitud 2^n - 1.
    """
    num_params = len(weights)
    assert num_params == 2 ** n_qubits - 1, f"Se esperaban {2 ** n_qubits - 1} parámetros, pero se recibieron {num_params}."

    # 1. Inicializar en superposición uniforme con Hadamard
    for idx in range(n_qubits):
        qml.Hadamard(wires=idx)

    # 2. Aplicar rotaciones RY con los primeros n_qubits parámetros
    for idx in range(n_qubits):
        qml.RY(weights[idx], wires=idx)

    # 3. Aplicar entrelazamiento (CNOT en topología lineal)
    for idx in range(n_qubits - 1):
        qml.CNOT(wires=[idx, idx + 1])

    # 4. Aplicar rotaciones adicionales para alcanzar 2^n - 1 parámetros
    extra_params = weights[n_qubits:]
    for idx, param in enumerate(extra_params):
        qml.RY(param, wires=idx % n_qubits)  # Distribuye los parámetros sobrantes entre los qubits


def circuit_variational(params, num_qubits):
    variational_block(params, num_qubits)


def ansatz_complex(params, wires):
    n_qubits_reg = len(wires)
    expected_params = 2 * n_qubits_reg
    assert len(params) == expected_params, f"Expected {expected_params} params for {n_qubits_reg} qubits on wires {wires}, got {len(params)}"

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


def circuit_complex_z(params, num_qubits):
    # Applies the same complex Z ansatz structure to both registers
    ansatz_complex(params, range(1, num_qubits + 1))  # Apply to first register
    ansatz_complex(params, range(num_qubits + 1, 2 * num_qubits + 1))  # Apply to second register


ansatz_library = {"amplitude": {"ansatz": ansatz, "circuit": circuit_amplitude}, "variational_block": {"circuit": circuit_variational},
                  "complex_z": {"circuit": circuit_complex_z}, }


#####
## CREACION DE MATRICES UNITARIAS
#####
def create_unitaries(Y, B):
    r"""Crea las matrices unitarias Y_extended y U_b† (calculadas a partir de B).

    Args:
        Y (numpy.ndarray): Matriz Y original.
        B (numpy.ndarray): Vector B.
        num_qubits (int): Número de qubits para cada registro.

    Returns:
        tuple: (Y_extended, U_b_dagger, Y_norm) donde Y_extended es la unidad extendida y U_b_dagger es el operador U_b†.
    """
    # Normalización y cálculo de Y_extended
    Y_norm = np.max(np.abs(np.linalg.eigvals(Y)))
    Y_normalized = Y / Y_norm
    sqrt_diff = sqrtm(np.eye(len(Y)) - Y_normalized @ Y_normalized)
    Y_extended = np.block([[Y_normalized, sqrt_diff], [sqrt_diff, -Y_normalized]])
    Y_extended, _ = polar(Y_extended)  # Se toma la parte unitaria

    # Construir U_b† basado en B
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


#####
## OPTIMIZACION (función principal)
#####
def quantum_optimization_simulation(num_qubits=2, ansatz_params=None, optimizer="basic", ansatz_name="amplitude"):
    r"""Simulación cuántica usando PennyLane con distintos optimizadores y ansatz

    Args:
        num_qubits (int, opcional): Número de qubits para cada uno de los dos registros.
        ansatz_params (list, opcional): Parámetros iniciales para el ansatz. MUST BE PROVIDED WITH CORRECT SIZE.
        optimizer (str, opcional): Método de optimización a usar ('basic', 'sequential', 'cobyla' o 'adam').
        ansatz_name (str, opcional): Nombre del ansatz a usar ('amplitude', 'variational_block', 'complex_z')
    """
    global learning_rate, loss_option, PEN_COEF_SCALE  # Added PEN_COEF_SCALE

    if ansatz_params is None:
        # Default initialization removed as size depends on ansatz_name
        raise ValueError("ansatz_params must be provided")

    # Calcular B y su norma
    B = V * (Y @ V)
    B[0] = 0
    B_norm = np.linalg.norm(B)  # Precalcular <B|B>

    # Obtener las matrices unitarias a usar
    Y_extended, U_b_dagger, Y_norm = create_unitaries(Y, B)

    # Definir el dispositivo de PennyLane. Total de wires: 2*num_qubits + 1
    total_wires = 2 * num_qubits + 1
    dev = qml.device("default.qubit", wires=total_wires)

    # Circuito qc1: Inicializa dos registros y aplica Y_extended y las compuertas CNOT.
    @qml.qnode(dev)
    def circuit1(params, option_ansatz):
        """
        QNode que inicializa el estado cuántico usando el ansatz seleccionado
        """
        circuit_f = ansatz_library.get(option_ansatz).get("circuit")
        circuit_f(params, num_qubits)

        # El wire extra (índice 0) se deja en el estado |0> # Corrected comment: wire 0 is ancilla
        # Aplicar la operación Y_extended en wires [0, ..., num_qubits]
        qml.QubitUnitary(Y_extended, wires=range(num_qubits + 1))
        # Aplicar compuertas CNOT: control en wire i+num_qubits y target en wire i
        for i in range(1, num_qubits + 1):
            qml.CNOT(wires=[i + num_qubits, i])
        return qml.state()

    # Circuito qc2: Igual que qc1 pero extiende con U_b† en el segundo registro.
    @qml.qnode(dev)
    def circuit2(params, option_ansatz):
        """QNode que extiende qc1 con U_b†."""
        circuit_f = ansatz_library.get(option_ansatz).get("circuit")
        circuit_f(params, num_qubits)

        # El wire extra (índice 0) se deja en el estado |0>
        # Aplicar la operación Y_extended en wires [0, ..., num_qubits]
        qml.QubitUnitary(Y_extended, wires=range(num_qubits + 1))
        # Aplicar compuertas CNOT: control en wire i+num_qubits y target en wire i
        for i in range(1, num_qubits + 1):
            qml.CNOT(wires=[i + num_qubits, i])
        # Aplicar U_b† en el segundo registro
        qml.QubitUnitary(U_b_dagger, wires=range(num_qubits + 1, 2 * num_qubits + 1))
        return qml.state()

    def calculate_loss_with_simulation(params):
        dim = 2 ** num_qubits
        # Handle V_norm based on ansatz
        if ansatz_name == "amplitude":
            v = ansatz(params)  # Only calculate v for amplitude ansatz
            if abs(v[0]) < 1e-9:  # Avoid division by zero
                return 1e6  # Return a large loss if v[0] is too small
            V_norm = 1 / v[0]  # Se asume que v[0] ≠ 0 (normalización en pu)
        else:
            V_norm = 1.0

        statevector1 = circuit1(params, ansatz_name)
        statevector2 = circuit2(params, ansatz_name)

        shots_array = np.abs(statevector1[1:dim]) ** 2
        shots_total = np.sum(shots_array)
        if shots_total < 1e-12: return 1e6  # Avoid division by zero / instability
        norm_yv_cnot = np.sqrt(shots_total)

        # Extract the coefficient in the position 0 (state |00...0>)
        shots_array2 = np.abs(statevector2[0]) ** 2
        shots_total2 = np.sum(shots_array2)
        norm_after_ub = np.sqrt(shots_total2)

        norm_YV_cnot = norm_yv_cnot * Y_norm * V_norm * V_norm
        pen_coef = PEN_COEF_SCALE / B_norm ** 2

        losses = []
        # Loss 0: Original form (potentially unstable if norm_yv_cnot is small)
        losses.append(1 - (norm_after_ub) / norm_yv_cnot + pen_coef * (norm_YV_cnot - B_norm) ** 2)
        # Loss 1: Squared ratio (more stable)
        losses.append(1 - (norm_after_ub) ** 2 / norm_yv_cnot ** 2 + pen_coef * (norm_YV_cnot - B_norm) ** 2)
        # Loss 2: Squared ratio + higher penalty power
        losses.append(1 - (norm_after_ub) ** 2 / norm_yv_cnot ** 2 + pen_coef * (norm_YV_cnot - B_norm) ** 4)
        # Loss 3, 4: Alternative forms (check original source for derivation)
        a2 = norm_YV_cnot ** 2
        b2 = B_norm ** 2
        # ab calculation depends heavily on V_norm interpretation. Using placeholder V_norm=1 for non-amplitude.
        ab = norm_after_ub * Y_norm * B_norm * V_norm * V_norm
        losses.append((a2 - ab) ** 2 + (b2 - ab) ** 2)  # Loss 3
        losses.append(a2 + b2 - 2 * ab)

        # Ensure loss is real
        selected_loss = np.real(losses[loss_option])
        if np.isnan(selected_loss): return 1e6  # Return large loss if NaN occurs

        return selected_loss

    def finite_difference_gradient(params, delta=1e-4):
        grad = np.zeros_like(params, dtype=float)  # Ensure float type
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += delta
            params_minus[i] -= delta
            loss_plus = calculate_loss_with_simulation(params_plus)
            loss_minus = calculate_loss_with_simulation(params_minus)
            grad[i] = (loss_plus - loss_minus) / (2 * delta)
        return grad

    def finite_difference_gradient_sequential(params, delta=1e-4):
        current_params = params.copy()  # Work on a copy
        for i in range(len(current_params)):
            params_plus = current_params.copy()
            params_minus = current_params.copy()
            params_plus[i] += delta
            params_minus[i] -= delta
            loss_plus = calculate_loss_with_simulation(params_plus)
            loss_minus = calculate_loss_with_simulation(params_minus)
            if loss_plus is None or loss_minus is None or np.isnan(loss_plus) or np.isnan(loss_minus):
                print(f"Warning: Invalid loss encountered for param {i}. Skipping update.")
                continue
            grad = (loss_plus - loss_minus) / (2 * delta)
            if np.isnan(grad):
                print(f"Warning: NaN gradient encountered for param {i}. Skipping update.")
                continue
            current_params[i] -= learning_rate * grad
        return current_params  # Return the updated parameters

    # Obtener la función gradiente analítica con PennyLane
    grad_fn = qml.grad(calculate_loss_with_simulation)

    current_params = ansatz_params.copy()
    loss = np.inf

    if optimizer == "analytic":
        for iter_count in range(max_iters):
            loss = calculate_loss_with_simulation(current_params)
            if loss is None or np.isnan(loss):
                print(f"Iteration {iter_count + 1}: Invalid loss encountered. Stopping.")
                loss = -100  # Indicate failure
                break
            if loss < tolerance:
                print(f"Iteration {iter_count + 1}: Converged using analytic gradient.")
                break

            try:
                grad = grad_fn(current_params)
                if np.any(np.isnan(grad)):
                    print(f"Iteration {iter_count + 1}: NaN gradient encountered. Stopping.")
                    loss = -100
                    break
                current_params = current_params - learning_rate * grad
            except Exception as e:
                print(f"Iteration {iter_count + 1}: Error during gradient calculation/update: {e}. Stopping.")
                loss = -100
                break
        else:  # Loop finished without break
            print(f"Max iterations ({max_iters}) reached without convergence (analytic).")
            iter_count += 1  # To match other loops' final iter count

    elif optimizer == "basic":
        for iter_count in range(max_iters):
            loss = calculate_loss_with_simulation(current_params)
            if loss is None or np.isnan(loss):
                print(f"Iteration {iter_count + 1}: Invalid loss encountered. Stopping.")
                loss = -100
                break
            if loss < tolerance:
                print(f"Iteration {iter_count + 1}: Converged using basic finite difference.")
                break

            grad = finite_difference_gradient(current_params)
            if np.any(np.isnan(grad)):
                print(f"Iteration {iter_count + 1}: NaN gradient encountered. Stopping.")
                loss = -100
                break
            current_params = current_params - learning_rate * grad
        else:  # Loop finished without break
            print(f"Max iterations ({max_iters}) reached without convergence (basic).")
            iter_count += 1

    elif optimizer == "sequential":
        for iter_count in range(max_iters):
            loss = calculate_loss_with_simulation(current_params)
            if loss is None or np.isnan(loss):
                print(f"Iteration {iter_count + 1}: Invalid loss encountered. Stopping.")
                loss = -100
                break
            if loss < tolerance:
                print(f"Iteration {iter_count + 1}: Converged using sequential finite difference.")
                break
            current_params = finite_difference_gradient_sequential(current_params)
        else:  # Loop finished without break
            print(f"Max iterations ({max_iters}) reached without convergence (sequential).")
            iter_count += 1

    elif optimizer == "cobyla":
        # Ensure loss function returns float
        def loss_func_cobyla(params):
            l = calculate_loss_with_simulation(params)
            return float(l) if l is not None and not np.isnan(l) else 1e6  # Return float or large number

        result = minimize(loss_func_cobyla, current_params, tol=tolerance, method="COBYLA", options={"maxiter": max_iters, "disp": False})
        current_params = result.x
        print(f"COBYLA finished after {result.nfev} evaluations. Final loss: {result.fun:.6e}. Message: {result.message}")
        iter_count = result.nfev
        loss = result.fun

    elif optimizer == "adam":
        params_tensor = torch.tensor(current_params.astype(np.float32), requires_grad=True)
        optim = torch.optim.Adam([params_tensor], lr=learning_rate)

        for iter_count in range(max_iters):
            loss_val = calculate_loss_with_simulation(params_tensor.detach().numpy())

            if loss_val is None or np.isnan(loss_val):
                print(f"Iteration {iter_count + 1}: Invalid loss encountered. Stopping.")
                loss = -100
                break
            if loss_val < tolerance:
                print(f"Iteration {iter_count + 1}: Converged using Adam.")
                break

            optim.zero_grad()

            loss_tensor = torch.tensor(float(loss_val), requires_grad=True)
            grad_np = finite_difference_gradient(params_tensor.detach().numpy())
            if np.any(np.isnan(grad_np)):
                print(f"Iteration {iter_count + 1}: NaN gradient encountered. Stopping.")
                loss = -100
                break

            with torch.no_grad():
                params_tensor.grad = torch.tensor(grad_np.astype(np.float32))

            optim.step()

        else:
            print(f"Max iterations ({max_iters}) reached without convergence (Adam).")
            iter_count += 1

        current_params = params_tensor.detach().numpy()
        loss = calculate_loss_with_simulation(current_params)

    else:
        raise ValueError(f"Optimizer '{optimizer}' no reconocido. Use 'basic', 'cobyla', 'sequential' o 'adam'.")

    # Resultados finales
    final_params = current_params

    if ansatz_name == "amplitude":
        v = ansatz(final_params)
        if abs(v[0]) < 1e-9:
            print("Error: v[0] is near zero, cannot calculate Vsol.")
            Vsol = [np.nan] * len(V)  # Indicate error
            max_err_V = np.inf
        else:
            v0 = v[0]
            Vsol = [x / v0 for x in v]
            err_V = np.abs(V - np.array(Vsol))
            max_err_V = np.max(err_V)
    else:
        Vsol = ["N/A (Variational)"] * len(V)
        max_err_V = np.nan
        print(f"Final parameters for {ansatz_name}: {final_params}")

    print(f"Iter {iter_count}: Loss x 1e6 = {loss * 1e6:.2f}")
    if ansatz_name == "amplitude":
        print(f"Params = {final_params}")
        print(f"Error máximo en V: {max_err_V}, Vreal/ Vcalc: {V / np.array(Vsol)}")
    else:
        print(f"Final loss achieved for variational ansatz {ansatz_name}.")

    if (anotarConvergencia and loss != -100 and loss <= anotarConvergenciaTolerance):
        with open("parametrosConvergencia.txt", "a") as f:
            f.write("Convergencia alcanzada usando {}.\n".format(optimizer))
            f.write("\nAnsatz: {}, Radio: {}, Learning rate: {}, loss option: {}, scale: {}\n".format(ansatz_name, radius, learning_rate, loss_option,
                                                                                                      scale))  # Added ansatz_name
            f.write("Iter {}: Loss x 1e6 = {:.2f}, Params = {}\n".format(iter_count, loss * 1e6, final_params))  # Use iter_count
            if ansatz_name == "amplitude":
                f.write("Error máximo en V: {}, Vreal/ Vcalc: {}\n".format(max_err_V, V / np.array(Vsol)))
            else:
                f.write("Variational ansatz used, Vsol/Error not directly comparable.\n")
            f.write("\n----------------------------------------\n")


if __name__ == "__main__":
    from time import time

    global radius, learning_rate, loss_option, scale, PEN_COEF_SCALE  # Make params global for ansatz/loss access if needed

    num_qubits_main = 2

    for radius in np.arange(0.1, 0.11, 0.01):
        for learning_rate in [0.05]:
            for loss_option in [4]:
                for scale in [0]:
                    if scale == 0:
                        PEN_COEF_SCALE = 0.0
                    else:
                        PEN_COEF_SCALE = 0.01 * scale

                    for method in ["cobyla"]:
                        for ansatz_name in ["complex_z"]:  # "amplitude", "variational_block"

                            print(
                                f"\nRadius: {radius:.2f}, LR: {learning_rate}, LossOpt: {loss_option}, Scale: {scale}, Method: {method}, Ansatz: {ansatz_name}")

                            if ansatz_name == "complex_z":
                                num_params_needed = 2 * num_qubits_main
                            elif ansatz_name == "variational_block":
                                num_params_needed = 2 ** num_qubits_main - 1
                            elif ansatz_name == "amplitude":
                                num_params_needed = 2 ** num_qubits_main - 1
                            else:
                                raise ValueError(f"Unknown ansatz_name: {ansatz_name}")

                            initial_params = np.random.uniform(0, 2 * np.pi, size=num_params_needed)
                            initial_params = np.array(initial_params, requires_grad=(method in ["analytic", "adam"]))

                            start = time()
                            try:
                                quantum_optimization_simulation(num_qubits=num_qubits_main, ansatz_params=initial_params, optimizer=method,
                                                                ansatz_name=ansatz_name)
                            except Exception as e:
                                print(f"!!! ERROR during simulation: {e}")
                                import traceback

                                traceback.print_exc()

                            print(f"Tiempo de ejecución: {time() - start:.2f} segundos")
                            print("----------------------------------------")
