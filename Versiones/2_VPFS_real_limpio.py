from qiskit import QuantumCircuit, QuantumRegister
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
import numpy as np
from scipy.linalg import sqrtm, polar
from scipy.optimize import minimize
import torch

Y = np.array([
    [1, -1, 0, 0],
    [-1, 2, -1, 0],
    [0, -1, 2, -1],
    [0, 0, -1, 1]
], dtype=complex) * 5
V = np.array([1, 1.1, 0.95, 0.9])

max_iters = 1  # Máximo número de iteraciones
tolerance = 1e-9  # Tolerancia para la convergencia# Crear Y y V de tamaño correspondiente
loss_th = 1e-12  # A partir de ese valor, el learning rate se adapta dinámicamente (si loss_th < tolerance no tiene efecto)


def ansatz(params):

    v = [1]  # Primera componente fija
    for i in range(len(params)):
        if radius > 0:
            v.append(np.sin(params[i]) * radius + 1)
        else:
            v.append(params[i] + 1)
    # Normalizar el vector v
    v = [x / np.linalg.norm(v) for x in v]
    return v


def create_circuits(Y, B, num_qubits):

    # Crear Y_extended
    Y_norm = np.max(np.abs(np.linalg.eigvals(Y)))
    Y_normalized = Y / Y_norm
    sqrt_diff = sqrtm(np.eye(len(Y)) - Y_normalized @ Y_normalized)
    Y_extended = np.block([
        [Y_normalized, sqrt_diff],
        [sqrt_diff, -Y_normalized]
    ])
    Y_extended, _ = polar(Y_extended)  # U es la parte unitaria
    Y_op = Operator(Y_extended)

    # Crear los registros cuánticos
    total_qubits = 2 * num_qubits + 1
    qreg = QuantumRegister(total_qubits, 'q')

    # Crear circuito base para qc1
    qc1 = QuantumCircuit(qreg, name="qc1")
    qc1.append(Y_op, list(range(num_qubits, 2 * num_qubits + 1)))
    for i in range(num_qubits, 2 * num_qubits):
        qc1.cx(i - num_qubits, i)

    # Construir U_b^\dagger basado en B
    b = B / np.linalg.norm(B)
    U_b = np.eye(len(b), dtype=complex)
    U_b[:, 0] = b
    for i in range(1, len(b) - 1):
        v = U_b[1:, i]
        for j in range(i):
            v -= np.dot(U_b[1:, j].conj(), v) * U_b[1:, j]
        v /= np.linalg.norm(v)
        U_b[1:, i] = v
    U_b[0, -1] = 1
    U_b[-1, -1] = 0
    U_b_dagger = U_b.conj().T
    U_b_op = Operator(U_b_dagger)

    # Circuito qc2 que extiende qc1 con U_b^\dagger
    qc2 = qc1.copy(name="qc2")
    qc2.append(U_b_op, list(range(num_qubits)))

    return qc1, qc2


def calculate_loss_with_simulation(params, qc1, qc2, simulator, num_qubits, B_norm, Y_norm):
    """
    Calcular la pérdida componiendo dinámicamente con inicialización.
    """
    dim = 2 ** num_qubits

    # Calcular el estado inicial v
    v = ansatz(params)

    # Obtener el valor de v[0] después de la normalización
    V_norm = 1 / v[0]  # Trabajamos en pu, luego v[0] = 1 y V_norm = 1

    # Crear circuito de inicialización
    qreg = QuantumRegister(2 * num_qubits + 1, 'q')
    init_circuit = QuantumCircuit(qreg, name="init")
    try:
        init_circuit.initialize(v, list(range(num_qubits)))  # Inicializar primer registro
        init_circuit.initialize(v, list(
            range(num_qubits, 2 * num_qubits)))  # Inicializar segundo registro
    except QiskitError as e:
        print(f"Error al inicializar el circuito: {e}")
        return None

    # Componer con qc1
    composed_qc1 = init_circuit.compose(qc1)
    composed_qc1.save_statevector(label="statevector_yv")  # Guardar estado vectorial

    # Simular el primer circuito
    result1 = simulator.run([composed_qc1]).result()
    statevector1 = np.array(
        result1.data(0)["statevector_yv"])  # Recuperar estado vectorial como array

    # Debug: Print statevector with binary labels for composed_qc1
    statevector1 = np.array(result1.data(0)["statevector_yv"])
    print("Qiskit Circuit1 Statevector:")
    total_qubits = 2 * num_qubits + 1
    for idx, amplitude in enumerate(statevector1):
        binary_label = format(idx, '0{}b'.format(total_qubits))
        print(f"Index {idx} (|{binary_label}⟩): {amplitude}")

    # Extraer coeficientes relevantes
    shots_array = np.abs(statevector1[1:dim]) ** 2
    shots_total = np.sum(shots_array)
    norm_yv_cnot = np.sqrt(shots_total)
    # Componer con qc2
    composed_qc2 = init_circuit.compose(qc2)
    composed_qc2.save_statevector(label="statevector_ub")  # Guardar estado vectorial

    # Simular el segundo circuito
    result2 = simulator.run([composed_qc2]).result()
    statevector2 = np.array(
        result2.data(0)["statevector_ub"])  # Recuperar estado vectorial como array

    # Debug: Print statevector with binary labels for composed_qc2
    statevector2 = np.array(result2.data(0)["statevector_ub"])
    print("Qiskit Circuit2 Statevector:")
    for idx, amplitude in enumerate(statevector2):
        binary_label = format(idx, '0{}b'.format(total_qubits))
        print(f"Index {idx} (|{binary_label}⟩): {amplitude}")

    # Calcular la pérdida
    shots_array = np.abs(statevector2[0]) ** 2
    shots_total = np.sum(shots_array)
    norm_after_ub = np.sqrt(shots_total)
    norm_YV_cnot = norm_yv_cnot * Y_norm * V_norm * V_norm
    pen_coef = PEN_COEF_SCALE / B_norm ** 2
    loss = [1 - (norm_after_ub) / norm_yv_cnot + pen_coef * (norm_YV_cnot - B_norm) ** 2]
    loss.append(
        1 - (norm_after_ub) ** 2 / norm_yv_cnot ** 2 + pen_coef * (norm_YV_cnot - B_norm) ** 2)
    loss.append(
        1 - (norm_after_ub) ** 2 / norm_yv_cnot ** 2 + pen_coef * (norm_YV_cnot - B_norm) ** 4)
    a2 = norm_YV_cnot ** 2
    b2 = B_norm ** 2
    ab = norm_after_ub * Y_norm * B_norm * V_norm * V_norm
    loss.append((a2 - ab) ** 2 + (b2 - ab) ** 2)
    loss.append(a2 + b2 - 2 * ab)
    return loss[loss_option]


def finite_difference_gradient(params, qc1, qc2, simulator, num_qubits, B_norm, Y_norm, delta=1e-4):
    """
    Calcular el gradiente usando diferencias finitas.
    """
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += delta
        params_minus[i] -= delta
        loss_plus = calculate_loss_with_simulation(params_plus, qc1, qc2, simulator, num_qubits,
                                                   B_norm, Y_norm)
        loss_minus = calculate_loss_with_simulation(params_minus, qc1, qc2, simulator, num_qubits,
                                                    B_norm, Y_norm)
        grad[i] = (loss_plus - loss_minus) / (2 * delta)
    return grad


def finite_difference_gradient_sequential(params, qc1, qc2, simulator, num_qubits, B_norm, Y_norm,
                                          delta=1e-4):
    """
    Calcular el gradiente y actualizar los parámetros de forma secuencial.
    """
    for i in range(len(params)):
        # Crear copias de los parámetros
        params_plus = params.copy()
        params_minus = params.copy()

        # Incrementar y decrementar el parámetro actual
        params_plus[i] += delta
        params_minus[i] -= delta

        # Calcular las pérdidas para los parámetros ajustados
        loss_plus = calculate_loss_with_simulation(params_plus, qc1, qc2, simulator, num_qubits,
                                                   B_norm, Y_norm)
        loss_minus = calculate_loss_with_simulation(params_minus, qc1, qc2, simulator, num_qubits,
                                                    B_norm, Y_norm)

        # Verificar si la pérdida es válida
        if loss_plus is None or loss_minus is None:
            print(
                f"Error en la evaluación de la pérdida para el parámetro {i}. Saltando actualización.")
            continue

        # Calcular el gradiente para el parámetro actual
        grad = (loss_plus - loss_minus) / (2 * delta)

        # Actualizar el parámetro actual
        params[i] -= learning_rate * grad

    return params


def quantum_optimization_simulation(num_qubits=2, ansatz_params=None, optimizer="basic"):
    """Simulación cuántica usando Qiskit AerSimulator con distintos optimizadores."""
    if ansatz_params is None:
        ansatz_params = [np.pi / 4] * (2 ** num_qubits - 1)  # Inicialización de parámetros

    B = V * (Y @ V)
    B[0] = 0
    B_norm = np.linalg.norm(B)

    # Crear los circuitos base una vez
    qc1, qc2 = create_circuits(Y, B, num_qubits)
    Y_norm = np.max(np.abs(np.linalg.eigvals(Y)))
    # Simulador
    simulator = AerSimulator()

    if optimizer == "basic":
        for iter in range(max_iters):
            loss = calculate_loss_with_simulation(ansatz_params, qc1, qc2, simulator, num_qubits,
                                                  B_norm, Y_norm)
            if loss == None:
                print(
                    "NO CONVERGIÓ!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                loss = - 100
                break
            elif loss < tolerance:
                print("Convergencia alcanzada usando basic.")
                break

            grad = finite_difference_gradient(ansatz_params, qc1, qc2, simulator, num_qubits,
                                              B_norm, Y_norm)
            ansatz_params -= learning_rate * grad
    elif optimizer == "sequential":
        for iter in range(max_iters):
            loss = calculate_loss_with_simulation(ansatz_params, qc1, qc2, simulator, num_qubits,
                                                  B_norm, Y_norm)
            if loss == None:
                print(
                    "NO CONVERGIÓ!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                loss = - 100
                break
            elif loss < tolerance:
                print("Convergencia alcanzada usando sequential.")
                break
            ansatz_params = finite_difference_gradient_sequential(ansatz_params, qc1, qc2,
                                                                  simulator, num_qubits, B_norm,
                                                                  Y_norm)
    elif optimizer == "cobyla":
        def loss_func(params):
            loss = calculate_loss_with_simulation(params, qc1, qc2, simulator, num_qubits, B_norm,
                                                  Y_norm)
            return loss

        result = minimize(loss_func, ansatz_params, tol=tolerance, method="COBYLA",
                          options={"maxiter": max_iters, "disp": True})
        ansatz_params = result.x
        print("Convergencia alcanzada usando COBYLA.\n")
        iter = result.nfev
        loss = result.fun


    elif optimizer == "adam":
        # Usar PyTorch para implementar Adam
        params = torch.tensor(ansatz_params, requires_grad=True, dtype=torch.float32)
        optimizer = torch.optim.Adam([params], lr=learning_rate)

        for iter in range(max_iters):
            optimizer.zero_grad()

            # Convertir params a numpy para calcular la pérdida
            loss = calculate_loss_with_simulation(
                params.detach().numpy(), qc1, qc2, simulator, num_qubits, B_norm, Y_norm
            )

            # Backpropagation en PyTorch
            loss_tensor = torch.tensor(loss, requires_grad=True)
            loss_tensor.backward()
            optimizer.step()

            if loss < tolerance:
                ansatz_params = params.detach().numpy()
                print("Convergencia alcanzada usando Adam.\n")
                break

    else:
        raise ValueError(f"Optimizer '{optimizer}' no reconocido. Use 'basic', 'cobyla' o 'adam'.")

    # Resultados finales
    v = ansatz(ansatz_params)
    v0 = v[0]
    Vsol = [x / v0 for x in v]
    print(f"Iter {iter + 1}: Loss x 1e6 = {loss * 1e6:.2f}, Params = {ansatz_params}")
    err_V = np.abs(V - Vsol)
    max_err_V = np.max(err_V)
    print(f"Error máximo en V: {max_err_V}, Vreal/ Vcalc: {V / Vsol}")


if __name__ == "__main__":
    from time import time

    for radius in [0.3]:
        for learning_rate in [0.1]:
            for loss_option in [0]:
                for scale in [0]:
                    if scale == 0:
                        loss_option = 4
                    PEN_COEF_SCALE = 0.01 * scale
                    for method in ["sequential"]:  # "basic",,"cobyla", "adam"
                        print(
                            f"\nRadio: {radius}, Learning rate: {learning_rate}, loss option: {loss_option}, scale: {scale} y metodo: {method}")
                        start = time()
                        quantum_optimization_simulation(num_qubits=2, ansatz_params=None,
                                                        optimizer=method)

# Optimo
# Radio: 0.3, Learning rate: 0.1, loss option: 4, scale: 0 y metodo: sequential
# Convergencia alcanzada usando sequential.
# Iter 279: Loss x 1e6 = 0.00, Params = [0.33988086703234277, -0.16735967645504335, -0.3397188350028957]
# Error máximo en V: 3.3397195060125284e-05, Vreal/ Vcalc: [1.         0.9999887  0.99997247 0.99996289]