import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pennylane as qml
import numpy as np
from scipy.linalg import sqrtm, polar

from scipy.optimize import minimize
import torch

# Crear Y y V de tamaño correspondiente
Y = np.array([
    [1, -1, 0, 0],
    [-1, 2, -1, 0],
    [0, -1, 2, -1],
    [0, 0, -1, 1]
], dtype=complex) * 5
V = np.array([1, 1.1, 0.95, 0.9])

max_iters = 1000      # Máximo número de iteraciones
tolerance = 1e-9      # Tolerancia para la convergencia

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


def variational_block(weights,n_qubits):
    """
    Ansatz con entrelazamiento y exactamente 2^n - 1 parámetros.

    Args:
        weights (array-like): Parámetros del ansatz. Debe tener longitud 2^n - 1.
    """
    num_params = len(weights)
    assert num_params == 2**n_qubits - 1, f"Se esperaban {2**n_qubits - 1} parámetros, pero se recibieron {num_params}."

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


def create_unitaries(Y, B, num_qubits):
    r"""Crea las matrices unitarias Y_extended y U_b† (calculadas a partir de B).

    Args:
        Y (numpy.ndarray): Matriz Y original.
        B (numpy.ndarray): Vector B.
        num_qubits (int): Número de qubits para cada registro.

    Returns:
        tuple: (Y_extended, U_b_dagger, Y_norm) donde Y_extended es la unidad extendida y U_b_dagger es el operador U_b†.
    """
    dim = 2 ** num_qubits
    # Normalización y cálculo de Y_extended
    Y_norm = np.max(np.abs(np.linalg.eigvals(Y)))
    Y_normalized = Y / Y_norm
    sqrt_diff = sqrtm(np.eye(len(Y)) - Y_normalized @ Y_normalized)
    Y_extended = np.block([
        [Y_normalized, sqrt_diff],
        [sqrt_diff, -Y_normalized]
    ])
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

def quantum_optimization_simulation(num_qubits=2, ansatz_params=None, optimizer="basic"):
    r"""Simulación cuántica usando PennyLane con distintos optimizadores.

    Args:
        num_qubits (int, opcional): Número de qubits para cada uno de los dos registros.
        ansatz_params (list, opcional): Parámetros iniciales para el ansatz.
        optimizer (str, opcional): Método de optimización a usar ('basic', 'sequential', 'cobyla' o 'adam').
    """
    global learning_rate, loss_option

    if ansatz_params is None:
        ansatz_params = [np.pi / 4] * (2 ** num_qubits - 1)  # Inicialización de parámetros

    # Calcular B y su norma
    B = V * (Y @ V)
    B[0] = 0
    B_norm = np.linalg.norm(B)  # Precalcular <B|B>

    # Obtener las matrices unitarias a usar
    Y_extended, U_b_dagger, Y_norm = create_unitaries(Y, B, num_qubits)

    # Definir el dispositivo de PennyLane. Total de wires: 2*num_qubits + 1
    total_wires = 2 * num_qubits + 1
    dev = qml.device("default.qubit", wires=total_wires)

    # Circuito qc1: Inicializa dos registros y aplica Y_extended y las compuertas CNOT.
    @qml.qnode(dev)
    def circuit1(params, option_ansatz="amplitude"):
        """
        QNode que inicializa el estado cuántico de dos maneras posibles:
        1) Amplitude embedding, si option_ansatz == 'amplitude'
        2) Bloques variacionales, si option_ansatz == 'variational_block'
        """
        if option_ansatz == "amplitude":
            # 1) Usar el ansatz vectorial
            v = ansatz(params)
                # Por ejemplo, si quieres inicializar 'num_qubits' wires
            # Inicializar primer registro con el vector de estado usando AmplitudeEmbedding
            qml.AmplitudeEmbedding(v, wires=range(1,num_qubits+1), normalize=False)
            # Inicializar segundo registro con el mismo vector
            qml.AmplitudeEmbedding(v, wires=range(num_qubits+1, 2 * num_qubits+1), normalize=False)

        elif option_ansatz == "variational_block":
            # 2) Usar el ansatz de compuertas
            # Asumimos que 'params' es un array del tamaño 'n_qubits'
            variational_block(params, num_qubits)

        # El wire extra (índice 2*num_qubits) se deja en el estado |0>
        # Aplicar la operación Y_extended en wires [num_qubits, ..., 2*num_qubits]
        qml.QubitUnitary(Y_extended, wires=range(num_qubits+1))
        # Aplicar compuertas CNOT: control en wire i y target en wire i+num_qubits
        for i in range(1,num_qubits+1):
            qml.CNOT(wires=[i + num_qubits, i])
        return qml.state()

    # Circuito qc2: Igual que qc1 pero extiende con U_b† en el primer registro.
    @qml.qnode(dev)
    def circuit2(params, option_ansatz="amplitude"):
        
        if option_ansatz == "amplitude":
            # 1) Usar el ansatz vectorial
            v = ansatz(params)
                # Por ejemplo, si quieres inicializar 'num_qubits' wires
            # Inicializar primer registro con el vector de estado usando AmplitudeEmbedding
            qml.AmplitudeEmbedding(v, wires=range(1,num_qubits+1), normalize=False)
            # Inicializar segundo registro con el mismo vector
            qml.AmplitudeEmbedding(v, wires=range(num_qubits+1, 2 * num_qubits+1), normalize=False)

        elif option_ansatz == "variational_block":
            # 2) Usar el ansatz de compuertas
            # Asumimos que 'params' es un array del tamaño 'n_qubits'
            variational_block(params, num_qubits)
        # El wire extra (índice 2*num_qubits) se deja en el estado |0>
        # Aplicar la operación Y_extended en wires [num_qubits, ..., 2*num_qubits]
        qml.QubitUnitary(Y_extended, wires=range(num_qubits+1))
        # Aplicar compuertas CNOT: control en wire i y target en wire i+num_qubits
        for i in range(1,num_qubits+1):
            qml.CNOT(wires=[i + num_qubits, i])
        # Aplicar U_b† en el primer registro
        qml.QubitUnitary(U_b_dagger, wires=range(num_qubits+1, 2 * num_qubits+1))
        return qml.state()

    def calculate_loss_with_simulation(params):
        dim = 2 ** num_qubits
        v = ansatz(params)
        V_norm = 1 / v[0]  # Se asume que v[0] ≠ 0 (normalización en pu)
        
        statevector1 = circuit1(params,"amplitude")
        # print(statevector1)
        # Extraer coeficientes relevantes (se toma statevector1[1:dim])
        shots_array = np.abs(statevector1[1:dim]) ** 2
        shots_total = np.sum(shots_array)
        norm_yv_cnot = np.sqrt(shots_total)
        
        statevector2 = circuit2(params,"amplitude")
        # print(statevector2)
        # import sys
        # sys.exit()

        # Extraer el coeficiente en la posición 0
        shots_array2 = np.abs(statevector2[0]) ** 2
        shots_total2 = np.sum(shots_array2)
        norm_after_ub = np.sqrt(shots_total2)
        
        norm_YV_cnot = norm_yv_cnot * Y_norm * V_norm * V_norm
        pen_coef = PEN_COEF_SCALE / B_norm**2
        
        losses = []
        losses.append(1 - (norm_after_ub) / norm_yv_cnot + pen_coef * (norm_YV_cnot - B_norm)**2)
        losses.append(1 - (norm_after_ub)**2 / norm_yv_cnot**2 + pen_coef * (norm_YV_cnot - B_norm)**2)
        losses.append(1 - (norm_after_ub)**2 / norm_yv_cnot**2 + pen_coef * (norm_YV_cnot - B_norm)**4)
        a2 = norm_YV_cnot**2
        b2 = B_norm**2
        ab = norm_after_ub * Y_norm * B_norm * V_norm * V_norm
        losses.append((a2 - ab)**2 + (b2 - ab)**2)
        losses.append(a2 + b2 - 2 * ab)
        return losses[loss_option]

    def finite_difference_gradient(params, delta=1e-4):
        grad = np.zeros_like(params)
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
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += delta
            params_minus[i] -= delta
            loss_plus = calculate_loss_with_simulation(params_plus)
            loss_minus = calculate_loss_with_simulation(params_minus)
            if loss_plus is None or loss_minus is None:
                print(f"Error en la evaluación de la pérdida para el parámetro {i}. Saltando actualización.")
                continue
            grad = (loss_plus - loss_minus) / (2 * delta)
            params[i] -= learning_rate * grad
        return params
        # Definir la función de costo (loss) que usa los circuitos

    # Obtener la función gradiente analítica con PennyLane
    grad_fn = qml.grad(calculate_loss_with_simulation(ansatz_params))

    if optimizer == "analytic":
        for iter in range(max_iters):
            loss = calculate_loss_with_simulation(ansatz_params)
            if loss is None:
                print("NO CONVERGIÓ!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                loss = -100
                break
            elif loss < tolerance:
                print("Convergencia alcanzada usando gradiente analítico.")
                break

            # Calcular el gradiente analítico
            grad = grad_fn(ansatz_params)
            grad = grad_fn(ansatz_params)
            print("Grad:", grad)
            print("Type of grad:", type(grad))

            # Si es un tuple y quieres ver el tipo de cada elemento:
            for i, g in enumerate(grad):
                print(f"grad[{i}] =", g, "| type:", type(g))

            ansatz_params = ansatz_params - learning_rate * grad

    # Selección del optimizador
    elif optimizer == "basic":
        for iter in range(max_iters):
            loss = calculate_loss_with_simulation(ansatz_params)
            if loss is None:
                print("NO CONVERGIÓ!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                loss = -100
                break
            elif loss < tolerance:
                print("Convergencia alcanzada usando basic.")
                break
            grad = finite_difference_gradient(ansatz_params)
            ansatz_params = ansatz_params - learning_rate * grad

    elif optimizer == "sequential":
        for iter in range(max_iters):
            loss = calculate_loss_with_simulation(ansatz_params)
            if loss is None:
                print("NO CONVERGIÓ!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                loss = -100
                break
            elif loss < tolerance:
                print("Convergencia alcanzada usando sequential.")
                break
            ansatz_params = finite_difference_gradient_sequential(ansatz_params)

    elif optimizer == "cobyla":
        def loss_func(params):
            return calculate_loss_with_simulation(params)
        result = minimize(loss_func, ansatz_params, tol=tolerance, method="COBYLA", options={"maxiter": max_iters, "disp": True})
        ansatz_params = result.x
        print("Convergencia alcanzada usando COBYLA.\n")
        iter = result.nfev
        loss = result.fun

    elif optimizer == "adam":
        params_tensor = torch.tensor(ansatz_params, requires_grad=True, dtype=torch.float32)
        optim = torch.optim.Adam([params_tensor], lr=learning_rate)
        for iter in range(max_iters):
            optim.zero_grad()
            loss_val = calculate_loss_with_simulation(params_tensor.detach().numpy())
            loss_tensor = torch.tensor(loss_val, requires_grad=True)
            loss_tensor.backward()
            optim.step()
            if loss_val < tolerance:
                ansatz_params = params_tensor.detach().numpy()
                print("Convergencia alcanzada usando Adam.\n")
                break
    else:
        raise ValueError(f"Optimizer '{optimizer}' no reconocido. Use 'basic', 'cobyla', 'sequential' o 'adam'.")

    # Resultados finales
    v = ansatz(ansatz_params)
    v0 = v[0]
    Vsol = [x / v0 for x in v]
    print(f"Iter {iter + 1}: Loss x 1e6 = {loss*1e6:.2f}, Params = {ansatz_params}")
    Bcalc = np.array(Vsol) * (Y @ np.array(Vsol))
    err_V = np.abs(V - np.array(Vsol))
    max_err_V = np.max(err_V)
    print(f"Error máximo en V: {max_err_V}, Vreal/ Vcalc: {V/np.array(Vsol)}")


if __name__ == "__main__":
    from time import time
    for radius in [0.3]:
        for learning_rate in [0.1]:
            for loss_option in [0]:
                for scale in [0]:
                    if scale == 0:
                        loss_option = 4
                    PEN_COEF_SCALE = 0.01 * scale
                    for method in ["sequential"]:  # Se puede probar también "basic", "cobyla" o "adam"
                        print(f"\nRadio: {radius}, Learning rate: {learning_rate}, loss option: {loss_option}, scale: {scale} y método: {method}")
                        start = time()
                        print(f"Optimizando con {method}")
                        quantum_optimization_simulation(num_qubits=2, ansatz_params=None, optimizer=method)
                        print(f"Tiempo de ejecución: {time() - start} segundos")   

