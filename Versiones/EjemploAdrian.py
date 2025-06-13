import pandapower as pp
import pandapower.networks as pn
from pandapower.pypower.makeYbus import makeYbus

import numpy as np
from scipy.sparse import csr_matrix

def newton_raphson_v3(Y_bus, P, Q, V_slack, slack_idx, tol=1e-8, max_iter=100, polar=True):
    """Método Newton-Raphson en coordenadas polares o rectangulares."""
    n = len(Y_bus)
    V = np.ones(n, dtype=complex)
    V[slack_idx] = V_slack  # Nodo slack fijado
    # dV_vec = []
    # dV_vec_cplx = []
    # ex_times = 0
    # ex_times_cplx = 0
    for iter_count in range(max_iter):
        # Cálculo de potencias nodales
        S_calc = V * np.conj(Y_bus @ V)
        P[0] = np.real(S_calc[0])
        Q[0] = np.imag(S_calc[0])
        dS = P + 1j * Q - S_calc
        if np.max(np.abs(dS)) < tol:
            # print(f"Tiempos de calculo: normal {ex_times*1000}")
            # print(f"Tiempos de calculo: complejo {ex_times_cplx*1000}")
            return V, iter_count

        # Calcular dP y dQ
        dP = dS.real  # Parte real (potencias activas)
        dQ = dS.imag  # Parte imaginaria (potencias reactivas)

         # Derivada de S respecto a V
        dSdV = np.diag(np.conj(Y_bus @ V))

        # Derivada de S respecto a V*
        dSdV_conj = np.conj(Y_bus) * V[:, np.newaxis]
        
        if polar:
            # Cálculo de dV/d|V|
            dVdMag = np.diag(np.exp(1j * np.angle(V)))
            dVdtheta = np.diag(1j*V)

            # Cálculo de dS/d|V|
            dSdMag = dSdV @ dVdMag + dSdV_conj @ np.conj(dVdMag)
            dSdtheta = dSdV @ dVdtheta + dSdV_conj @ np.conj(dVdtheta)

            # Separar en partes real e imaginaria
            dPdMag = dSdMag.real  # Derivadas de P respecto a |V|
            dQdMag = dSdMag.imag  # Derivadas de Q respecto a |V|
            dPdtheta = dSdtheta.real  # Derivadas de P respecto a θ
            dQdtheta = dSdtheta.imag  # Derivadas de Q respecto a θ
            # Construir la Jacobiana
            J = np.block([
                [dPdtheta[1:, 1:], dPdMag[1:, 1:]],
                [dQdtheta[1:, 1:], dQdMag[1:, 1:]]
            ])

            # Construir el vector columna [dP; dQ]
            dS_vec = np.hstack([dP[1:], dQ[1:]])  # Excluir el nodo slack
            # Resolver el sistema lineal
            dX = np.linalg.solve(J, dS_vec)
            # Separar dtheta y d|V|
            dtheta = dX[:n-1]  # Correcciones para los ángulos (sin slack)
            dV_mag = dX[n-1:]  # Correcciones para las magnitudes (sin slack)
            theta = np.angle(V)
            V_magnitude = np.abs(V)
            # Actualizar ángulos y magnitudes
            theta[1:] += dtheta  # Excluir el slack
            V_magnitude[1:] += dV_mag  # Excluir el slack

            # Reconstruir las tensiones en coordenadas polares
            V = V_magnitude * np.exp(1j * theta)
            # if iter_count < 2:
            #     print(f"{V[1:][-1]:.4f}")
        else:
            #Cálculo de dV/d|V|
            # dVdVre = 1
            # dVdVim = 1j

            # # Cálculo de dS/d|V| (se podría simplificar dado que es constante)
            # dSdVre = dSdV*dVdVre  + dSdV_conj*np.conj(dVdVre) 
            # dSdVim = dSdV*dVdVim  + dSdV_conj*np.conj(dVdVim)
            
            dSdVre = dSdV + dSdV_conj 
            dSdVim = 1j*(dSdV - dSdV_conj)

            # Separar en partes real e imaginaria
            dPdVre = dSdVre.real  # Derivadas de P respecto a |V|
            dQdVre = dSdVre.imag  # Derivadas de Q respecto a |V|
            dPdVim = dSdVim.real  # Derivadas de P respecto a θ
            dQdVim = dSdVim.imag  # Derivadas de Q respecto a θ
            # Construir la Jacobiana
            J = np.block([
                [dPdVim[1:, 1:], dPdVre[1:, 1:]],
                [dQdVim[1:, 1:], dQdVre[1:, 1:]]
            ])

            # Construir el vector columna [dP; dQ]
            dS_vec = np.hstack([dP[1:], dQ[1:]])  # Excluir el nodo slack
            # Resolver el sistema lineal
            dX = np.linalg.solve(J, dS_vec)
            # Separar dtheta y d|V|
            dVim = dX[:n-1]  # Correcciones para los ángulos (sin slack)
            dVre = dX[n-1:]  # Correcciones para las magnitudes (sin slack)
            Vim = np.imag(V)
            Vre = np.real(V)
            # Actualizar ángulos y magnitudes
            Vim[1:] += dVim  # Excluir el slack
            Vre[1:] += dVre  # Excluir el slack
            
            # Reconstruir las tensiones en coordenadas polares
            V = Vre + 1j*Vim
            # if iter_count < 2:
            #     print(f"{V[1:][-1]:.4f}")
    raise ValueError("Newton-Raphson no convergió.")
def newton_raphson_wirt(Y_bus, P, Q, V_slack, slack_idx, tol=1e-8, max_iter=100):
    """Método Newton-Raphson en coordenadas polares o rectangulares."""
    n = len(Y_bus)
    V = np.ones(n, dtype=complex)
    V[slack_idx] = V_slack  # Nodo slack fijado
    # dV_vec = []
    # dV_vec_cplx = []
    # ex_times = 0
    # ex_times_cplx = 0
    for iter_count in range(max_iter):
        # Cálculo de potencias nodales
        S_calc = V * np.conj(Y_bus @ V)
        P[0] = np.real(S_calc[0])
        Q[0] = np.imag(S_calc[0])
        dS = P + 1j * Q - S_calc
        if np.max(np.abs(dS)) < tol:
            # print(f"Tiempos de calculo: normal {ex_times*1000}")
            # print(f"Tiempos de calculo: complejo {ex_times_cplx*1000}")
            return V, iter_count

        # Calcular dP y dQ
        dP = dS.real  # Parte real (potencias activas)
        dQ = dS.imag  # Parte imaginaria (potencias reactivas)

         # Derivada de S respecto a V
        dSdV = np.diag(np.conj(Y_bus @ V))

        # Derivada de S respecto a V*
        dSdV_conj = np.conj(Y_bus) * V[:, np.newaxis]
        
        dS_vec = np.hstack([dS[1:], np.conj(dS[1:])])  # Excluir el nodo slack
        
        # Construir la Jacobiana
        J = np.block([
            [dSdV[1:, 1:], dSdV_conj[1:, 1:]],
            [np.conj(dSdV_conj[1:, 1:]), np.conj(dSdV[1:, 1:])]
        ])
        
        dX = np.linalg.solve(J, dS_vec)
        # Separar dtheta y d|V|
        dV = dX[:n-1]  # Correcciones  (sin slack)

        
        # Reconstruir las tensiones en coordenadas polares
        V[1:] += dV
        # if iter_count < 2:
        #     print(f"{V[1:][-1]:.4f}")
    raise ValueError("Newton-Raphson no convergió.")
# Ejecución y medición de tiempo
methods = {
    "Newton-Raphson (polar) v3": lambda: newton_raphson_v3(Y_bus, -P_load, -Q_load, V_slack, slack_idx, polar=True,max_iter=100),
    "Newton-Raphson (rect.) v3": lambda: newton_raphson_v3(Y_bus, -P_load, -Q_load, V_slack, slack_idx, polar=False,max_iter=100),
    "Newton-Raphson (Wirtinger) v3": lambda: newton_raphson_wirt(Y_bus, -P_load, -Q_load, V_slack, slack_idx, max_iter=100)
}   

import time

kVbase = 1.0
slack_idx = 0  # Índice del nodo slack
V_slack = 1.0  # Tensión del nodo slack
Y_bus = np.array([
    [1, -1, 0, 0],
    [-1, 2, -1, 0],
    [0, -1, 2, -1],
    [0, 0, -1, 1]
], dtype=complex)*10
nred = len(Y_bus)
print(Y_bus)
Ploads = []
Qloads = []
for _ in range(1):
    if _%100 == 0:
        print(_)
    # Crear vectores de prueba para P y Q
    P_load = np.random.uniform(0.05, 0.15, size=nred)  # Potencias activas (MW)
    Q_load = np.random.uniform(0, 0.1, size=nred)  # Potencias reactivas (MVAR) 
    Ploads.append(P_load)
    Qloads.append(Q_load)
    for name, method in methods.items():
        start_time = time.time()
        try:
            V_result, iterations = method()
            elapsed_time = time.time() - start_time
            if _ == 0:
                print(f"método: {name}, iteraciones: {iterations}")
                print(f"  Tensiones: {np.abs(V_result[-1]):.6f}, {np.rad2deg(np.angle(V_result[-1])):.6f}")
        except ValueError as e:
            print(f"\n{name} no convergió: {str(e)}")
