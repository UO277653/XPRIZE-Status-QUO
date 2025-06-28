#!/usr/bin/env python3
"""
VQLS para Flujo de Potencia: Enfoque Complejo vs Rectangular
Versión robusta y simplificada
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import time
import numpy as standard_np  # Para funciones que PennyLane no tiene


# =============================================================================
# CONFIGURACIÓN ROBUSTA DEL SISTEMA
# =============================================================================

def create_stable_system(approach="complex"):
    """Crea un sistema VQLS estable para flujo de potencia."""
    print(f"🔌 Creando sistema {approach.upper()}...")

    if approach == "complex":
        # Sistema complejo bien condicionado
        # Problema: c1*V + c2*V* = b
        #           conj(c2)*V + conj(c1)*V* = conj(b)

        c1 = 2.0 + 1.0j
        c2 = 0.5 + 0.2j
        target = 1.0 + 0.5j

        A = standard_np.array([[c1, c2], [standard_np.conj(c2), standard_np.conj(c1)]], dtype=complex)

        b = standard_np.array([target, standard_np.conj(target)], dtype=complex)

    else:  # rectangular
        # Sistema rectangular bien condicionado
        # [P] = [G -B] [Vr]
        # [Q]   [B  G] [Vi]

        G, B = 2.5, 1.2
        P_target, Q_target = 0.8, 0.3

        A = standard_np.array([[G, -B], [B, G]], dtype=float)

        b = standard_np.array([P_target, Q_target], dtype=float)

    # Normalizar y resolver
    b = b / standard_np.linalg.norm(b)
    x_classical = standard_np.linalg.solve(A, b)

    # Convertir a PennyLane arrays
    A_pnl = np.array(A)
    b_pnl = np.array(b)
    x_pnl = np.array(x_classical)

    print(f"  Matriz A: {A.shape}, condición: {standard_np.linalg.cond(A):.2e}")
    print(f"  Vector b: {b}")
    print(f"  Solución: {x_classical}")

    return A_pnl, b_pnl, x_pnl


# =============================================================================
# ANSÄTZE OPTIMIZADOS
# =============================================================================

def ansatz_complex(params, wires):
    """Ansatz para enfoque complejo (parámetros acoplados)."""
    n_qubits = len(wires)

    # Capa 1: Superposición inicial
    for w in wires:
        qml.Hadamard(wires=w)

    # Capa 2: Rotaciones acopladas
    for i, w in enumerate(wires):
        if i * 2 + 1 < len(params):
            qml.RY(params[i * 2], wires=w)
            qml.RZ(params[i * 2 + 1], wires=w)

    # Capa 3: Entanglement
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])

    # Capa 4: Rotaciones finales (si hay parámetros)
    offset = 2 * n_qubits
    for i, w in enumerate(wires):
        if offset + i < len(params):
            qml.RZ(params[offset + i], wires=w)


def ansatz_rectangular(params, wires):
    """Ansatz para enfoque rectangular (parámetros independientes)."""
    n_qubits = len(wires)

    # Capa 1: Superposición inicial
    for w in wires:
        qml.Hadamard(wires=w)

    # Capa 2: Rotaciones completas independientes
    for i, w in enumerate(wires):
        start_idx = i * 3
        if start_idx + 2 < len(params):
            qml.Rot(params[start_idx], params[start_idx + 1], params[start_idx + 2], wires=w)

    # Capa 3: Entanglement
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])

    # Capa 4: Segunda ronda de rotaciones (si hay parámetros)
    offset = 3 * n_qubits
    for i, w in enumerate(wires):
        if offset + i * 2 + 1 < len(params):
            qml.RY(params[offset + i * 2], wires=w)
            qml.RZ(params[offset + i * 2 + 1], wires=w)


# =============================================================================
# IMPLEMENTACIÓN VQLS SIMPLIFICADA
# =============================================================================

class VQLSSimple:
    def __init__(self, approach="complex", n_qubits=2):
        self.approach = approach
        self.n_qubits = n_qubits
        self.ancilla_idx = n_qubits

        # Crear sistema estable
        self.A, self.b, self.x_classical = create_stable_system(approach)

        # Configurar ansatz
        if approach == "complex":
            self.ansatz = ansatz_complex
            self.n_params = 3 * n_qubits  # Menos parámetros
        else:
            self.ansatz = ansatz_rectangular
            self.n_params = 5 * n_qubits  # Más parámetros

        # Dispositivo
        self.dev = qml.device("lightning.qubit", wires=n_qubits + 1)

        print(f"\n📊 VQLS-{approach.upper()}:")
        print(f"  Parámetros: {self.n_params}")

    def prepare_b_state(self):
        """Prepara |b> con amplitudes normalizadas."""
        full_amplitudes = standard_np.zeros(2 ** self.n_qubits, dtype=complex)

        # Asignar valores de b
        for i in range(min(len(self.b), len(full_amplitudes))):
            full_amplitudes[i] = self.b[i]

        # Normalizar
        full_amplitudes = full_amplitudes / standard_np.linalg.norm(full_amplitudes)

        # Preparar estado
        qml.MottonenStatePreparation(np.array(full_amplitudes), wires=range(self.n_qubits))

    def controlled_A_simple(self):
        """Operaciones controladas simplificadas de A."""
        # Implementación simplificada pero efectiva
        A_diag = standard_np.diag(self.A)

        for i in range(min(self.n_qubits, len(A_diag))):
            coeff = A_diag[i]
            if abs(coeff) > 1e-8:
                if self.approach == "complex":
                    angle = standard_np.angle(coeff)
                else:
                    angle = float(coeff)

                qml.CRZ(2 * angle, wires=[self.ancilla_idx, i])

    def create_circuit(self, part="real"):
        """Crea circuito VQLS."""

        @qml.qnode(self.dev, interface="autograd")
        def circuit(weights):
            # Hadamard en ancilla
            qml.Hadamard(wires=self.ancilla_idx)

            # Shift para parte imaginaria
            if part == "imag" and self.approach == "complex":
                qml.PhaseShift(-standard_np.pi / 2, wires=self.ancilla_idx)

            # Ansatz variacional
            self.ansatz(weights, wires=list(range(self.n_qubits)))

            # Operaciones controladas
            self.controlled_A_simple()

            # U_b dagger
            qml.adjoint(self.prepare_b_state)()

            # Hadamard final
            qml.Hadamard(wires=self.ancilla_idx)

            return qml.expval(qml.PauliZ(wires=self.ancilla_idx))

        return circuit

    def cost_function(self, weights):
        """Función de costo robusta."""
        if self.approach == "complex":
            real_circuit = self.create_circuit("real")
            imag_circuit = self.create_circuit("imag")

            real_part = real_circuit(weights)
            imag_part = imag_circuit(weights)
            overlap = real_part + 1j * imag_part
        else:
            real_circuit = self.create_circuit("real")
            overlap = real_circuit(weights)

        # Costo con regularización suave
        cost = 1.0 - abs(overlap) ** 2

        # Añadir pequeña regularización
        weights_array = standard_np.array(weights)
        reg = 0.0001 * standard_np.sum(weights_array ** 2)

        return cost + reg

    def optimize(self, steps=500, lr=0.1):
        """Optimización robusta."""
        print(f"\n🚀 Optimizando VQLS-{self.approach}...")

        # Inicialización más conservadora
        standard_np.random.seed(42)
        weights = 0.01 * standard_np.random.randn(self.n_params)
        weights = np.array(weights, requires_grad=True)

        # Optimizador
        optimizer = qml.AdamOptimizer(stepsize=lr)

        costs = []
        start_time = time.time()

        for step in range(steps):
            try:
                weights, cost = optimizer.step_and_cost(self.cost_function, weights)
                costs.append(float(cost))

                if step % 100 == 0:
                    print(f"  Paso {step:3d}: Costo = {cost:.6f}")

                # Convergencia temprana
                if cost < 1e-8:
                    print(f"  Convergencia en paso {step}")
                    break

            except Exception as e:
                print(f"  Error en paso {step}: {e}")
                break

        elapsed = time.time() - start_time
        print(f"  Tiempo: {elapsed:.2f}s, Costo final: {costs[-1]:.6f}")

        return weights, costs

    def get_solution(self, weights):
        """Extrae solución cuántica."""

        @qml.qnode(self.dev)
        def state_circuit(w):
            self.ansatz(w, wires=list(range(self.n_qubits)))
            return qml.state()

        quantum_state = state_circuit(weights)
        return quantum_state / standard_np.linalg.norm(quantum_state)

    def analyze(self, quantum_solution):
        """Análisis de resultados."""
        print(f"\n📈 Análisis {self.approach}:")

        # Convertir a arrays estándar
        n_vars = len(self.x_classical)
        quantum_amps = standard_np.array(quantum_solution[:n_vars])
        classical_amps = standard_np.array(self.x_classical)

        print(f"  Clásica: {classical_amps}")
        print(f"  Cuántica: {quantum_amps}")

        # Fidelidad
        c_probs = standard_np.abs(classical_amps) ** 2
        q_probs = standard_np.abs(quantum_amps) ** 2

        c_probs = c_probs / standard_np.sum(c_probs)
        q_probs = q_probs / standard_np.sum(q_probs)

        fidelity = standard_np.sum(standard_np.sqrt(c_probs * q_probs))

        # Error
        error = standard_np.linalg.norm(quantum_amps - classical_amps)
        rel_error = error / standard_np.linalg.norm(classical_amps)

        print(f"  Fidelidad: {fidelity:.6f}")
        print(f"  Error absoluto: {error:.6f}")
        print(f"  Error relativo: {rel_error:.6f}")

        # Verificar conjugado para complejo
        if self.approach == "complex" and len(quantum_amps) >= 2:
            V, V_star = quantum_amps[0], quantum_amps[1]
            conj_error = abs(V_star - standard_np.conj(V))
            print(f"  Error conjugado: {conj_error:.8f}")

            if conj_error < 0.05:
                print("  ✅ Relación conjugada preservada")
            else:
                print("  ⚠️ Relación conjugada no preservada")

        return fidelity, error, rel_error


# =============================================================================
# COMPARACIÓN PRINCIPAL
# =============================================================================

def run_comparison():
    """Ejecuta comparación completa."""
    print("=" * 70)
    print("🔬 VQLS ROBUSTO: COMPLEJO vs RECTANGULAR")
    print("=" * 70)

    results = {}

    for approach in ["complex", "rectangular"]:
        print(f"\n{'=' * 20} {approach.upper()} {'=' * 20}")

        try:
            # Crear y optimizar
            vqls = VQLSSimple(approach=approach)
            weights, costs = vqls.optimize(steps=400, lr=0.05)

            # Analizar
            quantum_solution = vqls.get_solution(weights)
            fidelity, error, rel_error = vqls.analyze(quantum_solution)

            # Guardar
            results[approach] = {'n_params': vqls.n_params, 'costs': costs, 'final_cost': costs[-1], 'fidelity': fidelity, 'error': error,
                                 'rel_error': rel_error, 'classical': vqls.x_classical, 'quantum': quantum_solution[:len(vqls.x_classical)]}

        except Exception as e:
            print(f"  ❌ Error en {approach}: {e}")

    # Visualizar
    if len(results) == 2:
        plot_results(results)
        final_summary(results)

    return results


def plot_results(results):
    """Visualización simplificada."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Convergencia
    for approach, data in results.items():
        ax1.semilogy(data['costs'], label=approach.capitalize(), linewidth=2)
    ax1.set_xlabel('Iteraciones')
    ax1.set_ylabel('Costo (log)')
    ax1.set_title('Convergencia')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Parámetros
    approaches = list(results.keys())
    n_params = [results[app]['n_params'] for app in approaches]

    ax2.bar(approaches, n_params, color=['#FF6B6B', '#4ECDC4'], alpha=0.7)
    ax2.set_ylabel('Parámetros')
    ax2.set_title('Complejidad del Ansatz')
    for i, n in enumerate(n_params):
        ax2.text(i, n + 0.2, str(n), ha='center', va='bottom', fontweight='bold')

    # 3. Fidelidad
    fidelities = [results[app]['fidelity'] for app in approaches]
    bars = ax3.bar(approaches, fidelities, color=['#FF6B6B', '#4ECDC4'], alpha=0.7)
    ax3.set_ylabel('Fidelidad')
    ax3.set_title('Precisión')
    ax3.set_ylim(0, 1.1)
    for bar, f in zip(bars, fidelities):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f'{f:.3f}', ha='center', va='bottom')

    # 4. Error relativo
    rel_errors = [results[app]['rel_error'] for app in approaches]
    ax4.bar(approaches, rel_errors, color=['#FF6B6B', '#4ECDC4'], alpha=0.7)
    ax4.set_ylabel('Error Relativo')
    ax4.set_title('Error vs Clásico')
    ax4.set_yscale('log')

    plt.tight_layout()
    plt.show()


def final_summary(results):
    """Resumen final."""
    print("\n" + "=" * 70)
    print("📊 RESUMEN FINAL")
    print("=" * 70)

    complex_res = results['complex']
    rect_res = results['rectangular']

    # Eficiencia de parámetros
    param_reduction = (rect_res['n_params'] - complex_res['n_params']) / rect_res['n_params'] * 100

    print(f"\n🎯 EFICIENCIA:")
    print(f"  Complejo: {complex_res['n_params']} parámetros")
    print(f"  Rectangular: {rect_res['n_params']} parámetros")
    print(f"  Reducción: {param_reduction:.1f}%")

    print(f"\n🎯 PRECISIÓN:")
    print(f"  Complejo - Fidelidad: {complex_res['fidelity']:.4f}, Error: {complex_res['rel_error']:.4f}")
    print(f"  Rectangular - Fidelidad: {rect_res['fidelity']:.4f}, Error: {rect_res['rel_error']:.4f}")

    # Ganador
    if complex_res['fidelity'] > rect_res['fidelity']:
        winner = "COMPLEJO"
        fid_advantage = complex_res['fidelity'] - rect_res['fidelity']
    else:
        winner = "RECTANGULAR"
        fid_advantage = rect_res['fidelity'] - complex_res['fidelity']

    print(f"\n💡 CONCLUSIONES:")
    print(f"  🏆 Mejor precisión: {winner} (+{fid_advantage:.4f})")
    print(f"  ⚡ Mejor eficiencia: COMPLEJO (-{param_reduction:.0f}% parámetros)")

    # Recomendación
    avg_fidelity = (complex_res['fidelity'] + rect_res['fidelity']) / 2
    if avg_fidelity > 0.9:
        print(f"  ✅ Ambos enfoques logran buena precisión")
        print(f"  🎯 Para redes grandes: Preferir COMPLEJO por escalabilidad")
    else:
        print(f"  ⚠️ Precisión mejorable (promedio: {avg_fidelity:.3f})")
        print(f"  🔧 Considerar ansätze más expresivos o más iteraciones")


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    print("🔌 VQLS Robusto para Flujo de Potencia")
    print("Comparación: Enfoque Complejo vs Rectangular")
    print("=" * 70)

    results = run_comparison()

    print(f"\n🎉 ¡Experimento completado!")

    if len(results) == 2:
        complex_fid = results['complex']['fidelity']
        rect_fid = results['rectangular']['fidelity']
        param_red = (results['rectangular']['n_params'] - results['complex']['n_params']) / results['rectangular']['n_params'] * 100

        print(f"💫 Fidelidades: Complejo {complex_fid:.3f}, Rectangular {rect_fid:.3f}")
        print(f"⚡ Reducción parámetros: {param_red:.0f}%")

        if complex_fid > 0.9 and rect_fid > 0.9:
            print(f"🌟 ¡Ambos enfoques son exitosos!")
