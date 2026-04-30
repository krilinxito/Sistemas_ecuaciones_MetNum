import numpy as np
import time


def solve_sor(A: np.ndarray, b: np.ndarray,
              omega: float = 1.25,
              tol: float = 1e-6, max_iter: int = 10000) -> dict:
    n = len(b)
    x = np.zeros(n)
    historial = []

    t0 = time.perf_counter()
    convergió = False
    k = 0

    for k in range(1, max_iter + 1):
        x_old = x.copy()

        for i in range(n):
            suma_nueva = np.dot(A[i, :i], x[:i])
            suma_vieja = np.dot(A[i, i + 1:], x_old[i + 1:])
            x_gs = (b[i] - suma_nueva - suma_vieja) / A[i, i]
            # Relajación: mezcla entre valor anterior y resultado GS
            x[i] = (1 - omega) * x_old[i] + omega * x_gs

        residual = float(np.linalg.norm(A @ x - b))
        historial.append(residual)

        if residual < tol:
            convergió = True
            break

    tiempo = time.perf_counter() - t0

    return {
        'solucion': x,
        'residual_final': historial[-1] if historial else float('inf'),
        'historial_residuales': historial,
        'iteraciones': k,
        'convergió': convergió,
        'tiempo': tiempo,
        'omega': omega,
    }


def buscar_omega_optimo(A: np.ndarray, b: np.ndarray,
                        omega_range: tuple = (1.0, 1.99),
                        n_puntos: int = 20) -> dict:
    omegas = np.linspace(omega_range[0], omega_range[1], n_puntos)
    iters_por_omega = {}
    mejor_omega = omegas[0]
    mejor_iters = 10000 + 1

    for w in omegas:
        res = solve_sor(A, b, omega=float(w), tol=1e-6, max_iter=500)
        iters = res['iteraciones'] if res['convergió'] else 500
        iters_por_omega[round(float(w), 4)] = iters
        if iters < mejor_iters:
            mejor_iters = iters
            mejor_omega = float(w)

    return {
        'omega_optimo': mejor_omega,
        'iteraciones_por_omega': iters_por_omega,
        'iters_optimo': mejor_iters,
    }
