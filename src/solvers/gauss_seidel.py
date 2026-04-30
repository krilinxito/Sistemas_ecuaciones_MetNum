import numpy as np
import time


def solve_gauss_seidel(A: np.ndarray, b: np.ndarray,
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
            # Usa valores YA actualizados para j < i
            suma_nueva = np.dot(A[i, :i], x[:i])
            # Usa valores anteriores para j > i
            suma_vieja = np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (b[i] - suma_nueva - suma_vieja) / A[i, i]

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
    }
