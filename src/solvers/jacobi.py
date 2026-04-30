import numpy as np
import time


def solve_jacobi(A: np.ndarray, b: np.ndarray,
                 tol: float = 1e-6, max_iter: int = 10000) -> dict:
    n = len(b)
    diag = np.diag(A)

    # Advertencia de dominancia diagonal
    dominancia_ok = all(
        abs(diag[i]) > sum(abs(A[i, j]) for j in range(n) if j != i)
        for i in range(n)
    )

    D_inv = 1.0 / diag
    R = A - np.diag(diag)   # A sin la diagonal

    x = np.zeros(n)
    historial = []

    t0 = time.perf_counter()
    convergió = False
    k = 0

    for k in range(1, max_iter + 1):
        x_nuevo = D_inv * (b - R @ x)   # actualización simultánea
        residual = float(np.linalg.norm(A @ x_nuevo - b))
        historial.append(residual)

        x = x_nuevo
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
        'dominancia_diagonal': dominancia_ok,
    }
