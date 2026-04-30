import numpy as np
import scipy.linalg
import time

from src.feature_engineering import calcular_numero_condicion


def solve_lu(A: np.ndarray, b: np.ndarray) -> dict:
    # Factorización LU con pivoteo parcial (scipy permitido aquí)
    kappa, _, _ = calcular_numero_condicion(A)

    t0 = time.perf_counter()
    lu_piv = scipy.linalg.lu_factor(A)
    x = scipy.linalg.lu_solve(lu_piv, b)
    tiempo = time.perf_counter() - t0

    residual = float(np.linalg.norm(A @ x - b))

    return {
        'solucion': x,
        'residual_final': residual,
        'historial_residuales': [residual],
        'iteraciones': 'N/A (método directo)',
        'convergió': True,
        'tiempo': tiempo,
        'numero_condicion': kappa,
    }
