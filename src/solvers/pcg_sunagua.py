import numpy as np
import time


def aplicar_precondicionador_inverso(r: np.ndarray, diag_A: np.ndarray) -> np.ndarray:
    # Resuelve Mz = r donde M = CCᵀ = diag(A)^{-1}
    # → z = M⁻¹ r = diag(A) * r  (multiplicación elemento a elemento)
    # M nunca se construye como matriz explícita.
    return diag_A * r


def solve_pcg_sunagua(A: np.ndarray, b: np.ndarray,
                      tol: float = 1e-6, max_iter: int = 10000) -> dict:
    """
    Gradiente Conjugado Precondicionado — Algoritmo 2 (Mz) de Suñagua (2020).

    Diferencia clave vs GCP estándar: β usa rᵀz (residuo precondicionado)
    en lugar de rᵀr. Esto da mayor estabilidad numérica y obtiene x
    directamente sin resolver un sistema adicional.

    Precondicionador diagonal: C = diag(A)^{-1/2}
    M = CCᵀ = diag(A)^{-1}  → resolver Mz=r es z = diag(A)*r
    """
    from src.feature_engineering import calcular_numero_condicion

    n = len(b)
    diag_A = np.diag(A)                       # vector diagonal de A

    kappa_antes, _, _ = calcular_numero_condicion(A)

    x = np.zeros(n)
    r = b - A @ x                             # r₀ = b − Ax₀
    z = aplicar_precondicionador_inverso(r, diag_A)   # z₀ = M⁻¹r₀

    p = None
    # Variables para rotación del historial de r y z
    r_km2, z_km2 = None, None   # r_{k-2}, z_{k-2}
    r_km1, z_km1 = r.copy(), z.copy()   # r_{k-1}, z_{k-1} (empiezan en r₀, z₀)

    historial = [float(np.linalg.norm(r))]

    t0 = time.perf_counter()
    convergió = False
    k = 0

    for k in range(1, max_iter + 1):
        if k == 1:
            p = z_km1.copy()                  # p₁ = z₀
        else:
            rz_km1 = np.dot(r_km1, z_km1)    # r_{k-1}ᵀ z_{k-1}
            rz_km2 = np.dot(r_km2, z_km2)    # r_{k-2}ᵀ z_{k-2}
            # β usa residuos precondicionados z, no r (diferencia clave del Algoritmo Mz)
            beta = rz_km1 / rz_km2
            p = z_km1 + beta * p              # actualizar dirección

        Ap = A @ p                            # A aplicado a la dirección actual
        rz = np.dot(r_km1, z_km1)            # r_{k-1}ᵀ z_{k-1}
        alpha = rz / np.dot(p, Ap)           # tamaño de paso

        x = x + alpha * p                    # actualizar solución
        r_new = r_km1 - alpha * Ap           # actualizar residuo
        z_new = aplicar_precondicionador_inverso(r_new, diag_A)  # precondicionar

        residual = float(np.linalg.norm(r_new))
        historial.append(residual)

        # Rotar variables para la siguiente iteración
        r_km2, z_km2 = r_km1, z_km1
        r_km1, z_km1 = r_new, z_new

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
        'numero_condicion_antes': kappa_antes,
        'referencia': 'Suñagua (2020), Algoritmo 2 (Mz)',
    }
