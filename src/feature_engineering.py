import numpy as np


def construir_sistema(X_norm: np.ndarray, y_norm: np.ndarray):
    # Ecuaciones normales: A = XᵀX, b = Xᵀy
    A = np.dot(X_norm.T, X_norm)   # (n, n)
    b = np.dot(X_norm.T, y_norm)   # (n,)
    return A, b


def calcular_numero_condicion(A: np.ndarray):
    # eigvalsh para matrices simétricas (más estable numéricamente)
    autovalores = np.linalg.eigvalsh(A)
    lam_min = float(autovalores.min())
    lam_max = float(autovalores.max())
    # Evitar división por cero en sistemas degenerados
    if abs(lam_min) < 1e-300:
        kappa = float('inf')
    else:
        kappa = abs(lam_max) / abs(lam_min)
    return kappa, lam_min, lam_max
