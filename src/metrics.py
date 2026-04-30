import numpy as np


def r_cuadrado(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - y_real.mean()) ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1 - ss_res / ss_tot)


def rmse(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_real - y_pred) ** 2)))


def analizar_convergencia(historial: list, tol: float) -> dict:
    if not historial:
        return {'iteraciones': 0, 'tasa_media': None, 'convergió': False}

    convergió = historial[-1] < tol
    iters = next((i + 1 for i, r in enumerate(historial) if r < tol), len(historial))

    # Tasa de convergencia: proporción media r_{k+1}/r_k
    ratios = []
    for i in range(1, len(historial)):
        if historial[i - 1] > 0:
            ratios.append(historial[i] / historial[i - 1])

    tasa_media = float(np.mean(ratios)) if ratios else None

    return {
        'iteraciones': iters,
        'tasa_media': tasa_media,
        'convergió': convergió,
    }
