import numpy as np
from src.feature_engineering import construir_sistema, calcular_numero_condicion
from src.data_loader import normalizar

# Con 3 variables: gdp(0), energy_per_capita(1), coal_co2_per_capita(2)
# Solo el carbón (índice 2) es combustible fósil directo
IDX_FOSIL = 2


def escenario_ideal(X_norm: np.ndarray, y_norm: np.ndarray):
    A, b = construir_sistema(X_norm, y_norm)
    kappa, lam_min, lam_max = calcular_numero_condicion(A)

    ridge_aplicado = False
    lam_ridge = 0.0
    if kappa > 500:
        lam_ridge = 1e-3
        A = A + lam_ridge * np.eye(3)
        kappa, lam_min, lam_max = calcular_numero_condicion(A)
        ridge_aplicado = True

    meta = {
        'nombre': 'Escenario Ideal',
        'descripcion': (
            'Dataset completo de 89 países con variables normalizadas. '
            'El sistema 3×3 AᵀA está razonablemente condicionado. '
            'Sirve de referencia para comparar los demás escenarios.'
        ),
        'n_paises': X_norm.shape[0],
        'n_variables': 3,
        'kappa': kappa,
        'lam_min': lam_min,
        'lam_max': lam_max,
        'ridge_aplicado': ridge_aplicado,
        'lam_ridge': lam_ridge,
    }
    return A, b, meta


def escenario_estres(X_norm: np.ndarray, y_norm: np.ndarray,
                     X_original: np.ndarray, y_original: np.ndarray,
                     params: dict):
    # Filtrar países en percentil > 90 de co2_per_capita (emisores extremos)
    umbral = np.percentile(y_original, 90)
    mascara = y_original > umbral

    X_sub = X_original[mascara].copy()
    y_sub = y_original[mascara].copy()

    # Escalar la columna de carbón (índice 2) × 50
    # Simula demanda energética extrema en combustibles sucios
    X_sub[:, IDX_FOSIL] *= 50

    X_stress, y_stress, _ = normalizar(X_sub, y_sub)
    A, b = construir_sistema(X_stress, y_stress)
    kappa, lam_min, lam_max = calcular_numero_condicion(A)

    meta = {
        'nombre': 'Escenario Bajo Estres',
        'descripcion': (
            'Países con CO2 > percentil 90 (emisores extremos). '
            'La columna de carbón se escala ×50, simulando demanda energética '
            'extrema. Esto dispara la magnitud de los elementos de AᵀA y aumenta κ.'
        ),
        'n_paises': int(mascara.sum()),
        'n_variables': 3,
        'kappa': kappa,
        'lam_min': lam_min,
        'lam_max': lam_max,
        'umbral_co2': float(umbral),
        'factor_escala': 50,
    }
    return A, b, meta


def escenario_mal_condicionado(X_norm: np.ndarray, y_norm: np.ndarray):
    np.random.seed(42)
    m = X_norm.shape[0]

    # Reemplazar x3 (carbón) con casi-copia de x2 (energía)
    # Simula tener dos instrumentos de medición casi idénticos
    ruido = np.random.normal(0, 1e-4, m)
    X_mc = X_norm.copy()
    X_mc[:, 2] = X_norm[:, 1] * 0.999 + ruido

    A, b = construir_sistema(X_mc, y_norm)
    kappa, lam_min, lam_max = calcular_numero_condicion(A)

    autovalores = np.sort(np.linalg.eigvalsh(A))

    meta = {
        'nombre': 'Escenario Mal Condicionado',
        'descripcion': (
            'x3 se reemplaza por x2 × 0.999 + ruido: dos columnas casi idénticas. '
            'AᵀA tiene un autovalor casi nulo → κ enorme. '
            'Los planos del sistema son casi paralelos → la solución es inestable.'
        ),
        'n_paises': m,
        'n_variables': 3,
        'kappa': kappa,
        'lam_min': lam_min,
        'lam_max': lam_max,
        'autovalores': autovalores.tolist(),
        'autovalor_minimo': float(autovalores[0]),
    }
    return A, b, meta
