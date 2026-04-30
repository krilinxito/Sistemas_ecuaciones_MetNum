import pandas as pd
import numpy as np

FEATURES = ['gdp', 'energy_per_capita', 'coal_co2_per_capita']
TARGET = 'co2_per_capita'


def cargar_datos(ruta_csv: str, ruta_codebook: str = None):
    df = pd.read_csv(ruta_csv)

    # Solo países reales: iso_code exactamente 3 letras mayúsculas
    df = df[df['iso_code'].str.match(r'^[A-Z]{3}$', na=False)]

    # Año con menor cantidad de NaN en las columnas de interés, rango 2018-2020
    cols_interes = FEATURES + [TARGET]
    df_rango = df[df['year'].between(2018, 2020)]
    nans_por_anio = df_rango.groupby('year')[cols_interes].apply(
        lambda g: g.isna().sum().sum()
    )
    anio_sel = int(nans_por_anio.idxmin())

    df_anio = df[df['year'] == anio_sel][['country', 'iso_code'] + cols_interes].copy()
    df_anio = df_anio.dropna(subset=cols_interes)
    df_anio = df_anio.set_index('country')

    return df_anio, anio_sel


def normalizar(X: np.ndarray, y: np.ndarray):
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    # Evitar división por cero
    X_std = np.where(X_std == 0, 1.0, X_std)

    y_mean = float(y.mean())
    y_std = float(y.std())
    if y_std == 0:
        y_std = 1.0

    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std

    params = {
        'X_mean': X_mean,
        'X_std': X_std,
        'y_mean': y_mean,
        'y_std': y_std,
    }
    return X_norm, y_norm, params


def desnormalizar_prediccion(y_pred_norm: np.ndarray, params: dict) -> np.ndarray:
    return y_pred_norm * params['y_std'] + params['y_mean']
