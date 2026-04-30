import numpy as np


def pca_manual(X: np.ndarray, n_componentes: int = 3):
    # Centrar
    media = X.mean(axis=0)
    X_c = X - media

    # Matriz de covarianza
    m = X_c.shape[0]
    C = (1 / m) * (X_c.T @ X_c)

    # Descomposición espectral (eigh para matrices simétricas)
    autovalores, autovectores = np.linalg.eigh(C)

    # Ordenar de mayor a menor autovalor
    idx = np.argsort(autovalores)[::-1]
    autovalores = autovalores[idx]
    autovectores = autovectores[:, idx]

    # Varianza explicada
    varianza_total = autovalores.sum()
    varianza_explicada = autovalores[:n_componentes] / varianza_total * 100

    # Proyección
    componentes = autovectores[:, :n_componentes]
    X_pca = X_c @ componentes

    return X_pca, varianza_explicada, componentes, media


def proyectar_hiperplano_en_pca(w: np.ndarray, componentes: np.ndarray,
                                 media: np.ndarray, X_pca: np.ndarray):
    # w puede ser de dimensión 12 o 14; tomar solo las primeras 12 si es necesario
    n_comp = componentes.shape[1]  # 3
    n_vars = componentes.shape[0]  # 12

    # Si w tiene más dimensiones que las variables del PCA (escenario mal cond.), truncar
    w_eff = w[:n_vars]

    # Pesos en espacio PCA: proyección de w sobre los componentes
    w_pca = componentes.T @ w_eff   # (3,)

    # Crear grilla sobre PC1-PC2
    pc1_min, pc1_max = X_pca[:, 0].min(), X_pca[:, 0].max()
    pc2_min, pc2_max = X_pca[:, 1].min(), X_pca[:, 1].max()

    pc1_lin = np.linspace(pc1_min, pc1_max, 30)
    pc2_lin = np.linspace(pc2_min, pc2_max, 30)
    xx, yy = np.meshgrid(pc1_lin, pc2_lin)

    # PC3 predicha por el hiperplano proyectado
    # ŷ_pca = w_pca[0]*PC1 + w_pca[1]*PC2 + w_pca[2]*PC3
    # → PC3 = (ŷ_pca - w_pca[0]*PC1 - w_pca[1]*PC2) / w_pca[2]
    if abs(w_pca[2]) < 1e-10:
        zz = np.zeros_like(xx)
    else:
        # Usar valor medio de y como referencia del plano
        y_medio = 0.0
        zz = (y_medio - w_pca[0] * xx - w_pca[1] * yy) / w_pca[2]
        # Recortar valores extremos para visualización limpia
        z_ref = X_pca[:, 2]
        zz = np.clip(zz, z_ref.min() * 2, z_ref.max() * 2)

    return xx, yy, zz
