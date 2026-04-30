# PROMPT: Predicción de Huella de Carbono mediante Regresión Lineal Múltiple y Métodos Numéricos

> **Curso:** Métodos Numéricos — Abril 2026  
> **Tema:** Resolución de Sistemas Lineales Complejos  
> **Contexto:** Regresión Lineal Múltiple aplicada a emisiones de CO₂ per cápita por país

---

## 0. Filosofía de Implementación

**Usar NumPy vanilla como base.** Esto significa:

- Los algoritmos iterativos (Jacobi, Gauss-Seidel, SOR, GCP) se implementan **desde cero con bucles y operaciones de NumPy**, sin llamar a `scipy.linalg.solve` ni similares dentro del solver.
- `scipy` solo se permite para la **Factorización LU** (único método directo, donde el punto es contrastar, no implementar desde cero).
- `pandas` solo para carga y limpieza inicial del CSV.
- `scikit-learn` solo para `StandardScaler` (normalización) y `PCA` (visualización).
- `plotly` para todas las gráficas del dashboard.
- `streamlit` para la interfaz.

El objetivo es que cada algoritmo sea legible, comentado línea a línea, y que refleje exactamente la matemática descrita.

---

## 1. Estructura del Proyecto

```
proyecto/
├── data/
│   ├── owid-co2-data.csv           # Dataset principal (Our World in Data)
│   └── owid-co2-codebook.csv       # Diccionario de variables
├── src/
│   ├── data_loader.py              # Carga, limpieza y normalización
│   ├── feature_engineering.py     # Selección de 12 variables + construcción AᵀA
│   ├── scenarios.py                # Los 3 escenarios de condicionamiento
│   ├── pca_viz.py                  # PCA manual + visualización 3D
│   ├── metrics.py                  # R², RMSE, número de condición, análisis convergencia
│   └── solvers/
│       ├── lu.py                   # Factorización LU (scipy permitido aquí)
│       ├── jacobi.py               # Jacobi desde cero con NumPy
│       ├── gauss_seidel.py         # Gauss-Seidel desde cero con NumPy
│       ├── sor.py                  # SOR desde cero con NumPy
│       └── pcg_sunagua.py          # Algoritmo Mz (Suñagua 2020) desde cero
├── app.py                          # Dashboard Streamlit
└── requirements.txt
```

---

## 2. El Problema Matemático

### 2.1 Contexto

Se busca predecir las **emisiones de CO₂ per cápita** de un país en función de 12 indicadores socioeconómicos y energéticos. Con `m` países y `n=12` variables, la relación es:

```
ŷ = Xw
```

donde `X` es la matriz de diseño (m × 12), `w` son los pesos a encontrar y `ŷ` las predicciones.

Minimizar el error cuadrático `||Xw - y||²` lleva al sistema de **ecuaciones normales**:

```
(XᵀX) w = Xᵀy
```

Este es el sistema lineal `Aw = b` que todos los métodos resuelven, donde:
- `A = XᵀX` es una matriz cuadrada (12×12), **simétrica y (idealmente) definida positiva**
- `b = Xᵀy` es el lado derecho
- `w` son los coeficientes de regresión (las incógnitas)

### 2.2 Por qué métodos iterativos

`A = XᵀX` tiene estructura especial que hace útiles los métodos iterativos:
- En datasets grandes (miles de países, decenas de variables) LU cuesta O(n³)
- Los métodos iterativos explotan la estructura de A
- El **número de condición** κ(A) determina qué tan difícil es el sistema:
  - κ pequeño → métodos iterativos convergen rápido
  - κ enorme → multicolinealidad, convergencia lenta o divergencia

### 2.3 Definición de variables

| Variable | Columna OWID | Interpretación como peso `wᵢ` |
|---|---|---|
| x₁ | `gdp` | Impacto del PIB total en las emisiones |
| x₂ | `energy_per_capita` | Impacto del consumo energético |
| x₃ | `population` | Efecto de escala poblacional |
| x₄ | `share_global_coal_co2` | Peso del carbón en emisiones globales |
| x₅ | `share_global_oil_co2` | Peso del petróleo en emisiones globales |
| x₆ | `share_global_gas_co2` | Peso del gas en emisiones globales |
| x₇ | `methane_per_capita` | Contribución del metano |
| x₈ | `nitrous_oxide_per_capita` | Contribución del óxido nitroso |
| x₉ | `renewables_share_energy` | Efecto mitigador de renovables |
| x₁₀ | `trade_co2` | CO₂ embebido en comercio |
| x₁₁ | `cement_co2_per_capita` | Emisiones industriales (cemento) |
| x₁₂ | `coal_co2_per_capita` | Emisiones directas de carbón |

**Target `b`:** `co2_per_capita` (emisiones de CO₂ en toneladas por persona)

Los pesos `wᵢ` resultantes indican cuánto contribuye marginalmente cada indicador a las emisiones per cápita, **controlando por todos los demás**.

---

## 3. Carga y Preprocesamiento (`data_loader.py`)

```python
import pandas as pd
import numpy as np

def cargar_datos(ruta_csv: str, ruta_codebook: str) -> pd.DataFrame:
    """
    Carga el dataset OWID y aplica limpieza inicial.
    
    Pasos:
    1. Leer CSV con pandas
    2. Filtrar solo países reales: iso_code de exactamente 3 letras
       sin dígitos (excluye "OWID_WRL", "OWID_EUR", etc.)
    3. Filtrar por el año con más datos completos en el rango 2018-2020
       (usar el año donde la suma de NaN en las 12 columnas sea mínima)
    4. Seleccionar las 12 columnas de features + columna target
    5. Eliminar filas con cualquier NaN en esas columnas
    6. Retornar DataFrame limpio con columna 'country' como índice
    
    Retorna: df_limpio (pd.DataFrame), año_seleccionado (int)
    """

def normalizar(X: np.ndarray, y: np.ndarray):
    """
    Normalización manual con NumPy (StandardScaler equivalente).
    
    Para cada columna j:
        X_norm[:, j] = (X[:, j] - mean_j) / std_j
    
    Normalizar también y (el target) para estabilidad numérica.
    
    Retorna: X_norm, y_norm, params_dict
    donde params_dict = {'X_mean': ..., 'X_std': ..., 'y_mean': ..., 'y_std': ...}
    Guardar params para desnormalizar predicciones después.
    """

def desnormalizar_prediccion(w_norm: np.ndarray, params: dict) -> np.ndarray:
    """
    Convierte los pesos aprendidos en espacio normalizado
    de vuelta a la escala original de CO₂ per cápita.
    """
```

---

## 4. Construcción del Sistema Lineal (`feature_engineering.py`)

```python
import numpy as np

def construir_sistema(X_norm: np.ndarray, y_norm: np.ndarray):
    """
    Construye las ecuaciones normales desde cero con NumPy.
    
    A = XᵀX   →  np.dot(X.T, X)    shape: (12, 12)
    b = Xᵀy   →  np.dot(X.T, y)    shape: (12,)
    
    NO usar np.linalg.lstsq ni similares.
    Retorna: A (ndarray 12x12), b (ndarray 12)
    """

def calcular_numero_condicion(A: np.ndarray) -> float:
    """
    Calcular κ(A) = λ_max / λ_min desde cero con NumPy.
    
    Pasos:
    1. Calcular autovalores con np.linalg.eigvalsh(A)
       (eigvalsh es para matrices simétricas, más estable)
    2. κ = max(autovalores) / min(autovalores)
    3. Retornar κ, λ_min, λ_max
    
    El número de condición determina:
    - κ < 100:     bien condicionado, todos los métodos convergen bien
    - 100 < κ < 1e6: moderado, iterativos pueden ser lentos
    - κ > 1e6:    mal condicionado, Jacobi/GS probablemente divergen
    """
```

---

## 5. Los Tres Escenarios (`scenarios.py`)

```python
import numpy as np

def escenario_ideal(X_norm: np.ndarray, y_norm: np.ndarray):
    """
    ESCENARIO 1: CASO IDEAL
    ========================
    Contexto: Dataset completo de países con datos balanceados.
    Las 12 variables tienen distribuciones independientes y el
    sistema es bien condicionado.
    
    Construcción:
    - Usar X_norm completo (todos los países disponibles)
    - Construir A = XᵀX, b = Xᵀy normalmente
    - Si κ(A) > 1000, aplicar regularización Ridge mínima:
        A_reg = A + λI   con λ = 1e-4
      Esto simula que tenemos priors débiles sobre los pesos.
    
    Verificar: κ(A) debe quedar < 1000 idealmente.
    
    Interpretación: países con mix energético diversificado,
    sin dependencias extremas en ningún combustible.
    
    Retorna: A, b, metadata_dict
    """

def escenario_estres(X_norm: np.ndarray, y_norm: np.ndarray,
                     X_original: np.ndarray, y_original: np.ndarray,
                     params: dict):
    """
    ESCENARIO 2: CASO BAJO ESTRÉS
    ==============================
    Contexto: Países con demanda carbonífera extrema.
    Análogo a migración en aves: el sistema opera en condiciones
    de máxima exigencia energética.
    
    Construcción:
    1. Filtrar países en percentil > 90 de co2_per_capita (emisores extremos)
    2. Sobre ese subconjunto, escalar las columnas de combustibles fósiles
       (índices correspondientes a x₄, x₅, x₆, x₁₁, x₁₂) por factor 50:
           X_stress[:, [3,4,5,10,11]] *= 50
    3. Re-normalizar ese bloque escalado
    4. Construir A_stress = Xᵀ_stress * X_stress, b_stress = Xᵀ_stress * y_stress
    
    Efecto matemático: las magnitudes de AᵀA se disparan, los coeficientes
    diagonales de A se vuelven muy grandes → sistema más difícil.
    κ(A_stress) será significativamente mayor que en el caso ideal.
    
    Verificar: mostrar cuánto aumentó κ respecto al caso ideal.
    
    Retorna: A, b, metadata_dict
    """

def escenario_mal_condicionado(X_norm: np.ndarray, y_norm: np.ndarray):
    """
    ESCENARIO 3: CASO MAL CONDICIONADO
    =====================================
    Contexto: Dos indicadores de carbón casi idénticos en el dataset.
    Análogo a dos fuentes de alimento nutricionalmente indistinguibles:
    el sistema no puede determinar cuánto peso asignar a cada una.
    
    Construcción:
    1. Tomar X_norm (12 columnas)
    2. Agregar columna 13: x_coal_redundante = x₁₂ * 1.001 + ruido
       donde ruido ~ N(0, 1e-5) generado con np.random.seed(42)
    3. Agregar columna 14: x_energy_redundante = x₂ * 0.999 + ruido
       (simula tener consumo energético medido con dos instrumentos levemente distintos)
    4. Construir A (14×14) y b (14,) con las 14 columnas
    
    Efecto matemático:
    - Las columnas 12 y 13 son casi linealmente dependientes
    - Las columnas 2 y 14 también
    - AᵀA tiene autovalores casi nulos → κ enorme (> 1e8)
    - Los hiperplanos en el espacio de soluciones son casi paralelos
    
    Verificar: κ(A) debe ser > 1e6. Mostrar los dos autovalores
    más pequeños para ilustrar el problema.
    
    Retorna: A, b, metadata_dict
    """
```

---

## 6. Solvers — Implementación desde Cero

### 6.1 Factorización LU (`solvers/lu.py`)

```python
import numpy as np
import scipy.linalg
import time

def solve_lu(A: np.ndarray, b: np.ndarray) -> dict:
    """
    FACTORIZACIÓN LU CON PIVOTEO PARCIAL
    =====================================
    Único método donde se permite scipy, porque es el método
    directo de referencia. No es iterativo: da la solución exacta
    en un solo paso (salvo errores de redondeo).
    
    El sistema A = PLU donde:
    - P: matriz de permutación (pivoteo)
    - L: triangular inferior con 1s en diagonal
    - U: triangular superior
    
    Resolución en dos pasos:
    1. Ly = Pb  →  sustitución hacia adelante
    2. Ux = y   →  sustitución hacia atrás
    
    Usar: scipy.linalg.lu_factor(A) y scipy.linalg.lu_solve(lu_piv, b)
    
    Retorna dict con:
    {
        'solucion': w (ndarray),
        'residual': ||Aw - b|| (float),
        'tiempo': segundos (float),
        'iteraciones': 'N/A (método directo)',
        'convergió': True,
        'numero_condicion': κ(A)
    }
    """
```

### 6.2 Jacobi (`solvers/jacobi.py`)

```python
import numpy as np
import time

def solve_jacobi(A: np.ndarray, b: np.ndarray,
                 tol: float = 1e-6, max_iter: int = 10000) -> dict:
    """
    MÉTODO DE JACOBI
    =================
    Idea: despejar cada variable xᵢ de la ecuación i,
    usando los valores de la iteración ANTERIOR para todo lo demás.
    
    Fórmula de actualización:
        x_i^{k+1} = (1/a_ii) * (b_i - sum_{j≠i} a_ij * x_j^k)
    
    Todas las variables se actualizan SIMULTÁNEAMENTE.
    Equivale a: x^{k+1} = D⁻¹(b - (L+U)x^k)
    donde D = diag(A), L+U = A - D
    
    Implementación con NumPy vanilla:
    
        n = len(b)
        x = np.zeros(n)              # x inicial
        D_inv = 1.0 / np.diag(A)    # inversa de diagonal (vector)
        R = A - np.diag(np.diag(A)) # A sin diagonal
        historial = []
        
        for k in range(max_iter):
            x_nuevo = D_inv * (b - R @ x)   # actualización simultánea
            residual = np.linalg.norm(A @ x_nuevo - b)
            historial.append(residual)
            if residual < tol:
                return ...  # convergió
            x = x_nuevo
    
    ADVERTENCIA: verificar dominancia diagonal antes de iterar.
    Si max(|a_ij| para j≠i) >= |a_ii| para algún i, advertir que
    Jacobi puede no converger.
    
    Retorna dict con:
    {
        'solucion': x,
        'residual_final': float,
        'historial_residuales': list[float],  # para graficar convergencia
        'iteraciones': int,
        'convergió': bool,
        'tiempo': float
    }
    """
```

### 6.3 Gauss-Seidel (`solvers/gauss_seidel.py`)

```python
import numpy as np
import time

def solve_gauss_seidel(A: np.ndarray, b: np.ndarray,
                        tol: float = 1e-6, max_iter: int = 10000) -> dict:
    """
    MÉTODO DE GAUSS-SEIDEL
    =======================
    Igual que Jacobi PERO usa los valores más recientes disponibles
    en la misma iteración. Cuando actualiza xᵢ, ya usa los x₁,...,xᵢ₋₁
    recién calculados.
    
    Fórmula para la variable i en iteración k:
        x_i^{k+1} = (1/a_ii) * (b_i
                     - sum_{j<i} a_ij * x_j^{k+1}   ← valores nuevos
                     - sum_{j>i} a_ij * x_j^k)        ← valores viejos
    
    Implementación con NumPy vanilla:
    
        n = len(b)
        x = np.zeros(n)
        historial = []
        
        for k in range(max_iter):
            x_old = x.copy()
            for i in range(n):
                # Suma con valores YA actualizados (j < i)
                suma_nueva = np.dot(A[i, :i], x[:i])
                # Suma con valores ANTERIORES (j > i)
                suma_vieja = np.dot(A[i, i+1:], x_old[i+1:])
                x[i] = (b[i] - suma_nueva - suma_vieja) / A[i, i]
            
            residual = np.linalg.norm(A @ x - b)
            historial.append(residual)
            if residual < tol:
                return ...
    
    El bucle interno sobre i es explícito e inevitable porque cada
    x[i] depende de x[i-1] recién calculado → no se puede vectorizar.
    Esto es lo que lo distingue de Jacobi.
    
    Retorna mismo formato dict que Jacobi.
    """
```

### 6.4 SOR (`solvers/sor.py`)

```python
import numpy as np
import time

def solve_sor(A: np.ndarray, b: np.ndarray,
              omega: float = 1.25,
              tol: float = 1e-6, max_iter: int = 10000) -> dict:
    """
    MÉTODO SOR (Successive Over-Relaxation)
    =========================================
    Extiende Gauss-Seidel con un parámetro de relajación ω.
    
    Idea: el resultado de Gauss-Seidel indica una dirección.
    SOR "apuesta" más fuerte en esa dirección (ω > 1) o
    frena (ω < 1) para estabilizar sistemas que divergirían.
    
    Fórmula:
        x_i^{GS}  = (1/a_ii)(b_i - sum_{j<i} a_ij x_j^{k+1} - sum_{j>i} a_ij x_j^k)
        x_i^{k+1} = (1 - ω) * x_i^k  +  ω * x_i^{GS}
    
    Con ω = 1 → Gauss-Seidel exacto.
    Con ω ∈ (1, 2) → sobrerelajación (acelera).
    Con ω ∈ (0, 1) → subrelajación (estabiliza).
    
    Implementación con NumPy vanilla:
    
        n = len(b)
        x = np.zeros(n)
        historial = []
        
        for k in range(max_iter):
            x_old = x.copy()
            for i in range(n):
                suma_nueva = np.dot(A[i, :i], x[:i])
                suma_vieja = np.dot(A[i, i+1:], x_old[i+1:])
                x_gs = (b[i] - suma_nueva - suma_vieja) / A[i, i]
                x[i] = (1 - omega) * x_old[i] + omega * x_gs  # ← relajación
            
            residual = np.linalg.norm(A @ x - b)
            historial.append(residual)
            if residual < tol:
                return ...
    
    Retorna mismo formato dict que Jacobi.
    """

def buscar_omega_optimo(A: np.ndarray, b: np.ndarray,
                         omega_range: tuple = (1.0, 1.99),
                         n_puntos: int = 20) -> dict:
    """
    Búsqueda en grilla del ω óptimo.
    
    Probar n_puntos valores de ω en omega_range.
    Para cada ω, correr solve_sor con max_iter=500.
    El ω óptimo es el que minimiza iteraciones hasta convergencia.
    Si no converge, registrar max_iter.
    
    Retorna:
    {
        'omega_optimo': float,
        'iteraciones_por_omega': dict{omega: iters},  # para graficar
        'iters_optimo': int
    }
    """
```

### 6.5 GCP — Algoritmo Mz de Suñagua (`solvers/pcg_sunagua.py`)

```python
import numpy as np
import time

def aplicar_precondicionador_inverso(r: np.ndarray, C_diag: np.ndarray) -> np.ndarray:
    """
    Resuelve Mz = r donde M = C Cᵀ y C = diag(A)^{-1/2}.
    
    Como M = CCᵀ = diag(A)^{-1/2} * diag(A)^{-1/2} = diag(A)^{-1}
    (para precondicionador diagonal), resolver Mz = r es simplemente:
        z = M⁻¹ r = diag(A) * r   (multiplicación elemento a elemento)
    
    M NUNCA se construye como matriz explícita.
    Solo se opera con el vector diagonal de C.
    
    Parámetros:
        r: vector residual actual (ndarray n)
        C_diag: vector diagonal de C = diag(A)^{-1/2} (ndarray n)
    Retorna: z (ndarray n)
    """

def solve_pcg_sunagua(A: np.ndarray, b: np.ndarray,
                       tol: float = 1e-6, max_iter: int = 10000) -> dict:
    """
    GRADIENTE CONJUGADO PRECONDICIONADO — ALGORITMO Mz
    ====================================================
    Implementación del Algoritmo 2 del paper:
    
        Suñagua, P. (2020). "Método de Gradientes Conjugados Precondicionado"
        Revista Boliviana de Matemática – UMSA 04 (2020), pp. 2–7.
    
    DIFERENCIA CON EL GCP ESTÁNDAR:
    --------------------------------
    El GCP estándar transforma el sistema explícitamente:
        Ax = b  →  (CACᵀ)x̃ = Cb  →  resuelve x̃  →  recupera x = C⁻ᵀx̃
    Son dos sistemas que resolver.
    
    El Algoritmo Mz de Suñagua redefine el vector de dirección:
        pₖ ← Cᵀ pₖ
    y trabaja con M = CCᵀ implícitamente, obteniendo x DIRECTAMENTE
    sin el paso adicional. Además β usa residuos precondicionados z,
    no los residuos crudos r, lo que da mayor estabilidad numérica.
    
    PRECONDICIONADOR:
    -----------------
    Se usa el precondicionador diagonal sugerido en el paper (sección 4):
        C = diag(A)^{-1/2}
    Es decir, C_diag[i] = 1 / sqrt(A[i,i])
    
    M = CCᵀ equivale a escalar cada ecuación por 1/A[i,i],
    reduciendo el número de condición cuando los elementos diagonales
    tienen magnitudes muy distintas (que es exactamente el caso
    en el escenario de estrés y mal condicionado).
    
    ALGORITMO Mz (línea a línea del paper, Algoritmo 2):
    -----------------------------------------------------
    
    Dados: A, b, x₀ = 0, tolerancia
    
    Calcular C_diag = 1 / sqrt(diag(A))      # precondicionador diagonal
    
    r₀ = b - A @ x₀                          # residuo inicial
    Resolver Mz₀ = r₀  →  z₀ = diag(A) * r₀ # paso de precondicionamiento
    k = 0
    
    Mientras ||rₖ|| > tol:
        k = k + 1
        
        si k == 1:
            p₁ = z₀                          # primera dirección = residuo precondicionado
        sino:
            βₖ = (rₖ₋₁ᵀ zₖ₋₁) / (rₖ₋₂ᵀ zₖ₋₂)  # ← USA z, no r (diferencia clave)
            pₖ = zₖ₋₁ + βₖ * pₖ₋₁
        
        αₖ = (rₖ₋₁ᵀ zₖ₋₁) / (pₖᵀ A pₖ)    # tamaño de paso
        xₖ = xₖ₋₁ + αₖ * pₖ                 # actualizar solución
        rₖ = rₖ₋₁ - αₖ * A @ pₖ             # actualizar residuo
        Resolver Mzₖ = rₖ  →  zₖ = diag(A) * rₖ  # precondicionar nuevo residuo
    
    retornar xₖ
    
    NOTA IMPORTANTE sobre A @ pₖ:
    No se calcula (CACᵀ)pₖ explícitamente.
    Se calcula directamente A @ pₖ con NumPy: np.dot(A, p)
    El precondicionamiento ya está absorbido en la definición de pₖ y zₖ.
    
    Implementación:
    
        n = len(b)
        x = np.zeros(n)
        diag_A = np.diag(A)                      # vector diagonal de A
        C_diag = 1.0 / np.sqrt(diag_A)          # precondicionador C diagonal
        
        r = b - A @ x                            # r₀
        z = diag_A * r                           # z₀ = M⁻¹r₀ = diag(A)*r (ver aplicar_precondicionador_inverso)
        p = None
        r_prev, z_prev = None, None
        historial = [np.linalg.norm(r)]
        
        for k in range(1, max_iter + 1):
            if k == 1:
                p = z.copy()                     # p₁ = z₀
            else:
                rz_actual = np.dot(r_prev, z_prev)
                rz_anterior = np.dot(r_prev_prev, z_prev_prev)
                beta = rz_actual / rz_anterior   # β con z, no con r
                p = z_prev + beta * p            # actualizar dirección
            
            Ap = A @ p                           # A aplicado a dirección
            rz = np.dot(r, z)                   # rᵀz actual
            alpha = rz / np.dot(p, Ap)          # tamaño de paso
            
            x = x + alpha * p                   # actualizar x
            r_nuevo = r - alpha * Ap             # actualizar residuo
            z_nuevo = diag_A * r_nuevo          # precondicionar (Mz = r)
            
            residual = np.linalg.norm(r_nuevo)
            historial.append(residual)
            
            # Rotar variables para siguiente iteración
            r_prev_prev, z_prev_prev = r_prev, z_prev
            r_prev, z_prev = r, z
            r, z = r_nuevo, z_nuevo
            
            if residual < tol:
                return {...}
    
    Retorna mismo formato dict que los otros solvers, más:
    {
        ...,
        'numero_condicion_antes': κ(A),
        'referencia': 'Suñagua (2020), Algoritmo 2 (Mz)'
    }
    """
```

---

## 7. PCA para Visualización 3D (`pca_viz.py`)

```python
import numpy as np

def pca_manual(X: np.ndarray, n_componentes: int = 3):
    """
    PCA implementado desde cero con NumPy.
    
    Pasos:
    1. Centrar X: X_c = X - mean(X, axis=0)
    2. Calcular matriz de covarianza: C = (1/m) * X_cᵀ X_c
    3. Descomposición espectral: autovalores, autovectores = np.linalg.eigh(C)
       (eigh para matrices simétricas, más estable)
    4. Ordenar autovectores de mayor a menor autovalor
    5. Proyectar: X_pca = X_c @ autovectores[:, :n_componentes]
    
    Retorna:
    - X_pca: proyección 3D (m × 3)
    - varianza_explicada: porcentaje de varianza por componente
    - componentes: matriz de loadings (12 × 3) — qué variables pesan más
    - media: para centrar nuevos puntos
    """

def proyectar_hiperplano_en_pca(w: np.ndarray, componentes: np.ndarray,
                                  media: np.ndarray, X_pca: np.ndarray):
    """
    Proyecta el hiperplano de regresión al espacio PCA para graficar en 3D.
    
    El hiperplano original: ŷ = Xw (en espacio de 12 dimensiones)
    En espacio PCA (3D): ŷ_pca = X_pca @ (componentes[:3].T @ w)
    
    Para graficar la superficie:
    - Crear grilla en PC1 y PC2
    - Calcular PC3 predicha por el modelo proyectado
    - Retornar meshgrid listo para Plotly surface
    
    Retorna: xx, yy, zz (meshgrids para go.Surface)
    """
```

---

## 8. Dashboard Streamlit (`app.py`)

### Estilo Visual

Minimalista oscuro. Al inicio del archivo agregar:

```python
st.set_page_config(
    page_title="CO₂ & Métodos Numéricos",
    page_icon="🌍",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1a1f2e; border-radius: 8px; padding: 12px; }
    .stDataFrame { border: 1px solid #2d3748; }
    h1, h2, h3 { color: #4fc3f7; font-weight: 300; letter-spacing: 0.5px; }
    .solver-card {
        background: #1a1f2e;
        border-left: 3px solid #4fc3f7;
        padding: 16px;
        border-radius: 4px;
        margin: 8px 0;
    }
    .converge-yes { color: #81c995; font-weight: bold; }
    .converge-no  { color: #f28b82; font-weight: bold; }
</style>
""", unsafe_allow_html=True)
```

Todas las gráficas usan plantilla Plotly `plotly_dark`. Colores principales:
- Azul: `#4fc3f7`
- Verde: `#81c995`
- Rojo/advertencia: `#f28b82`
- Fondo de cards: `#1a1f2e`

### Navegación

Sidebar con `st.radio` para seleccionar página. Estructura:

```
🌍  Exploración del Dataset
🔢  Sistema Lineal y Escenarios
⚙️  Métodos Numéricos
📐  Visualización 3D de Hiperplanos
🔮  Predictor Interactivo
📖  Documentación Matemática
```

---

### Página 1: 🌍 Exploración del Dataset

**Sección superior — métricas clave:**
- `st.metric` con: nº de países, año seleccionado, rango de CO₂ per cápita, nº de variables

**Mapa coroplético (Plotly):**
- `px.choropleth` con `color='co2_per_capita'`, escala viridis
- Hover: nombre país, CO₂, PIB per cápita

**Matriz de correlación:**
- Calcular con `np.corrcoef(X.T)` (no pandas)
- Graficar como `go.Heatmap` con anotaciones de valores
- Destacar con borde rojo las correlaciones > 0.9 (potencial multicolinealidad)
- Texto explicativo: *"Las celdas en rojo indican pares de variables casi colineales. En el Escenario 3 esto se amplifica artificialmente para estudiar su efecto en los sistemas lineales."*

**Gráfico de varianza explicada por PCA:**
- Barras de varianza por componente + línea acumulada
- Marcar con línea punteada el umbral del 95%

---

### Página 2: 🔢 Sistema Lineal y Escenarios

**Selector de escenario:** `st.selectbox`

Para cada escenario mostrar:

**Card de contexto:**
```
Escenario Ideal: Dataset balanceado de N países. El sistema AᵀA (12×12)
es bien condicionado. Todos los métodos iterativos deberían converger
en pocas iteraciones.
```

**Indicador de número de condición κ (semáforo):**
```python
kappa = calcular_numero_condicion(A)
if kappa < 100:
    color, label = "#81c995", "✅ Bien condicionado"
elif kappa < 1e6:
    color, label = "#ffd54f", "⚠️ Condicionamiento moderado"
else:
    color, label = "#f28b82", "🔴 Mal condicionado"
```
Mostrar como `st.metric` grande con delta indicando cambio vs escenario ideal.

**Heatmap de la matriz AᵀA:**
- `go.Heatmap` de la matriz 12×12 (o 14×14 en escenario 3)
- Escala de color divergente centrada en 0
- Texto: *"En el caso mal condicionado, observa que dos filas/columnas son casi idénticas — esto hace que det(A) ≈ 0 y el sistema sea casi singular."*

**Visualización 3D de hiperplanos (versión reducida a 3 variables):**
- Para ilustrar el concepto, mostrar el sistema reducido con PC1, PC2, PC3
- Tres superficies semitransparentes (una por cada ecuación del sistema 3×3 reducido)
- En caso mal condicionado: dos superficies casi paralelas visualmente
- Usar `go.Surface` con `opacity=0.6`

---

### Página 3: ⚙️ Métodos Numéricos

**Selector de escenario** (afecta el sistema a resolver)

**Botón "Ejecutar todos los métodos"** → corre los 5 solvers y cachea resultados con `st.session_state`

Para cada método, un card expandible (`st.expander`):

```
┌─────────────────────────────────────────────────┐
│  JACOBI                                          │
│  Actualización simultánea de todas las variables │
│  usando valores de la iteración anterior.        │
│                                                  │
│  Iteraciones: 847    Residual: 9.3e-7            │
│  Tiempo: 0.23s       Convergió: ✅               │
└─────────────────────────────────────────────────┘
```

Para el GCP agregar:
```
│  Implementa el Algoritmo Mz de Suñagua (2020),  │
│  que mejora el GCP estándar evitando construir   │
│  M explícitamente y usando residuos              │
│  precondicionados z en el cálculo de β.          │
```

**Gráfico de convergencia superpuesto (Plotly):**
- Eje X: iteración
- Eje Y: `||rₖ||` en escala logarítmica
- Una línea por método iterativo (colores distintos)
- Línea punteada horizontal en `tol = 1e-6`
- LU aparece como punto en iteración 0 (solución directa)

**Tabla comparativa (requerimiento del desafío):**

| Método | Iter. Ideal | Iter. Estrés | Iter. Mal C. | Convergió |
|---|---|---|---|---|
| Jacobi | — | — | — | — |
| Gauss-Seidel | — | — | — | — |
| SOR (ω=?) | — | — | — | — |
| GCP Suñagua | — | — | — | — |
| Fact. LU | N/A | N/A | N/A | ✅ |

**Subpágina de SOR — búsqueda de ω óptimo:**
- Gráfico de barras: ω vs iteraciones necesarias
- Slider para probar ω manualmente y ver cómo cambia la curva de convergencia en tiempo real

---

### Página 4: 📐 Visualización 3D de Hiperplanos

**Scatter 3D principal (Plotly):**
```python
fig = go.Figure()

# Puntos = países en espacio PCA
fig.add_trace(go.Scatter3d(
    x=X_pca[:,0], y=X_pca[:,1], z=X_pca[:,2],
    mode='markers',
    marker=dict(
        size=5,
        color=y,                    # color por CO₂ per cápita
        colorscale='Viridis',
        colorbar=dict(title='CO₂ per cápita'),
        opacity=0.8
    ),
    text=nombres_paises,            # hover con nombre del país
    hovertemplate='%{text}<br>CO₂: %{marker.color:.2f} t<extra></extra>'
))

# Hiperplano de regresión proyectado al espacio PCA
fig.add_trace(go.Surface(
    x=xx, y=yy, z=zz,
    opacity=0.4,
    colorscale=[[0,'#4fc3f7'],[1,'#4fc3f7']],
    showscale=False,
    name='Hiperplano de regresión'
))
```

**Controles en sidebar:**
- Toggle: mostrar/ocultar hiperplano
- Selector de escenario: ver cómo cambia la orientación del hiperplano
- Selector de método: comparar hiperplanos de distintos métodos (sutilmente distintos en caso mal condicionado)

**Panel de loadings PCA:**
- Tabla o barras horizontales mostrando qué variables contribuyen más a PC1, PC2, PC3
- Ayuda a interpretar los ejes del gráfico 3D

---

### Página 5: 🔮 Predictor Interactivo

**Sliders para las 12 variables:**
```python
gdp = st.slider("PIB total (GDP)", min_val, max_val, default)
energy = st.slider("Consumo energético per cápita", ...)
# ... (una por variable)
```
- Valores min/max tomados del dataset real
- Default: mediana del dataset (país "típico")

**Al presionar "Predecir":**
1. Construir vector de features x_new (12,)
2. Normalizar con los params del scaler entrenado: `x_norm = (x_new - X_mean) / X_std`
3. Para predecir con los pesos aprendidos `w`: `y_pred_norm = x_norm @ w`
4. Desnormalizar: `y_pred = y_pred_norm * y_std + y_mean`

**Visualización del resultado:**
- Gauge chart (`go.Indicator`) mostrando el CO₂ predicho
- Referencia: percentil en el dataset global
- Categoría cualitativa: "Emisor Bajo / Medio / Alto / Extremo"

**5 países más similares:**
- Calcular distancia euclídea en espacio normalizado: `||x_norm - X_norm[i]||`
- Mostrar tabla con: país, CO₂ real, CO₂ predicho, distancia
- Scatter mini: ubicar el punto predicho vs los 5 vecinos en espacio PC1-PC2

**Selector de método para la predicción:**
- Mostrar si el método converge bajo el escenario seleccionado
- Si no converge (ej. Jacobi en caso mal condicionado): advertencia en rojo

---

### Página 6: 📖 Documentación Matemática

Organizar con `st.tabs`:

**Tab 1 — ¿Por qué mínimos cuadrados?**
- Derivación de AᵀAx = Aᵀb con notación LaTeX via `st.latex`
- Interpretación geométrica: proyección de b sobre el espacio columna de A

**Tab 2 — Número de condición**
- Definición: κ(A) = λ_max / λ_min
- Interpretación geométrica: hiperplanos casi paralelos = autovalores casi iguales = κ enorme
- Tabla: cómo κ afecta la tasa de convergencia de cada método

**Tab 3 — Pseudocódigos**
- Un `st.code()` por algoritmo con el pseudocódigo limpio
- Para GCP: mostrar Algoritmo 2 del paper de Suñagua textualmente

**Tab 4 — GCP Suñagua (2020)**
- Explicación de la diferencia entre GCP estándar y Algoritmo Mz
- Por qué β usa residuos z y no r
- Por qué M nunca se construye explícitamente
- Cita completa: *Suñagua, P. (2020). Método de Gradientes Conjugados Precondicionado. Revista Boliviana de Matemática – UMSA 04, pp. 2–7.*

**Tab 5 — PCA**
- Por qué PCA es solo para visualización (no para el sistema lineal)
- Qué significa cada componente principal en términos de las variables originales
- Varianza explicada y su relación con la información perdida al proyectar a 3D

---

## 9. Métricas (`metrics.py`)

```python
import numpy as np

def r_cuadrado(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² = 1 - SS_res / SS_tot
    SS_res = ||y - ŷ||²
    SS_tot = ||y - mean(y)||²
    Implementar con NumPy, sin sklearn.
    """

def rmse(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE = sqrt(mean((y - ŷ)²))"""

def analizar_convergencia(historial: list, tol: float) -> dict:
    """
    Dado el historial de ||rₖ||, calcular:
    - iteraciones hasta convergencia (primer k donde historial[k] < tol)
    - tasa de convergencia promedio: mean(historial[k+1] / historial[k])
      (idealmente < 1, cuanto menor más rápido)
    - convergió: bool
    """
```

---

## 10. Requirements

```
numpy
pandas
streamlit
plotly
scipy
```

> **Nota:** `scipy` solo se usa en `solvers/lu.py`. Todo lo demás es NumPy.
> `scikit-learn` **no se usa**. Normalización y PCA implementados desde cero.

---

## 11. Notas Finales de Implementación

1. **GCP — implementar el Algoritmo Mz exactamente como describe el paper.** El cambio clave está en β: usa `rᵀz` (residuo precondicionado) en lugar de `rᵀr`. Documentar esto con comentarios en el código.

2. **Los tres escenarios son transformaciones del mismo dataset**, no datasets distintos. El escenario de estrés filtra un subconjunto y escala columnas; el mal condicionado agrega columnas redundantes.

3. **PCA es solo para visualización.** El sistema lineal siempre se resuelve con las variables originales (12 o 14). Nunca aplicar PCA antes de construir AᵀA.

4. **Si Jacobi o Gauss-Seidel no convergen** en el escenario mal condicionado, registrar `convergió=False` y mostrar el residual mínimo alcanzado. No interrumpir el dashboard.

5. **Tolerancia universal:** `tol = 1e-6` en todos los métodos iterativos, como especifica el desafío.

6. **Cachear resultados** de los solvers con `st.session_state` para evitar recomputar al cambiar de página.

7. **Los datos van en `data/`**. El código debe funcionar con rutas relativas desde la raíz del proyecto.
