# Sistemas de Ecuaciones Lineales aplicados a la Predicción de Emisiones de CO₂

**Materia:** Métodos Numéricos  
**Fecha:** Abril 2026  
**Dataset:** Our World in Data — CO2 Emissions (2019, 89 países)

---

> La pregunta de fondo es sencilla: ¿cuánto CO₂ emite un país por habitante, y en qué medida eso depende de su riqueza, su consumo energético y su uso del carbón? Responder esa pregunta con precisión implica resolver un sistema de ecuaciones lineales. El proyecto estudia cinco métodos para hacerlo, y qué tan frágiles se vuelven esos métodos cuando los datos son difíciles.

---

## 1. El Problema: ¿Qué queremos predecir y por qué?

### 1.1 Variable objetivo

El **CO₂ per cápita** (toneladas de CO₂ por persona y año) es la variable que queremos estimar. No es solo un número ambiental: refleja el estilo de vida energético de un país, su grado de industrialización y su mezcla de fuentes de energía.

En el dataset de 2019 (89 países con datos completos):

| Estadístico | Valor |
|---|---|
| Mínimo | ~0.1 t/persona |
| Mediana (P50) | ~3.5 t/persona |
| Percentil 90 | ~12 t/persona |
| Máximo | ~35 t/persona |

La distribución es fuertemente asimétrica a la derecha: la mayoría de países emite poco, pero unos pocos emiten muchísimo. Esa asimetría también hace que los escenarios de estrés sean interesantes numéricamente.

### 1.2 Variables predictoras

Se usan tres variables por su relevancia causal directa y por ser independientes entre sí (en el escenario ideal):

| Variable | Símbolo | Columna OWID | Interpretación del peso $w_i$ |
|---|---|---|---|
| PIB total (GDP) | $x_1$ | `gdp` | Mayor producción económica → más energía consumida → más CO₂. $w_1 > 0$ indica que países más ricos emiten más por habitante. |
| Energía per cápita (kWh) | $x_2$ | `energy_per_capita` | Consumo energético directo por persona. Proxy del nivel de vida e industrialización. $w_2 > 0$ casi siempre. |
| CO₂ por carbón per cápita (t) | $x_3$ | `coal_co2_per_capita` | Emisiones directas del carbón por persona. Indicador de dependencia de combustible sucio. $w_3 > 0$ y generalmente el peso más alto. |

Cada peso $w_i$ tiene una interpretación marginal: mide cuánto cambia el CO₂ per cápita si esa variable sube una unidad de desviación estándar, **manteniendo las otras dos constantes**.

---

## 2. El Modelo: Regresión Lineal Múltiple

### 2.1 Ecuación del modelo

Con $m = 89$ países y $n = 3$ variables, el modelo predice:

$$\hat{y}_i = w_1 x_{i1} + w_2 x_{i2} + w_3 x_{i3}, \quad i = 1, \ldots, m$$

En forma matricial:

$$\hat{y} = Xw, \quad X \in \mathbb{R}^{m \times 3},\; w \in \mathbb{R}^3,\; y \in \mathbb{R}^m$$

donde cada fila de $X$ es un país y cada columna es una variable.

### 2.2 Intuición geométrica

En el espacio de las variables predictoras $(x_1, x_2, x_3)$, cada país es un punto en $\mathbb{R}^3$ y su CO₂ real $y$ es un valor en una cuarta dimensión. El modelo busca un **hiperplano** $\hat{y} = w_1 x_1 + w_2 x_2 + w_3 x_3$ que pase lo más cerca posible de los 89 puntos, minimizando la suma de distancias cuadráticas en la dirección vertical (dirección $y$).

El vector $w$ es la **pendiente** de ese hiperplano en cada dirección.

### 2.3 Por qué normalizar

Las tres variables tienen escalas completamente distintas:

- $x_1$ (GDP): valores del orden de $10^{11}$ a $10^{13}$ dólares
- $x_2$ (energía): valores del orden de $10^3$ a $10^5$ kWh
- $x_3$ (CO₂ carbón): valores del orden de $10^{-2}$ a $10^1$ toneladas

Si no normalizamos, la matriz $A = X^T X$ tiene elementos que difieren en órdenes de magnitud. Eso dispara artificialmente el número de condición $\kappa(A)$ y hace que los métodos iterativos converjan muy lento o no converjan.

La normalización aplicada es la **estandarización** (media cero, desviación estándar uno):

$$X_{\text{norm}}[:,j] = \frac{X[:,j] - \mu_j}{\sigma_j}, \qquad y_{\text{norm}} = \frac{y - \mu_y}{\sigma_y}$$

Los pesos $w$ resultantes son comparables entre sí: un $w_i$ mayor significa genuinamente mayor influencia, no solo una escala diferente de la variable.

---

## 3. Mínimos Cuadrados — Derivación Paso a Paso

### Paso 1 — Función de costo

Queremos encontrar $w$ que minimice el error cuadrático total entre predicciones $\hat{y} = Xw$ y valores reales $y$:

$$\mathcal{L}(w) = \|Xw - y\|^2 = (Xw - y)^T(Xw - y) = \sum_{i=1}^{m}(\hat{y}_i - y_i)^2$$

Esta función es una paraboloide convexa en $w$: tiene un único mínimo global.

### Paso 2 — Condición de optimalidad

El mínimo ocurre donde el gradiente de $\mathcal{L}$ con respecto a $w$ es cero:

$$\frac{\partial \mathcal{L}}{\partial w} = 2X^T(Xw - y) = 0$$

Expandiendo:

$$X^T X w = X^T y$$

Esta es la condición necesaria y suficiente porque $\nabla^2 \mathcal{L} = 2X^TX \succeq 0$ (semidefinida positiva): la función no tiene puntos de silla, solo un mínimo.

### Paso 3 — Las ecuaciones normales

Definimos:

$$A = X^T X \in \mathbb{R}^{3 \times 3}, \qquad b = X^T y \in \mathbb{R}^3$$

El sistema que hay que resolver es:

$$Aw = b$$

Los elementos de $A$ y $b$ en términos de los datos son:

$$A_{ij} = \sum_{k=1}^{m} x_{ki}\, x_{kj}, \qquad b_i = \sum_{k=1}^{m} x_{ki}\, y_k$$

En forma explícita para $n = 3$:

$$A = \begin{pmatrix}
\sum x_{k1}^2 & \sum x_{k1}x_{k2} & \sum x_{k1}x_{k3} \\
\sum x_{k2}x_{k1} & \sum x_{k2}^2 & \sum x_{k2}x_{k3} \\
\sum x_{k3}x_{k1} & \sum x_{k3}x_{k2} & \sum x_{k3}^2
\end{pmatrix}, \qquad
b = \begin{pmatrix}
\sum x_{k1}\,y_k \\ \sum x_{k2}\,y_k \\ \sum x_{k3}\,y_k
\end{pmatrix}$$

### Paso 4 — Propiedades de A = XᵀX

Estas propiedades son la clave para entender por qué los métodos iterativos funcionan (o fallan):

- **Simétrica:** $A_{ij} = A_{ji}$ porque $\sum_k x_{ki} x_{kj} = \sum_k x_{kj} x_{ki}$. Esto garantiza que los autovalores son reales.
- **Semidefinida positiva:** $w^T A w = w^T X^T X w = \|Xw\|^2 \geq 0$ para todo $w$. No puede haber autovalores negativos.
- **Definida positiva** (invertible) si $X$ tiene rango columna completo, es decir, si las 3 columnas son linealmente independientes. Esto es lo que el escenario mal condicionado viola deliberadamente.
- **Diagonal positiva:** $A_{ii} = \|x_i\|^2 > 0$ (norma cuadrada de la columna $i$). Esto es lo que hace posibles los métodos de Jacobi y Gauss-Seidel.

---

## 4. El Número de Condición κ — Por Qué Importa

### 4.1 Definición

El número de condición de una matriz simétrica positiva definida es:

$$\kappa(A) = \frac{\lambda_{\max}}{\lambda_{\min}}$$

donde $\lambda_{\max}$ y $\lambda_{\min}$ son los autovalores extremos de $A$.

Se calcula a partir de la descomposición espectral: $A = Q \Lambda Q^T$ con $Q$ ortogonal y $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$.

### 4.2 Intuición geométrica

Las curvas de nivel de $\mathcal{L}(w) = \|Aw - b\|^2$ son elipsoides en $\mathbb{R}^n$. Los semiejes de esos elipsoides tienen longitudes proporcionales a $1/\sqrt{\lambda_i}$.

- Si $\kappa \approx 1$: todos los autovalores son similares → el elipsoide es casi una esfera → el gradiente apunta casi directamente al mínimo → convergencia rápida.
- Si $\kappa \gg 1$: autovalores muy dispares → elipsoide muy elongado → el gradiente "se equivoca de dirección" la mayor parte del tiempo → convergencia muy lenta o divergencia.

### 4.3 Efecto en el error numérico

Si el vector $b$ tiene una perturbación $\delta b$ (por redondeo, por ruido en los datos), el error relativo en la solución $w$ se amplifica como:

$$\frac{\|\delta w\|}{\|w\|} \leq \kappa(A) \cdot \frac{\|\delta b\|}{\|b\|}$$

Con $\kappa = 4.59 \times 10^8$: un error de $10^{-8}$ relativo en $b$ puede producir un error del $100\%$ en $w$. La solución se vuelve numéricamente inútil.

### 4.4 Criterio del semáforo

| κ(A) | Interpretación | Estado |
|---|---|---|
| < 100 | Bien condicionado. Todos los métodos convergen. | Bueno |
| 100 – 10⁶ | Condicionamiento moderado. Iterativos simples pueden fallar. | Advertencia |
| > 10⁶ | Mal condicionado. Solo métodos robustos (LU, PCG precondicionado) son confiables. | Critico |

### 4.5 Dominancia diagonal

Una condición suficiente (más fuerte que necesaria) para la convergencia de Jacobi y Gauss-Seidel es la **dominancia diagonal estricta**:

$$|a_{ii}| > \sum_{j \neq i} |a_{ij}|, \quad \forall i$$

Para $A = X^T X$ normalizada con variables relativamente independientes, esta condición suele cumplirse. Si las variables son casi colineales, la dominancia se pierde.

---

## 5. Los Tres Escenarios

Los tres escenarios están diseñados para poner los métodos en condiciones progresivamente más adversas.

### 5.1 Escenario Ideal

- **Datos:** 89 países, variables normalizadas tal como son.
- **Sistema:** $A_{3\times3}$ bien condicionado.
- **κ ≈ 1.72** — casi esfera. Dominancia diagonal cumplida.
- **Efecto:** todos los métodos convergen. Sirve de referencia.

La baja correlación entre las tres variables (PIB, energía, carbón) después de normalizar hace que las filas de $A$ sean casi diagonales. Ese es el caso ideal.

### 5.2 Escenario Bajo Estrés

- **Datos:** solo los países con CO₂ > percentil 90 (~9 países grandes emisores).
- **Perturbación:** la columna de carbón ($x_3$) se multiplica por 50, simulando una demanda energética extrema en combustibles sucios.
- **Re-normalización** del bloque reducido antes de construir $A$.
- **κ ≈ 6.89** — moderado pero más elongado que el ideal.
- **Efecto:** los métodos siguen convergiendo pero en más iteraciones. El sistema tiene menos datos (menor $m$) y una variable dominante.

La intuición: con solo 9 países y $x_3$ escalada ×50, el elipsoide se alarga porque la varianza de una columna eclipsa a las demás.

### 5.3 Escenario Mal Condicionado

- **Construcción:** se reemplaza $x_3$ (carbón) por $x_2 \times 0.999 + \varepsilon$ donde $\varepsilon \sim \mathcal{N}(0, 10^{-4})$.
- **Efecto:** las columnas 2 y 3 de $X$ son casi idénticas. La tercera fila/columna de $A$ es casi igual a la segunda.
- **κ ≈ 4.59 × 10⁸** — casi singular. El autovalor mínimo colapsa a cero.
- **Efecto:** los métodos iterativos simples divergen. Solo LU (directo) y GCP con precondicionamiento aguantan.

La intuición geométrica: tener dos variables casi iguales es como intentar triangular una posición con dos sensores que dicen lo mismo. El sistema no puede distinguir si subir $w_2$ o bajar $w_3$: infinitas soluciones tienen casi el mismo error.

---

## 6. Los Cinco Métodos

### 6.1 Factorización LU

#### Idea central

Descomponer $A = LU$ donde $L$ es triangular inferior (con 1s en la diagonal) y $U$ es triangular superior. Luego resolver en dos pasos triangulares:

$$Ly = b \quad \text{(sustitución hacia adelante)}$$
$$Ux = y \quad \text{(sustitución hacia atrás)}$$

En la práctica se usa pivoteo parcial: $PA = LU$ donde $P$ es una permutación que intercambia filas para poner el elemento más grande en la diagonal en cada paso, evitando divisiones por números pequeños.

#### Algoritmo (con pivoteo parcial)

```
Entrada: A (n×n), b (n,)
Salida:  w (n,)

(L, U, piv) = lu_factor(A)   # O(n³) — una sola vez
w = lu_solve((L, U, piv), b)  # O(n²)
```

Implementado con `scipy.linalg.lu_factor` / `lu_solve`.

#### Por qué es el método de referencia

- **Sin iteraciones:** solución exacta (hasta precisión de máquina) en un número fijo de operaciones.
- **Siempre termina:** no depende de condiciones de convergencia.
- **Costo O(n³):** aceptable para $n = 3$; impracticable para $n = 10^6$.
- **Limitación en sistemas mal condicionados:** LU calcula la solución del sistema perturbado por errores de redondeo. Si $\kappa \approx 10^8$, la solución puede tener ~8 dígitos incorrectos incluso con LU.

#### Resultado numérico

| Escenario | κ | Iteraciones | Residual $\|Aw - b\|$ |
|---|---|---|---|
| Ideal | 1.72 | N/A (directo) | ~10⁻¹⁶ |
| Bajo Estrés | 6.89 | N/A (directo) | ~10⁻¹⁶ |
| Mal Condicionado | 4.59×10⁸ | N/A (directo) | ~10⁻⁸ (limite de precision) |

---

### 6.2 Método de Jacobi

#### Idea central

Despejamos cada variable $w_i$ de la $i$-ésima ecuación, manteniendo fijas todas las demás en el valor de la iteración anterior:

$$w_i^{(k+1)} = \frac{1}{a_{ii}}\left(b_i - \sum_{j \neq i} a_{ij}\, w_j^{(k)}\right)$$

La actualización es **simultánea**: todas las $w_i$ se calculan usando los valores del paso $k$, y solo después se reemplazan.

#### Forma matricial

Separamos $A = D + (L + U)$ donde $D = \text{diag}(a_{11}, a_{22}, a_{33})$:

$$w^{(k+1)} = D^{-1}\bigl(b - (L+U)\, w^{(k)}\bigr) = D^{-1}b - D^{-1}(L+U)\, w^{(k)}$$

La matriz de iteración es $M_{\text{Jac}} = -D^{-1}(L+U)$.

#### Condición de convergencia

El método converge si y solo si el **radio espectral** de la matriz de iteración es menor que 1:

$$\rho(M_{\text{Jac}}) = \max_i |\lambda_i(M_{\text{Jac}})| < 1$$

Una condición **suficiente** (más fácil de verificar) es la dominancia diagonal estricta: si $|a_{ii}| > \sum_{j\neq i}|a_{ij}|$ para todo $i$, entonces $\rho < 1$ garantizado.

#### Pseudocódigo

```
Entrada: A (n×n), b (n,), tol, max_iter
Salida:  w, historial de residuales

D_inv = 1 / diag(A)          # vector de inversas diagonales
R = A - diag(diag(A))         # parte no diagonal
w = zeros(n)

para k = 1, 2, ..., max_iter:
    w_nuevo = D_inv * (b - R @ w)   # actualización simultánea
    res = norm(A @ w_nuevo - b)
    si res < tol: retornar w_nuevo, convergió=True
    w = w_nuevo

retornar w, convergió=False
```

La clave es que `R @ w` usa los valores del paso $k$ completos, no los parcialmente actualizados.

#### Resultado numérico (escenario ideal, κ = 1.72)

| k | w₁ | w₂ | w₃ | ‖rₖ‖ |
|---|---|---|---|---|
| 0 | 0.000000 | 0.000000 | 0.000000 | ~1.5e+00 |
| 1 | ... | ... | ... | ~8.0e-01 |
| ... | | | | |
| 17 | converge | | | < 1e-6 |

- **Iteraciones hasta convergencia:** ~17 (ideal), diverge (mal condicionado)
- **Tasa de convergencia:** ρ(M_Jac) ≈ 0.58 para el escenario ideal → reducción del residual por factor ~0.58 por iteración

---

### 6.3 Gauss-Seidel

#### Idea central

Igual que Jacobi, pero en cuanto calculamos $w_i^{(k+1)}$, lo usamos **inmediatamente** para calcular $w_{i+1}^{(k+1)}$. No esperamos terminar el ciclo completo.

$$w_i^{(k+1)} = \frac{1}{a_{ii}}\left(b_i - \sum_{j < i} a_{ij}\, w_j^{(k+1)} - \sum_{j > i} a_{ij}\, w_j^{(k)}\right)$$

Los términos con $j < i$ usan los **valores nuevos** (ya calculados en este ciclo); los de $j > i$ usan los **valores viejos**.

#### Por qué es más rápido que Jacobi

Para matrices simétricas positivas definidas (SPD), Gauss-Seidel converge con radio espectral $\rho_{\text{GS}} \approx \rho_{\text{Jac}}^2$. En la práctica, GS necesita aproximadamente la mitad de iteraciones que Jacobi para el mismo residual.

La intuición: usar información reciente es siempre mejor que descartarla. Cada variable ya corregida hace que la siguiente corrección sea más precisa.

#### Pseudocódigo

```
Entrada: A (n×n), b (n,), tol, max_iter
Salida:  w, historial de residuales

w = zeros(n)

para k = 1, 2, ..., max_iter:
    w_old = w.copy()
    para i = 0, 1, ..., n-1:
        # usa w[:i] ya actualizados, w_old[i+1:] viejos
        suma = dot(A[i, :i], w[:i]) + dot(A[i, i+1:], w_old[i+1:])
        w[i] = (b[i] - suma) / A[i, i]

    res = norm(A @ w - b)
    si res < tol: retornar w, convergió=True

retornar w, convergió=False
```

El bucle interno **no se puede vectorizar** directamente porque cada $w[i]$ depende del $w[i-1]$ recién calculado.

#### Resultado numérico (escenario ideal, κ = 1.72)

- **Iteraciones hasta convergencia:** ~7 (ideal), diverge (mal condicionado)
- Aproximadamente **2.4× menos iteraciones** que Jacobi, consistente con la relación $\rho_{\text{GS}} \approx \rho_{\text{Jac}}^2$

---

### 6.4 SOR — Sobrerrelajación Sucesiva

#### Idea central

SOR es una extensión de Gauss-Seidel que introduce un **parámetro de relajación** $\omega$:

$$w_i^{(k+1)} = (1 - \omega)\, w_i^{(k)} + \omega\, w_i^{\text{GS}}$$

donde $w_i^{\text{GS}}$ es el valor que daría Gauss-Seidel.

- $\omega = 1$: idéntico a Gauss-Seidel.
- $\omega \in (1, 2)$: **sobrerrelajación** — el nuevo valor se "dispara" más allá del GS, acelerando la convergencia si el sistema es bien portado.
- $\omega \in (0, 1)$: **subrelajación** — amortiguamos la corrección, útil para estabilizar sistemas casi inestables.

#### Por qué existe este parámetro

GS se mueve en la dirección correcta pero da pasos conservadores. Si sabemos que la solución está "más lejos" en esa dirección, $\omega > 1$ acorta el número de iteraciones. Para matrices SPD con $\rho_{\text{Jac}}$ conocido, el $\omega$ óptimo teórico es:

$$\omega_{\text{opt}} = \frac{2}{1 + \sqrt{1 - \rho_{\text{Jac}}^2}}$$

En la práctica (cuando $\rho_{\text{Jac}}$ no es conocido exactamente), se busca $\omega_{\text{opt}}$ numéricamente.

#### Búsqueda del ω óptimo

```
para omega en linspace(1.0, 1.99, 20):
    ejecutar SOR con ese omega
    registrar iteraciones hasta convergencia (o max_iter si no converge)

omega_optimo = omega con menos iteraciones
```

#### Pseudocódigo SOR

```
Entrada: A (n×n), b (n,), omega, tol, max_iter
Salida:  w, historial de residuales

w = zeros(n)

para k = 1, 2, ..., max_iter:
    w_old = w.copy()
    para i = 0, 1, ..., n-1:
        w_gs = (b[i] - dot(A[i,:i], w[:i]) - dot(A[i,i+1:], w_old[i+1:])) / A[i,i]
        w[i] = (1 - omega) * w_old[i] + omega * w_gs

    res = norm(A @ w - b)
    si res < tol: retornar w, convergió=True

retornar w, convergió=False
```

#### Resultado numérico (escenario ideal, κ = 1.72)

- **ω óptimo encontrado:** ≈ 1.25
- **Iteraciones:** ~15 (con ω = 1.25)
- Paradoja aparente: SOR necesita más iteraciones que GS en este escenario porque $\kappa$ ya es muy bajo → el sistema es tan bien condicionado que la aceleración de $\omega > 1$ no compensa el overhead; el ω óptimo real está muy cerca de 1.

---

### 6.5 GCP Suñagua — Gradiente Conjugado Precondicionado

#### Idea central: el método del gradiente conjugado

El gradiente conjugado (GC) es fundamentalmente diferente a los tres métodos anteriores. En lugar de iterar con una fórmula fija, en cada paso **elige la dirección de búsqueda** de forma que sea $A$-ortogonal (conjugada) a todas las anteriores:

$$p_i^T A\, p_j = 0, \quad i \neq j$$

Esta propiedad garantiza que en $n$ pasos exactos (aritmética exacta) el método encuentra la solución. Para $n = 3$ eso significa convergencia **en 3 iteraciones en el peor caso**.

La intuición: en lugar de corregir "una variable a la vez" como Jacobi/GS, GC busca el mínimo de $\mathcal{L}(w)$ de forma coordinada, progresando exactamente $1$ dimensión por paso.

#### El problema del condicionamiento

La velocidad de convergencia del GC estándar depende del número de condición:

$$\frac{\|w^{(k)} - w^*\|}{\|w^{(0)} - w^*\|} \leq 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k$$

Con $\kappa = 4.59 \times 10^8$, este factor es casi 1 y el método converge en prácticamente $n$ pasos de todas formas (porque $n = 3$ es tan pequeño que la garantía teórica ya dice que termina en 3). Para sistemas de mayor dimensión sería un problema severo.

#### El precondicionador: reducir κ efectivo

La idea del precondicionamiento es transformar el sistema $Aw = b$ en un sistema equivalente con $\kappa$ menor. Se introduce una matriz $M \approx A$ fácil de invertir:

$$M^{-1}Aw = M^{-1}b$$

El sistema transformado tiene el mismo conjunto de soluciones pero con $\kappa(M^{-1}A)$ potencialmente mucho menor que $\kappa(A)$.

En este proyecto el precondicionador es el **escalado diagonal**:

$$M = \text{diag}(A), \quad M^{-1} = \text{diag}(1/a_{11}, 1/a_{22}, 1/a_{33})$$

La acción de $M^{-1}$ sobre un vector $r$ es simplemente multiplicación elemento a elemento: $z_i = r_i / a_{ii}$.

El precondicionador escala cada ecuación para que todos los coeficientes diagonales sean 1. Esto "aplana" el elipsoide y agrupa los autovalores.

#### Algoritmo Mz de Suñagua (2020)

La versión estándar del GCP usa $\beta = r_k^T r_k / r_{k-1}^T r_{k-1}$. La variante de Suñagua (Algoritmo 2) usa en cambio el residuo **precondicionado** $z = M^{-1}r$:

$$\beta_k = \frac{r_{k-1}^T z_{k-1}}{r_{k-2}^T z_{k-2}}$$

Esto produce direcciones de búsqueda más alineadas con el espacio precondicionado, mejorando la convergencia cuando $M \neq I$.

#### Pseudocódigo completo

```
Entrada: A (n×n), b (n,), tol, max_iter
Salida:  w, historial de residuales

diag_A = diag(A)            # precondicionador (vector)
w = zeros(n)
r = b - A @ w               # residuo inicial
z = diag_A * r              # residuo precondicionado: M⁻¹r

r_prev = r.copy()
z_prev = z.copy()
r_pp = None
z_pp = None
p = None

para k = 1, 2, ..., max_iter:

    si k == 1:
        p = z_prev.copy()
    sino:
        beta = dot(r_prev, z_prev) / dot(r_pp, z_pp)
        p = z_prev + beta * p

    Ap = A @ p
    alpha = dot(r_prev, z_prev) / dot(p, Ap)

    w = w + alpha * p
    r_new = r_prev - alpha * Ap
    z_new = diag_A * r_new

    res = norm(r_new)
    si res < tol: retornar w, convergió=True

    r_pp, z_pp = r_prev, z_prev
    r_prev, z_prev = r_new, z_new

retornar w, convergió=False
```

La variable $p$ acumula la dirección conjugada; $\alpha$ es el tamaño de paso óptimo en esa dirección; $\beta$ ajusta la dirección para mantener la conjuguidad.

**Nota crítica:** nunca se construye la matriz $M$ explícitamente. La operación $M^{-1}r$ se implementa como `diag_A * r` (multiplicación elemento a elemento en O(n)).

#### Resultado numérico

| Escenario | κ | Iteraciones | Convergió |
|---|---|---|---|
| Ideal | 1.72 | 3 | Sí |
| Bajo Estrés | 6.89 | 3 | Sí |
| Mal Condicionado | 4.59×10⁸ | 3 | Sí (garantía teórica n=3) |

GCP converge en los tres escenarios porque para $n = 3$ variables el número máximo de iteraciones es 3 independientemente de $\kappa$.

---

## 7. Comparación y Análisis

### 7.1 Tabla comparativa de iteraciones

| Método | Ideal (κ=1.72) | Bajo Estrés (κ=6.89) | Mal Cond. (κ≈4.59e8) | Converge siempre |
|---|---|---|---|---|
| LU | Directo | Directo | Directo | Sí (directo) |
| Jacobi | ~17 | ~25 | Diverge | No |
| Gauss-Seidel | ~7 | ~12 | Diverge | No |
| SOR (ω opt.) | ~15 | ~20 | Diverge | No |
| GCP Suñagua | 3 | 3 | 3 | Sí |

### 7.2 Por qué divergen Jacobi, GS y SOR en el escenario mal condicionado

Cuando $x_3 \approx x_2 \times 0.999$, las columnas 2 y 3 de $X$ son casi iguales. Eso hace que las filas 2 y 3 de $A = X^TX$ sean casi iguales también. La **dominancia diagonal** se pierde: $a_{22} \approx a_{23}$ y $a_{33} \approx a_{23}$, entonces $a_{ii} \approx \sum_{j\neq i} a_{ij}$ en lugar de ser estrictamente mayor.

La matriz de iteración $M_{\text{Jac}} = -D^{-1}(L+U)$ adquiere un autovalor $|\lambda| \geq 1$, y el radio espectral $\rho \geq 1$ → divergencia garantizada.

### 7.3 Por qué GCP aguanta el escenario mal condicionado

La garantía de convergencia de GCP no requiere dominancia diagonal ni siquiera $\kappa$ pequeño: solo requiere que $A$ sea **simétrica definida positiva**. En el escenario mal condicionado, $A$ sigue siendo SPD (tiene un autovalor muy pequeño pero positivo). Por lo tanto GCP siempre converge, y para $n = 3$ lo hace en exactamente 3 pasos.

El precondicionador diagonal reduce $\kappa_{\text{efectivo}}$ aunque no llega a eliminar el problema cuando $\kappa \approx 10^8$: simplemente garantiza que los 3 pasos se ejecutan de forma numéricamente más estable.

### 7.4 LU vs GCP: ¿cuándo usar cada uno?

| Criterio | LU | GCP |
|---|---|---|
| $n$ pequeño ($n \leq 1000$) | Preferible | Funciona |
| $n$ grande ($n > 10^4$) | Impracticable (O(n³)) | Preferible (O(n·iter)) |
| $A$ densa | LU | GCP con precondicionador |
| $A$ dispersa | Ineficiente | GCP explota dispersión |
| κ alto | Solución perturbada | Precondicionador ayuda |

Para este proyecto ($n = 3$) ambos dan resultados equivalentes. La importancia es pedagógica: aprender a construirlos desde cero.

---

## 8. Regularización Ridge: el Antídoto contra κ Alto

Cuando el sistema está mal condicionado y no podemos cambiar los datos, la solución práctica es la **regularización de Tikhonov** (Ridge):

$$A_{\text{reg}} = A + \lambda I, \quad \lambda > 0$$

Esto añade $\lambda$ a todos los autovalores: si el mínimo era $\lambda_{\min} \approx 0$, ahora es $\lambda_{\min} + \lambda > 0$. El nuevo número de condición es:

$$\kappa(A_{\text{reg}}) = \frac{\lambda_{\max} + \lambda}{\lambda_{\min} + \lambda}$$

Eligiendo $\lambda$ apropiado, reducimos $\kappa$ a un rango manejable. El costo es una pequeña distorsión de la solución: $w_{\text{reg}}$ no minimiza exactamente $\|Xw - y\|^2$ sino $\|Xw - y\|^2 + \lambda\|w\|^2$ (penalización de norma grande).

En el escenario ideal del proyecto: se aplica Ridge con $\lambda = 10^{-3}$ si $\kappa > 500$.

---

## 9. Conclusiones

**El número de condición κ es el termómetro central del análisis numérico lineal.** Un sistema puede ser matemáticamente correcto pero numéricamente inutilizable. El paso de $\kappa = 1.72$ a $\kappa = 4.59 \times 10^8$ se logró simplemente duplicando una variable con ruido pequeño, algo que ocurre en datos reales (multicolinealidad).

**Los métodos iterativos simples (Jacobi, GS, SOR) dependen de la dominancia diagonal.** Son eficientes cuando se cumple, inútiles cuando no. Para un sistema $3 \times 3$ bien condicionado como el escenario ideal son suficientes. Para producción real o datos ruidosos, no son la primera opción.

**GCP con precondicionamiento es la estrategia más robusta aquí.** La garantía teórica de convergencia en $n$ pasos para matrices SPD, independientemente de κ, lo hace único. El precondicionador diagonal es la versión más simple; precondicionadores más sofisticados (ILU, AMG) pueden reducir κ efectivo de $10^8$ a $10^2$ en problemas reales.

**La elección del método importa más que la implementación.** Implementar Jacobi con NumPy vectorizado es elegante, pero si el sistema diverge, ninguna optimización lo salva. Entender por qué converge o diverge es más valioso que saber cómo implementarlo.

---

## Apéndice: Resumen de Notación

| Símbolo | Significado |
|---|---|
| $X \in \mathbb{R}^{m\times 3}$ | Matriz de datos normalizada ($m$ países, 3 variables) |
| $y \in \mathbb{R}^m$ | Vector de CO₂ per cápita normalizado |
| $w \in \mathbb{R}^3$ | Pesos del modelo (solución buscada) |
| $A = X^TX$ | Matriz de Gram (3×3), simétrica definida positiva |
| $b = X^Ty$ | Lado derecho de las ecuaciones normales |
| $\kappa(A) = \lambda_{\max}/\lambda_{\min}$ | Número de condición |
| $D = \text{diag}(A)$ | Parte diagonal de $A$ |
| $L, U$ | Partes triangular inferior/superior estrictamente |
| $\rho(M)$ | Radio espectral de la matriz de iteración $M$ |
| $\omega$ | Parámetro de relajación SOR |
| $r^{(k)} = b - Aw^{(k)}$ | Residuo en la iteración $k$ |
| $z^{(k)} = M^{-1}r^{(k)}$ | Residuo precondicionado (GCP) |
| $p^{(k)}$ | Dirección de búsqueda conjugada (GCP) |
| $\alpha_k$ | Tamaño de paso óptimo (GCP) |
| $\beta_k$ | Coeficiente de conjugación (GCP Suñagua) |
