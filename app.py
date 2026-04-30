import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import scipy.linalg
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.data_loader import cargar_datos, normalizar, FEATURES, TARGET
from src.scenarios import escenario_ideal, escenario_estres, escenario_mal_condicionado
from src.solvers.lu import solve_lu
from src.solvers.jacobi import solve_jacobi
from src.solvers.gauss_seidel import solve_gauss_seidel
from src.solvers.sor import solve_sor, buscar_omega_optimo
from src.solvers.pcg_sunagua import solve_pcg_sunagua
from src.metrics import r_cuadrado, rmse

# ── Configuracion ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Sistemas de Ecuaciones — Metodos Numericos", layout="wide")
st.markdown("""
<style>
  .main{background:#0e1117}
  section[data-testid="stSidebar"]{background:#111827}
  h1,h2,h3{color:#4fc3f7;font-weight:400}
  h4{color:#90caf9}
  .card{background:#1a1f2e;border-left:3px solid #4fc3f7;padding:14px 18px;
        border-radius:4px;margin:10px 0}
  .verde{color:#81c995;font-weight:bold}
  .rojo{color:#f28b82;font-weight:bold}
  .amarillo{color:#ffd54f;font-weight:bold}
  code{background:#12192a;padding:2px 6px;border-radius:3px}
</style>
""", unsafe_allow_html=True)

RUTA = os.path.join(os.path.dirname(__file__), 'data', 'owid-co2-data.csv')
TOL  = 1e-6

# Descripcion de las 3 variables
INFO = {
    'gdp':               ('x1', 'PIB total (GDP)',               'Produccion economica total. Mayor riqueza → mas energia → mas CO2.'),
    'energy_per_capita': ('x2', 'Energia per capita (kWh)',      'Consumo energetico por persona. Proxy directo del nivel de vida e industrializacion.'),
    'coal_co2_per_capita':('x3','CO2 por carbon per capita (t)', 'Emisiones directas de carbon por persona. Indicador clave de dependencia de combustible sucio.'),
}
VAR  = ['x1', 'x2', 'x3']
COLS = {m: c for m, c in zip(
    ['Inversa','LU','Jacobi','Gauss-Seidel','SOR','GCP Sunagua'],
    ['#ffb74d','#f28b82','#4fc3f7','#81c995','#ffd54f','#ce93d8']
)}

# ── Cache ──────────────────────────────────────────────────────────────────────
@st.cache_data
def _datos():
    df, yr = cargar_datos(RUTA)
    X  = df[FEATURES].values.astype(float)
    y  = df[TARGET].values.astype(float)
    Xn, yn, par = normalizar(X, y)
    return df, yr, X, y, Xn, yn, par, df.index.tolist(), df['iso_code'].values

@st.cache_data
def _esc():
    A0,b0,m0 = escenario_ideal(Xn, yn)
    A1,b1,m1 = escenario_estres(Xn, yn, Xo, yo, par)
    A2,b2,m2 = escenario_mal_condicionado(Xn, yn)
    return (A0,b0,m0),(A1,b1,m1),(A2,b2,m2)

df,YR,Xo,yo,Xn,yn,par,paises,isos = _datos()
ESC = _esc()
ENAMES = ['Ideal','Bajo Estres','Mal Condicionado']

# ── Utilidades LaTeX ───────────────────────────────────────────────────────────
def lmat(M, fmt='.2f'):
    """Genera string LaTeX para matriz o vector numpy."""
    if M.ndim == 1:
        rows = r' \\ '.join(f'{v:{fmt}}' for v in M)
    else:
        rows = r' \\ '.join(' & '.join(f'{v:{fmt}}' for v in row) for row in M)
    return r'\begin{pmatrix}' + rows + r'\end{pmatrix}'

def sistema_latex(A, b, fmt='.3f'):
    w = r'\begin{pmatrix}w_1\\w_2\\w_3\end{pmatrix}'
    return lmat(A, fmt) + w + '=' + lmat(b, fmt)

def semaforo(k):
    if k < 100:      return 'verde',    f'Bien condicionado (kappa = {k:.2e})'
    if k < 1e6:      return 'amarillo', f'Condicionamiento moderado (kappa = {k:.2e})'
    return              'rojo',         f'Mal condicionado (kappa = {k:.2e})'

# ── Iteraciones detalladas (primeras N) ────────────────────────────────────────
def _iters_jacobi(A, b, n=8):
    D_inv = 1.0 / np.diag(A)
    R = A - np.diag(np.diag(A))
    x = np.zeros(3)
    rows = [{'k':0,'w1':0,'w2':0,'w3':0,'||rk||': float(np.linalg.norm(b))}]
    for k in range(1, n+1):
        x = D_inv * (b - R @ x)
        res = float(np.linalg.norm(A @ x - b))
        rows.append({'k':k,'w1':round(x[0],6),'w2':round(x[1],6),'w3':round(x[2],6),'||rk||':f'{res:.4e}'})
        if not np.isfinite(res): break
    return rows

def _iters_gs(A, b, n=8):
    x = np.zeros(3)
    rows = [{'k':0,'w1':0,'w2':0,'w3':0,'||rk||': f'{np.linalg.norm(b):.4e}'}]
    for k in range(1, n+1):
        xo = x.copy()
        for i in range(3):
            x[i] = (b[i] - np.dot(A[i,:i],x[:i]) - np.dot(A[i,i+1:],xo[i+1:])) / A[i,i]
        res = float(np.linalg.norm(A @ x - b))
        rows.append({'k':k,'w1':round(x[0],6),'w2':round(x[1],6),'w3':round(x[2],6),'||rk||':f'{res:.4e}'})
        if not np.isfinite(res): break
    return rows

def _iters_sor(A, b, omega, n=8):
    x = np.zeros(3)
    rows = [{'k':0,'w1':0,'w2':0,'w3':0,'||rk||': f'{np.linalg.norm(b):.4e}'}]
    for k in range(1, n+1):
        xo = x.copy()
        for i in range(3):
            gs = (b[i]-np.dot(A[i,:i],x[:i])-np.dot(A[i,i+1:],xo[i+1:]))/A[i,i]
            x[i] = (1-omega)*xo[i] + omega*gs
        res = float(np.linalg.norm(A @ x - b))
        rows.append({'k':k,'w1':round(x[0],6),'w2':round(x[1],6),'w3':round(x[2],6),'||rk||':f'{res:.4e}'})
        if not np.isfinite(res): break
    return rows

def _iters_gcp(A, b, n=8):
    diag_A = np.diag(A)
    x = np.zeros(3)
    r = b - A @ x
    z = diag_A * r
    r_prev, z_prev, r_pp, z_pp, p = r.copy(), z.copy(), None, None, None
    rows = [{'k':0,'w1':0,'w2':0,'w3':0,'||rk||':f'{np.linalg.norm(r):.4e}','alpha':'—','beta':'—'}]
    for k in range(1, n+1):
        p = z_prev.copy() if k == 1 else z_prev + (np.dot(r_prev,z_prev)/np.dot(r_pp,z_pp))*p
        beta_str = '—' if k==1 else f'{np.dot(r_prev,z_prev)/np.dot(r_pp,z_pp):.4f}'
        Ap = A @ p
        alpha = np.dot(r_prev,z_prev)/np.dot(p,Ap)
        x = x + alpha*p
        r_new = r_prev - alpha*Ap
        z_new = diag_A * r_new
        res = float(np.linalg.norm(r_new))
        rows.append({'k':k,'w1':round(x[0],6),'w2':round(x[1],6),'w3':round(x[2],6),
                     '||rk||':f'{res:.4e}','alpha':f'{alpha:.4f}','beta':beta_str})
        r_pp,z_pp = r_prev,z_prev
        r_prev,z_prev = r_new,z_new
        if not np.isfinite(res) or res < TOL: break
    return rows

def _solve_inversa(A, b):
    import time
    t0 = time.perf_counter()
    A_inv = np.linalg.inv(A)
    w = A_inv @ b
    tiempo = time.perf_counter() - t0
    residual = float(np.linalg.norm(A @ w - b))
    return {
        'solucion': w,
        'residual_final': residual,
        'historial_residuales': [residual],
        'iteraciones': 'N/A (método directo)',
        'convergió': True,
        'tiempo': tiempo,
        'A_inv': A_inv,
    }

def solver(name, A, b, omega=1.25):
    if name=='Inversa':      return _solve_inversa(A, b)
    if name=='LU':           return solve_lu(A,b)
    if name=='Jacobi':       return solve_jacobi(A,b,tol=TOL,max_iter=10000)
    if name=='Gauss-Seidel': return solve_gauss_seidel(A,b,tol=TOL,max_iter=10000)
    if name=='SOR':          return solve_sor(A,b,omega=omega,tol=TOL,max_iter=10000)
    return solve_pcg_sunagua(A,b,tol=TOL,max_iter=10000)

def bar_solucion(w, titulo, nombre):
    fig = go.Figure(go.Bar(
        x=VAR, y=w,
        marker_color=['#81c995' if v>=0 else '#f28b82' for v in w],
        text=[f'{v:.5f}' for v in w], textposition='outside',
        customdata=[INFO[f][1] for f in FEATURES],
        hovertemplate='%{x}: %{y:.6f}<br>%{customdata}<extra></extra>',
    ))
    fig.update_layout(template='plotly_dark', height=280, title=titulo,
                      yaxis_title='Peso wi', xaxis_title='Variable')
    return fig

def conv_chart(res_dict, titulo):
    fig = go.Figure()
    for nm, r in res_dict.items():
        hist = r['historial_residuales']
        if nm in ('LU', 'Inversa'):
            fig.add_scatter(x=[0], y=[r['residual_final']], mode='markers',
                            marker=dict(size=14, color=COLS[nm], symbol='star'),
                            name=f"{nm}  res={r['residual_final']:.2e}")
        elif hist:
            lbl = f"{nm}  {'OK' if r['convergió'] else 'DIV'}  k={r['iteraciones']}"
            fig.add_scatter(x=list(range(1,len(hist)+1)), y=hist,
                            mode='lines', name=lbl,
                            line=dict(color=COLS[nm], width=2))
    fig.add_hline(y=TOL, line_dash='dot', line_color='white',
                  annotation_text=f'tol={TOL}', annotation_position='bottom right')
    fig.update_layout(template='plotly_dark', height=400,
                      xaxis_title='Iteracion k',
                      yaxis_title='||rk|| = ||Awk − b||',
                      yaxis_type='log', title=titulo,
                      legend=dict(x=0.01,y=0.01))
    return fig

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## Metodos Numericos')
    st.caption('Sistemas Lineales — Abril 2026')
    st.markdown('---')
    pag = st.radio('Seccion', [
        '1. Planteamiento del Problema',
        '2. Sistema Lineal y Escenarios',
        '3. Metodos Numericos',
        '4. Comparacion y Analisis',
        '5. Predictor Interactivo',
    ], label_visibility='collapsed')


# ══════════════════════════════════════════════════════════════════════════════
# 1. PLANTEAMIENTO
# ══════════════════════════════════════════════════════════════════════════════
if pag == '1. Planteamiento del Problema':
    st.title('Prediccion de CO2 per Capita por Regresion Lineal Multiple')
    st.markdown(f'**Dataset:** Our World in Data — CO2 Emissions | Año **{YR}** | {len(paises)} países')

    st.markdown('---')
    st.subheader('Modelo matematico')
    st.latex(r'\hat{y} = w_1 x_1 + w_2 x_2 + w_3 x_3 = X w')
    st.markdown("""
    Con **m** países y **n=3** variables, minimizar `||Xw − y||²` lleva al sistema de
    **ecuaciones normales** que todos los metodos resolveran:
    """)
    st.latex(r'\underbrace{X^T X}_{A\;(3\times 3)} \; w = \underbrace{X^T y}_{b\;(3\times 1)}')

    st.markdown('---')
    st.subheader('Variables seleccionadas')
    st.markdown(
        'Se usan 3 variables por su alta relevancia causal con las emisiones de CO2 '
        'y su facilidad de interpretacion. Cada peso **wᵢ** indica el efecto marginal '
        'de esa variable sobre el CO2 per capita, **controlando por las otras dos**.'
    )
    rows = []
    for feat in FEATURES:
        sym, nom, desc = INFO[feat]
        rows.append({'Variable': sym, 'Columna OWID': feat, 'Nombre': nom,
                     'Media': f'{Xo[:,FEATURES.index(feat)].mean():.3g}',
                     'Desv. std': f'{Xo[:,FEATURES.index(feat)].std():.3g}',
                     'Interpretacion del peso wi': desc})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.markdown('**Variable objetivo y:** `co2_per_capita` — toneladas de CO2 por persona y año.')

    st.markdown('---')
    st.subheader('Exploracion de los datos')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Paises', len(paises))
    c2.metric('Año', YR)
    c3.metric('CO2 min / max (t)', f'{yo.min():.1f} / {yo.max():.1f}')
    c4.metric('CO2 media (t)', f'{yo.mean():.2f}')

    fig_h = go.Figure()
    fig_h.add_histogram(x=yo, nbinsx=25, marker_color='#4fc3f7', opacity=0.8)
    for pct, lbl in [(50,'Mediana'),(90,'P90')]:
        v = np.percentile(yo, pct)
        fig_h.add_vline(x=v, line_dash='dash', line_color='#f28b82',
                        annotation_text=f'{lbl}={v:.1f}t')
    fig_h.update_layout(template='plotly_dark', height=280,
                        xaxis_title='CO2 per capita (t)', yaxis_title='Paises',
                        title='Distribucion del target y = co2_per_capita')
    st.plotly_chart(fig_h, use_container_width=True)

    # Scatter x1, x2, x3 vs y con linea de tendencia
    c1, c2, c3 = st.columns(3)
    for col_i, xi, lbl in [
        (c1, 0, 'x1: PIB total vs CO2'),
        (c2, 1, 'x2: Energia per capita vs CO2'),
        (c3, 2, 'x3: Carbon per capita vs CO2'),
    ]:
        m_fit, b_fit = np.polyfit(Xo[:, xi], yo, 1)
        x_line = np.array([Xo[:, xi].min(), Xo[:, xi].max()])
        fig_s = go.Figure()
        fig_s.add_scatter(
            x=Xo[:, xi], y=yo, mode='markers',
            marker=dict(size=5, color='#4fc3f7', opacity=0.7),
            text=paises,
            hovertemplate='%{text}<br>x=%{x:.3g}  y=%{y:.2f}<extra></extra>',
            name='Paises')
        fig_s.add_scatter(
            x=x_line, y=m_fit * x_line + b_fit, mode='lines',
            line=dict(color='#f28b82', width=2), name='Tendencia')
        fig_s.update_layout(
            template='plotly_dark', height=280, title=lbl,
            xaxis_title=FEATURES[xi], yaxis_title='CO2 per capita (t)',
            showlegend=False)
        col_i.plotly_chart(fig_s, use_container_width=True)

    if isos is not None:
        fig_m = px.choropleth(
            pd.DataFrame({'iso':isos,'pais':paises,'co2':yo}),
            locations='iso', color='co2', hover_name='pais',
            color_continuous_scale='Viridis', template='plotly_dark',
            labels={'co2':'CO2 per capita (t)'}, title=f'CO2 per capita — {YR}')
        fig_m.update_layout(height=380, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_m, use_container_width=True)

    # Scatter 3D + superficie de regresion
    st.markdown('---')
    st.subheader('Nube de datos y superficie de regresion')
    st.markdown(
        'Con 4 dimensiones (x1, x2, x3, y) no existe un unico grafico que lo muestre todo. '
        'Se usa **x3 (carbon)** como color del punto y se fija x3 = 0 (su media normalizada) '
        'para dibujar la superficie de regresion en el espacio (x1, x2, y).'
    )

    A0, b0, _ = ESC[0]
    try:
        w_ideal = np.linalg.solve(A0, b0)
    except np.linalg.LinAlgError:
        w_ideal = np.zeros(3)

    rng_s = np.linspace(Xn[:, 0].min(), Xn[:, 0].max(), 30)
    rng_e = np.linspace(Xn[:, 1].min(), Xn[:, 1].max(), 30)
    Sg, Eg = np.meshgrid(rng_s, rng_e)
    Zg_norm = w_ideal[0] * Sg + w_ideal[1] * Eg
    Zg = Zg_norm * par['y_std'] + par['y_mean']
    Sg_orig = Sg * par['X_std'][0] + par['X_mean'][0]
    Eg_orig = Eg * par['X_std'][1] + par['X_mean'][1]

    fig_3d = go.Figure()
    fig_3d.add_surface(
        x=Sg_orig, y=Eg_orig, z=Zg,
        opacity=0.45,
        colorscale=[[0, '#4fc3f7'], [1, '#81c995']],
        showscale=False, name='Superficie regresion (x3=media)')
    fig_3d.add_scatter3d(
        x=Xo[:, 0], y=Xo[:, 1], z=yo,
        mode='markers',
        marker=dict(
            size=4,
            color=Xo[:, 2],
            colorscale='Plasma',
            colorbar=dict(title='Carbon<br>(t/hab)', x=1.02, len=0.6),
            opacity=0.85),
        text=[f'{p}<br>Carbon: {Xo[i, 2]:.2f}' for i, p in enumerate(paises)],
        hovertemplate='%{text}<br>PIB: %{x:.2e}<br>Energia: %{y:.0f}<br>CO2: %{z:.2f}<extra></extra>',
        name='Paises')
    fig_3d.update_layout(
        template='plotly_dark', height=560,
        scene=dict(
            xaxis_title='PIB total (USD)',
            yaxis_title='Energia per capita (kWh)',
            zaxis_title='CO2 per capita (t)'),
        title='Datos reales + superficie de regresion (x3 = carbon fijo en media)',
        margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig_3d, use_container_width=True)

    # Planos en espacio de pesos (escenario ideal)
    st.markdown('---')
    st.subheader('Sistema de ecuaciones: interseccion de planos en espacio de pesos')
    st.markdown(
        'Cada ecuacion normal A[i,:] · w = b[i] define un **plano en R³** (espacio de pesos). '
        'La solucion w* son los pesos del modelo: el punto donde los tres planos se intersectan.'
    )

    rng_w = np.linspace(-3, 3, 22)
    cols_pl = ['#4fc3f7', '#81c995', '#f28b82']
    labels_eq = ['Ec. 1 — peso PIB (w1)', 'Ec. 2 — peso Energia (w2)', 'Ec. 3 — peso Carbon (w3)']
    fig_pl2 = go.Figure()
    for eq in range(3):
        a = A0[eq]; bi = b0[eq]
        Xg2, Yg2 = np.meshgrid(rng_w, rng_w)
        if abs(a[2]) > 1e-10:
            Zg2 = np.clip((bi - a[0] * Xg2 - a[1] * Yg2) / a[2], -8, 8)
        else:
            Zg2 = np.zeros_like(Xg2)
        fig_pl2.add_surface(
            x=Xg2, y=Yg2, z=Zg2, opacity=0.40,
            colorscale=[[0, cols_pl[eq]], [1, cols_pl[eq]]],
            showscale=False, name=labels_eq[eq])
    fig_pl2.add_scatter3d(
        x=[w_ideal[0]], y=[w_ideal[1]], z=[w_ideal[2]],
        mode='markers+text',
        marker=dict(size=8, color='white', symbol='diamond'),
        text=['w*'], textposition='top center', name='Solucion w*')
    fig_pl2.update_layout(
        template='plotly_dark', height=480,
        scene=dict(
            xaxis_title='w1 (peso PIB)',
            yaxis_title='w2 (peso Energia)',
            zaxis_title='w3 (peso Carbon)'),
        title='Planos del sistema Aw=b — Escenario Ideal')
    st.plotly_chart(fig_pl2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. SISTEMA LINEAL Y ESCENARIOS
# ══════════════════════════════════════════════════════════════════════════════
elif pag == '2. Sistema Lineal y Escenarios':
    st.title('Sistema Lineal y Escenarios')

    esc = st.selectbox('Escenario', ENAMES)
    idx = ENAMES.index(esc)
    A, b, meta = ESC[idx]

    st.markdown(f'<div class="card">{meta["descripcion"]}</div>', unsafe_allow_html=True)

    # Numero de condicion
    kappa = meta['kappa']
    cls, txt_k = semaforo(kappa)
    c1,c2,c3 = st.columns(3)
    c1.metric('kappa(A)', f'{kappa:.4e}')
    c2.metric('Autovalor minimo', f'{meta["lam_min"]:.4e}')
    c3.metric('Autovalor maximo', f'{meta["lam_max"]:.4e}')
    st.markdown(f'<span class="{cls}">{txt_k}</span>', unsafe_allow_html=True)

    st.markdown('---')
    st.subheader('Derivacion: de minimos cuadrados a las ecuaciones normales')

    st.markdown('**Paso 1 — Modelo lineal y funcion de error:**')
    st.latex(r'\hat{y} = Xw, \qquad X \in \mathbb{R}^{m \times 3},\; w \in \mathbb{R}^3,\; y \in \mathbb{R}^m')
    st.markdown(f'Con **m = {meta["n_paises"]} paises** (filas) y **n = 3 variables** (columnas):')
    st.latex(
        r'X = \begin{pmatrix}'
        r'x_{1,1} & x_{1,2} & x_{1,3} \\'
        r'x_{2,1} & x_{2,2} & x_{2,3} \\'
        r'\vdots  & \vdots  & \vdots  \\'
        r'x_{m,1} & x_{m,2} & x_{m,3}'
        r'\end{pmatrix}, \qquad '
        r'y = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_m \end{pmatrix}'
    )
    st.markdown('Queremos minimizar la suma de errores cuadraticos:')
    st.latex(r'\min_w \; \mathcal{L}(w) = \|Xw - y\|^2 = \sum_{i=1}^{m}(\hat{y}_i - y_i)^2')

    st.markdown('**Paso 2 — Condicion de optimalidad (derivada igual a cero):**')
    st.latex(
        r'\frac{\partial \mathcal{L}}{\partial w} = 2X^T(Xw - y) = 0'
        r'\quad \Longrightarrow \quad X^T X w = X^T y'
    )
    st.markdown(
        'Esta es la condicion necesaria y suficiente para el minimo porque '
        r'$\nabla^2\mathcal{L} = 2X^TX \succeq 0$ (semidefinida positiva).'
    )

    st.markdown('**Paso 3 — Construccion de A = XᵀX y b = Xᵀy:**')
    st.latex(
        r'A_{ij} = \sum_{k=1}^{m} x_{k,i}\; x_{k,j}'
        r'\qquad \Longrightarrow \qquad '
        r'A = \begin{pmatrix}'
        r'\sum x_{k1}^2      & \sum x_{k1}x_{k2} & \sum x_{k1}x_{k3} \\'
        r'\sum x_{k2}x_{k1} & \sum x_{k2}^2      & \sum x_{k2}x_{k3} \\'
        r'\sum x_{k3}x_{k1} & \sum x_{k3}x_{k2} & \sum x_{k3}^2'
        r'\end{pmatrix}'
    )
    st.latex(
        r'b_i = \sum_{k=1}^{m} x_{k,i}\; y_k'
        r'\qquad \Longrightarrow \qquad '
        r'b = \begin{pmatrix}'
        r'\sum x_{k1}\,y_k \\ \sum x_{k2}\,y_k \\ \sum x_{k3}\,y_k'
        r'\end{pmatrix}'
    )
    st.markdown("""
    **Propiedades de A = XᵀX:**
    - **Simetrica:** A[i,j] = A[j,i] porque suma_k x_ki x_kj = suma_k x_kj x_ki
    - **Semidefinida positiva:** wᵀAw = wᵀXᵀXw = ||Xw||² >= 0 para todo w
    - **Definida positiva** (invertible) si X tiene rango columna completo (n=3 columnas independientes)
    - La diagonal A[i,i] = ||xi||² es siempre positiva (norma al cuadrado)
    """)

    st.markdown('**Paso 4 — Normalizacion de los datos:**')
    st.latex(
        r'X_{norm}[:,j] = \frac{X[:,j] - \mu_j}{\sigma_j}'
        r'\qquad '
        r'y_{norm} = \frac{y - \mu_y}{\sigma_y}'
    )
    st.markdown(
        'Normalizar hace que todas las variables esten en la misma escala, '
        'lo que estabiliza numericamente el sistema y hace comparables los pesos w. '
        'Los estadisticos del dataset actual son:'
    )
    df_stats = pd.DataFrame({
        'Variable': ['x1: '+INFO['gdp'][1], 'x2: '+INFO['energy_per_capita'][1],
                     'x3: '+INFO['coal_co2_per_capita'][1], 'y: CO2 per capita'],
        'Media': [f'{par["X_mean"][i]:.4g}' for i in range(3)] + [f'{par["y_mean"]:.4g}'],
        'Std': [f'{par["X_std"][i]:.4g}' for i in range(3)] + [f'{par["y_std"]:.4g}'],
    })
    st.dataframe(df_stats, hide_index=True, use_container_width=True)

    st.markdown('---')
    st.subheader('Sistema Aw = b con valores numericos del escenario')
    st.markdown(
        f'Construido con los datos del **{meta["nombre"]}** ({meta["n_paises"]} paises). '
        'Cada elemento A[i,j] es el producto interno de las columnas i y j de X normalizada:'
    )
    st.latex(sistema_latex(A, b))

    with st.expander('Ver A y b como tabla'):
        st.dataframe(
            pd.DataFrame(A, index=VAR, columns=VAR).style.format('{:.4f}'),
            use_container_width=True)
        st.dataframe(
            pd.DataFrame({'b': b}, index=VAR).style.format('{:.4f}'),
            use_container_width=True)

    # Explicacion de kappa
    st.markdown('---')
    st.subheader('Numero de condicion kappa(A)')
    st.latex(r'\kappa(A) = \frac{\lambda_{\max}}{\lambda_{\min}}')
    avals = np.sort(np.linalg.eigvalsh(A))[::-1]
    st.markdown(f"""
    Para este escenario:
    - **lambda_max = {avals[0]:.4e}**
    - **lambda_min = {avals[-1]:.4e}**
    - **kappa = {avals[0]/avals[-1]:.4e}**

    **Interpretacion geometrica:** kappa mide que tan "alargado" es el elipsoide
    de la forma cuadratica φ(w) = wᵀAw. Con kappa grande las curvas de nivel son
    elipses muy excentricias → el gradiente apunta lejos del minimo → la convergencia
    de los metodos iterativos se vuelve muy lenta o imposible.

    **Regla practica:** si kappa ≈ 10^k, se pierden ~k digitos de precision numerica.
    """)

    # Autovalores
    fig_ev = go.Figure(go.Bar(
        x=[f'lambda_{i+1}' for i in range(len(avals))],
        y=avals,
        marker_color='#4fc3f7',
        text=[f'{v:.3e}' for v in avals], textposition='outside',
    ))
    fig_ev.update_layout(template='plotly_dark', height=280, yaxis_type='log',
                         yaxis_title='Autovalor (escala log)',
                         title=f'Espectro de A — kappa = {kappa:.2e}')
    st.plotly_chart(fig_ev, use_container_width=True)

    # Comparacion kappa entre escenarios
    st.markdown('---')
    st.subheader('Comparacion de kappa entre escenarios')
    df_k = pd.DataFrame([
        {'Escenario': ENAMES[i], 'kappa(A)': f'{ESC[i][2]["kappa"]:.3e}',
         'lambda_min': f'{ESC[i][2]["lam_min"]:.3e}',
         'lambda_max': f'{ESC[i][2]["lam_max"]:.3e}',
         'Ridge aplicado': 'Si' if ESC[i][2].get('ridge_aplicado') else 'No'}
        for i in range(3)
    ])
    st.dataframe(df_k, hide_index=True, use_container_width=True)
    st.markdown("""
    | Escenario | Por que cambia kappa |
    |---|---|
    | Ideal | Dataset completo normalizado. Sistema moderadamente condicionado. |
    | Bajo Estres | Subconjunto pequeño (emisores extremos) + carbón ×50. Las magnitudes de AᵀA se disparan. |
    | Mal condicionado | x3 ≈ x2: dos columnas casi iguales → autovalor minimo ≈ 0 → kappa enorme. |
    """)

    # Visualizacion 3D: planos
    st.markdown('---')
    st.subheader('Visualizacion 3D: interseccion de planos')
    st.markdown(
        'Cada ecuacion del sistema A[i,:] · w = b[i] define un plano en R³. '
        'La solucion w* es el punto donde los tres planos se cortan.'
    )
    try:
        w_ref = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w_ref = np.zeros(3)

    rng = np.linspace(-3, 3, 22)
    cols_pl = ['#4fc3f7', '#81c995', '#f28b82']
    fig_pl = go.Figure()
    for eq in range(3):
        a = A[eq]; bi = b[eq]
        Xg, Yg = np.meshgrid(rng, rng)
        if abs(a[2]) > 1e-10:
            Zg = np.clip((bi - a[0]*Xg - a[1]*Yg) / a[2], -8, 8)
        else:
            Zg = np.zeros_like(Xg)
        fig_pl.add_surface(x=Xg, y=Yg, z=Zg, opacity=0.40,
                           colorscale=[[0,cols_pl[eq]],[1,cols_pl[eq]]],
                           showscale=False,
                           name=f'Ecuacion {eq+1}: {VAR[eq]}')
    fig_pl.add_scatter3d(
        x=[w_ref[0]], y=[w_ref[1]], z=[w_ref[2]],
        mode='markers+text', marker=dict(size=8, color='white'),
        text=['w*'], textposition='top center', name='Solucion')
    fig_pl.update_layout(
        template='plotly_dark', height=480,
        scene=dict(xaxis_title='w1 (PIB)', yaxis_title='w2 (Energia)',
                   zaxis_title='w3 (Carbon)'),
        title=f'Planos del sistema — {meta["nombre"]}')
    st.plotly_chart(fig_pl, use_container_width=True)
    if idx == 2:
        st.warning(
            'En el escenario mal condicionado x3 ≈ x2, por lo que dos planos son '
            'casi paralelos. Cualquier perturbacion minima en A o b mueve la '
            'interseccion enormemente → solucion numericamente inestable.'
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3. METODOS NUMERICOS
# ══════════════════════════════════════════════════════════════════════════════
elif pag == '3. Metodos Numericos':
    st.title('Metodos Numericos')

    col_a, col_b = st.columns([1,1])
    with col_a:
        esc = st.selectbox('Escenario', ENAMES)
    with col_b:
        omega_v = st.slider('omega (SOR)', 1.0, 1.99, 1.25, step=0.01)

    idx = ENAMES.index(esc)
    A, b, meta = ESC[idx]
    kappa = meta['kappa']
    cls, txt_k = semaforo(kappa)

    st.markdown(
        f'Sistema: **{meta["nombre"]}** &nbsp;|&nbsp; '
        f'<span class="{cls}">{txt_k}</span>',
        unsafe_allow_html=True
    )

    if st.button('Ejecutar todos los metodos'):
        with st.spinner('Calculando...'):
            st.session_state['R'] = {
                'Inversa':      solver('Inversa', A, b),
                'LU':           solver('LU', A, b),
                'Jacobi':       solver('Jacobi', A, b),
                'Gauss-Seidel': solver('Gauss-Seidel', A, b),
                'SOR':          solver('SOR', A, b, omega_v),
                'GCP Sunagua':  solver('GCP Sunagua', A, b),
            }
            st.session_state['A_cur'] = A
            st.session_state['b_cur'] = b
            st.session_state['omega_cur'] = omega_v

    if 'R' not in st.session_state:
        st.info("Pulsa 'Ejecutar todos los metodos'.")
        st.stop()

    R   = st.session_state['R']
    Ac  = st.session_state['A_cur']
    bc  = st.session_state['b_cur']
    ow  = st.session_state['omega_cur']

    # Curva de convergencia superpuesta
    st.markdown('---')
    st.subheader('Convergencia comparada')
    st.plotly_chart(conv_chart(R, f'Convergencia — {esc}'), use_container_width=True)
    st.markdown("""
    **Como leer el grafico:** el eje Y en escala logaritmica muestra `||rk|| = ||Awk − b||`.
    Una linea que baja rapido indica convergencia rapida. Una linea plana o creciente indica
    divergencia. La linea punteada es la tolerancia tol = 1e-6.
    """)

    # Tabs por metodo
    st.markdown('---')
    st.subheader('Detalle por metodo')
    tabs = st.tabs(['Inversa de A', 'Factorizacion LU', 'Jacobi', 'Gauss-Seidel', 'SOR', 'GCP Sunagua'])

    # ── Inversa ───────────────────────────────────────────────────────────────
    with tabs[0]:
        r = R['Inversa']
        st.markdown('#### Metodo analitico: w = A⁻¹ b')
        st.markdown("""
        **Tipo:** Metodo directo. Calcula explicitamente la inversa de A y luego
        multiplica por b. Correcto matematicamente, pero **no se usa en la practica**
        porque calcular A⁻¹ cuesta ~3× mas que LU y acumula mas error de redondeo.
        """)

        st.markdown('**Formula:**')
        st.latex(r'A w = b \;\Longrightarrow\; w^* = A^{-1} b')

        st.markdown('**A⁻¹ con valores numericos:**')
        A_inv = r['A_inv']
        st.latex(r'A^{-1} = ' + lmat(A_inv, fmt='.5f'))

        st.markdown('**Verificacion A · A⁻¹ = I:**')
        I_check = Ac @ A_inv
        err_id = np.max(np.abs(I_check - np.eye(3)))
        st.latex(r'A \cdot A^{-1} = ' + lmat(I_check, fmt='.6f')
                 + rf'\quad \max|A A^{{-1}} - I| = {err_id:.2e}')

        st.markdown('**Solucion:**')
        st.latex(r'w^* = A^{-1} b = ' + lmat(A_inv, fmt='.5f')
                 + lmat(bc, fmt='.5f') + r'= ' + lmat(r['solucion'], fmt='.6f'))

        c1, c2, c3 = st.columns(3)
        c1.metric('Residual ||Aw*−b||', f"{r['residual_final']:.4e}")
        c2.metric('Tiempo', f"{r['tiempo']*1000:.3f} ms")
        c3.metric('Iteraciones', 'N/A (directo)')

        w = r['solucion']
        st.plotly_chart(bar_solucion(w, 'Coeficientes w — Inversa', 'Inversa'),
                        use_container_width=True)

        y_pred = Xn @ w * par['y_std'] + par['y_mean']
        c1, c2 = st.columns(2)
        c1.metric('R²', f'{r_cuadrado(yo, y_pred):.4f}')
        c2.metric('RMSE (t)', f'{rmse(yo, y_pred):.4f}')

        st.markdown("""
        **Por que no se usa en produccion:**
        - Complejidad **O(n³)** igual que LU, pero con una constante ~3× mayor.
        - Cada elemento de A⁻¹ acumula sumas de productos, amplificando el error de redondeo.
        - Con matrices grandes o mal condicionadas el resultado puede ser inutilizable.
        - LU es algebraicamente equivalente pero numericamente mas estable y eficiente.
        """)

    # ── LU ────────────────────────────────────────────────────────────────────
    with tabs[1]:
        r = R['LU']
        st.markdown('#### Factorizacion LU con pivoteo parcial')
        st.markdown("""
        **Tipo:** Metodo directo. Descompone A = PLU y resuelve en dos pasos.
        Siempre funciona (si A es invertible). Complejidad **O(n³)** — para n=3 es instantaneo.
        """)

        # Mostrar descomposicion
        P_perm, L_mat, U_mat = scipy.linalg.lu(Ac)
        st.markdown('**Descomposicion A = P L U con valores numericos:**')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.latex(r'P = ' + lmat(P_perm))
        with col2:
            st.latex(r'L = ' + lmat(L_mat))
        with col3:
            st.latex(r'U = ' + lmat(U_mat))

        st.markdown('**Verificacion:** P·L·U deberia igualar A:')
        A_rec = P_perm @ L_mat @ U_mat
        err_rec = np.max(np.abs(A_rec - Ac))
        st.latex(r'PLU = ' + lmat(A_rec) + rf'\quad \max|PLU - A| = {err_rec:.2e}')

        st.markdown('**Resolucion:**')
        st.latex(r'L y = P b \;\Rightarrow\; y = ' + lmat(scipy.linalg.solve_triangular(
            L_mat, P_perm @ bc, lower=True)))
        st.latex(r'U w = y \;\Rightarrow\; w^* = ' + lmat(r['solucion']))

        c1,c2,c3 = st.columns(3)
        c1.metric('Residual ||Aw*−b||', f"{r['residual_final']:.4e}")
        c2.metric('Tiempo', f"{r['tiempo']*1000:.2f} ms")
        c3.metric('Iteraciones', 'N/A (directo)')

        w = r['solucion']
        st.plotly_chart(bar_solucion(w, 'Coeficientes w — LU', 'LU'), use_container_width=True)

        y_pred = Xn @ w * par['y_std'] + par['y_mean']
        c1,c2 = st.columns(2)
        c1.metric('R²', f'{r_cuadrado(yo, y_pred):.4f}')
        c2.metric('RMSE (t)', f'{rmse(yo, y_pred):.4f}')
        st.markdown("""
        **Interpretacion de los pesos:**
        El peso `w1` (PIB) indica el efecto del nivel economico sobre las emisiones.
        `w2` (energia) captura la intensidad energetica per capita.
        `w3` (carbon) refleja la dependencia de combustible sucio.
        Un peso positivo grande indica que esa variable es un predictor fuerte de emisiones altas.
        Los pesos estan en espacio normalizado: sus magnitudes son directamente comparables.
        """)

    # ── Jacobi ────────────────────────────────────────────────────────────────
    with tabs[2]:
        r = R['Jacobi']
        st.markdown('#### Metodo de Jacobi')
        st.markdown("""
        **Idea:** Actualizar **todas las variables al mismo tiempo** usando solo los valores
        de la iteracion anterior. La variable `wi^(k+1)` se despeja de la ecuacion i y
        usa `wj^(k)` para j ≠ i.
        """)

        # Descomposicion D, L+U
        D = np.diag(np.diag(Ac))
        LpU = Ac - D
        D_inv = np.diag(1.0/np.diag(Ac))
        M_jac = -D_inv @ LpU

        st.markdown('**Descomposicion A = D + (L+U):**')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.latex(r'D = ' + lmat(D))
        with col2:
            st.latex(r'L+U = ' + lmat(LpU))
        with col3:
            st.latex(r'D^{-1} = ' + lmat(D_inv, fmt='.4f'))

        st.markdown('**Formula de actualizacion matricial:**')
        st.latex(r'w^{(k+1)} = D^{-1}\bigl(b - (L+U)\,w^{(k)}\bigr)')
        st.latex(r'w^{(k+1)} = \underbrace{-D^{-1}(L+U)}_{M_{Jac}} w^{(k)} + D^{-1}b')
        st.latex(r'M_{Jac} = ' + lmat(M_jac, fmt='.4f'))

        rho = max(abs(np.linalg.eigvals(M_jac)))
        if rho < 1:
            st.markdown(f'Radio espectral **rho(M_Jac) = {rho:.4f} < 1** → Jacobi converge.')
        else:
            st.markdown(f'Radio espectral **rho(M_Jac) = {rho:.4f} >= 1** → Jacobi diverge.')

        st.markdown('**Formula componente a componente (la que ejecuta el codigo):**')
        st.latex(
            r'w_1^{(k+1)} = \frac{1}{' + f'{Ac[0,0]:.3f}' + r'}\Bigl('
            + f'{bc[0]:.3f}' + r' - '
            + f'{Ac[0,1]:.3f}' + r'\,w_2^{(k)} - '
            + f'{Ac[0,2]:.3f}' + r'\,w_3^{(k)}\Bigr)'
        )
        st.latex(
            r'w_2^{(k+1)} = \frac{1}{' + f'{Ac[1,1]:.3f}' + r'}\Bigl('
            + f'{bc[1]:.3f}' + r' - '
            + f'{Ac[1,0]:.3f}' + r'\,w_1^{(k)} - '
            + f'{Ac[1,2]:.3f}' + r'\,w_3^{(k)}\Bigr)'
        )
        st.latex(
            r'w_3^{(k+1)} = \frac{1}{' + f'{Ac[2,2]:.3f}' + r'}\Bigl('
            + f'{bc[2]:.3f}' + r' - '
            + f'{Ac[2,0]:.3f}' + r'\,w_1^{(k)} - '
            + f'{Ac[2,1]:.3f}' + r'\,w_2^{(k)}\Bigr)'
        )

        st.markdown('**Primeras iteraciones (con w⁰ = [0, 0, 0]):**')
        df_it = pd.DataFrame(_iters_jacobi(Ac, bc, n=8))
        st.dataframe(df_it, hide_index=True, use_container_width=True)

        # Estado final
        conv = r['convergió']
        if conv:
            st.markdown(f'<span class="verde">Convergio en {r["iteraciones"]} iteraciones. '
                        f'Residual final: {r["residual_final"]:.4e}</span>', unsafe_allow_html=True)
            st.plotly_chart(bar_solucion(r['solucion'], 'Coeficientes w — Jacobi', 'Jacobi'),
                            use_container_width=True)
        else:
            st.markdown(f'<span class="rojo">No convergio ({r["iteraciones"]} iteraciones, '
                        f'residual={r["residual_final"]:.4e})</span>', unsafe_allow_html=True)
            st.markdown("""
            **Por que diverge:** Las ecuaciones normales XᵀX no tienen dominancia diagonal
            estricta (condicion suficiente para Jacobi). El radio espectral del operador
            de iteracion es >= 1, lo que hace que el error crezca en lugar de decrecer.
            """)

    # ── Gauss-Seidel ──────────────────────────────────────────────────────────
    with tabs[3]:
        r = R['Gauss-Seidel']
        st.markdown('#### Metodo de Gauss-Seidel')
        st.markdown("""
        **Idea:** Igual que Jacobi pero al calcular `wi^(k+1)` se usan **inmediatamente**
        los valores ya actualizados `w1^(k+1), ..., w_{i-1}^(k+1)`.
        El bucle interno no se puede vectorizar porque cada componente depende de la anterior.
        """)

        DpL = np.tril(Ac)
        U   = np.triu(Ac, 1)
        DpL_inv = np.linalg.inv(DpL)
        M_gs = -DpL_inv @ U

        col1, col2 = st.columns(2)
        with col1:
            st.latex(r'D + L = ' + lmat(DpL))
        with col2:
            st.latex(r'U = ' + lmat(U))

        st.markdown('**Formula matricial:**')
        st.latex(r'w^{(k+1)} = -(D+L)^{-1} U \, w^{(k)} + (D+L)^{-1} b')
        st.latex(r'M_{GS} = ' + lmat(M_gs, fmt='.4f'))
        rho_gs = max(abs(np.linalg.eigvals(M_gs)))
        st.markdown(f'Radio espectral **rho(M_GS) = {rho_gs:.4f}**'
                    + (' → converge.' if rho_gs < 1 else ' → diverge.'))

        st.markdown('**Formula componente a componente:**')
        st.latex(
            r'w_1^{(k+1)} = \frac{1}{'+f'{Ac[0,0]:.3f}'+r'}\Bigl('
            +f'{bc[0]:.3f}'+r' - '+f'{Ac[0,1]:.3f}'+r'\,w_2^{(k)}'
            +r' - '+f'{Ac[0,2]:.3f}'+r'\,w_3^{(k)}\Bigr)'
        )
        st.latex(
            r'w_2^{(k+1)} = \frac{1}{'+f'{Ac[1,1]:.3f}'+r'}\Bigl('
            +f'{bc[1]:.3f}'+r' - '+f'{Ac[1,0]:.3f}'+r'\,\mathbf{w_1^{(k+1)}}'
            +r' - '+f'{Ac[1,2]:.3f}'+r'\,w_3^{(k)}\Bigr) \quad\leftarrow\text{usa }w_1\text{ ya actualizado}'
        )
        st.latex(
            r'w_3^{(k+1)} = \frac{1}{'+f'{Ac[2,2]:.3f}'+r'}\Bigl('
            +f'{bc[2]:.3f}'+r' - '+f'{Ac[2,0]:.3f}'+r'\,\mathbf{w_1^{(k+1)}}'
            +r' - '+f'{Ac[2,1]:.3f}'+r'\,\mathbf{w_2^{(k+1)}}\Bigr)'
        )

        st.markdown('**Primeras iteraciones:**')
        st.dataframe(pd.DataFrame(_iters_gs(Ac, bc, n=8)), hide_index=True, use_container_width=True)

        if r['convergió']:
            st.markdown(f'<span class="verde">Convergio en {r["iteraciones"]} iter. '
                        f'Residual: {r["residual_final"]:.4e}</span>', unsafe_allow_html=True)
            st.plotly_chart(bar_solucion(r['solucion'], 'Coeficientes w — Gauss-Seidel', 'GS'),
                            use_container_width=True)
        else:
            st.markdown(f'<span class="rojo">No convergio. Residual: {r["residual_final"]:.4e}</span>',
                        unsafe_allow_html=True)

    # ── SOR ───────────────────────────────────────────────────────────────────
    with tabs[4]:
        r = R['SOR']
        st.markdown(f'#### SOR — Sobrerelajacion Sucesiva (omega = {ow:.2f})')
        st.markdown("""
        **Idea:** Extiende Gauss-Seidel con un parametro de relajacion omega.
        Primero calcula el valor GS y luego hace una interpolacion con el valor anterior.
        """)

        st.latex(
            r'w_i^{GS} = \frac{1}{a_{ii}}\Bigl(b_i '
            r'- \sum_{j<i}a_{ij}w_j^{(k+1)} - \sum_{j>i}a_{ij}w_j^{(k)}\Bigr)'
        )
        st.latex(
            r'w_i^{(k+1)} = (1-\omega)\,w_i^{(k)} + \omega\,w_i^{GS}'
        )
        st.markdown(f"""
        Con omega = {ow:.2f}:
        - **omega = 1** → Gauss-Seidel exacto
        - **omega > 1** → sobrerelajacion: acelera si la matriz lo permite
        - **omega < 1** → subrelajacion: estabiliza sistemas problematicos
        """)

        st.markdown(f'**Formula aplicada con omega = {ow:.2f}:**')
        st.latex(
            r'w_i^{(k+1)} = '+f'{1-ow:.2f}'+r'\,w_i^{(k)} + '+f'{ow:.2f}'+
            r'\cdot\frac{1}{a_{ii}}\Bigl(b_i - \sum_{j\neq i}a_{ij}w_j^{(\cdot)}\Bigr)'
        )

        st.markdown('**Primeras iteraciones:**')
        st.dataframe(pd.DataFrame(_iters_sor(Ac, bc, ow, n=8)), hide_index=True,
                     use_container_width=True)

        if r['convergió']:
            st.markdown(f'<span class="verde">Convergio en {r["iteraciones"]} iter. '
                        f'Residual: {r["residual_final"]:.4e}</span>', unsafe_allow_html=True)
            st.plotly_chart(bar_solucion(r['solucion'], f'Coeficientes w — SOR (omega={ow:.2f})', 'SOR'),
                            use_container_width=True)
        else:
            st.markdown(f'<span class="rojo">No convergio. Residual: {r["residual_final"]:.4e}</span>',
                        unsafe_allow_html=True)

        with st.expander('Busqueda de omega optimo'):
            if st.button('Calcular omega optimo (20 puntos en [1.0, 1.99])'):
                with st.spinner('Buscando...'):
                    res_om = buscar_omega_optimo(Ac, bc)
                st.metric('omega optimo', f"{res_om['omega_optimo']:.4f}",
                          f"converge en {res_om['iters_optimo']} iter")
                fig_om = go.Figure(go.Bar(
                    x=list(res_om['iteraciones_por_omega'].keys()),
                    y=list(res_om['iteraciones_por_omega'].values()),
                    marker_color='#ffd54f'))
                fig_om.update_layout(template='plotly_dark', height=280,
                                     xaxis_title='omega', yaxis_title='Iteraciones',
                                     title='Iteraciones hasta convergencia por omega')
                st.plotly_chart(fig_om, use_container_width=True)

    # ── GCP Sunagua ───────────────────────────────────────────────────────────
    with tabs[5]:
        r = R['GCP Sunagua']
        st.markdown('#### Gradiente Conjugado Precondicionado — Algoritmo Mz (Sunagua 2020)')
        st.markdown("""
        **Por que funciona cuando los otros no:** GCP no requiere dominancia diagonal.
        Solo necesita que A sea **simetrica definida positiva** (SPD), lo cual XᵀX siempre
        satisface si X tiene rango completo.

        **Referencia:** Sunagua, P. (2020). *Metodo de Gradientes Conjugados Precondicionado*.
        Revista Boliviana de Matematica – UMSA 04, pp. 2–7.
        """)

        # Precondicionador
        diag_A = np.diag(Ac)
        C_diag = 1.0 / np.sqrt(diag_A)
        st.markdown('**Precondicionador diagonal C = diag(A)^{-1/2}:**')
        st.latex(r'C = \text{diag}(A)^{-1/2} = ' + lmat(np.diag(C_diag), fmt='.4f'))
        st.latex(r'M = CC^T = \text{diag}(A)^{-1} \quad \Rightarrow \quad Mz = r \;\Longleftrightarrow\; z = \text{diag}(A) \cdot r')
        st.markdown('*M nunca se construye como matriz. Resolver Mz = r es una multiplicacion elemento a elemento.*')

        st.markdown('**Algoritmo Mz — diferencia clave con GCP estandar:**')
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('**GCP estandar** (usa r en beta):')
            st.latex(r'\beta_k = \frac{r_{k-1}^T r_{k-1}}{r_{k-2}^T r_{k-2}}')
        with col2:
            st.markdown('**Algoritmo Mz** (usa z en beta):')
            st.latex(r'\beta_k = \frac{r_{k-1}^T z_{k-1}}{r_{k-2}^T z_{k-2}}')
        st.markdown(
            'Usar los residuos precondicionados **z** garantiza que las direcciones de busqueda '
            'sean M-conjugadas, lo que da mayor estabilidad numerica y menos iteraciones.'
        )

        st.markdown('**Pseudocodigo del Algoritmo 2 de Sunagua:**')
        st.code("""
diag_A = diag(A)
r0 = b - A @ x0 ;  z0 = diag_A * r0

para k = 1, 2, ...:
    si k == 1:
        p1 = z0
    sino:
        beta  = (r_{k-1}^T z_{k-1}) / (r_{k-2}^T z_{k-2})   # usa z, no r
        pk    = z_{k-1} + beta * p_{k-1}

    alpha = (r_{k-1}^T z_{k-1}) / (pk^T A pk)
    xk    = x_{k-1} + alpha * pk
    rk    = r_{k-1} - alpha * A pk
    zk    = diag_A * rk                  # resuelve M zk = rk sin construir M

    si ||rk|| < tol: retornar xk
        """, language='text')

        st.markdown('**Primeras iteraciones (alpha y beta incluidos):**')
        st.dataframe(pd.DataFrame(_iters_gcp(Ac, bc, n=8)), hide_index=True,
                     use_container_width=True)

        if r['convergió']:
            st.markdown(f'<span class="verde">Convergio en {r["iteraciones"]} iter. '
                        f'Residual: {r["residual_final"]:.4e}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="rojo">No convergio. Residual: {r["residual_final"]:.4e}</span>',
                        unsafe_allow_html=True)

        st.plotly_chart(bar_solucion(r['solucion'], 'Coeficientes w — GCP Sunagua', 'GCP'),
                        use_container_width=True)

        w = r['solucion']
        y_pred = Xn @ w * par['y_std'] + par['y_mean']
        c1, c2 = st.columns(2)
        c1.metric('R²', f'{r_cuadrado(yo, y_pred):.4f}')
        c2.metric('RMSE (t)', f'{rmse(yo, y_pred):.4f}')

        st.markdown('**Cota teorica de convergencia (Golub & Van Loan):**')
        st.latex(
            r'\|w_k - w^*\|_A \leq 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k'
            r'\|w_0 - w^*\|_A'
        )
        st.markdown(
            f'Con kappa = {kappa:.2e}, el factor de reduccion por iteracion es '
            f'**{((kappa**0.5-1)/(kappa**0.5+1)):.6f}**. '
            'Mas cercano a 0 → convergencia mas rapida.'
        )


# ══════════════════════════════════════════════════════════════════════════════
# 4. COMPARACION Y ANALISIS
# ══════════════════════════════════════════════════════════════════════════════
elif pag == '4. Comparacion y Analisis':
    st.title('Comparacion y Analisis de Metodos')

    if st.button('Calcular tabla completa (6 metodos × 3 escenarios)'):
        with st.spinner('Ejecutando 18 combinaciones...'):
            filas = []
            for nm in ['Inversa','LU','Jacobi','Gauss-Seidel','SOR','GCP Sunagua']:
                fila = {'Metodo': nm}
                for i, en in enumerate(ENAMES):
                    Ai,bi,_ = ESC[i]
                    ri = solver(nm, Ai, bi, omega=1.25)
                    fila[f'Iter. {en}'] = str(ri['iteraciones'])
                    fila[f'Conv. {en}']  = 'Si' if ri['convergió'] else 'No'
                    fila[f'Res. {en}']   = f"{ri['residual_final']:.2e}"
                filas.append(fila)
        st.session_state['tabla'] = filas

    if 'tabla' in st.session_state:
        st.dataframe(pd.DataFrame(st.session_state['tabla']),
                     hide_index=True, use_container_width=True)

        # Grafico de iteraciones
        filas_it = [f for f in st.session_state['tabla'] if f['Metodo'] not in ('LU', 'Inversa')]
        fig_bar = go.Figure()
        for ci, en in enumerate(ENAMES):
            vals = []
            for f in filas_it:
                try:    vals.append(int(f[f'Iter. {en}']))
                except: vals.append(10000)
            fig_bar.add_bar(
                x=[f['Metodo'] for f in filas_it], y=vals, name=en,
                marker_color=['#4fc3f7','#ffd54f','#f28b82'][ci])
        fig_bar.update_layout(template='plotly_dark', barmode='group', height=350,
                               yaxis_type='log', yaxis_title='Iteraciones (escala log)',
                               title='Iteraciones por metodo y escenario')
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption('Barra en 10000 = no convergio dentro del limite de iteraciones.')

    # Tabla de analisis teorico
    st.markdown('---')
    st.subheader('Analisis teorico del condicionamiento y la convergencia')
    df_anal = pd.DataFrame([
        {'Escenario': ENAMES[i],
         'kappa(A)': f'{ESC[i][2]["kappa"]:.2e}',
         'Jacobi': 'Diverge (sin dom. diag.)' if i==0 else 'Diverge',
         'Gauss-Seidel': 'Diverge',
         'SOR': 'Diverge',
         'GCP': ['Converge rapido','Converge (mas lento)','Lento o diverge'][i]}
        for i in range(3)
    ])
    st.dataframe(df_anal, hide_index=True, use_container_width=True)

    st.markdown("""
    **Por que Jacobi, GS y SOR divergen en todos los escenarios:**

    Las ecuaciones normales XᵀX raramente tienen dominancia diagonal. Con 3 variables
    normalizadas, la matriz A ≈ m · R donde R es la matriz de correlaciones.
    Dominancia diagonal exigiria que |r_12| + |r_13| < 1 para cada fila i, lo cual
    falla cuando dos variables estan moderadamente correlacionadas entre si.

    **Por que la Inversa y LU siempre funcionan:**
    Los metodos directos no requieren ninguna condicion sobre la estructura de A (solo que sea invertible).
    La Inversa calcula A⁻¹ explicitamente (menos eficiente); LU factoriza y resuelve en dos pasos (preferido).

    **Por que GCP converge:**
    Solo requiere que A sea SPD (simetrica definida positiva). XᵀX siempre lo es
    si X tiene rango completo. El precondicionador diagonal reduce el kappa efectivo,
    acelerando la convergencia.
    """)

    # Convergencia GCP en los 3 escenarios
    st.subheader('Convergencia del GCP por escenario')
    fig_gcp = go.Figure()
    for ci, (Ai,bi,mi) in enumerate(ESC):
        rg = solve_pcg_sunagua(Ai, bi, tol=TOL, max_iter=10000)
        hist = rg['historial_residuales']
        lbl = (f"{ENAMES[ci]} — kappa={mi['kappa']:.1e}, "
               f"k={rg['iteraciones']}, {'OK' if rg['convergió'] else 'DIV'}")
        fig_gcp.add_scatter(x=list(range(1,len(hist)+1)), y=hist,
                            mode='lines', name=lbl,
                            line=dict(color=['#81c995','#ffd54f','#f28b82'][ci], width=2))
    fig_gcp.add_hline(y=TOL, line_dash='dot', line_color='white')
    fig_gcp.update_layout(template='plotly_dark', height=380,
                          xaxis_title='Iteracion k', yaxis_title='||rk||', yaxis_type='log',
                          title='GCP Sunagua: velocidad de convergencia segun kappa(A)')
    st.plotly_chart(fig_gcp, use_container_width=True)
    st.latex(
        r'\|w_k - w^*\|_A \leq 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k'
        r'\|w_0 - w^*\|_A'
    )
    st.markdown('Mayor kappa → factor de reduccion mas cercano a 1 → convergencia mas lenta.')


# ══════════════════════════════════════════════════════════════════════════════
# 5. PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif pag == '5. Predictor Interactivo':
    st.title('Predictor Interactivo de CO2 per Capita')
    st.markdown('Introduce valores para las 3 variables y el modelo predice las emisiones de CO2.')

    c1, c2 = st.columns(2)
    with c1:
        esc_p = st.selectbox('Escenario', ENAMES)
    with c2:
        met_p = st.selectbox('Metodo', ['LU','GCP Sunagua','SOR','Gauss-Seidel','Jacobi'])

    idx_p = ENAMES.index(esc_p)

    @st.cache_data
    def _wp(ei, met): return solver(met, ESC[ei][0], ESC[ei][1])

    rp = _wp(idx_p, met_p)
    w  = rp['solucion'][:3]

    if not rp['convergió'] and met_p != 'LU':
        st.error(f'{met_p} no convergio en {esc_p}. Predicciones pueden ser inexactas.')

    st.markdown('---')
    vals = {}
    cols_sl = st.columns(3)
    for ci, feat in enumerate(FEATURES):
        _,nom,_ = INFO[feat]
        xi = Xo[:, ci]
        with cols_sl[ci]:
            vals[feat] = st.slider(f'{VAR[ci]}: {nom}',
                                   float(xi.min()), float(xi.max()),
                                   float(np.median(xi)), key=f'sl_{feat}')

    x_new  = np.array([vals[f] for f in FEATURES])
    x_norm = (x_new - par['X_mean']) / par['X_std']
    y_pred = float(np.dot(x_norm, w) * par['y_std'] + par['y_mean'])

    st.markdown('---')
    st.subheader('Resultado')
    pct = float(np.mean(yo <= y_pred) * 100)
    cat = ['Emisor Bajo','Emisor Medio','Emisor Alto','Emisor Extremo'][
        sum([y_pred >= np.percentile(yo, p) for p in [25,50,75]])]
    c1,c2,c3 = st.columns(3)
    c1.metric('CO2 per capita predicho', f'{max(0,y_pred):.3f} t')
    c2.metric('Categoria', cat)
    c3.metric('Percentil global', f'{pct:.1f}%')

    st.markdown('**Calculo paso a paso:**')
    st.latex(
        r'\hat{y}_{norm} = '
        + f'{w[0]:.5f}' + r'\cdot\underbrace{' + f'{x_norm[0]:.3f}' + r'}_{x_1\;norm}'
        + ('+' if w[1]>=0 else '') + f'{w[1]:.5f}' + r'\cdot\underbrace{' + f'{x_norm[1]:.3f}' + r'}_{x_2\;norm}'
        + ('+' if w[2]>=0 else '') + f'{w[2]:.5f}' + r'\cdot\underbrace{' + f'{x_norm[2]:.3f}' + r'}_{x_3\;norm}'
        + r' = ' + f'{y_pred/par["y_std"]-par["y_mean"]/par["y_std"]:.5f}'
    )
    st.latex(
        r'\hat{y} = \hat{y}_{norm} \cdot \sigma_y + \mu_y = '
        + f'(\cdot)\cdot{par["y_std"]:.3f}+{par["y_mean"]:.3f} = {max(0,y_pred):.3f}\;\text{{t}}'
    )

    dists = np.linalg.norm(Xn - x_norm, axis=1)
    idx_v = np.argsort(dists)[:5]
    y_pred_v = Xn[idx_v] @ w * par['y_std'] + par['y_mean']
    st.subheader('5 paises mas similares al perfil introducido')
    st.dataframe(pd.DataFrame({
        'Pais': [paises[i] for i in idx_v],
        'CO2 real (t)': np.round(yo[idx_v], 3),
        'CO2 predicho (t)': np.round(y_pred_v, 3),
        'Error abs. (t)': np.round(np.abs(yo[idx_v]-y_pred_v), 4),
        'Distancia norm.': np.round(dists[idx_v], 4),
    }), hide_index=True, use_container_width=True)
