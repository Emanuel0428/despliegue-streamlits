# -*- coding: utf-8 -*-
"""Despliegue del modelo de clasificacion (Random Forest) con Streamlit.

Pipeline: cargar modelo -> capturar datos -> dummies -> reindexar a `variables`
-> predecir -> devolver la etiqueta original.
"""

import pickle

import pandas as pd
import streamlit as st

COLUMNAS = ['age', 'avg_glucose_level', 'hypertension', 'heart_disease',
            'ever_married', 'smoking_status']
CATEGORICAS = ['smoking_status', 'hypertension', 'heart_disease', 'ever_married']


@st.cache_resource
def cargar_modelo():
    # min_max_scaler no se usa: el modelo final es Random Forest (no normaliza)
    return pickle.load(open('modelo-cla.pkl', 'rb'))


modelo, labelencoder, variables, min_max_scaler = cargar_modelo()

st.set_page_config(page_title="Riesgo cardiovascular", page_icon="❤",
                   layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
  --bg:#07090c; --panel:#0d1117; --line:#1c232c;
  --ink:#e6edf3; --muted:#7d8896; --accent:#3ddc97; --alert:#ff4d5e;
}

[data-testid="stToolbar"], [data-testid="stDecoration"], footer { display:none !important; }
header[data-testid="stHeader"] { background:transparent !important; }

.stApp {
  background:
    repeating-linear-gradient(0deg, rgba(61,220,151,.035) 0 1px, transparent 1px 32px),
    repeating-linear-gradient(90deg, rgba(61,220,151,.035) 0 1px, transparent 1px 32px),
    radial-gradient(900px 500px at 50% -10%, #12202a 0%, var(--bg) 70%);
  color: var(--ink);
  font-family:'Inter', system-ui, sans-serif;
}
.block-container { padding-top:2.2rem; max-width:760px; }

/* ---- cabecera ---- */
.hdr { border:1px solid var(--line); border-radius:2px; background:var(--panel); padding:22px 24px 6px; }
.hdr .eyebrow {
  font-family:'JetBrains Mono', monospace; font-size:.68rem; letter-spacing:.22em;
  text-transform:uppercase; color:var(--muted);
}
.hdr h1 { font-size:1.9rem; font-weight:700; letter-spacing:-.02em; margin:.35rem 0 .1rem; color:var(--ink); }
.hdr .sub { color:var(--muted); font-size:.85rem; }
.ecg { width:100%; height:46px; display:block; margin-top:6px; }
.ecg polyline { fill:none; stroke:var(--accent); stroke-width:1.5; stroke-linejoin:round;
  stroke-dasharray:1400; animation:sweep 4s linear infinite;
  filter:drop-shadow(0 0 6px rgba(61,220,151,.5)); }
@keyframes sweep { from { stroke-dashoffset:1400 } to { stroke-dashoffset:0 } }
@media (prefers-reduced-motion: reduce) { .ecg polyline { animation:none; stroke-dasharray:none } }

/* ---- panel de captura ---- */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div .fld) {
  border:1px solid var(--line) !important; border-radius:2px !important;
  background:linear-gradient(180deg, rgba(61,220,151,.03), transparent 120px) !important;
  padding:4px 18px 14px !important;
}
.fld {
  display:flex; align-items:baseline; gap:.6rem; padding-top:14px;
  font-family:'JetBrains Mono', monospace; font-size:.7rem; letter-spacing:.14em;
  text-transform:uppercase;
}
.fld .n { color:var(--accent); opacity:.55; font-size:.62rem; }
.fld .l { color:var(--muted); white-space:nowrap; }
.fld .dots { flex:1; border-bottom:1px dotted #222b35; transform:translateY(-3px); }
.fld .v { color:var(--accent); letter-spacing:.04em; font-weight:700; text-transform:none; }

/* ---- controles ---- */
div[data-baseweb="select"] > div {
  background:#0a0e13 !important; border:1px solid var(--line) !important;
  border-radius:2px !important; color:var(--ink) !important;
}
div[data-baseweb="select"] > div:hover { border-color:rgba(61,220,151,.45) !important; }
ul[role="listbox"], div[data-baseweb="popover"] div { background:var(--panel) !important; color:var(--ink) !important; }
[data-testid="stThumbValue"] { display:none !important; }
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {
  font-family:'JetBrains Mono', monospace !important; font-size:.6rem !important; color:#4b5663 !important; }
.stSlider [data-baseweb="slider"] > div > div { background:#1a212a !important; }

/* ---- tarjeta de resultado ---- */
.res { border:1px solid var(--line); border-left:3px solid var(--tone); background:var(--panel);
  border-radius:2px; padding:20px 24px; margin-top:26px; }
.res .k { font-family:'JetBrains Mono', monospace; font-size:.68rem; letter-spacing:.22em;
  text-transform:uppercase; color:var(--muted); }
.res .v { font-size:2rem; font-weight:700; letter-spacing:-.02em; color:var(--tone); margin:.1rem 0 .6rem; }
.bar { height:6px; background:#161c24; border-radius:99px; overflow:hidden; }
.bar > span { display:block; height:100%; background:var(--tone); transition:width .3s ease; }
.meta { display:flex; justify-content:space-between; font-family:'JetBrains Mono', monospace;
  font-size:.72rem; color:var(--muted); margin-top:8px; }

.note { font-family:'JetBrains Mono', monospace; font-size:.7rem; color:var(--muted);
  border-top:1px solid var(--line); margin-top:28px; padding-top:12px; line-height:1.7; }

.stDataFrame { border:1px solid var(--line); }
details summary { color:var(--muted) !important; font-size:.8rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hdr">
  <div class="eyebrow">Random Forest &middot; clasificacion</div>
  <h1>Riesgo de ataque al corazon</h1>
  <div class="sub">Captura los datos del paciente. La estimacion se actualiza en vivo.</div>
  <svg class="ecg" viewBox="0 0 700 46" preserveAspectRatio="none">
    <polyline points="0,30 120,30 140,30 150,12 160,42 170,24 182,30 300,30 320,30 330,12 340,42 350,24 362,30 480,30 500,30 510,12 520,42 530,24 542,30 700,30"/>
  </svg>
</div>
""", unsafe_allow_html=True)

def campo(n, etiqueta, widget, fmt=str):
    """Cabecera propia (indice + etiqueta + lectura) sobre un widget sin label."""
    cabecera = st.empty()
    valor = widget()
    cabecera.markdown(
        f'<div class="fld"><span class="n">{n:02d}</span><span class="l">{etiqueta}</span>'
        f'<span class="dots"></span><span class="v">{fmt(valor)}</span></div>',
        unsafe_allow_html=True)
    return valor


SI_NO = ['No', 'Yes']
oculto = dict(label_visibility="collapsed")

st.write("")
izq, der = st.columns(2, gap="large")
with izq.container(border=True):
    age = campo(1, 'Edad', lambda: st.slider(
        'Edad', 1, 82, 45, 1, **oculto), lambda v: f'{v} años')
    hypertension = campo(3, 'Hipertension', lambda: st.selectbox(
        'Hipertension', SI_NO, **oculto))
    ever_married = campo(5, 'Alguna vez casado/a', lambda: st.selectbox(
        'Alguna vez casado/a', SI_NO, **oculto))
with der.container(border=True):
    avg_glucose_level = campo(2, 'Glucosa promedio', lambda: st.slider(
        'Glucosa promedio', 55.0, 272.0, 100.0, 0.1, **oculto), lambda v: f'{v:.1f} mg/dL')
    heart_disease = campo(4, 'Enfermedad cardiaca', lambda: st.selectbox(
        'Enfermedad cardiaca', SI_NO, **oculto))
    smoking_status = campo(6, 'Estado de fumador', lambda: st.selectbox(
        'Estado de fumador', ["'never smoked'", "'formerly smoked'", "smokes", "Unknown"],
        **oculto), lambda v: v.strip("'"))

data = pd.DataFrame([[age, avg_glucose_level, hypertension, heart_disease,
                      ever_married, smoking_status]], columns=COLUMNAS)

# Preparacion: dummies sin drop_first y reindex a las variables del entrenamiento
data_preparada = pd.get_dummies(data, columns=CATEGORICAS, drop_first=False, dtype=int)
data_preparada = data_preparada.reindex(columns=variables, fill_value=0)

etiqueta = labelencoder.inverse_transform(modelo.predict(data_preparada))[0]
probas = dict(zip(labelencoder.inverse_transform(modelo.classes_),
                  modelo.predict_proba(data_preparada)[0]))
p_riesgo = float(probas.get('Yes', 0.0))

riesgo = etiqueta == 'Yes'
st.markdown(f"""
<div class="res" style="--tone:{'var(--alert)' if riesgo else 'var(--accent)'}">
  <div class="k">Prediccion</div>
  <div class="v">{'RIESGO DETECTADO' if riesgo else 'SIN RIESGO'}</div>
  <div class="bar"><span style="width:{p_riesgo * 100:.1f}%"></span></div>
  <div class="meta"><span>probabilidad de ataque</span><span>{p_riesgo * 100:.1f}%</span></div>
</div>
""", unsafe_allow_html=True)

with st.expander("Datos enviados al modelo"):
    st.dataframe(data.assign(Prediccion=etiqueta), width='stretch')

st.markdown('<div class="note">El modelo tiene un error del 8% (MAPE).<br>'
            'Herramienta academica: no sustituye un diagnostico medico.</div>',
            unsafe_allow_html=True)
