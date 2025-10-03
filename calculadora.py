import math
import streamlit as st
import pandas as pd
import json
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

# =========================
# Configuración de página
# =========================
st.set_page_config(page_title="CAC Dashboard", layout="wide")
st.title("Calculadora Dinámica de CAC & Pricing")

# ============ Manejo de carga JSON ============
with st.sidebar:
    debug_mode = st.sidebar.checkbox("Mostrar tabla de cálculos", value=False)
    uploaded_file = st.file_uploader("Cargar parámetros desde JSON", type=["json"])
    if uploaded_file is not None:
        try:
            loaded_params = json.load(uploaded_file)
            for key, value in loaded_params.items():
                st.session_state[key] = value
            st.experimental_rerun()
        except Exception as e:
            st.sidebar.error(f"Error al cargar el archivo JSON: {e}")

# =========================
# Inicializar parámetros en session_state
# =========================
defaults = {
    "marketing": 0.0,
    "influencers": 0.0,
    "ventas_nomina": 0.0,
    "crm": 0.0,
    "equipo": 0.0,
    "instalacion": 0.0,
    "pasarelas": 0.0,
    "antifraude": 0.0,
    "plataforma_ecommerce": 0.0,
    "plataforma_ultima_milla": 0.0,
    "nomina_op": 0.0,
    "soporte": 0.0,
    "overhead": 0.0,
    "precio_mensual": 369.0,
    "auto_precio": False,
    "meses_plan": 12,
    "meses_gratis": 0,
    "churn_mensual_pct": 0,
    "costo_variable_mensual": 0.0,
    "tasa_desc_anual_pct": 0.0
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

with st.sidebar:
    st.subheader("Marketing & Publicidad")
    marketing_val = st.number_input("Inversión en campañas", min_value=0.0, step=100.0, value=st.session_state.get("marketing", defaults["marketing"]))
    st.session_state["marketing"] = marketing_val
    influencers_val = st.number_input("Costo de influencers / PR", min_value=0.0, step=100.0, value=st.session_state.get("influencers", defaults["influencers"]))
    st.session_state["influencers"] = influencers_val

    st.subheader("Ventas & Comercial")
    ventas_nomina_val = st.number_input("Sueldos + comisiones", min_value=0.0, step=100.0, value=st.session_state.get("ventas_nomina", defaults["ventas_nomina"]))
    st.session_state["ventas_nomina"] = ventas_nomina_val
    crm_val = st.number_input("Licencias CRM (HubSpot, etc.)", min_value=0.0, step=100.0, value=st.session_state.get("crm", defaults["crm"]))
    st.session_state["crm"] = crm_val

    st.subheader("Onboarding del Cliente")
    equipo_val = st.number_input("Subsidio/costo purificador", min_value=0.0, step=100.0, value=st.session_state.get("equipo", defaults["equipo"]))
    st.session_state["equipo"] = equipo_val
    instalacion_val = st.number_input("Logística de instalación/envío", min_value=0.0, step=100.0, value=st.session_state.get("instalacion", defaults["instalacion"]))
    st.session_state["instalacion"] = instalacion_val

    st.subheader("Plataformas & Procesamiento")
    pasarelas_val = st.number_input("Tarifas de pasarela de pago", min_value=0.0, step=100.0, value=st.session_state.get("pasarelas", defaults["pasarelas"]))
    st.session_state["pasarelas"] = pasarelas_val
    antifraude_val = st.number_input("Costos antifraude/validación", min_value=0.0, step=100.0, value=st.session_state.get("antifraude", defaults["antifraude"]))
    st.session_state["antifraude"] = antifraude_val
    plataforma_ecommerce_val = st.number_input("Costo de plataforma e-commerce", min_value=0.0, step=100.0, value=st.session_state.get("plataforma_ecommerce", defaults["plataforma_ecommerce"]))
    st.session_state["plataforma_ecommerce"] = plataforma_ecommerce_val
    plataforma_ultima_milla_val = st.number_input("Costo de plataforma última milla", min_value=0.0, step=100.0, value=st.session_state.get("plataforma_ultima_milla", defaults["plataforma_ultima_milla"]))
    st.session_state["plataforma_ultima_milla"] = plataforma_ultima_milla_val

    st.subheader("Operación Directa")
    nomina_op_val = st.number_input("Nómina de operaciones (altas)", min_value=0.0, step=100.0, value=st.session_state.get("nomina_op", defaults["nomina_op"]))
    st.session_state["nomina_op"] = nomina_op_val
    soporte_val = st.number_input("Soporte / Call center inicial", min_value=0.0, step=100.0, value=st.session_state.get("soporte", defaults["soporte"]))
    st.session_state["soporte"] = soporte_val

    st.subheader("Otros / Indirectos")
    overhead_val = st.number_input("Overhead / gastos indirectos", min_value=0.0, step=100.0, value=st.session_state.get("overhead", defaults["overhead"]))
    st.session_state["overhead"] = overhead_val

    st.subheader("Gestión de parámetros")
    params = {k: st.session_state.get(k, v) for k, v in defaults.items()}
    json_str = json.dumps(params, indent=4)
    st.download_button(
        label="Descargar parámetros (JSON)",
        data=json_str,
        file_name="parametros_cac.json",
        mime="application/json"
    )

# ---- CAC por cliente ----
CAC = (
    st.session_state["marketing"] + st.session_state["influencers"] +
    st.session_state["ventas_nomina"] + st.session_state["crm"] +
    st.session_state["equipo"] + st.session_state["instalacion"] +
    st.session_state["pasarelas"] + st.session_state["antifraude"] +
    st.session_state["plataforma_ecommerce"] + st.session_state["plataforma_ultima_milla"] +
    st.session_state["nomina_op"] + st.session_state["soporte"] +
    st.session_state["overhead"]
)

# =========================
# 2) Parámetros de simulación
# =========================
p1, p2, p3, p4 = st.columns(4)
with p1:
    precio_mensual_val = st.number_input("Precio mensual del plan ($)", min_value=0.0, step=10.0, value=st.session_state.get("precio_mensual", defaults["precio_mensual"]))
    st.session_state["precio_mensual"] = precio_mensual_val
    auto_precio_val = st.checkbox("Auto: calcular precio objetivo\n(romper CAC en N meses)", value=st.session_state.get("auto_precio", defaults["auto_precio"]))
    st.session_state["auto_precio"] = auto_precio_val
with p2:
    meses_plan_val = st.slider("Duración del plan (meses)", min_value=1, max_value=60, value=st.session_state.get("meses_plan", defaults["meses_plan"]))
    st.session_state["meses_plan"] = meses_plan_val
    meses_gratis_val = st.slider("Meses gratis/promoción", min_value=0, max_value=6, value=st.session_state.get("meses_gratis", defaults["meses_gratis"]))
    st.session_state["meses_gratis"] = meses_gratis_val
with p3:
    churn_mensual_pct_val = st.slider("Churn mensual (%)", min_value=0, max_value=30, value=st.session_state.get("churn_mensual_pct", defaults["churn_mensual_pct"]))
    st.session_state["churn_mensual_pct"] = churn_mensual_pct_val
    costo_variable_mensual_val = st.number_input("Costo variable mensual ($)", min_value=0.0, step=10.0, value=st.session_state.get("costo_variable_mensual", defaults["costo_variable_mensual"]))
    st.session_state["costo_variable_mensual"] = costo_variable_mensual_val
with p4:
    tasa_desc_anual_pct_val = st.number_input("Tasa de descuento anual (%)", min_value=0.0, step=0.5, value=st.session_state.get("tasa_desc_anual_pct", defaults["tasa_desc_anual_pct"]))
    st.session_state["tasa_desc_anual_pct"] = tasa_desc_anual_pct_val
    st.caption("Si > 0, los flujos se descuentan a valor presente.")

# =========================
# 3) Cálculos (ROI, break-even)
# =========================
def supervivencia_mes(m, churn_pct):
    return 1.0 if churn_pct <= 0 else (1 - churn_pct/100) ** (m-1)

def factor_descuento(m, tasa_anual_pct):
    if tasa_anual_pct <= 0: return 1.0
    tasa_mensual = (1 + tasa_anual_pct/100) ** (1/12) - 1
    return 1.0 / ((1 + tasa_mensual) ** m)

def factor_sumatorio(meses, churn_pct, tasa_anual_pct, meses_gratis):
    return sum(
        supervivencia_mes(m, churn_pct) * factor_descuento(m, tasa_anual_pct)
        for m in range(1, meses+1) if m > meses_gratis
    )

precio_objetivo = None
if CAC > 0:
    F = factor_sumatorio(st.session_state["meses_plan"], st.session_state["churn_mensual_pct"], st.session_state["tasa_desc_anual_pct"], st.session_state["meses_gratis"])
    if F > 0:
        neto_requerido = CAC / F
        precio_objetivo = neto_requerido + st.session_state["costo_variable_mensual"]

precio_efectivo = precio_objetivo if (st.session_state["auto_precio"] and precio_objetivo is not None) else st.session_state["precio_mensual"]
neto_mensual = max(0.0, precio_efectivo - st.session_state["costo_variable_mensual"])

ingresos_acum, ingresos_acum_desc, utilidad_acum, utilidad_acum_desc = 0.0, 0.0, -CAC, -CAC
serie_utilidad, serie_utilidad_desc, break_even_mes = [], [], None

for m in range(1, st.session_state["meses_plan"] + 1):
    ingreso_mes = 0.0 if m <= st.session_state["meses_gratis"] else neto_mensual
    ingreso_mes_aj = ingreso_mes * supervivencia_mes(m, st.session_state["churn_mensual_pct"])

    ingresos_acum += ingreso_mes_aj
    utilidad_acum += ingreso_mes_aj
    serie_utilidad.append(utilidad_acum)

    fd = factor_descuento(m, st.session_state["tasa_desc_anual_pct"])
    ingresos_acum_desc += ingreso_mes_aj * fd
    utilidad_acum_desc += ingreso_mes_aj * fd
    serie_utilidad_desc.append(utilidad_acum_desc)

    if break_even_mes is None and utilidad_acum >= 0:
        break_even_mes = m

roi = (ingresos_acum - CAC) / CAC if CAC > 0 else None
roi_desc = (ingresos_acum_desc - CAC) / CAC if CAC > 0 else None

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("CAC por cliente", f"${CAC:,.2f}")
m2.metric("Ingreso acumulado (esperado)", f"${ingresos_acum:,.2f}")
m3.metric("Break-even (mes)", f"{break_even_mes}" if break_even_mes else "No se recupera")
m4.metric("ROI (sin descuento)", f"{roi*100:,.1f}%" if roi is not None else "N/A")
if st.session_state["auto_precio"] and precio_objetivo is not None:
    m5.metric("Precio objetivo (romper CAC)", f"${precio_objetivo:,.2f}")
else:
    m5.metric("ROI (NPV)", f"{roi_desc*100:,.1f}%" if roi_desc is not None else "N/A")

st.caption("""
- Ingreso neto mensual = Precio mensual - Costo variable mensual  
- Supervivencia(m) = (1 - churn%)^(m-1)  
- Ingreso ajustado(m) = Ingreso neto × Supervivencia(m)  (si m > meses gratis, si no 0)  
- Factor descuento(m) = \\( \\frac{1}{(1 + tasa\\_mensual)^m} \\)  
- Utilidad acumulada = \\( \\sum Ingreso\\ ajustado(m) - CAC \\)  
- ROI = \\( \\frac{Ingreso\\ acumulado - CAC}{CAC} \\)  
- ROI (NPV) = \\( \\frac{Ingreso\\ acumulado\\ descontado - CAC}{CAC} \\)
""")

df = pd.DataFrame({
    "Mes": list(range(1, st.session_state["meses_plan"] + 1)),
    "Utilidad acumulada": serie_utilidad,
    "Utilidad acumulada (NPV)": serie_utilidad_desc
}).set_index("Mes")
st.line_chart(df, use_container_width=True, height=280)

# =========================
# Debug table de cálculos mensuales
# =========================
if debug_mode:
    meses_plan = st.session_state["meses_plan"]
    meses_gratis = st.session_state["meses_gratis"]
    churn_pct = st.session_state["churn_mensual_pct"]
    debug_rows = []
    for m in range(1, meses_plan + 1):
        ingreso_neto = neto_mensual if m > meses_gratis else 0
        supervivencia = supervivencia_mes(m, churn_pct)
        ingreso_ajustado = ingreso_neto * supervivencia
        utilidad_acum = serie_utilidad[m - 1] if m - 1 < len(serie_utilidad) else None
        debug_rows.append({
            "Mes": m,
            "Ingreso neto": ingreso_neto,
            "Supervivencia": supervivencia,
            "Ingreso ajustado": ingreso_ajustado,
            "Utilidad acumulada": utilidad_acum
        })
    df_debug = pd.DataFrame(debug_rows)
    st.dataframe(df_debug)

# =========================
# Gemini API integration
# =========================
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

if "gemini_insight" not in st.session_state:
    st.session_state["gemini_insight"] = None
if "gemini_chat" not in st.session_state:
    st.session_state["gemini_chat"] = []

if st.button("Generar insight con Inteligencia Artificial"):
    try:
        # Collect parameters from session_state
        params = {k: st.session_state.get(k) for k in st.session_state.keys()}
        roi_text = f"{roi*100:.1f}%" if roi is not None else "N/A"
        roi_desc_text = f"{roi_desc*100:.1f}%" if roi_desc is not None else "N/A"
        # Build prompt with key metrics
        prompt = f"""
Eres un asistente experto en análisis financiero y marketing. Aquí están los parámetros y resultados clave de un análisis CAC:

Parámetros:
{json.dumps(params, indent=4)}

Resultados clave:
- CAC: ${CAC:,.2f}
- Ingreso acumulado esperado: ${ingresos_acum:,.2f}
- Break-even en mes: {break_even_mes if break_even_mes else 'No se recupera'}
- ROI sin descuento: {roi_text}
- ROI con descuento (NPV): {roi_desc_text}

Por favor, genera un insight útil y conciso sobre estos resultados para ayudar a la toma de decisiones.
"""
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        st.session_state["gemini_insight"] = response.text
        st.session_state["gemini_chat"] = []
    except Exception as e:
        st.error(f"Error al generar insight con Gemini: {e}")

if st.session_state["gemini_insight"] is not None:
    st.subheader("Insight interactivo")
    st.markdown(st.session_state["gemini_insight"])

    # Display chat history
    for chat in st.session_state["gemini_chat"]:
        st.markdown(f"**Pregunta:** {chat['Q']}")
        st.markdown(f"**Respuesta:** {chat['A']}")

    followup = st.text_input("Escribe una pregunta o inquietud adicional sobre el análisis:", key="followup_input")

    if st.button("Enviar pregunta a la IA"):
        if followup.strip():
            try:
                model = genai.GenerativeModel("gemini-2.0-flash")
                followup_prompt = f"El usuario quiere profundizar en el análisis anterior con esta pregunta: {followup}\n\nConsidera los parámetros y resultados anteriores para responder."
                response_followup = model.generate_content(followup_prompt)
                st.session_state["gemini_chat"].append({"Q": followup, "A": response_followup.text})
                st.rerun()
            except Exception as e:
                st.error(f"Error al generar respuesta de seguimiento con Gemini: {e}")
