import io
from datetime import date
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="IPS Horizonte | Management Intelligence", page_icon="🏥", layout="wide")

DATA_FILE = "IPS_Horizonte_M4T4_datos_v2.xlsx"

@st.cache_data
def load_data():
    x = pd.ExcelFile(DATA_FILE)
    return {s: pd.read_excel(DATA_FILE, sheet_name=s) for s in x.sheet_names}

try:
    D = load_data()
except Exception as e:
    st.error(f"No se pudo cargar el archivo de datos: {e}")
    st.info("Verifica que IPS_Horizonte_M4T4_datos_v2.xlsx esté en la misma carpeta que app.py.")
    st.stop()

OPS = D["Operaciones_semanal"].copy()
HR = D["Talento_humano"].copy()
INV = D["Inventarios"].copy()
EQ = D["Equipos_biomedicos"].copy()
AUD = D["Auditoria_registros"].copy()
QUALITY = D["Control_calidad"].copy()

st.markdown("""
<style>
    .block-container {padding-top: 1.35rem; padding-bottom: 2rem; max-width: 1500px;}
    [data-testid="stMetric"] {background: #f4f7fb; border: 1px solid #e4eaf2; padding: 14px 16px; border-radius: 12px;}
    .small-note {color:#667085; font-size:0.88rem;}
    .section-kicker {color:#52708f; font-size:0.78rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase;}
    div[data-testid="stTabs"] button {font-weight:600;}
</style>
""", unsafe_allow_html=True)

def pct(n, d):
    return (n / d * 100) if d else 0

def metric_row(items):
    cols = st.columns(len(items))
    for c, (label, value, help_text) in zip(cols, items):
        c.metric(label, value, help=help_text)

def note(text):
    st.caption(text)

def chart(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=45, b=10), template="plotly_white",
                      font=dict(family="Arial", size=12), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

def df_view(df, height=330):
    st.dataframe(df, use_container_width=True, hide_index=True, height=height)

def export_csv(df, filename, label="Descargar datos"):
    st.download_button(label, df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=filename, mime="text/csv")

# Sidebar / navigation
st.sidebar.title("IPS Horizonte")
st.sidebar.caption("Centro de inteligencia gerencial")
page = st.sidebar.radio("Navegación", [
    "Visión institucional", "Demanda y capacidad", "Talento humano",
    "Inventarios", "Equipos biomédicos", "Calidad y auditoría",
    "Laboratorio de escenarios", "Decisiones y calidad de datos"
])
st.sidebar.divider()
st.sidebar.caption("Datos sintéticos para ejercicio académico · No representan una IPS real.")

st.title("Centro de inteligencia gerencial")
st.markdown("**De los indicadores a la decisión gerencial**")
st.caption("IPS Horizonte · Laboratorio de investigación y análisis")

if page == "Visión institucional":
    st.markdown('<div class="section-kicker">Vista ejecutiva</div>', unsafe_allow_html=True)
    st.subheader("¿Qué señales debería revisar la gerencia?")
    st.warning("Los indicadores siguientes provienen de fuentes y periodos distintos. Son señales institucionales independientes; no constituyen una fotografía consolidada de un mismo periodo.")
    attendance = pct(OPS.citas_atendidas.sum(), OPS.citas_asignadas.sum())
    util = pct(OPS.horas_utilizadas.sum(), OPS.horas_disponibles.sum())
    completeness = pct(AUD.registro_completo_basico.eq("Sí").sum(), len(AUD))
    out_of_service = int(EQ.Estado.eq("Fuera de servicio").sum())
    overtime = HR.Horas_extra.sum(min_count=1)
    metric_row([
        ("Atención / asignación", f"{attendance:.1f}%", "Operaciones: 8 semanas, 3 especialidades."),
        ("Utilización de horas", f"{util:.1f}%", "Horas utilizadas / horas disponibles; operaciones semanales."),
        ("Horas extra registradas", f"{overtime:,.0f}", "Talento humano: 12 meses; dos registros faltantes no se imputan."),
        ("Equipos fuera de servicio", f"{out_of_service}", "Inventario actual de equipos; no es una serie histórica."),
        ("Registros completos básicos", f"{completeness:.1f}%", "100 registros; definición compuesta descrita en Calidad y auditoría.")
    ])
    st.subheader("Señales para investigar")
    a, b = st.columns(2)
    with a:
        st.markdown("**Operación por especialidad**")
        g = OPS.groupby("especialidad", as_index=False).agg(
            asignadas=("citas_asignadas","sum"), atendidas=("citas_atendidas","sum"),
            horas_disp=("horas_disponibles","sum"), horas_uso=("horas_utilizadas","sum"))
        g["Asistencia (%)"] = (g.atendidas/g.asignadas*100).round(1)
        g["Utilización (%)"] = (g.horas_uso/g.horas_disp*100).round(1)
        chart(px.bar(g, x="especialidad", y="Asistencia (%)", text="Asistencia (%)",
                     title="Asistencia sobre citas asignadas"))
    with b:
        st.markdown("**Estado de equipos**")
        eqc = EQ.groupby(["Estado"], as_index=False).size()
        chart(px.pie(eqc, names="Estado", values="size", hole=.55, title="Equipos por estado"))
    st.subheader("Lectura gerencial")
    st.markdown("""
    - La asistencia y la utilización describen la operación ambulatoria observada en ocho semanas.
    - Las horas extra y las ausencias pertenecen a una fuente mensual de talento humano.
    - El estado de equipos es un corte de inventario; las fallas se cuentan en una ventana móvil de 90 días.
    - La completitud documental se calcula sobre registros individuales de auditoría.

    **No se deben sumar ni cruzar estas señales como si compartieran el mismo periodo, servicio o unidad de observación.**
    """)

elif page == "Demanda y capacidad":
    st.markdown('<div class="section-kicker">Pregunta 1 · Operaciones</div>', unsafe_allow_html=True)
    st.subheader("¿Dónde están las brechas entre demanda y capacidad?")
    specialties = ["Todas"] + sorted(OPS.especialidad.dropna().unique().tolist())
    selected = st.selectbox("Especialidad", specialties)
    x = OPS if selected == "Todas" else OPS[OPS.especialidad == selected]
    att = pct(x.citas_atendidas.sum(), x.citas_asignadas.sum())
    no_show = pct(x.inasistencias.sum(), x.citas_asignadas.sum())
    cancel = pct(x.cancelaciones.sum(), x.citas_asignadas.sum())
    utilization = pct(x.horas_utilizadas.sum(), x.horas_disponibles.sum())
    metric_row([
        ("Citas asignadas", f"{x.citas_asignadas.sum():,.0f}", "Suma del periodo seleccionado."),
        ("Atención / asignación", f"{att:.1f}%", "Citas atendidas / asignadas."),
        ("Inasistencia", f"{no_show:.1f}%", "Inasistencias / citas asignadas."),
        ("Cancelación", f"{cancel:.1f}%", "Cancelaciones / citas asignadas."),
        ("Utilización", f"{utilization:.1f}%", "Horas utilizadas / disponibles.")
    ])
    left, right = st.columns(2)
    with left:
        chart(px.line(x, x="semana", y=["solicitudes","citas_asignadas","citas_atendidas"],
                      markers=True, title="Demanda, asignación y atención por semana",
                      labels={"value":"Citas","semana":"Semana","variable":"Serie"}))
    with right:
        chart(px.line(x, x="semana", y=["horas_disponibles","horas_utilizadas"],
                      markers=True, title="Horas disponibles vs. utilizadas",
                      labels={"value":"Horas","semana":"Semana","variable":"Serie"}))
    st.markdown("**Detalle semanal**")
    df_view(x[["semana","especialidad","solicitudes","citas_asignadas","citas_atendidas",
               "cancelaciones","inasistencias","tasa_asistencia_pct","tasa_inasistencia_pct",
               "tasa_cancelacion_pct","horas_disponibles","horas_utilizadas","utilizacion_pct"]])
    note("Cobertura: semanas 1–8; no hay fechas calendario. No extrapolar tendencias a meses ni a toda la institución.")
    export_csv(x, "operaciones_filtradas.csv")

elif page == "Talento humano":
    st.markdown('<div class="section-kicker">Pregunta 2 · Recursos</div>', unsafe_allow_html=True)
    st.subheader("¿Cómo se distribuyen las horas y las señales de carga laboral?")
    months = ["Todos"] + sorted(HR.Mes.dropna().unique().tolist())
    services = ["Todos"] + sorted(HR.Servicio.dropna().unique().tolist())
    c1,c2,c3 = st.columns(3)
    m = c1.selectbox("Mes", months)
    s = c2.selectbox("Servicio", services)
    t = c3.selectbox("Turno", ["Todos"] + sorted(HR.Turno.dropna().unique().tolist()))
    x = HR.copy()
    if m != "Todos": x = x[x.Mes == m]
    if s != "Todos": x = x[x.Servicio == s]
    if t != "Todos": x = x[x.Turno == t]
    extra = x.Horas_extra.sum(min_count=1)
    metric_row([
        ("Horas programadas", f"{x.Horas_programadas.sum():,.0f}", "Suma de registros trabajador-mes."),
        ("Horas trabajadas", f"{x.Horas_trabajadas.sum():,.0f}", "Suma de registros trabajador-mes."),
        ("Horas extra", f"{extra:,.0f}" if pd.notna(extra) else "Sin dato", "Los valores faltantes no se convierten en cero."),
        ("Días de ausencia", f"{x.Ausencias_dias.sum():,.0f}", "Incluye un valor atípico pendiente de validar.")
    ])
    monthly = x.groupby("Mes", as_index=False).agg(
        programadas=("Horas_programadas","sum"), trabajadas=("Horas_trabajadas","sum"),
        extras=("Horas_extra","sum"))
    l,r = st.columns(2)
    with l:
        chart(px.bar(monthly, x="Mes", y=["programadas","trabajadas"], barmode="group",
                     title="Horas programadas y trabajadas por mes"))
    with r:
        chart(px.bar(monthly, x="Mes", y="extras", title="Horas extra registradas por mes"))
    st.markdown("**Registros que requieren validación**")
    flags = x[x.alerta_ausencias_atipicas | x.alerta_horas_extra_faltantes]
    df_view(flags[["Mes","Servicio","Turno","ID_trabajador","Cargo","Ausencias_dias","Horas_extra",
                   "alerta_ausencias_atipicas","alerta_horas_extra_faltantes"]])
    note("Granularidad: trabajador × mes × servicio × turno. No interpretar la suma de horas como número de trabajadores únicos.")
    export_csv(x, "talento_humano_filtrado.csv")

elif page == "Inventarios":
    st.markdown('<div class="section-kicker">Pregunta 3 · Abastecimiento</div>', unsafe_allow_html=True)
    st.subheader("¿Qué insumos requieren seguimiento?")
    items = ["Todos"] + sorted(INV.Insumo.dropna().unique().tolist())
    i = st.selectbox("Insumo", items)
    x = INV if i == "Todos" else INV[INV.Insumo == i]
    valid = x[x.Consumo_mensual > 0]
    metric_row([
        ("Registros", f"{len(x)}", "Filas mensuales por insumo."),
        ("Consumo promedio válido", f"{valid.Consumo_mensual.mean():,.0f}" if len(valid) else "Sin dato", "Excluye consumo negativo solo de este cálculo; el dato original se conserva."),
        ("Existencias promedio", f"{x.Existencias.mean():,.0f}" if x.Existencias.notna().any() else "Sin dato", "Ignora nulos; no los imputa."),
        ("Alertas de calidad", f"{int((x.alerta_consumo_negativo | x.alerta_lead_time_atipico | x.alerta_existencias_faltantes).sum())}", "Conteo de filas con al menos una alerta.")
    ])
    l,r = st.columns(2)
    with l:
        chart(px.line(x, x="Fecha", y=["Consumo_mensual","Existencias","Compras"],
                      markers=True, title="Consumo, existencias y compras",
                      labels={"value":"Unidades","Fecha":"Mes","variable":"Serie"}))
    with r:
        chart(px.bar(x, x="Fecha", y="ratio_existencias_consumo", title="Relación existencias / consumo positivo",
                     labels={"ratio_existencias_consumo":"Ratio","Fecha":"Mes"}))
    st.markdown("**Excepciones que deben validarse**")
    flags = x[x.alerta_consumo_negativo | x.alerta_lead_time_atipico | x.alerta_existencias_faltantes]
    df_view(flags[["Fecha","Insumo","Consumo_mensual","Existencias","Lead_time_dias",
                   "alerta_consumo_negativo","alerta_lead_time_atipico","alerta_existencias_faltantes"]])
    note("No hay dimensión de servicio. El ratio existencias/consumo es descriptivo; no se interpreta como cobertura futura ni predicción de quiebre.")
    export_csv(x, "inventarios_filtrados.csv")

elif page == "Equipos biomédicos":
    st.markdown('<div class="section-kicker">Pregunta 3 · Tecnología clínica</div>', unsafe_allow_html=True)
    st.subheader("¿Qué equipos concentran señales de riesgo?")
    c1,c2 = st.columns(2)
    svc = c1.selectbox("Servicio", ["Todos"] + sorted(EQ.Servicio.dropna().unique().tolist()))
    state = c2.selectbox("Estado", ["Todos"] + sorted(EQ.Estado.dropna().unique().tolist()))
    x = EQ.copy()
    if svc != "Todos": x = x[x.Servicio == svc]
    if state != "Todos": x = x[x.Estado == state]
    metric_row([
        ("Equipos", f"{len(x)}", "Inventario de equipos."),
        ("Fuera de servicio", f"{x.Estado.eq('Fuera de servicio').sum()}", "Estado reportado en el corte."),
        ("Fallas últimos 90 días", f"{x.Fallas_90_dias.sum():,.0f}", "Ventana móvil reportada por fuente."),
        ("Horas indisponibilidad", f"{x.Horas_indisponibilidad.sum():,.0f}", "Incluye un valor atípico pendiente de validar.")
    ])
    l,r = st.columns(2)
    with l:
        chart(px.bar(x.groupby("Servicio",as_index=False).agg(equipos=("ID_equipo","count"),
                     fallas=("Fallas_90_dias","sum")), x="Servicio", y="fallas",
                     title="Fallas reportadas por servicio"))
    with r:
        chart(px.scatter(x, x="Fallas_90_dias", y="Horas_indisponibilidad",
                         color="Criticidad", symbol="Estado", hover_name="ID_equipo",
                         title="Fallas vs. horas de indisponibilidad"))
    st.markdown("**Equipos para revisión**")
    df_view(x[(x.Estado == "Fuera de servicio") | x.alerta_indisponibilidad_atipica | x.alerta_criticidad_faltante])
    note("Es un inventario de corte, no una serie histórica. Las 140 horas de EQ-040 requieren validar ventana temporal y unidad.")
    export_csv(x, "equipos_filtrados.csv")

elif page == "Calidad y auditoría":
    st.markdown('<div class="section-kicker">Pregunta 4 · Calidad</div>', unsafe_allow_html=True)
    st.subheader("¿Qué componentes documentales muestran oportunidades de mejora?")
    services = ["Todos"] + sorted(AUD.Servicio.dropna().unique().tolist())
    s = st.selectbox("Servicio", services)
    x = AUD if s == "Todos" else AUD[AUD.Servicio == s]
    metric_row([
        ("Registros", f"{len(x)}", "Unidad de análisis: registro asistencial."),
        ("Completos básicos", f"{pct(x.registro_completo_basico.eq('Sí').sum(),len(x)):.1f}%", "Consentimiento + firma + diagnóstico + plan de alta."),
        ("Con observación previa", f"{x.Auditoría_previa.eq('Observación').sum()}", "Clasificación de auditoría previa."),
        ("Eventos adversos reportados", f"{x.Evento_adverso.eq('Sí').sum()}", "Conteo, no tasa de incidencia poblacional.")
    ])
    components = pd.DataFrame({
        "Componente":["Consentimiento presente","Firma profesional","Valoración del dolor","Diagnóstico registrado","Plan de alta registrado"],
        "Cumplimiento":[
            pct(x.Consentimiento.eq("Presente").sum(),len(x)),
            pct(x.Firma_profesional.eq("Sí").sum(),len(x)),
            pct(x["Valoración_dolor"].eq("Sí").sum(),x["Valoración_dolor"].notna().sum()),
            pct(x.Diagnóstico.eq("Registrado").sum(),len(x)),
            pct(x.Plan_alta.eq("Registrado").sum(),len(x))
        ]
    })
    chart(px.bar(components, x="Componente", y="Cumplimiento", text=components.Cumplimiento.round(1),
                 title="Cumplimiento por componente documental", range_y=[0,105]))
    st.markdown("**Registros con observación o información faltante**")
    df_view(x[(x.Auditoría_previa == "Observación") | x.alerta_fecha_faltante | x.alerta_valoracion_faltante])
    note("El indicador de completitud básica no incluye valoración del dolor. Un dato faltante no equivale a incumplimiento.")
    export_csv(x, "auditoria_filtrada.csv")

elif page == "Laboratorio de escenarios":
    st.markdown('<div class="section-kicker">Simulación gerencial</div>', unsafe_allow_html=True)
    st.subheader("¿Qué pasaría si cambia la demanda o la capacidad?")
    st.write("Selecciona una especialidad y modifica supuestos para observar cómo cambiaría la utilización de horas.")
    specialty = st.selectbox("Especialidad", sorted(OPS.especialidad.unique().tolist()))
    x = OPS[OPS.especialidad == specialty]
    baseline_hours = x.horas_utilizadas.sum() / len(x)
    baseline_capacity = x.horas_disponibles.sum() / len(x)
    demand_change = st.slider("Cambio hipotético de demanda (%)", -20, 40, 15, 5)
    capacity_change = st.slider("Cambio hipotético de horas disponibles (%)", -20, 30, 0, 5)
    demand_hours = baseline_hours * (1 + demand_change/100)
    new_capacity = baseline_capacity * (1 + capacity_change/100)
    new_util = pct(demand_hours, new_capacity)
    old_util = pct(baseline_hours, baseline_capacity)
    metric_row([
        ("Utilización base", f"{old_util:.1f}%", "Promedio de horas utilizadas / disponibles en las ocho semanas."),
        ("Utilización escenario", f"{new_util:.1f}%", "Supone que las horas utilizadas varían proporcionalmente con la demanda."),
        ("Cambio", f"{new_util-old_util:+.1f} pp", "Diferencia en puntos porcentuales.")
    ])
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Base", x=["Utilización"], y=[old_util]))
    fig.add_trace(go.Bar(name="Escenario", x=["Utilización"], y=[new_util]))
    fig.update_layout(barmode="group", title="Comparación de utilización", yaxis_title="Porcentaje", yaxis=dict(range=[0,max(120,new_util+10)]))
    chart(fig)
    if new_util > 100:
        st.error("El escenario supera el 100% de las horas disponibles: la capacidad modelada no cubriría la demanda bajo estos supuestos.")
    elif new_util > 85:
        st.warning("El escenario deja un margen reducido de capacidad. Conviene revisar turnos, duración de atención y variabilidad.")
    else:
        st.success("El escenario permanece por debajo del umbral de utilización del 85% usado aquí como referencia didáctica, no como estándar clínico.")
    st.info("Modelo simplificado: se asume una relación proporcional entre demanda y horas utilizadas. No es pronóstico, optimización de turnos ni recomendación clínica.")

elif page == "Decisiones y calidad de datos":
    st.markdown('<div class="section-kicker">Gobernanza de la decisión</div>', unsafe_allow_html=True)
    st.subheader("De la señal a una acción verificable")
    st.write("Registra una decisión basada en evidencia. Los registros viven durante esta sesión; descarga el archivo para conservarlos.")
    if "decisions" not in st.session_state:
        st.session_state.decisions = []
    with st.form("decision_form", clear_on_submit=True):
        problem = st.text_input("Señal o problema observado")
        evidence = st.text_area("Evidencia disponible")
        hypothesis = st.text_area("Hipótesis por validar")
        action = st.text_area("Acción propuesta")
        responsible = st.text_input("Responsable")
        indicator = st.text_input("Indicador de seguimiento")
        expected = st.text_input("Resultado esperado")
        submitted = st.form_submit_button("Guardar decisión")
        if submitted:
            if problem and action and responsible and indicator:
                st.session_state.decisions.append({
                    "Fecha de registro": date.today().isoformat(), "Problema":problem,
                    "Evidencia":evidence, "Hipótesis":hypothesis, "Acción":action,
                    "Responsable":responsible, "Indicador":indicator, "Resultado esperado":expected
                })
                st.success("Decisión agregada a la sesión.")
            else:
                st.error("Completa problema, acción, responsable e indicador.")
    if st.session_state.decisions:
        dec = pd.DataFrame(st.session_state.decisions)
        df_view(dec, height=260)
        export_csv(dec, "registro_decisiones_ips_horizonte.csv", "Descargar registro de decisiones")
    else:
        st.caption("Aún no hay decisiones registradas en esta sesión.")
    st.divider()
    st.subheader("Control de calidad de las fuentes")
    st.write("Los valores atípicos y faltantes se conservan y se muestran como asuntos por validar. No se corrigen silenciosamente. Las banderas usan umbrales exploratorios didácticos (ausencias >10 días, lead time >60 días, indisponibilidad >100 horas), no estándares regulatorios ni clínicos.")
    df_view(QUALITY, height=360)
    st.download_button("Descargar control de calidad", QUALITY.to_csv(index=False).encode("utf-8-sig"),
                       file_name="control_calidad_fuentes.csv", mime="text/csv")
    st.subheader("Modelo de datos: límites de comparabilidad")
    df_view(D["Diccionario_modelo"], height=260)
    st.warning("No existe una llave común válida para unir las cinco fuentes en una tabla única. Los filtros y cálculos son específicos de cada fuente.")
