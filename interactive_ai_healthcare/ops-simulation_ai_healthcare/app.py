import streamlit as st
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="IPS Horizonte | Centro de operaciones", page_icon="🏥", layout="wide")

# -----------------------------
# Datos y estado de la simulación
# -----------------------------
def now():
    return datetime.now().strftime("%H:%M:%S")

def init_state():
    defaults = {
        "stock": 520,
        "daily_use": 100,
        "min_stock": 600,
        "reorder_point": 650,
        "safety_stock": 150,
        "lead_days": 5,
        "target_days": 14,
        "order_qty": 0,
        "order_status": "Sin solicitar",
        "supplier_delay": 0,
        "nurse_absence": True,
        "hr_status": "Sin resolver",
        "hr_choice": "",
        "equipment_status": "En servicio",
        "equipment_priority": "Monitor de hospitalización",
        "equipment_order": "Pendiente",
        "backup_available": False,
        "events": [],
        "log": [],
        "surprise_used": [],
        "interpretations": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def log_action(area, action, detail):
    st.session_state.log.insert(0, {
        "Hora": now(), "Área": area, "Acción": action, "Detalle": detail
    })

def add_event(area, title, description, severity="Atención"):
    st.session_state.events.insert(0, {
        "hora": now(), "area": area, "title": title,
        "description": description, "severity": severity
    })

init_state()

# -----------------------------
# Motor de reglas (determinístico)
# -----------------------------
def inventory_metrics():
    coverage = st.session_state.stock / max(st.session_state.daily_use, 1)
    arrival = max(st.session_state.lead_days + st.session_state.supplier_delay, 0)
    projected_at_arrival = max(st.session_state.stock - st.session_state.daily_use * arrival, 0)
    suggested = max(
        st.session_state.daily_use * st.session_state.target_days
        + st.session_state.safety_stock
        - st.session_state.stock, 0
    )
    return coverage, arrival, projected_at_arrival, suggested

def inventory_risk():
    coverage, arrival, projected, suggested = inventory_metrics()
    if coverage <= arrival:
        return "Crítico", "El inventario actual no cubre el consumo esperado hasta la llegada del proveedor."
    if st.session_state.stock <= st.session_state.min_stock:
        return "Alto", "Las existencias están por debajo del mínimo definido."
    if st.session_state.stock <= st.session_state.reorder_point:
        return "Preventivo", "Se alcanzó el punto de reorden."
    return "Controlado", "Las existencias están por encima del punto de reorden."

def interpret_text(area, message):
    """Interpretación demostrativa basada en palabras clave; no llama a un LLM."""
    t = message.lower()
    if area == "Inventario":
        result = {
            "tipo_evento": "Solicitud de reposición" if any(x in t for x in ["acaban", "agot", "reponer", "pedido", "más"]) else "Consulta de inventario",
            "insumo": "Guantes de nitrilo" if "guante" in t else "Insumo por identificar",
            "urgencia": "Alta" if any(x in t for x in ["urgente", "crítica", "acaban", "agot"]) else "Media"
        }
    elif area == "Talento humano":
        result = {
            "tipo_evento": "Ausencia / cambio de turno" if any(x in t for x in ["no puede", "ausencia", "incapacidad", "falta"]) else "Solicitud de cobertura",
            "turno": "Noche" if "noche" in t else ("Día" if "día" in t else "Por confirmar"),
            "servicio": "Urgencias" if "urgencia" in t else ("Hospitalización" if "hospital" in t else "Por confirmar"),
            "perfil": "Enfermería profesional" if "enfermer" in t else "Por confirmar"
        }
    else:
        result = {
            "tipo_evento": "Incidente de equipo" if any(x in t for x in ["reinició", "falla", "alarma", "daño"]) else "Orden de mantenimiento",
            "equipo": "Monitor" if "monitor" in t else "Equipo por identificar",
            "servicio": "Hospitalización" if "hospital" in t else "Por confirmar",
            "mantenimiento": "Vencido" if any(x in t for x in ["vencido", "pendiente", "atrasado"]) else "Por verificar"
        }
    st.session_state.interpretations[area] = result
    return result

# -----------------------------
# Estilos
# -----------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {background: #f7f9fc; border: 1px solid #e6eaf0; padding: 12px 14px; border-radius: 12px;}
.alert-card {padding: 14px 16px; border: 1px solid #e4e8ef; border-left: 5px solid #d97706; border-radius: 10px; background: #fff; margin-bottom: 10px;}
.small-muted {color: #64748b; font-size: 0.88rem;}
.status-pill {font-weight: 600;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Encabezado y navegación
# -----------------------------
st.title("🏥 Centro de operaciones | IPS Horizonte")
st.caption("Simulación operativa · Gestión de inventarios, talento humano y equipos · Datos sintéticos")
st.warning("Entorno de demostración. No usar para decisiones clínicas, laborales, compras reales ni mantenimiento real.")

sections = ["Centro de operaciones", "Inventario", "Talento humano", "Equipos", "Decisiones y bitácora"]
if "nav_target" not in st.session_state:
    st.session_state.nav_target = sections[0]
page = st.sidebar.radio("Navegación", sections, index=sections.index(st.session_state.nav_target))
st.session_state.nav_target = page
st.sidebar.divider()
st.sidebar.markdown("**Estado de la simulación**")
cov, arr, proj, sug = inventory_metrics()
st.sidebar.metric("Cobertura inventario", f"{cov:.1f} días")
st.sidebar.metric("Pedido", st.session_state.order_status)
st.sidebar.metric("Cobertura urgencias", "Pendiente" if st.session_state.nurse_absence and st.session_state.hr_status != "Aprobada" else "Gestionada")
if st.sidebar.button("↺ Reiniciar simulación", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()
    st.rerun()

# -----------------------------
# Centro de operaciones
# -----------------------------
if page == "Centro de operaciones":
    st.subheader("Bandeja de excepciones")
    st.write("La jornada comienza con tres eventos simultáneos. La aplicación mantiene el estado de cada caso mientras tomas decisiones.")
    coverage, arrival, projected, suggested = inventory_metrics()
    risk, risk_text = inventory_risk()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inventario disponible", f"{st.session_state.stock:,} uds.")
    c2.metric("Cobertura estimada", f"{coverage:.1f} días")
    c3.metric("Turno de urgencias", "Descubierto" if st.session_state.nurse_absence and st.session_state.hr_status != "Aprobada" else "Gestionado")
    c4.metric("Equipo biomédico", st.session_state.equipment_status)

    st.markdown("### Alertas activas")
    a, b, c = st.columns(3)
    with a:
        st.markdown(f"""<div class="alert-card"><b>🔴 Inventario · {risk}</b><br/>Guantes de nitrilo por debajo del mínimo.<br/><span class="small-muted">{risk_text}</span></div>""", unsafe_allow_html=True)
        st.caption(f"Stock: {st.session_state.stock:,} · Mínimo: {st.session_state.min_stock:,} · Entrega: {st.session_state.lead_days + st.session_state.supplier_delay} días")
        if st.button("Gestionar inventario", key="go_inv", use_container_width=True):
            st.session_state.nav_target = "Inventario"
            st.rerun()
    with b:
        st.markdown(f"""<div class="alert-card"><b>🟠 Talento humano · {'Atención' if st.session_state.nurse_absence and st.session_state.hr_status != 'Aprobada' else 'Gestionada'}</b><br/>Ausencia de enfermería en turno noche de urgencias.<br/><span class="small-muted">Se requiere validar perfil, descanso y cobertura del servicio de origen.</span></div>""", unsafe_allow_html=True)
        st.caption(f"Estado: {st.session_state.hr_status}")
        if st.button("Gestionar turnos", key="go_hr", use_container_width=True):
            st.session_state.nav_target = "Talento humano"
            st.rerun()
    with c:
        st.markdown(f"""<div class="alert-card"><b>🟠 Equipamiento · {'Atención' if st.session_state.equipment_order == 'Pendiente' else 'En seguimiento'}</b><br/>Monitor de hospitalización con mantenimiento vencido.<br/><span class="small-muted">La priorización técnica corresponde al área biomédica.</span></div>""", unsafe_allow_html=True)
        st.caption(f"Orden: {st.session_state.equipment_order} · Estado: {st.session_state.equipment_status}")
        if st.button("Gestionar equipos", key="go_eq", use_container_width=True):
            st.session_state.nav_target = "Equipos"
            st.rerun()

    st.markdown("### ¿Qué información falta para decidir?")
    st.write("- Inventario: consumo real, pedidos en tránsito y fecha confirmada de entrega.")
    st.write("- Talento: disponibilidad, competencias habilitadas, descansos y cobertura mínima.")
    st.write("- Equipos: criticidad, evaluación técnica y disponibilidad de respaldo.")
    st.info("Pregunta para el grupo: ¿cuál de las tres excepciones atenderían primero y qué dato necesitan antes de actuar?")

    st.markdown("### Introducir un evento operativo")
    event_type = st.selectbox("Evento sorpresa", [
        "El proveedor retrasa la entrega tres días",
        "Se reporta una segunda ausencia en el turno de noche",
        "Se reporta una falla en un equipo de criticidad alta"
    ])
    if st.button("Registrar nuevo evento", type="primary"):
        if event_type not in st.session_state.surprise_used:
            st.session_state.surprise_used.append(event_type)
            if event_type.startswith("El proveedor"):
                st.session_state.supplier_delay += 3
                add_event("Inventario", "Retraso del proveedor", "La entrega pasa de 5 a 8 días.", "Crítica")
                log_action("Inventario", "Evento registrado", "Retraso de proveedor +3 días")
            elif event_type.startswith("Se reporta una segunda"):
                st.session_state.nurse_absence = True
                st.session_state.hr_status = "Requiere nueva revisión"
                add_event("Talento humano", "Segunda ausencia", "Aumenta la brecha de cobertura en turno noche.", "Crítica")
                log_action("Talento humano", "Evento registrado", "Segunda ausencia en turno noche")
            else:
                add_event("Equipos", "Falla de equipo crítico", "Se reporta una falla adicional; requiere evaluación técnica.", "Crítica")
                st.session_state.equipment_order = "Escalado"
                log_action("Equipos", "Evento registrado", "Falla adicional escalada a ingeniería biomédica")
            st.success("Evento incorporado. Revisa cómo cambia el estado de la operación.")
            st.rerun()
        else:
            st.info("Este evento ya fue utilizado en esta simulación.")

    if st.session_state.events:
        st.markdown("### Historial de eventos")
        for ev in st.session_state.events:
            st.write(f"**{ev['hora']} · {ev['area']} · {ev['severity']}** — {ev['title']}: {ev['description']}")

# -----------------------------
# Inventario
# -----------------------------
elif page == "Inventario":
    st.subheader("1 · Gestión de inventario")
    st.write("Caso: guantes de nitrilo. El sistema calcula cobertura, riesgo y cantidad sugerida; tú decides cuánto solicitar.")
    with st.expander("Mensaje recibido · interpretación del evento", expanded=True):
        msg = st.text_area("Descripción del área asistencial", "Se están acabando los guantes, necesitamos más para esta semana", height=75)
        if st.button("Interpretar mensaje", key="interpret_inv"):
            result = interpret_text("Inventario", msg)
            add_event("Inventario", "Solicitud interpretada", msg, "Atención")
            log_action("Inventario", "Interpretación", str(result))
        if "Inventario" in st.session_state.interpretations:
            st.json(st.session_state.interpretations["Inventario"])
            st.caption("Demostración local por palabras clave; no es una llamada a un modelo generativo.")

    coverage, arrival, projected, suggested = inventory_metrics()
    risk, explanation = inventory_risk()
    x1, x2, x3, x4 = st.columns(4)
    x1.metric("Existencias", f"{st.session_state.stock:,} uds.")
    x2.metric("Consumo diario", f"{st.session_state.daily_use:,} uds.")
    x3.metric("Cobertura", f"{coverage:.1f} días")
    x4.metric("Riesgo", risk)

    st.markdown("#### Parámetros operativos")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.session_state.daily_use = st.number_input("Consumo diario (uds.)", 1, 10000, int(st.session_state.daily_use), 10)
        st.session_state.stock = st.number_input("Stock actual (uds.)", 0, 100000, int(st.session_state.stock), 10)
    with p2:
        st.session_state.min_stock = st.number_input("Stock mínimo", 0, 100000, int(st.session_state.min_stock), 10)
        st.session_state.reorder_point = st.number_input("Punto de reorden", 0, 100000, int(st.session_state.reorder_point), 10)
    with p3:
        st.session_state.lead_days = st.number_input("Tiempo proveedor (días)", 0, 90, int(st.session_state.lead_days))
        st.session_state.safety_stock = st.number_input("Stock de seguridad", 0, 100000, int(st.session_state.safety_stock), 10)
        st.session_state.target_days = st.number_input("Días objetivo de cobertura", 1, 90, int(st.session_state.target_days))

    coverage, arrival, projected, suggested = inventory_metrics()
    risk, explanation = inventory_risk()
    st.markdown("#### Resultado de las reglas")
    st.write(explanation)
    r1, r2, r3 = st.columns(3)
    r1.metric("Días hasta la entrega", f"{arrival}")
    r2.metric("Stock proyectado al llegar", f"{projected:,} uds.")
    r3.metric("Cantidad sugerida", f"{suggested:,} uds.")
    st.caption("Fórmula demostrativa: consumo diario × días objetivo + stock de seguridad − existencias actuales. No contempla pedidos en tránsito, vencimientos, lotes ni variabilidad de demanda.")

    qty = st.slider("Cantidad que decides solicitar", 0, 3000, min(int(suggested), 3000), 10)
    projected_after = projected + qty
    st.metric("Stock proyectado después de recibir el pedido", f"{projected_after:,} uds.")
    if qty > suggested:
        st.warning("La cantidad supera la sugerencia del modelo de reposición. Evalúa almacenamiento, vencimiento y presupuesto.")
    elif qty < suggested:
        st.warning("La cantidad está por debajo de la sugerencia. Podría no alcanzar el horizonte objetivo.")
    else:
        st.success("La cantidad coincide con la sugerencia calculada.")

    if st.button("Aprobar solicitud de compra", type="primary", disabled=(qty == 0)):
        st.session_state.order_qty = qty
        st.session_state.order_status = "Aprobada · pendiente de entrega"
        log_action("Inventario", "Compra aprobada", f"{qty:,} unidades; entrega estimada en {arrival} días.")
        add_event("Inventario", "Orden de compra aprobada", f"{qty:,} unidades solicitadas.", "Información")
        st.success("La orden quedó aprobada. El stock NO aumenta hasta que llegue el pedido.")
    if st.session_state.order_qty:
        st.info(f"Orden actual: {st.session_state.order_qty:,} unidades · {st.session_state.order_status}")
        if st.button("Simular recepción del pedido"):
            st.session_state.stock += st.session_state.order_qty
            log_action("Inventario", "Pedido recibido", f"Se suman {st.session_state.order_qty:,} unidades al stock.")
            st.session_state.order_status = "Recibida"
            st.session_state.order_qty = 0
            st.session_state.supplier_delay = 0
            st.success("Recepción simulada. Se actualizó el inventario.")
            st.rerun()

# -----------------------------
# Talento humano
# -----------------------------
elif page == "Talento humano":
    st.subheader("2 · Gestión de talento humano")
    st.write("Caso: ausencia de enfermería profesional en el turno noche de urgencias. Las alternativas se validan con restricciones simuladas.")
    msg = st.text_area("Reporte recibido", "Una enfermera avisó que mañana en la noche no puede venir; hay que cubrir urgencias", height=75, key="hr_msg")
    if st.button("Interpretar reporte", key="interpret_hr"):
        result = interpret_text("Talento humano", msg)
        log_action("Talento humano", "Interpretación", str(result))
        add_event("Talento humano", "Ausencia interpretada", msg, "Atención")
    if "Talento humano" in st.session_state.interpretations:
        st.json(st.session_state.interpretations["Talento humano"])
        st.caption("Demostración local por palabras clave; no es una llamada a un modelo generativo.")

    st.markdown("#### Alternativas de cobertura")
    alternatives = {
        "A · Trasladar enfermera desde hospitalización": {
            "availability": True, "competency": True, "rest": True, "source_coverage": False,
            "reason": "Deja hospitalización por debajo de su cobertura mínima."
        },
        "B · Ofrecer turno extra a profesional de urgencias": {
            "availability": True, "competency": True, "rest": True, "source_coverage": True,
            "reason": "Cumple las restricciones configuradas en esta simulación."
        },
        "C · Asignar auxiliar sin habilitación requerida": {
            "availability": True, "competency": False, "rest": True, "source_coverage": True,
            "reason": "No cumple el perfil requerido para la cobertura."
        }
    }
    chosen = st.radio("Selecciona una alternativa para validar", list(alternatives.keys()))
    alt = alternatives[chosen]
    checks = [
        ("Disponibilidad", alt["availability"]),
        ("Competencia / habilitación", alt["competency"]),
        ("Descanso y jornada", alt["rest"]),
        ("Cobertura mínima del servicio de origen", alt["source_coverage"])
    ]
    df = pd.DataFrame([{"Restricción": n, "Resultado": "Cumple" if ok else "No cumple"} for n, ok in checks])
    st.dataframe(df, use_container_width=True, hide_index=True)
    valid = all(v for _, v in checks)
    if valid:
        st.success("Alternativa aprobable según las reglas ficticias de este caso.")
    else:
        st.error(f"Alternativa no aprobable: {alt['reason']}")
    if st.button("Aprobar alternativa seleccionada", type="primary", disabled=not valid):
        st.session_state.hr_status = "Aprobada"
        st.session_state.hr_choice = chosen
        st.session_state.nurse_absence = False
        log_action("Talento humano", "Cobertura aprobada", chosen)
        add_event("Talento humano", "Turno cubierto", chosen, "Información")
        st.success("La cobertura quedó registrada en la simulación.")
    st.info(f"Estado actual: {st.session_state.hr_status}" + (f" · {st.session_state.hr_choice}" if st.session_state.hr_choice else ""))
    st.caption("En una implementación real, las reglas laborales, habilitaciones, descansos y mínimos deben ser definidas y validadas por la institución. Esta simulación no sustituye la revisión de RR. HH. ni la normativa aplicable.")

# -----------------------------
# Equipos
# -----------------------------
elif page == "Equipos":
    st.subheader("3 · Gestión de equipos biomédicos")
    st.write("Caso: monitor de hospitalización con reinicios reportados y mantenimiento vencido. La aplicación prioriza la revisión; no realiza diagnóstico técnico.")
    msg = st.text_area("Reporte del servicio", "El monitor de hospitalización se reinició dos veces y ya tenía el mantenimiento vencido", height=75, key="eq_msg")
    if st.button("Interpretar reporte", key="interpret_eq"):
        result = interpret_text("Equipos", msg)
        log_action("Equipos", "Interpretación", str(result))
        add_event("Equipos", "Incidente interpretado", msg, "Atención")
    if "Equipos" in st.session_state.interpretations:
        st.json(st.session_state.interpretations["Equipos"])
        st.caption("Demostración local por palabras clave; no es una llamada a un modelo generativo.")

    equipment = pd.DataFrame([
        {"Equipo": "Monitor · Hospitalización", "Criticidad": "Alta", "Mantenimiento": "Vencido", "Respaldo": "No disponible", "Prioridad": 1},
        {"Equipo": "Bomba de infusión · UCI", "Criticidad": "Media", "Mantenimiento": "Próximo", "Respaldo": "Disponible", "Prioridad": 2},
        {"Equipo": "Ecógrafo · Consulta externa", "Criticidad": "Baja", "Mantenimiento": "Al día", "Respaldo": "Disponible", "Prioridad": 3},
    ])
    st.dataframe(equipment, use_container_width=True, hide_index=True)
    st.markdown("#### Intervención operativa")
    selected = st.selectbox("Equipo a priorizar", equipment["Equipo"].tolist(), index=0)
    backup = st.checkbox("Confirmar que existe equipo de respaldo disponible", value=st.session_state.backup_available)
    status = st.selectbox("Estado de la orden", ["Pendiente", "Asignada a ingeniería biomédica", "En evaluación técnica", "Completada"], index=["Pendiente", "Asignada a ingeniería biomédica", "En evaluación técnica", "Completada"].index(st.session_state.equipment_order) if st.session_state.equipment_order in ["Pendiente", "Asignada a ingeniería biomédica", "En evaluación técnica", "Completada"] else 0)
    if st.button("Registrar decisión de mantenimiento", type="primary"):
        st.session_state.equipment_priority = selected
        st.session_state.backup_available = backup
        st.session_state.equipment_order = status
        st.session_state.equipment_status = "Fuera de servicio" if not backup and status != "Completada" else ("Disponible" if status == "Completada" else "En seguimiento")
        log_action("Equipos", "Orden actualizada", f"{selected} · {status} · respaldo: {'sí' if backup else 'no'}")
        add_event("Equipos", "Orden de mantenimiento actualizada", f"{selected}: {status}", "Información")
        st.success("La decisión quedó registrada.")
    st.metric("Estado operativo simulado", st.session_state.equipment_status)
    st.caption("La criticidad y la prioridad son datos ficticios. La evaluación, aislamiento, reparación y liberación del equipo corresponden a personal biomédico autorizado.")

# -----------------------------
# Decisiones y bitácora
# -----------------------------
elif page == "Decisiones y bitácora":
    st.subheader("4 · Decisiones y bitácora")
    st.write("Esta pantalla muestra cómo la operación conserva el rastro de las decisiones y sus cambios de estado.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Orden inventario", st.session_state.order_status)
    col2.metric("Cobertura de turno", st.session_state.hr_status)
    col3.metric("Mantenimiento", st.session_state.equipment_order)

    if st.session_state.log:
        st.markdown("#### Registro de decisiones")
        st.dataframe(pd.DataFrame(st.session_state.log), use_container_width=True, hide_index=True)
    else:
        st.info("Todavía no hay decisiones registradas. Ve a Inventario, Talento humano o Equipos y ejecuta una acción.")

    st.markdown("#### Indicadores antes / después")
    coverage, arrival, projected, suggested = inventory_metrics()
    inv_after = f"{st.session_state.stock:,} uds. actuales"
    hr_after = "Cubierto" if not st.session_state.nurse_absence or st.session_state.hr_status == "Aprobada" else "Pendiente"
    eq_after = st.session_state.equipment_status
    comparison = pd.DataFrame([
        {"Indicador": "Cobertura inventario", "Inicio": "5.2 días", "Estado actual": f"{coverage:.1f} días"},
        {"Indicador": "Cobertura urgencias noche", "Inicio": "Descubierta", "Estado actual": hr_after},
        {"Indicador": "Monitor hospitalización", "Inicio": "Mantenimiento vencido", "Estado actual": eq_after},
        {"Indicador": "Orden de compra", "Inicio": "Sin solicitar", "Estado actual": st.session_state.order_status},
    ])
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.caption("Los valores de inicio corresponden al escenario sintético definido en el caso. No son datos de una IPS real.")

st.divider()
st.caption("IPS Horizonte · Prototipo didáctico · Datos sintéticos · La IA se representa en esta versión mediante una interpretación local simulada; no hay conexión a un modelo externo.")
