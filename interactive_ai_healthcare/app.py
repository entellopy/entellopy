import streamlit as st

st.set_page_config(page_title="Laboratorio de IA aplicada a salud", page_icon="🩺", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.block-container{max-width:1180px;padding-top:1.25rem;padding-bottom:3rem}
.hero{background:linear-gradient(120deg,#E5F4FF 0%,#E1F7F0 58%,#FFF3D5 100%);border:1px solid #C7E3E7;border-radius:24px;padding:25px 30px;display:flex;gap:22px;align-items:center;box-shadow:0 8px 22px #244e6410}
.hero h1{color:#17476B;font-size:clamp(1.7rem,3vw,2.5rem);margin:0 0 7px}.hero p{color:#416173;font-size:1.05rem;margin:0}
.med-icon{width:104px;height:104px;flex:0 0 104px;border-radius:24px;background:#fff;display:flex;align-items:center;justify-content:center;box-shadow:0 5px 15px #244e6412}
.notice{background:#FFF6D9;border-left:6px solid #E6B33F;border-radius:10px;padding:13px 16px;color:#624C19;margin:14px 0 22px}
.step{border-radius:17px;padding:16px 20px;margin:17px 0 10px;border:1px solid #d9e8ed;background:#f7fbfd}
.kicker{font-size:.76rem;letter-spacing:.1em;font-weight:850;text-transform:uppercase;color:#12818a;margin-bottom:4px}
.stitle{font-size:1.35rem;font-weight:850;color:#17476b}.sub{color:#5c7180;margin-top:3px}
.partbox{border-radius:18px;padding:18px 20px;min-height:154px;border:2px solid transparent;margin-bottom:8px}
.partblue{background:#e8f3ff;border-color:#b7d8fa}.partgreen{background:#e2f7ee;border-color:#b8e7d1}
.selected-blue{outline:3px solid #2f80c9;box-shadow:0 5px 18px #2f80c922}.selected-green{outline:3px solid #138a8a;box-shadow:0 5px 18px #138a8a22}
.parttag{display:inline-block;border-radius:99px;padding:5px 10px;font-size:.78rem;font-weight:850;margin-bottom:9px}
.blue-tag{background:#cce5ff;color:#155d99}.green-tag{background:#c9efd9;color:#176d4b}
.partbox h3{margin:0 0 6px;color:#17476b;font-size:1.18rem}.partbox p{margin:0;color:#3f5c6d}
div[data-testid="stSelectbox"] label{font-weight:850!important;color:#17476b!important;font-size:1.03rem!important}
div[data-testid="stSelectbox"] div[data-baseweb="select"]>div{min-height:3.5rem;background:#eaf5ff;border:2px solid #58a5d8;border-radius:13px;font-size:1.08rem;font-weight:700}
div[data-testid="stRadio"]>label{font-weight:850;color:#17476b;font-size:1.02rem}
div[role="radiogroup"]{gap:12px;width:100%}
div[role="radiogroup"] label{flex:1;background:#e8f3ff;border:2px solid #b7d8fa;border-radius:15px;padding:13px 15px;min-height:62px;font-weight:800}
div[role="radiogroup"] label:has(input:checked){background:#d1eaff;border-color:#2f80c9;color:#164e82}
div[role="radiogroup"] label p{font-size:1.02rem!important}
.casebox{background:#f5faff;border:1px solid #cfe4f6;border-left:6px solid #2f80c9;border-radius:15px;padding:17px 19px}
.goalbox{background:#fff8e5;border:1px solid #f2dda0;border-left:6px solid #e6b33f;border-radius:15px;padding:17px 19px;height:100%}
.stage{border-radius:15px;padding:14px 17px;margin:8px 0;border:1px solid #d7e5ec}
.s1{background:#f2f8ff;border-left:6px solid #2f80c9}.s2{background:#effaf5;border-left:6px solid #138a8a}.s3{background:#fff9e9;border-left:6px solid #e6b33f}
.stage h3{margin:0;color:#17476b;font-size:1.08rem}.stage small{color:#5c7180}
div.stButton>button{border-radius:11px;font-weight:750;min-height:2.7rem}
@media(max-width:700px){.hero{padding:18px;gap:13px}.med-icon{width:72px;height:72px;flex-basis:72px}.med-icon svg{width:62px;height:62px}.hero h1{font-size:1.55rem}}
</style>
""", unsafe_allow_html=True)

medical_svg="""<svg viewBox="0 0 100 100" width="88" height="88" aria-label="Ilustración médica"><circle cx="50" cy="50" r="47" fill="#EAF5FF"/><path d="M29 18v23c0 11 7 18 17 18s17-7 17-18V18" fill="none" stroke="#2F80C9" stroke-width="5.5" stroke-linecap="round"/><path d="M23 18h12M57 18h12" stroke="#2F80C9" stroke-width="5.5" stroke-linecap="round"/><circle cx="29" cy="17" r="6" fill="#F07D62"/><circle cx="63" cy="17" r="6" fill="#F07D62"/><path d="M46 59v8c0 10 7 16 16 16h4" fill="none" stroke="#138A8A" stroke-width="5.5" stroke-linecap="round"/><circle cx="75" cy="83" r="9" fill="#fff" stroke="#138A8A" stroke-width="5"/><path d="M46 38h13M52.5 31.5v13" stroke="#fff" stroke-width="3.5" stroke-linecap="round"/></svg>"""
st.markdown(f'<div class="hero"><div class="med-icon">{medical_svg}</div><div><h1>Laboratorio de IA aplicada a salud</h1><p>Explora un caso, trabaja con IA y convierte los resultados en oportunidades de mejora.</p></div></div>',unsafe_allow_html=True)
st.markdown('<div class="notice">⚠️ <b>Privacidad y uso responsable:</b> trabaja únicamente con los casos ficticios de esta aplicación. No ingreses historias clínicas reales ni datos personales. Revisa críticamente cada resultado de IA.</div>',unsafe_allow_html=True)

CASES={
"Atención clínica y rehabilitación":{
"intro":"Documentación clínica, síntesis de información y comunicación clara.","title":"Preparar mejor una consulta","goal":"Organizar información sin perder detalles ni automatizar decisiones clínicas.","case":"En una consulta externa ficticia, una persona de 52 años consulta por dolor de rodilla de 3 días y refiere dolor al caminar. No se aportan examen físico, antecedentes completos ni plan.","data":"Persona de 52 años; dolor de rodilla desde hace 3 días; dolor al caminar. Examen físico, antecedentes completos y plan: no informados.","steps":[
("Comprender la situación","Resume los hechos explícitos, los datos faltantes y las preguntas que el profesional debe revisar antes de la consulta. No diagnostiques ni recomiendes tratamientos."),
("Encontrar una oportunidad","Usa el caso y el análisis anterior. Propón hasta 3 mejoras del proceso de preparación de la consulta. Para cada una indica evidencia, qué debe confirmarse y cómo evitar duplicar trabajo."),
("Diseñar una prueba","Usa el caso y las respuestas anteriores. Diseña un piloto de una semana para probar la mejora más viable. Incluye pasos, responsable funcional, indicador con fórmula y validación humana. No inventes metas ni línea base.")],"deliverable":"Flujo de preparación + piloto e indicador."},
"Aseguramiento y gestión del riesgo en salud (EPS)":{
"intro":"Seguimiento poblacional, continuidad de atención e indicadores de programas de riesgo.","title":"Encontrar brechas en el seguimiento de diabetes","goal":"Detectar oportunidades que una EPS pueda investigar y medir.","case":"Una EPS ficticia revisa el seguimiento trimestral de 200 personas con diabetes tipo 2. Existen registros de control, HbA1c, presión arterial y citas pendientes, pero no se conocen las causas de los registros faltantes.","data":"Población: 200; control registrado en el trimestre: 132; sin control registrado: 68; HbA1c registrada en últimos 6 meses: 118; presión arterial registrada en trimestre: 145; citas pendientes reportadas: 54. Los grupos pueden superponerse y no se ha confirmado si las citas son personas únicas.","steps":[
("Comprender los indicadores","Analiza los datos ficticios. Calcula porcentajes cuando el denominador esté claro y muestra la fórmula. Separa hechos, incertidumbres y datos faltantes. No asumas que un registro ausente significa que la atención no ocurrió."),
("Priorizar oportunidades","Usa el caso y el análisis anterior. Identifica hasta 3 oportunidades de mejora para investigar. Para cada una indica evidencia, hipótesis por validar y dato adicional necesario. No atribuyas causas ni inventes metas."),
("Diseñar una prueba","Usa el caso y las respuestas anteriores. Diseña un piloto de mejora con acción, población o proceso objetivo, responsable funcional, indicador con fórmula y frecuencia de seguimiento. Incluye validaciones previas; no recomiendes tratamientos individuales.")],"deliverable":"Indicadores interpretados + oportunidad priorizada + piloto medible."},
"Operaciones y administración de IPS":{
"intro":"Oportunidad, tiempos de espera, cancelaciones y mejora de procesos.","title":"Reducir tiempos de espera","goal":"Convertir quejas por demoras en un proceso medible de mejora.","case":"Una IPS ficticia recibe quejas por largas esperas en consulta externa. Solo tiene un resumen de un día; todavía no sabe si las demoras varían por especialidad, hora o tipo de cita.","data":"Un día ficticio: 30 citas; espera media: 42 minutos; mediana: 35 minutos; 8 citas superaron 60 minutos. No hay datos de otros días ni causas registradas.","steps":[
("Entender el problema","Resume los datos disponibles y calcula qué porcentaje de citas superó 60 minutos, mostrando la fórmula. Señala qué no se puede concluir con datos de un solo día."),
("Investigar oportunidades","Usa el caso y el análisis anterior. Propón hasta 3 hipótesis operativas comprobables. Para cada una indica qué datos se necesitan y cómo segmentarlos. No presentes hipótesis como causas confirmadas."),
("Probar una mejora","Usa el caso y las respuestas anteriores. Diseña un piloto de bajo riesgo con cambio a probar, responsable funcional, indicador y fórmula, periodo de medición e indicador de balance. No inventes metas ni prometas resultados.")],"deliverable":"Diagnóstico inicial + hipótesis + piloto operativo."},
"Auditoría, calidad y regulación":{
"intro":"Revisión documental, trazabilidad de hallazgos y verificación contra criterios.","title":"Hacer más consistente la revisión documental","goal":"Mejorar trazabilidad y reducir retrabajo sin emitir dictámenes automáticos.","case":"Un equipo auditor ficticio revisa registros y quiere reducir el tiempo dedicado a localizar faltantes. Para este ejercicio se usan dos criterios internos ficticios: debe aparecer fecha y firma del responsable.","data":"Registro A: fecha y firma visibles. Registro B: fecha visible, firma ilegible. Registro C: fecha no visible y copia no legible. Criterios ficticios; no se suministró normativa aplicable.","steps":[
("Revisar la evidencia","Compara los registros con los criterios ficticios de fecha y firma. Clasifica la evidencia como visible, ausente en la copia o no evaluable. No inventes normas ni declares incumplimiento definitivo."),
("Estandarizar la revisión","Usa el caso y la respuesta anterior. Propón un flujo de revisión consistente e identifica qué casos deben escalarse a una persona auditora. Distingue evidencia de interpretación."),
("Medir la mejora","Usa el caso y las respuestas anteriores. Diseña un piloto del flujo de revisión con responsable funcional, indicador de tiempo o consistencia con fórmula, muestra y paso de validación humana. No inventes línea base ni meta.")],"deliverable":"Matriz de revisión + flujo estandarizado + piloto."},
"Dirección, investigación y salud pública":{
"intro":"Síntesis de evidencia, investigación y apoyo a decisiones institucionales.","title":"Convertir un informe en opciones de mejora","goal":"Distinguir evidencia y limitaciones antes de recomendar una intervención.","case":"Un equipo directivo revisa una encuesta ficticia sobre acceso a citas. Necesita distinguir hallazgos, limitaciones y acciones por explorar sin presentar los resultados como representativos de toda la población.","data":"Encuesta ficticia por conveniencia: 120 participantes; 45 reportaron dificultades para conseguir cita. La muestra no es necesariamente representativa y no permite concluir causalidad.","steps":[
("Sintetizar la evidencia","Resume el hallazgo explícito, calcula el porcentaje de participantes que reportó dificultades y señala qué no puede afirmarse debido al muestreo por conveniencia. No agregues datos externos."),
("Explorar opciones","Usa el caso y la respuesta anterior. Propón hasta 3 opciones de investigación o mejora. Para cada una indica evidencia, incertidumbre y dato adicional necesario. No presentes opciones como decisiones aprobadas."),
("Definir el siguiente paso","Usa el caso y las respuestas anteriores. Diseña un siguiente paso de bajo riesgo con actividad, responsable funcional, indicador de avance y validaciones necesarias. Distingue recomendación preliminar de decisión institucional.")],"deliverable":"Síntesis de evidencia + opciones + siguiente paso medible."}
}
PARTS={
"PARTE 1 · EXPLORAR":{"title":"De una necesidad a un buen análisis","desc":"Comprende el contexto, distingue evidencia de suposiciones y encuentra oportunidades.","color":"blue"},
"PARTE 2 · APLICAR":{"title":"De los hallazgos a una mejora verificable","desc":"Usa los resultados previos para diseñar una prueba y decidir cómo medirla.","color":"green"}}

# Select area once, prominently, without a duplicate selector.
st.markdown('<div class="step"><div class="kicker">PASO 1</div><div class="stitle">Selecciona tu área de trabajo</div><div class="sub">Elige el contexto que mejor representa tu trabajo; el caso práctico se adaptará a tu selección.</div></div>',unsafe_allow_html=True)
area=st.selectbox("Área profesional",list(CASES.keys()),key="area")
case=CASES[area]
st.caption(case["intro"])

st.markdown('<div class="step"><div class="kicker">PASO 2</div><div class="stitle">Elige una ruta de aprendizaje</div><div class="sub">Ambas rutas trabajan el mismo caso mediante tres pasos conectados.</div></div>',unsafe_allow_html=True)
if "part" not in st.session_state:
    st.session_state.part = "PARTE 1 · EXPLORAR"
c1,c2=st.columns(2,gap="large")
with c1:
    selected = st.session_state.part == "PARTE 1 · EXPLORAR"
    cls = "partbox partblue selected-blue" if selected else "partbox partblue"
    badge = '<span class="selection-badge">✓ SELECCIONADA</span>' if selected else ''
    st.markdown(f'<div class="{cls}"><span class="parttag blue-tag">PARTE 1 · EXPLORAR</span>{badge}<h3>De una necesidad a un buen análisis</h3><p>Comprende el contexto, separa hechos de hipótesis y descubre oportunidades de mejora.</p></div>',unsafe_allow_html=True)
    if st.button("Seleccionar Parte 1", key="choose_part1", use_container_width=True, type="primary" if selected else "secondary"):
        st.session_state.part = "PARTE 1 · EXPLORAR"
        st.rerun()
with c2:
    selected = st.session_state.part == "PARTE 2 · APLICAR"
    cls = "partbox partgreen selected-green" if selected else "partbox partgreen"
    badge = '<span class="selection-badge">✓ SELECCIONADA</span>' if selected else ''
    st.markdown(f'<div class="{cls}"><span class="parttag green-tag">PARTE 2 · APLICAR</span>{badge}<h3>De los hallazgos a una mejora</h3><p>Prioriza una oportunidad y diseña cómo probarla y medirla.</p></div>',unsafe_allow_html=True)
    if st.button("Seleccionar Parte 2", key="choose_part2", use_container_width=True, type="primary" if selected else "secondary"):
        st.session_state.part = "PARTE 2 · APLICAR"
        st.rerun()
part=st.session_state.part

st.markdown('<div class="step"><div class="kicker">CASO PRÁCTICO</div><div class="stitle">'+case["title"]+'</div></div>',unsafe_allow_html=True)
a,b=st.columns([1.45,1],gap="large")
with a: st.markdown('<div class="casebox"><b style="color:#24679B">SITUACIÓN FICTICIA</b><p style="color:#36586B;margin:.6rem 0 0">'+case["case"]+'</p></div>',unsafe_allow_html=True)
with b: st.markdown('<div class="goalbox"><b style="color:#8A6415">OBJETIVO</b><p style="font-weight:800;color:#624C19">'+case["goal"]+'</p><p style="color:#765F2A;font-size:.92rem"><b>Producto final:</b> '+case["deliverable"]+'</p></div>',unsafe_allow_html=True)
with st.expander("Ver los datos del caso"): st.write(case["data"])

st.markdown('<div class="step"><div class="kicker">PASO 3</div><div class="stitle">Trabaja con 3 prompts conectados</div><div class="sub">La IA se ejecuta dentro de esta página. Cada paso recibe el caso y las respuestas anteriores; no tienes que copiar y pegar entre aplicaciones.</div></div>',unsafe_allow_html=True)

# No API or external model required. This is a free guided simulation with curated
# case-specific outputs. It preserves each step's output and uses it in later steps.
chain_id=f"{area}::{part}"
if chain_id not in st.session_state:
    st.session_state[chain_id]={"answers":["","",""],"done":[False,False,False]}
chain=st.session_state[chain_id]

if part.startswith("PARTE 1"):
    st.info("Parte 1: explora cómo pasar de entender un problema a identificar una oportunidad de mejora.")
else:
    st.success("Parte 2: aplica el análisis para estructurar una prueba pequeña y medir si mejora el proceso.")

# Curated model-free response templates. They are transparent examples, not live AI output.
def make_answer(idx, case_data, prior):
    name=area
    data=case_data["data"]
    if name.startswith("Atención clínica"):
        answers=[
            "HECHOS: persona de 52 años; dolor de rodilla de 3 días; dolor al caminar.\nDATOS FALTANTES: examen físico, antecedentes completos, medicamentos y plan.\nPREGUNTAS PARA REVISAR: evolución del dolor, limitación funcional y signos relevantes según criterio clínico. No se puede establecer un diagnóstico con la información disponible.",
            "OPORTUNIDADES A INVESTIGAR: (1) usar una plantilla breve para ordenar notas; (2) marcar campos faltantes antes de la consulta; (3) evitar transcribir dos veces la misma información.\nVALIDAR: qué campos exige el flujo local y si la plantilla reduce retrabajo sin omitir datos importantes.",
            "PILOTO: probar la plantilla durante una semana con un grupo pequeño de consultas.\nINDICADORES: tiempo de preparación por consulta = minutos totales / consultas; completitud = campos completos / campos requeridos × 100.\nPENDIENTE: línea base y meta. Revisar una muestra con profesionales antes de ampliar el uso."
        ]
    elif name.startswith("Aseguramiento"):
        answers=[
            "HECHOS: 132/200 tienen control registrado (66%); 68/200 no tienen control registrado (34%); HbA1c registrada: 118/200 (59%); presión arterial registrada: 145/200 (72,5%).\nLÍMITES: los grupos pueden superponerse y 54 citas pendientes no equivalen necesariamente a 54 personas únicas. Registro ausente no prueba que no hubo atención.",
            "OPORTUNIDADES A INVESTIGAR: (1) verificar casos sin control registrado; (2) revisar oportunidad de medición de HbA1c; (3) depurar y caracterizar las citas pendientes.\nHIPÓTESIS, NO CONCLUSIONES: puede haber brechas de registro, acceso o seguimiento.\nDATOS NECESARIOS: fechas, unicidad de personas, contacto realizado y motivo de pendiente.",
            "PILOTO: validar una lista de seguimiento de un grupo pequeño y contactar según el protocolo de la EPS.\nINDICADORES: proporción con seguimiento confirmado = personas con seguimiento validado / personas elegibles × 100; citas pendientes resueltas / citas pendientes verificadas × 100.\nDEFINIR antes de iniciar: población, línea base, meta, responsable y frecuencia. No inferir necesidades clínicas individuales."
        ]
    elif name.startswith("Operaciones"):
        answers=[
            "HECHOS: espera media 42 min; mediana 35 min; 8 de 30 citas superaron 60 min (8/30 × 100 = 26,7%).\nLÍMITES: solo hay un día de datos. No se sabe si el patrón es habitual ni qué lo explica.",
            "HIPÓTESIS A COMPROBAR: (1) variación por especialidad; (2) concentración por franja horaria; (3) retrasos asociados a la hora de inicio o tipo de cita.\nDATOS: hora programada e inicio real, especialidad, tipo de cita y estado, recogidos durante varios días. Las hipótesis no son causas confirmadas.",
            "PILOTO: elegir una franja o servicio tras analizar varios días y probar un ajuste operativo acotado.\nINDICADORES: espera mediana; porcentaje de citas con espera >60 min = citas >60 / citas atendidas × 100; indicador de balance: cancelaciones o sobrecarga del personal.\nPENDIENTE: línea base, meta, periodo y responsable. Comparar periodos equivalentes."
        ]
    elif name.startswith("Auditoría"):
        answers=[
            "REGISTRO A: fecha y firma visibles; evidencia disponible para ambos criterios.\nREGISTRO B: fecha visible; firma no legible, requiere revisión.\nREGISTRO C: fecha no visible y copia no legible; no es posible concluir sobre el documento original.\nLÍMITE: los criterios son ficticios y no se proporcionó normativa aplicable.",
            "FLUJO PROPUESTO: (1) comprobar legibilidad; (2) marcar cada criterio como visible, ausente en la copia o no evaluable; (3) adjuntar evidencia; (4) escalar casos dudosos a auditoría humana.\nMEJORA: usar una matriz común para reducir criterios inconsistentes y retrabajo.",
            "PILOTO: aplicar la matriz a una muestra definida de registros.\nINDICADORES: tiempo medio de revisión = minutos de revisión / registros revisados; concordancia = decisiones coincidentes en doble revisión / registros revisados por duplicado × 100.\nPENDIENTE: línea base y meta. Validar criterios y escalamiento con el equipo responsable antes de usarla formalmente."
        ]
    else:
        answers=[
            "HECHO: 45 de 120 participantes reportaron dificultades (45/120 × 100 = 37,5%).\nLIMITACIONES: encuesta por conveniencia; puede no representar a la población y no permite establecer causalidad.\nPREGUNTAS: ¿cómo se reclutó la muestra?, ¿qué se entiende por dificultad?, ¿hay diferencias por zona o grupo? Esos datos no se suministraron.",
            "OPCIONES A EXPLORAR: (1) ampliar o diversificar la muestra; (2) segmentar dificultades por ubicación o tipo de servicio si hay datos; (3) mapear etapas del proceso de solicitud de citas.\nPARA VALIDAR: representatividad, definiciones, disponibilidad de datos y recursos. Los resultados actuales no demuestran la causa del problema.",
            "SIGUIENTE PASO: revisar el instrumento de encuesta y diseñar una pequeña validación con una muestra más diversa.\nINDICADORES: tasa de respuesta = respuestas completas / personas invitadas × 100; proporción que reporta dificultad por segmento, solo si el tamaño de muestra permite interpretarlo.\nDEFINIR: responsable, método de muestreo, criterios y resguardos de privacidad antes de ejecutar."
        ]
    answer=answers[idx]
    # Explicitly carry forward prior result in a compact context line to demonstrate the chain.
    if idx > 0 and prior:
        answer += "\n\nCONTEXTO USADO: se tomó en cuenta el resultado del paso anterior guardado en esta práctica."
    return answer

for i,(heading,instruction) in enumerate(case["steps"]):
    css=["s1","s2","s3"][i]
    st.markdown(f'<div class="stage {css}"><div style="font-size:.76rem;font-weight:850;letter-spacing:.08em;color:#13818A">PASO {i+1} DE 3</div><h3>{heading}</h3><small>{["Comprender","Analizar","Probar y medir"][i]}</small></div>',unsafe_allow_html=True)
    if i>0 and not chain["answers"][i-1]:
        st.info(f"Completa el paso {i} antes de avanzar. El resultado se incorporará al siguiente paso.")
    disabled=(i>0 and not chain["answers"][i-1].strip())
    if st.button("Generar ejemplo guiado" if not chain["answers"][i] else "Volver a generar ejemplo",key=f"generate_{i}_{area}_{part}",type="primary",use_container_width=True,disabled=disabled):
        prior="\n".join(chain["answers"][:i])
        chain["answers"][i]=make_answer(i,case,prior)
        chain["done"][i]=True
        if i<2:
            for j in range(i+1,3):
                chain["answers"][j]=""
                chain["done"][j]=False
        st.session_state[chain_id]=chain
        st.rerun()
    if chain["answers"][i]:
        st.markdown("**Resultado de ejemplo**")
        st.markdown(chain["answers"][i])
        st.caption("Resultado guiado predefinido para practicar el flujo; no es una respuesta generada en vivo por un modelo de IA.")
        if st.button("Borrar este paso y repetir desde aquí",key=f"clear_{i}_{area}_{part}"):
            for j in range(i,3):
                chain["answers"][j]=""
                chain["done"][j]=False
            st.session_state[chain_id]=chain
            st.rerun()

if all(x.strip() for x in chain["answers"]):
    st.markdown('<div class="step" style="background:#E2F7EE;border-color:#B8E7D1"><div class="kicker" style="color:#176D4B">RESULTADO FINAL</div><div class="stitle">Ruta de mejora completada</div><div class="sub">Revisa evidencia, indicadores y validación humana antes de actuar.</div></div>',unsafe_allow_html=True)
    export="\n\n".join(f"PASO {i+1}: {case['steps'][i][0]}\n{answer}" for i,answer in enumerate(chain["answers"]))
    st.download_button("Descargar resultados (.txt)",data=export,file_name="laboratorio_ia_salud_resultados.txt",mime="text/plain")
    if st.button("Reiniciar esta práctica"):
        st.session_state[chain_id]={"answers":["","",""],"done":[False,False,False]}
        st.rerun()

st.markdown('<div class="notice">💡 <b>Uso responsable:</b> la IA puede ayudar a organizar información, identificar oportunidades y estructurar pruebas; la validación de los datos y las decisiones siguen siendo responsabilidad del equipo.</div>',unsafe_allow_html=True)
