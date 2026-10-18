import streamlit as st

st.set_page_config(page_title="Laboratorio de IA aplicada a salud", page_icon="🩺", layout="wide")

# Workflows with a clear goal: diagnose a process, find improvement opportunities, and design a measurable pilot.
WORKFLOWS = {
    "Atención clínica y rehabilitación": {
        "description": "Mejorar la continuidad y la calidad del registro clínico sin automatizar decisiones clínicas.",
        "case": "En una consulta externa ficticia, una revisión de 30 notas encontró que 12 no dejaban claro el plan de seguimiento y 8 no documentaban de forma explícita las señales de alarma discutidas. La revisión no determinó si la información se comunicó verbalmente.",
        "goal": "Mejorar la integridad y claridad de las notas de evolución y del plan de seguimiento.",
        "data": "30 notas revisadas; 12 sin plan de seguimiento claramente documentado; 8 sin señales de alarma explícitas en la nota. Los grupos pueden solaparse. No se sabe si la información se comunicó verbalmente.",
        "stage1": "Actúa como analista de mejora de procesos clínicos. Usa solo este caso ficticio: {case}\nResume el problema en máximo 5 viñetas. Separa hechos, incertidumbres y datos faltantes. No infieras que algo no ocurrió solo porque no está documentado. No propongas todavía soluciones.",
        "stage2": "A partir del caso y del diagnóstico siguiente, identifica hasta 3 oportunidades de mejora del proceso de documentación: {previous}\nPara cada una indica evidencia, hipótesis por validar, dato adicional necesario y riesgo de una interpretación equivocada. No recomiendes diagnósticos ni tratamientos. Máximo 5 viñetas.",
        "stage3": "Usa el caso y el análisis previos para proponer un piloto pequeño que mejore la claridad de las notas y el seguimiento: caso={case}; diagnóstico y oportunidades={previous}\nIncluye: cambio de proceso, responsable funcional, indicador con fórmula, forma de muestreo y salvaguarda de revisión humana. No inventes metas ni prometas resultados. Máximo 120 palabras.",
    },
    "Aseguramiento y gestión del riesgo en salud (EPS)": {
        "description": "Detectar brechas de seguimiento poblacional y convertirlas en acciones medibles para validación.",
        "case": "Una EPS ficticia revisa 200 personas con diabetes tipo 2: 132 tienen un control registrado en el trimestre; 68 no tienen control registrado; 118 tienen HbA1c registrada en los últimos seis meses; 145 tienen presión arterial registrada en el trimestre. Se reportan 54 citas pendientes, pero falta confirmar si son personas únicas.",
        "goal": "Priorizar una brecha de seguimiento y diseñar una acción piloto con indicadores.",
        "data": "Población: 200; control registrado: 132; sin control registrado: 68; HbA1c registrada: 118; presión arterial registrada: 145; citas pendientes reportadas: 54. La falta de registro no demuestra ausencia de atención.",
        "stage1": "Actúa como analista de gestión del riesgo en salud. Analiza este caso ficticio: {case}\nResume máximo 5 hallazgos cuantitativos, muestra las fórmulas cuando calcules porcentajes y separa hechos de incertidumbres. No infieras causalidad ni equipares ausencia de registro con ausencia de atención. Señala los datos que deben validarse.",
        "stage2": "A partir del caso y de este diagnóstico: {previous}\nIdentifica hasta 3 oportunidades para mejorar el seguimiento de la población. Para cada una incluye evidencia, hipótesis por validar, información adicional necesaria y posible indicador. No recomiendes tratamientos individuales ni inventes metas. Máximo 5 viñetas.",
        "stage3": "Con base en el caso y el análisis: caso={case}; diagnóstico y oportunidades={previous}\nDiseña un piloto de mejora para discusión de la EPS. Incluye población objetivo, acción de proceso, responsable funcional, indicador con numerador/denominador, frecuencia de seguimiento y controles de calidad del dato. Distingue propuesta de decisión aprobada. Máximo 120 palabras.",
    },
    "Operaciones y administración de IPS": {
        "description": "Investigar demoras operativas y definir una prueba de mejora basada en medición.",
        "case": "En consulta externa de una IPS ficticia, durante un día se observaron 30 citas: la espera media fue de 42 minutos, la mediana de 35 minutos y 8 citas superaron los 60 minutos. No hay datos de otros días, especialidades ni causas de retraso.",
        "goal": "Identificar qué medir primero y diseñar un piloto operativo que pueda evaluarse.",
        "data": "30 citas de un solo día; espera media 42 minutos; mediana 35 minutos; 8 de 30 superaron 60 minutos. No hay segmentación ni causas registradas.",
        "stage1": "Actúa como analista de procesos de una IPS. Revisa este caso ficticio: {case}\nResume qué sabemos, qué no sabemos y qué no podemos concluir. Calcula el porcentaje de citas con espera superior a 60 minutos mostrando la fórmula. Máximo 5 viñetas. No inventes causas.",
        "stage2": "A partir del caso y este diagnóstico: {previous}\nPropón hasta 3 hipótesis comprobables sobre las demoras. Para cada hipótesis indica qué datos se necesitan, cómo segmentarlos y qué resultado la apoyaría o debilitaría. No presentes hipótesis como causas confirmadas. Máximo 5 viñetas.",
        "stage3": "Usa el caso y el análisis: caso={case}; diagnóstico y oportunidades={previous}\nDiseña un piloto de mejora operativo de bajo riesgo. Incluye proceso a probar, responsable funcional, indicador principal con fórmula, indicador de balance para evitar efectos negativos, periodo de medición y criterios de revisión. No inventes metas. Máximo 120 palabras.",
    },
    "Auditoría, calidad y regulación": {
        "description": "Convertir hallazgos documentales en una revisión trazable y un plan de mejora verificable.",
        "case": "En una auditoría ficticia de 40 registros, 10 no muestran fecha de revisión en la copia disponible y 6 tienen una firma ilegible. No se ha revisado si existen documentos complementarios ni se ha definido aquí una norma aplicable.",
        "goal": "Distinguir hallazgos documentales de incumplimientos confirmados y mejorar la trazabilidad.",
        "data": "40 registros revisados; 10 sin fecha visible en la copia; 6 con firma ilegible. Puede haber solapamiento. No se suministró normativa ni se revisaron anexos.",
        "stage1": "Actúa como asistente de análisis para auditoría y calidad. Examina este caso ficticio: {case}\nOrganiza los hallazgos observables, las limitaciones de la evidencia y los documentos o criterios que faltan. No cites normas no suministradas ni conviertas automáticamente falta de evidencia en incumplimiento. Máximo 5 viñetas.",
        "stage2": "Con base en el caso y el diagnóstico: {previous}\nPropón hasta 3 oportunidades para mejorar la trazabilidad documental. Para cada una indica evidencia, posible causa por validar, información necesaria y criterio que debe confirmar el auditor. No emitas un dictamen de cumplimiento. Máximo 5 viñetas.",
        "stage3": "Usa el caso y el análisis: caso={case}; diagnóstico y oportunidades={previous}\nDiseña un piloto para mejorar la revisión documental. Incluye cambio de proceso, responsable funcional, indicador con fórmula, muestreo y paso de validación humana. No inventes obligaciones legales ni metas. Máximo 120 palabras.",
    },
    "Dirección, investigación y salud pública": {
        "description": "Transformar evidencia limitada en preguntas de decisión y opciones de mejora con cautelas explícitas.",
        "case": "Una encuesta ficticia por conveniencia a 120 personas encontró que 45 reportaron dificultades para conseguir una cita. No es una muestra representativa y no permite concluir que el resultado describa a toda la población ni establecer causas.",
        "goal": "Definir qué información adicional se necesita antes de priorizar una intervención institucional.",
        "data": "Encuesta ficticia por conveniencia; 120 participantes; 45 reportaron dificultades para conseguir cita. El muestreo limita la generalización y no permite inferir causalidad.",
        "stage1": "Actúa como analista de investigación y salud pública. Analiza este caso ficticio: {case}\nResume el hallazgo cuantitativo, qué significa y qué no permite afirmar. Muestra el cálculo del porcentaje y señala las limitaciones del muestreo. Máximo 5 viñetas; no agregues datos externos.",
        "stage2": "A partir del caso y del diagnóstico: {previous}\nIdentifica hasta 3 preguntas prioritarias para comprender el problema y orientar una decisión institucional. Para cada pregunta explica qué dato recoger, de quién y cómo ayudaría a decidir. No asumas causas ni generalices más allá de la muestra. Máximo 5 viñetas.",
        "stage3": "Usa el caso y el análisis: caso={case}; diagnóstico y oportunidades={previous}\nPropón un siguiente paso de mejora o aprendizaje institucional que sea proporcional a la evidencia disponible. Incluye acción, población a consultar, indicador, método de seguimiento y limitaciones que se comunicarán a dirección. No presentes la propuesta como una conclusión definitiva. Máximo 120 palabras.",
    },
}

st.markdown("""
<style>
:root {--navy:#103e66;--teal:#087f8c;--ink:#172b3a;}
.block-container{padding-top:1.3rem;padding-bottom:3rem;max-width:1280px}
.hero{background:linear-gradient(125deg,#103e66 0%,#155a7d 60%,#078b91 100%);color:#fff;padding:1.8rem 1.5rem;border-radius:20px;text-align:center;margin-bottom:1rem;box-shadow:0 10px 28px #103e6620}
.hero h1{color:#fff!important;font-size:2.05rem!important;margin:.15rem 0 .4rem!important}.hero p{color:#eaf6ff;font-size:1.1rem;margin:0}
.step-card{border:1px solid #dce5ea;background:#fff;border-radius:15px;padding:1rem 1.2rem;margin:.65rem 0 .7rem;box-shadow:0 3px 12px #18364909}
.step-kicker{font-size:.76rem;letter-spacing:.1em;text-transform:uppercase;font-weight:800;color:#087f8c}.step-title{font-size:1.3rem;font-weight:800;color:#172b3a}.step-copy{color:#60717c;margin:.2rem 0 0}
div[data-testid="stSelectbox"] div[data-baseweb="select"]>div{background:#eaf5ff;border:2px solid #4b9ce8;border-radius:12px;min-height:3.35rem;font-size:1.08rem;font-weight:700}
div[data-testid="stSelectbox"] label{font-size:1rem;font-weight:800;color:#164e82}
div[data-testid="stRadio"]>label{font-size:1.02rem;font-weight:800;color:#172b3a}
div[role="radiogroup"]{gap:1rem;width:100%}
div[role="radiogroup"] label{background:#e8f4ff;border:2px solid #b9dafa;border-radius:16px;padding:1rem 1.1rem;min-height:74px;font-weight:800!important;flex:1}
div[role="radiogroup"] label:has(input:checked){background:#d5edff;border-color:#2f80ed;color:#164e82}
div[role="radiogroup"] label p{font-size:1.03rem!important}
.stCode pre{white-space:pre-wrap}.stButton>button,.stLinkButton a{border-radius:10px;font-weight:650}
@media(max-width:700px){.hero h1{font-size:1.55rem!important}.hero{padding:1.35rem .8rem}div[role="radiogroup"]{flex-direction:column}}
</style>
<div class="hero"><div style="font-size:2rem">🩺</div><h1>Laboratorio de IA aplicada a salud</h1><p>Selecciona tu perfil y aprende haciendo</p></div>
""", unsafe_allow_html=True)

st.warning("⚠️ Usa solo los casos ficticios de esta aplicación. No pegues historias clínicas reales ni datos personales o sensibles. Las respuestas de IA requieren revisión humana.")

st.markdown('<div class="step-card"><div class="step-kicker">Paso 1</div><div class="step-title">Selecciona tu área de trabajo</div><p class="step-copy">Elige el contexto que mejor representa tu trabajo.</p></div>', unsafe_allow_html=True)
segment = st.selectbox("Área profesional", list(WORKFLOWS.keys()), label_visibility="visible")
flow = WORKFLOWS[segment]
st.caption(flow["description"])

st.markdown('<div class="step-card"><div class="step-kicker">Paso 2</div><div class="step-title">Elige una etapa del laboratorio</div><p class="step-copy">Sigue una secuencia con un objetivo concreto: entender el proceso, identificar oportunidades y diseñar una mejora medible.</p></div>', unsafe_allow_html=True)
part = st.radio("Parte del laboratorio", ["PARTE 1  ·  ENTENDER Y ANALIZAR", "PARTE 2  ·  DISEÑAR Y VALIDAR"], horizontal=True, label_visibility="collapsed")

st.markdown("### Caso práctico ficticio")
st.write(flow["case"])
st.markdown(f"**Objetivo del ejercicio:** {flow['goal']}")

try:
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    model = st.secrets.get("OPENAI_MODEL", "gpt-4.1-mini")
except Exception:
    api_key = ""
    model = "gpt-4.1-mini"

if api_key:
    st.success(f"Asistente de IA conectado · modelo configurado: {model}")
else:
    st.info("Modo guiado: puedes copiar cada prompt y pegarlo en ChatGPT o Copilot. Para ejecutar las 3 etapas dentro de esta app, configura OPENAI_API_KEY en Streamlit Secrets.")

# Each step uses the prior answer, so this is a connected workflow rather than three unrelated prompts.
state_prefix = f"{segment}_{'p1' if part.startswith('PARTE 1') else 'p2'}"
for stage_num in (1, 2) if part.startswith("PARTE 1") else (3,):
    stage_name = {1:"Prompt 1 · Entender el problema", 2:"Prompt 2 · Encontrar oportunidades", 3:"Prompt 3 · Diseñar una mejora medible"}[stage_num]
    prompt_template = flow[f"stage{stage_num}"]
    if stage_num == 1:
        prompt = prompt_template.format(case=flow["case"])
        prior_text = ""
    elif stage_num == 2:
        prior_text = st.session_state.get(f"{segment}_stage1", "")
        prompt = prompt_template.format(previous=prior_text or "Aún no hay diagnóstico de la etapa 1; usa el caso y señala esta limitación.")
    else:
        first = st.session_state.get(f"{segment}_stage1", "")
        second = st.session_state.get(f"{segment}_stage2", "")
        combined = f"Diagnóstico etapa 1: {first or 'No completado'}\nOportunidades etapa 2: {second or 'No completado'}"
        prompt = prompt_template.format(case=flow["case"], previous=combined)

    with st.container(border=True):
        st.subheader(stage_name)
        if stage_num == 2 and not st.session_state.get(f"{segment}_stage1"):
            st.caption("Sugerencia: ejecuta primero el Prompt 1 y usa su resultado como contexto.")
        if stage_num == 3 and (not st.session_state.get(f"{segment}_stage1") or not st.session_state.get(f"{segment}_stage2")):
            st.caption("Para obtener una propuesta más coherente, completa primero los dos prompts de la Parte 1.")
        st.code(prompt + "\n\nRegla de salida: máximo 120 palabras o 5 viñetas. Usa solo la información suministrada; no inventes datos.", language=None)
        col_run, col_manual = st.columns([1, 2])
        with col_run:
            run_label = "Ejecutar con IA" if api_key else "IA no conectada"
            run_disabled = not bool(api_key)
            run_now = st.button(run_label, key=f"run_{segment}_{stage_num}", disabled=run_disabled, type="primary")
        with col_manual:
            st.link_button("Abrir ChatGPT", "https://chatgpt.com/")
            st.link_button("Abrir Copilot", "https://copilot.microsoft.com/")

        if run_now and api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role":"system", "content":"Eres un asistente de mejora de procesos en salud. Trabaja solo con casos ficticios. No des consejo clínico individual, no inventes hechos, distingue evidencia e hipótesis y responde con la brevedad indicada por el usuario."},
                        {"role":"user", "content":prompt + "\n\nResponde en máximo 120 palabras o 5 viñetas."},
                    ],
                    temperature=0.2,
                    max_tokens=300,
                )
                answer = response.choices[0].message.content or "No se recibió texto de respuesta."
                st.session_state[f"{segment}_stage{stage_num}"] = answer
            except Exception as ex:
                st.error(f"No se pudo ejecutar el modelo. Revisa la clave, el nombre del modelo y la configuración. Detalle: {ex}")

        saved = st.session_state.get(f"{segment}_stage{stage_num}", "")
        manual = st.text_area("Resultado de esta etapa", value=saved, key=f"manual_{segment}_{stage_num}", height=140, placeholder="Pega aquí la respuesta de tu asistente si estás trabajando en modo guiado.")
        if st.button("Guardar resultado de esta etapa", key=f"save_{segment}_{stage_num}"):
            st.session_state[f"{segment}_stage{stage_num}"] = manual
            st.success("Resultado guardado para usarlo en el siguiente prompt.")

st.markdown("### ¿Cómo sabrás si la propuesta es útil?")
st.markdown("- Parte de un problema descrito con datos, no de una suposición.\n- Distingue hechos, hipótesis y datos faltantes.\n- Propone un cambio concreto y un indicador medible.\n- Incluye validación humana antes de implementar.")
st.caption("Laboratorio educativo. Los resultados son preliminares y deben validarse con los equipos responsables y las fuentes institucionales aplicables.")
