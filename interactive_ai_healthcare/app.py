import streamlit as st
st.set_page_config(page_title='Laboratorio de IA aplicada a salud', page_icon='🩺', layout='wide')

# Cada segmento contiene dos ejercicios por sesión.
DATA = {
'Atención clínica y rehabilitación': ('Documentación clínica, síntesis de historias y comunicación con pacientes.', [
('Estructurar una nota de evolución','Organizar apuntes sin agregar información.','Organiza estos apuntes ficticios en secciones: motivo de consulta, antecedentes mencionados, hallazgos reportados, evaluación y plan mencionado. No inventes datos; marca vacíos como “No especificado”. No propongas diagnósticos ni tratamientos. Devuelve una tabla con sección, información extraída y dudas por validar.\n\nApuntes: Paciente ficticia de 52 años consulta por dolor de rodilla desde hace 3 días. Refiere dolor al caminar. No se aportan examen físico, antecedentes ni plan.','Nota organizada y lista de datos por confirmar.'),
('Comparar resúmenes de historia clínica','Preparar una síntesis breve para revisión profesional.','Resume esta historia clínica ficticia en máximo 180 palabras: motivo de atención, antecedentes explícitos, eventos en orden temporal, medicamentos solo si aparecen y preguntas pendientes. Distingue hechos de incertidumbres. No infieras diagnósticos ni recomiendes tratamientos. Si falta un dato, escribe “No mencionado”.\n\nHistoria: Paciente ficticio consulta por tos de 4 días. El relato no menciona dificultad respiratoria. No se describen signos vitales ni examen físico.','Resumen y preguntas pendientes.'),
('Extraer información de un caso clínico','Localizar información puntual y señalar vacíos.','Extrae solo información explícita del caso ficticio en una tabla: dato, valor encontrado, fragmento que lo respalda y estado (explícito, ambiguo o ausente). Busca motivo de consulta, antecedentes, medicamentos mencionados, signos/resultados y seguimiento. No interpretes ni completes vacíos.\n\nCaso: Paciente ficticio de 45 años consulta por cefalea desde ayer. Se menciona antecedente de hipertensión. No se informan medicamentos actuales ni signos vitales.','Tabla de extracción con evidencia textual.'),
('Traducir una explicación a lenguaje claro','Simplificar sin alterar el significado.','Reescribe este texto educativo ficticio para una persona adulta sin formación sanitaria. Usa frases cortas, define términos técnicos y conserva el significado. No agregues recomendaciones médicas ni lo presentes como consejo personalizado. Termina con 3 preguntas para conversar con el equipo de salud.\n\nTexto: La hipertensión arterial es una condición en la que la presión de la sangre se mantiene elevada y requiere seguimiento profesional.','Versión clara y preguntas para el equipo.')]),
'Aseguramiento y gestión del riesgo en salud (EPS)': ('Seguimiento poblacional, indicadores y continuidad de la atención.', [
('Mejorar un prompt de seguimiento de diabetes','Analizar indicadores sin confundir registro ausente con atención no realizada.','Analiza estos datos agregados ficticios de diabetes tipo 2. Calcula porcentajes si el denominador está disponible y muestra fórmula. Separa hallazgos, hipótesis y datos faltantes. No infieras causalidad ni afirmes que un registro ausente demuestra que no hubo atención.\n\nPoblación: 200; control registrado en trimestre: 132; sin control registrado: 68; HbA1c registrada en últimos 6 meses: 118; presión arterial registrada en trimestre: 145; citas pendientes reportadas: 54. Devuelve indicador, numerador, denominador, cálculo, resultado y cautela interpretativa.','Tabla de indicadores con fórmulas y cautelas.'),
('Identificar datos faltantes','Formular preguntas de calidad antes de analizar.','Antes de analizar esta tabla ficticia, no saques conclusiones. Identifica dudas sobre definición de indicadores, periodo, población objetivo, denominadores, duplicados, datos faltantes, actualización y sesgos de registro. Formula hasta 8 preguntas y explica por qué importa cada una.\n\nDescripción: Columnas ficticias: identificador, grupo de edad, fecha de control, HbA1c registrada (sí/no), presión arterial registrada (sí/no), cita pendiente (sí/no).','Preguntas priorizadas para el responsable de datos.'),
('Separar hallazgos e hipótesis','Clasificar afirmaciones como hechos, hipótesis o información insuficiente.','Con estos datos ficticios (132/200 con control registrado; 68/200 sin control registrado; 118 con HbA1c registrada; 145 con presión arterial registrada), clasifica cada afirmación como hecho respaldado, hipótesis por validar o información insuficiente. Incluye evidencia, límite interpretativo y dato adicional necesario. No atribuyas causas ni asumas que un registro ausente equivale a atención no realizada.','Matriz de hechos, hipótesis y preguntas.'),
('Proponer acciones e indicadores','Generar opciones preliminares vinculadas a evidencia y métricas.','Propón hasta 3 acciones preliminares para discusión de una EPS a partir de estos hallazgos ficticios: De 200 personas ficticias, 132 tienen control registrado en el trimestre y 118 tienen HbA1c registrada en los últimos seis meses.. Para cada acción incluye hallazgo, hipótesis por validar, responsable funcional, indicador, fórmula y datos necesarios. No inventes metas, costos o resultados; no recomiendes tratamientos individuales. Distingue propuesta de decisión aprobada.','Matriz de acciones e indicadores por validar.')]),
'Operaciones y administración de IPS': ('Capacidad operativa, oportunidad, cancelaciones y mejora de procesos.', [
('Analizar retrasos y cancelaciones','Diseñar un análisis descriptivo antes de buscar causas.','Diseña un análisis descriptivo para datos ficticios de citas y retrasos. Indica variables, definiciones, segmentaciones y métricas útiles. Separa lo que puede medirse de las causas posibles que requieren evidencia. No inventes resultados ni atribuyas responsabilidades sin datos.\n\nColumnas disponibles: fecha_cita, especialidad, hora_programada, hora_inicio, estado_cita, motivo_cancelacion','Plan de análisis descriptivo.'),
('Investigar un problema operativo','Convertir una queja general en preguntas comprobables.','Estructura una investigación inicial sobre demoras en una IPS. No asumas la causa. Entrega: definición medible del problema, preguntas de diagnóstico, datos necesarios, segmentos de análisis, hipótesis comprobables y criterios para evaluar una intervención. Distingue hechos de supuestos.\n\nContexto ficticio: En una IPS ficticia, usuarios reportan esperas largas en consulta externa. Aún no hay medición por especialidad ni por hora del día.','Guía breve de diagnóstico operativo.'),
('Resumir tiempos de espera','Identificar patrones sin ocultar limitaciones.','Resume esta tabla ficticia de tiempos de espera: periodo, número de registros, estadísticas disponibles, diferencias entre grupos solo si los datos lo permiten, posibles atípicos y limitaciones. No inventes estadísticas ni explicaciones.\n\nDatos: Consulta externa ficticia en un día: 30 citas; espera media 42 minutos; mediana 35 minutos; 8 citas superaron 60 minutos.','Resumen operativo con limitaciones.'),
('Diseñar un piloto de mejora','Proponer una prueba pequeña con medición y seguimiento.','Propón un plan preliminar de mejora basado solo en estos hallazgos ficticios: En un día ficticio, la espera media fue 42 minutos y 8 de 30 citas superaron 60 minutos. No hay datos de otros días ni causas registradas.. Incluye problema medible, intervención piloto, proceso afectado, responsable funcional, indicador y fórmula, frecuencia de revisión y riesgos. No prometas resultados ni inventes metas. Indica qué debe validarse antes de ejecutar.','Ficha de piloto y medición.')]),
'Auditoría, calidad y regulación': ('Revisión documental, criterios, clasificación de hallazgos y trazabilidad.', [
('Clasificar motivos ficticios de glosa','Aplicar criterios entregados y señalar casos ambiguos.','Clasifica estos casos ficticios solo con los criterios suministrados. Para cada uno incluye texto del caso, criterio posiblemente aplicable, evidencia textual, confianza cualitativa y dato faltante. Si no es suficiente, escribe “Requiere revisión humana”. No inventes normas ni decidas automáticamente quién tiene razón.\n\nCriterios: Criterio ficticio A: el registro debe incluir fecha. Criterio ficticio B: debe incluir firma del responsable.\nCasos: Caso A: registro sin fecha visible. Caso B: registro con fecha y firma. Caso C: copia ilegible.','Tabla preliminar de clasificación.'),
('Revisar un documento contra criterios','Crear una lista de verificación trazable.','Revisa el documento ficticio contra estos criterios. Por criterio devuelve evidencia textual exacta, estado (cumple, no cumple, no evaluable) y explicación. No infieras evidencia ni inventes requisitos. Termina con puntos para revisión del auditor.\n\nCriterios: Criterio ficticio A: el registro debe incluir fecha. Criterio ficticio B: debe incluir firma del responsable.\nDocumento: Estudio ficticio transversal con 120 participantes: 45 reportaron dificultades para conseguir cita. La muestra fue por conveniencia; no permite concluir causalidad ni generalizar a toda la población.','Lista de verificación trazable.'),
('Clasificar hallazgos frente a criterios','Construir una matriz de revisión con evidencia.','Compara cada extracto ficticio con los criterios. Devuelve identificador, criterio, fragmento de evidencia, clasificación (posible incumplimiento, evidencia de cumplimiento o no evaluable), justificación y pregunta de validación. No conviertas ausencia de evidencia en prueba automática de incumplimiento ni inventes normativa.\n\nCriterios: Criterio ficticio A: el registro debe incluir fecha. Criterio ficticio B: debe incluir firma del responsable.\nExtractos: Registro 01: contiene fecha y firma. Registro 02: contiene fecha, pero la firma no es legible.','Matriz de hallazgos preliminares.'),
('Detectar conclusiones no sustentadas','Auditar una respuesta de IA contra su fuente.','Compara la respuesta de IA con el documento fuente ficticio. Clasifica cada afirmación como respaldada, parcialmente respaldada, no respaldada o contradictoria. Cita el fragmento pertinente e identifica omisiones, inferencias y datos inventados. No decidas el caso; indica qué debe validar el auditor.\n\nFuente: Documento ficticio: “El informe cubre 40 registros del trimestre. En 10 registros falta la fecha de revisión.”\nRespuesta IA: Respuesta de IA: “El 25% de los registros no tiene fecha de revisión. Esto demuestra incumplimiento sistemático del equipo.”','Informe breve de verificación.')]),
'Dirección, investigación y salud pública': ('Síntesis de evidencia, investigación clínica, salud pública y decisiones institucionales.', [
('Resumir un informe ejecutivo','Definir audiencia, estructura y límites de un resumen.','Resume este informe ficticio para una reunión directiva: propósito, 3 hallazgos, evidencia de cada uno, limitaciones, decisiones que podrían discutirse y preguntas abiertas. Distingue lo que afirma el documento de tus inferencias. No agregues datos externos.\n\nInforme: Informe ficticio: en una encuesta de 120 personas, 45 reportaron dificultades para conseguir cita. La muestra fue por conveniencia y no representa necesariamente a toda la población.','Resumen ejecutivo trazable.'),
('Extraer hallazgos y limitaciones','Leer críticamente un documento de investigación o salud pública.','Extrae del documento ficticio: objetivo, diseño si está explícito, población/fuente, hallazgos reportados, limitaciones mencionadas y preguntas abiertas. Para cada hallazgo cita una frase breve. Si falta un elemento, indica “No especificado”. No atribuyas causalidad sin respaldo.\n\nDocumento: Estudio ficticio transversal con 120 participantes: 45 reportaron dificultades para conseguir cita. La muestra fue por conveniencia; no permite concluir causalidad ni generalizar a toda la población.','Ficha de lectura crítica.'),
('Separar evidencia, incertidumbre y opciones','Preparar una conversación institucional sin convertir sugerencias en decisiones.','Organiza el documento ficticio en: evidencia explícita, incertidumbres/datos faltantes y opciones de acción para discusión. Para cada opción indica evidencia que la motiva y qué debe validarse. No presentes opciones como decisiones aprobadas ni añadas datos externos.\n\nDocumento: Estudio ficticio transversal con 120 participantes: 45 reportaron dificultades para conseguir cita. La muestra fue por conveniencia; no permite concluir causalidad ni generalizar a toda la población.','Matriz de evidencia, incertidumbre y opciones.'),
('Preparar una síntesis ejecutiva responsable','Comunicar hechos, límites y próximos pasos de validación.','Prepara una síntesis ejecutiva breve con contexto, hallazgos respaldados, implicaciones posibles, limitaciones, preguntas pendientes y próximos pasos de validación. Evita lenguaje categórico cuando haya incertidumbre. No inventes cifras ni causalidad.\n\nDocumento: Estudio ficticio transversal con 120 participantes: 45 reportaron dificultades para conseguir cita. La muestra fue por conveniencia; no permite concluir causalidad ni generalizar a toda la población.','Síntesis ejecutiva para revisión humana.')])
}

# Estilo visual con colores, tarjetas y jerarquía visual.
st.markdown('''
<style>
:root { --navy:#103e66; --blue:#2f80ed; --teal:#078b91; --ink:#172b3a; --muted:#637582; }
.block-container { padding-top:1.7rem; padding-bottom:3rem; max-width:1280px; }
.hero { background:linear-gradient(125deg,#103e66 0%,#155a7d 62%,#087f8c 100%); color:white; padding:2.1rem 2rem 1.8rem; border-radius:20px; text-align:center; margin-bottom:1.2rem; box-shadow:0 12px 30px rgba(16,62,102,.16); }
.hero-icon { font-size:2.2rem; line-height:1.2; margin-bottom:.45rem; }
.hero h1 { color:white !important; font-size:2.15rem !important; margin:.15rem 0 .45rem !important; }
.hero p { color:#eaf6ff; font-size:1.15rem; margin:0; }
.step-card { border:1px solid #dce5ea; background:#fff; border-radius:15px; padding:1.1rem 1.35rem; margin:.8rem 0 1.15rem; box-shadow:0 3px 12px rgba(24,54,73,.035); }
.step-kicker { text-transform:uppercase; font-size:.75rem; letter-spacing:.11em; font-weight:800; color:#087f8c; margin-bottom:.25rem; }
.step-title { color:#172b3a; font-size:1.35rem; font-weight:750; margin-bottom:.35rem; }
.step-copy { color:#60717c; font-size:1rem; margin:0; }
.part-card { border-radius:17px; padding:1.25rem 1.3rem 1.15rem; min-height:178px; border:1px solid transparent; }
.part-one { background:linear-gradient(135deg,#e8f4ff,#dceefa); border-color:#c8e4f7; }
.part-two { background:linear-gradient(135deg,#e2f7f1,#d4f1e9); border-color:#bce8dc; }
.pill { display:inline-block; border-radius:999px; padding:.28rem .75rem; font-weight:700; font-size:.86rem; margin-bottom:.75rem; }
.pill-blue { color:#1769c2; background:#c7e4ff; }
.pill-green { color:#087e57; background:#bfeedd; }
.part-title { color:#172b3a; font-size:1.3rem; font-weight:800; margin:.1rem 0 .45rem; }
.part-copy { color:#314b5b; font-size:.98rem; line-height:1.5; margin:0; }
.part-icon { font-size:1.65rem; margin-top:.55rem; }
.down-arrow { text-align:center; font-size:1.6rem; color:#80919b; margin:.4rem 0 .8rem; }
.section-heading { color:#163e5d; font-size:1.25rem; font-weight:800; margin:1.3rem 0 .3rem; }
div[data-testid="stSelectbox"] label, div[data-testid="stRadio"] label { font-weight:650; }
div[role="radiogroup"] { gap:.8rem; }
div[role="radiogroup"] label { background:#f0f6fa; border:2px solid #d5e3eb; border-radius:13px; padding:.85rem 1.05rem; min-height:58px; font-weight:750 !important; }
div[role="radiogroup"] label:has(input:checked) { background:#dceefa; border-color:#2f80ed; color:#164e82; }
div[data-testid="stRadio"] > label { font-size:1.05rem; margin-bottom:.6rem; }
.stCode pre { white-space:pre-wrap; }
.stButton>button, .stLinkButton a { border-radius:10px; }
[data-testid="stAlert"] { border-radius:12px; }
@media (max-width:700px) { .hero h1 {font-size:1.55rem !important;} .hero {padding:1.4rem 1rem;} }
</style>
<div class="hero"><div class="hero-icon">🩺</div><h1>Laboratorio de IA aplicada a salud</h1><p>Selecciona tu perfil y aprende haciendo</p></div>
''', unsafe_allow_html=True)

st.warning('⚠️ Usa solo los casos ficticios de esta aplicación. No pegues nombres, historias clínicas reales ni datos personales o sensibles. La IA puede equivocarse y sus respuestas requieren revisión humana.')

st.markdown('<div class="step-card"><div class="step-kicker">Paso 1</div><div class="step-title">Selecciona tu área de trabajo</div><p class="step-copy">Elige el contexto que mejor representa tu trabajo. Puedes explorar otras áreas cuando quieras.</p></div>', unsafe_allow_html=True)
segment = st.selectbox('Área profesional', list(DATA.keys()), label_visibility='collapsed')
st.caption(DATA[segment][0])
st.markdown('<div class="down-arrow">↓</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2, gap='medium')
with c1:
    st.markdown('''<div class="part-card part-one"><span class="pill pill-blue">Parte 1 · Explorar</span><div class="part-title">Interactuar y mejorar prompts</div><p class="part-copy">Explorar asistentes, comparar instrucciones y mejorar resultados.</p><div class="part-icon">💬</div></div>''', unsafe_allow_html=True)
with c2:
    st.markdown('''<div class="part-card part-two"><span class="pill pill-green">Parte 2 · Aplicar</span><div class="part-title">Aplicar y verificar</div><p class="part-copy">Extraer información, resumir, comunicar y evaluar propuestas de IA.</p><div class="part-icon">☑️</div></div>''', unsafe_allow_html=True)

st.markdown('<div class="down-arrow">↓</div>', unsafe_allow_html=True)
st.markdown('<div class="step-card"><div class="step-kicker">Paso 2</div><div class="step-title">Elige por dónde quieres empezar</div><p class="step-copy">Cada parte incluye ejercicios prácticos, prompts sugeridos y una lista para revisar los resultados.</p></div>', unsafe_allow_html=True)
part = st.radio('Parte de aprendizaje', ['Parte 1 · Interactuar y mejorar prompts', 'Parte 2 · Aplicar y verificar'], horizontal=True, label_visibility='collapsed')
sidx = 0 if part.startswith('Parte 1') else 1
exercises = DATA[segment][1][sidx*2:sidx*2+2]
title = st.selectbox('Ejercicio práctico', [e[0] for e in exercises])
e = next(x for x in exercises if x[0] == title)
left, right = st.columns([1.15, .85], gap='large')
with left:
    st.markdown('<div class="section-heading">Tu reto</div>', unsafe_allow_html=True)
    st.write(e[1])
    st.markdown('<div class="section-heading">Prompt sugerido</div>', unsafe_allow_html=True)
    st.caption('Cópialo y reemplaza los campos entre corchetes con información ficticia.')
    st.code(e[2] + '\n\nLímite de respuesta: máximo 120 palabras o 5 viñetas. Usa solo los datos proporcionados; si falta información, indícalo brevemente. No inventes datos.', language=None)
    st.markdown('<div class="section-heading">Producto esperado</div>', unsafe_allow_html=True)
    st.info(e[3])
with right:
    st.markdown('<div class="section-heading">Pruébalo con un asistente</div>', unsafe_allow_html=True)
    st.write('Abre una herramienta externa, pega el prompt y revisa la respuesta.')
    st.caption('Para mantener respuestas breves, el prompt incluye un límite de 120 palabras o 5 viñetas.')
    st.link_button('Abrir ChatGPT', 'https://chatgpt.com/')
    st.link_button('Abrir Microsoft Copilot', 'https://copilot.microsoft.com/')
    st.markdown('<div class="section-heading">Verifica el resultado</div>', unsafe_allow_html=True)
    for i, q in enumerate(['¿La respuesta se basa en la información suministrada?', '¿Señala datos faltantes y límites?', '¿Distingue hechos de hipótesis?', '¿Evita inventar datos o conclusiones?', '¿Indica qué debe validar una persona responsable?']):
        st.checkbox(q, key=f'{sidx}_{segment}_{title}_{i}')
    st.text_area('Resultado (opcional; solo datos ficticios)', height=150, placeholder='Pega aquí la respuesta para discutirla. No se envía a un modelo ni se guarda en una base de datos.')
st.divider()
st.markdown('<div class="section-heading">Preguntas para cerrar</div>', unsafe_allow_html=True)
st.markdown('- ¿Qué afirmaciones están respaldadas por la entrada?\n- ¿Qué falta o necesita validación?\n- ¿Qué decisión puede apoyar la IA y cuál sigue bajo responsabilidad humana?')
st.caption('Laboratorio educativo. Usa solo casos ficticios. No sustituye protocolos institucionales, fuentes verificadas ni criterio profesional.')
