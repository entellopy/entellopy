import streamlit as st
st.set_page_config(page_title='Laboratorio de IA aplicada a salud', page_icon='🩺', layout='wide')
st.title('🩺 Laboratorio de IA aplicada a salud')
st.caption('Módulo 2 · Sesiones 1 y 2 | Casos ficticios para practicar el uso crítico y seguro de la IA')
st.warning('Usa solo información ficticia. No pegues nombres, historias clínicas reales ni datos personales o sensibles. La IA puede equivocarse y sus respuestas requieren revisión humana.')

# Cada segmento contiene dos ejercicios por sesión.
DATA = {
'Atención clínica y rehabilitación': ('Documentación clínica, síntesis de historias y comunicación con pacientes.', [
('Estructurar una nota de evolución','Organizar apuntes sin agregar información.','Organiza estos apuntes ficticios en secciones: motivo de consulta, antecedentes mencionados, hallazgos reportados, evaluación y plan mencionado. No inventes datos; marca vacíos como “No especificado”. No propongas diagnósticos ni tratamientos. Devuelve una tabla con sección, información extraída y dudas por validar.\n\nApuntes: [PEGA APUNTES FICTICIOS]','Nota organizada y lista de datos por confirmar.'),
('Comparar resúmenes de historia clínica','Preparar una síntesis breve para revisión profesional.','Resume esta historia clínica ficticia en máximo 180 palabras: motivo de atención, antecedentes explícitos, eventos en orden temporal, medicamentos solo si aparecen y preguntas pendientes. Distingue hechos de incertidumbres. No infieras diagnósticos ni recomiendes tratamientos. Si falta un dato, escribe “No mencionado”.\n\nHistoria: [PEGA TEXTO FICTICIO]','Resumen y preguntas pendientes.'),
('Extraer información de un caso clínico','Localizar información puntual y señalar vacíos.','Extrae solo información explícita del caso ficticio en una tabla: dato, valor encontrado, fragmento que lo respalda y estado (explícito, ambiguo o ausente). Busca motivo de consulta, antecedentes, medicamentos mencionados, signos/resultados y seguimiento. No interpretes ni completes vacíos.\n\nCaso: [PEGA CASO FICTICIO]','Tabla de extracción con evidencia textual.'),
('Traducir una explicación a lenguaje claro','Simplificar sin alterar el significado.','Reescribe este texto educativo ficticio para una persona adulta sin formación sanitaria. Usa frases cortas, define términos técnicos y conserva el significado. No agregues recomendaciones médicas ni lo presentes como consejo personalizado. Termina con 3 preguntas para conversar con el equipo de salud.\n\nTexto: [PEGA TEXTO]','Versión clara y preguntas para el equipo.')]),
'Aseguramiento y gestión del riesgo en salud (EPS)': ('Seguimiento poblacional, indicadores y continuidad de la atención.', [
('Mejorar un prompt de seguimiento de diabetes','Analizar indicadores sin confundir registro ausente con atención no realizada.','Analiza estos datos agregados ficticios de diabetes tipo 2. Calcula porcentajes si el denominador está disponible y muestra fórmula. Separa hallazgos, hipótesis y datos faltantes. No infieras causalidad ni afirmes que un registro ausente demuestra que no hubo atención.\n\nPoblación: 200; control registrado en trimestre: 132; sin control registrado: 68; HbA1c registrada en últimos 6 meses: 118; presión arterial registrada en trimestre: 145; citas pendientes reportadas: 54. Devuelve indicador, numerador, denominador, cálculo, resultado y cautela interpretativa.','Tabla de indicadores con fórmulas y cautelas.'),
('Identificar datos faltantes','Formular preguntas de calidad antes de analizar.','Antes de analizar esta tabla ficticia, no saques conclusiones. Identifica dudas sobre definición de indicadores, periodo, población objetivo, denominadores, duplicados, datos faltantes, actualización y sesgos de registro. Formula hasta 8 preguntas y explica por qué importa cada una.\n\nDescripción: [PEGA DESCRIPCIÓN FICTICIA]','Preguntas priorizadas para el responsable de datos.'),
('Separar hallazgos e hipótesis','Clasificar afirmaciones como hechos, hipótesis o información insuficiente.','Con estos datos ficticios (132/200 con control registrado; 68/200 sin control registrado; 118 con HbA1c registrada; 145 con presión arterial registrada), clasifica cada afirmación como hecho respaldado, hipótesis por validar o información insuficiente. Incluye evidencia, límite interpretativo y dato adicional necesario. No atribuyas causas ni asumas que un registro ausente equivale a atención no realizada.','Matriz de hechos, hipótesis y preguntas.'),
('Proponer acciones e indicadores','Generar opciones preliminares vinculadas a evidencia y métricas.','Propón hasta 3 acciones preliminares para discusión de una EPS a partir de estos hallazgos ficticios: [PEGA HALLAZGOS VALIDADOS]. Para cada acción incluye hallazgo, hipótesis por validar, responsable funcional, indicador, fórmula y datos necesarios. No inventes metas, costos o resultados; no recomiendes tratamientos individuales. Distingue propuesta de decisión aprobada.','Matriz de acciones e indicadores por validar.')]),
'Operaciones y administración de IPS': ('Capacidad operativa, oportunidad, cancelaciones y mejora de procesos.', [
('Analizar retrasos y cancelaciones','Diseñar un análisis descriptivo antes de buscar causas.','Diseña un análisis descriptivo para datos ficticios de citas y retrasos. Indica variables, definiciones, segmentaciones y métricas útiles. Separa lo que puede medirse de las causas posibles que requieren evidencia. No inventes resultados ni atribuyas responsabilidades sin datos.\n\nColumnas disponibles: [PEGA ENCABEZADOS]','Plan de análisis descriptivo.'),
('Investigar un problema operativo','Convertir una queja general en preguntas comprobables.','Estructura una investigación inicial sobre demoras en una IPS. No asumas la causa. Entrega: definición medible del problema, preguntas de diagnóstico, datos necesarios, segmentos de análisis, hipótesis comprobables y criterios para evaluar una intervención. Distingue hechos de supuestos.\n\nContexto ficticio: [DESCRIBE PROBLEMA]','Guía breve de diagnóstico operativo.'),
('Resumir tiempos de espera','Identificar patrones sin ocultar limitaciones.','Resume esta tabla ficticia de tiempos de espera: periodo, número de registros, estadísticas disponibles, diferencias entre grupos solo si los datos lo permiten, posibles atípicos y limitaciones. No inventes estadísticas ni explicaciones.\n\nDatos: [PEGA DATOS FICTICIOS]','Resumen operativo con limitaciones.'),
('Diseñar un piloto de mejora','Proponer una prueba pequeña con medición y seguimiento.','Propón un plan preliminar de mejora basado solo en estos hallazgos ficticios: [PEGA HALLAZGOS]. Incluye problema medible, intervención piloto, proceso afectado, responsable funcional, indicador y fórmula, frecuencia de revisión y riesgos. No prometas resultados ni inventes metas. Indica qué debe validarse antes de ejecutar.','Ficha de piloto y medición.')]),
'Auditoría, calidad y regulación': ('Revisión documental, criterios, clasificación de hallazgos y trazabilidad.', [
('Clasificar motivos ficticios de glosa','Aplicar criterios entregados y señalar casos ambiguos.','Clasifica estos casos ficticios solo con los criterios suministrados. Para cada uno incluye texto del caso, criterio posiblemente aplicable, evidencia textual, confianza cualitativa y dato faltante. Si no es suficiente, escribe “Requiere revisión humana”. No inventes normas ni decidas automáticamente quién tiene razón.\n\nCriterios: [PEGA CRITERIOS]\nCasos: [PEGA CASOS FICTICIOS]','Tabla preliminar de clasificación.'),
('Revisar un documento contra criterios','Crear una lista de verificación trazable.','Revisa el documento ficticio contra estos criterios. Por criterio devuelve evidencia textual exacta, estado (cumple, no cumple, no evaluable) y explicación. No infieras evidencia ni inventes requisitos. Termina con puntos para revisión del auditor.\n\nCriterios: [PEGA CRITERIOS]\nDocumento: [PEGA DOCUMENTO FICTICIO]','Lista de verificación trazable.'),
('Clasificar hallazgos frente a criterios','Construir una matriz de revisión con evidencia.','Compara cada extracto ficticio con los criterios. Devuelve identificador, criterio, fragmento de evidencia, clasificación (posible incumplimiento, evidencia de cumplimiento o no evaluable), justificación y pregunta de validación. No conviertas ausencia de evidencia en prueba automática de incumplimiento ni inventes normativa.\n\nCriterios: [PEGA CRITERIOS]\nExtractos: [PEGA EXTRACTOS]','Matriz de hallazgos preliminares.'),
('Detectar conclusiones no sustentadas','Auditar una respuesta de IA contra su fuente.','Compara la respuesta de IA con el documento fuente ficticio. Clasifica cada afirmación como respaldada, parcialmente respaldada, no respaldada o contradictoria. Cita el fragmento pertinente e identifica omisiones, inferencias y datos inventados. No decidas el caso; indica qué debe validar el auditor.\n\nFuente: [PEGA DOCUMENTO]\nRespuesta IA: [PEGA RESPUESTA]','Informe breve de verificación.')]),
'Dirección, investigación y salud pública': ('Síntesis de evidencia, investigación clínica, salud pública y decisiones institucionales.', [
('Resumir un informe ejecutivo','Definir audiencia, estructura y límites de un resumen.','Resume este informe ficticio para una reunión directiva: propósito, 3 hallazgos, evidencia de cada uno, limitaciones, decisiones que podrían discutirse y preguntas abiertas. Distingue lo que afirma el documento de tus inferencias. No agregues datos externos.\n\nInforme: [PEGA INFORME FICTICIO]','Resumen ejecutivo trazable.'),
('Extraer hallazgos y limitaciones','Leer críticamente un documento de investigación o salud pública.','Extrae del documento ficticio: objetivo, diseño si está explícito, población/fuente, hallazgos reportados, limitaciones mencionadas y preguntas abiertas. Para cada hallazgo cita una frase breve. Si falta un elemento, indica “No especificado”. No atribuyas causalidad sin respaldo.\n\nDocumento: [PEGA DOCUMENTO FICTICIO]','Ficha de lectura crítica.'),
('Separar evidencia, incertidumbre y opciones','Preparar una conversación institucional sin convertir sugerencias en decisiones.','Organiza el documento ficticio en: evidencia explícita, incertidumbres/datos faltantes y opciones de acción para discusión. Para cada opción indica evidencia que la motiva y qué debe validarse. No presentes opciones como decisiones aprobadas ni añadas datos externos.\n\nDocumento: [PEGA DOCUMENTO FICTICIO]','Matriz de evidencia, incertidumbre y opciones.'),
('Preparar una síntesis ejecutiva responsable','Comunicar hechos, límites y próximos pasos de validación.','Prepara una síntesis ejecutiva breve con contexto, hallazgos respaldados, implicaciones posibles, limitaciones, preguntas pendientes y próximos pasos de validación. Evita lenguaje categórico cuando haya incertidumbre. No inventes cifras ni causalidad.\n\nDocumento: [PEGA DOCUMENTO FICTICIO]','Síntesis ejecutiva para revisión humana.')])
}

with st.sidebar:
    st.header('Explora el laboratorio')
    session = st.radio('Sesión', ['Módulo 2 — Sesión 1: prompts e interacción', 'Módulo 2 — Sesión 2: aplicación y revisión crítica'])
    sidx = 0 if 'Sesión 1' in session else 1
    segment = st.selectbox('Contexto profesional', list(DATA.keys()))
    st.caption('MVP educativo: sin API, cuentas ni base de datos.')

st.subheader(segment)
st.write(DATA[segment][0])
all_exercises = DATA[segment][1]
exercises = all_exercises[sidx*2:sidx*2+2]
title = st.selectbox('Elige un ejercicio', [e[0] for e in exercises])
e = next(x for x in exercises if x[0] == title)
left, right = st.columns([1.15, .85], gap='large')
with left:
    st.markdown('### Situación y reto')
    st.write(e[1])
    st.markdown('### Prompt sugerido')
    st.caption('Cópialo desde el bloque y reemplaza los campos entre corchetes con información ficticia.')
    st.code(e[2], language=None)
    st.markdown('### Producto esperado')
    st.info(e[3])
with right:
    st.markdown('### Prueba con un asistente')
    st.write('Abre una herramienta externa, pega el prompt y revisa la respuesta.')
    st.link_button('Abrir ChatGPT', 'https://chatgpt.com/')
    st.link_button('Abrir Microsoft Copilot', 'https://copilot.microsoft.com/')
    st.markdown('### Lista de verificación')
    for i, q in enumerate(['¿La respuesta se basa en la información suministrada?', '¿Señala datos faltantes y límites?', '¿Distingue hechos de hipótesis?', '¿Evita inventar datos o conclusiones?', '¿Indica qué debe validar una persona responsable?']):
        st.checkbox(q, key=f'{sidx}_{segment}_{title}_{i}')
    st.text_area('Resultado (opcional; solo datos ficticios)', height=150, placeholder='Pega aquí la respuesta para discutirla. No se envía a un modelo ni se guarda en una base de datos.')
st.divider()
st.subheader('Preguntas de cierre')
st.markdown('- ¿Qué afirmaciones están respaldadas por la entrada?\n- ¿Qué falta o necesita validación?\n- ¿Qué decisión puede apoyar la IA y cuál sigue bajo responsabilidad humana?')
st.caption('Laboratorio educativo. No sustituye protocolos institucionales, fuentes verificadas ni criterio profesional.')
