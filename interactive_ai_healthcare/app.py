import streamlit as st
import re
import html
import textwrap
import zipfile
from pathlib import Path

st.set_page_config(page_title="Casos prácticos de IA en salud", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.block-container {
    max-width: 1180px;
    padding-top: 1.1rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(120deg, #e7f3ff, #e2f7ef, #fff4d8);
    border: 1px solid #cfe2e8;
    border-radius: 22px;
    padding: 22px 26px;
    margin-bottom: 14px;
}

.hero h1 {
    color: #17476b;
    margin: 0 0 7px;
    font-size: 2rem;
}

.hero p {
    margin: 0;
    color: #466273;
    font-size: 1.02rem;
}

.intro {
    background: #f4f8fb;
    border-left: 6px solid #2f80c9;
    border-radius: 12px;
    padding: 14px 17px;
    margin: 12px 0 20px;
    color: #28485b;
}

.sectionbox {
    border-radius: 16px;
    padding: 18px 20px;
    margin: 12px 0 18px;
    border: 1px solid;
}

.blue {
    background: #eaf4ff;
    border-color: #b8d8f5;
}

.green {
    background: #e8f7ef;
    border-color: #b9e4cc;
}

.yellow {
    background: #fff7df;
    border-color: #f0d88a;
}

.purple {
    background: #f2edff;
    border-color: #d6c9f7;
}

.sectionbox h4 {
    margin: 0 0 7px;
    color: #17476b;
}

.sectionbox p {
    margin: 0;
    color: #3e5a6a;
}


/* Default prompt background */
div[data-testid="stCode"] {
    background-color: #f8fafc !important;
    border: 1px solid #d9e4ed !important;
    border-radius: 12px !important;
}

div[data-testid="stCode"] pre,
div[data-testid="stCode"] code {
    background-color: transparent !important;
}

/* Match each prompt to its preceding title color */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.blue)
    + div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stCode"] {
    background-color: #eaf4ff !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.purple)
    + div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stCode"] {
    background-color: #f2edff !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.yellow)
    + div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stCode"] {
    background-color: #fff7df !important;
}


/* Main instruction labels */
div[data-testid="stSelectbox"] [data-testid="stWidgetLabel"] p,
div[data-testid="stRadio"] [data-testid="stWidgetLabel"] p {
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    line-height: 1.5 !important;
    color: #17476b !important;
}

/* Selected text in dropdowns */
div[data-baseweb="select"] * {
    font-size: 1.15rem !important;
}

/* Dropdown menu options */
div[role="listbox"] *,
ul[role="listbox"] * {
    font-size: 1.15rem !important;
}

/* Small explanatory text */
.small {
    font-size: 0.9rem;
    color: #5c7180;
}

/* Buttons */
div.stButton > button {
    border-radius: 10px;
    font-weight: 650;
    min-height: 2.5rem;
}

/* Links */
a {
    font-weight: 600;
}


</style>
""", unsafe_allow_html=True)

st.markdown("<div class='hero'><h1>🩺 Laboratorio de casos prácticos de IA en salud</h1><p>Practica con situaciones cercanas a tu trabajo. Cada caso incluye hasta tres prompts conectados: comprender, aplicar y verificar.</p></div>", unsafe_allow_html=True)
st.markdown("<div class='intro'><b>Cómo usar el laboratorio</b><br>1. Elige el segmento de tu interes.<br> 2. Abre un caso y copia sus prompts en el asistente de IA que prefieras.<br> 3. Sigue el orden de los prompts: el segundo prompt revisa o profundiza el primero; el tercero convierte el resultado en un producto útil y pide verificaciones.<br> Usa solo los datos ficticios y no ingreses información identificable de pacientes.</div>", unsafe_allow_html=True)

st.markdown("### Asistentes de IA para probar los prompts")
links = [
    ("ChatGPT", "https://chatgpt.com/", "#eaf3ff"),
    ("Gemini", "https://gemini.google.com/", "#f2edff"),
    ("Microsoft Copilot", "https://copilot.microsoft.com/", "#e5f7f1"),
    ("Perplexity", "https://www.perplexity.ai/", "#fff5dc"),
    (“Claude", "https://claude.ai/login", "#d97757"),
]
link_cols = st.columns(4)
for col, (label, url, color) in zip(link_cols, links):
    with col:
        st.markdown(f"<div style='background:{color};border-radius:12px;padding:13px;text-align:center'><a href='{url}' target='_blank'>{label} ↗</a></div>", unsafe_allow_html=True)
st.caption("Usa únicamente herramientas autorizadas por tu institución. No pegues datos reales, identificables o confidenciales.")

# Each prompt sequence is specific to the segment and case, and follows a connected workflow.
SEGMENTS = {
"Atención clínica y rehabilitación": {
"Aplicación práctica y prompt engineering": [
 {"title":"Consulta externa: preparar una nota clínica sin inventar datos", "context":"Caso: consulta de fisioterapia por dolor de hombro. La nota inicial dice: ‘dolor al elevar el brazo desde hace dos semanas; paciente quiere volver a nadar’. No incluye examen físico, antecedentes ni evaluación funcional.", "prompts":[
  "Actúa como asistente de documentación clínica, no como diagnosticador. Con esta nota ficticia: ‘dolor al elevar el brazo desde hace dos semanas; paciente quiere volver a nadar’. Organiza la información en: motivo de consulta, evolución referida, objetivo del paciente y datos faltantes por preguntar. No inventes examen físico, antecedentes ni diagnóstico.",
  "Revisa la estructura que acabas de proponer. Separa explícitamente los datos aportados de los campos pendientes. Sugiere hasta cinco preguntas neutrales que el profesional podría considerar para completar la valoración, sin indicar tratamiento ni asumir una causa del dolor.",
  "Convierte la estructura revisada en una plantilla breve de nota inicial para fisioterapia, con campos editables y una sección ‘por confirmar’. Añade una lista final para que el profesional verifique exactitud, pertinencia y consentimiento antes de incorporar cualquier texto a la historia clínica."
 ]},
 {"title":"Resumen rápido de una evolución para preparar la consulta", "context":"Caso: tres notas consecutivas describen que una persona mayor inició rehabilitación tras una caída; persiste dificultad para caminar y se menciona una recomendación de apoyo, pero las notas no precisan si se entregó el dispositivo.", "prompts":[
  "Resume este caso para preparar una consulta: ‘tras una caída inició rehabilitación; persiste dificultad para caminar; se menciona recomendación de apoyo, pero no consta si se entregó’. Devuelve máximo cinco viñetas bajo: evolución documentada, situación funcional descrita, acciones mencionadas y datos que no constan. No infieras diagnósticos.",
  "Revisa el resumen anterior como si fueras un segundo revisor. Marca cada frase como ‘explícita en la nota’ o ‘requiere confirmación’. Identifica si se confundió una recomendación con una acción realizada y corrige el resumen para evitar esa confusión.",
  "Entrega una versión final de máximo 80 palabras para el profesional que recibirá al paciente, seguida de tres preguntas de continuidad asistencial. Mantén la incertidumbre donde corresponda y no añadas recomendaciones terapéuticas."
 ]}
],
"Extraer, explicar y verificar": [
 {"title":"Extraer información clínica y diagnósticos ya documentados", "context":"Caso de egreso: ‘Consulta por tos y fiebre. En impresión diagnóstica se registra infección respiratoria aguda. Se indica control por consulta externa. No se consigna alergia en este resumen’.", "prompts":[
  "Extrae únicamente la información explícita de este resumen: ‘Consulta por tos y fiebre. En impresión diagnóstica se registra infección respiratoria aguda. Se indica control por consulta externa. No se consigna alergia en este resumen’. Organiza en: síntomas referidos, diagnóstico escrito en la fuente, plan documentado y dato no consignado. No propongas diagnósticos nuevos.",
  "Verifica la extracción contra el texto original. Para cada elemento, incluye la frase de origen que lo respalda. Distingue ‘no consignado en este resumen’ de ‘el paciente no tiene ese antecedente’. Señala cualquier dato que no pueda confirmarse.",
  "Genera una tabla de revisión humana con columnas: campo extraído, texto fuente, estado (confirmado/no consignado/ambiguo) y acción de verificación. No completes campos vacíos ni conviertas la tabla en una recomendación clínica."
 ]},
 {"title":"Explicar instrucciones de seguimiento en lenguaje sencillo", "context":"Caso: ‘Solicitar control ambulatorio en 7 días y llevar resultados de exámenes disponibles’. El equipo quiere preparar una explicación comprensible para el paciente.", "prompts":[
  "Reescribe en lenguaje sencillo esta instrucción ficticia: ‘Solicitar control ambulatorio en 7 días y llevar resultados de exámenes disponibles’. Conserva exactamente el plazo y la acción solicitada. No agregues síntomas de alarma ni instrucciones que no estén en el texto.",
  "Compara la versión sencilla con la instrucción original. Lista qué información se conservó y si se añadió algo no respaldado. Corrige cualquier cambio de significado y evita tecnicismos innecesarios.",
  "Prepara una versión final de dos frases y una pregunta de comprobación para confirmar que el paciente entendió. Indica al profesional que valide la redacción y añada, por separado, cualquier indicación clínica autorizada que falte."
 ]}
]},
"Aseguramiento y gestión del riesgo en salud (EPS)": {
"Aplicación práctica y prompt engineering": [
 {"title":"Seguimiento de personas con diabetes: construir un prompt de análisis", "context":"Caso de una EPS: cohorte de 200 personas con diabetes tipo 2; 132 tienen control registrado en el trimestre y 68 no tienen control registrado. No se conocen aún las razones de los registros faltantes.", "prompts":[
  "Actúa como analista de gestión del riesgo de una EPS. Con estos datos ficticios: 200 personas con diabetes tipo 2; 132 con control registrado y 68 sin control registrado, calcula las proporciones y muestra la fórmula. Aclara que ‘sin registro’ no demuestra que no hubo atención. No atribuyas causas.",
  "Revisa los cálculos y la interpretación anterior. Comprueba que los denominadores sumen la cohorte y enumera tres explicaciones posibles para un registro faltante, etiquetándolas como hipótesis que requieren validación, no como hallazgos.",
  "Mejora el prompt para que un equipo de la EPS pueda repetir el análisis cada mes. Incluye variables requeridas, reglas para evitar duplicados, fecha de corte, formato de salida y una advertencia explícita contra inferir falta de atención a partir de ausencia de registro."
 ]},
 {"title":"Preparar una reunión de seguimiento de riesgo sin sobregeneralizar", "context":"Caso: en un listado de seguimiento hay personas con control pendiente, citas sin confirmar y resultados de laboratorio no visibles. No se ha conciliado la base con prestadores.", "prompts":[
  "A partir de este caso de EPS —control pendiente, citas sin confirmar y resultados de laboratorio no visibles— crea una tabla que separe cada señal operativa, qué se sabe y qué falta validar. No asumas que las personas no recibieron atención.",
  "Para cada señal de la tabla, formula una pregunta concreta que el equipo de gestión del riesgo pueda resolver contrastando la base con el prestador. Prioriza calidad de datos, continuidad de atención y trazabilidad; no recomiendes cambios de tratamiento.",
  "Redacta una agenda de reunión de 15 minutos con tres decisiones operativas posibles, los datos que deben revisarse antes de decidir y un responsable funcional sugerido. Marca las decisiones como pendientes de validación, no como acciones ya aprobadas."
 ]}
],
"Extraer, explicar y verificar": [
 {"title":"Extraer diagnósticos documentados y brechas de seguimiento", "context":"Caso de registro: diagnóstico escrito ‘diabetes mellitus tipo 2’; último control registrado hace 5 meses; HbA1c no visible en la base consultada; cita de seguimiento aparece como pendiente.", "prompts":[
  "Extrae del registro: diagnóstico tal como aparece escrito, fecha relativa del último control, estado del registro de HbA1c y estado de la cita. Usa ‘no visible en la base consultada’ en vez de asumir que el examen no se realizó. No clasifiques gravedad ni recomiendes tratamiento.",
  "Verifica cada campo con una columna de evidencia textual y otra de limitación. Identifica qué requiere confirmar el prestador o la fuente de datos antes de clasificar a la persona dentro de una ruta de seguimiento.",
  "Entrega una tabla operativa para el equipo EPS: campo, dato extraído, evidencia, verificación pendiente y área responsable. No incluyas nombres ni identificadores y no infieras condiciones clínicas no registradas."
 ]},
 {"title":"Comunicar un pendiente de seguimiento en lenguaje sencillo", "context":"Caso: mensaje interno dice ‘No se evidencia cargue del resultado de HbA1c y la cita de control figura pendiente de confirmación’. Debe convertirse en un mensaje claro para el equipo de contacto.", "prompts":[
  "Reescribe este mensaje para un equipo de contacto de una EPS: ‘No se evidencia cargue del resultado de HbA1c y la cita de control figura pendiente de confirmación’. Mantén la diferencia entre información no visible y actividad no realizada. No afirmes que la persona incumplió.",
  "Revisa el mensaje para detectar lenguaje que pueda culpabilizar a la persona o afirmar hechos no verificados. Propón una versión neutral que invite a confirmar la información y verificar el estado de la cita.",
  "Prepara un guion breve de contacto que primero confirme si la cita se realizó y si el examen está disponible, y después explique el siguiente paso según la respuesta. No des recomendaciones clínicas; deja cualquier decisión asistencial al personal autorizado."
 ]}
]},
"Operaciones y administración de IPS": {
"Aplicación práctica y prompt engineering": [
 {"title":"Quejas por espera: preparar un análisis operativo", "context":"Caso de consulta externa en una IPS: 30 citas en un día; espera media de 42 minutos; 8 citas superaron 60 minutos. No hay datos de otros días ni de especialidad o franja horaria.", "prompts":[
  "Actúa como apoyo de mejora de procesos de una IPS. Resume los datos: 30 citas, espera media 42 minutos y 8 esperas mayores de 60 minutos en un solo día. Calcula el porcentaje de citas que superaron 60 minutos y aclara qué no se puede concluir con un solo día de datos.",
  "A partir del resumen, propone hasta cuatro hipótesis operativas comprobables (por ejemplo, concentración por franja horaria), sin afirmar que sean causas. Para cada hipótesis, especifica qué variable se debería recolectar para comprobarla.",
  "Redacta un prompt reutilizable para analizar semanalmente la oportunidad de consulta. Incluye segmentación por servicio y hora, definición del inicio y fin de la espera, tratamiento de datos faltantes y la obligación de separar hallazgos de hipótesis."
 ]},
 {"title":"Rediseñar un flujo de admisiones con datos incompletos", "context":"Caso: el personal reporta que algunas admisiones se demoran porque faltan documentos o hay que corregir datos de autorización; no se han medido frecuencia ni duración por causa.", "prompts":[
  "Con este reporte de admisiones —documentos faltantes y correcciones de autorización— crea un mapa preliminar del proceso con pasos conocidos y puntos por observar. Etiqueta lo que es reporte del personal y lo que aún no está medido.",
  "Diseña una tabla sencilla para registrar durante una semana: hora de llegada, inicio y fin de admisión, tipo de trámite, causa de reproceso observada y resultado. Evita incluir datos personales innecesarios y define cómo registrar ‘causa desconocida’.",
  "Usando la tabla propuesta, plantea un piloto de mejora pequeño que solo se active tras revisar los datos. Incluye indicador de tiempo, indicador de reproceso, riesgo de trasladar la demora a otra etapa y responsable funcional; no inventes metas."
 ]}
],
"Extraer, explicar y verificar": [
 {"title":"Extraer y estructurar motivos de reproceso", "context":"Caso: tres notas operativas dicen: ‘faltaba autorización’, ‘documento ilegible’ y ‘se corrigió el número de orden’. No hay categorías oficiales definidas.", "prompts":[
  "Clasifica estas notas ficticias: ‘faltaba autorización’, ‘documento ilegible’, ‘se corrigió el número de orden’. Propón una categoría provisional para cada una y conserva el texto original. No afirmes que las categorías son oficiales.",
  "Revisa si las categorías propuestas se solapan o dependen de interpretación. Sugiere un pequeño diccionario de categorías con definiciones y un estado ‘requiere revisión’ para los casos ambiguos.",
  "Entrega una plantilla de registro para analizar reprocesos en admisiones: texto original, categoría provisional, etapa del proceso, evidencia, revisión requerida y acción siguiente. No añadas causas que no aparezcan en las notas."
 ]},
 {"title":"Explicar un cambio de proceso en lenguaje claro", "context":"Caso: la IPS quiere pedir que los documentos se revisen antes de asignar turno de admisión, pero el cambio aún no está aprobado.", "prompts":[
  "Explica en lenguaje claro para el personal de admisiones esta propuesta aún no aprobada: ‘revisar documentos antes de asignar turno’. Distingue propuesta de procedimiento vigente y no inventes nuevas obligaciones.",
  "Revisa el texto para detectar si suena como una instrucción ya aprobada. Reescríbelo para invitar a evaluar un piloto, indicando qué aspectos operativos necesitan validación.",
  "Prepara un mensaje breve para socializar el piloto con campos editables para fecha, área y responsable. Incluye cómo reportar problemas y recuerda que el proceso actual sigue vigente hasta que la institución autorice el cambio."
 ]}
]},
"Auditoría, calidad y regulación": {
"Aplicación práctica y prompt engineering": [
 {"title":"Revisión documental: instruir a la IA para no inventar criterios", "context":"Caso: criterio interno del ejercicio indica que el formato debe tener fecha y firma legible. Registro A tiene fecha y firma visibles; B tiene fecha y firma ilegible; C no muestra fecha en la copia.", "prompts":[
  "Actúa como asistente de organización de evidencias para auditoría, no como autoridad regulatoria. Clasifica A: fecha y firma visibles; B: fecha visible y firma ilegible; C: fecha no visible en la copia. Usa estados ‘evidencia visible’, ‘no legible’ y ‘no visible en copia’; no cites normas externas.",
  "Verifica la clasificación contra cada descripción original. Distingue entre ‘no visible en esta copia’ y ‘no existe en el expediente’. Señala qué debe comprobar un auditor en la fuente antes de registrar un incumplimiento.",
  "Mejora el prompt para futuras revisiones documentales. Incluye criterio suministrado, evidencia textual, estado, limitación, trazabilidad y obligación de no inferir incumplimientos ni inventar normativa."
 ]},
 {"title":"Convertir notas de hallazgo en observaciones trazables", "context":"Caso: nota del auditor: ‘en dos registros la fecha no se distingue en la imagen; se requiere revisar el original’. No se ha verificado el expediente fuente.", "prompts":[
  "Organiza la nota ficticia del auditor en: observación, evidencia disponible, alcance de la revisión y acción pendiente. No la conviertas en un hallazgo definitivo porque aún no se ha revisado el original.",
  "Revisa el texto y marca expresiones que impliquen más certeza de la que permite la evidencia. Sugiere redacción neutral y conserva explícitamente la necesidad de verificar el expediente fuente.",
  "Crea una ficha de seguimiento con campos para identificador interno no nominativo, criterio, evidencia, estado de verificación, revisor y fecha de cierre. No inventes resultados ni conclusiones normativas."
 ]}
],
"Extraer, explicar y verificar": [
 {"title":"Extraer datos clave y diagnósticos documentados de una auditoría", "context":"Caso: en una nota de egreso se lee ‘diagnóstico registrado: neumonía adquirida en la comunidad’; la firma aparece borrosa en el archivo digital; el documento original aún no se ha consultado.", "prompts":[
  "Extrae de la nota ficticia solo estos campos: diagnóstico tal como está escrito, estado de legibilidad de la firma en el archivo digital y verificación pendiente. No valides clínicamente el diagnóstico ni concluyas que el documento original carece de firma.",
  "Crea una tabla de evidencia con el fragmento fuente, campo extraído, certeza, limitación y paso de verificación. Diferencia ‘borroso en archivo digital’ de ‘ausente en el original’.",
  "Redacta una observación provisional para revisión humana que describa el problema de legibilidad sin declarar incumplimiento definitivo. Añade qué evidencia debe consultar el auditor antes de cerrar el caso."
 ]},
 {"title":"Explicar un hallazgo documental sin tecnicismos", "context":"Caso: el equipo debe comunicar a un área clínica que una firma no se puede leer en la copia digital y que el original debe revisarse antes de concluir.", "prompts":[
  "Traduce a lenguaje sencillo esta nota: ‘la firma no es legible en la copia digital y debe consultarse el original antes de concluir’. Conserva la incertidumbre y evita palabras que sugieran incumplimiento ya confirmado.",
  "Comprueba que la versión sencilla no cambie el sentido ni convierta una limitación de la copia en una afirmación sobre el documento original. Señala cualquier frase que necesite corrección.",
  "Entrega un mensaje breve y respetuoso para el área responsable: explica qué se observó, qué falta verificar y cuál es el siguiente paso documental. No atribuyas responsabilidad ni cites normas no proporcionadas."
 ]}
]},
"Dirección, investigación y salud pública": {
"Aplicación práctica y prompt engineering": [
 {"title":"Encuesta de acceso a citas: resumir sin generalizar", "context":"Caso: encuesta por conveniencia con 120 participantes; 45 reportaron dificultad para conseguir cita. No se conoce si la muestra representa a toda la población usuaria.", "prompts":[
  "Resume este resultado para un equipo directivo: encuesta por conveniencia, 120 participantes, 45 reportaron dificultad para conseguir cita. Calcula la proporción y aclara que no necesariamente representa a toda la población. No atribuyas causas.",
  "Revisa el resumen e identifica conclusiones que excedan la muestra o impliquen causalidad. Reescribe cada una como hallazgo limitado o pregunta pendiente de investigación.",
  "Mejora el prompt para futuras síntesis de encuestas en salud pública. Exige tamaño y tipo de muestra, denominador, resultado, limitaciones, posibles sesgos y separación explícita entre hallazgo e hipótesis."
 ]},
 {"title":"Resumir un informe largo para una reunión de dirección", "context":"Caso: informe de 12 páginas sobre barreras de acceso contiene resultados de encuesta, testimonios y recomendaciones; el equipo necesita un resumen para una reunión de 10 minutos.", "prompts":[
  "A partir de un informe sobre barreras de acceso, prepara una estructura de resumen ejecutivo de una página con: pregunta, método, hallazgos cuantitativos, testimonios, limitaciones y recomendaciones del propio informe. No inventes cifras ni atribuyas recomendaciones que no estén en la fuente.",
  "Revisa el resumen contra el informe fuente. Para cada cifra o recomendación, solicita página o fragmento de respaldo. Marca como ‘sin respaldo localizado’ cualquier afirmación que no puedas vincular con la fuente.",
  "Convierte el resumen verificado en una agenda de discusión de 10 minutos: decisiones que sí pueden tomarse con la evidencia actual, preguntas abiertas y datos adicionales necesarios. Distingue evidencia de opinión."
 ]}
],
"Extraer, explicar y verificar": [
 {"title":"Extraer resultados y categorías desde una síntesis de investigación", "context":"Caso: una síntesis informa que 45 de 120 participantes reportaron dificultades de acceso; los temas cualitativos mencionados incluyen horarios, transporte y disponibilidad de citas. No se proporciona frecuencia por tema.", "prompts":[
  "Extrae los datos explícitos del caso: 45 de 120 reportaron dificultades y los temas mencionados fueron horarios, transporte y disponibilidad de citas. Calcula el porcentaje global; no asignes frecuencias a los temas porque no se proporcionaron.",
  "Verifica el cálculo y revisa si la redacción permite confundir temas mencionados con causas demostradas. Produce una versión corregida que distinga resultado cuantitativo, temas cualitativos y límites de la información.",
  "Entrega una tabla para el equipo de investigación con: resultado, fuente, método o muestra, limitación, dato que falta y uso permitido. No generalices más allá de la muestra ni hagas afirmaciones causales."
 ]},
 {"title":"Comunicar resultados a una audiencia no técnica", "context":"Caso: el equipo necesita explicar a una comunidad el resultado de una encuesta de acceso a citas, sin sugerir que describe necesariamente a todas las personas usuarias.", "prompts":[
  "Explica en lenguaje sencillo este resultado: 45 de 120 personas encuestadas reportaron dificultad para conseguir cita; la muestra fue por conveniencia. Evita jerga estadística y aclara que el resultado no representa necesariamente a todas las personas usuarias.",
  "Revisa el mensaje en busca de generalizaciones, lenguaje alarmista o conclusiones causales. Propón una versión más clara y respetuosa que conserve la limitación de la muestra.",
  "Prepara una versión oral de 45 segundos y una pregunta para escuchar experiencias de la comunidad. No presentes la encuesta como diagnóstico completo del sistema ni prometas acciones no aprobadas."
 ]}
]}
}

# Top navigation: the two parts are parallel sections with different class themes, both case-based.
seg = st.selectbox("1. Selecciona un segmento", list(SEGMENTS.keys()), key="segment")
part = st.radio("2. Selecciona un tema", ["Aplicación práctica y prompt engineering", "Extraer, explicar y verificar"], horizontal=True, key="part")

if part == "Aplicación práctica y prompt engineering":
    st.markdown("<div class='sectionbox blue'><h4> Aplicación práctica y prompt engineering</h4><p><b>Temas de la sesión:</b> <br> • Uso de ChatGPT, Copilot y otros asistentes en el entorno clínico <br>• Formulación de instrucciones claras y seguras <br>• Copilotos clínicos para redactar notas y evoluciones <br>• Resumen de historias clínicas extensas para consultas rápidas.<br>Trabajaremos estos temas mediante casos de cada segmento.</p></div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='sectionbox green'><h4>Extraer, explicar y verificar</h4><p><b>Temas de la sesión:</b> <br>• Extracción de información clave y diagnósticos ya documentados <br>• Traducción de lenguaje técnico a lenguaje sencillo <br>• Buenas prácticas y riesgos, incluidas alucinaciones, verificación de información y uso responsable.<br> Trabajaremos estos temas mediante casos de cada segmento.</p></div>", unsafe_allow_html=True)

cases = SEGMENTS[seg][part]
st.markdown(f"### Casos prácticos: {seg}")
case_titles = [c["title"] for c in cases]
chosen_title = st.selectbox("3. Elige un caso", case_titles, key=f"case::{seg}::{part}")
case = next(c for c in cases if c["title"] == chosen_title)
st.markdown(f"<div class='sectionbox yellow'><h4>Situación del caso</h4><p>{case['context']}</p></div>", unsafe_allow_html=True)
st.markdown("### Flujo de Prompts")
st.caption("El Prompt 1 incluye la estructura completa. Ejecuta los tres prompts en la misma conversación: los prompts 2 y 3 son más breves y construyen sobre la respuesta anterior.")
step_colors = ["blue", "purple", "yellow"]
if part == "Aplicación práctica y prompt engineering":
    step_names = [
        "Prompt 1 · Dar contexto e instrucciones claras",
        "Prompt 2 · Refinar y revisar la respuesta",
        "Prompt 3 · Preparar un producto para revisión profesional",
    ]
else:
    step_names = [
        "Prompt 1 · Extraer o explicar información de la fuente",
        "Prompt 2 · Verificar evidencia y detectar errores",
        "Prompt 3 · Entregar un resultado verificado y utilizable",
    ]

def readable_prompt(prompt: str, width: int = 72) -> str:
    """Break every prompt into readable lines while preserving its wording."""
    text = prompt.strip().replace("\r\n", "\n").replace("\r", "\n")
    # Separate sentences and key instructions into paragraphs first.
    text = re.sub(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÜÑ¿¡])", "\n\n", text)
    text = re.sub(
        r":\s+(?=(?:Organiza|Devuelve|Entrega|Incluye|Separa|Marca|Distingue|Calcula|Conserva|No inventes|No añadas|No asumas|No propongas|Identifica|Sugiere|Prepara|Genera|Revisa|Comprueba|Enumera|Para cada|Usa solo|A partir|Con base|Transforma|Extrae|Clasifica|Redacta|Explica|Actúa|Resume|Convierte|Crea|Diseña|Propón|Mantén|Aclara|Limita|Solicita|Pide|Verifica)\b)",
        ":\n",
        text,
    )
    # Hard-wrap long sentences so code blocks are readable on desktop and mobile.
    wrapped_paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text):
        lines = paragraph.splitlines()
        if not paragraph.strip():
            continue
        wrapped_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Preserve list formatting and indent wrapped continuation lines.
            prefix = ""
            content = line
            match = re.match(r"^(\s*(?:[-•*]|\d+[.)])\s+)(.*)$", line)
            if match:
                prefix, content = match.group(1), match.group(2)
            wrapped_lines.append(textwrap.fill(
                content,
                width=max(40, width - len(prefix)),
                initial_indent=prefix,
                subsequent_indent=" " * len(prefix) if prefix else "",
                break_long_words=False,
                break_on_hyphens=False,
            ))
        wrapped_paragraphs.append("\n".join(wrapped_lines))
    return "\n\n".join(wrapped_paragraphs)

prompt_backgrounds = {
    "blue": "#eaf4ff",
    "purple": "#f2edff",
    "yellow": "#fff7df",
}

def build_structured_prompt(prompt: str, case: dict, segment: str, part: str, step: int) -> str:
    """Use a fully structured first prompt, then concise follow-ups that build on prior output."""
    if part == "Aplicación práctica y prompt engineering":
        role = (
            "Actúa como copiloto de apoyo al trabajo en salud. Tu función es organizar, resumir o redactar "
            "información para revisión profesional; no reemplazas al personal de salud ni tomas decisiones clínicas."
        )
        objective = "Apoyar una tarea concreta del flujo clínico, administrativo o de gestión descrito en el caso."
        base_constraints = [
            "Trabaja únicamente con la información proporcionada.",
            "No inventes datos, antecedentes, diagnósticos, resultados, acciones realizadas ni normas.",
            "Distingue los hechos documentados de los datos faltantes, las hipótesis y las propuestas.",
            "Usa lenguaje claro, respetuoso y adecuado al equipo destinatario.",
            "La salida debe ser revisada por una persona autorizada antes de utilizarse."
        ]
        format_instruction = (
            "Respeta el formato y la extensión solicitados en la tarea. Si no se especifican, "
            "organiza la respuesta con subtítulos y viñetas breves."
        )
    else:
        role = (
            "Actúa como asistente de extracción, comunicación y control de calidad de información en salud. "
            "No confirmas diagnósticos por cuenta propia ni sustituyes la verificación humana."
        )
        objective = (
            "Extraer, explicar o verificar información sin alterar el significado de la fuente "
            "ni añadir afirmaciones no respaldadas."
        )
        base_constraints = [
            "Usa la fuente proporcionada como referencia principal y no agregues datos externos.",
            "Conserva los diagnósticos tal como estén documentados; no propongas diagnósticos nuevos.",
            "Distingue entre ‘no consta’, ‘no está visible’, ‘es ambiguo’ y ‘se confirmó que no existe’.",
            "Señala cualquier afirmación que no pueda vincularse con la fuente; no rellenes vacíos por inferencia.",
            "No incluyas datos personales identificables ni conviertas la respuesta en una decisión clínica."
        ]
        format_instruction = (
            "Usa el formato solicitado en la tarea. Cuando extraigas información, incluye el dato, "
            "su evidencia textual y cualquier limitación o verificación pendiente."
        )

    # Prompt 1 teaches the full structure: role, objective, context, task, constraints and output.
    if step == 0:
        constraints = "\n".join(f"- {item}" for item in base_constraints)
        return (
            f"ROL:\n{role}\n\n"
            f"OBJETIVO:\n{objective}\n\n"
            f"CONTEXTO DEL CASO Y FUENTE:\n{case['context']}\n\n"
            f"TAREA E INSTRUCCIONES ESPECÍFICAS:\n{prompt}\n\n"
            f"REGLAS Y LÍMITES:\n{constraints}\n\n"
            f"FORMATO DE SALIDA:\n{format_instruction}\n\n"
            "Antes de responder, comprueba que la salida se base en la información disponible "
            "y deja explícito cualquier dato faltante."
        )

    # Prompts 2 and 3 are intentionally shorter: students continue in the same chat.
    if step == 1:
        if part == "Aplicación práctica y prompt engineering":
            safety_line = "No agregues datos ni supongas que la información faltante es verdadera."
            heading = "PROMPT 2 · REVISAR Y MEJORAR"
        else:
            safety_line = (
                "Contrasta la respuesta con la fuente del Prompt 1. Señala posibles alucinaciones, "
                "afirmaciones sin respaldo y datos que requieran confirmación; no completes vacíos por inferencia."
            )
            heading = "PROMPT 2 · VERIFICAR Y CORREGIR"
        return (
            f"{heading}\n\n"
            "Revisa la respuesta anterior y realiza esta tarea:\n"
            f"{prompt}\n\n"
            f"{safety_line}"
        )

    if part == "Aplicación práctica y prompt engineering":
        safety_line = (
            "Entrega un resultado claro y utilizable, mantén visibles los datos pendientes "
            "y deja el contenido listo para revisión profesional."
        )
        heading = "PROMPT 3 · PREPARAR EL RESULTADO FINAL"
    else:
        safety_line = (
            "Conserva la evidencia y las limitaciones relevantes. Si algo no puede verificarse, "
            "indícalo expresamente en lugar de inventar una respuesta."
        )
        heading = "PROMPT 3 · ENTREGAR EL RESULTADO VERIFICADO"
    return (
        f"{heading}\n\n"
        "A partir de la respuesta revisada en el paso anterior, realiza esta tarea:\n"
        f"{prompt}\n\n"
        f"{safety_line}"
    )


for i, prompt in enumerate(case["prompts"]):
    color = step_colors[i]
    background = prompt_backgrounds[color]
    structured = build_structured_prompt(prompt, case, seg, part, i)
    safe_prompt = html.escape(readable_prompt(structured))

    st.markdown(
        f"""
        <div class="sectionbox {color}">
            <h4>{step_names[i]}</h4>
            <p>{([
                'Define el rol, el objetivo, el contexto, las instrucciones y los límites de la tarea.',
                'Usa la respuesta anterior como borrador; revisa su calidad y corrige lo necesario.',
                'Prepara el resultado solicitado para revisión humana antes de usarlo.'
            ] if part == "Aplicación práctica y prompt engineering" else [
                'Identifica la información explícita sin completar vacíos ni inferir datos.',
                'Contrasta cada afirmación con la fuente y señala errores, ambigüedades o datos no respaldados.',
                'Entrega la información verificada e indica las limitaciones y comprobaciones pendientes.'
            ])[i]}</p>
            <div style="
                background-color: {background};
                border: 1px solid rgba(120, 150, 180, 0.25);
                border-radius: 12px;
                padding: 12px 15px;
                margin-top: 12px;
                color: #263746;
                white-space: pre-wrap;
                overflow-wrap: anywhere;
                font-family: monospace;
                font-size: 0.9rem;
            ">{safe_prompt}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption("Selecciona el texto del bloque y cópialo para pegarlo en el asistente elegido.")

    if i < 2:
        st.markdown(
            "<p class='small'>Después de probarlo, conserva la respuesta. "
            "El siguiente prompt te ayudará a revisarla o utilizarla.</p>",
            unsafe_allow_html=True
        )

with st.expander("Lista rápida de verificación antes de usar un resultado"):
    st.markdown("- ¿Cada dato coincide con el caso o la fuente original?\n- ¿La IA distinguió hechos, hipótesis y datos faltantes?\n- ¿Inventó diagnósticos, citas, normas, cifras o recomendaciones?\n- ¿Se puede rastrear cada afirmación importante hasta su fuente?\n- ¿Una persona responsable revisó y aprobó el resultado antes de utilizarlo?")

st.markdown("<div class='sectionbox purple'><h4>Uso responsable</h4><p>Estos casos son ficticios y educativos. La IA puede equivocarse o inventar información. <br>No uses datos identificables, no copies automáticamente la salida a una historia clínica y no sustituyas el juicio profesional,<br> la normativa aplicable ni los protocolos institucionales.</p></div>", unsafe_allow_html=True)
