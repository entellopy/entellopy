import streamlit as st
st.set_page_config(page_title="Laboratorio de IA aplicada a salud", page_icon="🩺", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container{max-width:1180px;padding-top:1.2rem;padding-bottom:3rem}.hero{background:linear-gradient(120deg,#E5F4FF,#E1F7F0 58%,#FFF3D5);border:1px solid #C7E3E7;border-radius:22px;padding:24px 28px;margin-bottom:16px}.hero h1{color:#17476B;margin:0 0 6px}.hero p{color:#416173;margin:0}.notice{background:#FFF6D9;border-left:6px solid #E6B33F;border-radius:10px;padding:13px 16px;color:#624C19;margin:14px 0 22px}.part-card{box-sizing:border-box;height:210px;border-radius:18px;padding:18px 20px;border:2px solid;display:flex;flex-direction:column}.part-card h3{margin:8px 0;color:#17476b;font-size:1.2rem}.part-card p{margin:0;color:#3f5c6d;line-height:1.4}.part-blue{background:#e8f3ff;border-color:#b7d8fa}.part-green{background:#e2f7ee;border-color:#b8e7d1}.sel-blue{outline:3px solid #2f80c9}.sel-green{outline:3px solid #138a8a}.parttag{display:inline-block;border-radius:99px;padding:5px 10px;font-size:.76rem;font-weight:850;align-self:flex-start}.blue-tag{background:#cce5ff;color:#155d99}.green-tag{background:#c9efd9;color:#176d4b}.selection{font-weight:800;font-size:.82rem;margin-top:auto;padding-top:12px;color:#176d4b}.stage{border-radius:14px;padding:12px 16px;margin:14px 0 8px;border:1px solid #d7e5ec}.s1{background:#f2f8ff;border-left:6px solid #2f80c9}.s2{background:#effaf5;border-left:6px solid #138a8a}.s3{background:#fff9e9;border-left:6px solid #e6b33f}.stage h3{margin:0;color:#17476b;font-size:1.05rem}div.stButton>button{border-radius:11px;font-weight:750;min-height:2.6rem}@media(max-width:700px){.part-card{height:240px;padding:14px}.hero{padding:18px}}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🩺 Laboratorio de IA aplicada a salud</h1><p>Parte 1: formular mejores prompts · Parte 2: aplicar, transformar y verificar información.</p></div>',unsafe_allow_html=True)
st.markdown('<div class="notice">⚠️ Usa solo los casos ficticios. Los prompts se copian a la herramienta de IA que uses en clase. Esta aplicación no llama a un modelo, no requiere API ni genera respuestas en vivo.</div>',unsafe_allow_html=True)

CASES={
"Atención clínica y rehabilitación":{
"case":"Consulta ficticia: persona de 52 años con dolor de rodilla desde hace 3 días y dolor al caminar. No se aportan examen físico, antecedentes completos ni plan.","data":"52 años; dolor de rodilla de 3 días; dolor al caminar. Examen físico, antecedentes completos y plan no informados.",
"p1":[("Preparar una consulta sin inventar información","Distinguir hechos, datos faltantes y preguntas para el profesional."),("Estructurar una nota clínica","Diseñar una instrucción para ordenar la información y marcar campos pendientes.")],"p2":[("Crear una lista de verificación","Transformar el resumen en apoyo operativo para preparar la consulta."),("Verificar la salida clínica","Revisar omisiones, afirmaciones sin evidencia y necesidad de validación profesional.")]},
"Aseguramiento y gestión del riesgo en salud (EPS)":{
"case":"Una EPS ficticia revisa el seguimiento trimestral de 200 personas con diabetes tipo 2; se desconocen las causas de los registros faltantes.","data":"Población 200; control registrado 132; sin control registrado 68; HbA1c registrada 118; presión arterial registrada 145; citas pendientes reportadas 54. Los grupos pueden superponerse.",
"p1":[("Aclarar una solicitud de análisis poblacional","Separar cifras, denominadores, incertidumbres y datos faltantes."),("Priorizar brechas investigables","Pedir hipótesis por validar, sin asumir causas ni decidir sobre atención individual.")],"p2":[("Transformar cifras en indicadores","Calcular porcentajes con fórmulas y denominadores claros."),("Crear una matriz de seguimiento","Organizar campos para validar fechas, personas únicas, contacto y citas pendientes.")]},
"Operaciones y administración de IPS":{
"case":"Una IPS ficticia recibe quejas por demoras en consulta externa. Solo dispone de datos de un día; no sabe si varía por especialidad, hora o tipo de cita.","data":"Un día: 30 citas; espera media 42 minutos; mediana 35 minutos; 8 citas superaron 60 minutos. No hay otros días ni causas registradas.",
"p1":[("Definir el problema de oportunidad","Separar datos observados de preguntas aún abiertas."),("Explorar posibles causas sin asumirlas","Solicitar hipótesis comprobables y datos necesarios para investigarlas.")],"p2":[("Convertir el resumen en tabla de análisis","Organizar medidas, denominadores, segmentos y variables por recolectar."),("Diseñar un piloto operativo","Proponer una prueba acotada con indicador de resultado y de balance.")]},
"Auditoría, calidad y regulación":{
"case":"Un equipo auditor ficticio busca reducir el tiempo de revisión documental. Criterios internos ficticios: debe aparecer fecha y firma del responsable.","data":"Registro A: fecha y firma visibles. B: fecha visible, firma ilegible. C: fecha no visible y copia no legible. No se aportó normativa aplicable.",
"p1":[("Revisar evidencia contra criterios entregados","Evitar inventar normas o emitir dictámenes automáticos."),("Clasificar hallazgos consistentemente","Diferenciar evidencia visible, ausente en la copia y no evaluable.")],"p2":[("Transformar notas en matriz de auditoría","Estructurar criterio, evidencia, estado, observación y revisión requerida."),("Verificar trazabilidad","Comprobar que cada clasificación tenga evidencia y que dudas escalen a una persona.")]},
"Dirección, investigación y salud pública":{
"case":"Un equipo directivo analiza una encuesta ficticia sobre acceso a citas y necesita distinguir hallazgos, limitaciones y acciones posibles.","data":"Encuesta por conveniencia: 120 participantes; 45 reportaron dificultades para conseguir cita. La muestra puede no ser representativa y no demuestra causalidad.",
"p1":[("Sintetizar críticamente un hallazgo","Distinguir resultado observado, límites de la muestra y conclusiones no sustentadas."),("Generar preguntas de investigación","Proponer preguntas y datos adicionales sin convertir hipótesis en hechos.")],"p2":[("Convertir resultados en nota ejecutiva","Resumir cifras, limitaciones y próximos pasos verificables."),("Comparar opciones de mejora","Vincular cada opción con evidencia, información faltante y forma de validación.")]}}
PARTS={"Parte 1":{"tag":"PARTE 1 · EXPLORAR","subtitle":"Sesión 1 · Interactuar con asistentes y mejorar prompts","desc":"Comprender el contexto, separar hechos de supuestos y mejorar instrucciones.","cls":"part-blue","sel":"sel-blue"},"Parte 2":{"tag":"PARTE 2 · APLICAR","subtitle":"Sesión 2 · Aplicar, transformar información y verificar","desc":"Analizar, resumir y estructurar información; comprobar resultados antes de usarlos.","cls":"part-green","sel":"sel-green"}}

def key_for(a,p,i): return f"{a}::{p}::ej{i+1}"
def prompts_for(area, case, part, idx):
    title,goal=case['p1' if part=='Parte 1' else 'p2'][idx]
    if part=='Parte 1':
        if idx==0:
            prompts=[f"Actúa como asistente de apoyo a procesos en salud. Área: {area}. Caso: {case['case']} Datos: {case['data']} Separa en una tabla: hechos explícitos, datos faltantes y preguntas por resolver. No inventes información ni hagas diagnósticos o recomendaciones clínicas individuales.",
            f"Revisa tu respuesta anterior para el caso de {area}. Etiqueta cada afirmación como hecho, hipótesis o dato faltante. Identifica cualquier frase que no esté respaldada por el caso y corrígela. No añadas datos externos.",
            f"Mejora el prompt inicial para el área {area}. Debe incluir rol, contexto, tarea concreta, formato de salida y límites. Caso: {case['case']} Devuelve el prompt mejorado listo para copiar y una frase que explique por qué es más claro."]
        else:
            prompts=[f"Para el área {area}, analiza este caso: {case['case']} Datos: {case['data']} Formula tres preguntas que ayuden a entender el proceso antes de recomendar cambios. Separa lo conocido de lo que falta averiguar.",
            f"Usa el caso de {area} y las preguntas anteriores. Propón hasta tres oportunidades que valga la pena investigar; para cada una indica evidencia disponible, hipótesis por validar y dato adicional necesario. No afirmes causalidad ni inventes metas.",
            f"Escribe un prompt mejorado para investigar oportunidades de mejora en {area}. Incluye este contexto: {case['data']}; solicita una tabla con oportunidad, evidencia, hipótesis y dato faltante; máximo tres filas; prohíbe inventar datos o presentar hipótesis como hechos."]
    else:
        if idx==0:
            prompts=[f"Usa solo estos datos ficticios del área {area}: {case['data']} Organiza la información en tabla: dato, valor, denominador/unidad, interpretación prudente y limitación. Calcula porcentajes solo si el denominador es claro y muestra la fórmula. No inventes datos.",
            f"Verifica la tabla anterior para {area}. Recalcula los porcentajes, comprueba denominadores y señala interpretaciones que excedan la evidencia. Si algo no se puede comprobar, escribe 'no verificable con los datos disponibles'.",
            f"Convierte la tabla verificada del caso de {area} en una síntesis breve con tres apartados: hallazgos, limitaciones y datos que deben recogerse después. Usa solo: {case['data']} Distingue hechos e hipótesis."]
        else:
            prompts=[f"Para este caso de {area}: {case['case']} Datos: {case['data']} Crea una matriz con oportunidad, evidencia, hipótesis por validar, dato necesario, acción piloto, indicador y responsable funcional. Marca lo que aún no se puede completar; no inventes metas.",
            f"Audita la matriz anterior para {area}. Comprueba que cada oportunidad tenga evidencia, que las hipótesis estén identificadas y que cada indicador tenga definición o fórmula. Señala riesgos de privacidad, sesgo y puntos que requieren revisión humana.",
            f"A partir de la matriz revisada del caso de {area}, redacta un plan de prueba pequeño y reversible con pasos, responsable funcional, duración por definir, indicador de resultado, indicador de balance y validación humana. No inventes línea base, metas ni causas. Datos: {case['data']}"]
    return prompts,title,goal

if 'selected_part' not in st.session_state: st.session_state.selected_part='Parte 1'
if 'completed' not in st.session_state: st.session_state.completed=set()
area=st.selectbox('Selecciona tu área profesional',list(CASES.keys()),key='area')
case=CASES[area]
part1_done=all(key_for(area,'Parte 1',i) in st.session_state.completed for i in range(2))
if st.session_state.selected_part=='Parte 2' and not part1_done: st.session_state.selected_part='Parte 1'
st.markdown('### Elige la parte de la práctica')
c1,c2=st.columns(2,gap='large')
with c1:
    sel=st.session_state.selected_part=='Parte 1'; cl='part-card '+PARTS['Parte 1']['cls']+(' '+PARTS['Parte 1']['sel'] if sel else '')
    badge='✓ SELECCIONADA' if sel else 'PARA EMPEZAR'
    st.markdown(f'<div class="{cl}"><span class="parttag blue-tag">PARTE 1 · EXPLORAR</span><h3>De una necesidad a un buen prompt</h3><p>Sesión 1: comprender el contexto, separar hechos de supuestos y mejorar instrucciones para la IA.</p><div class="selection">{badge}</div></div>',unsafe_allow_html=True)
    if st.button('Abrir Parte 1',key='go1',use_container_width=True): st.session_state.selected_part='Parte 1'; st.rerun()
with c2:
    sel=st.session_state.selected_part=='Parte 2'; cl='part-card '+PARTS['Parte 2']['cls']+(' '+PARTS['Parte 2']['sel'] if sel else '')
    badge='✓ SELECCIONADA' if sel else ('🔒 Completa los dos ejercicios de Parte 1' if not part1_done else 'LISTA PARA CONTINUAR')
    st.markdown(f'<div class="{cl}"><span class="parttag green-tag">PARTE 2 · APLICAR</span><h3>De la información a una mejora verificable</h3><p>Sesión 2: analizar, resumir, transformar información y verificar resultados antes de utilizarlos.</p><div class="selection">{badge}</div></div>',unsafe_allow_html=True)
    if st.button('Abrir Parte 2' if part1_done else 'Parte 2 bloqueada',key='go2',use_container_width=True,disabled=not part1_done): st.session_state.selected_part='Parte 2'; st.rerun()
part=st.session_state.selected_part
st.markdown(f"### {PARTS[part]['subtitle']}")
st.write(PARTS[part]['desc'])
st.markdown(f"**Caso del área:** {case['case']}")
with st.expander('Ver datos ficticios'): st.write(case['data'])
st.markdown('### Ejercicios de esta parte')
st.caption('Los prompts son visibles y específicos para tu área. Cópialos en la herramienta de IA que uses en clase, revisa sus respuestas y marca cada paso. La aplicación no envía datos a servicios externos.')
for idx in range(2):
    k=key_for(area,part,idx); prompts,title,goal=prompts_for(area,case,part,idx); done=k in st.session_state.completed
    with st.expander(f"{'✅' if done else '🧩'} Ejercicio {idx+1}: {title}",expanded=not done):
        st.write(goal)
        steps=st.session_state.get('steps::'+k,[False,False,False])
        names=['Contextualizar','Revisar y profundizar','Mejorar o aplicar']
        for j,prompt in enumerate(prompts):
            st.markdown(f'<div class="stage s{j+1}"><h3>Paso {j+1} · {names[j]}</h3></div>',unsafe_allow_html=True)
            st.code(prompt,language='text')
            steps[j]=st.checkbox(f'Ya probé y revisé el prompt {j+1}',value=steps[j],key=f'check::{k}::{j}')
        st.session_state['steps::'+k]=steps
        if st.button('Marcar ejercicio como completado' if not done else 'Ejercicio completado ✓',key='finish::'+k,disabled=not all(steps) or done,use_container_width=True,type='primary' if all(steps) and not done else 'secondary'):
            st.session_state.completed.add(k); st.rerun()
        if done: st.success('Ejercicio completado. Puedes volver a consultar sus prompts.')
count=sum(key_for(area,part,i) in st.session_state.completed for i in range(2))
st.progress(count/2,text=f'Progreso de {part}: {count} de 2 ejercicios completados')
if part=='Parte 1':
    if part1_done: st.success('¡Parte 1 completada para esta área! Ya puedes abrir Parte 2. Ningún botón de los pasos cambia de parte automáticamente.')
    else: st.info('Completa los dos ejercicios de Parte 1 para desbloquear Parte 2 en esta área. Los botones de los pasos solo registran progreso.')
else: st.success('Estás en Parte 2. Puedes volver a Parte 1 cuando quieras; el progreso se conserva por área y ejercicio.')
if all(key_for(area,'Parte 2',i) in st.session_state.completed for i in range(2)) and part1_done:
    st.markdown('<div class="notice">🎉 <b>Ruta completada:</b> practicaste cómo mejorar prompts y cómo aplicar, transformar y verificar información. Valida datos y decisiones con el equipo responsable.</div>',unsafe_allow_html=True)
    lines=[f'Laboratorio de IA en salud — {area}',f'Caso: {case["case"]}',f'Datos: {case["data"]}']
    for pname in ['Parte 1','Parte 2']:
        lines.append('\n'+PARTS[pname]['subtitle'])
        for i in range(2):
            ps,title,_=prompts_for(area,case,pname,i); lines.append('\nEjercicio '+str(i+1)+': '+title)
            for j,p in enumerate(ps): lines.append(f'\nPaso {j+1}:\n{p}')
    st.download_button('Descargar todos los prompts de esta área (.txt)',data='\n'.join(lines),file_name='prompts_ia_salud.txt',mime='text/plain')
st.markdown('<div class="notice">💡 <b>Uso responsable:</b> los resultados de IA pueden contener errores. Verifica datos, criterios y conclusiones; la validación humana es indispensable.</div>',unsafe_allow_html=True)
