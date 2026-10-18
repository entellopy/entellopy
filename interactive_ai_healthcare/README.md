# Laboratorio de casos prácticos de IA en salud

Aplicación Streamlit educativa con dos partes temáticas, ambas basadas en casos prácticos personalizados para cinco segmentos profesionales.

## Partes
- **Parte 1:** uso práctico de asistentes conversacionales, prompt engineering, copilotos para documentación clínica y resumen de historias clínicas extensas.
- **Parte 2:** extracción de información y diagnósticos ya documentados, traducción a lenguaje sencillo, verificación, alucinaciones y uso responsable.

Cada caso incluye un flujo de hasta tres prompts: comprender/estructurar, revisar/profundizar y convertir el resultado en un producto útil. Los prompts cambian según el segmento y el caso.

La aplicación no llama a modelos ni requiere API key. Los usuarios copian los prompts en ChatGPT, Gemini, Copilot, Perplexity u otra herramienta aprobada por su institución.

## Ejecutar
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Privacidad
Todos los casos son ficticios. No ingresar historias clínicas reales ni datos identificables o confidenciales. Los resultados de la IA deben verificarse por una persona responsable antes de cualquier uso.
