# Laboratorio de IA aplicada a salud

Aplicación educativa de Streamlit con rutas visuales y casos ficticios para practicar mejora de procesos en salud.

## Características
- Cinco áreas profesionales con casos y datos ficticios.
- Dos rutas seleccionables en tarjetas de colores; la opción activa muestra una marca de selección.
- Tres pasos encadenados por caso: comprender, identificar oportunidades y diseñar una mejora medible.
- Sin API, sin claves, sin pagos y sin cuentas de IA para ejecutar la aplicación.
- Resultados guiados predefinidos y descargables.

## Importante sobre el modo gratuito
Esta versión NO llama a un modelo de IA. Para que los estudiantes puedan generar respuestas libres de un modelo dentro de Streamlit se necesitaría alojar un modelo propio o utilizar un servicio de inferencia externo; no existe una API de LLM gratuita, ilimitada y garantizada para una aplicación pública. Para evitar costos, este MVP usa respuestas guiadas predefinidas que muestran cómo se encadenan los pasos.

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Desplegar en Streamlit Community Cloud
Sube `app.py`, `requirements.txt`, `README.md`, `.gitignore` y `.streamlit/config.toml` a GitHub. En https://share.streamlit.io/ selecciona el repositorio, la rama `main` y `app.py`. No se requieren Secrets.

## Privacidad
Usa únicamente los casos ficticios. No introduzcas historias clínicas reales ni datos personales.
