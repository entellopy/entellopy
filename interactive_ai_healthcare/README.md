# Laboratorio de IA aplicada a salud

Aplicación educativa de Streamlit para practicar cómo usar IA en la mejora de procesos en cinco áreas de salud.

## Funcionalidad
- Interfaz con tarjetas de color, selector destacado e ilustración médica clara.
- Cinco áreas profesionales con casos ficticios completos.
- Dos rutas visuales: Parte 1 — Explorar y Parte 2 — Aplicar.
- Tres prompts conectados: comprender el problema, analizar oportunidades y diseñar una mejora medible.
- La respuesta del paso anterior se incorpora al siguiente prompt automáticamente.
- Respuestas acotadas con instrucciones de máximo 120 palabras/5 viñetas y límite de tokens.
- La API se configura una sola vez por la persona propietaria en Streamlit Secrets; los estudiantes no necesitan acceder a la configuración.

## Ejecutar localmente
Python 3.10+:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Configure una variable de entorno `OPENAI_API_KEY` o cree `.streamlit/secrets.toml` localmente:

```toml
OPENAI_API_KEY = "tu_clave_api"
OPENAI_MODEL = "gpt-4.1-mini"
```

No suba `secrets.toml` ni claves a GitHub.

## Publicar en Streamlit Community Cloud
1. Sube `app.py`, `requirements.txt`, `README.md`, `.gitignore` y `.streamlit/config.toml` a GitHub.
2. Despliega desde https://share.streamlit.io/ seleccionando el repositorio, la rama `main` y `app.py`.
3. En la configuración de la app, la persona propietaria agrega `OPENAI_API_KEY` y opcionalmente `OPENAI_MODEL` en **Settings → Secrets**. Esto es una configuración del despliegue, no de cada estudiante.
4. Reinicia/redeploy la app. Los estudiantes podrán usar el modelo en la página sin ver la clave.

## Privacidad y límites
Usar solo casos ficticios; no introducir historias clínicas reales ni datos personales. Las respuestas se conservan en la sesión de Streamlit mientras dure la sesión y no se guardan en una base de datos. El proveedor de IA recibe el prompt y los datos ficticios enviados. El límite de palabras es una instrucción más un límite de tokens, no una garantía exacta. La IA puede equivocarse; valida todo resultado.
