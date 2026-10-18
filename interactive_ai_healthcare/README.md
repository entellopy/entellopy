# Laboratorio de IA aplicada a salud

Aplicación educativa en Streamlit para explorar cómo la IA puede apoyar la mejora de procesos en cinco áreas del sector salud.

## Qué incluye
- Cinco áreas profesionales con casos ficticios y objetivos concretos.
- Un flujo de tres prompts conectados: entender el problema, identificar oportunidades y diseñar una mejora medible.
- Parte 1 para diagnóstico y análisis; Parte 2 para propuesta y validación.
- Ejecución opcional dentro de la app mediante la API de OpenAI, o modo guiado para copiar prompts en ChatGPT/Copilot y pegar los resultados.
- Límites de salida de 120 palabras o cinco viñetas y controles para no inventar datos.

## Ejecutar localmente
Requiere Python 3.10 o superior.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Conectar el modelo de IA en Streamlit Community Cloud
1. Sube los archivos del proyecto a GitHub.
2. Entra a https://share.streamlit.io/ y despliega `app.py`.
3. En la configuración de la app, abre **Settings → Secrets** y añade:

```toml
OPENAI_API_KEY = "tu_clave_de_api"
OPENAI_MODEL = "gpt-4.1-mini"
```

4. Guarda y reinicia/reinicia el despliegue. No guardes la clave en `app.py` ni en GitHub.

La API de OpenAI se factura por uso y es independiente de una suscripción de ChatGPT. No añadas la clave hasta decidir activar esa conexión. Sin clave, la app sigue funcionando en modo guiado con prompts copiables y resultados que se pueden pegar para continuar el flujo.

## Seguridad y límites
- Usa solo casos ficticios; no pegues datos personales ni historias clínicas reales.
- Las respuestas se limitan en el prompt y en el máximo de tokens, pero el límite de palabras no puede garantizarse perfectamente.
- La app no sustituye protocolos, fuentes institucionales ni revisión profesional.
