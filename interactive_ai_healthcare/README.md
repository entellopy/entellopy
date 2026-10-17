# Laboratorio de IA aplicada a salud

Aplicación educativa en Streamlit para practicar prompts y revisión crítica con casos ficticios.

## Incluye
- 5 segmentos profesionales.
- 2 ejercicios por segmento en cada sesión (20 ejercicios en total).
- Prompts copiables, listas de verificación y productos esperados.
- Enlaces a ChatGPT y Microsoft Copilot.
- Sin API, claves ni base de datos.

## Ejecutar localmente
Requiere Python 3.10 o superior.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Publicar en GitHub y Streamlit Community Cloud
1. Crea un repositorio GitHub llamado `laboratorio-ia-salud`.
2. Sube `app.py`, `requirements.txt`, `README.md`, `.gitignore` y `.streamlit/config.toml`.
3. Abre https://share.streamlit.io/ e inicia sesión con GitHub.
4. Selecciona **Create app**, el repositorio, la rama `main` y `app.py`.
5. Pulsa **Deploy**. No hacen falta Secrets para esta versión.
6. Los cambios subidos a GitHub se reflejan en la app conectada tras el redeploy.

## Seguridad y alcance
Usa solo datos ficticios. No pegues información identificable ni historias clínicas reales. La app no llama a un modelo de IA y no valida la exactitud clínica de las respuestas. El área de texto es para trabajar durante la sesión; no hay base de datos.
