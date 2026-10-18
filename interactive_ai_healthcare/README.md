# Laboratorio de IA aplicada a salud

Aplicación educativa de Streamlit con cinco áreas profesionales y dos partes alineadas con las sesiones de clase.

- **Parte 1 / Sesión 1:** interactuar con asistentes y mejorar prompts.
- **Parte 2 / Sesión 2:** aplicar IA para analizar, resumir, transformar y verificar información.
- Dos ejercicios por parte y por área, con tres prompts visibles por ejercicio.
- La Parte 2 queda bloqueada hasta completar los dos ejercicios de la Parte 1 en la misma área.
- Sin API, claves, pagos ni llamadas externas. Los estudiantes copian los prompts a la herramienta de IA utilizada en clase.

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

En Streamlit Community Cloud, despliega desde `app.py`; no se requieren Secrets. Usa solo casos ficticios y nunca introduzcas datos reales de pacientes.
