# Centro de operaciones — IPS Horizonte

Prototipo didáctico en Streamlit para el walkthrough del Módulo 4, Tema 3.

## Ejecutar localmente

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Qué incluye
- Bandeja inicial con tres alertas simultáneas.
- Inventario: cobertura, riesgo, cantidad sugerida, aprobación y recepción simulada.
- Talento humano: alternativas de cobertura y validación de restricciones.
- Equipos: priorización y estados de orden de mantenimiento.
- Decisiones y bitácora: registro de acciones y comparación de estado.

## Importante
Todos los datos son sintéticos. La interpretación de mensajes está simulada localmente mediante reglas de palabras clave; no se conecta a un modelo generativo ni a sistemas reales. Las reglas de turnos, compras y mantenimiento son ilustrativas y deben ser definidas por cada institución antes de cualquier uso real.
