# IPS Horizonte — Centro de inteligencia gerencial

Aplicación Streamlit para un laboratorio académico de inteligencia gerencial en salud.

## Archivos
- `app.py`: aplicación.
- `IPS_Horizonte_M4T4_datos_v2.xlsx`: fuentes preparadas y banderas de calidad.
- `requirements.txt`: dependencias.

## Ejecutar localmente
```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\\Scripts\\activate     # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Decisiones metodológicas
- Las cinco fuentes se analizan a su granularidad original; no se hace un join general.
- Operaciones: 8 semanas y 3 especialidades; sin fechas calendario.
- Talento humano: trabajador × mes × servicio × turno.
- Inventario: mes × insumo; no tiene dimensión de servicio.
- Equipos: inventario de corte; fallas en ventana móvil de 90 días.
- Auditoría: registro asistencial individual.
- No hay datos contables: no se calculan costos ni costo por atención.
- Los valores atípicos se conservan y se marcan; no se imputan ni corrigen automáticamente.
- El laboratorio de escenarios es una simulación proporcional, no un pronóstico.
- El registro de decisiones se conserva solo en la sesión; se puede descargar como CSV.

## Nota para la clase
Los datos son sintéticos y se usan con fines académicos. La aplicación no es un sistema clínico ni una herramienta de decisión asistencial real.


## Umbrales de alerta
Las banderas automáticas de ausencias (>10 días), lead time (>60 días) e indisponibilidad (>100 horas) son umbrales exploratorios didácticos, no estándares regulatorios ni clínicos. Deben validarse antes de cualquier uso operativo.
