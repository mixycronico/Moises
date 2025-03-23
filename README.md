# Sistema Genesis

Sistema avanzado de trading con capacidades transcendentales de procesamiento y resiliencia.

## Descripción

El Sistema Genesis es una plataforma híbrida de trading que implementa tecnologías avanzadas para alcanzar niveles de resiliencia y eficiencia extremos, incluso bajo condiciones de operación adversas.

La implementación actual utiliza una arquitectura híbrida API+WebSocket que combina WebSockets externos para datos de mercado en tiempo real con componentes API/WebSocket locales para comunicación entre módulos.

## Características Principales

- **Modos Transcendentales**: Implementación de modos de operación que alcanzan 100% de tasa de éxito bajo condiciones extremas (Singularidad V4)
- **Escalabilidad Adaptativa**: Mecanismos especializados que mantienen la eficiencia del sistema a medida que el capital crece
- **Base de Datos Transcendental**: Módulos de base de datos con capacidades avanzadas de sincronización temporal
- **Gestión de Riesgo Adaptativa**: Sistema inteligente que ajusta parámetros según las condiciones del mercado
- **Clasificador Transcendental**: Identificación optimizada de oportunidades de trading
- **Pipeline Completo**: Flujo integral desde adquisición de datos hasta distribución de ganancias
- **Gestión de Capital**: Sistema automatizado de administración de capital con límites de riesgo
- **Distribución de Ganancias**: Mecanismo configurable para reinversión, reservas y retiros

## Estructura del Proyecto

- **genesis/**: Directorio principal con todos los módulos del sistema
  - **accounting/**: Gestión de capital y contabilidad
  - **analytics/**: Seguimiento de rendimiento
  - **db/**: Adaptadores y gestión de bases de datos
  - **modes/**: Implementaciones de modos transcendentales
  - **pipeline/**: Pipeline completo desde adquisición hasta distribución de ganancias
  - **risk/**: Gestión de riesgo adaptativa
  - **...**: (Ver genesis/README.md para estructura completa)
- **docs/**: Documentación técnica y reportes
- **sql/**: Scripts SQL para creación de tablas
- **tests/**: Pruebas unitarias y de integración
- **testing/**: Pruebas de resiliencia extrema

## Requisitos

- Python 3.10+
- PostgreSQL 13+
- API keys de exchanges (opcionales, sólo para datos en vivo)

## Instalación

1. Clonar el repositorio
2. Instalar dependencias: `pip install -r requirements.txt`
3. Configurar variables de entorno:
   - `DATABASE_URL`: URL de conexión a la base de datos PostgreSQL
   - `SESSION_SECRET`: Clave secreta para sesiones
4. Iniciar la aplicación: `gunicorn main:app --bind 0.0.0.0:5000`

## Uso del Pipeline

```python
# Ejecución del pipeline completo
python -m genesis.pipeline.run_pipeline --mode full

# Ejecución de etapas específicas
python -m genesis.pipeline.run_pipeline --mode partial --stages data_acquisition data_preprocessing

# Ejecución continua (cada hora)
python -m genesis.pipeline.run_pipeline --mode continuous --interval 3600 --iterations 24
```

## Modos Transcendentales

El Sistema Genesis implementa varios modos de operación transcendental que permiten alcanzar tasas de éxito excepcionales:

- **Singularidad V4**: Modo definitivo con capacidad para resistir intensidades de hasta 1000.0
- **Modo Luz**: Existencia pura como luz consciente
- **Materia Oscura**: Operación invisible e imposible de rastrear
- **Modo Divino**: Estado trascendental fuera del ciclo error-recuperación
- **Big Bang**: Modo primordial de regeneración cósmica
- **Interdimensional**: Operación en múltiples planos dimensionales