# Sistema de Pipeline Trascendental Genesis

Este módulo implementa un pipeline completo para el Sistema Genesis con capacidades trascendentales de resiliencia extrema. El pipeline procesa datos desde la adquisición hasta la ejecución de órdenes y distribución de ganancias.

## Arquitectura

El pipeline está compuesto por las siguientes etapas:

1. **Adquisición de Datos** - Obtiene datos de mercado, noticias y sentimiento de múltiples fuentes.
2. **Procesamiento de Datos** - Limpia, transforma y enriquece los datos con indicadores técnicos.
3. **Análisis** - Genera señales de trading basadas en análisis técnico, sentimiento y modelos predictivos.
4. **Decisión** - Convierte señales en decisiones de trading considerando gestión de capital y riesgo.
5. **Ejecución** - Implementa las decisiones como órdenes de mercado y monitorea su estado.
6. **Seguimiento de Rendimiento** - Registra y analiza el rendimiento de las operaciones.
7. **Distribución de Ganancias** - Distribuye las ganancias según reglas predefinidas.
8. **Gestión de Capital** - Administra el capital de trading con enfoque de preservación.

## Características Trascendentales

- **Resiliencia Extrema**: Operación continua incluso bajo condiciones de fallo parcial.
- **Adaptabilidad Dimensional**: Ajuste automático a diferentes condiciones de mercado.
- **Capacidades de Transmutación**: Conversión de errores en resultados exitosos.
- **Mecanismos de Autoconocimiento**: Métricas avanzadas y autoajuste.
- **Sincronización Atemporal**: Coherencia entre diferentes marcos temporales.

## Uso del Pipeline

### Ejecución Completa

```python
from genesis.pipeline.run_pipeline import run_full_pipeline
import asyncio

# Ejecutar pipeline completo
result = asyncio.run(run_full_pipeline())
```

### Ejecución Parcial

```python
from genesis.pipeline.run_pipeline import run_partial_pipeline
import asyncio

# Ejecutar solo adquisición y procesamiento
stages = ["data_acquisition", "data_preprocessing"]
result = asyncio.run(run_partial_pipeline(stages))
```

### Ejecución Continua

```python
from genesis.pipeline.run_pipeline import run_continuous_pipeline
import asyncio

# Ejecutar cada hora indefinidamente
asyncio.run(run_continuous_pipeline(interval=3600, iterations=-1))
```

## Integración con Exchanges

El sistema soporta tanto operaciones simuladas como reales a través de exchanges:

- **Binance/Binance Testnet** - Implementado completamente
- **Simulador Interno** - Para desarrollo y pruebas sin riesgo

## Gestión de Capital

El sistema implementa estrategias avanzadas de gestión de capital:

- Asignación de capital por operación basada en riesgo
- Límites de riesgo global (portfolio risk)
- Trailing stops automáticos
- Distribución de ganancias configurable

## Distribución de Ganancias

El distribuidor automatizado divide las ganancias en tres categorías:

1. **Reinversión** - Capital que regresa al sistema de trading
2. **Reserva** - Capital almacenado para contingencias
3. **Retiro** - Capital disponible para el usuario

Las proporciones son ajustables basadas en umbrales de ganancia.

## Componentes Adicionales

- **Analizador de Sentimiento** - Incorpora análisis de noticias y sentimiento de mercado
- **Procesador de Señales Mixtas** - Combina señales técnicas y fundamentales
- **Gestor de Posiciones** - Monitoreo de posiciones con trailing stops
- **Motor de Ejecución** - Interfaz con exchanges para órdenes

## Ejecución desde Línea de Comandos

```bash
# Ejecutar pipeline completo
python -m genesis.pipeline.run_pipeline --mode full

# Ejecutar etapas específicas
python -m genesis.pipeline.run_pipeline --mode partial --stages data_acquisition data_preprocessing

# Ejecutar en modo continuo
python -m genesis.pipeline.run_pipeline --mode continuous --interval 3600 --iterations 24
```