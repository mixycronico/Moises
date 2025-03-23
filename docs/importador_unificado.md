# Adaptadores de Base de Datos Divinos y Operaciones Unificadas

## Introducción

Este documento describe dos componentes principales de la capa de datos del Sistema Genesis:

1. **Importador Unificado**: Módulo avanzado que proporciona una interfaz única para importar datos JSON en la base de datos del Sistema Genesis. Este módulo elimina la duplicación existente entre los scripts previos (`import_json_data.py`, `import_json_data_gen.py` e `import_json_data_sync.py`).

2. **Adaptadores Extendidos**: Conjunto de adaptadores especializados derivados del `DivineDatabaseAdapter` base, diseñados para casos de uso específicos como analítica y series temporales.

Ambos componentes aprovechan todas las capacidades resilientes del `DivineDatabaseAdapter` para soportar operaciones síncronas y asíncronas con máxima fiabilidad.

## Ventajas

- **Unificación de interfaces**: Una única API para todos los tipos de importaciones
- **Soporte híbrido**: Funciona tanto en contextos síncronos como asíncronos
- **Resiliencia extrema**: Aprovecha todas las capacidades del adaptador divino
- **Manejo transaccional**: Importaciones en transacciones atómicas para garantizar consistencia
- **Procesamiento por lotes**: Capacidad para importar múltiples archivos en paralelo
- **Caching inteligente**: Aprovecha el sistema de caché del adaptador divino
- **Código mantenible**: Elimina la duplicación y mejora la modularidad

## Arquitectura

El sistema está compuesto por tres componentes principales:

1. **Módulo base (`unified_import.py`)**: Implementa la clase `UnifiedImporter` y funciones de utilidad
2. **Script de línea de comandos (`import_data_unified.py`)**: Proporciona una interfaz por consola para importaciones
3. **Ejemplos de uso (`unified_import_examples.py`)**: Muestra cómo integrar el importador en otros scripts

## Uso

### Importación Síncrona (Simple)

```python
from genesis.db.unified_import import import_file_sync

success = import_file_sync("resultados_prueba.json")
if success:
    print("Importación exitosa")
else:
    print("Error en la importación")
```

### Importación Asíncrona (Simple)

```python
import asyncio
from genesis.db.unified_import import import_file_async

async def importar():
    success = await import_file_async("resultados_prueba.json")
    if success:
        print("Importación exitosa")
    else:
        print("Error en la importación")

asyncio.run(importar())
```

### Importación de Múltiples Archivos

```python
import asyncio
from genesis.db.unified_import import batch_import_files_async

async def importar_multiples():
    files = ["archivo1.json", "archivo2.json", "archivo3.json"]
    results = await batch_import_files_async(files)
    
    for file_path, success in results.items():
        print(f"{file_path}: {'Éxito' if success else 'Error'}")

asyncio.run(importar_multiples())
```

### Línea de Comandos

El script `import_data_unified.py` proporciona una interfaz por línea de comandos:

```bash
# Importación síncrona
python import_data_unified.py archivo1.json archivo2.json

# Importación asíncrona
python import_data_unified.py --async archivo1.json archivo2.json

# Importar todos los archivos JSON de un directorio
python import_data_unified.py --dir ./datos

# Importar en paralelo (modo asíncrono)
python import_data_unified.py --async --concurrent archivo1.json archivo2.json
```

## Estructura de Datos Soportada

El importador soporta los siguientes tipos de datos en los archivos JSON:

- **Resultados de intensidad**: Información general sobre la prueba
- **Ciclos de procesamiento**: Datos de ciclos individuales
- **Componentes**: Estadísticas de los componentes del sistema
- **Eventos**: Registros detallados de eventos durante la prueba
- **Métricas**: Valores medidos durante la ejecución

Ejemplo de estructura JSON soportada:

```json
{
  "intensity": 10.0,
  "mode": "SINGULARITY_V4",
  "average_success_rate": 0.99,
  "components_count": 10,
  "total_events": 500,
  "execution_time": 60.5,
  "timestamp": "2025-03-23T12:00:00Z",
  "system_version": "1.0.0",
  "metadata": {
    "additional_info": "Información adicional"
  },
  "ciclos": [
    {
      "cycle_number": 1,
      "success_rate": 0.98,
      "processing_time": 0.5
    }
  ],
  "componentes": [
    {
      "component_id": "comp_1",
      "component_type": "PROCESSOR",
      "success_rate": 0.99
    }
  ],
  "eventos": [
    {
      "cycle_number": 1,
      "event_type": "REQUEST",
      "source_component": "comp_1",
      "target_component": "comp_2"
    }
  ],
  "metricas": [
    {
      "cycle_number": 1,
      "metric_name": "latencia",
      "metric_value": 0.05,
      "component_id": "comp_1"
    }
  ]
}
```

## Integración con el Sistema Genesis

El Importador Unificado está diseñado para integrarse perfectamente con el Sistema Genesis:

1. **Transacción Atómica**: Cada importación se ejecuta en una única transacción atómica
2. **Compatibilidad con Modo Singularidad**: Aprovecha la resiliencia extrema del sistema
3. **Adaptabilidad**: Detecta automáticamente el contexto (síncrono o asíncrono)
4. **Escalabilidad**: Puede importar tanto archivos pequeños como grandes conjuntos de datos
5. **Monitoreo**: Integración con el sistema de logging para seguimiento detallado

## Comparación con Implementaciones Anteriores

| Característica | Scripts Antiguos | Importador Unificado |
|----------------|------------------|----------------------|
| Soporte síncrono | Parcial (scripts separados) | ✓ |
| Soporte asíncrono | Parcial (scripts separados) | ✓ |
| Transacciones atómicas | ✓ | ✓ |
| Manejo de errores | Básico | Avanzado |
| Procesamiento por lotes | ✗ | ✓ |
| Caching | ✗ | ✓ |
| Reintentos adaptativos | ✗ | ✓ |
| Código duplicado | Alto | Ninguno |
| Interfaz unificada | ✗ | ✓ |

## Próximos Pasos

- **Migración gradual**: Se recomienda migrar gradualmente a usar el nuevo importador
- **Pruebas comparativas**: Evaluar rendimiento contra implementaciones anteriores
- **Ampliar cobertura**: Extender para soportar más tipos de datos
- **Documentación adicional**: Incluir más ejemplos de uso avanzado

## Adaptadores Extendidos

Además del Importador Unificado, el Sistema Genesis proporciona adaptadores especializados derivados del `DivineDatabaseAdapter` base, diseñados para casos de uso específicos:

### AnalyticsDBAdapter

Adaptador especializado para operaciones analíticas con características optimizadas para consultas complejas:

- **Caché extendido**: Mayor tamaño y tiempo de vida para análisis
- **Patrones de consulta**: Registro de consultas analíticas frecuentes
- **Precarga**: Capacidad para precargar resultados de consultas comunes

Ejemplo de uso:

```python
from genesis.db.extended_divine_adapter import get_analytics_db_adapter

# Obtener instancia
analytics_db = get_analytics_db_adapter()

# Registrar patrón de consulta
analytics_db.register_query_pattern(
    "rendimiento_por_componente",
    """
    SELECT component_id, AVG(success_rate) as avg_success_rate
    FROM gen_components
    WHERE results_id IN (
        SELECT id FROM gen_intensity_results 
        WHERE intensity >= %(min_intensity)s
    )
    GROUP BY component_id
    ORDER BY avg_success_rate DESC
    """
)

# Ejecutar consulta analítica
resultados = analytics_db.fetch_analytics_sync(
    "rendimiento_por_componente", 
    {"min_intensity": 5.0}
)
```

### TimeSeriesDBAdapter

Adaptador especializado para datos de series temporales como precios, métricas y eventos secuenciales:

- **Registro de tablas temporales**: Gestión específica para tablas de series temporales
- **API especializada**: Métodos optimizados para inserción y consulta de datos temporales
- **Particionamiento**: Soporte para tablas particionadas por tiempo

Ejemplo de uso:

```python
from genesis.db.extended_divine_adapter import get_timeseries_db_adapter
import datetime

# Obtener instancia
timeseries_db = get_timeseries_db_adapter()

# Registrar tabla de series temporales
timeseries_db.register_timeseries_table("gen_crypto_prices")

# Definir rango temporal (últimas 24 horas)
end_time = datetime.datetime.now()
start_time = end_time - datetime.timedelta(hours=24)

# Consultar datos de series temporales
async def get_crypto_prices():
    prices = await timeseries_db.get_timeseries_async(
        "gen_crypto_prices",
        start_time.isoformat(),
        end_time.isoformat(),
        fields=["timestamp", "symbol", "price", "volume"],
        limit=100
    )
    return prices
```

## Conclusión

El Importador Unificado y los Adaptadores Extendidos representan una evolución significativa en el manejo de datos del Sistema Genesis, eliminando duplicación, mejorando la resiliencia y proporcionando interfaces especializadas para diferentes casos de uso. Se recomienda su adopción para todas las operaciones nuevas de acceso a datos en el sistema.