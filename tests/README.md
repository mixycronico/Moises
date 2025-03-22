# Sistema de Pruebas para Genesis

Este directorio contiene las pruebas para el sistema de trading Genesis. Se ha optimizado para garantizar mayor resiliencia, detección de problemas y eficiencia en la ejecución.

## Estructura del Sistema de Pruebas

```
tests/
├── utils/                  # Utilidades compartidas para pruebas
│   ├── __init__.py
│   └── timeout_helpers.py  # Funciones para manejo de timeouts
├── unit/                   # Pruebas unitarias
│   ├── core/               # Pruebas para el motor de eventos
│   │   ├── conftest.py     # Fixtures compartidos
│   │   ├── test_core_extreme_scenarios.py
│   │   ├── test_core_extreme_scenarios_optimized.py
│   │   ├── test_core_load_concurrency.py
│   │   └── test_core_load_concurrency_optimized.py
│   └── ... (otros módulos)
├── integration/            # Pruebas de integración
└── conftest.py             # Configuración global de pytest
```

## Mejoras Implementadas

Se han implementado las siguientes mejoras en el sistema de pruebas:

1. **Manejo de Timeouts**: Todas las operaciones asíncronas ahora tienen límites de tiempo para evitar bloqueos indefinidos.
2. **Medición de Rendimiento**: Se agregó medición de tiempos para identificar cuellos de botella.
3. **Limpieza de Recursos**: Se implementó limpieza robusta después de cada prueba para evitar interferencias.
4. **Manejo Defensivo**: Se agregó manejo defensivo de respuestas nulas o inesperadas.

## Herramientas Proporcionadas

### Utilidades de Timeout (`tests/utils/timeout_helpers.py`)

- `emit_with_timeout()`: Permite emitir eventos con un límite de tiempo máximo.
- `check_component_status()`: Verifica el estado de un componente con timeout.
- `run_test_with_timing()`: Ejecuta y mide el tiempo de una función de prueba.
- `cleanup_engine()`: Realiza limpieza completa de un motor y sus recursos.

### Fixtures Optimizados (`tests/unit/core/conftest.py`)

- `non_blocking_engine`: Proporciona un `EngineNonBlocking` con limpieza.
- `configurable_engine`: Proporciona un `ConfigurableTimeoutEngine` con timeouts razonables.
- `priority_engine`: Proporciona un `EnginePriorityBlocks` con limpieza.
- `dynamic_engine`: Proporciona un `DynamicExpansionEngine` con limpieza.

## Uso de las Utilidades

### Ejemplo: Emisión de Eventos con Timeout

```python
from tests.utils.timeout_helpers import emit_with_timeout

# Emitir evento con un timeout de 2 segundos
response = await emit_with_timeout(
    engine, 
    "process_data", 
    {"size": 10}, 
    "component_a",
    timeout=2.0
)

# El resultado nunca será None, sino una lista con un elemento de error
# en caso de fallo o timeout
```

### Ejemplo: Verificación de Estado con Timeout

```python
from tests.utils.timeout_helpers import check_component_status

# Verificar estado con timeout de 1 segundo
status = await check_component_status(engine, "component_a", timeout=1.0)

# Acceder a campos de forma segura
is_healthy = status.get("healthy", False)
```

### Ejemplo: Medición de Tiempo de Ejecución

```python
from tests.utils.timeout_helpers import run_test_with_timing

async def my_test_function(engine):
    # Código de prueba...
    return True

# Ejecutar y medir
result = await run_test_with_timing(
    engine, 
    "mi_prueba_importante", 
    my_test_function
)
```

## Mejores Prácticas

1. **Siempre Usar Timeouts**: 
   - Use `emit_with_timeout()` en lugar de `emit_event_with_response()`
   - Establezca tiempos límite razonables (1-5 segundos según el contexto)

2. **Verificar Respuestas Defensivamente**:
   - Siempre maneje el caso de respuestas nulas o incorrectas
   - Use patrones como `valor = respuesta.get("campo", valor_por_defecto)`

3. **Limpiar Recursos**:
   - Use los fixtures proporcionados en lugar de crear motores directamente
   - Asegúrese de que todos los componentes se desregistren después de las pruebas

4. **Monitorear Rendimiento**:
   - Agregue mediciones de tiempo en pruebas lentas para identificar cuellos de botella
   - Use `run_test_with_timing()` para pruebas complejas

5. **Usar Marcadores para Pruebas Lentas**:
   ```python
   @pytest.mark.slow
   async def test_very_slow_operation():
       # Prueba que toma mucho tiempo...
   ```
   - Ejecute solo pruebas rápidas durante desarrollo: `pytest -m "not slow"`

## Actualización de Pruebas Existentes

Para cada prueba problemática, se ha creado una versión optimizada con sufijo `_optimized`. Las principales diferencias son:

1. Uso de `emit_with_timeout()` en lugar de `emit_event()`/`emit_event_with_response()`
2. Manejo defensivo de respuestas potencialmente nulas
3. Registro detallado de tiempos de ejecución
4. Timeouts y parámetros ajustados para pruebas más eficientes

## Ejecutando las Pruebas

Para ejecutar todas las pruebas:
```bash
pytest
```

Para ejecutar pruebas específicas:
```bash
# Pruebas optimizadas
pytest tests/unit/core/test_core_extreme_scenarios_optimized.py

# Pruebas de un módulo específico
pytest tests/unit/core/

# Solo pruebas rápidas
pytest -m "not slow"
```