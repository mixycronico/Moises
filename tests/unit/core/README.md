# Tests del Módulo Core de Genesis

## Descripción General

Este directorio contiene las pruebas unitarias para el módulo core del sistema Genesis, incluyendo componentes fundamentales como el Engine, EventBus, Component y Settings.

## Estructura de Pruebas

Las pruebas están organizadas en varios archivos según su propósito y complejidad:

### Pruebas Básicas
- `test_core_basic.py`: Pruebas básicas de componentes del core
- `test_core_engine_basic.py`: Pruebas básicas del motor
- `test_core_event_bus.py`: Pruebas del bus de eventos
- `test_core_settings.py`: Pruebas de la configuración del sistema

### Pruebas Intermedias
- `test_core_intermediate.py`: Pruebas más complejas de componentes del core
- `test_core_engine_intermediate.py`: Pruebas intermedias del motor
- `test_core_intermediate_optimized.py`: Versión optimizada de pruebas intermedias
- `test_core_engine_intermediate_optimized.py`: Versión optimizada de pruebas intermedias del motor

### Pruebas Avanzadas
- `test_core_advanced.py`: Pruebas avanzadas de componentes del core
- `test_core_engine_advanced.py`: Pruebas avanzadas del motor
- `test_core_advanced_simplified.py`: Versión simplificada de pruebas avanzadas

### Pruebas Especializadas
- `test_engine_non_blocking.py`: Pruebas del motor no bloqueante
- `test_non_blocking_emit.py`: Pruebas específicas del método emit_event en el motor no bloqueante
- `test_engine_manual_events.py`: Pruebas con envío manual de eventos a través del motor
- `test_error_handling_direct.py`: Pruebas directas de manejo de errores en componentes
- `test_engine_solution.py`: Prueba final que demuestra la solución al problema de manejo de errores

### Pruebas de Prioridad
- `test_core_priority_simplified.py`: Pruebas simplificadas de prioridad de componentes
- `test_core_priority_minimal.py`: Pruebas mínimas de prioridad
- `test_core_priority_ultra_minimal.py`: Pruebas ultra mínimas de prioridad
- `test_core_priority_alternative.py`: Enfoque alternativo para pruebas de prioridad
- `test_core_priority_simplified_v2.py`: Segunda versión de pruebas simplificadas de prioridad

### Pruebas Minimalistas
- `test_engine_minimal.py`: Pruebas mínimas del motor
- `test_engine_ultra_minimal.py`: Pruebas ultra mínimas del motor
- `test_engine_optimized_minimal.py`: Pruebas mínimas del motor optimizado
- `test_final_minimal.py`: Prueba minimalista final para verificación rápida

## Solución de Manejo de Errores

Un enfoque especial se ha dado a la solución del problema de manejo de errores en el sistema. Esta solución incluye:

1. **EngineNonBlocking**: Una implementación no bloqueante del motor que maneja correctamente los errores en componentes.

2. **EventBusMinimal**: Una implementación simplificada del bus de eventos para pruebas más controladas.

3. **Estrategias de Prueba**: Diversas estrategias para probar el manejo de errores, desde pruebas directas de componentes hasta pruebas completas del sistema.

Para más detalles sobre la solución de manejo de errores, consultar `docs/error_handling_solution.md`.

## Cómo Ejecutar las Pruebas

Para ejecutar todas las pruebas del módulo core:

```bash
python -m pytest tests/unit/core
```

Para ejecutar un archivo de prueba específico:

```bash
python -m pytest tests/unit/core/test_engine_non_blocking.py
```

Para ejecutar con detalles verbosos:

```bash
python -m pytest tests/unit/core -v
```

## Notas Importantes

- Algunas pruebas utilizan configuraciones especiales como modo de prueba y timeouts reducidos.
- Las pruebas que verifican el manejo de errores pueden mostrar trazas de error en la consola, lo cual es esperado.
- Las advertencias sobre clases de prueba que no pueden ser recolectadas debido a constructores __init__ son normales y pueden ignorarse.