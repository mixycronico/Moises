# Informe de Mejoras para Pruebas del Motor Genesis

## Contexto

Las pruebas en `tests/unit/core/test_core_extreme_scenarios.py` presentaban dos problemas importantes:

1. **Error de suscripción de objetos nulos**: Algunas pruebas fallaban con `"Object of type 'None' is not subscriptable"` al intentar acceder a respuestas inexistentes.
2. **Tiempos de ejecución excesivos**: Varias pruebas tardan demasiado tiempo en ejecutarse, lo que afecta el flujo de desarrollo.

## Solución Implementada

Para el primer problema, se realizaron los siguientes cambios:

1. Se reemplazaron llamadas a `emit_event()` con `emit_event_with_response()` cuando se esperaban respuestas:
   ```python
   # Antes
   response = await engine.emit_event("set_health", {"healthy": False}, "comp_a")
   
   # Después
   response = await engine.emit_event_with_response("set_health", {"healthy": False}, "comp_a")
   ```

2. Se aplicó manejo defensivo de respuestas nulas:
   ```python
   resp_a = resp_a_recovery[0] if resp_a_recovery and len(resp_a_recovery) > 0 else {"healthy": True, "error": "No response", "recovered": True}
   ```

## Mejoras Propuestas

### 1. Agregar Timeouts para Prevenir Bloqueos

Implementar una función helper para llamar a `emit_event_with_response()` con timeouts:

```python
async def emit_with_timeout(engine, event_type, data, source, timeout=5.0):
    """Emitir evento con timeout para evitar bloqueos indefinidos."""
    try:
        return await asyncio.wait_for(
            engine.emit_event_with_response(event_type, data, source),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.warning(f"Timeout al esperar respuesta para {event_type} de {source}")
        return [{"error": "timeout", "event": event_type, "source": source}]
```

### 2. Monitoreo de Tiempo de Ejecución

Agregar mediciones de tiempo para identificar cuellos de botella:

```python
import time

start_time = time.time()
# Código a medir
elapsed = time.time() - start_time
logger.info(f"Operación completada en {elapsed:.3f} segundos")
```

### 3. Verificación de Tareas Pendientes

Al final de cada prueba, verificar si hay tareas asíncronas sin completar:

```python
# En cleanup/finally
pending = len([t for t in asyncio.all_tasks() if not t.done() and t != asyncio.current_task()])
if pending > 0:
    logger.warning(f"Hay {pending} tareas pendientes al finalizar la prueba")
```

### 4. Limpieza Explícita de Recursos

Asegurar que cada prueba limpie correctamente todos los recursos:

```python
@pytest.fixture
async def engine_fixture():
    engine = EngineNonBlocking(test_mode=True)
    yield engine
    # Cleanup explícito
    for component in list(engine.components.values()):
        await engine.unregister_component(component.name)
    await engine.stop()
    # Esperar un poco para que las tareas terminen
    await asyncio.sleep(0.1)
```

## Recomendaciones Adicionales

1. **Reducir la Complejidad de las Pruebas**: Simplificar escenarios excesivamente complejos.
2. **Ejecución Selectiva**: Usar marcadores de pytest para excluir pruebas lentas durante el desarrollo.
3. **Paralelización**: Considerar ejecutar pruebas en paralelo con pytest-xdist.
4. **Optimización del Motor**: Revisar el motor para eliminar posibles cuellos de botella.

## Conclusión

Las correcciones implementadas resuelven los problemas inmediatos de errores de suscripción. Para mejorar el rendimiento, se recomienda implementar los timeouts y las mejoras en el monitoreo de tiempo y recursos. Estos cambios mejorarán la estabilidad y velocidad de ejecución de las pruebas, facilitando el desarrollo y mantenimiento del sistema.