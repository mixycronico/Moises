# Problema en tests de eventos y manejo de respuestas

## Descripción del problema

En el archivo `tests/unit/core/test_core_extreme_scenarios.py` se identificaron errores críticos relacionados con el manejo incorrecto de respuestas de eventos:

```
Error: Object of type 'None' is not subscriptable
```

## Causa raíz

1. **Uso incorrecto del método `emit_event()`**:
   - Se utilizaba `emit_event()` cuando se necesitaba capturar respuestas
   - Este método no está diseñado para devolver respuestas utilizables
   - Al intentar acceder a valores de respuesta nulos, se producía el error

2. **Líneas problemáticas**:
   ```python
   # Línea 987
   response = await engine.emit_event("set_health", {"healthy": False}, "comp_a")
   
   # Línea 1028
   await engine.emit_event("set_health", {"healthy": True}, "comp_a")
   ```

## Solución aplicada

Se reemplazaron las llamadas a `emit_event()` por llamadas a `emit_event_with_response()`:

```python
# Línea 987 - corregida
response = await engine.emit_event_with_response("set_health", {"healthy": False}, "comp_a")

# Línea 1028 - corregida
await engine.emit_event_with_response("set_health", {"healthy": True}, "comp_a")
```

## Resultados

1. ✅ La prueba `test_cascading_failures` ahora pasa correctamente
2. ⚠️ Otras pruebas en el mismo archivo parecen tardar demasiado tiempo en ejecutarse
3. ℹ️ Los errores LSP en otros archivos son solo advertencias del analizador estático

## Problemas pendientes

Hay pruebas adicionales en el archivo que están tardando demasiado tiempo en ejecutarse, lo que podría indicar:
- Posibles problemas de rendimiento
- Timeouts no manejados adecuadamente
- Eventos asíncronos que no se resuelven correctamente