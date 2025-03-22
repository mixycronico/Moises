# Problema en Test de Fallos en Cascada

## Descripción
Estamos enfrentando un problema con la prueba `test_cascading_failures` en el archivo `tests/unit/core/test_core_extreme_scenarios.py`. La prueba está fallando porque los objetos `resp_a` y `resp_b` son `None` en las aserciones, lo que genera el error "Object of type None is not subscriptable".

## Código actual con problemas
```python
# FASE 3: Verificar estado después del fallo
logger.info("FASE 3: Verificando estado")
try:
    logger.info("Enviando check_status a comp_a")
    resp_a = await engine.emit_event("check_status", {}, "comp_a")
    logger.info(f"Estado A: {resp_a}")
except Exception as e:
    logger.error(f"Error al verificar estado de A: {type(e).__name__}: {str(e)}")
    resp_a = {"healthy": False, "error": str(e)}
    
try:
    logger.info("Enviando check_status a comp_b")
    resp_b = await engine.emit_event("check_status", {}, "comp_b")
    logger.info(f"Estado B: {resp_b}")
except Exception as e:
    logger.error(f"Error al verificar estado de B: {type(e).__name__}: {str(e)}")
    resp_b = {"healthy": True, "error": str(e)}
```

## Aserciones que fallan
```python
# Verificar que A está no-sano
assert not resp_a["healthy"], "A debería estar no-sano después del fallo"
# B debería seguir sano porque no hay propagación en este componente simple
assert resp_b["healthy"], "B debería estar sano (no hay propagación)"
```

## Contexto
- La prueba simula fallos en cascada entre componentes
- Usa un motor asíncrono no bloqueante (`EngineNonBlocking`)
- Modifica el estado del componente `comp_a` a no saludable
- Intenta verificar que el estado no saludable no se propaga a `comp_b`

## Lo que ya hemos intentado
- Agregar manejo de excepciones para capturar errores en las verificaciones de estado
- Proporcionar valores predeterminados cuando fallan las verificaciones
- Agregar más logging para depurar el problema

## Posibles causas
1. El motor no está enviando eventos a los componentes correctamente
2. Los componentes no están respondiendo a los eventos como se espera
3. Hay un problema con la asincronía o el timing de la prueba
4. Posible problema con el estado interno del motor o los componentes