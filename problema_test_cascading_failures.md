# Problema: Tests de Fallos en Cascada

## Descripción del Problema

Se han identificado problemas recurrentes al ejecutar las pruebas de fallos en cascada en el sistema Genesis. Estos problemas se manifiestan de las siguientes formas:

1. **Error de suscripción de objetos nulos**: `"Object of type 'None' is not subscriptable"` aparece frecuentemente en las pruebas.
2. **Timeouts en la ejecución de pruebas**: Algunas pruebas nunca terminan o tardan demasiado tiempo.
3. **Resultados inconsistentes**: La misma prueba puede pasar o fallar en diferentes ejecuciones.
4. **Fallos en cascada no reproducibles**: Ciertos patrones de fallos no se reproducen de manera consistente.

## Escenarios Problemáticos

### Caso 1: Test de Propagación de Fallos

```python
@pytest.mark.asyncio
async def test_cascading_failure_propagation():
    engine = EngineNonBlocking()
    
    # Configurar componentes en cadena
    comp_a = TestComponent("comp_a")
    comp_b = DependentComponent("comp_b", ["comp_a"])
    comp_c = DependentComponent("comp_c", ["comp_b"])
    
    await engine.register_component(comp_a)
    await engine.register_component(comp_b)
    await engine.register_component(comp_c)
    
    # Verificar estado inicial
    response_a = await engine.emit_event("check_status", {}, "comp_a")
    response_b = await engine.emit_event("check_status", {}, "comp_b")
    response_c = await engine.emit_event("check_status", {}, "comp_c")
    
    # ERROR: Los objetos response_X pueden ser None
    assert response_a[0]["healthy"] == True
    assert response_b[0]["healthy"] == True
    assert response_c[0]["healthy"] == True
    
    # Fallar el componente A
    await engine.emit_event("set_health", {"healthy": False}, "comp_a")
    
    # Verificar propagación de fallos
    response_a = await engine.emit_event("check_status", {}, "comp_a")
    response_b = await engine.emit_event("check_status", {}, "comp_b")
    response_c = await engine.emit_event("check_status", {}, "comp_c")
    
    # ERROR: Los objetos response_X pueden ser None
    assert response_a[0]["healthy"] == False
    assert response_b[0]["healthy"] == False
    assert response_c[0]["healthy"] == False
```

### Caso 2: Recuperación Automática Tras Fallo

```python
@pytest.mark.asyncio
async def test_auto_recovery_after_failure():
    engine = EngineNonBlocking()
    
    # Configurar componentes
    comp_a = TestComponent("comp_a")
    comp_b = RecoveringComponent("comp_b", depends_on=["comp_a"])
    
    await engine.register_component(comp_a)
    await engine.register_component(comp_b)
    
    # Fallar componente principal
    await engine.emit_event("set_health", {"healthy": False}, "comp_a")
    
    # Verificar fallo propagado
    response_b = await engine.emit_event("check_status", {}, "comp_b")
    
    # ERROR: response_b puede ser None
    assert response_b[0]["healthy"] == False
    
    # Restaurar componente principal
    await engine.emit_event("set_health", {"healthy": True}, "comp_a")
    
    # Dar tiempo para recuperación
    await asyncio.sleep(1.0)  # Posible problema de carrera
    
    # Verificar recuperación
    response_b = await engine.emit_event("check_status", {}, "comp_b")
    
    # ERROR: timeout posible, response_b puede ser None
    assert response_b[0]["healthy"] == True
```

## Análisis de Causas Raíz

### 1. Uso Incorrecto de Métodos de Emisión de Eventos

El patrón más común es utilizar `emit_event()` cuando se espera obtener respuestas:

```python
# Código problemático
response = await engine.emit_event("check_status", {}, "component_id")
status = response[0]["healthy"]  # Error si response es None
```

Este método no garantiza que devuelva una estructura utilizable para acceder a campos como `response[0]["healthy"]`.

### 2. Falta de Verificación de Respuestas Nulas

No se verifica que la respuesta sea válida antes de acceder a sus campos:

```python
# Código problemático
status = response[0]["healthy"]  # Error si response es None o vacío
```

### 3. Operaciones Asíncronas Sin Timeout

Las operaciones asíncronas carecen de límites de tiempo, lo que puede llevar a bloqueos indefinidos:

```python
# Código problemático - puede bloquear indefinidamente
response = await engine.emit_event_with_response("operation", data, "component_id")
```

### 4. Dependencias Circulares No Detectadas

Algunos tests crean dependencias circulares entre componentes:

```python
# Potencial problema
comp_a = DependentComponent("comp_a", ["comp_b"])
comp_b = DependentComponent("comp_b", ["comp_c"])
comp_c = DependentComponent("comp_c", ["comp_a"])  # Dependencia circular
```

## Solución Propuesta

### 1. Reemplazar `emit_event()` por `emit_event_with_response()` o `emit_with_timeout()`

```python
# Solución
response = await engine.emit_event_with_response("check_status", {}, "component_id")
```

O mejor aún, utilizar la función helper con timeout:

```python
response = await emit_with_timeout(engine, "check_status", {}, "component_id", timeout=2.0)
```

### 2. Implementar Manejo Defensivo de Respuestas

```python
# Solución
healthy = False
if response and isinstance(response, list) and len(response) > 0:
    if isinstance(response[0], dict) and "healthy" in response[0]:
        healthy = response[0]["healthy"]
```

O utilizar el helper seguro:

```python
healthy = safe_get_response(response, "healthy", default=False)
```

### 3. Agregar Timeouts a Todas las Operaciones Asíncronas

```python
# Solución
try:
    response = await asyncio.wait_for(
        engine.emit_event_with_response("operation", data, "component_id"),
        timeout=5.0
    )
except asyncio.TimeoutError:
    logger.warning("Operación ha excedido el tiempo máximo")
    response = [{"error": "timeout"}]
```

### 4. Implementar Fixtures para Limpieza de Recursos

```python
@pytest.fixture
async def test_engine():
    engine = EngineNonBlocking()
    yield engine
    # Limpieza explícita
    await cleanup_engine(engine)
```

### 5. Detectar y Prevenir Dependencias Circulares

```python
def detect_circular_dependencies(component_dependencies):
    """Detectar dependencias circulares en un conjunto de componentes."""
    visited = set()
    path = []
    
    def visit(component):
        if component in path:
            cycle = path[path.index(component):] + [component]
            return cycle
        if component in visited:
            return None
            
        visited.add(component)
        path.append(component)
        
        for dep in component_dependencies.get(component, []):
            cycle = visit(dep)
            if cycle:
                return cycle
                
        path.pop()
        return None
    
    for component in component_dependencies:
        cycle = visit(component)
        if cycle:
            return cycle
            
    return None
```

## Plan de Implementación

1. **Fase 1**: Corregir los métodos de emisión de eventos y agregar verificaciones defensivas
   - Reemplazar `emit_event()` por `emit_event_with_response()` cuando se necesiten respuestas
   - Implementar función helper `emit_with_timeout()`
   - Implementar función helper `safe_get_response()`

2. **Fase 2**: Mejorar la gestión de recursos
   - Implementar `cleanup_engine()` para limpieza correcta de recursos
   - Crear fixture genérico para motor con limpieza automática

3. **Fase 3**: Detectar y prevenir condiciones problemáticas
   - Implementar detección de dependencias circulares
   - Agregar verificaciones de componentes huérfanos (con dependencias no registradas)

4. **Fase 4**: Refactorizar pruebas de fallos en cascada
   - Separar pruebas complejas en casos de prueba más simples
   - Agregar logging detallado para diagnosticar problemas

## Conclusión

Los tests de fallos en cascada son fundamentales para garantizar la resiliencia del sistema Genesis, pero requieren un manejo cuidadoso de las interacciones asíncronas y las dependencias entre componentes. Las soluciones propuestas abordan las causas raíz de los problemas identificados y proporcionarán un marco más robusto para probar estos escenarios críticos.

Al implementar estas soluciones, no solo resolveremos los problemas actuales, sino que también estableceremos patrones que prevendrán la aparición de problemas similares en el futuro.