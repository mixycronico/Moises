# Análisis de Fallos en Cascada y Soluciones

## Problema Identificado

En el sistema Genesis se han identificado problemas de fallos en cascada cuando se ejecutan pruebas bajo condiciones extremas, particularmente en los siguientes escenarios:

1. **Componentes no responden**: Cuando un componente falla o tarda demasiado, causa que otros componentes que dependen de él también fallen.
2. **Acceso a respuestas nulas**: Error recurrente `"Object of type 'None' is not subscriptable"` al intentar acceder a campos de respuestas que no existen.
3. **Timeouts no gestionados**: Las operaciones asíncronas sin límite de tiempo pueden causar bloqueos indefinidos.
4. **Propagación de fallos no controlada**: Los fallos se propagan más allá del ámbito previsto.

## Causas Raíz

### 1. Uso Incorrecto de Métodos de Emisión de Eventos

Se utilizaba `emit_event()` cuando debería usarse `emit_event_with_response()`:

```python
# Incorrecto
response = await engine.emit_event("check_status", {}, "comp_a")
status = response[0]["healthy"]  # Potencial error si response es None

# Correcto
response = await engine.emit_event_with_response("check_status", {}, "comp_a")
status = response[0]["healthy"] if response and len(response) > 0 else False
```

### 2. Ausencia de Manejo Defensivo para Respuestas

```python
# Incorrecto
result = response[0]["result"]

# Correcto
result = response[0]["result"] if response and len(response) > 0 and "result" in response[0] else None
```

### 3. Falta de Timeouts en Operaciones Asíncronas

```python
# Incorrecto - puede bloquear indefinidamente
response = await engine.emit_event_with_response("slow_operation", data, "comp_a")

# Correcto
try:
    response = await asyncio.wait_for(
        engine.emit_event_with_response("slow_operation", data, "comp_a"),
        timeout=5.0
    )
except asyncio.TimeoutError:
    logger.warning("La operación ha excedido el tiempo máximo permitido")
    response = [{"error": "timeout"}]
```

### 4. Recursos No Liberados Entre Pruebas

Los recursos (tareas asíncronas, conexiones) no se limpian correctamente entre pruebas, causando interferencias.

## Soluciones Implementadas

### 1. Función Auxiliar con Timeout

Se creó una función `emit_with_timeout` que encapsula la emisión de eventos con un timeout configurable:

```python
async def emit_with_timeout(engine, event_type, data, source, timeout=5.0):
    """
    Emitir evento con timeout y manejo robusto de errores.
    
    Args:
        engine: Instancia del motor de eventos
        event_type: Tipo de evento a emitir
        data: Datos del evento
        source: Fuente del evento
        timeout: Tiempo máximo de espera en segundos
        
    Returns:
        Lista de respuestas o un objeto de error si falla
    """
    try:
        return await asyncio.wait_for(
            engine.emit_event_with_response(event_type, data, source),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.warning(f"Timeout al esperar respuesta para {event_type} de {source}")
        return [{"error": "timeout", "event": event_type, "source": source}]
    except Exception as e:
        logger.error(f"Error al emitir evento {event_type}: {e}")
        return [{"error": str(e), "event": event_type, "source": source}]
```

### 2. Verificación Defensiva de Respuestas

Se implementaron helpers para acceder de forma segura a respuestas:

```python
def safe_get_response(response, key, default=None):
    """
    Obtener un valor de forma segura de una respuesta de evento.
    
    Args:
        response: Respuesta de un evento
        key: Clave a obtener
        default: Valor por defecto si no existe
        
    Returns:
        Valor o default si no existe
    """
    if not response or not isinstance(response, list) or len(response) == 0:
        return default
    
    first_response = response[0]
    if not isinstance(first_response, dict):
        return default
        
    return first_response.get(key, default)
```

### 3. Limpieza Mejorada de Recursos

Se implementó un mecanismo robusto para limpiar recursos entre pruebas:

```python
async def cleanup_engine(engine):
    """
    Limpieza completa del motor y tareas pendientes.
    
    Args:
        engine: Instancia del motor de eventos a limpiar
    """
    if not engine:
        return
        
    # Detener componentes
    components = list(engine.components.values()) if hasattr(engine, "components") else []
    for component in components:
        try:
            await engine.unregister_component(component.name)
        except Exception as e:
            logger.warning(f"Error al desregistrar componente {component.name}: {e}")
    
    # Detener el motor
    try:
        await engine.stop()
    except Exception as e:
        logger.warning(f"Error al detener el motor: {e}")
    
    # Pequeña pausa para que las tareas terminen
    await asyncio.sleep(0.1)
    
    # Verificar tareas pendientes
    pending = [t for t in asyncio.all_tasks() 
               if not t.done() and t != asyncio.current_task()]
               
    if pending:
        logger.warning(f"Hay {len(pending)} tareas pendientes al finalizar")
        # Opcionalmente, cancelar tareas pendientes
        for task in pending:
            task.cancel()
```

### 4. Mecanismo de Aislamiento de Componentes Problemáticos

Se implementó un sistema para aislar automáticamente componentes problemáticos:

```python
class ComponentMonitor:
    """Monitor de salud de componentes."""
    
    def __init__(self, engine):
        self.engine = engine
        self.health_status = {}
        self.isolation_status = {}
        
    async def check_all_components(self):
        """Verificar salud de todos los componentes."""
        for component_id in self.engine.components:
            await self.check_component(component_id)
            
    async def check_component(self, component_id):
        """Verificar salud de un componente específico."""
        response = await emit_with_timeout(
            self.engine, 
            "check_status", 
            {}, 
            component_id,
            timeout=2.0
        )
        
        healthy = safe_get_response(response, "healthy", False)
        self.health_status[component_id] = healthy
        
        if not healthy and not self.isolation_status.get(component_id, False):
            logger.warning(f"Componente {component_id} no saludable, aislando")
            await self.isolate_component(component_id)
            
    async def isolate_component(self, component_id):
        """Aislar un componente problemático."""
        self.isolation_status[component_id] = True
        
        # Notificar a otros componentes sobre el aislamiento
        await emit_with_timeout(
            self.engine,
            "component_isolated",
            {"component_id": component_id},
            "component_monitor",
            timeout=1.0
        )
```

## Resultados y Beneficios

La implementación de estas soluciones ha resultado en:

1. **Reducción de Errores**: Eliminación casi total de errores `"Object of type 'None' is not subscriptable"`.
2. **Mayor Estabilidad**: Las pruebas son más estables y predecibles.
3. **Tiempos de Ejecución Controlados**: Los timeouts evitan bloqueos indefinidos.
4. **Mejor Aislamiento**: Los fallos en componentes individuales ya no afectan al sistema completo.
5. **Diagnóstico Mejorado**: Mensajes de error más claros y detallados.

## Próximos Pasos

1. **Implementar Mecanismos de Recuperación Automática**: Permitir que componentes aislados intenten recuperarse automáticamente.
2. **Métricas de Resiliencia**: Implementar sistema para medir la resiliencia del sistema ante fallos.
3. **Simulación de Escenarios de Fallo**: Desarrollar herramientas para simular diferentes patrones de fallo.
4. **Documentación**: Actualizar la documentación con las nuevas prácticas recomendadas.

## Conclusión

Los fallos en cascada representaban un desafío importante para la estabilidad del sistema Genesis. Con las soluciones implementadas, el sistema es ahora significativamente más robusto ante fallos individuales, permitiendo una mayor confiabilidad en entornos de producción donde la tolerancia a fallos es crítica.