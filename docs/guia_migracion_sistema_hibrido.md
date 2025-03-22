# Guía de Migración al Sistema Híbrido API + WebSocket

Este documento proporciona una guía paso a paso para migrar componentes y aplicaciones existentes del sistema Genesis original al nuevo sistema híbrido API + WebSocket, diseñado para eliminar los problemas de deadlock y mejorar la escalabilidad.

## Índice

1. [Introducción](#introducción)
2. [Arquitectura del Sistema Híbrido](#arquitectura-del-sistema-híbrido)
3. [Ventajas del Sistema Híbrido](#ventajas-del-sistema-híbrido)
4. [Opciones de Migración](#opciones-de-migración)
5. [Migración Paso a Paso](#migración-paso-a-paso)
6. [Adaptación de Componentes Existentes](#adaptación-de-componentes-existentes)
7. [Ejemplos de Código](#ejemplos-de-código)
8. [Solución de Problemas](#solución-de-problemas)
9. [Mejores Prácticas](#mejores-prácticas)
10. [Referencias](#referencias)

## Introducción

El sistema Genesis original sufría de problemas de deadlock debido a la comunicación circular y recursiva entre componentes, así como a bloqueos prolongados en operaciones asíncronas. El nuevo sistema híbrido resuelve estos problemas mediante la separación clara de las comunicaciones:

- **API**: Para solicitudes directas con respuesta inmediata
- **WebSocket Local**: Para eventos internos en tiempo real entre componentes (en memoria)
- **WebSocket Externo**: Para comunicación con sistemas externos (a través de la red)

Esta separación elimina los deadlocks y mejora el rendimiento del sistema.

## Arquitectura del Sistema Híbrido

El sistema híbrido se compone de los siguientes elementos clave:

1. **GenesisHybridCoordinator**: Componente central que gestiona la comunicación entre componentes.
2. **ComponentAPI**: Interfaz base para componentes en el nuevo sistema.
3. **WebSocket Local**: Canal de comunicación en memoria para eventos entre componentes.
4. **WebSocket Externo**: Canal de comunicación por red para eventos con sistemas externos.
5. **API**: Canal para solicitudes directas entre componentes.

La arquitectura separa claramente las responsabilidades:

- Las **solicitudes** se realizan a través de la API con timeout, evitando bloqueos indefinidos.
- Los **eventos internos** se comunican a través del WebSocket local, sin overhead de red.
- Los **eventos externos** se comunican a través del WebSocket externo, para sistemas remotos.

## Ventajas del Sistema Híbrido

Las principales ventajas del sistema híbrido son:

1. **Eliminación de deadlocks**: La separación de canales y el uso de timeouts previene deadlocks.
2. **Mayor rendimiento**: Comunicación local en memoria sin overhead de red.
3. **Mejor escalabilidad**: Desacoplamiento claro entre los componentes.
4. **Comunicación en tiempo real**: Eventos asíncronos para comunicación no bloqueante.
5. **Compatibilidad con sistemas externos**: WebSocket externo para integración con otros sistemas.

## Opciones de Migración

Existen dos enfoques principales para migrar al sistema híbrido:

### 1. Migración Completa

Reimplementar todos los componentes utilizando la nueva interfaz `ComponentAPI`. Esta opción ofrece el mejor rendimiento y aprovecha todas las ventajas del sistema híbrido.

### 2. Migración Gradual con Adaptadores

Utilizar el adaptador `ComponentAdapter` para encapsular componentes existentes y hacerlos compatibles con el nuevo sistema. Esta opción permite una migración incremental sin modificar los componentes existentes.

## Migración Paso a Paso

### Migración Completa

1. Instalar las dependencias necesarias:
   ```bash
   pip install aiohttp websockets
   ```

2. Importar los módulos necesarios:
   ```python
   from genesis.core.genesis_hybrid import ComponentAPI, GenesisHybridCoordinator
   ```

3. Crear un nuevo coordinador:
   ```python
   coordinator = GenesisHybridCoordinator(host="localhost", port=8080)
   ```

4. Reimplementar los componentes heredando de `ComponentAPI`:
   ```python
   class MyComponent(ComponentAPI):
       async def process_request(self, request_type, data, source):
           # Procesar solicitudes directas
           pass
           
       async def on_local_event(self, event_type, data, source):
           # Procesar eventos locales
           await super().on_local_event(event_type, data, source)
   ```

5. Registrar los componentes en el coordinador:
   ```python
   my_component = MyComponent("my_component")
   coordinator.register_component("my_component", my_component)
   ```

6. Iniciar el coordinador:
   ```python
   await coordinator.start()
   ```

### Migración Gradual

1. Instalar las dependencias necesarias:
   ```bash
   pip install aiohttp websockets
   ```

2. Importar los módulos necesarios:
   ```python
   from genesis.core.component_adapter import HybridEngineAdapter
   ```

3. Crear un adaptador de motor:
   ```python
   engine_adapter = HybridEngineAdapter(host="localhost", port=8080)
   ```

4. Registrar componentes existentes en el adaptador:
   ```python
   engine_adapter.register_component(existing_component)
   ```

5. Iniciar el adaptador:
   ```python
   await engine_adapter.start()
   ```

## Adaptación de Componentes Existentes

Para adaptar componentes existentes sin modificarlos, utiliza el adaptador `ComponentAdapter`:

```python
from genesis.core.component_adapter import ComponentAdapter, HybridEngineAdapter

# Crear adaptador de motor
engine_adapter = HybridEngineAdapter()

# Registrar componente existente
engine_adapter.register_component(existing_component)

# Uso del motor adaptado
await engine_adapter.emit_event("event_type", data, "source_id")
result = await engine_adapter.request("target_id", "request_type", data, "source_id")
```

El adaptador se encarga de traducir entre las interfaces antigua y nueva, permitiendo que los componentes existentes funcionen en el nuevo sistema sin modificaciones.

## Ejemplos de Código

### Ejemplo de Componente Nativo del Sistema Híbrido

```python
from genesis.core.genesis_hybrid import ComponentAPI

class DataProcessor(ComponentAPI):
    def __init__(self, id):
        super().__init__(id)
        self.processed_data = []
    
    async def process_request(self, request_type, data, source):
        if request_type == "process_data":
            value = data.get("value", 0)
            processed = value * 2
            self.processed_data.append(processed)
            return {"original": value, "processed": processed}
        
        elif request_type == "get_results":
            return {"results": self.processed_data}
        
        return None
    
    async def on_local_event(self, event_type, data, source):
        await super().on_local_event(event_type, data, source)
        
        if event_type == "new_data_available":
            # Procesar datos disponibles
            pass
```

### Ejemplo de Uso del Adaptador

```python
import asyncio
from genesis.core.component import Component
from genesis.core.component_adapter import HybridEngineAdapter

# Componente existente
class ExistingComponent(Component):
    def __init__(self, id):
        super().__init__(id)
    
    async def handle_event(self, event_type, data, source):
        if event_type == "hello":
            return {"message": f"Hello from {self.id}"}
        return None

# Configurar adaptador
async def main():
    # Crear componentes existentes
    comp1 = ExistingComponent("comp1")
    comp2 = ExistingComponent("comp2")
    
    # Crear adaptador
    adapter = HybridEngineAdapter()
    
    # Registrar componentes
    adapter.register_component(comp1)
    adapter.register_component(comp2)
    
    # Iniciar adaptador
    await adapter.start()
    
    # Usar como antes
    result = await adapter.request("comp2", "hello", {}, "comp1")
    print(result)  # {"message": "Hello from comp2"}
    
    # Detener adaptador
    await adapter.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## Solución de Problemas

### Problema: Timeouts en Solicitudes API

**Síntoma**: Las solicitudes API a un componente específico dan timeout frecuentemente.

**Solución**: 
1. Aumentar el timeout para ese tipo específico de solicitud.
2. Revisar si el componente está realizando operaciones bloqueantes en `process_request`.
3. Considerar mover procesamiento pesado a tareas en segundo plano y usar eventos para notificar resultados.

### Problema: Acumulación de Eventos en Cola

**Síntoma**: Advertencias sobre colas de eventos llenas o componentes sin respuesta.

**Solución**:
1. Ajustar el tamaño máximo de la cola (`maxsize` en `asyncio.Queue`).
2. Optimizar el procesamiento de eventos en los componentes.
3. Considerar filtrar eventos menos importantes.

### Problema: Componente Adaptado No Recibe Eventos

**Síntoma**: Un componente adaptado no procesa eventos que debería recibir.

**Solución**:
1. Verificar que el adaptador esté correctamente configurado y el componente esté registrado.
2. Comprobar si hay excepciones en el manejo de eventos.
3. Verificar que los tipos de eventos sean correctos.

## Mejores Prácticas

1. **Solicitudes vs Eventos**:
   - Usa **solicitudes API** para operaciones que requieren respuesta inmediata.
   - Usa **eventos** para notificaciones sin respuesta requerida.

2. **Timeouts Adecuados**:
   - Configura timeouts apropiados según la operación.
   - Operaciones rápidas: 0.5-1.0 segundos.
   - Operaciones complejas: 2.0-5.0 segundos.

3. **Manejo de Errores**:
   - Implementa manejo de errores robusto en `process_request` y `on_local_event`.
   - Nunca permitas que excepciones no controladas escapen de estos métodos.

4. **Pruebas**:
   - Crea pruebas específicas para verificar que no hay deadlocks.
   - Prueba escenarios de comunicación circular y recursiva.
   - Verifica el comportamiento con múltiples componentes ejecutándose simultáneamente.

5. **Monitoreo**:
   - Utiliza las métricas proporcionadas por el sistema híbrido para monitorear el rendimiento.
   - Establece alertas para tiempos de respuesta anómalos.

## Referencias

- [Código Fuente: Sistema Híbrido](../genesis/core/genesis_hybrid.py)
- [Código Fuente: Adaptador de Componentes](../genesis/core/component_adapter.py)
- [Ejemplo: Sistema Híbrido Básico](../examples/hybrid_system_example.py)
- [Ejemplo: Adaptador de Componentes](../examples/hybrid_adapter_example.py)
- [Pruebas: Sistema Híbrido](../tests/unit/core/test_hybrid_system_basic.py)
- [Sistema Híbrido Optimizado](../genesis/core/genesis_hybrid_optimized.py)
- [Prueba del Sistema Optimizado](../examples/hybrid_system_optimized_test.py)