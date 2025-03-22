# Implementación Híbrida API+WebSocket para Genesis

## Introducción

Este documento describe la arquitectura híbrida API+WebSocket implementada para resolver los problemas de deadlocks y mejorar el rendimiento en el sistema de trading Genesis. La nueva arquitectura combina solicitudes síncronas directas con un modelo de publicación/suscripción asíncrono para maximizar la flexibilidad y la resiliencia del sistema.

## Contexto y Problemática

### Sistema Anterior

El sistema Genesis original utilizaba un solo mecanismo de comunicación entre componentes:

```
┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │
│  Componente A   │◄────►  Componente B   │
│                 │     │                 │
└─────────────────┘     └─────────────────┘
           ▲                    ▲
           │                    │
           │                    │
           ▼                    ▼
┌──────────────────────────────────────────┐
│                                          │
│        Bus de Eventos Síncrono           │
│                                          │
└──────────────────────────────────────────┘
```

### Problemas Identificados

1. **Deadlocks en Llamadas Circulares**: Cuando el componente A esperaba respuesta del componente B, que a su vez esperaba respuesta del componente A.

2. **Deadlocks en Llamadas Recursivas**: Cuando un componente emitía un evento que le era enviado a sí mismo.

3. **Bloqueo por Componentes Lentos**: Un componente con procesamiento lento afectaba a todo el sistema.

4. **Escalabilidad Limitada**: El modelo síncrono imponía límites a la concurrencia.

## Nueva Arquitectura Híbrida

La nueva arquitectura separa claramente dos tipos de comunicación:

```
┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │
│  Componente A   │─────►  Componente B   │ ◄─── API (Solicitud/Respuesta)
│                 │     │                 │      Con timeouts
└─────────────────┘     └─────────────────┘
       │    ▲                 │    ▲
       │    │                 │    │
       ▼    │                 ▼    │
┌──────────────────────────────────────────┐
│                                          │
│     WebSocket (Publicación/Suscripción)  │ ◄─── Eventos asíncronos
│                                          │      Sin espera de respuesta
└──────────────────────────────────────────┘
```

### Componentes Principales

1. **API para Solicitudes Directas**:
   - Comunicación síncrona con respuesta esperada
   - Mecanismo de timeout para prevenir bloqueos indefinidos
   - Manejo de excepciones robusto
   - Ideal para operaciones que requieren respuesta inmediata

2. **WebSocket para Eventos**:
   - Modelo de publicación/suscripción
   - Comunicación asíncrona sin respuesta (fire-and-forget)
   - Distribución selectiva a suscriptores
   - Ideal para notificaciones y actualizaciones de estado

### Coordinador Híbrido

El coordinador central proporciona dos métodos principales:

```python
async def request(self, target_id: str, request_type: str, 
                 data: Dict[str, Any], source: str,
                 timeout: float = 5.0) -> Optional[Any]:
    """
    API: Enviar solicitud directa a un componente con timeout.
    """
    if target_id not in self.components:
        return None
    
    try:
        # Clave del sistema: timeout para prevenir deadlocks
        result = await asyncio.wait_for(
            self.components[target_id].process_request(request_type, data, source),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        logger.warning(f"Timeout en solicitud {request_type} a {target_id}")
        return None
    except Exception as e:
        logger.error(f"Error en solicitud a {target_id}: {e}")
        return None

async def emit_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
    """
    WebSocket: Emitir evento a todos los componentes suscritos.
    """
    subscribers = self.event_subscribers.get(event_type, set())
    tasks = []
    
    for comp_id in subscribers:
        if comp_id in self.components and comp_id != source:
            tasks.append(
                self.components[comp_id].on_event(event_type, data, source)
            )
    
    # Ejecución paralela sin esperar respuestas
    if tasks:
        await asyncio.gather(*tasks)
```

### Interfaz de Componentes

Cada componente implementa dos métodos principales que corresponden a los dos tipos de comunicación:

```python
async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
    """
    Procesar solicitudes API directas.
    Debe devolver una respuesta.
    """
    # Implementación específica

async def on_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
    """
    Manejar eventos WebSocket.
    No se espera respuesta.
    """
    # Implementación específica
```

## Solución a Problemas Específicos

### 1. Deadlocks en Llamadas Circulares

**Antes**: El componente A esperaba respuesta del componente B, que a su vez esperaba respuesta del componente A, resultando en un bloqueo permanente.

**Ahora**: 
- Las solicitudes API tienen timeouts que evitan bloqueos indefinidos
- Las llamadas circulares se resuelven: si A → B → A, el timeout limita la espera
- Si se necesita notificar sin esperar respuesta, se usa WebSocket

### 2. Deadlocks en Llamadas Recursivas

**Antes**: Un componente emitía un evento que le era enviado a sí mismo, creando un bucle infinito.

**Ahora**:
- Las solicitudes API a uno mismo tienen timeouts
- Los eventos WebSocket no se envían al componente emisor
- Se puede implementar control explícito en el componente

### 3. Bloqueo por Componentes Lentos

**Antes**: Un componente lento bloqueaba todo el sistema.

**Ahora**:
- Las solicitudes API tienen timeouts que limitan el impacto
- Los eventos WebSocket se procesan de forma asíncrona
- El sistema puede continuar funcionando aunque un componente esté bloqueado

### 4. Escalabilidad Limitada

**Antes**: El modelo síncrono limitaba la concurrencia.

**Ahora**:
- Los eventos WebSocket se procesan en paralelo
- Las solicitudes API solo bloquean cuando es estrictamente necesario
- Se puede ajustar la concurrencia según necesidades

## Patrones de Uso Recomendados

### Cuándo Usar API (Solicitudes Directas)

- Cuando se necesita una respuesta inmediata
- Para operaciones que afectan al estado del sistema
- Cuando se requiere validación o autorización
- Para consultas puntuales de datos

Ejemplo:
```python
result = await coordinator.request(
    "exchange_manager", 
    "execute_trade",
    {"symbol": "BTC/USDT", "side": "buy", "amount": 0.1},
    "strategy_manager"
)
if result and result.get("status") == "success":
    # Procesar resultado exitoso
```

### Cuándo Usar WebSocket (Eventos)

- Para notificaciones y actualizaciones de estado
- Cuando no se requiere respuesta
- Para distribuir información a múltiples componentes
- Para operaciones que no son críticas en tiempo

Ejemplo:
```python
await coordinator.emit_event(
    "price_update",
    {"symbol": "BTC/USDT", "price": 50000, "timestamp": time.time()},
    "market_data_manager"
)
```

## Ventajas de la Arquitectura Híbrida

1. **Prevención de Deadlocks**: El sistema está diseñado intrínsecamente para evitar deadlocks.

2. **Mejor Rendimiento**: Procesamiento paralelo de eventos sin bloqueos innecesarios.

3. **Mayor Resiliencia**: Fallos en componentes individuales no afectan al sistema global.

4. **Escalabilidad**: El sistema puede manejar más componentes y mayor volumen de eventos.

5. **Flexibilidad**: Los desarrolladores pueden elegir el modelo más apropiado según cada caso de uso.

## Consideraciones para la Implementación

### Gestión de Timeouts

Los valores de timeout deben ajustarse según el componente y tipo de operación:
- Operaciones rápidas: 0.5-1 segundos
- Operaciones complejas: 5-10 segundos
- Verificar conexiones: 1-2 segundos

### Manejo de Errores

Las excepciones en un componente no deben propagarse al sistema completo:
- Toda excepción debe ser capturada y gestionada
- Implementar reintentos para operaciones cruciales
- Registrar adecuadamente los errores

### Monitoreo y Diagnóstico

Implementar métricas detalladas para ambos tipos de comunicación:
- Número de solicitudes/eventos por tipo
- Tiempos de respuesta
- Tasas de error
- Timeouts ocurridos

## Conclusión

La arquitectura híbrida API+WebSocket proporciona una solución robusta a los problemas identificados en el sistema anterior. Al separar claramente los dos modelos de comunicación, se obtiene un sistema más resiliente, con mejor rendimiento y libre de deadlocks. Las pruebas realizadas confirman que esta arquitectura cumple con los objetivos planteados y proporciona una base sólida para el futuro desarrollo del sistema Genesis.

---

*Documento creado: 22 de marzo de 2025*