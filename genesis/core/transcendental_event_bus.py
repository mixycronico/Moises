"""
Adaptador Trascendental para reemplazar el EventBus con WebSocket/API Local.

Este módulo implementa un adaptador que reemplaza completamente el event_bus
tradicional con el sistema híbrido de WebSocket/API Trascendental, manteniendo
compatibilidad con la interfaz existente mientras proporciona todas las 
capacidades trascendentales.
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, Optional, Callable, Awaitable, List
from websockets.server import WebSocketServerProtocol as WebSocket

# Importar mecanismos y componentes trascendentales
from genesis_singularity_transcendental_v4 import (
    TranscendentalWebSocket, 
    TranscendentalAPI,
    DimensionalCollapseV4,
    EventHorizonV4,
    QuantumTimeV4,
    InfiniteDensityV4,
    OmniversalSharedMemory,
    PredictiveRecoverySystem,
    EvolvingConsciousInterface,
    QuantumTunnelV4,
    ResilientReplicationV4,
    EntanglementV4,
    RealityMatrixV4,
    OmniConvergenceV4,
    QuantumFeedbackLoop
)

# Configuración de logging
logger = logging.getLogger("Genesis.TranscendentalEventBus")

# Tipo para manejadores de eventos
EventHandler = Callable[[str, Dict[str, Any], str], Awaitable[None]]

class TranscendentalEventBus:
    """
    Implementación de EventBus con capacidades trascendentales V4.
    
    Reemplaza el EventBus tradicional con un sistema híbrido WebSocket/API,
    integrando los 13 mecanismos trascendentales para operar a intensidad 1000.0.
    """
    
    def __init__(self, ws_uri: str = "ws://localhost:8080", api_url: str = "http://localhost:8000", test_mode: bool = False):
        """
        Inicializar el EventBus Trascendental.
        
        Args:
            ws_uri: URI del WebSocket local
            api_url: URL de la API local
            test_mode: Si se ejecuta en modo prueba
        """
        # Parsear host y puerto del URI WebSocket
        try:
            host = ws_uri.split("://")[1].split(":")[0]
            port = int(ws_uri.split(":")[-1])
        except Exception:
            host = "localhost"
            port = 8080
            logger.warning(f"Error al parsear URI WebSocket {ws_uri}, usando valores por defecto")
        
        # Inicializar componentes trascendentales
        self.ws = TranscendentalWebSocket(ws_uri)
        self.api = TranscendentalAPI(api_url)
        
        # Mecanismos trascendentales completos (los 13)
        self.mechanisms = {
            "dimensional": DimensionalCollapseV4(),
            "horizon": EventHorizonV4(),
            "quantum_time": QuantumTimeV4(),
            "density": InfiniteDensityV4(),
            "memory": OmniversalSharedMemory(),
            "predictive": PredictiveRecoverySystem(),
            "conscious": EvolvingConsciousInterface(),
            "tunnel": QuantumTunnelV4(),
            "replication": ResilientReplicationV4(),
            "entanglement": EntanglementV4(),
            "reality": RealityMatrixV4(),
            "convergence": OmniConvergenceV4(),
            "feedback": QuantumFeedbackLoop()
        }
        
        # Estado del bus
        self.running = False
        self.test_mode = test_mode
        
        # Mapa de suscriptores trascendental
        # Estructura: {event_type: [(priority, handler, component_id)]}
        self.subscribers: Dict[str, List[tuple[int, EventHandler, str]]] = {}
        
        # Estadísticas avanzadas
        self.stats = {"events_emitted": 0, "events_delivered": 0, "errors_transmuted": 0}
        
        # Iniciar tareas en segundo plano
        asyncio.create_task(self._run_background_tasks())
        
        # Inicializar en modo prueba si es necesario
        if self.test_mode:
            logger.debug("EventBus Trascendental: inicializado en modo prueba")
            self.running = True
    
    async def start(self) -> None:
        """Iniciar el EventBus Trascendental."""
        if self.running:
            return
            
        logger.info("Iniciando EventBus Trascendental...")
        
        # Inicializar componentes en paralelo con entrelazamiento
        await asyncio.gather(
            self.api.initialize(),
            self.mechanisms["entanglement"].entangle_components([self.ws, self.api])
        )
        
        # Iniciar websocket en tarea separada
        asyncio.create_task(self.ws.run())
        
        self.running = True
        logger.info("EventBus Trascendental activo y operando en todas las dimensiones")
    
    async def stop(self) -> None:
        """Detener el EventBus Trascendental."""
        if not self.running:
            return
            
        logger.info("Deteniendo EventBus Trascendental...")
        
        # Preservar estado en memoria omniversal
        await self.mechanisms["memory"].store_state({"shutdown": True}, self.stats)
        
        # Detener websocket y cerrar sesión API
        await self.api.close()
        self.running = False
        
        logger.info("EventBus Trascendental: detenido")
    
    async def subscribe(self, event_type: str, handler: EventHandler, priority: int = 0, component_id: str = "unknown") -> None:
        """
        Suscribir un manejador a un tipo de evento con capacidades trascendentales.
        
        Args:
            event_type: Tipo de evento
            handler: Función manejadora
            priority: Prioridad (menor = mayor prioridad)
            component_id: ID del componente que se suscribe
        """
        # Crear lista de suscriptores si no existe
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        # Añadir suscriptor con su prioridad
        self.subscribers[event_type].append((priority, handler, component_id))
        
        # Ordenar por prioridad (menor número = mayor prioridad)
        self.subscribers[event_type].sort(key=lambda x: x[0])
        
        # Almacenar en memoria omniversal para recuperación
        await self.mechanisms["memory"].store_state(
            {"event_type": event_type, "component_id": component_id},
            {"priority": priority, "timestamp": time.time()}
        )
        
        logger.debug(f"Suscripción trascendental: {component_id} a {event_type}, prioridad={priority}")
    
    async def subscribe_once(self, event_type: str, handler: EventHandler, priority: int = 0, component_id: str = "unknown") -> None:
        """
        Suscribir un manejador para un solo evento con capacidades trascendentales.
        
        Args:
            event_type: Tipo de evento
            handler: Función manejadora
            priority: Prioridad (menor = mayor prioridad)
            component_id: ID del componente que se suscribe
        """
        # Crear un wrapper que se auto-desuscribirá después de la primera ejecución
        async def one_time_wrapper(event_type: str, data: Dict[str, Any], source: str) -> None:
            # Ejecutar manejador original
            await handler(event_type, data, source)
            
            # Desuscribir este wrapper
            await self.unsubscribe(event_type, one_time_wrapper)
        
        # Suscribir el wrapper
        await self.subscribe(event_type, one_time_wrapper, priority, component_id)
    
    async def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """
        Desuscribir un manejador de un tipo de evento.
        
        Args:
            event_type: Tipo de evento
            handler: Función manejadora a desuscribir
        """
        if event_type in self.subscribers:
            # Filtrar la lista para eliminar el manejador
            self.subscribers[event_type] = [(p, h, c) for p, h, c in self.subscribers[event_type] if h != handler]
            
            # Si no quedan suscriptores, eliminar la clave
            if not self.subscribers[event_type]:
                del self.subscribers[event_type]
    
    async def emit(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Emitir un evento con capacidades trascendentales.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente que emite el evento
        """
        if not self.running and not self.test_mode:
            logger.warning(f"Intento de emitir evento {event_type} cuando el bus no está activo")
            return
        
        self.stats["events_emitted"] += 1
        
        # Crear evento con metadatos
        event = {
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": time.time(),
            "_transcendental": True
        }
        
        # Procesar evento a través de cascada trascendental
        try:
            async with self.mechanisms["quantum_time"].nullify_time():
                # Cascada de procesamiento trascendental
                collapsed = await self.mechanisms["dimensional"].collapse_data(event)
                compressed = await self.mechanisms["density"].compress(collapsed)
                tunneled = await self.mechanisms["tunnel"].tunnel_data(compressed)
                feedback = await self.mechanisms["feedback"].feedback(tunneled)
                optimized = await self.mechanisms["reality"].optimize(feedback)
                
                # Verificar convergencia antes de procesar
                if await self.mechanisms["convergence"].converge():
                    # Verificar suscriptores
                    if event_type in self.subscribers:
                        # En modo prueba o sin cola, entregar directamente
                        if self.test_mode:
                            await self._deliver_event(event_type, data, source)
                        else:
                            # Enviar a través de WebSocket/API trascendental
                            await self._process_event(optimized)
                    else:
                        logger.debug(f"Sin suscriptores para el evento {event_type}")
        
        except Exception as e:
            # Transmutación de error en energía
            error_info = await self.mechanisms["horizon"].transmute_error(e, {"event_type": event_type})
            logger.debug(f"Error transmutado al emitir evento: {error_info['original_error']}")
            self.stats["errors_transmuted"] += 1
    
    async def _process_event(self, event: Dict[str, Any]) -> None:
        """
        Procesar evento a través del sistema híbrido.
        
        Args:
            event: Evento completo con metadatos
        """
        event_type = event["type"]
        data = event["data"]
        source = event["source"]

        # Procesamiento híbrido WebSocket/API en paralelo
        await asyncio.gather(
            self.ws.process_message(event),
            self.api.send_request("event", event),
            self._deliver_event(event_type, data, source)
        )
        
        # Actualizar estadísticas
        self.stats["events_delivered"] += len(self.subscribers.get(event_type, []))
    
    async def _deliver_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Entregar evento a los suscriptores de forma trascendental.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente que emite el evento
        """
        # Solo procesar si hay suscriptores
        if event_type not in self.subscribers:
            return
        
        # Crear tareas para procesamiento paralelo
        tasks = []
        for priority, handler, component_id in self.subscribers[event_type]:
            async def handle_with_resilience(h: EventHandler):
                try:
                    await h(event_type, data, source)
                except Exception as e:
                    # Transmutación de error en energía
                    await self.mechanisms["horizon"].transmute_error(e, {
                        "event_type": event_type,
                        "component_id": component_id
                    })
                    self.stats["errors_transmuted"] += 1
            
            tasks.append(handle_with_resilience(handler))
        
        # Ejecutar manejadores en paralelo
        await asyncio.gather(*tasks)
    
    async def _run_background_tasks(self) -> None:
        """Ejecutar tareas en segundo plano (evolución y sincronización)."""
        while True:
            await asyncio.sleep(60)  # Intervalo de 60 segundos
            if self.running:
                # Tareas de mantenimiento y evolución
                await asyncio.gather(
                    self.mechanisms["conscious"].evolve_system(self.stats),
                    self.mechanisms["replication"].replicate_state()
                )
                
                # Regenerar y replicar componentes dañados
                await self.mechanisms["replication"].regenerate()

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del EventBus Trascendental.
        
        Returns:
            Estadísticas detalladas
        """
        # Combinar estadísticas básicas con específicas
        stats = self.stats.copy()
        stats.update({
            "running": self.running,
            "subscribers_count": sum(len(handlers) for handlers in self.subscribers.values()),
            "event_types": len(self.subscribers),
            "test_mode": self.test_mode,
            "subscribers": {k: len(v) for k, v in self.subscribers.items()}
        })
        
        # Añadir estadísticas de cada mecanismo
        for name, mechanism in self.mechanisms.items():
            if hasattr(mechanism, "get_stats"):
                stats[f"{name}_stats"] = mechanism.get_stats()
                
        # Añadir estadísticas de WebSocket y API
        stats["ws_stats"] = self.ws.get_stats()
        stats["api_stats"] = self.api.get_stats()
        
        return stats

# Ejemplo de uso para pruebas
async def test_event_bus():
    """Test básico del EventBus Trascendental."""
    bus = TranscendentalEventBus(test_mode=True)
    await bus.start()

    async def handler(event_type: str, data: Dict[str, Any], source: str):
        logger.info(f"Evento recibido: {event_type} desde {source} con datos {data}")

    await bus.subscribe("test_event", handler, component_id="test_component")
    await bus.emit("test_event", {"value": 42}, "test_source")
    await asyncio.sleep(1)  # Dar tiempo para procesar
    logger.info(f"Estadísticas: {bus.get_stats()}")
    await bus.stop()

if __name__ == "__main__":
    asyncio.run(test_event_bus())
"""