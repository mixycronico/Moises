"""
Sistema Genesis híbrido optimizado con características avanzadas de resiliencia:
- Reintentos adaptativos con backoff exponencial y jitter
- Circuit Breaker por componente
- Checkpointing y Safe Mode
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
import json
from time import time
from random import uniform

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # Tipo ficticio para que no falle la importación

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Implementación del patrón Circuit Breaker."""
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 10.0):
        self.state = "CLOSED"
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = 0

    async def call(self, coro):
        """Ejecutar una coroutine con Circuit Breaker."""
        if self.state == "OPEN":
            if time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit Breaker abierto")
        
        try:
            result = await coro
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                self.last_failure_time = time()
            raise e

class ComponentAPI:
    """
    Componente del sistema Genesis con capacidades de resiliencia.
    Esta clase implementa la interfaz API + WebSocket con características
    de resiliencia integradas.
    """
    def __init__(self, id: str):
        self.id = id
        self.local_events = []
        self.external_events = []
        self.local_queue = asyncio.Queue(maxsize=50)
        self.last_active = time()
        self.failed = False
        self.checkpoint = {}  # Estado persistido
        self.circuit_breaker = CircuitBreaker()

    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """
        Procesar una solicitud API.
        
        Args:
            request_type: Tipo de solicitud
            data: Datos de la solicitud
            source: Origen de la solicitud
            
        Returns:
            Resultado de la solicitud
            
        Raises:
            NotImplementedError: Debe ser implementado por subclases
        """
        self.last_active = time()
        raise NotImplementedError

    async def on_local_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar evento local del WebSocket interno.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Origen del evento
        """
        self.last_active = time()
        self.local_events.append((event_type, data, source))

    async def on_external_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar evento externo del WebSocket.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Origen del evento
        """
        self.last_active = time()
        self.external_events.append((event_type, data, source))

    async def listen_local(self):
        """Escuchar eventos en la cola local con mecanismos de resiliencia."""
        while True:
            try:
                event_type, data, source = await asyncio.wait_for(self.local_queue.get(), timeout=2.0)
                if not self.failed:
                    await self.on_local_event(event_type, data, source)
                self.local_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Fallo en {self.id} al procesar evento local: {e}")
                self.failed = True
                await asyncio.sleep(1)
                self.failed = False

    def save_checkpoint(self):
        """Guardar estado crítico para recuperación."""
        self.checkpoint = {
            "local_events": self.local_events[-10:],  # Últimos 10 eventos
            "external_events": self.external_events[-10:],
            "last_active": self.last_active
        }
        logger.debug(f"Checkpoint guardado para {self.id}")

    async def restore_from_checkpoint(self):
        """Restaurar desde checkpoint después de un fallo."""
        if self.checkpoint:
            self.local_events = self.checkpoint.get("local_events", [])
            self.external_events = self.checkpoint.get("external_events", [])
            self.last_active = self.checkpoint.get("last_active", time())
            logger.info(f"{self.id} restaurado desde checkpoint")

class GenesisHybridCoordinator:
    """
    Coordinador del sistema híbrido Genesis con características avanzadas de resiliencia.
    
    Este coordinador implementa:
    - API REST para solicitudes síncronas
    - WebSockets para eventos asíncronos
    - Reintentos adaptativos con backoff exponencial y jitter
    - Circuit Breaker por componente
    - Checkpointing y Safe Mode
    """
    def __init__(self, host: str = "localhost", port: int = 8080, max_ws_connections: int = 100):
        self.components: Dict[str, ComponentAPI] = {}
        self.host = host
        self.port = port
        self.websocket_clients: Dict[str, web.WebSocketResponse] = {} if AIOHTTP_AVAILABLE else {}
        self.running = False
        self.mode = "NORMAL"  # NORMAL, SAFE, EMERGENCY
        self.request_count = 0
        self.local_event_count = 0
        self.external_event_count = 0
        self.max_ws_connections = max_ws_connections
        self.app = None
        
        if AIOHTTP_AVAILABLE:
            self.app = web.Application()
            self.app.add_routes([
                web.get("/ws", self._external_websocket_handler),
                web.post("/request/{target}", self._api_request_handler),
            ])
        
        self._monitor_task = None

    def register_component(self, component_id: str, component: ComponentAPI) -> None:
        """
        Registrar un componente en el coordinador.
        
        Args:
            component_id: ID único del componente
            component: Instancia del componente
        """
        if component_id in self.components:
            logger.warning(f"Componente {component_id} ya registrado, reemplazando")
        self.components[component_id] = component
        
        if self.running:
            asyncio.create_task(component.listen_local())
        
        logger.debug(f"Componente {component_id} registrado")

    async def _retry_with_backoff(self, coro, max_retries: int = 5, base_delay: float = 0.1):
        """
        Reintentos adaptativos con backoff exponencial y jitter.
        
        Args:
            coro: Coroutine a ejecutar
            max_retries: Número máximo de reintentos
            base_delay: Retraso base en segundos
            
        Returns:
            Resultado de la coroutine
            
        Raises:
            Exception: Si todos los reintentos fallan
        """
        attempt = 0
        while attempt < max_retries:
            try:
                return await coro
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                # Fórmula de backoff exponencial con jitter
                delay = base_delay * (2 ** attempt) + uniform(0, 0.1 * (2 ** attempt))
                logger.warning(f"Reintento {attempt + 1}/{max_retries} tras fallo: {e}. Espera: {delay:.2f}s")
                await asyncio.sleep(delay)
                attempt += 1

    async def request(self, target_id: str, request_type: str, data: Dict[str, Any], source: str, timeout: float = 5.0) -> Optional[Any]:
        """
        Realizar una solicitud API a un componente con todas las características de resiliencia.
        
        Args:
            target_id: ID del componente objetivo
            request_type: Tipo de solicitud
            data: Datos de la solicitud
            source: Origen de la solicitud
            timeout: Tiempo límite para la solicitud
            
        Returns:
            Resultado de la solicitud o None si falla
        """
        if target_id not in self.components or self.components[target_id].failed:
            return None
        
        # En modo de emergencia, solo permitir operaciones críticas
        if self.mode == "EMERGENCY" and request_type not in ["ping", "status", "health", "emergency_action"]:
            return None
        
        # En modo seguro, limitar tipos de operaciones para componentes no esenciales
        if self.mode == "SAFE" and not self._is_essential(target_id):
            if not (request_type.startswith("get") or request_type.startswith("read") or 
                    request_type in ["ping", "status", "health"]):
                logger.warning(f"Operación {request_type} rechazada en modo SAFE para componente no esencial {target_id}")
                return None
        
        async def call():
            return await asyncio.wait_for(
                self.components[target_id].process_request(request_type, data, source),
                timeout=timeout
            )

        try:
            self.request_count += 1
            return await self.components[target_id].circuit_breaker.call(
                self._retry_with_backoff(call)
            )
        except Exception as e:
            logger.error(f"Error en solicitud a {target_id}: {e}")
            self.components[target_id].failed = True
            return None

    async def emit_local(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Emitir evento local a todos los componentes.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Origen del evento
        """
        if not self.running or self.mode == "EMERGENCY":
            return
        
        self.local_event_count += 1
        tasks = []
        
        for cid, component in self.components.items():
            if cid != source and not component.failed:
                if component.local_queue.qsize() < 40:  # Prevenir overflow
                    tasks.append(component.local_queue.put((event_type, data, source)))
                else:
                    logger.warning(f"Cola local de {cid} llena")
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def emit_external(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Emitir evento externo a todos los clientes WebSocket.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Origen del evento
        """
        if not self.websocket_clients or self.mode == "EMERGENCY" or not AIOHTTP_AVAILABLE:
            return
        
        message = json.dumps({"type": event_type, "data": data, "source": source})
        self.external_event_count += 1
        
        tasks = [
            ws.send_str(message)
            for cid, ws in self.websocket_clients.items()
            if cid != source and not ws.closed
        ]
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _external_websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """
        Manejador de conexiones WebSocket externas.
        
        Args:
            request: Solicitud HTTP
            
        Returns:
            Respuesta WebSocket
        """
        if not AIOHTTP_AVAILABLE:
            return None
            
        if len(self.websocket_clients) >= self.max_ws_connections:
            return web.Response(status=503, text="Límite de conexiones alcanzado")
            
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        component_id = request.query.get("id")
        if component_id not in self.components:
            await ws.close(code=1008, message=b"Componente no registrado")
            return ws

        self.websocket_clients[component_id] = ws
        logger.info(f"WebSocket externo conectado para {component_id}")

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT and self.mode != "EMERGENCY":
                    data = json.loads(msg.data)
                    if not self.components[component_id].failed:
                        await self.components[component_id].on_external_event(
                            data.get("type"), data.get("data", {}), data.get("source", component_id)
                        )
        finally:
            self.websocket_clients.pop(component_id, None)
            logger.info(f"WebSocket externo desconectado para {component_id}")
        
        return ws

    async def _api_request_handler(self, request: web.Request) -> web.Response:
        """
        Manejador de solicitudes API REST.
        
        Args:
            request: Solicitud HTTP
            
        Returns:
            Respuesta HTTP
        """
        if not AIOHTTP_AVAILABLE:
            return None
            
        target_id = request.match_info["target"]
        
        if target_id not in self.components:
            return web.Response(status=404, text=f"Componente {target_id} no encontrado")
            
        try:
            data = await request.json()
        except Exception:
            return web.Response(status=400, text="JSON inválido")
            
        result = await self.request(
            target_id, data.get("type"), data.get("data", {}), data.get("source", "external")
        )
        
        return web.json_response({"result": result})

    def _is_essential(self, component_id: str) -> bool:
        """
        Determinar si un componente es esencial.
        Esta implementación se puede extender para definir componentes esenciales.
        
        Args:
            component_id: ID del componente
            
        Returns:
            True si el componente es esencial
        """
        # Esta es una implementación básica
        # En una implementación real, se podría establecer esto en la configuración
        essential_components = ["exchange", "risk_manager", "wallet", "order_manager"]
        return component_id in essential_components

    async def _monitor_and_checkpoint(self):
        """
        Monitoreo continuo, checkpointing y gestión de modo.
        Esta tarea se ejecuta periódicamente para:
        - Realizar checkpoints de los componentes
        - Restaurar componentes fallidos
        - Ajustar el modo del sistema según su estado
        """
        while self.running:
            try:
                # Contar componentes fallidos para determinar el modo
                failed_count = sum(1 for c in self.components.values() if c.failed)
                total = len(self.components)
                
                if total > 0:
                    failure_rate = failed_count / total
                    
                    # Ajustar modo según tasa de fallos
                    if failure_rate > 0.5:  # Más del 50% de componentes fallidos
                        if self.mode != "EMERGENCY":
                            logger.critical(f"Activando modo EMERGENCY. {failed_count}/{total} componentes fallidos")
                            self.mode = "EMERGENCY"
                    elif failure_rate > 0.2:  # Más del 20% de componentes fallidos
                        if self.mode != "SAFE" and self.mode != "EMERGENCY":
                            logger.warning(f"Activando modo SAFE. {failed_count}/{total} componentes fallidos")
                            self.mode = "SAFE"
                    else:
                        if self.mode != "NORMAL":
                            logger.info(f"Volviendo a modo NORMAL. {failed_count}/{total} componentes fallidos")
                            self.mode = "NORMAL"
                
                # Realizar checkpointing y recuperación
                for cid, component in self.components.items():
                    # Guardar checkpoint
                    component.save_checkpoint()
                    
                    # Intentar recuperar componentes fallidos
                    if component.failed:
                        logger.info(f"Recuperando componente {cid}")
                        await component.restore_from_checkpoint()
                        component.failed = False
                        asyncio.create_task(component.listen_local())
                    
                    # Verificar inactividad
                    elif time() - component.last_active > 5:
                        logger.warning(f"Componente {cid} inactivo demasiado tiempo")
                        if self._is_essential(cid):
                            logger.error(f"Componente esencial {cid} inactivo")
                            if self.mode == "NORMAL":
                                self.mode = "SAFE"
                    
                    # Verificar sobrecarga de cola
                    elif component.local_queue.qsize() > 40:
                        logger.warning(f"Cola local de {cid} cerca del límite")
            
            except Exception as e:
                logger.error(f"Error en monitor: {e}")
            
            # ~150ms para checkpointing frecuente
            await asyncio.sleep(0.15)

    async def start(self) -> None:
        """Iniciar el coordinador y todos los componentes."""
        if self.running:
            return
            
        self.running = True
        
        # Iniciar tareas de monitoreo y checkpointing
        self._monitor_task = asyncio.create_task(self._monitor_and_checkpoint())
        
        # Iniciar escucha de eventos para todos los componentes
        for component in self.components.values():
            asyncio.create_task(component.listen_local())
        
        # Iniciar servidor web si aiohttp está disponible
        if AIOHTTP_AVAILABLE and self.app:
            runner = web.AppRunner(self.app)
            await runner.setup()
            site = web.TCPSite(runner, self.host, self.port)
            await site.start()
            logger.info(f"Coordinador iniciado en {self.host}:{self.port}")
            
            # En pruebas, podemos querer que esto retorne en lugar de esperar indefinidamente
            return
            # Para uso en producción, descomentar:
            # await asyncio.Future()
        else:
            logger.info("Coordinador iniciado en modo headless (sin servidor web)")

    async def stop(self) -> None:
        """Detener el coordinador y todos los componentes."""
        if not self.running:
            return
            
        self.running = False
        
        # Cancelar tarea de monitoreo
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Cerrar conexiones WebSocket
        if AIOHTTP_AVAILABLE:
            for ws in self.websocket_clients.values():
                await ws.close()
            
            # Apagar servidor web
            if self.app:
                await self.app.shutdown()
                await self.app.cleanup()
        
        logger.info(f"Coordinador detenido. Estadísticas: Solicitudes: {self.request_count}, " +
                  f"Eventos locales: {self.local_event_count}, Eventos externos: {self.external_event_count}")

# Ejemplo de componente resiliente
class ResilientComponent(ComponentAPI):
    """
    Componente resiliente de ejemplo.
    
    Implementa todas las capacidades de resiliencia del sistema Genesis:
    - Circuit Breaker incorporado
    - Checkpointing automático
    - Manejo de fallos con recuperación
    """
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """Procesar solicitud API con resiliencia."""
        if request_type == "ping":
            await asyncio.sleep(0.1)  # Simular trabajo
            return f"Pong desde {self.id}"
        elif request_type == "echo":
            return data.get("message", "No message")
        elif request_type == "status":
            return {
                "id": self.id,
                "healthy": not self.failed,
                "events_processed": len(self.local_events) + len(self.external_events),
                "circuit_state": self.circuit_breaker.state
            }
        elif request_type == "simulate_failure":
            self.failed = True
            raise Exception("Fallo simulado")
        elif request_type == "restore":
            await self.restore_from_checkpoint()
            return {"restored": True, "events": len(self.local_events)}
        
        return None