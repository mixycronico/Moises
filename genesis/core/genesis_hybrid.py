"""
Sistema Genesis híbrido con API y WebSocket.

Este módulo implementa un enfoque híbrido que combina:
- Una API interna síncrona para solicitudes directas entre componentes
- WebSockets para eventos en tiempo real
- Un coordinador central que gestiona ambos tipos de comunicación

Este enfoque elimina los deadlocks al separar claramente las operaciones
síncronas de las asíncronas y gestionarlas de forma robusta.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Set, Callable, Awaitable, Union
import websockets
from aiohttp import web, WSMsgType

logger = logging.getLogger(__name__)

class ComponentAPI:
    """
    Interfaz base para componentes en el sistema híbrido.
    
    Cada componente debe implementar al menos process_request para manejar
    solicitudes directas a través de la API.
    """
    def __init__(self, id: str):
        """
        Inicializa un componente con ID único.
        
        Args:
            id: Identificador único del componente
        """
        self.id = id
        self.events_received = []
        self.status = "created"
        self.ws_connected = False
        self.metrics = {
            "requests_processed": 0,
            "events_processed": 0,
            "errors": 0,
            "last_activity": time.time()
        }
    
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """
        Procesa una solicitud directa de otro componente o fuente externa.
        
        Este método debe ser implementado por cada componente concreto.
        
        Args:
            request_type: Tipo de solicitud
            data: Datos asociados a la solicitud
            source: ID del componente o sistema que origina la solicitud
            
        Returns:
            Resultado de la solicitud, puede ser cualquier tipo serializable
        """
        raise NotImplementedError(f"{self.id} debe implementar process_request")
    
    async def on_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Maneja un evento recibido por WebSocket.
        
        Por defecto, registra el evento recibido. Los componentes concretos
        pueden sobrescribir este método para manejar eventos específicos.
        
        Args:
            event_type: Tipo de evento
            data: Datos asociados al evento
            source: ID del componente que originó el evento
        """
        self.events_received.append((event_type, data, source))
        self.metrics["events_processed"] += 1
        self.metrics["last_activity"] = time.time()
    
    async def start(self) -> None:
        """
        Inicializa el componente. Llamado por el coordinador durante el registro.
        
        Los componentes concretos pueden sobrescribir este método para 
        realizar inicializaciones específicas.
        """
        self.status = "started"
        logger.info(f"Componente {self.id} iniciado")
    
    async def stop(self) -> None:
        """
        Detiene el componente. Llamado por el coordinador durante la parada.
        
        Los componentes concretos pueden sobrescribir este método para
        realizar limpiezas específicas.
        """
        self.status = "stopped"
        logger.info(f"Componente {self.id} detenido")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del componente.
        
        Returns:
            Diccionario con información de estado
        """
        return {
            "id": self.id,
            "status": self.status,
            "ws_connected": self.ws_connected,
            "metrics": self.metrics,
            "events_count": len(self.events_received)
        }

class GenesisHybridCoordinator:
    """
    Coordinador híbrido que gestiona tanto API como WebSocket.
    
    Este es el componente central que:
    1. Registra componentes
    2. Maneja solicitudes API directas entre componentes
    3. Gestiona conexiones WebSocket para eventos en tiempo real
    4. Monitoriza la salud del sistema
    """
    def __init__(self, host: str = "localhost", port: int = 8080, 
                 default_timeout: float = 2.0, monitor_interval: float = 10.0):
        """
        Inicializa el coordinador híbrido.
        
        Args:
            host: Host para el servidor web
            port: Puerto para el servidor web
            default_timeout: Timeout por defecto para solicitudes API (segundos)
            monitor_interval: Intervalo para monitorización de componentes (segundos)
        """
        self.components: Dict[str, ComponentAPI] = {}
        self.component_dependencies: Dict[str, Set[str]] = {}
        self.app = web.Application()
        self.host = host
        self.port = port
        self.default_timeout = default_timeout
        self.monitor_interval = monitor_interval
        self.websocket_clients: Dict[str, web.WebSocketResponse] = {}
        self.running = False
        self.stats = {
            "request_count": 0,
            "event_count": 0,
            "error_count": 0,
            "start_time": None
        }
        
        # Configurar rutas para API y WebSocket
        self.app.add_routes([
            web.get("/ws", self._websocket_handler),
            web.post("/request/{target}", self._api_request_handler),
            web.get("/status", self._status_handler),
            web.get("/health", self._health_handler),
        ])
        
        # Tarea de monitorización
        self._monitor_task = None
    
    def register_component(self, component_id: str, component: ComponentAPI, 
                          depends_on: Optional[List[str]] = None) -> None:
        """
        Registra un componente en el sistema.
        
        Args:
            component_id: ID único del componente
            component: Instancia del componente
            depends_on: Lista de IDs de componentes de los que depende (opcional)
        """
        if component_id in self.components:
            logger.warning(f"Componente {component_id} ya registrado, reemplazando")
        
        self.components[component_id] = component
        
        # Registrar dependencias para detección de ciclos
        if depends_on:
            self.component_dependencies[component_id] = set(depends_on)
            # Verificar ciclos
            if self._check_dependency_cycle(component_id):
                logger.warning(f"Ciclo de dependencias detectado para {component_id}")
        else:
            self.component_dependencies[component_id] = set()
        
        logger.info(f"Componente {component_id} registrado")
    
    def _check_dependency_cycle(self, component_id: str, visited: Optional[Set[str]] = None, 
                               path: Optional[Set[str]] = None) -> bool:
        """
        Verifica si hay ciclos de dependencias para un componente.
        
        Args:
            component_id: ID del componente a verificar
            visited: Conjunto de componentes ya visitados
            path: Camino actual en la búsqueda
            
        Returns:
            True si se detectó un ciclo, False en caso contrario
        """
        if visited is None:
            visited = set()
        if path is None:
            path = set()
        
        visited.add(component_id)
        path.add(component_id)
        
        for dep in self.component_dependencies.get(component_id, set()):
            if dep not in visited:
                if self._check_dependency_cycle(dep, visited, path):
                    return True
            elif dep in path:
                logger.error(f"Ciclo de dependencias: {component_id} -> {dep}")
                return True
        
        path.remove(component_id)
        return False
    
    async def request(self, target_id: str, request_type: str, 
                     data: Dict[str, Any], source: str, 
                     timeout: Optional[float] = None) -> Optional[Any]:
        """
        Envia una solicitud directa a un componente a través de la API.
        
        Args:
            target_id: ID del componente destino
            request_type: Tipo de solicitud
            data: Datos de la solicitud
            source: ID del componente origen
            timeout: Timeout para la solicitud (opcional)
            
        Returns:
            Resultado de la solicitud o None si hubo error
        """
        if not self.running:
            logger.warning(f"Sistema detenido, solicitud {request_type} ignorada")
            return None
        
        if target_id not in self.components:
            logger.error(f"Componente destino {target_id} no encontrado")
            return None
        
        self.stats["request_count"] += 1
        start_time = time.time()
        
        timeout_value = timeout if timeout is not None else self.default_timeout
        
        try:
            logger.debug(f"Solicitud {request_type} de {source} a {target_id}")
            result = await asyncio.wait_for(
                self.components[target_id].process_request(request_type, data, source),
                timeout=timeout_value
            )
            logger.debug(f"Solicitud {request_type} completada en {time.time() - start_time:.2f}s")
            return result
        except asyncio.TimeoutError:
            logger.error(f"Timeout en {target_id} para {request_type}")
            self.stats["error_count"] += 1
            return None
        except Exception as e:
            logger.error(f"Error en {target_id} procesando {request_type}: {e}")
            self.stats["error_count"] += 1
            return None
    
    async def broadcast_event(self, event_type: str, data: Dict[str, Any], 
                             source: str, exclude: Optional[List[str]] = None) -> None:
        """
        Emite un evento a todos los componentes conectados por WebSocket.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: ID del componente que origina el evento
            exclude: Lista de IDs de componentes a excluir (opcional)
        """
        if not self.running:
            logger.warning(f"Sistema detenido, evento {event_type} ignorado")
            return
        
        if not self.websocket_clients:
            logger.warning("No hay clientes WebSocket conectados")
            return
        
        self.stats["event_count"] += 1
        exclude_set = set(exclude) if exclude else set()
        exclude_set.add(source)  # No enviar al origen
        
        message = json.dumps({
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": time.time()
        })
        
        # Enviar a todos los clientes WebSocket activos, excepto los excluidos
        send_tasks = []
        for cid, ws in list(self.websocket_clients.items()):
            if cid not in exclude_set and not ws.closed:
                send_tasks.append(ws.send_str(message))
        
        if send_tasks:
            # Usar gather para enviar en paralelo con manejo de excepciones
            results = await asyncio.gather(*send_tasks, return_exceptions=True)
            
            # Comprobar errores
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error enviando evento a cliente: {result}")
    
    async def _monitor_components(self) -> None:
        """Monitoriza la salud de los componentes periódicamente."""
        while self.running:
            try:
                logger.debug("Ejecutando monitorización de componentes")
                now = time.time()
                
                # Verificar estado de componentes
                for comp_id, comp in list(self.components.items()):
                    status = comp.get_status()
                    inactive_time = now - status["metrics"]["last_activity"]
                    
                    # Detectar componentes sin actividad por largo tiempo
                    if inactive_time > self.monitor_interval * 3:
                        logger.warning(f"Componente {comp_id} sin actividad durante {inactive_time:.1f}s")
                
                await self.broadcast_event("system.heartbeat", {
                    "timestamp": now,
                    "active_components": len(self.components),
                    "connected_websockets": len(self.websocket_clients)
                }, "system")
                
            except Exception as e:
                logger.error(f"Error en monitorización: {e}")
                
            await asyncio.sleep(self.monitor_interval)
    
    async def _websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """
        Manejador para conexiones WebSocket de componentes.
        
        Args:
            request: Solicitud HTTP con conexión WebSocket
            
        Returns:
            Respuesta WebSocket
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # Obtener ID del componente de los query params
        component_id = request.query.get("id")
        if not component_id or component_id not in self.components:
            logger.error(f"ID de componente inválido o no registrado: {component_id}")
            await ws.close(code=1008, message=b"Componente no registrado")
            return ws
        
        # Registrar el WebSocket para este componente
        self.websocket_clients[component_id] = ws
        comp = self.components[component_id]
        comp.ws_connected = True
        logger.info(f"WebSocket conectado para {component_id}")
        
        try:
            # Notificar conexión
            await self.broadcast_event("system.component.connected", {
                "component_id": component_id
            }, "system")
            
            # Bucle principal de recepción de mensajes
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        event_type = data.get("type")
                        event_data = data.get("data", {})
                        source = data.get("source", component_id)
                        
                        if event_type:
                            # Manejo local en el componente
                            await comp.on_event(event_type, event_data, source)
                            
                            # Reenviar evento a otros componentes si tiene flag broadcast
                            if data.get("broadcast", False):
                                await self.broadcast_event(
                                    event_type, event_data, component_id
                                )
                        else:
                            logger.warning(f"Mensaje sin tipo de evento de {component_id}")
                            
                    except json.JSONDecodeError:
                        logger.error(f"Mensaje JSON inválido de {component_id}")
                    except Exception as e:
                        logger.error(f"Error procesando mensaje de {component_id}: {e}")
                        
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"Error WebSocket de {component_id}: {ws.exception()}")
        finally:
            # Limpiar conexión
            self.websocket_clients.pop(component_id, None)
            comp.ws_connected = False
            logger.info(f"WebSocket desconectado para {component_id}")
            
            # Notificar desconexión
            await self.broadcast_event("system.component.disconnected", {
                "component_id": component_id
            }, "system")
            
        return ws
    
    async def _api_request_handler(self, request: web.Request) -> web.Response:
        """
        Manejador para solicitudes API HTTP.
        
        Args:
            request: Solicitud HTTP
            
        Returns:
            Respuesta HTTP
        """
        # Obtener componente destino
        target_id = request.match_info.get("target")
        if not target_id or target_id not in self.components:
            return web.json_response({
                "error": f"Componente destino {target_id} no encontrado"
            }, status=404)
        
        try:
            # Leer payload JSON
            data = await request.json()
            request_type = data.get("type")
            request_data = data.get("data", {})
            source = data.get("source", "external")
            timeout = data.get("timeout", self.default_timeout)
            
            if not request_type:
                return web.json_response({
                    "error": "Falta campo 'type' en la solicitud"
                }, status=400)
            
            # Ejecutar solicitud
            result = await self.request(
                target_id, request_type, request_data, source, timeout
            )
            
            # Devolver resultado
            return web.json_response({
                "success": result is not None,
                "result": result
            })
            
        except json.JSONDecodeError:
            return web.json_response({
                "error": "Formato JSON inválido"
            }, status=400)
        except Exception as e:
            logger.error(f"Error en API request handler: {e}")
            return web.json_response({
                "error": str(e)
            }, status=500)
    
    async def _status_handler(self, request: web.Request) -> web.Response:
        """Manejador para endpoint de estado del sistema."""
        uptime = time.time() - (self.stats["start_time"] or time.time())
        
        # Recolectar estado de componentes
        component_status = {}
        for comp_id, comp in self.components.items():
            component_status[comp_id] = comp.get_status()
        
        status_data = {
            "system": {
                "running": self.running,
                "uptime": uptime,
                "stats": self.stats,
                "active_websockets": len(self.websocket_clients)
            },
            "components": component_status
        }
        
        return web.json_response(status_data)
    
    async def _health_handler(self, request: web.Request) -> web.Response:
        """Manejador para endpoint de salud del sistema."""
        return web.json_response({
            "status": "ok" if self.running else "stopped",
            "components": len(self.components),
            "connected": len(self.websocket_clients)
        })
    
    async def start(self) -> None:
        """Inicia el coordinador híbrido."""
        if self.running:
            logger.warning("El coordinador ya está en ejecución")
            return
        
        logger.info(f"Iniciando coordinador híbrido en {self.host}:{self.port}")
        self.running = True
        self.stats["start_time"] = time.time()
        
        # Iniciar componentes
        for comp_id, comp in self.components.items():
            try:
                await comp.start()
            except Exception as e:
                logger.error(f"Error iniciando componente {comp_id}: {e}")
        
        # Iniciar tarea de monitorización
        self._monitor_task = asyncio.create_task(self._monitor_components())
        
        # Iniciar servidor web
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        logger.info(f"Coordinador híbrido iniciado y escuchando en {self.host}:{self.port}")
    
    async def stop(self) -> None:
        """Detiene el coordinador híbrido."""
        if not self.running:
            logger.warning("El coordinador ya está detenido")
            return
        
        logger.info("Deteniendo coordinador híbrido")
        self.running = False
        
        # Cancelar tarea de monitorización
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Notificar parada
        await self.broadcast_event("system.shutdown", {
            "timestamp": time.time()
        }, "system")
        
        # Cerrar conexiones WebSocket
        for comp_id, ws in list(self.websocket_clients.items()):
            if not ws.closed:
                await ws.close(code=1000, message="Sistema detenido")
        
        # Detener componentes
        for comp_id, comp in self.components.items():
            try:
                await comp.stop()
            except Exception as e:
                logger.error(f"Error deteniendo componente {comp_id}: {e}")
        
        # Cerrar servidor web
        await self.app.shutdown()
        await self.app.cleanup()
        
        logger.info(f"Coordinador híbrido detenido. Estadísticas finales: " +
                  f"Solicitudes: {self.stats['request_count']}, " +
                  f"Eventos: {self.stats['event_count']}")