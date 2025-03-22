"""
Sistema Genesis híbrido optimizado con API, WebSocket local y externo.

Esta versión optimizada del sistema híbrido combina:
- API para solicitudes síncronas directas
- WebSocket local (en memoria) para eventos internos entre componentes
- WebSocket externo (red) para eventos con sistemas externos

La arquitectura elimina deadlocks al separar claramente los diferentes
tipos de comunicación y sus mecanismos.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Set
from aiohttp import web, WSMsgType
import json

logger = logging.getLogger(__name__)

class ComponentAPI:
    """
    Interfaz base para componentes en el sistema híbrido optimizado.
    
    Cada componente debe implementar al menos process_request para la API
    y puede sobrescribir on_local_event y on_external_event para
    manejar eventos de WebSocket local y externo.
    """
    def __init__(self, id: str):
        """
        Inicializa un componente.
        
        Args:
            id: Identificador único del componente
        """
        self.id = id
        self.local_events = []  # Eventos recibidos localmente
        self.external_events = []  # Eventos recibidos externamente
        self.local_queue = asyncio.Queue(maxsize=50)  # Límite para evitar acumulación
        self.metrics = {
            "requests_processed": 0,
            "local_events_processed": 0,
            "external_events_processed": 0,
            "errors": 0,
            "last_activity": time.time()
        }

    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """
        Procesa una solicitud directa a través de la API.
        
        Args:
            request_type: Tipo de solicitud
            data: Datos asociados a la solicitud
            source: ID del componente o sistema que origina la solicitud
            
        Returns:
            Resultado de la solicitud (cualquier tipo serializable)
        """
        raise NotImplementedError(f"{self.id} debe implementar process_request")

    async def on_local_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Maneja un evento recibido del WebSocket local (en memoria).
        
        Args:
            event_type: Tipo de evento
            data: Datos asociados al evento
            source: ID del componente que originó el evento
        """
        self.local_events.append((event_type, data, source))
        self.metrics["local_events_processed"] += 1
        self.metrics["last_activity"] = time.time()

    async def on_external_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Maneja un evento recibido del WebSocket externo (red).
        
        Args:
            event_type: Tipo de evento
            data: Datos asociados al evento
            source: ID del componente que originó el evento
        """
        self.external_events.append((event_type, data, source))
        self.metrics["external_events_processed"] += 1
        self.metrics["last_activity"] = time.time()

    async def listen_local(self):
        """
        Escucha eventos del WebSocket local (en memoria).
        
        Este método se ejecuta en una tarea separada para cada componente
        y procesa los eventos de la cola local.
        """
        while True:
            try:
                # Usar timeout para permitir una terminación limpia
                event_type, data, source = await asyncio.wait_for(
                    self.local_queue.get(), 
                    timeout=2.0
                )
                
                try:
                    # Procesar el evento
                    await self.on_local_event(event_type, data, source)
                except Exception as e:
                    logger.error(f"Error en {self.id} procesando evento local {event_type}: {e}")
                    self.metrics["errors"] += 1
                finally:
                    # Siempre marcar la tarea como completada
                    self.local_queue.task_done()
                    
            except asyncio.TimeoutError:
                # Timeout normal, continuar esperando
                continue
            except asyncio.CancelledError:
                # Tarea cancelada, salir del bucle
                logger.debug(f"Escucha local de {self.id} cancelada")
                break
            except Exception as e:
                # Error inesperado
                logger.error(f"Error en escucha local de {self.id}: {e}")
                self.metrics["errors"] += 1
                # Pequeña pausa para evitar ciclos de error
                await asyncio.sleep(0.1)

class GenesisHybridCoordinator:
    """
    Coordinador del sistema híbrido optimizado.
    
    Este es el componente central que:
    1. Registra componentes
    2. Maneja solicitudes API directas entre componentes
    3. Distribuye eventos por WebSocket local (en memoria)
    4. Gestiona conexiones WebSocket externas (red)
    """
    def __init__(self, host: str = "localhost", port: int = 8080):
        """
        Inicializa el coordinador híbrido.
        
        Args:
            host: Host para el servidor web externo
            port: Puerto para el servidor web externo
        """
        self.components: Dict[str, ComponentAPI] = {}
        self.app = web.Application()
        self.host = host
        self.port = port
        self.websocket_clients: Dict[str, web.WebSocketResponse] = {}  # WebSocket externo
        self.running = False
        self.request_count = 0
        self.local_event_count = 0
        self.external_event_count = 0
        self.start_time = None
        self.listen_tasks = {}  # Tareas de escucha para cada componente
        
        # Configurar rutas para WebSocket externo y API
        self.app.add_routes([
            web.get("/ws", self._external_websocket_handler),
            web.post("/request/{target}", self._api_request_handler),
            web.get("/status", self._status_handler),
        ])

    def register_component(self, component_id: str, component: ComponentAPI) -> None:
        """
        Registra un componente en el sistema.
        
        Args:
            component_id: ID único del componente
            component: Instancia del componente
        """
        if component_id in self.components:
            logger.warning(f"Componente {component_id} ya registrado, reemplazando")
            
            # Cancelar tarea de escucha anterior si existe
            task = self.listen_tasks.pop(component_id, None)
            if task:
                task.cancel()
                
        # Registrar componente
        self.components[component_id] = component
        
        # Iniciar escucha local (en memoria)
        listen_task = asyncio.create_task(component.listen_local())
        self.listen_tasks[component_id] = listen_task
        
        logger.debug(f"Componente {component_id} registrado")

    async def request(self, target_id: str, request_type: str, data: Dict[str, Any], 
                     source: str, timeout: float = 2.0) -> Optional[Any]:
        """
        Envía una solicitud directa a un componente a través de la API.
        
        Args:
            target_id: ID del componente destino
            request_type: Tipo de solicitud
            data: Datos de la solicitud
            source: ID del componente origen
            timeout: Tiempo máximo de espera (segundos)
            
        Returns:
            Resultado de la solicitud o None si hubo error
        """
        if target_id not in self.components:
            logger.error(f"Componente {target_id} no encontrado")
            return None
            
        try:
            self.request_count += 1
            start_time = time.time()
            
            # Realizar solicitud con timeout
            result = await asyncio.wait_for(
                self.components[target_id].process_request(request_type, data, source),
                timeout=timeout
            )
            
            # Registrar tiempo de procesamiento
            processing_time = time.time() - start_time
            if processing_time > timeout / 2:
                # Advertencia para solicitudes que toman más de la mitad del timeout
                logger.warning(f"Solicitud lenta: {request_type} a {target_id} tomó {processing_time:.2f}s")
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout en {target_id} para {request_type}")
            return None
        except Exception as e:
            logger.error(f"Error en solicitud a {target_id}: {e}")
            return None

    async def emit_local(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Emite un evento al WebSocket local (en memoria).
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: ID del componente origen
        """
        if not self.running:
            logger.warning(f"Sistema detenido, evento local {event_type} ignorado")
            return
            
        self.local_event_count += 1
        
        # Agregar timestamp si no existe
        if "timestamp" not in data:
            data["timestamp"] = time.time()
            
        # Crear tareas para enviar a todos los componentes excepto el origen
        tasks = [
            component.local_queue.put((event_type, data, source))
            for cid, component in self.components.items()
            if cid != source  # No enviar al origen
        ]
        
        # Ejecutar todas las tareas en paralelo
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
        logger.debug(f"Evento local {event_type} emitido desde {source}")

    async def emit_external(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Emite un evento al WebSocket externo (red).
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: ID del componente origen
        """
        if not self.websocket_clients:
            logger.debug("No hay clientes WebSocket externos conectados")
            return
            
        # Formato del mensaje
        message = json.dumps({
            "type": event_type, 
            "data": data, 
            "source": source,
            "timestamp": time.time()
        })
        
        self.external_event_count += 1
        
        # Crear tareas para enviar a todos los clientes excepto el origen
        tasks = [
            ws.send_str(message)
            for cid, ws in self.websocket_clients.items()
            if cid != source and not ws.closed  # No enviar al origen y solo a conexiones activas
        ]
        
        # Ejecutar todas las tareas en paralelo
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verificar errores
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error enviando evento externo: {result}")
                    
        logger.debug(f"Evento externo {event_type} emitido desde {source}")

    async def _external_websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """
        Manejador para conexiones WebSocket externas.
        
        Args:
            request: Solicitud HTTP con conexión WebSocket
            
        Returns:
            Respuesta WebSocket
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # Obtener ID del componente o cliente
        component_id = request.query.get("id")
        if not component_id:
            await ws.close(code=1008, message=b"ID no especificado")
            return ws
            
        # Para clientes externos, verificar si el componente existe o es un cliente externo
        if component_id not in self.components and not component_id.startswith("external_"):
            await ws.close(code=1008, message=b"Componente no registrado")
            return ws
        
        # Registrar cliente WebSocket
        self.websocket_clients[component_id] = ws
        logger.info(f"WebSocket externo conectado para {component_id}")
        
        try:
            # Bucle principal para recibir mensajes
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        # Procesar mensaje JSON
                        data = json.loads(msg.data)
                        event_type = data.get("type")
                        event_data = data.get("data", {})
                        source = data.get("source", component_id)
                        
                        if component_id in self.components:
                            # Si es un componente registrado, procesar evento
                            await self.components[component_id].on_external_event(
                                event_type, event_data, source
                            )
                        else:
                            # Si es un cliente externo, emitir el evento localmente
                            await self.emit_local(
                                event_type,
                                {"external_data": event_data, "external_source": source},
                                component_id
                            )
                            
                    except json.JSONDecodeError:
                        logger.error(f"Mensaje WebSocket inválido desde {component_id}")
                    except Exception as e:
                        logger.error(f"Error procesando mensaje WebSocket de {component_id}: {e}")
                        
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"Error WebSocket de {component_id}: {ws.exception()}")
                    
        finally:
            # Limpiar conexión
            self.websocket_clients.pop(component_id, None)
            logger.info(f"WebSocket externo desconectado para {component_id}")
            
        return ws

    async def _api_request_handler(self, request: web.Request) -> web.Response:
        """
        Manejador para solicitudes API externas.
        
        Args:
            request: Solicitud HTTP
            
        Returns:
            Respuesta HTTP
        """
        # Obtener componente destino
        target_id = request.match_info["target"]
        if target_id not in self.components:
            return web.Response(
                status=404, 
                text=f"Componente {target_id} no encontrado"
            )
            
        try:
            # Leer y procesar solicitud
            data = await request.json()
            timeout = data.get("timeout", 2.0)  # Timeout personalizable
            
            result = await self.request(
                target_id, 
                data.get("type"), 
                data.get("data", {}), 
                data.get("source", "external"),
                timeout
            )
            
            # Devolver resultado como JSON
            return web.json_response({"result": result})
            
        except json.JSONDecodeError:
            return web.Response(status=400, text="Formato JSON inválido")
        except Exception as e:
            logger.error(f"Error en API request handler: {e}")
            return web.Response(status=500, text=str(e))
    
    async def _status_handler(self, request: web.Request) -> web.Response:
        """
        Manejador para obtener estado del sistema.
        
        Args:
            request: Solicitud HTTP
            
        Returns:
            Respuesta HTTP con estado del sistema
        """
        uptime = time.time() - (self.start_time or time.time())
        
        # Recolectar estado de componentes
        component_status = {}
        for comp_id, comp in self.components.items():
            component_status[comp_id] = {
                "events_received_local": len(comp.local_events),
                "events_received_external": len(comp.external_events),
                "metrics": comp.metrics
            }
        
        status = {
            "uptime": uptime,
            "running": self.running,
            "request_count": self.request_count,
            "local_event_count": self.local_event_count,
            "external_event_count": self.external_event_count,
            "active_websockets": len(self.websocket_clients),
            "components": component_status
        }
        
        return web.json_response(status)

    async def start(self) -> None:
        """
        Inicia el coordinador híbrido.
        
        Esta función inicia el servidor web para WebSocket externo
        y marca el sistema como en ejecución.
        """
        if self.running:
            logger.warning("El coordinador ya está en ejecución")
            return
            
        self.running = True
        self.start_time = time.time()
        
        # Iniciar servidor web
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        logger.info(f"Coordinador híbrido iniciado en {self.host}:{self.port}")
        
        # Mantener el servidor corriendo
        while self.running:
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
        
        # Cuando se detiene el bucle, limpiar recursos
        await runner.cleanup()

    async def stop(self) -> None:
        """
        Detiene el coordinador híbrido.
        
        Esta función cierra todas las conexiones y detiene el servidor web.
        """
        if not self.running:
            logger.warning("El coordinador ya está detenido")
            return
            
        self.running = False
        
        # Cerrar todas las conexiones WebSocket externas
        for component_id, ws in list(self.websocket_clients.items()):
            if not ws.closed:
                await ws.close(code=1000, message=b"Sistema detenido")
        
        # Cancelar tareas de escucha local
        for comp_id, task in list(self.listen_tasks.items()):
            if not task.done():
                task.cancel()
        
        # Limpiar recursos
        await self.app.shutdown()
        await self.app.cleanup()
        
        # Estadísticas de ejecución
        uptime = time.time() - (self.start_time or time.time())
        logger.info(
            f"Coordinador híbrido detenido. "
            f"Uptime: {uptime:.1f}s, "
            f"Solicitudes: {self.request_count}, "
            f"Eventos locales: {self.local_event_count}, "
            f"Eventos externos: {self.external_event_count}"
        )