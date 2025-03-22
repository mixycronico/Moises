"""
Sistema Genesis híbrido optimizado con API, WebSocket local y WebSocket externo.

Esta implementación integra las mejoras de resiliencia:
1. Sistema de Reintentos Adaptativo
2. Arquitectura de Circuit Breaker  
3. Sistema de Checkpointing y Safe Mode

Proporciona un sistema robusto con recuperación automática, detección proactiva
y alto rendimiento bajo carga.
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, Optional, List, Set, Tuple, Callable, Coroutine
from aiohttp import web
import random

# Importar nuestros módulos de resiliencia
from genesis.core.retry_adaptive import with_retry, retry_operation, RetryConfig
from genesis.core.circuit_breaker import with_circuit_breaker, CircuitState, registry
from genesis.core.checkpoint_recovery import (
    CheckpointManager, CheckpointType, RecoveryMode,
    SafeModeManager, RecoveryManager
)

# Configuración de logging
logger = logging.getLogger("genesis.hybrid")

class ComponentAPI:
    """
    Componente básico para el sistema híbrido.
    
    Cada componente puede:
    1. Procesar solicitudes API síncronas
    2. Manejar eventos locales asíncronos
    3. Manejar eventos externos asíncronos
    """
    
    def __init__(self, id: str):
        """
        Inicializar componente.
        
        Args:
            id: Identificador único del componente
        """
        self.id = id
        self.local_events = []
        self.external_events = []
        self.local_queue = asyncio.Queue(maxsize=100)
        self.last_active = time.time()
        self.healthy = True
        self.state: Dict[str, Any] = {"id": id, "started": False}
        
        # Variables para checkpointing
        self.checkpoint_dir = None
        self.checkpoint_mgr = None
        self.last_checkpoint_time = 0
    
    def setup_checkpointing(self, 
                           checkpoint_dir: str, 
                           interval_ms: float = 150.0,
                           checkpoint_type: CheckpointType = CheckpointType.DISK) -> None:
        """
        Configurar sistema de checkpointing.
        
        Args:
            checkpoint_dir: Directorio para almacenar checkpoints
            interval_ms: Intervalo de checkpointing en milisegundos
            checkpoint_type: Tipo de checkpoint (MEMORY, DISK, DISTRIBUTED)
        """
        self.checkpoint_dir = os.path.join(checkpoint_dir, self.id)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.checkpoint_mgr = CheckpointManager(
            component_id=self.id,
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_interval=interval_ms,
            max_checkpoints=5,
            checkpoint_type=checkpoint_type
        )
    
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """
        Procesar solicitud API síncrona.
        
        Args:
            request_type: Tipo de solicitud
            data: Datos de la solicitud
            source: Fuente de la solicitud
            
        Returns:
            Resultado de la solicitud
            
        Raises:
            NotImplementedError: Debe ser implementado por subclases
        """
        self.last_active = time.time()
        raise NotImplementedError("Las subclases deben implementar process_request")
    
    async def on_local_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar evento local asíncrono.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
        """
        self.last_active = time.time()
        self.local_events.append((event_type, data, source))
        # Las subclases pueden sobrescribir para comportamiento específico
    
    async def on_external_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar evento externo asíncrono.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
        """
        self.last_active = time.time()
        self.external_events.append((event_type, data, source))
        # Las subclases pueden sobrescribir para comportamiento específico
    
    @with_retry(base_delay=0.1, max_retries=3, jitter_factor=0.2)
    async def listen_local(self) -> None:
        """
        Escuchar eventos locales de la cola.
        Esta función se ejecuta continuamente mientras el componente esté activo.
        """
        while True:
            try:
                # Esperar evento con timeout para permitir cancelación
                event_type, data, source = await asyncio.wait_for(
                    self.local_queue.get(), 
                    timeout=2.0
                )
                
                if self.healthy:
                    await self.on_local_event(event_type, data, source)
                
                self.local_queue.task_done()
                
                # Crear checkpoint si es necesario
                if (self.checkpoint_mgr and 
                    time.time() - self.last_checkpoint_time > 5.0):  # Cada 5 segundos
                    await self.checkpoint_mgr.checkpoint(self.state)
                    self.last_checkpoint_time = time.time()
                
            except asyncio.TimeoutError:
                # Timeout esperado, continuar
                continue
                
            except asyncio.CancelledError:
                # Cancelación solicitada, salir
                logger.info(f"Listener local de {self.id} cancelado")
                break
                
            except Exception as e:
                # Error inesperado
                logger.error(f"Error en {self.id} al procesar evento local: {e}")
                self.healthy = False
                
                # Esperar antes de reintentar
                await asyncio.sleep(1)
                self.healthy = True
    
    async def start(self) -> None:
        """Iniciar el componente."""
        self.state["started"] = True
        
        # Iniciar checkpointing automático si está configurado
        if self.checkpoint_mgr:
            await self.checkpoint_mgr.start_automatic_checkpointing(lambda: self.state)
    
    async def stop(self) -> None:
        """Detener el componente."""
        self.state["started"] = False
        
        # Detener checkpointing automático si está configurado
        if self.checkpoint_mgr:
            await self.checkpoint_mgr.stop_automatic_checkpointing()

class GenesisHybridCoordinator:
    """
    Coordinador central del sistema híbrido optimizado.
    
    Maneja:
    1. Registro de componentes
    2. Comunicación API directa entre componentes
    3. Eventos locales asíncronos
    4. Eventos externos vía WebSockets
    5. Detección proactiva y recuperación
    6. Checkpointing y modo seguro
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        max_ws_connections: int = 100,
        checkpoint_dir: str = "./checkpoints",
        essential_components: List[str] = None
    ):
        """
        Inicializar coordinador.
        
        Args:
            host: Host para el servidor HTTP
            port: Puerto para el servidor HTTP
            max_ws_connections: Máximo de conexiones WebSocket simultáneas
            checkpoint_dir: Directorio para almacenar checkpoints
            essential_components: Componentes esenciales para modo seguro
        """
        # Componentes registrados
        self.components: Dict[str, ComponentAPI] = {}
        
        # Servidor HTTP/WebSocket
        self.app = web.Application()
        self.host = host
        self.port = port
        self.websocket_clients: Dict[str, web.WebSocketResponse] = {}
        
        # Estado
        self.running = False
        self.request_count = 0
        self.local_event_count = 0
        self.external_event_count = 0
        self.max_ws_connections = max_ws_connections
        
        # Configurar rutas HTTP
        self.app.add_routes([
            web.get("/ws", self._external_websocket_handler),
            web.post("/request/{target}", self._api_request_handler),
            web.get("/health", self._health_check_handler),
            web.get("/status", self._status_handler),
        ])
        
        # Configurar checkpointing y recuperación
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Recovery Manager y Safe Mode
        self.recovery_mgr = RecoveryManager(
            checkpoint_dir=checkpoint_dir,
            essential_components=essential_components or []
        )
        
        self.safe_mode_mgr = self.recovery_mgr.safe_mode_manager
        
        # Iniciar monitor de componentes
        self._monitor_task = None
    
    def register_component(
        self, 
        component_id: str, 
        component: ComponentAPI,
        essential: bool = False,
        setup_checkpointing: bool = True
    ) -> None:
        """
        Registrar un componente en el sistema.
        
        Args:
            component_id: ID único del componente
            component: Instancia del componente
            essential: Si es un componente esencial
            setup_checkpointing: Configurar checkpointing automático
        """
        if component_id in self.components:
            logger.warning(f"Componente {component_id} ya registrado, reemplazando")
        
        # Registrar componente
        self.components[component_id] = component
        
        # Configurar checkpointing si se solicita
        if setup_checkpointing:
            component.setup_checkpointing(
                checkpoint_dir=self.checkpoint_dir,
                interval_ms=150.0,
                checkpoint_type=CheckpointType.DISK
            )
        
        # Iniciar tarea de escucha de eventos locales
        asyncio.create_task(component.listen_local())
        
        # Marcar como esencial si es necesario
        if essential and component_id not in self.recovery_mgr.essential_components:
            self.recovery_mgr.essential_components.append(component_id)
        
        logger.debug(f"Componente {component_id} registrado")
    
    @with_retry(base_delay=0.1, max_retries=2, jitter_factor=0.1)
    @with_circuit_breaker(failure_threshold=3, recovery_timeout=5.0)
    async def request(
        self, 
        target_id: str, 
        request_type: str, 
        data: Dict[str, Any], 
        source: str,
        timeout: float = 2.0
    ) -> Optional[Any]:
        """
        Realizar solicitud API a un componente.
        
        Args:
            target_id: ID del componente objetivo
            request_type: Tipo de solicitud
            data: Datos de la solicitud
            source: Fuente de la solicitud
            timeout: Tiempo máximo de espera en segundos
            
        Returns:
            Resultado de la solicitud o None si hay error
        """
        # Verificar disponibilidad del componente
        if target_id not in self.components:
            logger.error(f"Componente {target_id} no encontrado")
            return None
        
        component = self.components[target_id]
        if not component.healthy:
            logger.error(f"Componente {target_id} no disponible (unhealthy)")
            return None
        
        # Verificar modo seguro
        if (self.safe_mode_mgr.current_mode != RecoveryMode.NORMAL and
            not self.safe_mode_mgr.is_operation_allowed("request", target_id)):
            logger.warning(f"Solicitud a {target_id} rechazada en modo {self.safe_mode_mgr.current_mode.name}")
            return None
        
        # Ejecutar solicitud con timeout
        try:
            self.request_count += 1
            return await asyncio.wait_for(
                component.process_request(request_type, data, source),
                timeout=timeout
            )
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout en {target_id} para {request_type}")
            component.healthy = False
            
            # Intentar recuperar el componente
            asyncio.create_task(self._attempt_component_recovery(target_id))
            return None
            
        except Exception as e:
            logger.error(f"Error en {target_id}: {e}")
            component.healthy = False
            
            # Intentar recuperar el componente
            asyncio.create_task(self._attempt_component_recovery(target_id))
            return None
    
    async def emit_local(
        self, 
        event_type: str, 
        data: Dict[str, Any], 
        source: str,
        excluded_components: List[str] = None
    ) -> int:
        """
        Emitir evento local a todos los componentes.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            excluded_components: Componentes a excluir
            
        Returns:
            Número de componentes a los que se envió el evento
        """
        if not self.running:
            return 0
            
        self.local_event_count += 1
        excluded = set(excluded_components or [])
        excluded.add(source)  # No enviar al emisor
        
        # Verificar modo seguro
        if self.safe_mode_mgr.current_mode != RecoveryMode.NORMAL:
            # En modo seguro, solo enviar a componentes esenciales
            for cid in list(self.components.keys()):
                if not self.safe_mode_mgr.is_component_essential(cid):
                    excluded.add(cid)
        
        # Preparar tareas para envío paralelo
        tasks = []
        skipped = 0
        
        for cid, component in self.components.items():
            if cid in excluded or not component.healthy:
                continue
                
            # Detección proactiva: verificar si la cola está llena
            if component.local_queue.qsize() >= 90:  # 90% de capacidad
                logger.warning(f"Cola local de {cid} cerca del límite, evento descartado")
                skipped += 1
                continue
                
            # Añadir tarea
            tasks.append(component.local_queue.put((event_type, data, source)))
        
        # Ejecutar envíos en paralelo
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
        # Registrar componentes omitidos
        total_sent = len(tasks)
        if skipped > 0:
            logger.info(f"Evento local {event_type}: {total_sent} enviados, {skipped} omitidos")
            
        return total_sent
    
    async def emit_external(
        self, 
        event_type: str, 
        data: Dict[str, Any], 
        source: str,
        excluded_components: List[str] = None
    ) -> int:
        """
        Emitir evento externo a todos los clientes WebSocket.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            excluded_components: Componentes a excluir
            
        Returns:
            Número de clientes a los que se envió el evento
        """
        if not self.running or not self.websocket_clients:
            return 0
            
        # Verificar límite de conexiones
        if len(self.websocket_clients) >= self.max_ws_connections:
            logger.warning(f"Límite de conexiones WebSocket alcanzado, evento externo descartado")
            return 0
        
        # Preparar mensaje
        message = json.dumps({
            "type": event_type, 
            "data": data, 
            "source": source,
            "timestamp": time.time()
        })
        
        self.external_event_count += 1
        excluded = set(excluded_components or [])
        excluded.add(source)  # No enviar al emisor
        
        # Verificar modo seguro
        if self.safe_mode_mgr.current_mode != RecoveryMode.NORMAL:
            # En modo seguro, solo enviar a componentes esenciales
            for cid in list(self.websocket_clients.keys()):
                if not self.safe_mode_mgr.is_component_essential(cid):
                    excluded.add(cid)
        
        # Enviar en paralelo
        tasks = []
        for cid, ws in self.websocket_clients.items():
            if cid in excluded or ws.closed:
                continue
            tasks.append(ws.send_str(message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
        return len(tasks)
    
    async def _external_websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """
        Manejador para conexiones WebSocket externas.
        
        Args:
            request: Solicitud HTTP
            
        Returns:
            Respuesta WebSocket
        """
        # Verificar límite de conexiones
        if len(self.websocket_clients) >= self.max_ws_connections:
            return web.Response(status=503, text="Límite de conexiones alcanzado")
        
        # Preparar WebSocket
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # Verificar ID de componente
        component_id = request.query.get("id")
        if not component_id:
            await ws.close(code=1008, message=b"ID de componente requerido")
            return ws
            
        if component_id not in self.components:
            await ws.close(code=1008, message=b"Componente no registrado")
            return ws
        
        # Verificar modo seguro
        if (self.safe_mode_mgr.current_mode != RecoveryMode.NORMAL and
            not self.safe_mode_mgr.is_component_essential(component_id)):
            await ws.close(code=1008, message=f"Componente no esencial rechazado en modo {self.safe_mode_mgr.current_mode.name}".encode())
            return ws
        
        # Registrar cliente
        self.websocket_clients[component_id] = ws
        logger.info(f"WebSocket externo conectado para {component_id}")
        
        # Procesar mensajes
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        component = self.components.get(component_id)
                        
                        if component and component.healthy:
                            await component.on_external_event(
                                data.get("type"), 
                                data.get("data", {}), 
                                data.get("source", component_id)
                            )
                    except json.JSONDecodeError:
                        logger.error(f"Formato JSON inválido de {component_id}")
                    except Exception as e:
                        logger.error(f"Error al procesar mensaje WebSocket de {component_id}: {e}")
                
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f"Error WebSocket de {component_id}: {ws.exception()}")
        
        finally:
            # Limpiar al desconectar
            self.websocket_clients.pop(component_id, None)
            logger.info(f"WebSocket externo desconectado para {component_id}")
        
        return ws
    
    async def _api_request_handler(self, request: web.Request) -> web.Response:
        """
        Manejador para solicitudes API HTTP.
        
        Args:
            request: Solicitud HTTP
            
        Returns:
            Respuesta HTTP
        """
        # Extraer objetivo
        target_id = request.match_info["target"]
        if target_id not in self.components:
            return web.Response(status=404, text=f"Componente {target_id} no encontrado")
        
        # Procesar solicitud
        try:
            # Extraer datos
            data = await request.json()
            
            # Ejecutar solicitud
            result = await self.request(
                target_id, 
                data.get("type"), 
                data.get("data", {}), 
                data.get("source", "external")
            )
            
            # Generar respuesta
            return web.json_response({
                "result": result,
                "status": "success" if result is not None else "error",
                "timestamp": time.time()
            })
            
        except json.JSONDecodeError:
            return web.Response(status=400, text="Formato JSON inválido")
            
        except Exception as e:
            logger.error(f"Error en solicitud API a {target_id}: {e}")
            return web.Response(status=500, text=f"Error interno: {str(e)}")
    
    async def _health_check_handler(self, request: web.Request) -> web.Response:
        """
        Manejador para verificación de salud.
        
        Args:
            request: Solicitud HTTP
            
        Returns:
            Respuesta HTTP con estado de salud
        """
        # Contar componentes por estado
        total = len(self.components)
        healthy = sum(1 for c in self.components.values() if c.healthy)
        
        # Calcular métricas
        health_ratio = healthy / total if total > 0 else 0
        system_healthy = health_ratio >= 0.6  # 60% mínimo
        
        # Preparar respuesta
        response = {
            "status": "healthy" if system_healthy else "degraded",
            "components": {
                "total": total,
                "healthy": healthy,
                "unhealthy": total - healthy
            },
            "health_ratio": health_ratio,
            "safe_mode": self.safe_mode_mgr.current_mode.name,
            "timestamp": time.time()
        }
        
        status_code = 200 if system_healthy else 503
        return web.json_response(response, status=status_code)
    
    async def _status_handler(self, request: web.Request) -> web.Response:
        """
        Manejador para información de estado.
        
        Args:
            request: Solicitud HTTP
            
        Returns:
            Respuesta HTTP con estado detallado
        """
        # Recopilar métricas
        component_status = {}
        for cid, comp in self.components.items():
            component_status[cid] = {
                "healthy": comp.healthy,
                "queue_size": comp.local_queue.qsize(),
                "last_active": time.time() - comp.last_active,
                "essential": cid in self.recovery_mgr.essential_components
            }
        
        # Estadísticas
        stats = {
            "request_count": self.request_count,
            "local_event_count": self.local_event_count,
            "external_event_count": self.external_event_count,
            "websocket_clients": len(self.websocket_clients)
        }
        
        # Sistema
        system = {
            "mode": self.safe_mode_mgr.current_mode.name,
            "running": self.running,
            "uptime": time.time() - self._start_time if hasattr(self, "_start_time") else 0
        }
        
        # Preparar respuesta
        response = {
            "components": component_status,
            "stats": stats,
            "system": system,
            "timestamp": time.time()
        }
        
        return web.json_response(response)
    
    async def _monitor_components(self) -> None:
        """
        Monitor proactivo de componentes.
        
        Detecta problemas y realiza recuperación automática.
        """
        while self.running:
            try:
                # Iterar sobre componentes
                for cid, component in list(self.components.items()):
                    # Detectar inactividad prolongada
                    if time.time() - component.last_active > 30.0:  # 30 segundos
                        logger.warning(f"Componente {cid} inactivo por demasiado tiempo")
                        component.healthy = False
                        
                        # Recuperación
                        await self._attempt_component_recovery(cid)
                    
                    # Detectar colas saturadas
                    elif component.local_queue.qsize() > 80:  # 80% de capacidad
                        logger.warning(f"Cola local de {cid} cerca del límite: {component.local_queue.qsize()}/100")
                        
                        # Medida preventiva: procesar eventos más rápido
                        for _ in range(min(10, component.local_queue.qsize())):
                            component.local_queue.get_nowait()
                            component.local_queue.task_done()
                
                # Verificar si debemos activar modo seguro
                unhealthy_count = sum(1 for c in self.components.values() if not c.healthy)
                if (unhealthy_count > len(self.components) * 0.3 and  # >30% componentes fallidos
                    self.safe_mode_mgr.current_mode == RecoveryMode.NORMAL):
                    
                    await self.safe_mode_mgr.activate_safe_mode(
                        f"Activación automática: {unhealthy_count} componentes fallidos"
                    )
                
                # Verificar si podemos desactivar modo seguro
                elif (unhealthy_count == 0 and 
                      self.safe_mode_mgr.current_mode != RecoveryMode.NORMAL):
                    
                    await self.safe_mode_mgr.deactivate_safe_mode()
                
                # Esperar hasta el siguiente ciclo
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                # Cancelación solicitada, salir
                break
                
            except Exception as e:
                logger.error(f"Error en monitor de componentes: {e}")
                await asyncio.sleep(5.0)  # Esperar más tiempo tras error
    
    async def _attempt_component_recovery(self, component_id: str) -> bool:
        """
        Intentar recuperar un componente fallido.
        
        Args:
            component_id: ID del componente a recuperar
            
        Returns:
            True si se recuperó correctamente
        """
        if component_id not in self.components:
            return False
        
        component = self.components[component_id]
        
        # Actualizar estado en Recovery Manager
        self.recovery_mgr.update_component_state(component_id, "failed")
        
        logger.info(f"Intentando recuperar componente {component_id}")
        
        # Verificar si tiene checkpointing
        if component.checkpoint_mgr:
            # Función de recuperación
            async def restore_component(state):
                component.state = state
                component.healthy = True
                # Reiniciar escucha local
                asyncio.create_task(component.listen_local())
            
            # Intentar recuperación
            success = await self.recovery_mgr.attempt_recovery(
                component_id, restore_component
            )
            
            if success:
                logger.info(f"Componente {component_id} recuperado exitosamente")
                return True
        
        # Si no tiene checkpointing o falló la recuperación, reiniciar
        component.healthy = True
        # Reiniciar escucha local
        asyncio.create_task(component.listen_local())
        
        # Actualizar estado
        self.recovery_mgr.update_component_state(component_id, "healthy")
        
        logger.info(f"Componente {component_id} reiniciado")
        return True
    
    async def start(self) -> None:
        """Iniciar el coordinador y todos los componentes."""
        if self.running:
            return
        
        # Marcar como iniciado
        self.running = True
        self._start_time = time.time()
        
        # Iniciar componentes
        for component in self.components.values():
            await component.start()
        
        # Iniciar servidor HTTP
        runner = web.AppRunner(self.app)
        await runner.setup()
        self.site = web.TCPSite(runner, self.host, self.port)
        await self.site.start()
        
        # Iniciar monitor de componentes
        self._monitor_task = asyncio.create_task(self._monitor_components())
        
        # Iniciar monitoreo de recuperación
        await self.recovery_mgr.start_monitoring()
        
        logger.info(f"Coordinador iniciado en {self.host}:{self.port}")
    
    async def stop(self) -> None:
        """Detener el coordinador y todos los componentes."""
        if not self.running:
            return
        
        # Marcar como detenido
        self.running = False
        
        # Detener monitor
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Detener monitoreo de recuperación
        await self.recovery_mgr.stop_monitoring()
        
        # Cerrar conexiones WebSocket
        for ws in self.websocket_clients.values():
            await ws.close()
        
        # Detener componentes
        for component in self.components.values():
            await component.stop()
        
        # Detener servidor HTTP
        await self.app.shutdown()
        await self.app.cleanup()
        
        # Registrar estadísticas
        logger.info(
            f"Coordinador detenido. "
            f"Solicitudes: {self.request_count}, "
            f"Eventos locales: {self.local_event_count}, "
            f"Eventos externos: {self.external_event_count}"
        )

# Ejemplo de componente simple
class TestComponent(ComponentAPI):
    """Componente de prueba para el sistema híbrido."""
    
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """Procesar solicitud de prueba."""
        await asyncio.sleep(0.1)  # Simular trabajo
        
        if request_type == "ping":
            return {"message": f"Pong desde {self.id}", "timestamp": time.time()}
        elif request_type == "echo":
            return data
        elif request_type == "error":
            raise Exception("Error simulado")
        
        return None
    
    async def on_local_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Manejar evento local de prueba."""
        await super().on_local_event(event_type, data, source)
        
        # Actualizar estado
        if "events" not in self.state:
            self.state["events"] = []
        
        self.state["events"].append({
            "type": event_type,
            "source": source,
            "timestamp": time.time()
        })
        
        # Limitar cantidad de eventos almacenados
        if len(self.state["events"]) > 100:
            self.state["events"] = self.state["events"][-100:]