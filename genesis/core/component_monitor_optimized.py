"""
Monitor de componentes optimizado para el sistema Genesis.

Este módulo implementa un monitor de componentes mejorado que utiliza 
el sistema de colas dedicadas para detectar y gestionar componentes problemáticos,
evitando fallos en cascada y bloqueos.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Set, Tuple, Callable

from genesis.core.base import Component

# Configuración del logger
logger = logging.getLogger(__name__)

class OptimizedComponentMonitor(Component):
    """
    Monitor de componentes optimizado con detección avanzada de fallos.
    
    Características clave:
    1. Utiliza colas dedicadas para comunicación sin bloqueos
    2. Implementa timeouts agresivos para operaciones críticas
    3. Mantiene historial detallado de estados de componentes
    4. Detecta y aísla componentes problemáticos automáticamente
    5. Implementa políticas de recuperación configurables
    """
    
    def __init__(self, name: str = "component_monitor", 
                check_interval: float = 5.0,
                max_failures: int = 3,
                recovery_interval: float = 30.0,
                test_mode: bool = False):
        """
        Inicializar el monitor de componentes optimizado.
        
        Args:
            name: Nombre del componente monitor
            check_interval: Intervalo en segundos entre verificaciones de salud
            max_failures: Número máximo de fallos consecutivos antes de aislar
            recovery_interval: Intervalo en segundos entre intentos de recuperación
            test_mode: Si es True, usa timeouts más agresivos para pruebas
        """
        super().__init__(name)
        self.check_interval = check_interval
        self.max_failures = max_failures
        self.recovery_interval = recovery_interval
        self.test_mode = test_mode
        
        # Timeouts adaptativos
        self.check_timeout = 0.5 if test_mode else 2.0
        
        # Estado de componentes
        self.health_status: Dict[str, bool] = {}  # Componente -> Estado actual (True=sano, False=no sano)
        self.failure_counts: Dict[str, int] = {}  # Componente -> Contador de fallos consecutivos
        self.isolated_components: Set[str] = set()  # Componentes actualmente aislados
        self.component_metadata: Dict[str, Dict[str, Any]] = {}  # Metadatos por componente
        
        # Histórico de estados
        self.status_history: Dict[str, List[Tuple[float, bool]]] = {}  # Componente -> Lista de (timestamp, estado)
        
        # Variables de control
        self.monitor_task: Optional[asyncio.Task] = None
        self.recovery_task: Optional[asyncio.Task] = None
        self.active_tasks: Set[asyncio.Task] = set()
        self.running = False
        self.last_check_time = 0
        self.isolation_events = 0
        self.recovery_events = 0
        
        # Callbacks para notificaciones
        self.status_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
    async def start(self) -> None:
        """Iniciar el monitor de componentes optimizado."""
        logger.info(f"Iniciando monitor de componentes optimizado ({self.name})")
        self.running = True
        
        # Iniciar tarea de monitoreo
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        self.monitor_task.set_name(f"{self.name}_monitoring_loop")
        self.active_tasks.add(self.monitor_task)
        self.monitor_task.add_done_callback(lambda t: self.active_tasks.discard(t))
        
        # Iniciar tarea de recuperación
        self.recovery_task = asyncio.create_task(self._recovery_loop())
        self.recovery_task.set_name(f"{self.name}_recovery_loop")
        self.active_tasks.add(self.recovery_task)
        self.recovery_task.add_done_callback(lambda t: self.active_tasks.discard(t))
        
        # Notificar inicio del monitor
        if self.event_bus:
            try:
                await asyncio.wait_for(
                    self.event_bus.emit(
                        "system.monitor.started",
                        {"monitor": self.name, "interval": self.check_interval},
                        self.name
                    ),
                    timeout=self.check_timeout
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.error(f"Error al emitir evento de inicio: {e}")
        
    async def stop(self) -> None:
        """Detener el monitor de componentes."""
        logger.info(f"Deteniendo monitor de componentes ({self.name})")
        self.running = False
        
        # Cancelar tareas activas
        for task in self.active_tasks:
            if not task.done():
                task.cancel()
        
        # Esperar a que las tareas terminen
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
        
        # Notificar detención del monitor
        if self.event_bus:
            try:
                await asyncio.wait_for(
                    self.event_bus.emit(
                        "system.monitor.stopped",
                        {"monitor": self.name},
                        self.name
                    ),
                    timeout=self.check_timeout
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.error(f"Error al emitir evento de detención: {e}")
            
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """
        Manejar eventos recibidos por el monitor.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Opcional, respuesta al evento
            
        Eventos soportados:
        - check_status: Verificar estado del monitor
        - check_component: Verificar estado de un componente específico
        - isolate_component: Aislar manualmente un componente
        - recover_component: Intentar recuperar un componente aislado
        - get_health_report: Obtener informe completo del estado de los componentes
        """
        # Verificar estado del monitor
        if event_type == "check_status":
            return {
                "component": self.name,
                "healthy": self.running,
                "monitored_components": len(self.health_status),
                "isolated_components": len(self.isolated_components),
                "last_check_time": self.last_check_time
            }
            
        # Verificar estado de un componente específico
        elif event_type == "check_component":
            component_id = data.get("component_id")
            if not component_id:
                return {"error": "Falta component_id en los datos"}
                
            try:
                result = await asyncio.wait_for(
                    self._check_component_health(component_id),
                    timeout=self.check_timeout
                )
                return {
                    "component_id": component_id,
                    "healthy": result.get("healthy", False),
                    "isolated": component_id in self.isolated_components,
                    "failure_count": self.failure_counts.get(component_id, 0),
                    "metadata": self.component_metadata.get(component_id, {})
                }
            except asyncio.TimeoutError:
                logger.warning(f"Timeout al verificar componente {component_id}")
                return {
                    "component_id": component_id,
                    "healthy": False,
                    "error": "timeout",
                    "isolated": component_id in self.isolated_components
                }
            
        # Aislar componente manualmente
        elif event_type == "isolate_component":
            component_id = data.get("component_id")
            if not component_id:
                return {"error": "Falta component_id en los datos"}
                
            reason = data.get("reason", "Manual isolation")
            try:
                await asyncio.wait_for(
                    self._isolate_component(component_id, reason),
                    timeout=self.check_timeout
                )
                
                return {
                    "component_id": component_id,
                    "isolated": True,
                    "reason": reason
                }
            except asyncio.TimeoutError:
                logger.warning(f"Timeout al aislar componente {component_id}")
                # Forzar aislamiento en caso de timeout
                self.isolated_components.add(component_id)
                return {
                    "component_id": component_id,
                    "isolated": True,
                    "reason": f"{reason} (forzado por timeout)"
                }
            
        # Recuperar componente manualmente
        elif event_type == "recover_component":
            component_id = data.get("component_id")
            if not component_id:
                return {"error": "Falta component_id en los datos"}
                
            try:
                success = await asyncio.wait_for(
                    self._attempt_recovery(component_id),
                    timeout=self.check_timeout * 2
                )
                
                return {
                    "component_id": component_id,
                    "recovered": success,
                    "currently_isolated": component_id in self.isolated_components
                }
            except asyncio.TimeoutError:
                logger.warning(f"Timeout al recuperar componente {component_id}")
                return {
                    "component_id": component_id,
                    "recovered": False,
                    "error": "timeout",
                    "currently_isolated": component_id in self.isolated_components
                }
            
        # Obtener informe completo de salud
        elif event_type == "get_health_report":
            return {
                "monitor": self.name,
                "timestamp": time.time(),
                "component_status": self.health_status,
                "isolated_components": list(self.isolated_components),
                "failure_counts": self.failure_counts,
                "isolation_events": self.isolation_events,
                "recovery_events": self.recovery_events,
                "monitor_healthy": self.running,
                "component_metadata": self.component_metadata
            }
            
        # Registrar componente para monitoreo explícito
        elif event_type == "register_for_monitoring":
            component_id = data.get("component_id")
            metadata = data.get("metadata", {})
            if not component_id:
                return {"error": "Falta component_id en los datos"}
                
            # Inicializar registros si no existen
            if component_id not in self.health_status:
                self.health_status[component_id] = True
                self.failure_counts[component_id] = 0
                self.status_history[component_id] = []
                
            # Almacenar metadatos
            self.component_metadata[component_id] = metadata
            
            return {
                "component_id": component_id,
                "registered": True,
                "monitor": self.name
            }
            
        # Para otros tipos de eventos
        return {"component": self.name, "event": event_type, "processed": True}
        
    async def _monitoring_loop(self) -> None:
        """
        Bucle principal de monitoreo para verificar periódicamente 
        el estado de los componentes.
        """
        logger.info(f"Iniciando bucle de monitoreo con intervalo de {self.check_interval}s")
        
        while self.running:
            try:
                # Registrar inicio de verificación
                self.last_check_time = time.time()
                
                # Obtener lista de componentes del engine
                components_to_check = set()
                if self.engine and hasattr(self.engine, "components"):
                    # Solo verificar componentes en el motor
                    for component_id in list(self.engine.components.keys()):
                        # Omitir monitor para evitar recursión
                        if component_id == self.name:
                            continue
                        components_to_check.add(component_id)
                        
                # Añadir componentes registrados explícitamente
                for component_id in list(self.health_status.keys()):
                    if component_id != self.name:
                        components_to_check.add(component_id)
                
                logger.debug(f"Verificando {len(components_to_check)} componentes")
                
                # Verificar componentes en paralelo para mayor eficiencia
                if components_to_check:
                    check_tasks = []
                    for component_id in components_to_check:
                        # Omitir componentes ya aislados
                        if component_id in self.isolated_components:
                            continue
                            
                        # Crear tarea para verificar salud
                        task = asyncio.create_task(self._check_component_health(component_id))
                        check_tasks.append(task)
                    
                    # Esperar a que todas las verificaciones terminen, con timeout global
                    if check_tasks:
                        await asyncio.wait(
                            check_tasks,
                            timeout=self.check_interval * 0.8,  # Dejar margen para el sleep
                            return_when=asyncio.ALL_COMPLETED
                        )
                
                # Esperar hasta la próxima verificación
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                logger.info("Bucle de monitoreo cancelado")
                break
            except Exception as e:
                logger.error(f"Error en bucle de monitoreo: {e}")
                # Continuar monitoreando a pesar del error
                await asyncio.sleep(1.0)
    
    async def _recovery_loop(self) -> None:
        """
        Bucle de recuperación que intenta periódicamente
        recuperar componentes aislados.
        """
        logger.info(f"Iniciando bucle de recuperación con intervalo de {self.recovery_interval}s")
        
        while self.running:
            try:
                # Solo continuar si hay componentes aislados
                if self.isolated_components:
                    logger.info(f"Intentando recuperar {len(self.isolated_components)} componentes aislados")
                    
                    # Intentar recuperar en paralelo
                    recovery_tasks = []
                    for component_id in list(self.isolated_components):
                        task = asyncio.create_task(self._attempt_recovery(component_id))
                        recovery_tasks.append(task)
                    
                    # Esperar recuperaciones con timeout global
                    if recovery_tasks:
                        await asyncio.wait(
                            recovery_tasks,
                            timeout=self.recovery_interval * 0.8,
                            return_when=asyncio.ALL_COMPLETED
                        )
                
                # Esperar hasta el próximo intento
                await asyncio.sleep(self.recovery_interval)
                
            except asyncio.CancelledError:
                logger.info("Bucle de recuperación cancelado")
                break
            except Exception as e:
                logger.error(f"Error en bucle de recuperación: {e}")
                await asyncio.sleep(1.0)
    
    async def _check_component_health(self, component_id: str) -> Dict[str, Any]:
        """
        Verificar la salud de un componente específico.
        
        Args:
            component_id: ID del componente a verificar
            
        Returns:
            Diccionario con estado del componente
        """
        # Verificar que el componente existe en el engine
        component_exists = False
        if self.engine and hasattr(self.engine, "components"):
            component_exists = component_id in self.engine.components
            
        if not component_exists and component_id not in self.health_status:
            logger.warning(f"Componente {component_id} no registrado en el motor ni en monitor")
            return {"component_id": component_id, "healthy": False, "error": "Componente no registrado"}
        
        try:
            # Enviar evento de verificación con timeout
            if self.event_bus:
                response = await asyncio.wait_for(
                    self.event_bus.emit_with_response("check_status", {}, component_id),
                    timeout=self.check_timeout
                )
                
                # Procesar respuesta
                healthy = False
                if response and isinstance(response, list) and len(response) > 0:
                    # Obtener estado de salud de la respuesta
                    if isinstance(response[0], dict) and "healthy" in response[0]:
                        healthy = response[0]["healthy"]
                
                # Actualizar registro de salud
                previous_state = self.health_status.get(component_id, True)  # Asumir sano por defecto
                self.health_status[component_id] = healthy
                
                # Almacenar en historial
                if component_id not in self.status_history:
                    self.status_history[component_id] = []
                self.status_history[component_id].append((time.time(), healthy))
                
                # Mantener historial limitado
                if len(self.status_history[component_id]) > 100:
                    self.status_history[component_id] = self.status_history[component_id][-100:]
                
                # Actualizar contador de fallos
                if healthy:
                    # Resetear contador si está sano
                    self.failure_counts[component_id] = 0
                else:
                    # Incrementar contador si no está sano
                    current_count = self.failure_counts.get(component_id, 0)
                    self.failure_counts[component_id] = current_count + 1
                    
                    # Verificar si se debe aislar
                    if self.failure_counts[component_id] >= self.max_failures:
                        await self._isolate_component(
                            component_id, 
                            f"Componente no saludable durante {self.max_failures} verificaciones consecutivas"
                        )
                
                # Emitir evento de cambio de estado si cambió
                if previous_state != healthy and self.event_bus:
                    await self.event_bus.emit(
                        "system.component.health_changed",
                        {
                            "component_id": component_id,
                            "healthy": healthy,
                            "previous": previous_state,
                            "failure_count": self.failure_counts.get(component_id, 0),
                            "metadata": self.component_metadata.get(component_id, {})
                        },
                        self.name
                    )
                
                return {
                    "component_id": component_id,
                    "healthy": healthy,
                    "response": response
                }
            else:
                logger.warning(f"No hay event_bus disponible para verificar {component_id}")
                return {
                    "component_id": component_id,
                    "healthy": False,
                    "error": "No event_bus disponible"
                }
            
        except asyncio.TimeoutError:
            # El componente no respondió a tiempo
            logger.warning(f"Timeout al verificar componente {component_id}")
            
            # Actualizar estado y contador
            self.health_status[component_id] = False
            current_count = self.failure_counts.get(component_id, 0)
            self.failure_counts[component_id] = current_count + 1
            
            # Almacenar en historial
            if component_id not in self.status_history:
                self.status_history[component_id] = []
            self.status_history[component_id].append((time.time(), False))
            
            # Verificar si se debe aislar
            if self.failure_counts[component_id] >= self.max_failures:
                await self._isolate_component(
                    component_id,
                    f"Componente no responde después de {self.max_failures} intentos"
                )
            
            return {
                "component_id": component_id,
                "healthy": False,
                "error": "timeout"
            }
            
        except Exception as e:
            logger.error(f"Error al verificar componente {component_id}: {e}")
            
            # Actualizar estado pero no aislar automáticamente por errores de verificación
            self.health_status[component_id] = False
            
            return {
                "component_id": component_id,
                "healthy": False,
                "error": str(e)
            }
    
    async def _isolate_component(self, component_id: str, reason: str) -> None:
        """
        Aislar un componente del sistema.
        
        Args:
            component_id: ID del componente a aislar
            reason: Razón del aislamiento
        """
        # Verificar si ya está aislado
        if component_id in self.isolated_components:
            logger.debug(f"Componente {component_id} ya está aislado")
            return
            
        logger.info(f"Aislando componente {component_id}: {reason}")
        
        # Añadir a la lista de componentes aislados
        self.isolated_components.add(component_id)
        self.isolation_events += 1
        
        # Almacenar la razón en metadatos
        if component_id not in self.component_metadata:
            self.component_metadata[component_id] = {}
        self.component_metadata[component_id]["isolation_reason"] = reason
        self.component_metadata[component_id]["isolation_time"] = time.time()
        
        # Notificar aislamiento al sistema
        if self.event_bus:
            try:
                # Usar create_task para no bloquear
                task = asyncio.create_task(
                    self.event_bus.emit(
                        "system.component.isolated",
                        {
                            "component_id": component_id,
                            "reason": reason,
                            "timestamp": time.time(),
                            "monitor": self.name,
                            "metadata": self.component_metadata.get(component_id, {})
                        },
                        self.name
                    )
                )
                self.active_tasks.add(task)
                task.add_done_callback(lambda t: self.active_tasks.discard(t))
                
            except Exception as e:
                logger.error(f"Error al emitir evento de aislamiento para {component_id}: {e}")
        
        # Notificar a componentes dependientes
        await self._notify_dependencies(component_id, False)
        
    async def _notify_dependencies(self, component_id: str, healthy: bool) -> None:
        """
        Notificar a componentes dependientes sobre cambio de estado.
        
        Args:
            component_id: ID del componente que cambió de estado
            healthy: Nuevo estado del componente
        """
        if not self.event_bus:
            return
            
        try:
            # Emitir evento de cambio de estado para que lo reciban los dependientes
            await self.event_bus.emit(
                "dependency.status_changed",
                {
                    "component_id": component_id,
                    "status": healthy,
                    "timestamp": time.time(),
                    "monitor": self.name
                },
                self.name
            )
            
        except Exception as e:
            logger.error(f"Error al notificar dependencias de {component_id}: {e}")
    
    async def _attempt_recovery(self, component_id: str) -> bool:
        """
        Intentar recuperar un componente aislado.
        
        Args:
            component_id: ID del componente a recuperar
            
        Returns:
            True si se recuperó correctamente, False en caso contrario
        """
        # Verificar si está aislado
        if component_id not in self.isolated_components:
            logger.debug(f"Componente {component_id} no está aislado, no se requiere recuperación")
            return True
            
        logger.info(f"Intentando recuperar componente {component_id}")
        
        # Verificar estado actual
        check_result = await self._check_component_health(component_id)
        healthy = check_result.get("healthy", False)
        
        if healthy:
            logger.info(f"Componente {component_id} está saludable, recuperando")
            
            # Eliminar de la lista de aislados
            self.isolated_components.remove(component_id)
            self.recovery_events += 1
            
            # Actualizar metadatos
            if component_id in self.component_metadata:
                self.component_metadata[component_id]["recovery_time"] = time.time()
                self.component_metadata[component_id]["recovery_count"] = self.component_metadata.get(component_id, {}).get("recovery_count", 0) + 1
                
            # Notificar recuperación
            if self.event_bus:
                try:
                    await self.event_bus.emit(
                        "system.component.recovered",
                        {
                            "component_id": component_id,
                            "timestamp": time.time(),
                            "monitor": self.name,
                            "metadata": self.component_metadata.get(component_id, {})
                        },
                        self.name
                    )
                except Exception as e:
                    logger.error(f"Error al emitir evento de recuperación para {component_id}: {e}")
            
            # Notificar a componentes dependientes
            await self._notify_dependencies(component_id, True)
            
            return True
        else:
            logger.warning(f"Componente {component_id} sigue no saludable, no se puede recuperar")
            return False
            
    def register_status_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """
        Registrar callback para notificaciones de cambio de estado.
        
        Args:
            callback: Función que recibe (component_id, metadata)
        """
        self.status_callbacks.append(callback)