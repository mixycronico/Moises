"""
Monitor de componentes para el sistema Genesis.

Este módulo implementa un sistema de monitoreo para detectar y manejar
componentes problemáticos, evitando que sus fallos afecten al resto del sistema.
El monitor permite aislar componentes que no responden o están en estado no saludable,
mitigando así los fallos en cascada.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Set, Tuple

from genesis.core.component import Component

# Configuración del logger
logger = logging.getLogger(__name__)

class ComponentMonitor(Component):
    """
    Monitor para detectar y aislar componentes problemáticos.
    
    Esta clase implementa un componente especial que:
    1. Monitorea periódicamente la salud de otros componentes
    2. Aísla componentes que no responden o fallan consistentemente
    3. Intenta recuperar componentes aislados cuando sea posible
    4. Notifica sobre cambios de estado y problemas detectados
    """
    
    def __init__(self, name: str = "component_monitor", 
                check_interval: float = 5.0,
                max_failures: int = 3,
                recovery_interval: float = 30.0):
        """
        Inicializar el monitor de componentes.
        
        Args:
            name: Nombre del componente monitor
            check_interval: Intervalo en segundos entre verificaciones de salud
            max_failures: Número máximo de fallos consecutivos antes de aislar
            recovery_interval: Intervalo en segundos entre intentos de recuperación
        """
        super().__init__(name)
        self.check_interval = check_interval
        self.max_failures = max_failures
        self.recovery_interval = recovery_interval
        
        # Estado de componentes
        self.health_status: Dict[str, bool] = {}  # Componente -> Estado actual (True=sano, False=no sano)
        self.failure_counts: Dict[str, int] = {}  # Componente -> Contador de fallos consecutivos
        self.isolated_components: Set[str] = set()  # Componentes actualmente aislados
        
        # Histórico de estados
        self.status_history: Dict[str, List[Tuple[float, bool]]] = {}  # Componente -> Lista de (timestamp, estado)
        
        # Variables de control
        self.monitor_task = None
        self.recovery_task = None
        self.running = False
        self.last_check_time = 0
        self.isolation_events = 0
        self.recovery_events = 0
        
    async def start(self) -> None:
        """Iniciar el monitor de componentes."""
        logger.info(f"Iniciando monitor de componentes ({self.name})")
        self.running = True
        
        # Iniciar tarea de monitoreo
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        self.monitor_task.set_name(f"{self.name}_monitoring_loop")
        
        # Iniciar tarea de recuperación
        self.recovery_task = asyncio.create_task(self._recovery_loop())
        self.recovery_task.set_name(f"{self.name}_recovery_loop")
        
        # Notificar inicio del monitor
        try:
            await self.event_bus.emit(
                "system.monitor.started",
                {"monitor": self.name, "interval": self.check_interval},
                self.name
            )
        except Exception as e:
            logger.error(f"Error al emitir evento de inicio: {e}")
        
    async def stop(self) -> None:
        """Detener el monitor de componentes."""
        logger.info(f"Deteniendo monitor de componentes ({self.name})")
        self.running = False
        
        # Detener tareas de monitoreo y recuperación
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
                
        if self.recovery_task and not self.recovery_task.done():
            self.recovery_task.cancel()
            try:
                await self.recovery_task
            except asyncio.CancelledError:
                pass
        
        # Notificar detención del monitor
        try:
            await self.event_bus.emit(
                "system.monitor.stopped",
                {"monitor": self.name},
                self.name
            )
        except Exception as e:
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
                
            result = await self._check_component_health(component_id)
            return {
                "component_id": component_id,
                "healthy": result.get("healthy", False),
                "isolated": component_id in self.isolated_components,
                "failure_count": self.failure_counts.get(component_id, 0)
            }
            
        # Aislar componente manualmente
        elif event_type == "isolate_component":
            component_id = data.get("component_id")
            if not component_id:
                return {"error": "Falta component_id en los datos"}
                
            reason = data.get("reason", "Manual isolation")
            await self._isolate_component(component_id, reason)
            
            return {
                "component_id": component_id,
                "isolated": True,
                "reason": reason
            }
            
        # Recuperar componente manualmente
        elif event_type == "recover_component":
            component_id = data.get("component_id")
            if not component_id:
                return {"error": "Falta component_id en los datos"}
                
            success = await self._attempt_recovery(component_id)
            
            return {
                "component_id": component_id,
                "recovered": success,
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
                "monitor_healthy": self.running
            }
            
        # Para otros tipos de eventos
        return {"component": self.name, "event": event_type, "processed": True}
        
    async def _monitoring_loop(self) -> None:
        """
        Bucle principal de monitoreo que verifica periódicamente 
        el estado de los componentes.
        """
        logger.info(f"Iniciando bucle de monitoreo con intervalo de {self.check_interval}s")
        
        while self.running:
            try:
                # Registrar inicio de verificación
                self.last_check_time = time.time()
                logger.debug(f"Iniciando verificación de componentes ({len(self.engine.components)} registrados)")
                
                # Verificar todos los componentes registrados en el motor
                for component_id in list(self.engine.components.keys()):
                    # Omitir monitor para evitar recursión
                    if component_id == self.name:
                        continue
                        
                    # Omitir componentes ya aislados
                    if component_id in self.isolated_components:
                        continue
                        
                    # Verificar salud del componente
                    await self._check_component_health(component_id)
                
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
                    
                    # Copiar la lista para evitar modificarla durante la iteración
                    for component_id in list(self.isolated_components):
                        await self._attempt_recovery(component_id)
                
                # Esperar hasta el próximo intento de recuperación
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
        # Verificar que el componente existe
        if component_id not in self.engine.components:
            logger.warning(f"Componente {component_id} no está registrado en el motor")
            return {"component_id": component_id, "healthy": False, "error": "Componente no registrado"}
            
        try:
            # Enviar evento de verificación con timeout
            timeout = 1.0  # timeout corto para detectar componentes bloqueados
            response = await asyncio.wait_for(
                self.event_bus.emit_with_response("check_status", {}, component_id),
                timeout=timeout
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
            
            # Mantener historial limitado (máximo 100 registros por componente)
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
            if previous_state != healthy:
                await self.event_bus.emit(
                    "system.component.health_changed",
                    {
                        "component_id": component_id,
                        "healthy": healthy,
                        "previous": previous_state,
                        "failure_count": self.failure_counts.get(component_id, 0)
                    },
                    self.name
                )
            
            return {
                "component_id": component_id,
                "healthy": healthy,
                "response": response
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
    
    async def _isolate_component(self, component_id: str, reason: str) -> bool:
        """
        Aislar un componente problemático para prevenir fallos en cascada.
        
        Args:
            component_id: ID del componente a aislar
            reason: Razón del aislamiento
            
        Returns:
            True si se aisló correctamente, False en caso contrario
        """
        # Verificar si ya está aislado
        if component_id in self.isolated_components:
            logger.debug(f"Componente {component_id} ya está aislado")
            return True
            
        logger.warning(f"Aislando componente {component_id}: {reason}")
        
        # Registrar aislamiento
        self.isolated_components.add(component_id)
        self.isolation_events += 1
        
        # Notificar aislamiento
        try:
            await self.event_bus.emit(
                "system.component.isolated",
                {
                    "component_id": component_id,
                    "reason": reason,
                    "failure_count": self.failure_counts.get(component_id, 0),
                    "timestamp": time.time()
                },
                self.name
            )
            
            # Notificar a los componentes dependientes
            await self.event_bus.emit(
                "dependency_status_change",
                {
                    "dependency_id": component_id,
                    "status": False,
                    "reason": "component_isolated"
                },
                self.name
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error al notificar aislamiento de {component_id}: {e}")
            return False
    
    async def _attempt_recovery(self, component_id: str) -> bool:
        """
        Intentar recuperar un componente aislado.
        
        Args:
            component_id: ID del componente a recuperar
            
        Returns:
            True si se recuperó exitosamente, False en caso contrario
        """
        # Verificar si está aislado
        if component_id not in self.isolated_components:
            logger.debug(f"Componente {component_id} no está aislado, no se requiere recuperación")
            return True
            
        # Verificar si el componente sigue registrado
        if component_id not in self.engine.components:
            logger.warning(f"Componente {component_id} no está registrado, eliminando de aislados")
            self.isolated_components.discard(component_id)
            return False
            
        logger.info(f"Intentando recuperar componente {component_id}")
        
        try:
            # Verificar estado actual
            result = await self._check_component_health(component_id)
            healthy = result.get("healthy", False)
            
            if healthy:
                # El componente está sano, recuperarlo
                self.isolated_components.discard(component_id)
                self.failure_counts[component_id] = 0
                self.recovery_events += 1
                
                logger.info(f"Componente {component_id} recuperado exitosamente")
                
                # Notificar recuperación
                await self.event_bus.emit(
                    "system.component.recovered",
                    {
                        "component_id": component_id,
                        "timestamp": time.time()
                    },
                    self.name
                )
                
                # Notificar a los componentes dependientes
                await self.event_bus.emit(
                    "dependency_status_change",
                    {
                        "dependency_id": component_id,
                        "status": True,
                        "reason": "component_recovered"
                    },
                    self.name
                )
                
                return True
            else:
                logger.info(f"Componente {component_id} sigue no saludable, permanece aislado")
                return False
                
        except Exception as e:
            logger.error(f"Error al intentar recuperar componente {component_id}: {e}")
            return False
            
    def get_status_report(self) -> Dict[str, Any]:
        """
        Generar un informe detallado del estado del sistema.
        
        Returns:
            Diccionario con el informe detallado
        """
        return {
            "monitor": self.name,
            "timestamp": time.time(),
            "components": {
                "total": len(self.health_status),
                "healthy": sum(1 for status in self.health_status.values() if status),
                "unhealthy": sum(1 for status in self.health_status.values() if not status),
                "isolated": len(self.isolated_components)
            },
            "events": {
                "isolation_events": self.isolation_events,
                "recovery_events": self.recovery_events
            },
            "status": {
                component_id: {
                    "healthy": status,
                    "isolated": component_id in self.isolated_components,
                    "failure_count": self.failure_counts.get(component_id, 0)
                }
                for component_id, status in self.health_status.items()
            }
        }