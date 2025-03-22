"""
Motor con colas dedicadas para el sistema Genesis.

Este módulo implementa un motor que utiliza el bus de eventos con colas dedicadas
para evitar deadlocks y mejorar la escalabilidad del sistema.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Set, Optional, Union, Tuple

from genesis.core.base import Component
from genesis.core.event_bus_dedicated_queues import DedicatedQueueEventBus

# Configuración del logger
logger = logging.getLogger(__name__)

class DedicatedQueueEngine:
    """
    Motor de procesamiento con colas dedicadas por componente.
    
    Esta implementación:
    1. Utiliza un bus de eventos con colas dedicadas
    2. Implementa monitoreo de componentes
    3. Mantiene estadísticas de rendimiento
    4. Incluye mecanismos de recuperación de componentes
    """
    
    def __init__(self, test_mode: bool = False):
        """
        Inicializar el motor.
        
        Args:
            test_mode: Si es True, usa timeouts más agresivos para pruebas
        """
        self.event_bus = DedicatedQueueEventBus(test_mode=test_mode)
        self.components: Dict[str, Component] = {}
        self.running = False
        self.monitor_task = None
        self.test_mode = test_mode
        
        # Estadísticas
        self.start_time = 0
        self.component_stats: Dict[str, Dict[str, int]] = {}
        
        # Registro de errores
        self.component_errors: Dict[str, List[Tuple[float, str]]] = {}
        
        # Callbacks
        self.status_callbacks = []
        
    async def register_component(self, component: Component) -> None:
        """
        Registrar un componente en el motor.
        
        Args:
            component: Componente a registrar
        """
        if component.name in self.components:
            logger.warning(f"Componente {component.name} ya registrado, actualizando referencia")
            
        logger.info(f"Registrando componente {component.name}")
        
        # Almacenar referencia al componente
        self.components[component.name] = component
        
        # Conectar el bus de eventos al componente
        component.attach_event_bus(self.event_bus)
        
        # Registrar el componente en el bus de eventos
        self.event_bus.attach_component(component.name, component)
        
        # Inicializar estadísticas
        self.component_stats[component.name] = {
            "events_processed": 0,
            "errors": 0,
            "warnings": 0
        }
        
        # Inicializar registro de errores
        self.component_errors[component.name] = []
        
        # Notificar a los callbacks
        for callback in self.status_callbacks:
            await callback("component.registered", {
                "component": component.name,
                "timestamp": time.time()
            })
        
    async def remove_component(self, component_name: str) -> bool:
        """
        Eliminar un componente del motor.
        
        Args:
            component_name: Nombre del componente a eliminar
            
        Returns:
            True si se eliminó correctamente, False si no se encontró
        """
        if component_name not in self.components:
            logger.warning(f"Componente {component_name} no encontrado para eliminar")
            return False
            
        logger.info(f"Eliminando componente {component_name}")
        
        # Obtener referencia al componente
        component = self.components[component_name]
        
        # Detener el componente si está en ejecución
        if component.running:
            try:
                await asyncio.wait_for(component.stop(), timeout=1.0)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout al detener componente {component_name}")
            except Exception as e:
                logger.error(f"Error al detener componente {component_name}: {e}")
        
        # Eliminar del diccionario de componentes
        del self.components[component_name]
        
        # Notificar a los callbacks
        for callback in self.status_callbacks:
            await callback("component.removed", {
                "component": component_name,
                "timestamp": time.time()
            })
            
        return True
        
    async def start(self) -> None:
        """Iniciar el motor y todos los componentes registrados."""
        if self.running:
            logger.warning("Motor ya en ejecución")
            return
            
        logger.info("Iniciando motor")
        self.running = True
        self.start_time = time.time()
        
        # Iniciar el bus de eventos
        await self.event_bus.start_monitoring()
        
        # Iniciar el monitor de componentes
        self.monitor_task = asyncio.create_task(self._monitor_components())
        
        # Iniciar todos los componentes
        for name, component in list(self.components.items()):
            try:
                logger.debug(f"Iniciando componente {name}")
                await asyncio.wait_for(component.start(), timeout=2.0)
                logger.debug(f"Componente {name} iniciado")
            except asyncio.TimeoutError:
                logger.error(f"Timeout al iniciar componente {name}")
                self._register_component_error(name, "Timeout al iniciar")
            except Exception as e:
                logger.error(f"Error al iniciar componente {name}: {e}")
                self._register_component_error(name, f"Error al iniciar: {e}")
                
        logger.info(f"Motor iniciado con {len(self.components)} componentes")
        
    async def stop(self) -> None:
        """Detener el motor y todos los componentes registrados."""
        if not self.running:
            logger.warning("Motor ya detenido")
            return
            
        logger.info("Deteniendo motor")
        self.running = False
        
        # Detener el monitor
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
                
        # Detener todos los componentes en orden inverso de registro
        for name, component in reversed(list(self.components.items())):
            try:
                logger.debug(f"Deteniendo componente {name}")
                await asyncio.wait_for(component.stop(), timeout=2.0)
                logger.debug(f"Componente {name} detenido")
            except asyncio.TimeoutError:
                logger.error(f"Timeout al detener componente {name}")
            except Exception as e:
                logger.error(f"Error al detener componente {name}: {e}")
                
        # Detener el bus de eventos
        await self.event_bus.stop()
        
        logger.info("Motor detenido")
        
    async def _monitor_components(self) -> None:
        """Monitorear el estado de los componentes."""
        logger.info("Iniciando monitoreo de componentes")
        
        while self.running:
            try:
                # Verificar componentes
                for name, component in list(self.components.items()):
                    if not component.running and self.running:
                        logger.warning(f"Componente {name} no está en ejecución, intentando reiniciar")
                        try:
                            await asyncio.wait_for(component.start(), timeout=2.0)
                            logger.info(f"Componente {name} reiniciado")
                        except Exception as e:
                            logger.error(f"Error al reiniciar componente {name}: {e}")
                            self._register_component_error(name, f"Error al reiniciar: {e}")
                            
                # Verificar estado del motor
                uptime = time.time() - self.start_time
                if uptime > 3600 and uptime % 3600 < 10:  # Cada hora
                    logger.info(f"Motor en ejecución durante {uptime/3600:.1f} horas")
                    self._log_component_stats()
                    
                # Esperar antes de siguiente verificación
                await asyncio.sleep(5.0 if not self.test_mode else 1.0)
                
            except asyncio.CancelledError:
                logger.info("Monitoreo de componentes cancelado")
                break
                
            except Exception as e:
                logger.error(f"Error en monitoreo de componentes: {e}")
                await asyncio.sleep(1.0)
                
        logger.info("Monitoreo de componentes finalizado")
        
    def _register_component_error(self, component_name: str, error_msg: str) -> None:
        """
        Registrar un error de componente.
        
        Args:
            component_name: Nombre del componente
            error_msg: Mensaje de error
        """
        if component_name in self.component_stats:
            self.component_stats[component_name]["errors"] += 1
            
        if component_name in self.component_errors:
            # Limitar a 100 errores por componente
            if len(self.component_errors[component_name]) >= 100:
                self.component_errors[component_name].pop(0)
                
            self.component_errors[component_name].append((time.time(), error_msg))
            
    def _log_component_stats(self) -> None:
        """Registrar estadísticas de componentes en el log."""
        total_events = sum(stats["events_processed"] for stats in self.component_stats.values())
        total_errors = sum(stats["errors"] for stats in self.component_stats.values())
        
        logger.info(f"Estadísticas del motor: {len(self.components)} componentes, "
                   f"{total_events} eventos procesados, {total_errors} errores")
                   
        # Componentes con más errores
        error_components = sorted(
            [(name, stats["errors"]) for name, stats in self.component_stats.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        if error_components and error_components[0][1] > 0:
            top_errors = error_components[:3]
            logger.warning(f"Componentes con más errores: " + 
                          ", ".join([f"{name}: {errors}" for name, errors in top_errors]))
                          
    def register_status_callback(self, callback) -> None:
        """
        Registrar un callback para notificaciones de estado.
        
        Args:
            callback: Función asíncrona que recibe (event_type, data)
        """
        self.status_callbacks.append(callback)
        
    async def get_component_status(self) -> Dict[str, Any]:
        """
        Obtener estado de todos los componentes.
        
        Returns:
            Diccionario con estado de componentes
        """
        status = {
            "total_components": len(self.components),
            "running_components": sum(1 for comp in self.components.values() if comp.running),
            "uptime": time.time() - self.start_time,
            "components": {}
        }
        
        # Estado por componente
        for name, component in self.components.items():
            status["components"][name] = {
                "running": component.running,
                "errors": self.component_stats.get(name, {}).get("errors", 0),
                "events_processed": self.component_stats.get(name, {}).get("events_processed", 0)
            }
            
        return status