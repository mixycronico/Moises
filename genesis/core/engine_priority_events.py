"""
Motor con manejo de eventos prioritarios.

Este módulo implementa un motor que procesa eventos basados
en niveles de prioridad, esencial para sistemas de trading
donde algunos eventos son más críticos que otros.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Set
from enum import IntEnum
import heapq

from genesis.core.component import Component

# Configurar logging
logger = logging.getLogger(__name__)

class EventPriority(IntEnum):
    """Niveles de prioridad para eventos."""
    CRITICAL = 0   # Inmediato (ej: límites de riesgo, stop-loss)
    HIGH = 1       # Alta prioridad (ej: señales de trading, órdenes)
    MEDIUM = 2     # Prioridad media (ej: actualización de datos de mercado)
    LOW = 3        # Baja prioridad (ej: eventos informativos, métricas)
    BACKGROUND = 4 # Proceso en segundo plano (ej: limpieza, logging)


class PriorityEvent:
    """Evento con prioridad para procesamiento."""
    
    def __init__(self, 
                priority: EventPriority,
                event_type: str,
                data: Dict[str, Any],
                source: str,
                created_at: float):
        """
        Inicializar evento prioritario.
        
        Args:
            priority: Nivel de prioridad del evento
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            created_at: Timestamp de creación
        """
        self.priority = priority
        self.event_type = event_type
        self.data = data
        self.source = source
        self.created_at = created_at
        
    def __lt__(self, other):
        """Comparar eventos para ordenamiento en cola de prioridades."""
        if self.priority != other.priority:
            # Menor número = mayor prioridad
            return self.priority < other.priority
        # A igual prioridad, se procesa primero el más antiguo
        return self.created_at < other.created_at


class PriorityEventEngine:
    """
    Motor asíncrono con manejo de eventos prioritarios.
    
    Esta implementación ordena y procesa eventos basados en su
    nivel de prioridad, crucial para sistemas de trading donde
    algunos eventos son más importantes que otros.
    """
    
    def __init__(self, 
                component_timeout: float = 0.5,
                max_queue_size: int = 1000,
                priority_mappings: Optional[Dict[str, EventPriority]] = None):
        """
        Inicializar motor con eventos prioritarios.
        
        Args:
            component_timeout: Tiempo máximo para operaciones de componentes
            max_queue_size: Tamaño máximo de la cola de eventos
            priority_mappings: Mapeo de tipos de eventos a niveles de prioridad
        """
        self.component_timeout = component_timeout
        self.max_queue_size = max_queue_size
        self.priority_mappings = priority_mappings or {}
        
        # Estado del motor
        self.components = {}
        self.running = False
        self.event_queue = []  # Cola de prioridad (heap)
        self.queue_lock = asyncio.Lock()
        self.event_processor_task = None
        
        # Contadores para estadísticas
        self.processed_events = 0
        self.dropped_events = 0
        self.timeout_events = 0
        
        logger.info(f"Motor de eventos prioritarios creado: timeout={component_timeout}s, "
                   f"max_queue={max_queue_size}")
    
    def register_component(self, component: Component) -> None:
        """
        Registrar componente.
        
        Args:
            component: Componente a registrar
        """
        self.components[component.name] = component
        logger.info(f"Componente {component.name} registrado")
    
    def get_priority_for_event(self, event_type: str) -> EventPriority:
        """
        Determinar prioridad para un tipo de evento.
        
        Args:
            event_type: Tipo de evento
            
        Returns:
            Nivel de prioridad para el evento
        """
        # Buscar en mapeos específicos primero
        for pattern, priority in self.priority_mappings.items():
            if pattern in event_type:
                return priority
        
        # Prioridades por defecto basadas en prefijos comunes
        if event_type.startswith("risk."):
            return EventPriority.CRITICAL
        elif event_type.startswith("trade."):
            return EventPriority.HIGH
        elif event_type.startswith("market."):
            return EventPriority.MEDIUM
        elif event_type.startswith("info."):
            return EventPriority.LOW
        else:
            return EventPriority.MEDIUM  # Prioridad por defecto
    
    async def start(self) -> None:
        """Iniciar motor y componentes."""
        if self.running:
            logger.warning("Motor ya está en ejecución")
            return
        
        logger.info("Iniciando motor de eventos prioritarios")
        
        # Iniciar componentes con timeout
        for name, component in self.components.items():
            try:
                task = asyncio.create_task(component.start())
                try:
                    await asyncio.wait_for(task, timeout=self.component_timeout)
                    logger.info(f"Componente {name} iniciado")
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout al iniciar componente {name}")
                    if not task.done() and not task.cancelled():
                        task.cancel()
                except Exception as e:
                    logger.error(f"Error al iniciar componente {name}: {str(e)}")
            except Exception as e:
                logger.error(f"Error general al iniciar componente {name}: {str(e)}")
        
        # Iniciar procesador de eventos en segundo plano
        self.running = True
        self.event_processor_task = asyncio.create_task(self._process_event_queue())
        logger.info("Motor de eventos prioritarios iniciado")
    
    async def stop(self) -> None:
        """Detener motor y componentes."""
        if not self.running:
            logger.warning("Motor ya está detenido")
            return
        
        logger.info("Deteniendo motor de eventos prioritarios")
        
        # Detener procesador de eventos
        self.running = False
        if self.event_processor_task and not self.event_processor_task.done():
            self.event_processor_task.cancel()
            try:
                await self.event_processor_task
            except asyncio.CancelledError:
                pass
        
        # Detener componentes con timeout
        for name, component in self.components.items():
            try:
                task = asyncio.create_task(component.stop())
                try:
                    await asyncio.wait_for(task, timeout=self.component_timeout)
                    logger.info(f"Componente {name} detenido")
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout al detener componente {name}")
                    if not task.done() and not task.cancelled():
                        task.cancel()
                except Exception as e:
                    logger.error(f"Error al detener componente {name}: {str(e)}")
            except Exception as e:
                logger.error(f"Error general al detener componente {name}: {str(e)}")
        
        logger.info("Motor de eventos prioritarios detenido")
    
    async def emit_event(self, event_type: str, data: Optional[Dict[str, Any]] = None, 
                       source: str = "system", priority: Optional[EventPriority] = None) -> bool:
        """
        Emitir evento con prioridad específica o automática.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            priority: Prioridad del evento (opcional, se determina automáticamente si no se proporciona)
            
        Returns:
            True si el evento se aceptó, False si se descartó
        """
        if not self.running:
            logger.warning(f"Intento de emitir evento {event_type} con motor detenido")
            return False
        
        # Determinar prioridad (usar la proporcionada o inferir automáticamente)
        event_priority = priority if priority is not None else self.get_priority_for_event(event_type)
        
        event_data = data or {}
        now = time.time()
        
        # Crear objeto de evento prioritario
        event = PriorityEvent(
            priority=event_priority,
            event_type=event_type,
            data=event_data,
            source=source,
            created_at=now
        )
        
        # Agregar evento a la cola con prioridad
        async with self.queue_lock:
            if len(self.event_queue) >= self.max_queue_size:
                # Si la cola está llena, sólo descartar eventos de baja prioridad
                if event_priority >= EventPriority.LOW:
                    self.dropped_events += 1
                    logger.warning(f"Cola llena, descartando evento de baja prioridad: {event_type}")
                    return False
                else:
                    # Para eventos importantes, eliminar el evento de menor prioridad
                    # para hacer espacio
                    heapq.heappush(self.event_queue, event)
                    lowest_priority_event = heapq.heappop(self.event_queue)
                    self.dropped_events += 1
                    logger.warning(f"Cola llena, eliminando evento de menor prioridad para insertar: {event_type}")
            else:
                # Agregar a la cola si hay espacio
                heapq.heappush(self.event_queue, event)
                
            priority_name = event_priority.name if hasattr(event_priority, 'name') else str(event_priority)
            logger.info(f"Evento {event_type} añadido a la cola (prioridad: {priority_name}, cola: {len(self.event_queue)})")
        
        return True
    
    async def _process_event_queue(self) -> None:
        """
        Procesar continuamente la cola de eventos en segundo plano,
        ordenando por prioridad.
        """
        logger.info("Iniciando procesador de cola de eventos prioritarios")
        
        try:
            while self.running:
                event = None
                
                # Obtener el evento de mayor prioridad
                async with self.queue_lock:
                    if self.event_queue:
                        event = heapq.heappop(self.event_queue)
                
                if event:
                    priority_name = event.priority.name if hasattr(event.priority, 'name') else str(event.priority)
                    logger.debug(f"Procesando evento {event.event_type} (prioridad: {priority_name})")
                    
                    # Enviar evento a todos los componentes
                    for name, component in self.components.items():
                        try:
                            task = asyncio.create_task(
                                component.handle_event(
                                    event.event_type, event.data, event.source
                                )
                            )
                            try:
                                await asyncio.wait_for(task, timeout=self.component_timeout)
                                logger.debug(f"Componente {name} procesó evento {event.event_type}")
                            except asyncio.TimeoutError:
                                logger.warning(f"Timeout al procesar evento {event.event_type} en componente {name}")
                                self.timeout_events += 1
                                if not task.done() and not task.cancelled():
                                    task.cancel()
                            except Exception as e:
                                logger.error(f"Error al procesar evento {event.event_type} en componente {name}: {str(e)}")
                        except Exception as e:
                            logger.error(f"Error general al procesar evento {event.event_type}: {str(e)}")
                    
                    self.processed_events += 1
                    
                    # Calcular tiempo de procesamiento total
                    processing_time = time.time() - event.created_at
                    logger.debug(f"Evento {event.event_type} procesado en {processing_time:.3f}s")
                else:
                    # Si no hay eventos, esperar un poco
                    await asyncio.sleep(0.01)
                    
        except asyncio.CancelledError:
            logger.info("Procesador de cola de eventos cancelado")
        except Exception as e:
            logger.error(f"Error en procesador de cola de eventos: {str(e)}")
            # Reintentar si el motor sigue en ejecución
            if self.running:
                logger.info("Reiniciando procesador de cola de eventos")
                self.event_processor_task = asyncio.create_task(self._process_event_queue())
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del motor.
        
        Returns:
            Diccionario con estadísticas
        """
        queue_size = 0
        async def get_queue_size():
            nonlocal queue_size
            async with self.queue_lock:
                queue_size = len(self.event_queue)
        
        # Ejecutar de forma síncrona
        loop = asyncio.get_event_loop()
        if loop.is_running():
            future = asyncio.run_coroutine_threadsafe(get_queue_size(), loop)
            future.result()  # Esperar el resultado
        
        return {
            "processed_events": self.processed_events,
            "dropped_events": self.dropped_events,
            "timeout_events": self.timeout_events,
            "queue_size": queue_size,
            "running": self.running
        }