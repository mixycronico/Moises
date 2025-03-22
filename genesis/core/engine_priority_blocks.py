"""
Motor con bloques priorizados y componente seguro.

Este módulo implementa un motor que combina procesamiento en bloques
con priorización de eventos, incluyendo un componente seguro entre bloques
para operaciones críticas.
"""

import asyncio
import logging
import time
import heapq
from typing import Dict, List, Any, Optional, Set, Tuple

from genesis.core.component import Component

# Configurar logging
logger = logging.getLogger(__name__)

# Función auxiliar para garantizar que los parámetros nunca sean None
def safe_handle_event_params(event_type, event_data, source):
    """
    Asegura que los parámetros para handle_event nunca sean None.
    
    Args:
        event_type: Tipo de evento o None
        event_data: Datos del evento o None
        source: Fuente del evento o None
        
    Returns:
        Tuple con valores seguros (nunca None)
    """
    return (
        event_type or "",
        event_data or {},
        source or ""
    )

class EventPriority:
    """Niveles de prioridad para eventos."""
    CRITICAL = 0   # Inmediato (stop-loss, límites de riesgo)
    HIGH = 1       # Alta prioridad (señales de trading, órdenes)
    MEDIUM = 2     # Media prioridad (actualización de datos)
    LOW = 3        # Baja prioridad (eventos informativos)
    BACKGROUND = 4 # Procesos en segundo plano (limpieza, logs)


class PriorityBlock:
    """Bloque de componentes con prioridad."""
    
    def __init__(self, 
                priority: int, 
                components: List[Tuple[str, Component]],
                is_safe_block: bool = False):
        """
        Inicializar bloque con prioridad.
        
        Args:
            priority: Nivel de prioridad del bloque
            components: Lista de tuplas (nombre, componente)
            is_safe_block: Si es un bloque de seguridad
        """
        self.priority = priority
        self.components = components
        self.is_safe_block = is_safe_block
        
    def __lt__(self, other):
        """Comparar bloques para ordenamiento."""
        # Los bloques seguros siempre tienen prioridad
        if self.is_safe_block != other.is_safe_block:
            return self.is_safe_block
        # Ordenar por prioridad
        return self.priority < other.priority


class PriorityBlockEngine:
    """
    Motor con bloques priorizados y componente seguro entre bloques.
    
    Esta implementación combina procesamiento en bloques con
    priorización de eventos, e incluye un bloque seguro entre
    bloques regulares para operaciones críticas.
    """
    
    def __init__(self, 
                block_size: int = 3,
                timeout: float = 0.5,
                max_concurrent_blocks: int = 2,
                priority_mappings: Optional[Dict[str, int]] = None):
        """
        Inicializar motor con bloques priorizados.
        
        Args:
            block_size: Tamaño de cada bloque de componentes
            timeout: Tiempo máximo para operaciones
            max_concurrent_blocks: Número máximo de bloques concurrentes
            priority_mappings: Mapeos de tipos de eventos a prioridades
        """
        self.block_size = block_size
        self.timeout = timeout
        self.max_concurrent_blocks = max_concurrent_blocks
        self.priority_mappings = priority_mappings or {}
        
        # Componentes y estado
        self.components = {}
        self.safe_components = set()  # Nombres de componentes seguros
        self.paused_components = set()  # Componentes pausados temporalmente
        self.running = False
        
        # Semáforo para controlar concurrencia
        self.semaphore = asyncio.Semaphore(max_concurrent_blocks)
        
        # Estadísticas
        self.processed_blocks = 0
        self.timeout_blocks = 0
        self.isolation_events = 0
        
        # Monitor de componentes (se inicializa si es necesario)
        self.component_monitor = None
        
        logger.info(f"Motor de bloques priorizados creado: "
                   f"block_size={block_size}, timeout={timeout}s, "
                   f"max_concurrent={max_concurrent_blocks}")
    
    def register_component(self, component: Component, safe: bool = False) -> None:
        """
        Registrar componente en el motor.
        
        Args:
            component: Componente a registrar
            safe: Si es un componente seguro (procesado entre bloques)
        """
        self.components[component.name] = component
        if safe:
            self.safe_components.add(component.name)
            logger.info(f"Componente SEGURO {component.name} registrado")
        else:
            logger.info(f"Componente {component.name} registrado")
    
    def get_priority_for_event(self, event_type: str) -> int:
        """
        Determinar prioridad para un tipo de evento.
        
        Args:
            event_type: Tipo de evento
            
        Returns:
            Nivel de prioridad para el evento
        """
        # Buscar en mapeos específicos
        for pattern, priority in self.priority_mappings.items():
            if pattern in event_type:
                return priority
        
        # Prioridades por prefijo
        if event_type.startswith("risk."):
            return EventPriority.CRITICAL
        elif event_type.startswith("trade."):
            return EventPriority.HIGH
        elif event_type.startswith("market."):
            return EventPriority.MEDIUM
        elif event_type.startswith("info."):
            return EventPriority.LOW
        else:
            return EventPriority.MEDIUM
    
    def _create_priority_blocks(self) -> List[PriorityBlock]:
        """
        Crear bloques de componentes priorizados.
        
        Returns:
            Lista de bloques priorizados
        """
        # Separar componentes seguros y regulares
        safe_items = []
        regular_items = []
        
        for name, comp in self.components.items():
            if name in self.safe_components:
                safe_items.append((name, comp))
            else:
                regular_items.append((name, comp))
        
        # Crear bloque seguro si hay componentes seguros
        blocks = []
        if safe_items:
            safe_block = PriorityBlock(
                priority=0,  # Máxima prioridad
                components=safe_items,
                is_safe_block=True
            )
            blocks.append(safe_block)
        
        # Crear bloques regulares
        for i in range(0, len(regular_items), self.block_size):
            block_components = regular_items[i:i + self.block_size]
            # Asignar prioridad basada en la posición
            priority = (i // self.block_size) + 1  # Empezar en 1 después del bloque seguro
            block = PriorityBlock(
                priority=priority,
                components=block_components,
                is_safe_block=False
            )
            blocks.append(block)
        
        return blocks
    
    async def _process_block(self, 
                            block: PriorityBlock, 
                            operation: str,
                            event_type: Optional[str] = None,
                            event_data: Optional[Dict[str, Any]] = None,
                            event_source: Optional[str] = None) -> None:
        """
        Procesar un bloque de componentes.
        
        Args:
            block: Bloque a procesar
            operation: Operación a realizar ('start', 'stop', 'handle_event')
            event_type: Tipo de evento (solo para 'handle_event')
            event_data: Datos del evento (solo para 'handle_event')
            event_source: Fuente del evento (solo para 'handle_event')
        """
        # Si es bloque seguro, procesar secuencialmente con alta prioridad
        if block.is_safe_block:
            for name, component in block.components:
                # Verificar si el componente está pausado (excepto para 'stop' que siempre se procesa)
                if name in self.paused_components and operation != 'stop':
                    logger.debug(f"Saltando componente SEGURO pausado {name} ({operation})")
                    continue
                    
                try:
                    logger.info(f"Procesando componente SEGURO {name} ({operation})")
                    if operation == 'start':
                        await asyncio.wait_for(component.start(), timeout=self.timeout)
                    elif operation == 'stop':
                        await asyncio.wait_for(component.stop(), timeout=self.timeout)
                    elif operation == 'handle_event':
                        # Verificar si es un evento de sistema para componentes pausados
                        if event_type and event_type in ["component_paused", "component_resumed"]:
                            # Eventos de sistema de componentes se entregan a todos
                            pass
                        elif name in self.paused_components:
                            # Saltar procesamiento para componentes pausados
                            continue
                            
                        # Asegurar que los parámetros nunca sean None
                        evt_type, evt_data, evt_source = safe_handle_event_params(event_type, event_data, event_source)
                        await asyncio.wait_for(
                            component.handle_event(evt_type, evt_data, evt_source),
                            timeout=self.timeout
                        )
                    logger.info(f"Componente SEGURO {name} procesado exitosamente ({operation})")
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout en componente SEGURO {name} ({operation})")
                    # Considerar pausar componentes con timeout en eventos críticos
                    if operation == 'handle_event' and event_type and event_type.startswith("risk."):
                        await self.pause_component(name)
                except Exception as e:
                    logger.error(f"Error en componente SEGURO {name} ({operation}): {str(e)}")
                    # Considerar pausar componentes con errores en eventos críticos
                    if operation == 'handle_event' and event_type and event_type.startswith("risk."):
                        await self.pause_component(name)
        else:
            # Para bloques regulares, procesar con semáforo de concurrencia
            async with self.semaphore:
                logger.info(f"Procesando bloque {block.priority} con {len(block.components)} componentes")
                
                # Crear tareas para todos los componentes en el bloque que no están pausados
                tasks = []
                # Diccionario para almacenar metadatos de las tareas
                task_metadata = {}
                
                for name, component in block.components:
                    # Verificar si el componente está pausado (excepto para 'stop' que siempre se procesa)
                    if name in self.paused_components and operation != 'stop':
                        logger.debug(f"Saltando componente pausado {name} ({operation})")
                        continue
                        
                    # Verificar si es un evento de sistema para componentes pausados
                    if (operation == 'handle_event' and event_type and 
                        not (event_type in ["component_paused", "component_resumed"]) and
                        name in self.paused_components):
                        # Saltar procesamiento para componentes pausados
                        continue
                        
                    task = None
                    if operation == 'start':
                        task = asyncio.create_task(component.start())
                    elif operation == 'stop':
                        task = asyncio.create_task(component.stop())
                    elif operation == 'handle_event':
                        # Asegurar que los parámetros nunca sean None
                        evt_type, evt_data, evt_source = safe_handle_event_params(event_type, event_data, event_source)
                        task = asyncio.create_task(
                            component.handle_event(evt_type, evt_data, evt_source)
                        )
                    
                    # Solo agregar la tarea si se creó correctamente
                    if task is not None:
                        # Guardar metadata en diccionario usando id de la tarea como clave
                        task_metadata[id(task)] = {
                            'component_name': name,
                            'operation': operation
                        }
                        tasks.append(task)
                
                # Verificar si hay tareas para procesar
                if not tasks:
                    logger.info(f"No hay componentes activos en el bloque {block.priority} para procesar")
                    return
                
                # Esperar a que todas las tareas completen con timeout
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=self.timeout * len(tasks)  # Timeout proporcional al número de tareas
                    )
                    self.processed_blocks += 1
                    logger.info(f"Bloque {block.priority} procesado exitosamente")
                except asyncio.TimeoutError:
                    self.timeout_blocks += 1
                    logger.warning(f"Timeout en bloque {block.priority}")
                    # Cancelar tareas pendientes
                    for task in tasks:
                        if not task.done() and not task.cancelled():
                            task.cancel()
                
                # Verificar resultados individuales
                for task in tasks:
                    task_id = id(task)
                    task_meta = task_metadata.get(task_id, {})
                    name = task_meta.get('component_name', 'unknown')
                    op = task_meta.get('operation', 'unknown')
                    
                    if task.done() and not task.cancelled():
                        try:
                            task.result()  # Esto re-lanzará cualquier excepción
                            logger.debug(f"Tarea exitosa: {name} ({op})")
                        except Exception as e:
                            logger.error(f"Error en tarea: {name} ({op}): {str(e)}")
                            # Considerar pausar componentes con errores en eventos críticos
                            if (op == 'handle_event' and event_type and 
                                (event_type.startswith("risk.") or 
                                 event_type.startswith("trade.") or
                                 event_type.startswith("critical."))):
                                await self.pause_component(name)
    
    async def start(self) -> None:
        """Iniciar motor y componentes en bloques priorizados."""
        if self.running:
            logger.warning("Motor ya está en ejecución")
            return
        
        logger.info("Iniciando motor de bloques priorizados")
        
        # Crear bloques de componentes
        blocks = self._create_priority_blocks()
        logger.info(f"Creados {len(blocks)} bloques para inicio ({len(self.components)} componentes)")
        
        # Primero procesar bloque seguro (si existe)
        safe_blocks = [b for b in blocks if b.is_safe_block]
        regular_blocks = [b for b in blocks if not b.is_safe_block]
        
        # Procesar bloques seguros primero
        for block in safe_blocks:
            await self._process_block(block, 'start')
        
        # Luego procesar bloques regulares en paralelo
        start_tasks = []
        for block in regular_blocks:
            task = asyncio.create_task(
                self._process_block(block, 'start')
            )
            start_tasks.append(task)
        
        # Esperar a que todos los bloques regulares completen
        if start_tasks:
            await asyncio.gather(*start_tasks, return_exceptions=True)
        
        self.running = True
        logger.info("Motor de bloques priorizados iniciado")
    
    async def stop(self) -> None:
        """Detener motor y componentes en bloques priorizados."""
        if not self.running:
            logger.warning("Motor ya está detenido")
            return
        
        logger.info("Deteniendo motor de bloques priorizados")
        
        # Crear bloques de componentes
        blocks = self._create_priority_blocks()
        logger.info(f"Creados {len(blocks)} bloques para detención")
        
        # Procesar bloques regulares primero (inverso al inicio)
        regular_blocks = [b for b in blocks if not b.is_safe_block]
        safe_blocks = [b for b in blocks if b.is_safe_block]
        
        # Detener bloques regulares en paralelo
        stop_tasks = []
        for block in regular_blocks:
            task = asyncio.create_task(
                self._process_block(block, 'stop')
            )
            stop_tasks.append(task)
        
        # Esperar a que completen
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        # Finalmente detener bloques seguros
        for block in safe_blocks:
            await self._process_block(block, 'stop')
        
        self.running = False
        logger.info("Motor de bloques priorizados detenido")
    
    async def emit_event(self, event_type: str, data: Optional[Dict[str, Any]] = None, 
                       source: str = "system", priority: Optional[int] = None) -> None:
        """
        Emitir evento a componentes en bloques priorizados.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            priority: Prioridad opcional (se infiere si no se proporciona)
        """
        if not self.running:
            logger.warning(f"Intento de emitir evento {event_type} con motor detenido")
            return
        
        # Determinar prioridad
        event_priority = priority if priority is not None else self.get_priority_for_event(event_type)
        logger.info(f"Emitiendo evento {event_type} con prioridad {event_priority}")
        
        event_data = data or {}
        
        # Crear bloques y ordenarlos por prioridad
        blocks = self._create_priority_blocks()
        
        # Procesar bloque seguro primero si existe
        safe_blocks = [b for b in blocks if b.is_safe_block]
        regular_blocks = [b for b in blocks if not b.is_safe_block]
        
        # Procesar bloques seguros secuencialmente
        for block in safe_blocks:
            await self._process_block(block, 'handle_event', event_type, event_data, source)
        
        # Procesar bloques regulares con prioridad
        # Si el evento es crítico, procesar todos los bloques
        # Para otros eventos, limitar según prioridad
        process_blocks = []
        
        if event_priority == EventPriority.CRITICAL:
            # Eventos críticos van a todos los bloques
            process_blocks = regular_blocks
        elif event_priority == EventPriority.HIGH:
            # Eventos de alta prioridad van a los primeros N bloques
            process_blocks = regular_blocks[:max(1, len(regular_blocks) // 2)]
        elif event_priority == EventPriority.MEDIUM:
            # Eventos de media prioridad van a algunos bloques
            process_blocks = regular_blocks[:max(1, len(regular_blocks) // 3)]
        else:
            # Eventos de baja prioridad solo van al primer bloque
            if regular_blocks:
                process_blocks = [regular_blocks[0]]
        
        # Procesar bloques seleccionados en paralelo
        event_tasks = []
        for block in process_blocks:
            task = asyncio.create_task(
                self._process_block(block, 'handle_event', event_type, event_data, source)
            )
            event_tasks.append(task)
        
        # Esperar a que completen
        if event_tasks:
            await asyncio.gather(*event_tasks, return_exceptions=True)
        
        logger.info(f"Evento {event_type} procesado por {len(process_blocks)} bloques")
    
    async def pause_component(self, component_id: str) -> bool:
        """
        Pausar temporalmente un componente para evitar que procese eventos.
        
        Esta función aisla un componente problemático permitiendo que el
        resto del sistema siga funcionando normalmente.
        
        Args:
            component_id: Identificador del componente a pausar
            
        Returns:
            True si el componente fue pausado, False en caso contrario
        """
        if component_id not in self.components:
            logger.warning(f"Intento de pausar componente inexistente: {component_id}")
            return False
            
        if component_id in self.paused_components:
            logger.info(f"Componente {component_id} ya estaba pausado")
            return True
            
        logger.warning(f"Pausando componente: {component_id}")
        self.paused_components.add(component_id)
        self.isolation_events += 1
        
        # Notificar al componente que está siendo pausado
        try:
            component = self.components[component_id]
            if hasattr(component, "on_pause"):
                await asyncio.wait_for(
                    component.on_pause(),
                    timeout=self.timeout
                )
                logger.info(f"Método on_pause ejecutado para {component_id}")
        except Exception as e:
            logger.error(f"Error al pausar componente {component_id}: {e}")
            
        # Notificar a otros componentes que este ha sido pausado
        await self.emit_event(
            "component_paused", 
            {"component_id": component_id},
            "system",
            EventPriority.HIGH  # Alta prioridad para este tipo de eventos
        )
        
        return True
        
    async def resume_component(self, component_id: str) -> bool:
        """
        Reanudar un componente previamente pausado.
        
        Args:
            component_id: Identificador del componente a reanudar
            
        Returns:
            True si el componente fue reanudado, False en caso contrario
        """
        if component_id not in self.components:
            logger.warning(f"Intento de reanudar componente inexistente: {component_id}")
            return False
            
        if component_id not in self.paused_components:
            logger.info(f"Componente {component_id} no estaba pausado")
            return True
            
        logger.info(f"Reanudando componente: {component_id}")
        self.paused_components.remove(component_id)
        
        # Notificar al componente que está siendo reanudado
        try:
            component = self.components[component_id]
            if hasattr(component, "on_resume"):
                await asyncio.wait_for(
                    component.on_resume(),
                    timeout=self.timeout
                )
                logger.info(f"Método on_resume ejecutado para {component_id}")
        except Exception as e:
            logger.error(f"Error al reanudar componente {component_id}: {e}")
            
        # Notificar a otros componentes que este ha sido reanudado
        await self.emit_event(
            "component_resumed", 
            {"component_id": component_id},
            "system",
            EventPriority.HIGH  # Alta prioridad para este tipo de eventos
        )
        
        return True
        
    async def enable_component_monitoring(self, check_interval: float = 10.0) -> None:
        """
        Habilitar el monitoreo automático de componentes para detectar y aislar
        componentes problemáticos.
        
        Args:
            check_interval: Intervalo de verificación en segundos
        """
        # Importar aquí para evitar importación circular
        from genesis.core.component_monitor import ComponentMonitor
        
        if self.component_monitor is None:
            self.component_monitor = ComponentMonitor(self)
            await self.component_monitor.start(check_interval)
            logger.info(f"Monitoreo de componentes habilitado (intervalo: {check_interval}s)")
        else:
            logger.warning("El monitoreo de componentes ya estaba habilitado")
            
    async def disable_component_monitoring(self) -> None:
        """Deshabilitar el monitoreo automático de componentes."""
        if self.component_monitor is not None:
            await self.component_monitor.stop()
            self.component_monitor = None
            logger.info("Monitoreo de componentes deshabilitado")
        else:
            logger.warning("El monitoreo de componentes no estaba habilitado")
            
    def is_component_paused(self, component_id: str) -> bool:
        """
        Verificar si un componente está pausado.
        
        Args:
            component_id: Identificador del componente
            
        Returns:
            True si el componente está pausado, False en caso contrario
        """
        return component_id in self.paused_components
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del motor.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "processed_blocks": self.processed_blocks,
            "timeout_blocks": self.timeout_blocks,
            "components": len(self.components),
            "safe_components": len(self.safe_components),
            "paused_components": len(self.paused_components),
            "isolation_events": self.isolation_events,
            "running": self.running,
            "monitoring_enabled": self.component_monitor is not None
        }