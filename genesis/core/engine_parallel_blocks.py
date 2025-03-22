"""
Motor con procesamiento paralelo en bloques.

Este módulo implementa un motor que procesa eventos en bloques 
paralelos para mejorar el rendimiento mientras mantiene control
sobre timeouts y recursos.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Set

from genesis.core.component import Component

# Configurar logging
logger = logging.getLogger(__name__)

class ParallelBlockEngine:
    """
    Motor asíncrono con procesamiento en bloques paralelos.
    
    Esta implementación procesa eventos en bloques paralelos
    para mejorar el rendimiento mientras controla los recursos.
    """
    
    def __init__(self, 
                block_size: int = 3,
                timeout: float = 0.5,
                max_concurrent_blocks: int = 2):
        """
        Inicializar motor con procesamiento en bloques paralelos.
        
        Args:
            block_size: Número de componentes por bloque para procesamiento paralelo
            timeout: Tiempo máximo (en segundos) para operaciones
            max_concurrent_blocks: Número máximo de bloques a procesar en paralelo
        """
        self.timeout = timeout
        self.block_size = block_size
        self.max_concurrent_blocks = max_concurrent_blocks
        self.components = {}
        self.running = False
        
        # Semáforo para limitar concurrencia
        self.semaphore = asyncio.Semaphore(max_concurrent_blocks)
        
        logger.info(f"Motor de bloques paralelos creado: "
                   f"block_size={block_size}, timeout={timeout}s, "
                   f"max_concurrent_blocks={max_concurrent_blocks}")
    
    def register_component(self, component: Component) -> None:
        """
        Registrar componente.
        
        Args:
            component: Componente a registrar
        """
        self.components[component.name] = component
        logger.info(f"Componente {component.name} registrado")
    
    def _create_component_blocks(self) -> List[List[tuple]]:
        """
        Crear bloques de componentes para procesamiento paralelo.
        
        Returns:
            Lista de bloques, donde cada bloque es una lista de tuplas (nombre, componente)
        """
        items = list(self.components.items())
        blocks = []
        
        for i in range(0, len(items), self.block_size):
            block = items[i:i + self.block_size]
            blocks.append(block)
            
        return blocks
    
    async def _process_component_block(self, 
                                      block: List[tuple], 
                                      operation: str,
                                      event_type: Optional[str] = None,
                                      event_data: Optional[Dict[str, Any]] = None,
                                      event_source: Optional[str] = None) -> None:
        """
        Procesar un bloque de componentes en paralelo.
        
        Args:
            block: Lista de tuplas (nombre, componente) a procesar
            operation: Operación a realizar ('start', 'stop', o 'handle_event')
            event_type: Tipo de evento (solo para 'handle_event')
            event_data: Datos del evento (solo para 'handle_event')
            event_source: Fuente del evento (solo para 'handle_event')
        """
        # Adquirir semáforo para limitar concurrencia
        async with self.semaphore:
            # Crear tareas para todos los componentes en el bloque
            tasks = []
            # Diccionario para almacenar metadatos de las tareas
            task_metadata = {}
            
            for name, component in block:
                if operation == 'start':
                    task = asyncio.create_task(component.start())
                    # Guardar metadata en diccionario usando id de la tarea como clave
                    task_metadata[id(task)] = {
                        'component_name': name,
                        'operation': 'start'
                    }
                elif operation == 'stop':
                    task = asyncio.create_task(component.stop())
                    task_metadata[id(task)] = {
                        'component_name': name,
                        'operation': 'stop'
                    }
                elif operation == 'handle_event':
                    task = asyncio.create_task(
                        component.handle_event(event_type, event_data, event_source)
                    )
                    task_metadata[id(task)] = {
                        'component_name': name,
                        'operation': 'handle_event',
                        'event_type': event_type
                    }
                
                tasks.append(task)
            
            # Procesar todas las tareas con timeout individual
            for task in tasks:
                task_id = id(task)
                task_meta = task_metadata.get(task_id, {})
                
                try:
                    await asyncio.wait_for(task, timeout=self.timeout)
                    name = task_meta.get('component_name', 'unknown')
                    op = task_meta.get('operation', 'unknown')
                    
                    if op == 'handle_event':
                        evt = task_meta.get('event_type', 'unknown')
                        logger.debug(f"Componente {name} procesó evento {evt}")
                    else:
                        logger.info(f"Operación {op} completada en componente {name}")
                        
                except asyncio.TimeoutError:
                    name = task_meta.get('component_name', 'unknown')
                    op = task_meta.get('operation', 'unknown')
                    
                    if op == 'handle_event':
                        evt = task_meta.get('event_type', 'unknown')
                        logger.warning(f"Timeout en componente {name} procesando evento {evt}")
                    else:
                        logger.warning(f"Timeout en operación {op} para componente {name}")
                    
                    # Cancelar la tarea para liberar recursos
                    if not task.done() and not task.cancelled():
                        task.cancel()
                        
                except Exception as e:
                    name = task_meta.get('component_name', 'unknown')
                    op = task_meta.get('operation', 'unknown')
                    
                    if op == 'handle_event':
                        evt = task_meta.get('event_type', 'unknown')
                        logger.error(f"Error en componente {name} procesando evento {evt}: {str(e)}")
                    else:
                        logger.error(f"Error en operación {op} para componente {name}: {str(e)}")
    
    async def start(self) -> None:
        """Iniciar motor y componentes en bloques paralelos."""
        if self.running:
            logger.warning("Motor ya está en ejecución")
            return
        
        logger.info("Iniciando motor de bloques paralelos")
        self.running = True
        
        # Crear bloques de componentes
        blocks = self._create_component_blocks()
        logger.info(f"Procesando {len(blocks)} bloques de componentes para inicio")
        
        # Procesar cada bloque
        start_tasks = []
        for block in blocks:
            task = asyncio.create_task(
                self._process_component_block(block, 'start')
            )
            start_tasks.append(task)
        
        # Esperar a que todos los bloques completen
        await asyncio.gather(*start_tasks, return_exceptions=True)
        
        logger.info("Motor de bloques paralelos iniciado")
    
    async def stop(self) -> None:
        """Detener motor y componentes en bloques paralelos."""
        if not self.running:
            logger.warning("Motor ya está detenido")
            return
        
        logger.info("Deteniendo motor de bloques paralelos")
        
        # Crear bloques de componentes
        blocks = self._create_component_blocks()
        logger.info(f"Procesando {len(blocks)} bloques de componentes para detención")
        
        # Procesar cada bloque
        stop_tasks = []
        for block in blocks:
            task = asyncio.create_task(
                self._process_component_block(block, 'stop')
            )
            stop_tasks.append(task)
        
        # Esperar a que todos los bloques completen
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        self.running = False
        logger.info("Motor de bloques paralelos detenido")
    
    async def emit_event(self, event_type: str, data: Optional[Dict[str, Any]] = None, 
                       source: str = "system") -> None:
        """
        Emitir evento a componentes registrados en bloques paralelos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
        """
        if not self.running:
            logger.warning(f"Intento de emitir evento {event_type} con motor detenido")
            return
        
        logger.info(f"Emitiendo evento {event_type} desde {source}")
        event_data = data or {}
        
        # Crear bloques de componentes
        blocks = self._create_component_blocks()
        logger.info(f"Procesando evento en {len(blocks)} bloques de componentes")
        
        # Procesar cada bloque
        event_tasks = []
        for block in blocks:
            task = asyncio.create_task(
                self._process_component_block(
                    block, 'handle_event', event_type, event_data, source
                )
            )
            event_tasks.append(task)
        
        # Esperar a que todos los bloques completen
        await asyncio.gather(*event_tasks, return_exceptions=True)
        
        logger.info(f"Evento {event_type} procesado por todos los bloques")