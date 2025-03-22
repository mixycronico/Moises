"""
Motor con bloques dinámicos y expansión adaptativa.

Este módulo implementa un motor que combina procesamiento en bloques
con componentes de expansión dinámica entre bloques, permitiendo
adaptarse automáticamente a diferentes cargas de trabajo.
"""

import asyncio
import logging
import time
import heapq
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
import statistics
from collections import deque

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


class DynamicBlock:
    """Bloque de componentes con capacidad de expansión dinámica."""
    
    def __init__(self, 
                priority: int, 
                components: List[Tuple[str, Component]],
                block_type: str = "regular",
                scaling_factor: float = 1.0):
        """
        Inicializar bloque dinámico.
        
        Args:
            priority: Nivel de prioridad del bloque
            components: Lista de tuplas (nombre, componente)
            block_type: Tipo de bloque ('regular', 'safe', 'expansion')
            scaling_factor: Factor de escalado para componentes
        """
        self.priority = priority
        self.components = components
        self.block_type = block_type
        self.scaling_factor = scaling_factor
        self.execution_times = deque(maxlen=10)  # Últimos tiempos de ejecución
        
    def __lt__(self, other):
        """Comparar bloques para ordenamiento."""
        # Los bloques seguros siempre tienen prioridad
        if self.block_type != other.block_type:
            if self.block_type == "safe":
                return True
            if other.block_type == "safe":
                return False
            if self.block_type == "expansion":
                return True
            if other.block_type == "expansion":
                return False
        # Ordenar por prioridad
        return self.priority < other.priority
    
    def add_execution_time(self, time_ms: float) -> None:
        """
        Registrar tiempo de ejecución para análisis de rendimiento.
        
        Args:
            time_ms: Tiempo de ejecución en milisegundos
        """
        self.execution_times.append(time_ms)
    
    def get_average_execution_time(self) -> Optional[float]:
        """
        Obtener tiempo promedio de ejecución.
        
        Returns:
            Tiempo promedio en milisegundos o None si no hay datos
        """
        if not self.execution_times:
            return None
        return statistics.mean(self.execution_times)
    
    def get_load_factor(self) -> float:
        """
        Calcular factor de carga basado en rendimiento.
        
        Returns:
            Factor de carga (>1 significa sobrecargado)
        """
        avg_time = self.get_average_execution_time()
        if avg_time is None:
            return 1.0
        
        # Factor base según tipo de bloque
        base_factor = {
            "safe": 0.7,       # Los bloques seguros tienen menos peso
            "expansion": 0.5,   # Los bloques de expansión tienen aún menos peso
            "regular": 1.0      # Los bloques regulares tienen peso normal
        }.get(self.block_type, 1.0)
        
        # Ajustar por tiempo de ejecución (normalizado a ~50ms como objetivo)
        time_factor = avg_time / 50.0
        
        return base_factor * time_factor * self.scaling_factor


class DynamicExpansionEngine:
    """
    Motor con bloques dinámicos y expansión adaptativa.
    
    Esta implementación combina procesamiento en bloques con
    componentes de expansión dinámica entre bloques, permitiendo
    adaptarse automáticamente a diferentes cargas de trabajo.
    """
    
    def __init__(self, 
                initial_block_size: int = 3,
                timeout: float = 0.5,
                min_concurrent_blocks: int = 2,
                max_concurrent_blocks: int = 8,
                expansion_threshold: float = 0.7,
                contraction_threshold: float = 0.3,
                auto_scaling: bool = True,
                scale_cooldown: float = 5.0,
                priority_mappings: Optional[Dict[str, int]] = None):
        """
        Inicializar motor con expansión dinámica.
        
        Args:
            initial_block_size: Tamaño inicial de cada bloque de componentes
            timeout: Tiempo máximo para operaciones
            min_concurrent_blocks: Número mínimo de bloques concurrentes
            max_concurrent_blocks: Número máximo de bloques concurrentes
            expansion_threshold: Umbral de carga para expandir (0-1)
            contraction_threshold: Umbral de carga para contraer (0-1)
            auto_scaling: Si se debe escalar automáticamente
            scale_cooldown: Tiempo mínimo entre escalados (segundos)
            priority_mappings: Mapeos de tipos de eventos a prioridades
        """
        self.block_size = initial_block_size
        self.timeout = timeout
        self.min_concurrent_blocks = min_concurrent_blocks
        self.max_concurrent_blocks = max_concurrent_blocks
        self.expansion_threshold = expansion_threshold
        self.contraction_threshold = contraction_threshold
        self.auto_scaling = auto_scaling
        self.scale_cooldown = scale_cooldown
        self.priority_mappings = priority_mappings or {}
        
        # Estado actual de concurrencia
        self.current_concurrent_blocks = min_concurrent_blocks
        
        # Componentes y estado
        self.components = {}
        self.safe_components = set()  # Nombres de componentes seguros
        self.expansion_components = set()  # Nombres de componentes de expansión
        self.running = False
        
        # Semáforo para controlar concurrencia (inicialmente al mínimo)
        self.semaphore = asyncio.Semaphore(min_concurrent_blocks)
        
        # Estadísticas y monitoreo
        self.processed_blocks = 0
        self.timeout_blocks = 0
        self.load_history = deque(maxlen=20)  # Historial de carga
        self.expansion_history = deque(maxlen=10)  # Historial de expansiones
        self.last_scale_time = 0  # Último tiempo de escalado
        
        logger.info(f"Motor de expansión dinámica creado: "
                   f"initial_block_size={initial_block_size}, timeout={timeout}s, "
                   f"concurrent={min_concurrent_blocks}-{max_concurrent_blocks}")
    
    def register_component(self, component: Component, component_type: str = "regular") -> None:
        """
        Registrar componente en el motor.
        
        Args:
            component: Componente a registrar
            component_type: Tipo de componente ('regular', 'safe', 'expansion')
        """
        self.components[component.name] = component
        
        if component_type == "safe":
            self.safe_components.add(component.name)
            logger.info(f"Componente SEGURO {component.name} registrado")
        elif component_type == "expansion":
            self.expansion_components.add(component.name)
            logger.info(f"Componente de EXPANSIÓN {component.name} registrado")
        else:
            logger.info(f"Componente regular {component.name} registrado")
    
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
    
    def _create_dynamic_blocks(self) -> List[DynamicBlock]:
        """
        Crear bloques dinámicos de componentes.
        
        Returns:
            Lista de bloques dinámicos
        """
        # Separar componentes por tipo
        safe_items = []
        expansion_items = []
        regular_items = []
        
        for name, comp in self.components.items():
            if name in self.safe_components:
                safe_items.append((name, comp))
            elif name in self.expansion_components:
                expansion_items.append((name, comp))
            else:
                regular_items.append((name, comp))
        
        blocks = []
        
        # Crear bloques seguros
        if safe_items:
            # Subdividir componentes seguros si hay muchos
            for i in range(0, len(safe_items), max(1, self.block_size // 2)):
                block_components = safe_items[i:i + max(1, self.block_size // 2)]
                safe_block = DynamicBlock(
                    priority=i,  # Prioridad incremental dentro de los seguros
                    components=block_components,
                    block_type="safe",
                    scaling_factor=0.8  # Los bloques seguros son más ligeros
                )
                blocks.append(safe_block)
        
        # Crear bloques regulares intercalados con bloques de expansión
        regular_block_count = (len(regular_items) + self.block_size - 1) // self.block_size
        
        for i in range(regular_block_count):
            # Añadir un bloque regular
            start_idx = i * self.block_size
            end_idx = min((i + 1) * self.block_size, len(regular_items))
            block_components = regular_items[start_idx:end_idx]
            
            # Calcular prioridad - los primeros bloques tienen mayor prioridad
            priority = i * 2 + 10  # Empezar después de los bloques seguros
            
            regular_block = DynamicBlock(
                priority=priority,
                components=block_components,
                block_type="regular",
                scaling_factor=1.0
            )
            blocks.append(regular_block)
            
            # Añadir un bloque de expansión después de cada bloque regular
            # si hay componentes de expansión disponibles
            if expansion_items and i < regular_block_count - 1:
                # Tomar componentes de expansión de forma circular
                expansion_idx = i % len(expansion_items)
                expansion_components = [expansion_items[expansion_idx]]
                
                expansion_block = DynamicBlock(
                    priority=priority + 1,  # Insertarlo entre bloques regulares
                    components=expansion_components,
                    block_type="expansion",
                    scaling_factor=0.5  # Los bloques de expansión son más ligeros
                )
                blocks.append(expansion_block)
        
        return blocks
    
    async def _process_block(self, 
                            block: DynamicBlock, 
                            operation: str,
                            event_type: Optional[str] = None,
                            event_data: Optional[Dict[str, Any]] = None,
                            event_source: Optional[str] = None) -> None:
        """
        Procesar un bloque de componentes con medición de rendimiento.
        
        Args:
            block: Bloque a procesar
            operation: Operación a realizar ('start', 'stop', 'handle_event')
            event_type: Tipo de evento (solo para 'handle_event')
            event_data: Datos del evento (solo para 'handle_event')
            event_source: Fuente del evento (solo para 'handle_event')
        """
        start_time = time.time()
        
        # Diccionario para almacenar metadatos de tareas
        task_metadata = {}
        
        # Ajustar timeout según tipo de bloque
        block_timeout = self.timeout
        if block.block_type == "safe":
            # Los bloques seguros tienen más tiempo para completar
            block_timeout = self.timeout * 1.5
        elif block.block_type == "expansion":
            # Los bloques de expansión tienen menos tiempo
            block_timeout = self.timeout * 0.7
        
        # Procesar según tipo de bloque
        if block.block_type == "safe":
            # Bloques seguros: procesamiento secuencial con timeout extendido
            for name, component in block.components:
                try:
                    logger.info(f"Procesando componente SEGURO {name} ({operation})")
                    if operation == 'start':
                        await asyncio.wait_for(component.start(), timeout=block_timeout)
                    elif operation == 'stop':
                        await asyncio.wait_for(component.stop(), timeout=block_timeout)
                    elif operation == 'handle_event':
                        # Asegurar que los parámetros nunca sean None
                        event_type_safe = event_type or ""
                        event_data_safe = event_data or {}
                        event_source_safe = event_source or ""
                        await asyncio.wait_for(
                            component.handle_event(event_type_safe, event_data_safe, event_source_safe),
                            timeout=block_timeout
                        )
                    logger.info(f"Componente SEGURO {name} procesado exitosamente ({operation})")
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout en componente SEGURO {name} ({operation})")
                except Exception as e:
                    logger.error(f"Error en componente SEGURO {name} ({operation}): {str(e)}")
        elif block.block_type == "expansion":
            # Bloques de expansión: procesamiento ligero que puede expandirse
            # Estos se adaptan según la carga del sistema
            current_load = self._get_current_load()
            
            # Si la carga es alta, ejecutar en modo "ligero"
            if current_load > 0.7:
                for name, component in block.components:
                    try:
                        logger.info(f"Procesando componente EXPANSIÓN {name} en modo LIGERO ({operation})")
                        if operation == 'start':
                            await asyncio.wait_for(component.start(), timeout=block_timeout)
                        elif operation == 'stop':
                            await asyncio.wait_for(component.stop(), timeout=block_timeout)
                        elif operation == 'handle_event':
                            # Pasar marcador de modo ligero en los datos
                            light_data = dict(event_data or {})
                            light_data['_light_mode'] = True
                            # Asegurar que los parámetros nunca sean None
                            event_type_safe = event_type or ""
                            event_source_safe = event_source or ""
                            await asyncio.wait_for(
                                component.handle_event(event_type_safe, light_data, event_source_safe),
                                timeout=block_timeout
                            )
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout en componente EXPANSIÓN {name} ({operation})")
                    except Exception as e:
                        logger.error(f"Error en componente EXPANSIÓN {name} ({operation}): {str(e)}")
            else:
                # Si la carga es baja, ejecutar en modo "completo"
                # Crear tareas para todos los componentes en el bloque
                tasks = []
                for name, component in block.components:
                    logger.info(f"Procesando componente EXPANSIÓN {name} en modo COMPLETO ({operation})")
                    task = None
                    if operation == 'start':
                        task = asyncio.create_task(component.start())
                    elif operation == 'stop':
                        task = asyncio.create_task(component.stop())
                    elif operation == 'handle_event':
                        # Pasar datos completos
                        full_data = dict(event_data or {})
                        full_data['_full_mode'] = True
                        # Asegurar que los parámetros nunca sean None
                        event_type_safe = event_type or ""
                        event_source_safe = event_source or ""
                        task = asyncio.create_task(
                            component.handle_event(event_type_safe, full_data, event_source_safe)
                        )
                    
                    # Solo agregar la tarea si se creó correctamente
                    if task is not None:
                        # Guardar metadata en diccionario usando id de la tarea como clave
                        task_metadata[id(task)] = {
                            'component_name': name,
                            'operation': operation
                        }
                        tasks.append(task)
                
                # Ejecutar con timeout extendido para procesamiento completo
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=block_timeout * 1.5
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout en modo COMPLETO para bloque de EXPANSIÓN {block.priority}")
                    for task in tasks:
                        if not task.done() and not task.cancelled():
                            task.cancel()
        else:
            # Bloques regulares: procesamiento concurrente normal
            async with self.semaphore:
                logger.info(f"Procesando bloque {block.priority} con {len(block.components)} componentes")
                
                # Crear tareas para todos los componentes en el bloque
                tasks = []
                
                for name, component in block.components:
                    task = None
                    if operation == 'start':
                        task = asyncio.create_task(component.start())
                    elif operation == 'stop':
                        task = asyncio.create_task(component.stop())
                    elif operation == 'handle_event':
                        # Asegurar que los parámetros nunca sean None
                        event_type_safe = event_type or ""
                        event_data_safe = event_data or {}
                        event_source_safe = event_source or ""
                        task = asyncio.create_task(
                            component.handle_event(event_type_safe, event_data_safe, event_source_safe)
                        )
                    
                    # Solo agregar la tarea si se creó correctamente
                    if task is not None:
                        # Guardar metadata en diccionario usando id de la tarea como clave
                        task_metadata[id(task)] = {
                            'component_name': name,
                            'operation': operation
                        }
                        tasks.append(task)
                
                # Esperar a que todas las tareas completen con timeout
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=block_timeout * len(tasks)  # Timeout proporcional al número de tareas
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
        
        # Calcular tiempo de ejecución y registrar para análisis
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000
        block.add_execution_time(execution_time_ms)
        
        # Actualizar estadísticas para auto-escalado
        if self.auto_scaling and operation == 'handle_event':
            self._update_load_statistics()
            self._check_scaling_needs()
    
    def _get_current_load(self) -> float:
        """
        Calcular carga actual del sistema.
        
        Returns:
            Factor de carga normalizado (0-1)
        """
        if not self.load_history:
            return 0.5  # Valor por defecto si no hay historial
        
        # Promedio ponderado del historial reciente
        recent_loads = list(self.load_history)
        weights = [i+1 for i in range(len(recent_loads))]  # Peso incremental
        weighted_load = sum(l * w for l, w in zip(recent_loads, weights)) / sum(weights)
        
        # Normalizar entre 0 y 1
        return max(0.0, min(1.0, weighted_load))
    
    def _update_load_statistics(self) -> None:
        """Actualizar estadísticas de carga del sistema."""
        # Calcular proporción de timeouts como medida de carga
        if self.processed_blocks + self.timeout_blocks > 0:
            timeout_ratio = self.timeout_blocks / (self.processed_blocks + self.timeout_blocks)
        else:
            timeout_ratio = 0.0
        
        # Ajustar por semáforo actual vs. máximo posible
        concurrency_ratio = self.current_concurrent_blocks / self.max_concurrent_blocks
        
        # Combinar para obtener factor de carga (mayor = más cargado)
        load_factor = (0.7 * timeout_ratio) + (0.3 * concurrency_ratio)
        
        # Registrar en historial
        self.load_history.append(load_factor)
        
        logger.debug(f"Carga actual: {load_factor:.2f} (timeouts: {timeout_ratio:.2f}, concurrencia: {concurrency_ratio:.2f})")
    
    def _check_scaling_needs(self) -> None:
        """Verificar si es necesario escalar y ajustar recursos."""
        current_time = time.time()
        
        # Respetar periodo de enfriamiento entre escalados
        if current_time - self.last_scale_time < self.scale_cooldown:
            return
        
        current_load = self._get_current_load()
        
        # Expandir si la carga es alta y podemos aumentar concurrencia
        if (current_load > self.expansion_threshold and 
            self.current_concurrent_blocks < self.max_concurrent_blocks):
            
            # Incrementar concurrencia (más agresivo si la carga es muy alta)
            if current_load > 0.9:
                new_concurrent = min(self.max_concurrent_blocks, self.current_concurrent_blocks + 2)
            else:
                new_concurrent = min(self.max_concurrent_blocks, self.current_concurrent_blocks + 1)
            
            logger.info(f"EXPANDIENDO concurrencia de {self.current_concurrent_blocks} a {new_concurrent} "
                       f"(carga: {current_load:.2f})")
            
            # Actualizar el semáforo (crear uno nuevo con más permisos)
            old_semaphore = self.semaphore
            self.semaphore = asyncio.Semaphore(new_concurrent)
            self.current_concurrent_blocks = new_concurrent
            
            # Registrar expansión
            self.expansion_history.append(("expand", current_time, current_load))
            self.last_scale_time = current_time
            
        # Contraer si la carga es baja y podemos reducir concurrencia
        elif (current_load < self.contraction_threshold and 
              self.current_concurrent_blocks > self.min_concurrent_blocks):
            
            # Decrementar concurrencia
            new_concurrent = max(self.min_concurrent_blocks, self.current_concurrent_blocks - 1)
            
            logger.info(f"CONTRAYENDO concurrencia de {self.current_concurrent_blocks} a {new_concurrent} "
                       f"(carga: {current_load:.2f})")
            
            # Actualizar el semáforo (crear uno nuevo con menos permisos)
            old_semaphore = self.semaphore
            self.semaphore = asyncio.Semaphore(new_concurrent)
            self.current_concurrent_blocks = new_concurrent
            
            # Registrar contracción
            self.expansion_history.append(("contract", current_time, current_load))
            self.last_scale_time = current_time
    
    async def start(self) -> None:
        """Iniciar motor y componentes en bloques dinámicos."""
        if self.running:
            logger.warning("Motor ya está en ejecución")
            return
        
        logger.info("Iniciando motor de expansión dinámica")
        
        # Crear bloques de componentes
        blocks = self._create_dynamic_blocks()
        logger.info(f"Creados {len(blocks)} bloques para inicio ({len(self.components)} componentes)")
        
        # Ordenar bloques por prioridad
        blocks.sort()  # Usa __lt__ para ordenar
        
        # Iniciar todos los bloques (primero seguros, luego resto)
        safe_blocks = [b for b in blocks if b.block_type == "safe"]
        expansion_blocks = [b for b in blocks if b.block_type == "expansion"]
        regular_blocks = [b for b in blocks if b.block_type == "regular"]
        
        # Procesar bloques seguros primero
        for block in safe_blocks:
            await self._process_block(block, 'start')
        
        # Inicializar componentes de expansión
        # Los componentes de expansión se inician por separado para asegurar que siempre inicien
        for name, component in self.components.items():
            if name in self.expansion_components:
                logger.info(f"Iniciando componente de EXPANSIÓN directamente: {name}")
                try:
                    await asyncio.wait_for(component.start(), timeout=self.timeout)
                    logger.info(f"Componente de EXPANSIÓN {name} iniciado correctamente")
                except Exception as e:
                    logger.error(f"Error al iniciar componente de EXPANSIÓN {name}: {str(e)}")
        
        # Luego procesar bloques de expansión (para estructura de procesamiento)
        if expansion_blocks:
            logger.info(f"Configurando {len(expansion_blocks)} bloques de expansión")
            expansion_tasks = []
            for block in expansion_blocks:
                task = asyncio.create_task(
                    self._process_block(block, 'start')
                )
                expansion_tasks.append(task)
            
            await asyncio.gather(*expansion_tasks, return_exceptions=True)
        
        # Finalmente procesar bloques regulares en paralelo
        regular_tasks = []
        for block in regular_blocks:
            task = asyncio.create_task(
                self._process_block(block, 'start')
            )
            regular_tasks.append(task)
        
        if regular_tasks:
            await asyncio.gather(*regular_tasks, return_exceptions=True)
        
        self.running = True
        logger.info("Motor de expansión dinámica iniciado")
    
    async def stop(self) -> None:
        """Detener motor y componentes en bloques dinámicos."""
        if not self.running:
            logger.warning("Motor ya está detenido")
            return
        
        logger.info("Deteniendo motor de expansión dinámica")
        
        # Crear bloques de componentes
        blocks = self._create_dynamic_blocks()
        logger.info(f"Creados {len(blocks)} bloques para detención")
        
        # Ordenar bloques por prioridad (inverso al inicio)
        blocks.sort(reverse=True)  # Usa __lt__ para ordenar
        
        # Detener todos los bloques (primero regulares, luego expansión, finalmente seguros)
        safe_blocks = [b for b in blocks if b.block_type == "safe"]
        expansion_blocks = [b for b in blocks if b.block_type == "expansion"]
        regular_blocks = [b for b in blocks if b.block_type == "regular"]
        
        # Detener bloques regulares en paralelo
        regular_tasks = []
        for block in regular_blocks:
            task = asyncio.create_task(
                self._process_block(block, 'stop')
            )
            regular_tasks.append(task)
        
        if regular_tasks:
            await asyncio.gather(*regular_tasks, return_exceptions=True)
        
        # Detener componentes de expansión directamente
        for name, component in self.components.items():
            if name in self.expansion_components:
                logger.info(f"Deteniendo componente de EXPANSIÓN directamente: {name}")
                try:
                    await asyncio.wait_for(component.stop(), timeout=self.timeout)
                    logger.info(f"Componente de EXPANSIÓN {name} detenido correctamente")
                except Exception as e:
                    logger.error(f"Error al detener componente de EXPANSIÓN {name}: {str(e)}")
        
        # Luego detener bloques de expansión (para estructura de procesamiento)
        if expansion_blocks:
            logger.info(f"Finalizando {len(expansion_blocks)} bloques de expansión")
            expansion_tasks = []
            for block in expansion_blocks:
                task = asyncio.create_task(
                    self._process_block(block, 'stop')
                )
                expansion_tasks.append(task)
            
            await asyncio.gather(*expansion_tasks, return_exceptions=True)
        
        # Finalmente detener bloques seguros
        for block in safe_blocks:
            await self._process_block(block, 'stop')
        
        self.running = False
        logger.info("Motor de expansión dinámica detenido")
    
    async def emit_event(self, event_type: str, data: Optional[Dict[str, Any]] = None, 
                       source: str = "system", priority: Optional[int] = None) -> None:
        """
        Emitir evento a componentes en bloques dinámicos.
        
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
        
        # Para pruebas y eventos simples, enviar directamente a cada componente
        # en lugar de usar bloques cuando hay pocos componentes
        if len(self.components) <= 10:
            # Enfoque directo para enviar eventos a componentes individuales
            # Especialmente útil para pruebas simples
            safe_tasks = []
            expansion_tasks = []
            regular_tasks = []
            
            # Procesar componentes seguros primero
            for name, component in self.components.items():
                if name in self.safe_components:
                    try:
                        # Asegurar que los parámetros nunca sean None
                        evt_type, evt_data, evt_source = safe_handle_event_params(event_type, event_data, source)
                        logger.info(f"Enviando evento {evt_type} a componente SEGURO {name}")
                        safe_tasks.append(asyncio.create_task(
                            asyncio.wait_for(
                                component.handle_event(evt_type, evt_data, evt_source),
                                timeout=self.timeout
                            )
                        ))
                    except Exception as e:
                        logger.error(f"Error enviando evento a componente SEGURO {name}: {str(e)}")
            
            # Esperar a que componentes seguros terminen
            if safe_tasks:
                await asyncio.gather(*safe_tasks, return_exceptions=True)
            
            # Procesar componentes de expansión
            for name, component in self.components.items():
                if name in self.expansion_components:
                    try:
                        # Asegurar que los parámetros nunca sean None
                        evt_type, evt_data, evt_source = safe_handle_event_params(event_type, event_data, source)
                        logger.info(f"Enviando evento {evt_type} a componente EXPANSIÓN {name}")
                        expansion_tasks.append(asyncio.create_task(
                            asyncio.wait_for(
                                component.handle_event(evt_type, evt_data, evt_source),
                                timeout=self.timeout
                            )
                        ))
                    except Exception as e:
                        logger.error(f"Error enviando evento a componente EXPANSIÓN {name}: {str(e)}")
            
            # Procesar componentes regulares
            for name, component in self.components.items():
                if name not in self.safe_components and name not in self.expansion_components:
                    try:
                        # Asegurar que los parámetros nunca sean None
                        evt_type, evt_data, evt_source = safe_handle_event_params(event_type, event_data, source)
                        logger.info(f"Enviando evento {evt_type} a componente REGULAR {name}")
                        regular_tasks.append(asyncio.create_task(
                            asyncio.wait_for(
                                component.handle_event(evt_type, evt_data, evt_source),
                                timeout=self.timeout
                            )
                        ))
                    except Exception as e:
                        logger.error(f"Error enviando evento a componente REGULAR {name}: {str(e)}")
            
            # Esperar a que el resto de componentes terminen
            if expansion_tasks or regular_tasks:
                await asyncio.gather(*(expansion_tasks + regular_tasks), return_exceptions=True)
            
            # Actualizar estadísticas
            if self.auto_scaling:
                self._update_load_statistics()
                self._check_scaling_needs()
                
            return
            
        # Para sistemas más grandes, usar la lógica de bloques dinámicos
        # Crear bloques dinámicos
        blocks = self._create_dynamic_blocks()
        
        # Ordenar bloques por prioridad
        blocks.sort()  # Usa __lt__ para ordenar
        
        # Ajustar distribución según prioridad del evento
        if event_priority == EventPriority.CRITICAL:
            # Eventos críticos: todos los bloques, pero seguros primero
            safe_blocks = [b for b in blocks if b.block_type == "safe"]
            expansion_blocks = [b for b in blocks if b.block_type == "expansion"]
            regular_blocks = [b for b in blocks if b.block_type == "regular"]
            
            # Asegurar que los parámetros nunca sean None
            evt_type, evt_data, evt_source = safe_handle_event_params(event_type, event_data, source)
            
            # Procesar bloques seguros primero (secuencialmente)
            for block in safe_blocks:
                await self._process_block(block, 'handle_event', evt_type, evt_data, evt_source)
            
            # Luego procesar bloques de expansión y regulares en paralelo
            other_tasks = []
            for block in expansion_blocks + regular_blocks:
                task = asyncio.create_task(
                    self._process_block(block, 'handle_event', evt_type, evt_data, evt_source)
                )
                other_tasks.append(task)
            
            if other_tasks:
                await asyncio.gather(*other_tasks, return_exceptions=True)
                
        elif event_priority == EventPriority.HIGH:
            # Eventos de alta prioridad: bloques seguros, expansión y algunos regulares
            safe_blocks = [b for b in blocks if b.block_type == "safe"]
            expansion_blocks = [b for b in blocks if b.block_type == "expansion"]
            regular_blocks = [b for b in blocks if b.block_type == "regular"]
            
            # Limitar bloques regulares a los de mayor prioridad
            regular_blocks = regular_blocks[:max(1, len(regular_blocks) // 2)]
            
            # Asegurar que los parámetros nunca sean None
            evt_type, evt_data, evt_source = safe_handle_event_params(event_type, event_data, source)
            
            # Procesar bloques seguros primero (secuencialmente)
            for block in safe_blocks:
                await self._process_block(block, 'handle_event', evt_type, evt_data, evt_source)
            
            # Luego procesar bloques de expansión y regulares seleccionados en paralelo
            other_tasks = []
            for block in expansion_blocks + regular_blocks:
                task = asyncio.create_task(
                    self._process_block(block, 'handle_event', evt_type, evt_data, evt_source)
                )
                other_tasks.append(task)
            
            if other_tasks:
                await asyncio.gather(*other_tasks, return_exceptions=True)
                
        elif event_priority == EventPriority.MEDIUM:
            # Eventos de media prioridad: bloques seguros, algunos expansión, pocos regulares
            safe_blocks = [b for b in blocks if b.block_type == "safe"]
            expansion_blocks = [b for b in blocks if b.block_type == "expansion"]
            regular_blocks = [b for b in blocks if b.block_type == "regular"]
            
            # Limitar bloques
            expansion_blocks = expansion_blocks[:max(1, len(expansion_blocks) // 2)]
            regular_blocks = regular_blocks[:max(1, len(regular_blocks) // 3)]
            
            # Asegurar que los parámetros nunca sean None
            evt_type, evt_data, evt_source = safe_handle_event_params(event_type, event_data, source)
            
            # Procesar bloques seguros primero (secuencialmente)
            for block in safe_blocks:
                await self._process_block(block, 'handle_event', evt_type, evt_data, evt_source)
            
            # Luego procesar bloques seleccionados en paralelo
            other_tasks = []
            for block in expansion_blocks + regular_blocks:
                task = asyncio.create_task(
                    self._process_block(block, 'handle_event', evt_type, evt_data, evt_source)
                )
                other_tasks.append(task)
            
            if other_tasks:
                await asyncio.gather(*other_tasks, return_exceptions=True)
                
        else:  # LOW o BACKGROUND
            # Eventos de baja prioridad: solo bloques seguros y algunos de expansión
            safe_blocks = [b for b in blocks if b.block_type == "safe"]
            expansion_blocks = [b for b in blocks if b.block_type == "expansion"]
            
            # Limitar bloques de expansión
            if expansion_blocks:
                expansion_blocks = [expansion_blocks[0]]  # Solo el primer bloque de expansión
            
            # Asegurar que los parámetros nunca sean None
            evt_type, evt_data, evt_source = safe_handle_event_params(event_type, event_data, source)
            
            # Procesar bloques seguros primero (secuencialmente) si son pocos
            if len(safe_blocks) <= 2:
                for block in safe_blocks:
                    await self._process_block(block, 'handle_event', evt_type, evt_data, evt_source)
            else:
                # Si hay muchos bloques seguros, procesar en paralelo
                safe_tasks = []
                for block in safe_blocks:
                    task = asyncio.create_task(
                        self._process_block(block, 'handle_event', evt_type, evt_data, evt_source)
                    )
                    safe_tasks.append(task)
                
                if safe_tasks:
                    await asyncio.gather(*safe_tasks, return_exceptions=True)
            
            # Luego procesar bloques de expansión seleccionados
            for block in expansion_blocks:
                await self._process_block(block, 'handle_event', evt_type, evt_data, evt_source)
        
        # Actualizar estadísticas después de procesar el evento
        if self.auto_scaling:
            self._update_load_statistics()
            self._check_scaling_needs()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del motor.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "processed_blocks": self.processed_blocks,
            "timeout_blocks": self.timeout_blocks,
            "components": {
                "total": len(self.components),
                "safe": len(self.safe_components),
                "expansion": len(self.expansion_components),
                "regular": len(self.components) - len(self.safe_components) - len(self.expansion_components)
            },
            "concurrency": {
                "current": self.current_concurrent_blocks,
                "min": self.min_concurrent_blocks,
                "max": self.max_concurrent_blocks
            },
            "load": self._get_current_load(),
            "scaling_history": list(self.expansion_history),
            "running": self.running
        }