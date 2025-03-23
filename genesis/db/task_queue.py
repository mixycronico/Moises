"""
Módulo de cola de tareas para operaciones de base de datos en Genesis.

Este módulo proporciona una implementación robusta para manejar operaciones
asíncronas de base de datos, evitando problemas de bucles de eventos y
permitiendo operaciones en segundo plano sin bloquear el hilo principal.
"""
import asyncio
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Coroutine
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseTaskQueue:
    """
    Cola de tareas para operaciones de base de datos en Genesis.
    
    Esta clase implementa una cola de tareas que permite encolar operaciones
    de base de datos para ser ejecutadas en un bucle de eventos específico,
    evitando problemas de "Task got Future attached to a different loop".
    """
    
    def __init__(self, max_workers: int = 3, queue_size: int = 100):
        """
        Inicializar cola de tareas.
        
        Args:
            max_workers: Número máximo de trabajadores concurrentes
            queue_size: Tamaño máximo de la cola
        """
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.max_workers = max_workers
        self.workers: List[asyncio.Task] = []
        self.running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.stats = {
            "tasks_enqueued": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "last_error": None,
            "start_time": None,
            "processing_time": 0
        }
    
    async def start(self):
        """Iniciar procesamiento de cola."""
        if self.running:
            logger.warning("La cola de tareas ya está en ejecución")
            return
        
        self.running = True
        self.loop = asyncio.get_event_loop()
        self.stats["start_time"] = datetime.now()
        
        # Iniciar trabajadores
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)
        
        logger.info(f"Cola de tareas iniciada con {self.max_workers} trabajadores")
    
    async def stop(self):
        """Detener procesamiento de cola."""
        if not self.running:
            logger.warning("La cola de tareas no está en ejecución")
            return
        
        self.running = False
        
        # Esperar a que se completen todas las tareas pendientes
        if not self.queue.empty():
            logger.info(f"Esperando a que se completen {self.queue.qsize()} tareas pendientes")
            await self.queue.join()
        
        # Cancelar trabajadores
        for worker in self.workers:
            worker.cancel()
        
        # Esperar a que se cancelen todos los trabajadores
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        self.workers = []
        logger.info("Cola de tareas detenida")
    
    async def _worker(self, worker_id: int):
        """
        Trabajador que procesa tareas de la cola.
        
        Args:
            worker_id: ID del trabajador
        """
        logger.debug(f"Trabajador {worker_id} iniciado")
        
        while self.running:
            try:
                # Obtener siguiente tarea
                task_tuple = await self.queue.get()
                func, args, kwargs, future = task_tuple
                
                start_time = time.time()
                try:
                    # Ejecutar tarea
                    result = await func(*args, **kwargs)
                    future.set_result(result)
                    self.stats["tasks_completed"] += 1
                except Exception as e:
                    # Capturar errores
                    error_info = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                    logger.error(f"Error en tarea de base de datos: {error_info}")
                    self.stats["tasks_failed"] += 1
                    self.stats["last_error"] = error_info
                    
                    if not future.done():
                        future.set_exception(e)
                finally:
                    # Marcar tarea como completada
                    self.queue.task_done()
                    elapsed = time.time() - start_time
                    self.stats["processing_time"] += elapsed
                    
                    if elapsed > 1.0:  # Advertir si toma más de 1 segundo
                        logger.warning(f"Tarea de base de datos tomó {elapsed:.2f} segundos")
            
            except asyncio.CancelledError:
                # Manejar cancelación
                logger.debug(f"Trabajador {worker_id} cancelado")
                break
            except Exception as e:
                # Manejar otros errores
                logger.error(f"Error en trabajador {worker_id}: {e}")
                await asyncio.sleep(1)  # Evitar bucle infinito si hay errores
        
        logger.debug(f"Trabajador {worker_id} finalizado")
    
    async def enqueue(self, func: Callable[..., Coroutine], *args, **kwargs) -> Any:
        """
        Encolar una tarea para ejecución asíncrona.
        
        Args:
            func: Función asíncrona a ejecutar
            *args: Argumentos para la función
            **kwargs: Argumentos con nombre para la función
            
        Returns:
            Resultado de la función
        """
        # Crear future para recibir el resultado
        future = asyncio.Future()
        
        # Encolar tarea
        await self.queue.put((func, args, kwargs, future))
        self.stats["tasks_enqueued"] += 1
        
        # Iniciar cola si no está activa
        if not self.running:
            asyncio.create_task(self.start())
        
        # Esperar resultado
        return await future
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de la cola.
        
        Returns:
            Estadísticas de la cola
        """
        stats = self.stats.copy()
        
        # Calcular tiempo total de ejecución
        if stats["start_time"]:
            stats["uptime_seconds"] = (datetime.now() - stats["start_time"]).total_seconds()
        
        # Calcular tareas pendientes
        stats["pending_tasks"] = self.queue.qsize() if self.queue else 0
        
        # Calcular tiempo promedio por tarea
        if stats["tasks_completed"] > 0:
            stats["avg_processing_time"] = stats["processing_time"] / stats["tasks_completed"]
        else:
            stats["avg_processing_time"] = 0
        
        return stats

# Instancia global de la cola
db_task_queue = DatabaseTaskQueue()

def async_db_operation(func):
    """
    Decorador para operaciones asíncronas de base de datos.
    
    Este decorador encola la operación en la cola de tareas, asegurando
    que se ejecute en el bucle de eventos correcto.
    
    Args:
        func: Función asíncrona a decorar
        
    Returns:
        Función decorada
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await db_task_queue.enqueue(func, *args, **kwargs)
    return wrapper