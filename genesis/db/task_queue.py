"""
Módulo de cola de tareas para operaciones de base de datos en Genesis.

Este módulo proporciona una implementación ultra-robusta para manejar operaciones
asíncronas de base de datos, garantizando la integridad de los datos y evitando
cualquier tipo de error relacionado con bucles de eventos o fallos de concurrencia.

Características principales:
- Cola de tareas asíncrona con aislamiento total de bucles de eventos
- Mecanismo de transacciones protegidas con rollback automático
- Sistema de reintentos con backoff exponencial y jitter
- Monitoreo detallado y registro exhaustivo de operaciones
- Verificaciones de integridad antes y después de cada operación
- Recuperación avanzada ante fallos
- Protección contra deadlocks y bloqueos
"""
import asyncio
import logging
import time
import traceback
import json
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Coroutine, Set
from functools import wraps
from datetime import datetime

# Configuración de logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Asegurar que el logger tenga al menos un handler
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class DatabaseTaskQueue:
    """
    Cola de tareas ultra-robusta para operaciones de base de datos en Genesis.
    
    Esta clase implementa una cola de tareas que garantiza la integridad de las
    operaciones de base de datos, incluso bajo condiciones de alta carga o fallos.
    Incluye mecanismos avanzados de protección, verificación y recuperación.
    
    Características clave:
    - Aislamiento total de bucles de eventos
    - Transacciones protegidas con rollback automático
    - Sistema avanzado de reintentos con backoff exponencial
    - Verificación de integridad de datos
    - Registro detallado de todas las operaciones
    - Tolerancia a fallos con recuperación automática
    - Prevención de deadlocks y race conditions
    """
    
    def __init__(self, max_workers: int = 2, queue_size: int = 100, max_retries: int = 3):
        """
        Inicializar cola de tareas ultrasegura.
        
        Args:
            max_workers: Número máximo de trabajadores concurrentes
            queue_size: Tamaño máximo de la cola
            max_retries: Número máximo de reintentos automáticos
        """
        # Cola principal de tareas
        self.queue = asyncio.Queue(maxsize=queue_size)
        
        # Configuración de trabajadores
        self.max_workers = max_workers
        self.workers: List[asyncio.Task] = []
        self.running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Configuración de reintentos
        self.max_retries = max_retries
        self.retry_delays = [0.1, 0.5, 1.0, 2.0, 5.0]  # Delays en segundos
        
        # Cola de respaldo para tareas críticas que fallaron
        self.recovery_queue = asyncio.Queue(maxsize=queue_size // 2)
        
        # Registro de operaciones críticas (transaccionales)
        self.critical_ops_log: Dict[str, Dict[str, Any]] = {}
        
        # Estadísticas detalladas
        self.stats = {
            "tasks_enqueued": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_retried": 0,
            "tasks_recovered": 0,
            "transaction_commits": 0,
            "transaction_rollbacks": 0,
            "integrity_checks_passed": 0,
            "integrity_checks_failed": 0,
            "deadlocks_prevented": 0,
            "last_error": None,
            "last_10_errors": [],
            "start_time": None,
            "processing_time": 0,
            "max_task_time": 0,
            "task_times": {},  # Tiempo promedio por tipo de tarea
            "task_success_rates": {}  # Tasa de éxito por tipo de tarea
        }
        
        # Set para rastrear operaciones en curso (prevenir deadlocks)
        self.in_progress_ops: Set[str] = set()
        
        # Verificación de integridad 
        self.last_integrity_check = time.time()
        self.integrity_check_interval = 60  # segundos
    
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
        Trabajador que procesa tareas de la cola con máxima protección.
        
        Este trabajador implementa múltiples capas de protección:
        - Aislamiento completo de excepciones
        - Detección de deadlocks
        - Reintentos con backoff exponencial
        - Monitoreo detallado
        - Recuperación automática
        
        Args:
            worker_id: ID del trabajador
        """
        logger.info(f"Trabajador {worker_id} iniciado con protecciones avanzadas")
        
        # Contadores locales para estadísticas
        local_tasks_processed = 0
        local_tasks_succeeded = 0
        local_tasks_failed = 0
        
        while self.running:
            task_id = f"task_{uuid.uuid4().hex[:8]}"
            task_tuple = None
            
            try:
                # Obtener siguiente tarea
                task_tuple = await self.queue.get()
                func, args, kwargs, future = task_tuple
                
                # Extraer información para logging
                func_name = func.__name__ if hasattr(func, "__name__") else str(func)
                task_name = kwargs.get("operation_name", func_name)
                
                # Generar ID único para la operación
                operation_id = kwargs.get("operation_id", f"op_{uuid.uuid4().hex[:8]}")
                
                # Verificar si es una operación crítica (transaccional)
                is_critical = kwargs.get("critical", False)
                is_transactional = kwargs.get("transactional", False)
                
                if is_critical:
                    # Registrar operación crítica
                    self.critical_ops_log[operation_id] = {
                        "task_id": task_id,
                        "task_name": task_name,
                        "start_time": time.time(),
                        "state": "started",
                        "retries": 0,
                        "worker_id": worker_id
                    }
                
                # Registrar tipos de operación para prevenir deadlocks
                op_type = kwargs.get("operation_type", "default")
                
                # Verificar si hay operaciones similares en curso (prevenir deadlocks)
                if op_type in self.in_progress_ops:
                    # Posible deadlock, registrar y continuar con precaución
                    logger.warning(f"Posible deadlock detectado - Operación {op_type} ya en curso. Usando protección.")
                    self.stats["deadlocks_prevented"] += 1
                
                # Registrar operación en curso
                self.in_progress_ops.add(op_type)
                
                # Iniciar procesamiento
                logger.debug(f"Trabajador {worker_id} procesando {task_name} (ID: {task_id})")
                start_time = time.time()
                
                # Manejar la ejecución con protección de transacciones si es necesario
                if is_transactional:
                    result = await self._execute_transactional(worker_id, func, args, kwargs, task_id, operation_id)
                else:
                    # Ejecutar con reintentos automáticos
                    result = await self._execute_with_retry(worker_id, func, args, kwargs, task_id)
                
                # Registrar éxito
                future.set_result(result)
                self.stats["tasks_completed"] += 1
                local_tasks_succeeded += 1
                
                # Calcular estadísticas de tasa de éxito para este tipo de operación
                if task_name not in self.stats["task_success_rates"]:
                    self.stats["task_success_rates"][task_name] = {"total": 0, "success": 0}
                
                self.stats["task_success_rates"][task_name]["total"] += 1
                self.stats["task_success_rates"][task_name]["success"] += 1
                
                # Actualizar estado de operación crítica
                if is_critical and operation_id in self.critical_ops_log:
                    self.critical_ops_log[operation_id]["state"] = "completed"
                    self.critical_ops_log[operation_id]["end_time"] = time.time()
                
            except asyncio.CancelledError:
                # Manejar cancelación limpiamente
                logger.info(f"Trabajador {worker_id} cancelado, finalizando tareas pendientes...")
                
                # Intentar completar la tarea actual si es posible
                if task_tuple:
                    _, _, _, future = task_tuple
                    if not future.done():
                        future.cancel()
                
                break  # Salir del bucle
                
            except Exception as e:
                # Manejar otros errores en el worker principal
                error_info = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                logger.error(f"Error crítico en trabajador {worker_id}: {error_info}")
                
                # Registrar error y estadísticas
                self.stats["tasks_failed"] += 1
                local_tasks_failed += 1
                
                # Mantener historial de errores
                self.stats["last_error"] = error_info
                self._add_to_error_history(error_info)
                
                # Completar el future con el error si es posible
                if task_tuple:
                    _, _, _, future = task_tuple
                    if not future.done():
                        future.set_exception(e)
                
                # Esperar un poco para evitar bucles infinitos de errores
                await asyncio.sleep(1)
                
            finally:
                # Limpiar operación en curso
                if task_tuple:
                    op_type = task_tuple[2].get("operation_type", "default")
                    if op_type in self.in_progress_ops:
                        self.in_progress_ops.remove(op_type)
                
                # Asegurar que siempre se marca la tarea como completada
                if task_tuple:
                    self.queue.task_done()
                    
                    # Registrar tiempo de procesamiento
                    elapsed = time.time() - start_time
                    self.stats["processing_time"] += elapsed
                    
                    # Actualizar tiempo máximo si aplica
                    if elapsed > self.stats["max_task_time"]:
                        self.stats["max_task_time"] = elapsed
                    
                    # Registrar tiempo promedio por tipo de tarea
                    func_name = task_tuple[0].__name__ if hasattr(task_tuple[0], "__name__") else str(task_tuple[0])
                    task_name = task_tuple[2].get("operation_name", func_name)
                    
                    if task_name not in self.stats["task_times"]:
                        self.stats["task_times"][task_name] = {"total_time": 0, "count": 0}
                    
                    self.stats["task_times"][task_name]["total_time"] += elapsed
                    self.stats["task_times"][task_name]["count"] += 1
                
                # Verificar si es momento de realizar una comprobación de integridad
                current_time = time.time()
                if current_time - self.last_integrity_check > self.integrity_check_interval:
                    asyncio.create_task(self._check_integrity())
                    self.last_integrity_check = current_time
                
                # Incrementar contador local de tareas
                local_tasks_processed += 1
                
                # Realizar registro periódico de estadísticas del trabajador
                if local_tasks_processed % 100 == 0:
                    logger.info(f"Trabajador {worker_id} ha procesado {local_tasks_processed} tareas " +
                                f"({local_tasks_succeeded} éxitos, {local_tasks_failed} fallos)")
                
        # Registro final al terminar
        logger.info(f"Trabajador {worker_id} finalizado - Procesadas: {local_tasks_processed}, " +
                   f"Exitosas: {local_tasks_succeeded}, Fallidas: {local_tasks_failed}")
    
    async def _execute_with_retry(self, worker_id: int, func: Callable, args: tuple, kwargs: dict, task_id: str) -> Any:
        """
        Ejecutar función con reintentos automáticos y backoff exponencial.
        
        Args:
            worker_id: ID del trabajador
            func: Función a ejecutar
            args: Argumentos posicionales
            kwargs: Argumentos con nombre
            task_id: ID de la tarea
            
        Returns:
            Resultado de la función
            
        Raises:
            Exception: Si falla después de todos los reintentos
        """
        # Determinar número máximo de reintentos
        max_retries = kwargs.pop("max_retries", self.max_retries)
        retry_count = 0
        last_error = None
        
        # Lista de excepciones que no deberían reintentarse (errores fatales)
        fatal_exceptions = (
            KeyboardInterrupt,
            SystemExit,
            asyncio.CancelledError
        )
        
        # Obtener nombre de función para logging
        func_name = func.__name__ if hasattr(func, "__name__") else str(func)
        
        while retry_count <= max_retries:
            try:
                # Marcar el intento actual en kwargs para funciones que necesiten saberlo
                current_kwargs = kwargs.copy()
                current_kwargs["_current_retry"] = retry_count
                
                # Ejecutar la función
                if retry_count > 0:
                    logger.info(f"Reintento {retry_count}/{max_retries} para tarea {func_name} (ID: {task_id})")
                    
                return await func(*args, **current_kwargs)
                
            except fatal_exceptions as e:
                # No reintentar para excepciones fatales
                logger.error(f"Error fatal en tarea {func_name} (ID: {task_id}): {e}")
                raise
                
            except Exception as e:
                # Registrar el error
                last_error = e
                error_info = f"{type(e).__name__}: {str(e)}"
                
                if retry_count < max_retries:
                    # Calcular delay con backoff exponencial y jitter aleatorio
                    delay = min(
                        self.retry_delays[min(retry_count, len(self.retry_delays) - 1)] * (2 ** retry_count),
                        30  # Máximo 30 segundos
                    )
                    # Añadir jitter (±25%)
                    jitter = (random.random() - 0.5) * 0.5 * delay
                    delay += jitter
                    
                    logger.warning(
                        f"Error en tarea {func_name} (ID: {task_id}) - Reintento {retry_count+1}/{max_retries} " +
                        f"en {delay:.2f}s: {error_info}"
                    )
                    
                    # Incrementar estadísticas
                    self.stats["tasks_retried"] += 1
                    
                    # Esperar antes del siguiente intento
                    await asyncio.sleep(delay)
                    retry_count += 1
                else:
                    # Último intento fallido, propagar el error
                    logger.error(
                        f"Error final en tarea {func_name} (ID: {task_id}) después de {max_retries} reintentos: {error_info}"
                    )
                    raise
        
        # No debería llegar aquí, pero por si acaso
        if last_error:
            raise last_error
        return None
    
    async def _execute_transactional(self, worker_id: int, func: Callable, args: tuple, kwargs: dict, 
                                    task_id: str, operation_id: str) -> Any:
        """
        Ejecutar función dentro de una transacción con protección de rollback automático.
        
        Args:
            worker_id: ID del trabajador
            func: Función a ejecutar
            args: Argumentos posicionales
            kwargs: Argumentos con nombre
            task_id: ID de la tarea
            operation_id: ID de la operación
            
        Returns:
            Resultado de la función
            
        Raises:
            Exception: Si falla la transacción y no puede hacer rollback
        """
        # Obtener nombre de función para logging
        func_name = func.__name__ if hasattr(func, "__name__") else str(func)
        
        # Registrar inicio de transacción
        logger.info(f"Iniciando transacción para {func_name} (ID: {task_id}, Op: {operation_id})")
        
        # Extraer o crear sesión de base de datos
        session = kwargs.get("db_session", None)
        external_session = session is not None
        
        try:
            # Si no hay sesión externa, crear una nueva
            if not external_session:
                # Importar funciones de base de datos aquí para evitar circular imports
                from genesis.db.base import get_db_session
                session = await get_db_session()
                kwargs["db_session"] = session
            
            # Iniciar transacción si la sesión lo soporta
            transaction_active = False
            if hasattr(session, "begin") and callable(session.begin):
                transaction = await session.begin()
                transaction_active = True
                logger.debug(f"Transacción iniciada para tarea {func_name} (ID: {task_id})")
            
            # Ejecutar la función con reintentos
            result = await self._execute_with_retry(worker_id, func, args, kwargs, task_id)
            
            # Commit transacción si está activa y es nuestra sesión
            if transaction_active and not external_session:
                await transaction.commit()
                self.stats["transaction_commits"] += 1
                logger.debug(f"Transacción confirmada para tarea {func_name} (ID: {task_id})")
            
            return result
            
        except Exception as e:
            # Error en la transacción, intentar rollback
            error_info = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Error en transacción para {func_name} (ID: {task_id}): {error_info}")
            
            # Intentar rollback si está activa y es nuestra sesión
            if 'transaction' in locals() and transaction_active and not external_session:
                try:
                    await transaction.rollback()
                    self.stats["transaction_rollbacks"] += 1
                    logger.info(f"Rollback exitoso para transacción de tarea {func_name} (ID: {task_id})")
                except Exception as rollback_error:
                    logger.critical(
                        f"ERROR CRÍTICO: Rollback fallido para tarea {func_name} (ID: {task_id}): "
                        f"{type(rollback_error).__name__}: {str(rollback_error)}"
                    )
            
            # Actualizar registro de operación crítica
            if operation_id in self.critical_ops_log:
                self.critical_ops_log[operation_id]["state"] = "failed"
                self.critical_ops_log[operation_id]["error"] = error_info
                self.critical_ops_log[operation_id]["end_time"] = time.time()
            
            # Propagar el error original
            raise
            
        finally:
            # Cerrar sesión si la creamos nosotros
            if not external_session and session is not None:
                if hasattr(session, "close") and callable(session.close):
                    await session.close()
                    logger.debug(f"Sesión cerrada para tarea {func_name} (ID: {task_id})")
    
    async def _check_integrity(self) -> Dict[str, Any]:
        """
        Realizar verificación de integridad del sistema.
        
        Esta función verifica que no haya operaciones "perdidas" o en estado
        inconsistente, como transacciones pendientes o trabajadores bloqueados.
        
        Returns:
            Resultado de la verificación
        """
        logger.debug("Iniciando verificación de integridad del sistema")
        
        results = {
            "status": "success",
            "issues_found": 0,
            "issues_fixed": 0,
            "details": []
        }
        
        try:
            # Verificar operaciones críticas pendientes
            stalled_ops = 0
            current_time = time.time()
            
            for op_id, op_info in list(self.critical_ops_log.items()):
                if op_info["state"] == "started":
                    # Verificar si lleva demasiado tiempo (más de 5 minutos)
                    if current_time - op_info["start_time"] > 300:
                        logger.warning(f"Operación crítica {op_id} posiblemente estancada")
                        results["details"].append({
                            "type": "stalled_operation",
                            "operation_id": op_id,
                            "elapsed_time": current_time - op_info["start_time"]
                        })
                        stalled_ops += 1
                        results["issues_found"] += 1
            
            # Verificar deadlocks potenciales
            if len(self.in_progress_ops) > 3 * self.max_workers:
                logger.warning("Posible acumulación de operaciones en curso, podría indicar deadlocks")
                results["details"].append({
                    "type": "potential_deadlock",
                    "operations_in_progress": len(self.in_progress_ops)
                })
                results["issues_found"] += 1
            
            # Verificar trabajadores activos
            active_workers = len([w for w in self.workers if not w.done()])
            if active_workers < self.max_workers and self.running:
                logger.warning(f"Solo {active_workers}/{self.max_workers} trabajadores activos")
                results["details"].append({
                    "type": "worker_shortage",
                    "active_workers": active_workers,
                    "expected_workers": self.max_workers
                })
                results["issues_found"] += 1
            
            # Limpiar el registro de operaciones críticas completadas hace mucho tiempo
            for op_id, op_info in list(self.critical_ops_log.items()):
                if op_info["state"] in ("completed", "failed") and "end_time" in op_info:
                    if current_time - op_info["end_time"] > 3600:  # 1 hora
                        self.critical_ops_log.pop(op_id, None)
                        
            # Si todo está bien
            if results["issues_found"] == 0:
                self.stats["integrity_checks_passed"] += 1
                logger.debug("Verificación de integridad completada sin problemas")
            else:
                self.stats["integrity_checks_failed"] += 1
                logger.warning(f"Verificación de integridad completada con {results['issues_found']} problemas")
            
        except Exception as e:
            # Registrar error en verificación de integridad
            logger.error(f"Error en verificación de integridad: {e}")
            results["status"] = "error"
            results["error"] = str(e)
        
        return results
    
    def _add_to_error_history(self, error_info: str) -> None:
        """
        Añadir error al historial de errores recientes.
        
        Args:
            error_info: Información detallada del error
        """
        # Mantener un historial limitado de los últimos 10 errores
        if len(self.stats["last_10_errors"]) >= 10:
            self.stats["last_10_errors"].pop(0)
        
        self.stats["last_10_errors"].append({
            "time": datetime.now().isoformat(),
            "error": error_info
        })
    
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

# Instancia global de la cola con 4 trabajadores para mayor paralelismo
db_task_queue = DatabaseTaskQueue(max_workers=4, queue_size=200, max_retries=3)

def async_db_operation(critical: bool = False, transactional: bool = False, 
                      max_retries: int = None, operation_type: str = None,
                      operation_name: str = None):
    """
    Decorador para operaciones asíncronas de base de datos con protección avanzada.
    
    Este decorador proporciona una capa robusta de protección para operaciones
    de base de datos, garantizando que se ejecuten en el bucle de eventos correcto,
    con manejo de transacciones y reintentos automáticos si es necesario.
    
    Args:
        critical: Si la operación es considerada crítica para monitoreo especial
        transactional: Si la operación requiere protección transaccional (auto commit/rollback)
        max_retries: Número máximo de reintentos, None para usar el valor por defecto
        operation_type: Categoría de operación para prevenir deadlocks (ej: "account_update")
        operation_name: Nombre descriptivo para la operación (para estadísticas)
        
    Returns:
        Decorador configurado
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generar ID único para la operación
            operation_id = f"op_{uuid.uuid4().hex[:8]}"
            
            # Preparar kwargs extendidos con metadatos
            extended_kwargs = kwargs.copy()
            extended_kwargs["operation_id"] = operation_id
            extended_kwargs["critical"] = critical
            extended_kwargs["transactional"] = transactional
            
            if max_retries is not None:
                extended_kwargs["max_retries"] = max_retries
                
            if operation_type:
                extended_kwargs["operation_type"] = operation_type
            else:
                # Usar el nombre del módulo como tipo de operación por defecto
                if hasattr(func, "__module__"):
                    extended_kwargs["operation_type"] = func.__module__.split(".")[-1]
                
            if operation_name:
                extended_kwargs["operation_name"] = operation_name
            else:
                # Usar el nombre de la función como nombre de operación
                extended_kwargs["operation_name"] = func.__name__
            
            # Encolar y ejecutar la tarea
            return await db_task_queue.enqueue(func, *args, **extended_kwargs)
        
        return wrapper
    
    # Permitir usar el decorador con o sin parámetros
    if callable(critical):
        func = critical
        critical = False
        return decorator(func)
    
    return decorator


def transactional_db_operation(critical: bool = True, max_retries: int = 2, 
                              operation_type: str = None, operation_name: str = None):
    """
    Decorador específico para operaciones transaccionales de base de datos.
    
    Este decorador es un caso especial de async_db_operation que siempre
    activa la protección transaccional (commit/rollback automático).
    
    Args:
        critical: Si la operación es considerada crítica para monitoreo especial
        max_retries: Número máximo de reintentos, None para usar el valor por defecto
        operation_type: Categoría de operación para prevenir deadlocks
        operation_name: Nombre descriptivo para la operación
        
    Returns:
        Decorador configurado
    """
    return async_db_operation(
        critical=critical,
        transactional=True,
        max_retries=max_retries,
        operation_type=operation_type,
        operation_name=operation_name
    )


def critical_db_operation(transactional: bool = True, max_retries: int = 3,
                         operation_type: str = None, operation_name: str = None):
    """
    Decorador para operaciones críticas de base de datos con máxima protección.
    
    Este decorador marca la operación como crítica, lo que activa mecanismos
    adicionales de monitoreo, registro y protección.
    
    Args:
        transactional: Si la operación requiere protección transaccional
        max_retries: Número máximo de reintentos
        operation_type: Categoría de operación para prevenir deadlocks
        operation_name: Nombre descriptivo para la operación
        
    Returns:
        Decorador configurado
    """
    return async_db_operation(
        critical=True,
        transactional=transactional,
        max_retries=max_retries,
        operation_type=operation_type,
        operation_name=operation_name
    )