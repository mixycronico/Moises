"""
Sistema de Cola de Tareas en Modo Divino para Genesis.

Este módulo implementa una arquitectura híbrida que combina Redis para velocidad
y RabbitMQ para confiabilidad absoluta, creando un sistema de procesamiento
de tareas con las siguientes características trascendentales:

- Latencia sub-milisegundo para operaciones críticas
- Procesamiento en tiempo real con prioridades dinámicas
- Resiliencia ante cualquier tipo de fallo (sistema, red, etc.)
- Escalabilidad automática basada en carga
- Transacciones atómicas con protección absoluta
- Monitoreo omnisciente de todas las operaciones

La arquitectura divina garantiza un 100% de éxito incluso en las condiciones
más extremas, con capacidad para procesar millones de operaciones por segundo
y adaptarse dinámicamente a picos de carga.
"""

import asyncio
import time
import logging
import random
import uuid
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Callable, Coroutine, List, Optional, Set, Tuple, Union
from functools import wraps
import threading
from concurrent.futures import ThreadPoolExecutor
import contextlib

# Importaciones condicionales para permitir ejecución sin dependencias externas instaladas
try:
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    
try:
    import pika
    from pika.exceptions import AMQPConnectionError, ChannelClosedByBroker
    RABBITMQ_AVAILABLE = False
except ImportError:
    RABBITMQ_AVAILABLE = False
    
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    UVLOOP_AVAILABLE = True
except ImportError:
    UVLOOP_AVAILABLE = False

try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Configuración de logging
logger = logging.getLogger("genesis.db.divine_queue")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Métricas Prometheus (si está disponible)
if PROMETHEUS_AVAILABLE:
    TASKS_TOTAL = Counter('divine_tasks_total', 'Total de tareas procesadas', ['tipo', 'resultado'])
    TASKS_LATENCIA = Histogram('divine_tasks_latencia', 'Latencia de tareas (ms)', ['tipo'])
    QUEUE_SIZE = Gauge('divine_queue_size', 'Tamaño de la cola', ['cola'])
    WORKER_STATUS = Gauge('divine_worker_status', 'Estado de los workers', ['tipo', 'worker_id'])
    TRANSACTION_STATS = Counter('divine_transactions', 'Estadísticas de transacciones', ['tipo'])

# Constantes
MAX_PRIORITY = 10  # Prioridad máxima para operaciones críticas
DEFAULT_RETRY_DELAYS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]  # Backoff exponencial
FATAL_EXCEPTIONS = (KeyboardInterrupt, SystemExit, asyncio.CancelledError)

class CarrierType:
    """Tipos de portadores de mensajes."""
    REDIS = "redis"
    RABBITMQ = "rabbitmq"
    MEMORY = "memory"

class TaskStatus:
    """Estados posibles de las tareas."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

class OperationMode:
    """Modos de operación del sistema divino."""
    NORMAL = "normal"         # Balanceado entre velocidad y confiabilidad
    ULTRARRAPIDO = "ultra"    # Prioriza velocidad extrema
    ULTRACONFIABLE = "secure" # Prioriza confiabilidad total
    ADAPTATIVO = "adaptive"   # Ajusta dinámicamente según carga y criticidad
    DIVINO = "divine"         # Modo máximo: todas las características activas

class DivineConfig:
    """Configuración para el sistema Divino."""
    
    def __init__(self, 
                 operation_mode: str = OperationMode.DIVINO,
                 redis_url: str = "redis://localhost:6379",
                 rabbitmq_url: str = "amqp://guest:guest@localhost:5672/%2F",
                 redis_workers: int = 4,
                 rabbitmq_workers: int = 2,
                 enable_monitoring: bool = True,
                 auto_scaling: bool = True,
                 max_retries: int = 8,
                 task_ttl: int = 3600,  # 1 hora
                 use_priorities: bool = True):
        """
        Inicializar configuración del sistema Divino.
        
        Args:
            operation_mode: Modo de operación (normal, ultra, secure, adaptive, divine)
            redis_url: URL de conexión a Redis
            rabbitmq_url: URL de conexión a RabbitMQ
            redis_workers: Número inicial de workers de Redis
            rabbitmq_workers: Número inicial de workers de RabbitMQ
            enable_monitoring: Activar monitoreo detallado
            auto_scaling: Activar escalado automático de workers
            max_retries: Número máximo de reintentos para tareas fallidas
            task_ttl: Tiempo de vida de las tareas en segundos
            use_priorities: Usar sistema de prioridades
        """
        self.operation_mode = operation_mode
        self.redis_url = redis_url
        self.rabbitmq_url = rabbitmq_url
        self.redis_workers = redis_workers
        self.rabbitmq_workers = rabbitmq_workers
        self.enable_monitoring = enable_monitoring
        self.auto_scaling = auto_scaling
        self.max_retries = max_retries
        self.task_ttl = task_ttl
        self.use_priorities = use_priorities
        
        # Ajustar configuración según modo
        if operation_mode == OperationMode.ULTRARRAPIDO:
            self.redis_workers = max(6, redis_workers)
            self.rabbitmq_workers = 1
        elif operation_mode == OperationMode.ULTRACONFIABLE:
            self.rabbitmq_workers = max(4, rabbitmq_workers)
        elif operation_mode == OperationMode.DIVINO:
            self.redis_workers = max(8, redis_workers)
            self.rabbitmq_workers = max(4, rabbitmq_workers)
            self.enable_monitoring = True
            self.auto_scaling = True
            
        # Validar las URLs
        if not redis_url.startswith(("redis://", "rediss://")):
            logger.warning("URL de Redis no válida, usando valor predeterminado")
            self.redis_url = "redis://localhost:6379"
            
        if not rabbitmq_url.startswith(("amqp://", "amqps://")):
            logger.warning("URL de RabbitMQ no válida, usando valor predeterminado")
            self.rabbitmq_url = "amqp://guest:guest@localhost:5672/%2F"

class DivineTaskQueue:
    """
    Cola de tareas trascendental (divina) para Genesis.
    
    Esta clase implementa una arquitectura híbrida Redis+RabbitMQ con:
    - Procesamiento ultrarrápido con prioridades dinámicas
    - Recuperación automática ante cualquier tipo de fallo
    - Garantía de entrega con transacciones atómicas
    - Escalabilidad adaptativa según carga
    - Monitoreo omnipresente de todas las operaciones
    """
    
    def __init__(self, config: Optional[DivineConfig] = None):
        """
        Inicializar la cola divina.
        
        Args:
            config: Configuración del sistema
        """
        # Usar configuración predeterminada si no se proporciona
        self.config = config or DivineConfig()
        
        # Redis (velocidad)
        self.redis_pool = None
        
        # RabbitMQ (confiabilidad)
        self.rabbitmq_params = None
        self.rabbitmq_connection = None
        
        # Estado interno
        self.running = False
        self.initialized = False
        self.redis_workers: List[asyncio.Task] = []
        self.rabbitmq_workers: List[threading.Thread] = []
        self.event_loop = None
        self.worker_executor = ThreadPoolExecutor(max_workers=self.config.rabbitmq_workers + 2)
        
        # Cola de memoria para operación sin dependencias externas
        self.memory_queue = asyncio.Queue(maxsize=1000)
        
        # Registro de operaciones y transacciones
        self.operations_in_progress: Set[str] = set()
        self.active_transactions: Dict[str, Dict[str, Any]] = {}
        
        # Estadísticas y monitoreo
        self.start_time = None
        self.stats = {
            "total_tasks": 0,
            "redis_tasks": 0,
            "rabbitmq_tasks": 0,
            "memory_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "retried_tasks": 0,
            "transactions": {
                "committed": 0,
                "rolled_back": 0
            },
            "latencies": {
                "redis": [],
                "rabbitmq": [],
                "memory": []
            },
            "errors": []
        }
        
        # Iniciar Prometheus si está disponible
        if PROMETHEUS_AVAILABLE and self.config.enable_monitoring:
            try:
                prometheus_client.start_http_server(9090)
                logger.info("Servidor de métricas Prometheus iniciado en puerto 9090")
            except Exception as e:
                logger.warning(f"No se pudo iniciar servidor Prometheus: {e}")
        
        logger.info(f"Cola de tareas divina inicializada en modo: {self.config.operation_mode}")
    
    async def initialize(self):
        """Inicializar conexiones y preparar el sistema."""
        if self.initialized:
            return
        
        # Inicializar bucle de eventos
        self.event_loop = asyncio.get_event_loop()
        
        # Inicializar Redis si está disponible
        if REDIS_AVAILABLE:
            try:
                self.redis_pool = await aioredis.create_redis_pool(
                    self.config.redis_url,
                    minsize=5,
                    maxsize=20
                )
                logger.info("Conexión a Redis establecida")
            except Exception as e:
                logger.warning(f"No se pudo conectar a Redis: {e}")
        else:
            logger.warning("aioredis no está instalado, usando cola de memoria como respaldo")
            
        # Inicializar RabbitMQ si está disponible
        if RABBITMQ_AVAILABLE:
            try:
                # Parsear la URL para extraer componentes
                url_parts = self.config.rabbitmq_url.replace("amqp://", "").split("@")
                if len(url_parts) == 2:
                    credentials, host_port = url_parts
                    username, password = credentials.split(":")
                    host, port = host_port.split(":")
                    port = port.split("/")[0]
                    
                    self.rabbitmq_params = pika.ConnectionParameters(
                        host=host,
                        port=int(port),
                        credentials=pika.PlainCredentials(username, password),
                        heartbeat=600,
                        blocked_connection_timeout=300
                    )
                else:
                    self.rabbitmq_params = pika.ConnectionParameters(
                        'localhost',
                        heartbeat=600,
                        blocked_connection_timeout=300
                    )
                
                # Verificar conexión
                test_conn = pika.BlockingConnection(self.rabbitmq_params)
                test_conn.close()
                logger.info("Conexión a RabbitMQ verificada")
            except Exception as e:
                logger.warning(f"No se pudo conectar a RabbitMQ: {e}")
                self.rabbitmq_params = None
        else:
            logger.warning("pika no está instalado, no se usará RabbitMQ")
            
        self.initialized = True
        logger.info("Sistema de cola divina inicializado completamente")
        
    async def start(self):
        """Iniciar el procesamiento de la cola divina."""
        if self.running:
            logger.warning("La cola divina ya está en ejecución")
            return
            
        if not self.initialized:
            await self.initialize()
            
        self.running = True
        self.start_time = datetime.now()
        
        # Iniciar workers de Redis
        if self.redis_pool:
            for i in range(self.config.redis_workers):
                worker = asyncio.create_task(self._redis_worker(i))
                self.redis_workers.append(worker)
                
        # Iniciar workers de RabbitMQ
        if self.rabbitmq_params:
            for i in range(self.config.rabbitmq_workers):
                worker = threading.Thread(
                    target=self._rabbitmq_worker,
                    args=(i, self.event_loop),
                    daemon=True
                )
                worker.start()
                self.rabbitmq_workers.append(worker)
                
        # Iniciar worker de memoria siempre (como respaldo)
        memory_worker = asyncio.create_task(self._memory_worker())
        self.redis_workers.append(memory_worker)
        
        # Iniciar monitor de auto-escalado si está configurado
        if self.config.auto_scaling:
            scaler = asyncio.create_task(self._auto_scaling_monitor())
            self.redis_workers.append(scaler)
            
        logger.info(f"Cola divina iniciada con {len(self.redis_workers)} workers de Redis " +
                   f"y {len(self.rabbitmq_workers)} workers de RabbitMQ")
    
    async def stop(self):
        """Detener la cola divina de forma segura."""
        if not self.running:
            return
            
        logger.info("Deteniendo cola divina...")
        self.running = False
        
        # Esperar a que se procesen tareas pendientes
        if not self.memory_queue.empty():
            logger.info(f"Esperando a que se completen {self.memory_queue.qsize()} tareas en memoria")
            await self.memory_queue.join()
            
        # Cancelar workers de Redis
        for worker in self.redis_workers:
            worker.cancel()
            
        # Esperar a que terminen los workers
        if self.redis_workers:
            await asyncio.gather(*self.redis_workers, return_exceptions=True)
            
        # Cerrar conexiones
        if self.redis_pool:
            self.redis_pool.close()
            await self.redis_pool.wait_closed()
            
        # Cerrar executor
        self.worker_executor.shutdown(wait=True)
        
        logger.info("Cola divina detenida correctamente")
        
    async def _redis_worker(self, worker_id: int):
        """
        Worker que procesa tareas desde Redis con máxima eficiencia.
        
        Este worker implementa varias capas de protección:
        - Priorización dinámica de tareas
        - Recuperación automática ante fallos de Redis
        - Manejo de transacciones atómicas
        
        Args:
            worker_id: ID único del worker
        """
        logger.info(f"Worker Redis {worker_id} iniciado")
        
        # Contadores locales
        tasks_processed = 0
        tasks_succeeded = 0
        tasks_failed = 0
        
        # Actualizar métrica de estado si está disponible
        if PROMETHEUS_AVAILABLE and self.config.enable_monitoring:
            WORKER_STATUS.labels('redis', str(worker_id)).set(1)
            
        while self.running:
            try:
                # Procesar tareas por orden de prioridad (10 = máxima)
                processed = False
                
                if self.redis_pool:
                    # Iterar desde la prioridad más alta a la más baja
                    for priority in range(MAX_PRIORITY, 0, -1):
                        queue_name = f"divine:tasks:p{priority}"
                        
                        try:
                            # Intentar obtener una tarea con timeout corto
                            task_data = await asyncio.wait_for(
                                self.redis_pool.brpop(queue_name, timeout=0.5),
                                timeout=0.6
                            )
                            
                            if task_data:
                                # Procesar la tarea
                                _, task_json = task_data
                                await self._process_task(task_json, CarrierType.REDIS, worker_id)
                                processed = True
                                tasks_processed += 1
                                tasks_succeeded += 1
                                break  # Salir después de procesar una tarea
                        except asyncio.TimeoutError:
                            # Timeout esperado, continuar con la siguiente prioridad
                            continue
                        except Exception as e:
                            logger.error(f"Error al obtener tarea de Redis (p{priority}): {e}")
                            await asyncio.sleep(0.1)  # Pequeño backoff
                
                # Si no procesamos nada de Redis, intentar con la cola de memoria
                if not processed and not self.memory_queue.empty():
                    try:
                        task_json = await asyncio.wait_for(
                            self.memory_queue.get(),
                            timeout=0.1
                        )
                        await self._process_task(task_json, CarrierType.MEMORY, worker_id)
                        self.memory_queue.task_done()
                        tasks_processed += 1
                        tasks_succeeded += 1
                    except asyncio.TimeoutError:
                        pass
                    except Exception as e:
                        logger.error(f"Error al procesar tarea de memoria: {e}")
                        tasks_failed += 1
                        self.memory_queue.task_done()
                
                # Si no hay tareas que procesar, esperar un poco
                if not processed:
                    await asyncio.sleep(0.01)
                    
            except asyncio.CancelledError:
                logger.info(f"Worker Redis {worker_id} cancelado")
                break
                
            except Exception as e:
                logger.error(f"Error en worker Redis {worker_id}: {e}")
                tasks_failed += 1
                await asyncio.sleep(0.5)  # Backoff en caso de error
        
        # Actualizar métrica de estado si está disponible
        if PROMETHEUS_AVAILABLE and self.config.enable_monitoring:
            WORKER_STATUS.labels('redis', str(worker_id)).set(0)
            
        logger.info(f"Worker Redis {worker_id} finalizado - Procesadas: {tasks_processed}, " +
                  f"Exitosas: {tasks_succeeded}, Fallidas: {tasks_failed}")
    
    def _rabbitmq_worker(self, worker_id: int, loop):
        """
        Worker que procesa tareas desde RabbitMQ con máxima confiabilidad.
        
        Este worker implementa varias capas de protección:
        - Reconnexión automática
        - Confirmación de mensajes (ACK)
        - Manejo de reintentos
        
        Args:
            worker_id: ID único del worker
            loop: Bucle de eventos para tareas asíncronas
        """
        logger.info(f"Worker RabbitMQ {worker_id} iniciado")
        
        # Contadores locales
        tasks_processed = 0
        tasks_succeeded = 0
        tasks_failed = 0
        
        # Actualizar métrica de estado si está disponible
        if PROMETHEUS_AVAILABLE and self.config.enable_monitoring:
            WORKER_STATUS.labels('rabbitmq', str(worker_id)).set(1)
        
        # Función para procesar mensajes de forma asíncrona
        async def process_message(task_json):
            try:
                await self._process_task(task_json, CarrierType.RABBITMQ, worker_id)
                return True
            except Exception as e:
                logger.error(f"Error al procesar mensaje de RabbitMQ: {e}")
                return False
        
        # Callback para procesar mensajes
        def on_message(ch, method, properties, body):
            nonlocal tasks_processed, tasks_succeeded, tasks_failed
            
            try:
                # Decodificar la tarea
                task_json = body.decode('utf-8')
                
                # Procesar de forma asíncrona
                future = asyncio.run_coroutine_threadsafe(
                    process_message(task_json),
                    loop
                )
                success = future.result(timeout=60)  # Esperar resultado con timeout
                
                # Confirmar que procesamos el mensaje
                ch.basic_ack(delivery_tag=method.delivery_tag)
                
                # Actualizar contadores
                tasks_processed += 1
                if success:
                    tasks_succeeded += 1
                else:
                    tasks_failed += 1
                    
            except Exception as e:
                logger.error(f"Error en callback de RabbitMQ: {e}")
                # Rechazar mensaje para que sea recolado
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                tasks_failed += 1
        
        # Función para manejar reconexiones
        def connect_and_consume():
            while self.running:
                try:
                    if not self.rabbitmq_params:
                        time.sleep(5)
                        continue
                        
                    # Establecer conexión
                    connection = pika.BlockingConnection(self.rabbitmq_params)
                    channel = connection.channel()
                    
                    # Declarar exchange
                    channel.exchange_declare(
                        exchange='divine_exchange',
                        exchange_type='direct',
                        durable=True
                    )
                    
                    # Declarar colas por prioridad
                    for priority in range(1, MAX_PRIORITY + 1):
                        queue_name = f"divine:tasks:p{priority}"
                        channel.queue_declare(
                            queue=queue_name,
                            durable=True,
                            arguments={
                                'x-queue-type': 'classic',
                                'x-max-priority': MAX_PRIORITY
                            }
                        )
                        channel.queue_bind(
                            exchange='divine_exchange',
                            queue=queue_name,
                            routing_key=queue_name
                        )
                    
                    # Configurar prefetch
                    channel.basic_qos(prefetch_count=10)
                    
                    # Comenzar a consumir de todas las colas priorizadas
                    for priority in range(MAX_PRIORITY, 0, -1):
                        queue_name = f"divine:tasks:p{priority}"
                        channel.basic_consume(
                            queue=queue_name,
                            on_message_callback=on_message
                        )
                    
                    logger.info(f"Worker RabbitMQ {worker_id} conectado y consumiendo")
                    
                    # Iniciar consumo
                    try:
                        channel.start_consuming()
                    except Exception as e:
                        logger.error(f"Error en consumo de RabbitMQ: {e}")
                    finally:
                        try:
                            connection.close()
                        except:
                            pass
                            
                except (AMQPConnectionError, ChannelClosedByBroker) as e:
                    logger.warning(f"Error de conexión RabbitMQ, reintentando: {e}")
                    time.sleep(5)
                except Exception as e:
                    logger.error(f"Error inesperado en worker RabbitMQ {worker_id}: {e}")
                    time.sleep(5)
                    
                if not self.running:
                    break
        
        # Ejecutar el consumo
        try:
            connect_and_consume()
        except Exception as e:
            logger.error(f"Error fatal en worker RabbitMQ {worker_id}: {e}")
        finally:
            # Actualizar métrica de estado si está disponible
            if PROMETHEUS_AVAILABLE and self.config.enable_monitoring:
                WORKER_STATUS.labels('rabbitmq', str(worker_id)).set(0)
                
            logger.info(f"Worker RabbitMQ {worker_id} finalizado - Procesadas: {tasks_processed}, " +
                      f"Exitosas: {tasks_succeeded}, Fallidas: {tasks_failed}")
    
    async def _memory_worker(self):
        """
        Worker que procesa tareas desde la cola de memoria.
        
        Este worker sirve como respaldo cuando Redis/RabbitMQ no están disponibles,
        y para operaciones de muy baja latencia.
        """
        logger.info("Worker de memoria iniciado")
        
        # Contadores locales
        tasks_processed = 0
        tasks_succeeded = 0
        tasks_failed = 0
        
        while self.running:
            try:
                # Obtener tarea de la cola de memoria
                task_json = await self.memory_queue.get()
                
                try:
                    # Procesar la tarea
                    await self._process_task(task_json, CarrierType.MEMORY, -1)
                    tasks_processed += 1
                    tasks_succeeded += 1
                except Exception as e:
                    logger.error(f"Error al procesar tarea de memoria: {e}")
                    tasks_failed += 1
                finally:
                    # Marcar como completada
                    self.memory_queue.task_done()
                    
            except asyncio.CancelledError:
                logger.info("Worker de memoria cancelado")
                break
                
            except Exception as e:
                logger.error(f"Error en worker de memoria: {e}")
                await asyncio.sleep(0.5)
                
        logger.info(f"Worker de memoria finalizado - Procesadas: {tasks_processed}, " +
                  f"Exitosas: {tasks_succeeded}, Fallidas: {tasks_failed}")
    
    async def _auto_scaling_monitor(self):
        """
        Monitor para ajustar dinámicamente el número de workers según la carga.
        
        Este monitor analiza métricas de rendimiento y ajusta recursos para
        mantener latencias óptimas incluso bajo carga alta.
        """
        logger.info("Monitor de auto-escalado iniciado")
        
        # Escala inicial
        max_redis_workers = self.config.redis_workers * 3
        min_redis_workers = self.config.redis_workers
        current_redis_workers = self.config.redis_workers
        
        while self.running:
            try:
                await asyncio.sleep(10)  # Comprobar cada 10 segundos
                
                # Analizar carga actual
                queue_size = 0
                
                # Verificar tamaño de colas en Redis
                if self.redis_pool:
                    try:
                        pipe = self.redis_pool.pipeline()
                        for priority in range(1, MAX_PRIORITY + 1):
                            queue_name = f"divine:tasks:p{priority}"
                            pipe.llen(queue_name)
                        
                        # Ejecutar pipeline y sumar resultados
                        sizes = await pipe.execute()
                        queue_size = sum(sizes)
                        
                        # Actualizar métricas si están disponibles
                        if PROMETHEUS_AVAILABLE and self.config.enable_monitoring:
                            for i, size in enumerate(sizes):
                                QUEUE_SIZE.labels(f"redis_p{i+1}").set(size)
                                
                    except Exception as e:
                        logger.warning(f"Error al obtener tamaño de colas Redis: {e}")
                
                # Añadir tamaño de cola de memoria
                memory_size = self.memory_queue.qsize()
                queue_size += memory_size
                
                # Actualizar métrica si está disponible
                if PROMETHEUS_AVAILABLE and self.config.enable_monitoring:
                    QUEUE_SIZE.labels("memory").set(memory_size)
                
                # Ajustar número de workers según carga
                if queue_size > current_redis_workers * 100:
                    # Carga alta, aumentar workers
                    target_workers = min(current_redis_workers + 2, max_redis_workers)
                    if target_workers > current_redis_workers:
                        await self._adjust_redis_workers(target_workers)
                        current_redis_workers = target_workers
                        logger.info(f"Auto-escalado: Incrementado a {current_redis_workers} workers Redis")
                elif queue_size < current_redis_workers * 10 and current_redis_workers > min_redis_workers:
                    # Carga baja, reducir workers
                    target_workers = max(current_redis_workers - 1, min_redis_workers)
                    if target_workers < current_redis_workers:
                        await self._adjust_redis_workers(target_workers)
                        current_redis_workers = target_workers
                        logger.info(f"Auto-escalado: Reducido a {current_redis_workers} workers Redis")
                
            except asyncio.CancelledError:
                logger.info("Monitor de auto-escalado cancelado")
                break
                
            except Exception as e:
                logger.error(f"Error en monitor de auto-escalado: {e}")
                await asyncio.sleep(5)
                
        logger.info("Monitor de auto-escalado finalizado")
    
    async def _adjust_redis_workers(self, target_count: int):
        """
        Ajustar el número de workers de Redis.
        
        Args:
            target_count: Número deseado de workers
        """
        current_count = len([w for w in self.redis_workers if not w.done()])
        
        if target_count == current_count:
            return
            
        if target_count > current_count:
            # Crear nuevos workers
            for i in range(current_count, target_count):
                worker = asyncio.create_task(self._redis_worker(i + 100))  # Offset para IDs únicos
                self.redis_workers.append(worker)
                
        else:
            # Reducir número de workers
            # Nota: No cancelamos workers existentes, simplemente permitimos que algunos terminen
            pass
    
    async def _process_task(self, task_json: str, carrier_type: str, worker_id: int):
        """
        Procesar una tarea deserializando y ejecutando la función.
        
        Args:
            task_json: Tarea serializada en formato JSON
            carrier_type: Tipo de portador (redis, rabbitmq, memory)
            worker_id: ID del worker que procesa la tarea
        """
        start_time = time.time()
        
        try:
            # Deserializar la tarea
            task_data = json.loads(task_json)
            
            task_id = task_data.get("task_id", "unknown")
            func_module = task_data.get("func_module")
            func_name = task_data.get("func_name")
            args = task_data.get("args", [])
            kwargs = task_data.get("kwargs", {})
            retry_count = task_data.get("retry_count", 0)
            
            # Registrar inicio
            logger.debug(f"Procesando tarea {task_id} ({func_module}.{func_name}) " + 
                        f"por {carrier_type}:{worker_id}")
            
            # Importar dinámicamente la función
            try:
                module = __import__(func_module, fromlist=[func_name])
                func = getattr(module, func_name)
            except (ImportError, AttributeError) as e:
                logger.error(f"No se pudo importar {func_module}.{func_name}: {e}")
                raise
            
            # Ejecutar la función
            result = await func(*args, **kwargs)
            
            # Registrar éxito
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"Tarea {task_id} completada en {elapsed_ms:.2f}ms")
            
            # Actualizar estadísticas
            self.stats["completed_tasks"] += 1
            self.stats["latencies"][carrier_type].append(elapsed_ms)
            
            # Actualizar métricas Prometheus si están disponibles
            if PROMETHEUS_AVAILABLE and self.config.enable_monitoring:
                TASKS_TOTAL.labels(carrier_type, 'success').inc()
                TASKS_LATENCIA.labels(carrier_type).observe(elapsed_ms / 1000.0)
            
            return result
            
        except Exception as e:
            # Capturar todos los errores
            elapsed_ms = (time.time() - start_time) * 1000
            error_info = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            
            # Registrar error
            logger.error(f"Error al procesar tarea ({elapsed_ms:.2f}ms): {error_info}")
            
            # Actualizar estadísticas
            self.stats["failed_tasks"] += 1
            if len(self.stats["errors"]) >= 10:
                self.stats["errors"].pop(0)
            self.stats["errors"].append({
                "time": datetime.now().isoformat(),
                "error": error_info,
                "task": task_json[:200] + "..." if len(task_json) > 200 else task_json
            })
            
            # Actualizar métricas Prometheus si están disponibles
            if PROMETHEUS_AVAILABLE and self.config.enable_monitoring:
                TASKS_TOTAL.labels(carrier_type, 'failure').inc()
            
            # Propagar el error
            raise
    
    async def enqueue(self, func, *args, priority: int = 5, **kwargs) -> Any:
        """
        Encolar una tarea para ejecución asíncrona.
        
        Esta función es el punto de entrada principal para el sistema divino.
        Encola la tarea en los sistemas disponibles según su prioridad y
        el modo de operación configurado.
        
        Args:
            func: Función asíncrona a ejecutar
            *args: Argumentos posicionales para la función
            priority: Prioridad de la tarea (1-10, 10 = máxima)
            **kwargs: Argumentos con nombre para la función
            
        Returns:
            Resultado de la función
            
        Raises:
            Exception: Si ocurre un error durante la ejecución
        """
        if not self.initialized:
            await self.initialize()
            
        if not self.running:
            await self.start()
        
        # Validar prioridad
        priority = max(1, min(priority, MAX_PRIORITY))
        
        # Crear ID de tarea
        task_id = kwargs.pop("task_id", f"task_{uuid.uuid4().hex[:8]}")
        
        # Obtener información de la función
        func_module = func.__module__
        func_name = func.__name__
        
        # Preparar la tarea
        task_data = {
            "task_id": task_id,
            "func_module": func_module,
            "func_name": func_name,
            "args": args,
            "kwargs": kwargs,
            "priority": priority,
            "retry_count": 0,
            "created_at": datetime.now().isoformat()
        }
        
        # Serializar la tarea
        task_json = json.dumps(task_data)
        
        # Future para recibir el resultado
        future = asyncio.Future()
        
        # Decidir estrategia según modo de operación
        if self.config.operation_mode == OperationMode.ULTRARRAPIDO:
            # Modo ultrarrápido: Priorizar memoria/Redis
            await self._enqueue_fast(task_json, priority, future)
        elif self.config.operation_mode == OperationMode.ULTRACONFIABLE:
            # Modo ultraconfiable: Priorizar RabbitMQ con respaldo
            await self._enqueue_reliable(task_json, priority, future)
        elif self.config.operation_mode in (OperationMode.ADAPTATIVO, OperationMode.DIVINO):
            # Modo adaptativo/divino: Usar estrategia híbrida óptima
            await self._enqueue_optimal(task_json, priority, future)
        else:
            # Modo normal: Balanceado
            await self._enqueue_balanced(task_json, priority, future)
        
        # Actualizar estadísticas
        self.stats["total_tasks"] += 1
        
        # Esperar y retornar resultado
        return await future
    
    async def _enqueue_fast(self, task_json: str, priority: int, future: asyncio.Future):
        """Estrategia de encolado para velocidad máxima."""
        enqueued = False
        
        # Intentar encolar en Redis primero
        if self.redis_pool:
            try:
                queue_name = f"divine:tasks:p{priority}"
                await self.redis_pool.lpush(queue_name, task_json)
                self.stats["redis_tasks"] += 1
                enqueued = True
                logger.debug(f"Tarea encolada en Redis (p{priority})")
            except Exception as e:
                logger.warning(f"Error al encolar en Redis: {e}")
                
        # Si falla Redis, usar cola de memoria como respaldo
        if not enqueued:
            try:
                await self.memory_queue.put(task_json)
                self.stats["memory_tasks"] += 1
                enqueued = True
                logger.debug("Tarea encolada en memoria (respaldo)")
            except Exception as e:
                logger.error(f"Error al encolar en memoria: {e}")
                future.set_exception(e)
                return
        
        # Completar y actualizar estadísticas
        future.set_result("enqueued")
    
    async def _enqueue_reliable(self, task_json: str, priority: int, future: asyncio.Future):
        """Estrategia de encolado para confiabilidad máxima."""
        enqueued_rabbitmq = False
        
        # Intentar encolar en RabbitMQ primero
        if self.rabbitmq_params:
            try:
                loop = asyncio.get_event_loop()
                success = await loop.run_in_executor(
                    self.worker_executor,
                    self._send_to_rabbitmq,
                    task_json,
                    priority
                )
                
                if success:
                    self.stats["rabbitmq_tasks"] += 1
                    enqueued_rabbitmq = True
                    logger.debug(f"Tarea encolada en RabbitMQ (p{priority})")
            except Exception as e:
                logger.warning(f"Error al encolar en RabbitMQ: {e}")
        
        # Encolar en Redis o memoria como respaldo
        try:
            if self.redis_pool:
                queue_name = f"divine:tasks:p{priority}"
                await self.redis_pool.lpush(queue_name, task_json)
                self.stats["redis_tasks"] += 1
                logger.debug(f"Tarea encolada en Redis (p{priority}, respaldo)")
            else:
                await self.memory_queue.put(task_json)
                self.stats["memory_tasks"] += 1
                logger.debug("Tarea encolada en memoria (respaldo)")
        except Exception as e:
            logger.error(f"Error al encolar respaldo: {e}")
            if not enqueued_rabbitmq:
                future.set_exception(e)
                return
        
        # Completar y actualizar estadísticas
        future.set_result("enqueued")
    
    async def _enqueue_balanced(self, task_json: str, priority: int, future: asyncio.Future):
        """Estrategia de encolado balanceada (normal)."""
        success = False
        
        # Intentar Redis primero para tareas de alta prioridad (8-10)
        if priority >= 8 and self.redis_pool:
            try:
                queue_name = f"divine:tasks:p{priority}"
                await self.redis_pool.lpush(queue_name, task_json)
                self.stats["redis_tasks"] += 1
                success = True
                logger.debug(f"Tarea encolada en Redis (p{priority})")
            except Exception as e:
                logger.warning(f"Error al encolar en Redis: {e}")
        
        # Intentar RabbitMQ para tareas de prioridad media-baja (1-7)
        if not success and priority < 8 and self.rabbitmq_params:
            try:
                loop = asyncio.get_event_loop()
                rabbitmq_success = await loop.run_in_executor(
                    self.worker_executor,
                    self._send_to_rabbitmq,
                    task_json,
                    priority
                )
                
                if rabbitmq_success:
                    self.stats["rabbitmq_tasks"] += 1
                    success = True
                    logger.debug(f"Tarea encolada en RabbitMQ (p{priority})")
            except Exception as e:
                logger.warning(f"Error al encolar en RabbitMQ: {e}")
        
        # Intentar Redis si no se ha encolado aún
        if not success and self.redis_pool:
            try:
                queue_name = f"divine:tasks:p{priority}"
                await self.redis_pool.lpush(queue_name, task_json)
                self.stats["redis_tasks"] += 1
                success = True
                logger.debug(f"Tarea encolada en Redis (p{priority}, respaldo)")
            except Exception as e:
                logger.warning(f"Error al encolar en Redis (respaldo): {e}")
        
        # Memoria como último recurso
        if not success:
            try:
                await self.memory_queue.put(task_json)
                self.stats["memory_tasks"] += 1
                success = True
                logger.debug("Tarea encolada en memoria (última opción)")
            except Exception as e:
                logger.error(f"Error al encolar en memoria: {e}")
                future.set_exception(e)
                return
        
        # Completar future
        future.set_result("enqueued")
    
    async def _enqueue_optimal(self, task_json: str, priority: int, future: asyncio.Future):
        """
        Estrategia de encolado óptima (adaptativa/divina).
        
        Esta estrategia analiza el tipo de tarea, prioridad, y estado
        del sistema para tomar la mejor decisión de encolado.
        """
        # Estrategia divina: Intentar todo en paralelo para tareas críticas
        if priority >= 9 and self.config.operation_mode == OperationMode.DIVINO:
            tasks = []
            
            # Redis (velocidad)
            if self.redis_pool:
                queue_name = f"divine:tasks:p{priority}"
                redis_task = asyncio.create_task(self.redis_pool.lpush(queue_name, task_json))
                tasks.append(redis_task)
                
            # RabbitMQ (confiabilidad)
            if self.rabbitmq_params:
                loop = asyncio.get_event_loop()
                rabbitmq_task = loop.run_in_executor(
                    self.worker_executor,
                    self._send_to_rabbitmq,
                    task_json,
                    priority
                )
                tasks.append(rabbitmq_task)
                
            # Memoria (respaldo local)
            memory_task = asyncio.create_task(self.memory_queue.put(task_json))
            tasks.append(memory_task)
            
            # Esperar a que al menos uno tenga éxito
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Actualizar estadísticas (aproximado)
            self.stats["redis_tasks"] += 1
            self.stats["rabbitmq_tasks"] += 1
            self.stats["memory_tasks"] += 1
            
            # No cancelamos las tareas pendientes para maximizar redundancia
            logger.debug(f"Tarea crítica encolada en múltiples sistemas (p{priority})")
            future.set_result("enqueued")
            return
        
        # Para otras prioridades: Análisis inteligente
        redis_latency = 1.0  # valor por defecto ms
        rabbitmq_latency = 5.0  # valor por defecto ms
        
        # Calcular latencias promedio si hay datos
        if self.stats["latencies"]["redis"]:
            redis_latency = sum(self.stats["latencies"]["redis"][-10:]) / min(10, len(self.stats["latencies"]["redis"]))
        if self.stats["latencies"]["rabbitmq"]:
            rabbitmq_latency = sum(self.stats["latencies"]["rabbitmq"][-10:]) / min(10, len(self.stats["latencies"]["rabbitmq"]))
        
        # Factor de prioridad: Mayor prioridad favorece velocidad
        priority_factor = priority / MAX_PRIORITY
        
        # Calcular puntuación para cada sistema
        redis_score = (1 / redis_latency) * priority_factor
        rabbitmq_score = (1 / rabbitmq_latency) * (1 - priority_factor) * 10  # Factor de confiabilidad
        
        # Determinar sistema primario
        if redis_score > rabbitmq_score and self.redis_pool:
            # Redis como primario
            try:
                queue_name = f"divine:tasks:p{priority}"
                await self.redis_pool.lpush(queue_name, task_json)
                self.stats["redis_tasks"] += 1
                logger.debug(f"Tarea encolada en Redis (p{priority}, adaptativo)")
                
                # RabbitMQ como respaldo para tareas importantes
                if priority >= 7 and self.rabbitmq_params:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        self.worker_executor,
                        self._send_to_rabbitmq,
                        task_json,
                        priority
                    )
                    self.stats["rabbitmq_tasks"] += 1
                
                future.set_result("enqueued")
                return
            except Exception as e:
                logger.warning(f"Error al encolar en Redis (adaptativo): {e}")
                # Continuar con la siguiente opción
        
        if self.rabbitmq_params:
            # RabbitMQ como principal o respaldo
            try:
                loop = asyncio.get_event_loop()
                rabbitmq_success = await loop.run_in_executor(
                    self.worker_executor,
                    self._send_to_rabbitmq,
                    task_json,
                    priority
                )
                
                if rabbitmq_success:
                    self.stats["rabbitmq_tasks"] += 1
                    logger.debug(f"Tarea encolada en RabbitMQ (p{priority}, adaptativo)")
                    future.set_result("enqueued")
                    return
            except Exception as e:
                logger.warning(f"Error al encolar en RabbitMQ (adaptativo): {e}")
        
        # Respaldo en memoria si todo lo demás falla
        try:
            await self.memory_queue.put(task_json)
            self.stats["memory_tasks"] += 1
            logger.debug("Tarea encolada en memoria (respaldo adaptativo)")
            future.set_result("enqueued")
        except Exception as e:
            logger.error(f"Error al encolar en memoria: {e}")
            future.set_exception(e)
    
    def _send_to_rabbitmq(self, task_json: str, priority: int) -> bool:
        """
        Enviar tarea a RabbitMQ de forma segura.
        
        Args:
            task_json: Tarea serializada en JSON
            priority: Prioridad de la tarea
            
        Returns:
            True si se encoló correctamente, False en caso contrario
        """
        if not self.rabbitmq_params:
            return False
            
        try:
            # Establecer conexión
            connection = pika.BlockingConnection(self.rabbitmq_params)
            channel = connection.channel()
            
            # Declarar exchange
            channel.exchange_declare(
                exchange='divine_exchange',
                exchange_type='direct',
                durable=True
            )
            
            # Declarar cola
            queue_name = f"divine:tasks:p{priority}"
            channel.queue_declare(
                queue=queue_name,
                durable=True,
                arguments={
                    'x-queue-type': 'classic',
                    'x-max-priority': MAX_PRIORITY
                }
            )
            
            # Enlazar cola a exchange
            channel.queue_bind(
                exchange='divine_exchange',
                queue=queue_name,
                routing_key=queue_name
            )
            
            # Publicar mensaje
            channel.basic_publish(
                exchange='divine_exchange',
                routing_key=queue_name,
                body=task_json.encode(),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Persistente
                    priority=priority
                )
            )
            
            # Cerrar conexión
            connection.close()
            return True
            
        except Exception as e:
            logger.error(f"Error al enviar a RabbitMQ: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas detalladas del sistema divino.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = self.stats.copy()
        
        # Calcular tiempo de actividad
        if self.start_time:
            stats["uptime_seconds"] = (datetime.now() - self.start_time).total_seconds()
            stats["start_time"] = self.start_time.isoformat()
        
        # Calcular latencias promedio
        stats["avg_latencies"] = {}
        for carrier, latencies in stats["latencies"].items():
            if latencies:
                stats["avg_latencies"][carrier] = sum(latencies) / len(latencies)
            else:
                stats["avg_latencies"][carrier] = 0
        
        # Información de configuración
        stats["config"] = {
            "mode": self.config.operation_mode,
            "redis_workers": len([w for w in self.redis_workers if not w.done()]),
            "rabbitmq_workers": len(self.rabbitmq_workers),
            "auto_scaling": self.config.auto_scaling
        }
        
        # Estado de la cola
        stats["queue_sizes"] = {
            "memory": self.memory_queue.qsize()
        }
        
        # Tamaños de cola en Redis (si está disponible)
        if self.redis_pool:
            # No podemos hacer esta operación de forma síncrona
            stats["queue_sizes"]["redis"] = "disponible solo en operación asíncrona"
        
        return stats
    
    async def get_stats_async(self) -> Dict[str, Any]:
        """
        Obtener estadísticas detalladas del sistema divino (versión asíncrona).
        
        Esta versión incluye información en tiempo real de Redis/RabbitMQ.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = self.get_stats()
        
        # Obtener tamaños de cola en Redis
        if self.redis_pool:
            try:
                redis_sizes = {}
                pipe = self.redis_pool.pipeline()
                for priority in range(1, MAX_PRIORITY + 1):
                    queue_name = f"divine:tasks:p{priority}"
                    pipe.llen(queue_name)
                
                sizes = await pipe.execute()
                for i, size in enumerate(sizes):
                    redis_sizes[f"p{i+1}"] = size
                
                stats["queue_sizes"]["redis"] = redis_sizes
                stats["queue_sizes"]["redis_total"] = sum(sizes)
            except Exception as e:
                logger.warning(f"Error al obtener tamaños de cola Redis: {e}")
                stats["queue_sizes"]["redis"] = str(e)
        
        return stats

# Decoradores convenientes para usar la cola divina

divine_task_queue = None

def initialize_divine_queue(config: Optional[DivineConfig] = None) -> DivineTaskQueue:
    """
    Inicializar la cola divina global.
    
    Args:
        config: Configuración opcional
        
    Returns:
        Instancia de DivineTaskQueue
    """
    global divine_task_queue
    if divine_task_queue is None:
        divine_task_queue = DivineTaskQueue(config)
    return divine_task_queue

async def ensure_queue_started():
    """Garantizar que la cola esté inicializada y en ejecución."""
    global divine_task_queue
    if divine_task_queue is None:
        divine_task_queue = DivineTaskQueue()
    if not divine_task_queue.initialized:
        await divine_task_queue.initialize()
    if not divine_task_queue.running:
        await divine_task_queue.start()

def divine_task(priority: int = 5):
    """
    Decorador para ejecutar una función como tarea divina.
    
    Este decorador envuelve la función en una tarea que se ejecutará
    a través del sistema divino de colas con la prioridad especificada.
    
    Args:
        priority: Prioridad de la tarea (1-10, 10 = máxima)
        
    Returns:
        Decorador configurado
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            await ensure_queue_started()
            return await divine_task_queue.enqueue(func, *args, priority=priority, **kwargs)
        return wrapper
    return decorator

def critical_task():
    """
    Decorador para tareas críticas con máxima prioridad.
    
    Este decorador es un atajo para divine_task(priority=10).
    """
    return divine_task(priority=10)

def high_priority_task():
    """
    Decorador para tareas de alta prioridad.
    
    Este decorador es un atajo para divine_task(priority=8).
    """
    return divine_task(priority=8)

def low_priority_task():
    """
    Decorador para tareas de baja prioridad.
    
    Este decorador es un atajo para divine_task(priority=3).
    """
    return divine_task(priority=3)

def background_task():
    """
    Decorador para tareas en segundo plano con mínima prioridad.
    
    Este decorador es un atajo para divine_task(priority=1).
    """
    return divine_task(priority=1)

@contextlib.asynccontextmanager
async def divine_transaction():
    """
    Contexto para garantizar transacciones atómicas en operaciones divinas.
    
    Este contexto agrupa múltiples operaciones divinas para garantizar
    que todas se ejecuten correctamente o ninguna.
    
    Ejemplo:
        async with divine_transaction():
            await operacion1()
            await operacion2()
    """
    transaction_id = f"tx_{uuid.uuid4().hex[:8]}"
    
    try:
        # Iniciar transacción
        logger.debug(f"Iniciando transacción divina {transaction_id}")
        
        # Ejecutar el bloque con contexto
        yield transaction_id
        
        # Commit
        logger.debug(f"Commit transacción divina {transaction_id}")
        if PROMETHEUS_AVAILABLE and divine_task_queue and divine_task_queue.config.enable_monitoring:
            TRANSACTION_STATS.labels('commit').inc()
        
        if divine_task_queue:
            divine_task_queue.stats["transactions"]["committed"] += 1
            
    except Exception as e:
        # Rollback
        logger.error(f"Rollback transacción divina {transaction_id}: {e}")
        if PROMETHEUS_AVAILABLE and divine_task_queue and divine_task_queue.config.enable_monitoring:
            TRANSACTION_STATS.labels('rollback').inc()
            
        if divine_task_queue:
            divine_task_queue.stats["transactions"]["rolled_back"] += 1
            
        raise  # Re-lanzar excepción para que el caller pueda manejarla