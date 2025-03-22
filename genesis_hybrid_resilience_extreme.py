"""
Sistema Genesis híbrido con optimizaciones extremas de resiliencia.

Esta versión incorpora optimizaciones avanzadas adicionales para llevar
el rendimiento del 93.58% al rango 95-98% bajo condiciones extremas:

Mejoras principales:
- Timeout global para evitar que operaciones lentas consuman tiempo excesivo
- Circuit Breaker con modo predictivo para anticipar fallos
- Checkpointing diferencial y comprimido para mayor eficiencia
- Procesamiento por lotes para colas con alta carga
- Modo PRE-SAFE para transiciones más suaves entre estados del sistema
- Optimización de jitter para reintentos más rápidos en componentes esenciales
"""

import asyncio
import logging
import time
import random
import json
import zlib
import base64
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Set, Tuple, Callable, Union, Coroutine

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enumeraciones para estados
class CircuitState(Enum):
    """Estados posibles del Circuit Breaker."""
    CLOSED = auto()        # Funcionamiento normal
    PARTIAL = auto()       # Parcialmente abierto, mayor monitoreo
    OPEN = auto()          # Circuito abierto, rechaza llamadas
    HALF_OPEN = auto()     # Semi-abierto, permite algunas llamadas

class SystemMode(Enum):
    """Modos de operación del sistema."""
    NORMAL = "normal"       # Funcionamiento normal
    PRE_SAFE = "pre_safe"   # Modo precaución, monitoreo intensivo
    SAFE = "safe"           # Modo seguro
    EMERGENCY = "emergency" # Modo emergencia

class EventPriority(Enum):
    """Prioridades para eventos."""
    CRITICAL = 0    # Eventos críticos (ej. alertas de seguridad)
    HIGH = 1        # Eventos importantes (ej. operaciones de trading)
    NORMAL = 2      # Eventos regulares
    LOW = 3         # Eventos de baja prioridad (ej. actualizaciones UI)
    BACKGROUND = 4  # Eventos de fondo, pueden descartarse bajo estrés

# Circuit Breaker con modo predictivo
class CircuitBreaker:
    """
    Implementación optimizada del Circuit Breaker con modo predictivo.
    
    Mejoras:
    - Modo predictivo para anticipar fallos
    - Recovery timeout reducido para componentes esenciales
    - Transición gradual con estado PARTIAL antes de OPEN
    - Métricas avanzadas para mejor monitoreo
    """
    
    def __init__(
        self, 
        name: str, 
        failure_threshold: int = 3,
        recovery_timeout: float = 1.0,
        half_open_max_calls: int = 1,
        success_threshold: int = 1,
        essential: bool = False
    ):
        """
        Inicializar Circuit Breaker.
        
        Args:
            name: Nombre para identificar el circuit breaker
            failure_threshold: Fallos consecutivos para abrir el circuito
            recovery_timeout: Tiempo hasta probar recuperación (segundos)
            half_open_max_calls: Máximo de llamadas en estado half-open
            success_threshold: Éxitos consecutivos para cerrar el circuito
            essential: Si es un componente esencial
        """
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_state_change = time.time()
        self.essential = essential
        
        # Parámetros ajustados según si es esencial
        if essential:
            # Componentes esenciales tienen recovery más rápido
            recovery_timeout = max(0.5, recovery_timeout / 2)
        
        # Configuración
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        
        # Estadísticas
        self.call_count = 0
        self.success_count_total = 0
        self.failure_count_total = 0
        self.rejection_count = 0
        self.recent_failures = []
        self.recent_latencies = []
        self.degradation_score = 0  # 0-100, para modo predictivo
        
    async def execute(self, func: Callable[..., Coroutine], *args, **kwargs) -> Any:
        """
        Ejecutar función con protección del Circuit Breaker.
        
        Args:
            func: Función a ejecutar
            *args, **kwargs: Argumentos para la función
            
        Returns:
            Resultado de la función o None si el circuito está abierto
            
        Raises:
            Exception: Si ocurre un error y el circuito no está abierto
        """
        self.call_count += 1
        
        # Modo predictivo: detectar degradación antes de que ocurran los fallos completos
        if self.state == CircuitState.CLOSED and self.degradation_score > 75:
            logger.warning(f"Circuit Breaker '{self.name}' en modo PARTIAL (predictivo)")
            self.state = CircuitState.PARTIAL
            self.last_state_change = time.time()
        
        # Si está abierto, verificar si debemos transicionar a half-open
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                logger.debug(f"Circuit Breaker '{self.name}' pasando a HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.last_state_change = time.time()
                self.success_count = 0  # Reset contador de éxitos
            else:
                self.rejection_count += 1
                return None  # Rechazar llamada, no lanzar excepción
        
        # Si está half-open, limitar calls
        if self.state == CircuitState.HALF_OPEN:
            # Verificar si estamos permitiendo más llamadas
            if self.success_count + self.failure_count >= self.half_open_max_calls:
                self.rejection_count += 1
                return None  # Rechazar llamada, no lanzar excepción
        
        # Si está en PARTIAL, registrar métricas pero permitir 90% de las llamadas
        if self.state == CircuitState.PARTIAL:
            if random.random() < 0.1:  # Rechazar 10% de llamadas para reducir carga
                self.rejection_count += 1
                return None
        
        # Ejecutar la función
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            
            # Medir latencia
            latency = time.time() - start_time
            self.recent_latencies.append(latency)
            if len(self.recent_latencies) > 10:
                self.recent_latencies.pop(0)
            
            # Actualizar degradation score (disminuir)
            self.degradation_score = max(0, self.degradation_score - 15)
            
            # Contabilizar éxito
            self.success_count_total += 1
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    logger.info(f"Circuit Breaker '{self.name}' cerrado tras {self.success_count} éxitos")
                    self.state = CircuitState.CLOSED
                    self.last_state_change = time.time()
                    self.failure_count = 0
                    self.degradation_score = 0
            elif self.state == CircuitState.PARTIAL:
                if self.degradation_score <= 25:
                    logger.info(f"Circuit Breaker '{self.name}' vuelve a CLOSED desde PARTIAL")
                    self.state = CircuitState.CLOSED
                    self.last_state_change = time.time()
            else:
                # En estado normal, resetear contador de fallos con cada éxito
                self.failure_count = 0
                
            return result
        except Exception as e:
            # Medir latencia incluso en error
            latency = time.time() - start_time
            
            # Contabilizar fallo
            self.failure_count_total += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # Registrar tipo de fallo para análisis
            self.recent_failures.append(str(type(e).__name__))
            if len(self.recent_failures) > 5:
                self.recent_failures.pop(0)
            
            # Actualizar degradation score (aumentar)
            self.degradation_score = min(100, self.degradation_score + 25)
            
            # Actualizar estado si necesario
            if self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
                logger.warning(f"Circuit Breaker '{self.name}' abierto tras {self.failure_count} fallos")
                self.state = CircuitState.OPEN
                self.last_state_change = time.time()
            elif self.state == CircuitState.PARTIAL:
                if self.failure_count >= self.failure_threshold - 1:
                    logger.warning(f"Circuit Breaker '{self.name}' pasa de PARTIAL a OPEN")
                    self.state = CircuitState.OPEN
                    self.last_state_change = time.time()
            elif self.state == CircuitState.HALF_OPEN:
                logger.warning(f"Circuit Breaker '{self.name}' vuelve a OPEN tras fallo en HALF_OPEN")
                self.state = CircuitState.OPEN
                self.last_state_change = time.time()
            
            # Propagar la excepción
            raise e
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del circuit breaker."""
        avg_latency = sum(self.recent_latencies) / max(len(self.recent_latencies), 1)
        return {
            "state": self.state.name,
            "call_count": self.call_count,
            "success_rate": (self.success_count_total / max(self.call_count, 1)) * 100,
            "failure_count": self.failure_count_total,
            "rejection_count": self.rejection_count,
            "avg_latency": avg_latency,
            "degradation_score": self.degradation_score,
            "last_failure": time.time() - self.last_failure_time if self.last_failure_time > 0 else None,
            "recent_failures": self.recent_failures
        }

# Sistema de reintentos adaptativos optimizado con timeout global
async def with_retry(
    func: Callable[..., Coroutine], 
    max_retries: int = 3, 
    base_delay: float = 0.05, 
    max_delay: float = 0.3,
    jitter: float = 0.05,
    global_timeout: float = 0.8,  # Tiempo máximo total incluyendo todos los reintentos
    essential: bool = False
) -> Any:
    """
    Ejecutar una función con reintentos adaptativos optimizados y timeout global.
    
    Mejoras:
    - Timeout global para limitar el tiempo total de operación
    - Jitter optimizado para componentes esenciales
    - Detección mejorada de éxito temprano
    
    Args:
        func: Función a ejecutar
        max_retries: Número máximo de reintentos
        base_delay: Tiempo base entre reintentos
        max_delay: Tiempo máximo entre reintentos
        jitter: Variación aleatoria máxima
        global_timeout: Tiempo máximo total para la operación
        essential: Si es un componente esencial
        
    Returns:
        Resultado de la función
        
    Raises:
        Exception: Si se agotan los reintentos o se excede el timeout global
    """
    # Ajustar parámetros para componentes esenciales
    if essential:
        # Menos jitter para respuestas más predecibles
        jitter *= 0.5
        # Priorizar rapidez en componentes esenciales
        base_delay *= 0.8
    
    start_time = time.time()
    retries = 0
    last_exception = None
    
    while retries <= max_retries:
        # Verificar si excedimos el timeout global
        if time.time() - start_time >= global_timeout:
            if last_exception:
                logger.warning(f"Timeout global alcanzado tras {retries} intentos. Último error: {str(last_exception)[:50]}")
                raise last_exception
            else:
                logger.warning(f"Timeout global alcanzado tras {retries} intentos sin excepción específica")
                raise asyncio.TimeoutError(f"Timeout global de {global_timeout}s excedido")
            
        try:
            result = await func()
            if result is not None:
                # Éxito temprano - retornar inmediatamente sin esperar más intentos
                return result
        except asyncio.TimeoutError as e:
            # Manejar timeouts específicamente
            last_exception = e
            retries += 1
            if retries > max_retries:
                break
                
            # Mayor backoff para timeouts (asumimos congestión)
            delay = min(base_delay * (2 ** retries) + random.uniform(0, jitter), max_delay)
            time_left = global_timeout - (time.time() - start_time)
            if time_left <= 0:
                break
                
            # Asegurar que el delay no exceda el tiempo restante
            delay = min(delay, time_left * 0.7)  # Usar solo 70% del tiempo restante para dejar margen
            
            logger.info(f"Reintento {retries}/{max_retries} tras timeout. Esperando {delay:.2f}s")
            await asyncio.sleep(delay)
            continue
        except Exception as e:
            # Otras excepciones
            last_exception = e
            retries += 1
            if retries > max_retries:
                break
                
            delay = min(base_delay * (2 ** (retries - 1)) + random.uniform(0, jitter), max_delay)
            time_left = global_timeout - (time.time() - start_time)
            if time_left <= 0:
                break
                
            # Asegurar que el delay no exceda el tiempo restante
            delay = min(delay, time_left * 0.7)
            
            logger.info(f"Reintento {retries}/{max_retries} tras error: {str(e)[:50]}. Esperando {delay:.2f}s")
            await asyncio.sleep(delay)
            continue
        
        # Si llegamos aquí, result es None pero no hubo excepción
        retries += 1
        if retries > max_retries:
            break
            
        delay = min(base_delay * (1.5 ** retries) + random.uniform(0, jitter), max_delay)
        time_left = global_timeout - (time.time() - start_time)
        if time_left <= delay:
            break
            
        await asyncio.sleep(delay)
    
    if last_exception:
        logger.error(f"Fallo final tras {retries} reintentos: {last_exception}")
        raise last_exception
    
    # Si llegamos aquí, todos los intentos devolvieron None sin excepción
    return None

# Compresión y diferencial para checkpoints
class CheckpointCompressor:
    """Utilidad para comprimir y descomprimir checkpoints."""
    
    @staticmethod
    def compress(data: Dict[str, Any]) -> str:
        """
        Comprimir datos de checkpoint.
        
        Args:
            data: Datos a comprimir
            
        Returns:
            Datos comprimidos en formato base64
        """
        json_str = json.dumps(data)
        compressed = zlib.compress(json_str.encode('utf-8'))
        return base64.b64encode(compressed).decode('utf-8')
    
    @staticmethod
    def decompress(compressed_data: str) -> Dict[str, Any]:
        """
        Descomprimir datos de checkpoint.
        
        Args:
            compressed_data: Datos comprimidos en formato base64
            
        Returns:
            Datos descomprimidos
        """
        decoded = base64.b64decode(compressed_data)
        decompressed = zlib.decompress(decoded)
        return json.loads(decompressed.decode('utf-8'))
    
    @staticmethod
    def create_differential(current: Dict[str, Any], previous: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crear checkpoint diferencial (solo cambios).
        
        Args:
            current: Estado actual
            previous: Estado anterior
            
        Returns:
            Checkpoint diferencial
        """
        diff = {"__is_differential": True, "__timestamp": time.time()}
        
        # Solo procesar datos de estado, mantener metadatos
        for key in ["state", "local_events", "external_events"]:
            if key in current:
                # Para eventos, solo guardar los nuevos
                if key in ["local_events", "external_events"]:
                    prev_events = previous.get(key, [])
                    curr_events = current.get(key, [])
                    if len(prev_events) < len(curr_events):
                        diff[key] = curr_events[len(prev_events):]
                    else:
                        diff[key] = curr_events[-3:] if curr_events else []
                # Para el estado, comparar y guardar solo las diferencias
                elif key == "state":
                    prev_state = previous.get(key, {})
                    curr_state = current.get(key, {})
                    changes = {}
                    
                    # Detectar cambios y nuevas claves
                    for k, v in curr_state.items():
                        if k not in prev_state or prev_state[k] != v:
                            changes[k] = v
                    
                    # Detectar claves eliminadas
                    for k in prev_state:
                        if k not in curr_state:
                            changes[k] = None  # Marcar como eliminado
                            
                    diff[key] = changes
        
        # Incluir una referencia al timestamp del checkpoint anterior
        if "created_at" in previous:
            diff["__previous_timestamp"] = previous["created_at"]
            
        # Incluir metadatos adicionales
        for key in current:
            if key not in ["state", "local_events", "external_events"] and key not in diff:
                diff[key] = current[key]
                
        return diff
    
    @staticmethod
    def apply_differential(base_checkpoint: Dict[str, Any], diff_checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplicar checkpoint diferencial.
        
        Args:
            base_checkpoint: Checkpoint base
            diff_checkpoint: Checkpoint diferencial
            
        Returns:
            Checkpoint completo actualizado
        """
        if not diff_checkpoint.get("__is_differential", False):
            return diff_checkpoint
            
        result = base_checkpoint.copy()
        
        # Actualizar estado
        if "state" in diff_checkpoint:
            state_changes = diff_checkpoint["state"]
            result_state = result.get("state", {}).copy()
            
            # Aplicar cambios
            for k, v in state_changes.items():
                if v is None:
                    result_state.pop(k, None)  # Eliminar clave
                else:
                    result_state[k] = v  # Actualizar o añadir
                    
            result["state"] = result_state
        
        # Actualizar eventos
        for key in ["local_events", "external_events"]:
            if key in diff_checkpoint:
                result[key] = result.get(key, []) + diff_checkpoint[key]
        
        # Actualizar timestamp
        if "__timestamp" in diff_checkpoint:
            result["created_at"] = diff_checkpoint["__timestamp"]
            
        # Incluir metadatos adicionales del diferencial
        for key in diff_checkpoint:
            if key not in ["__is_differential", "__timestamp", "__previous_timestamp",
                          "state", "local_events", "external_events"] and key not in result:
                result[key] = diff_checkpoint[key]
                
        return result

# Clase base de componente con mejoras extremas
class ComponentAPI:
    """
    Componente base con características de resiliencia extremas.
    
    Mejoras:
    - Checkpointing diferencial y comprimido
    - Procesamiento por lotes para alta carga
    - Mejor detección predictiva de fallos
    - Recuperación más rápida bajo estrés
    - Cola prioritaria con optimización de memoria
    """
    
    def __init__(self, id: str, essential: bool = False):
        """
        Inicializar componente.
        
        Args:
            id: Identificador único del componente
            essential: Si es un componente esencial
        """
        self.id = id
        self.essential = essential
        self.local_events: List[Tuple[str, Dict[str, Any], str]] = []
        self.external_events: List[Tuple[str, Dict[str, Any], str]] = []
        self.state: Dict[str, Any] = {}
        self.checkpoint: Dict[str, Any] = {}
        self.previous_checkpoint: Dict[str, Any] = {}  # Para diferenciales
        
        # Mejoras de resiliencia
        self.circuit_breaker = CircuitBreaker(f"cb_{id}", essential=essential)
        self.last_checkpoint_time = time.time()
        self.checkpoint_interval = 0.15  # 150ms entre checkpoints (normal)
        self.checkpoint_interval_stress = 0.1  # 100ms bajo estrés
        
        # Estado y métricas
        self.active = True
        self.last_active = time.time()
        self.task = None
        self.stats = {
            "processed_events": 0,
            "failed_events": 0,
            "checkpoint_count": 0,
            "restore_count": 0,
            "batch_count": 0,
            "batch_size_avg": 0,
            "processing_time_avg": 0
        }
        
        # Cola de eventos con prioridad mejorada
        self._event_queues: Dict[EventPriority, asyncio.Queue] = {
            EventPriority.CRITICAL: asyncio.Queue(maxsize=100),
            EventPriority.HIGH: asyncio.Queue(maxsize=200),
            EventPriority.NORMAL: asyncio.Queue(maxsize=300),
            EventPriority.LOW: asyncio.Queue(maxsize=200),
            EventPriority.BACKGROUND: asyncio.Queue(maxsize=100)
        }
        
        # Control del procesamiento por lotes
        self._batch_size = 5  # Procesar hasta 5 eventos a la vez
        self._processing_latencies = []
        
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """
        Procesar solicitud directa (API).
        
        Args:
            request_type: Tipo de solicitud
            data: Datos de la solicitud
            source: Origen de la solicitud
            
        Returns:
            Resultado de la solicitud
        """
        self.last_active = time.time()
        # Implementación específica en subclases
        raise NotImplementedError()
        
    async def on_local_event(self, event_type: str, data: Dict[str, Any], source: str, 
                           priority: EventPriority = EventPriority.NORMAL) -> None:
        """
        Manejar evento local.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Origen del evento
            priority: Prioridad del evento
        """
        # Si estamos bajo estrés severo y es un evento de baja prioridad, descartar
        if priority == EventPriority.BACKGROUND and self._is_under_severe_stress():
            return
            
        # Si la cola de prioridad correspondiente está llena, intentar con prioridad inferior
        try:
            await asyncio.wait_for(
                self._event_queues[priority].put((event_type, data, source)),
                timeout=0.1  # Timeout para no bloquear indefinidamente
            )
        except (asyncio.QueueFull, asyncio.TimeoutError):
            # Si la cola está llena o timeout, degradar prioridad
            if priority.value < len(EventPriority) - 1:
                next_priority = EventPriority(priority.value + 1)
                try:
                    await self._event_queues[next_priority].put((event_type, data, source))
                except (asyncio.QueueFull, asyncio.TimeoutError):
                    # Descartar silenciosamente si todas las colas están llenas
                    pass
    
    async def on_external_event(self, event_type: str, data: Dict[str, Any], source: str,
                              priority: EventPriority = EventPriority.NORMAL) -> None:
        """
        Manejar evento externo.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Origen del evento
            priority: Prioridad del evento
        """
        # Si estamos bajo estrés severo y es un evento de baja prioridad, descartar
        if priority == EventPriority.BACKGROUND and self._is_under_severe_stress():
            return
            
        # Mismo mecanismo que eventos locales pero con diferente procesamiento
        try:
            await asyncio.wait_for(
                self._event_queues[priority].put(("external", event_type, data, source)),
                timeout=0.1
            )
        except (asyncio.QueueFull, asyncio.TimeoutError):
            if priority.value < len(EventPriority) - 1:
                next_priority = EventPriority(priority.value + 1)
                try:
                    await self._event_queues[next_priority].put(("external", event_type, data, source))
                except (asyncio.QueueFull, asyncio.TimeoutError):
                    # Descartar silenciosamente si todas las colas están llenas
                    pass
            
    async def _process_events_loop(self):
        """Procesar eventos de las colas por prioridad, con soporte para procesamiento por lotes."""
        while self.active:
            try:
                # Verificar colas en orden de prioridad
                processed = False
                
                for priority in list(EventPriority):
                    queue = self._event_queues[priority]
                    
                    if not queue.empty():
                        try:
                            # Procesar por lotes para mayor eficiencia
                            batch_size = min(self._batch_size, queue.qsize())
                            if batch_size > 1:
                                # Extraer varios eventos como batch
                                batch = []
                                for _ in range(batch_size):
                                    try:
                                        event_data = await asyncio.wait_for(queue.get(), timeout=0.03)
                                        batch.append(event_data)
                                        queue.task_done()
                                    except asyncio.TimeoutError:
                                        break
                                
                                # Procesar batch
                                if batch:
                                    start_process = time.time()
                                    await self._process_event_batch(batch, priority)
                                    batch_time = time.time() - start_process
                                    
                                    # Actualizar estadísticas de batch
                                    self.stats["batch_count"] += 1
                                    self.stats["batch_size_avg"] = (
                                        (self.stats["batch_size_avg"] * (self.stats["batch_count"] - 1) + len(batch)) /
                                        self.stats["batch_count"]
                                    )
                                    self.stats["processing_time_avg"] = (
                                        (self.stats["processing_time_avg"] * (self.stats["batch_count"] - 1) + batch_time) /
                                        self.stats["batch_count"]
                                    )
                                    processed = True
                            else:
                                # Procesar evento único normalmente
                                event_data = await asyncio.wait_for(queue.get(), timeout=0.05)
                                await self._process_single_event(event_data)
                                queue.task_done()
                                processed = True
                                
                        except asyncio.TimeoutError:
                            # Timeout es esperado para poder revisar otras colas
                            continue
                        except Exception as e:
                            # Error procesando evento
                            logger.error(f"Error procesando eventos en {self.id}: {e}")
                            self.stats["failed_events"] += 1
                            queue.task_done()  # Asegurar que marcamos como completado
                
                # Si no procesamos nada, esperar un poco
                if not processed:
                    await asyncio.sleep(0.01)
                
                # Verificar si es momento de crear checkpoint
                now = time.time()
                interval = self.checkpoint_interval_stress if self._is_under_stress() else self.checkpoint_interval
                if now - self.last_checkpoint_time >= interval:
                    await self._create_checkpoint()
                    self.last_checkpoint_time = now
                    
            except asyncio.CancelledError:
                # Tarea cancelada, salir limpiamente
                break
            except Exception as e:
                # Error inesperado, registrar pero seguir ejecutando
                logger.error(f"Error en bucle de eventos de {self.id}: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_single_event(self, event_data):
        """Procesar un solo evento."""
        try:
            if len(event_data) == 3:  # Evento local
                event_type, data, source = event_data
                self.local_events.append((event_type, data, source))
                
                # Implementación específica del evento
                await self._handle_local_event(event_type, data, source)
            else:  # Evento externo (4 elementos)
                _, event_type, data, source = event_data
                self.external_events.append((event_type, data, source))
                
                # Implementación específica del evento
                await self._handle_external_event(event_type, data, source)
                
            # Actualizar estadísticas
            self.stats["processed_events"] += 1
            self.last_active = time.time()
        except Exception as e:
            logger.error(f"Error procesando evento en {self.id}: {e}")
            self.stats["failed_events"] += 1
    
    async def _process_event_batch(self, batch, priority):
        """Procesar un lote de eventos de la misma prioridad."""
        # Dividir el batch en locales y externos
        local_batch = []
        external_batch = []
        
        for event_data in batch:
            if len(event_data) == 3:  # Evento local
                self.local_events.append(event_data)
                local_batch.append(event_data)
            else:  # Evento externo (4 elementos)
                _, event_type, data, source = event_data
                self.external_events.append((event_type, data, source))
                external_batch.append((event_type, data, source))
        
        # Procesar por tipo
        if local_batch:
            try:
                await self._handle_local_batch(local_batch)
            except Exception as e:
                logger.error(f"Error procesando batch local en {self.id}: {e}")
                self.stats["failed_events"] += len(local_batch)
            else:
                self.stats["processed_events"] += len(local_batch)
                
        if external_batch:
            try:
                await self._handle_external_batch(external_batch)
            except Exception as e:
                logger.error(f"Error procesando batch externo en {self.id}: {e}")
                self.stats["failed_events"] += len(external_batch)
            else:
                self.stats["processed_events"] += len(external_batch)
                
        self.last_active = time.time()
    
    async def _handle_local_event(self, event_type: str, data: Dict[str, Any], source: str):
        """
        Implementación específica para manejar eventos locales.
        Sobrescribir en subclases para comportamiento personalizado.
        """
        pass
    
    async def _handle_external_event(self, event_type: str, data: Dict[str, Any], source: str):
        """
        Implementación específica para manejar eventos externos.
        Sobrescribir en subclases para comportamiento personalizado.
        """
        pass
    
    async def _handle_local_batch(self, batch: List[Tuple[str, Dict[str, Any], str]]):
        """
        Implementación específica para manejar lotes de eventos locales.
        Sobrescribir en subclases para comportamiento optimizado.
        Por defecto, procesa cada evento individualmente.
        """
        for event_type, data, source in batch:
            await self._handle_local_event(event_type, data, source)
    
    async def _handle_external_batch(self, batch: List[Tuple[str, Dict[str, Any], str]]):
        """
        Implementación específica para manejar lotes de eventos externos.
        Sobrescribir en subclases para comportamiento optimizado.
        Por defecto, procesa cada evento individualmente.
        """
        for event_type, data, source in batch:
            await self._handle_external_event(event_type, data, source)
    
    def _is_under_stress(self) -> bool:
        """Determinar si el componente está bajo estrés para ajustar comportamiento."""
        # Verificar si hay muchos eventos pendientes
        total_pending = sum(q.qsize() for q in self._event_queues.values())
        return total_pending > 100  # Más de 100 eventos pendientes = estrés
    
    def _is_under_severe_stress(self) -> bool:
        """Determinar si el componente está bajo estrés severo."""
        total_pending = sum(q.qsize() for q in self._event_queues.values())
        return total_pending > 500  # Más de 500 eventos pendientes = estrés severo
    
    async def _create_checkpoint(self):
        """Crear checkpoint optimizado del estado actual."""
        # Guardamos el checkpoint anterior para calcular diferencial
        if self.checkpoint:
            self.previous_checkpoint = self.checkpoint.copy()
        
        # Crear checkpoint completo
        current_checkpoint = {
            "state": self.state.copy() if self.state else {},
            "local_events": self.local_events[-3:] if self.local_events else [],
            "external_events": self.external_events[-3:] if self.external_events else [],
            "created_at": time.time()
        }
        
        # Si tenemos checkpoint anterior, crear diferencial
        if self.previous_checkpoint:
            diff_checkpoint = CheckpointCompressor.create_differential(
                current_checkpoint, self.previous_checkpoint
            )
            # Usar el diferencial si es más pequeño que el completo
            if len(str(diff_checkpoint)) < len(str(current_checkpoint)):
                self.checkpoint = diff_checkpoint
            else:
                self.checkpoint = current_checkpoint
        else:
            self.checkpoint = current_checkpoint
            
        # Estadísticas
        self.stats["checkpoint_count"] += 1
        
        logger.debug(f"Checkpoint creado para {self.id}")
    
    async def restore_from_checkpoint(self) -> bool:
        """
        Restaurar desde el último checkpoint.
        
        Returns:
            True si se restauró correctamente
        """
        if not self.checkpoint:
            logger.warning(f"No hay checkpoint disponible para {self.id}")
            return False
            
        try:
            # Restaurar estado desde checkpoint
            if self.checkpoint.get("__is_differential", False) and self.previous_checkpoint:
                # Si es diferencial, aplicarlo al checkpoint anterior
                full_checkpoint = CheckpointCompressor.apply_differential(
                    self.previous_checkpoint, self.checkpoint
                )
            else:
                full_checkpoint = self.checkpoint
                
            if "state" in full_checkpoint:
                self.state = full_checkpoint["state"].copy() if full_checkpoint["state"] else {}
                
            if "local_events" in full_checkpoint:
                self.local_events = list(full_checkpoint["local_events"])
                
            if "external_events" in full_checkpoint:
                self.external_events = list(full_checkpoint["external_events"])
            
            # Resetear estado activo
            self.active = True
            self.last_active = time.time()
            
            # Estadísticas
            self.stats["restore_count"] += 1
            
            logger.info(f"Componente {self.id} restaurado desde checkpoint")
            return True
        except Exception as e:
            logger.error(f"Error restaurando checkpoint para {self.id}: {e}")
            return False
        
    async def start(self):
        """Iniciar el componente."""
        if self.task is None or self.task.done():
            self.active = True
            self.task = asyncio.create_task(self._process_events_loop())
            logger.debug(f"Componente {self.id} iniciado")
    
    async def stop(self):
        """Detener el componente."""
        self.active = False
        if self.task and not self.task.done():
            try:
                self.task.cancel()
                await asyncio.shield(asyncio.wait_for(self.task, timeout=0.5))
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        logger.debug(f"Componente {self.id} detenido")
        
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del componente."""
        pending_events = {p.name: self._event_queues[p].qsize() for p in EventPriority}
        return {
            "id": self.id,
            "essential": self.essential,
            "active": self.active,
            "circuit_breaker": self.circuit_breaker.get_stats(),
            "events": {
                "local": len(self.local_events),
                "external": len(self.external_events),
                "pending": pending_events,
                "processed": self.stats["processed_events"],
                "failed": self.stats["failed_events"]
            },
            "batching": {
                "batch_count": self.stats["batch_count"],
                "avg_size": self.stats["batch_size_avg"],
                "avg_time": self.stats["processing_time_avg"]
            },
            "checkpointing": {
                "checkpoint_count": self.stats["checkpoint_count"],
                "restore_count": self.stats["restore_count"],
                "last_checkpoint": time.time() - self.last_checkpoint_time,
                "differential": self.checkpoint.get("__is_differential", False) if self.checkpoint else False
            }
        }

# Coordinador central mejorado con modo PRE-SAFE
class HybridCoordinator:
    """
    Coordinador central del sistema híbrido con optimizaciones extremas.
    
    Mejoras:
    - Modo PRE-SAFE para transiciones más suaves
    - Timeout global para todas las operaciones
    - Mejor gestión de componentes esenciales
    - Monitoreo predictivo avanzado
    - Priorización dinámica bajo estrés
    """
    
    def __init__(self):
        """Inicializar coordinador."""
        self.components: Dict[str, ComponentAPI] = {}
        self.mode = SystemMode.NORMAL
        self.essential_components: Set[str] = set()
        self.start_time = time.time()
        self.stats = {
            "api_calls": 0,
            "local_events": 0,
            "external_events": 0,
            "failures": 0,
            "recoveries": 0,
            "mode_transitions": {
                "to_normal": 0,
                "to_pre_safe": 0,
                "to_safe": 0,
                "to_emergency": 0
            },
            "timeouts": 0
        }
        self.monitor_task = None
        self.checkpoint_task = None
        self.system_health = 100  # 0-100, 100 = perfecta salud
        
        # Historial para mejor detección predictiva
        self._failure_history = []
        self._latency_history = []
        
    def register_component(self, component_id: str, component: ComponentAPI) -> None:
        """
        Registrar un componente en el sistema.
        
        Args:
            component_id: ID del componente
            component: Instancia del componente
        """
        if component_id in self.components:
            logger.warning(f"Componente {component_id} ya registrado, reemplazando")
        self.components[component_id] = component
        if component.essential:
            self.essential_components.add(component_id)
        logger.debug(f"Componente {component_id} registrado")
        
    async def request(self, target_id: str, request_type: str, data: Dict[str, Any], source: str, 
                    timeout: float = 0.8, priority: bool = False) -> Optional[Any]:
        """
        Realizar solicitud a un componente (API).
        
        Args:
            target_id: ID del componente destino
            request_type: Tipo de solicitud
            data: Datos de la solicitud
            source: Origen de la solicitud
            timeout: Timeout para la solicitud
            priority: Si es una solicitud prioritaria (ignora restricciones de modo)
            
        Returns:
            Resultado de la solicitud o None si falla
        """
        # Verificar si el componente existe
        if target_id not in self.components:
            logger.warning(f"Componente {target_id} no encontrado")
            return None
            
        component = self.components[target_id]
        
        # Verificar modo del sistema y si el componente es esencial
        if not priority:
            if self.mode == SystemMode.EMERGENCY and target_id not in self.essential_components:
                if request_type not in ["ping", "status", "health"]:
                    logger.warning(f"Solicitud {request_type} rechazada para {target_id} en modo EMERGENCY")
                    return None
            elif self.mode == SystemMode.SAFE and request_type not in ["ping", "status", "health", "critical"]:
                # En modo SAFE limitar operaciones en componentes no esenciales
                if target_id not in self.essential_components and random.random() < 0.3:
                    logger.info(f"Solicitud {request_type} diferida en {target_id} (modo SAFE)")
                    return None
        
        # Incrementar contador
        self.stats["api_calls"] += 1
        
        # Registrar tiempo de inicio para latencia
        start_time = time.time()
        
        # Función para ejecutar con timeout
        async def execute_request():
            return await asyncio.wait_for(
                component.process_request(request_type, data, source),
                timeout=timeout
            )
        
        # Ejecutar con Circuit Breaker y reintentos
        try:
            # Usar timeout más corto para componentes no esenciales en PRE-SAFE o SAFE
            actual_timeout = timeout
            if (self.mode in [SystemMode.PRE_SAFE, SystemMode.SAFE] and 
                target_id not in self.essential_components):
                actual_timeout = timeout * 0.8  # 20% menos tiempo
            
            # Circuit Breaker maneja fallos persistentes
            result = await component.circuit_breaker.execute(
                # Retry maneja fallos temporales
                lambda: with_retry(
                    execute_request,
                    max_retries=2,
                    base_delay=0.05,
                    max_delay=0.3,
                    global_timeout=actual_timeout,
                    essential=component.essential
                )
            )
            
            # Registrar latencia exitosa
            latency = time.time() - start_time
            self._latency_history.append((target_id, latency))
            if len(self._latency_history) > 100:
                self._latency_history.pop(0)
                
            return result
            
        except asyncio.TimeoutError:
            self.stats["timeouts"] += 1
            latency = time.time() - start_time
            self._latency_history.append((target_id, latency))
            logger.warning(f"Timeout en solicitud a {target_id}: {request_type}")
            return None
        except Exception as e:
            self.stats["failures"] += 1
            self._failure_history.append((target_id, str(type(e).__name__)))
            if len(self._failure_history) > 50:
                self._failure_history.pop(0)
                
            logger.error(f"Error en solicitud a {target_id}: {e}")
            component.active = False  # Marcar como inactivo para recuperación
            return None
            
    async def emit_local(self, event_type: str, data: Dict[str, Any], source: str, 
                        priority: EventPriority = EventPriority.NORMAL) -> None:
        """
        Emitir evento local a todos los componentes.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Origen del evento
            priority: Prioridad del evento
        """
        # En modo emergencia solo procesar eventos críticos
        if self.mode == SystemMode.EMERGENCY and priority != EventPriority.CRITICAL:
            return
            
        # En modos degradados, ajustar prioridades
        if self.mode == SystemMode.PRE_SAFE:
            # Degradar levemente eventos no críticos
            if priority == EventPriority.NORMAL:
                if random.random() < 0.3:  # 30% de probabilidad de degradar
                    priority = EventPriority.LOW
        elif self.mode == SystemMode.SAFE:
            # Degradar significativamente eventos no críticos
            if priority == EventPriority.NORMAL:
                priority = EventPriority.LOW
            elif priority == EventPriority.LOW:
                priority = EventPriority.BACKGROUND
            
        # Incrementar contador
        self.stats["local_events"] += 1
        
        # Filtrar componentes según modo
        filtered_components = {}
        if self.mode == SystemMode.EMERGENCY:
            # Solo componentes esenciales
            filtered_components = {cid: comp for cid, comp in self.components.items() 
                                if comp.essential or cid in self.essential_components}
        else:
            filtered_components = self.components
            
        # Crear tareas para enviar eventos
        tasks = []
        for cid, component in filtered_components.items():
            if cid != source and component.active:
                tasks.append(component.on_local_event(event_type, data, source, priority))
                
        # Ejecutar tareas sin esperar
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def emit_external(self, event_type: str, data: Dict[str, Any], source: str,
                          priority: EventPriority = EventPriority.NORMAL) -> None:
        """
        Emitir evento externo a todos los componentes.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Origen del evento
            priority: Prioridad del evento
        """
        # En modo emergencia no procesar eventos externos excepto críticos
        if self.mode == SystemMode.EMERGENCY and priority != EventPriority.CRITICAL:
            return
            
        # En modos degradados, ajustar prioridades
        if self.mode in [SystemMode.PRE_SAFE, SystemMode.SAFE]:
            # Degradar eventos externos en modos degradados
            if priority.value < EventPriority.LOW.value:
                priority = EventPriority(priority.value + 1)
            
        # Incrementar contador
        self.stats["external_events"] += 1
        
        # Crear tareas para enviar eventos
        tasks = []
        for cid, component in self.components.items():
            if cid != source and component.active:
                tasks.append(component.on_external_event(event_type, data, source, priority))
                
        # Ejecutar tareas sin esperar
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
    async def _monitor_system(self):
        """Monitorear el estado del sistema y ajustar modo."""
        while True:
            try:
                # Contar componentes fallidos
                failed_components = [cid for cid, comp in self.components.items() 
                                    if not comp.active or comp.circuit_breaker.state == CircuitState.OPEN]
                failed_count = len(failed_components)
                
                # Contar componentes parcialmente fallidos
                partial_components = [cid for cid, comp in self.components.items() 
                                     if comp.active and comp.circuit_breaker.state == CircuitState.PARTIAL]
                partial_count = len(partial_components)
                
                # Contar componentes esenciales fallidos
                essential_failed = [cid for cid in failed_components if cid in self.essential_components]
                essential_partial = [cid for cid in partial_components if cid in self.essential_components]
                
                # Actualizar modo del sistema
                total_components = len(self.components) or 1  # Evitar división por cero
                
                # Calcular puntuación de salud del sistema (0-100)
                failure_weight = 100 / total_components
                partial_weight = 50 / total_components
                
                new_health = 100
                new_health -= len(failed_components) * failure_weight
                new_health -= len(partial_components) * partial_weight
                # Más peso a componentes esenciales
                new_health -= len(essential_failed) * failure_weight * 2
                new_health -= len(essential_partial) * partial_weight * 2
                
                # Limitar el valor entre 0 y 100
                self.system_health = max(0, min(100, new_health))
                
                # Determinar nuevo modo
                new_mode = SystemMode.NORMAL
                
                if len(essential_failed) > 0 or self.system_health < 40:
                    new_mode = SystemMode.EMERGENCY
                elif self.system_health < 60 or len(essential_partial) > 0:
                    new_mode = SystemMode.SAFE
                elif self.system_health < 80 or partial_count > total_components * 0.2:
                    new_mode = SystemMode.PRE_SAFE
                    
                # Registrar cambio de modo
                if new_mode != self.mode:
                    prev_mode = self.mode
                    self.mode = new_mode
                    self.stats["mode_transitions"][f"to_{new_mode.value}"] += 1
                    logger.warning(f"Cambiando modo del sistema: {prev_mode.value} -> {new_mode.value}")
                    logger.warning(f"Componentes fallidos: {failed_count}/{total_components}, Parciales: {partial_count}/{total_components}")
                    logger.warning(f"Salud del sistema: {self.system_health:.2f}%")
                
                # Verificar componentes para recuperación proactiva
                for cid, component in self.components.items():
                    if not component.active:
                        # Intentar restaurar desde checkpoint
                        if await component.restore_from_checkpoint():
                            component.active = True
                            # Reiniciar task si es necesario
                            if component.task is None or component.task.done():
                                component.task = asyncio.create_task(component._process_events_loop())
                            self.stats["recoveries"] += 1
                            logger.info(f"Componente {cid} recuperado")
                    elif component.circuit_breaker.state == CircuitState.OPEN and cid in self.essential_components:
                        # Intentar reset proactivo para componentes esenciales
                        component.circuit_breaker.state = CircuitState.HALF_OPEN
                        logger.info(f"Reset proactivo de Circuit Breaker para componente esencial {cid}")
                    elif not component.task or component.task.done():
                        # Tarea terminó inesperadamente, reiniciar
                        component.task = asyncio.create_task(component._process_events_loop())
                        logger.info(f"Tarea reiniciada para componente {cid}")
                            
                # Dormir hasta próxima comprobación
                sleep_time = 0.1 if self.mode != SystemMode.NORMAL else 0.15
                await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error en monitor del sistema: {e}")
                await asyncio.sleep(0.5)  # Esperar más en caso de error
                
    async def _checkpoint_system(self):
        """Crear checkpoints periódicos del estado del sistema."""
        while True:
            try:
                # Crear checkpoint del estado del sistema
                system_state = {
                    "mode": self.mode.value,
                    "health": self.system_health,
                    "stats": self.stats.copy(),
                    "components": {cid: comp.get_stats() for cid, comp in self.components.items()},
                    "timestamp": time.time()
                }
                
                # Aquí se podría persistir el estado en disco o base de datos
                
                # Esperar hasta próximo checkpoint
                sleep_time = 1.0  # 1 segundo entre checkpoints del sistema
                await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error en checkpoint del sistema: {e}")
                await asyncio.sleep(0.5)
                
    async def start(self):
        """Iniciar todos los componentes y el sistema."""
        # Iniciar componentes
        start_tasks = [comp.start() for comp in self.components.values()]
        await asyncio.gather(*start_tasks)
        
        # Iniciar monitor y checkpoint
        self.monitor_task = asyncio.create_task(self._monitor_system())
        self.checkpoint_task = asyncio.create_task(self._checkpoint_system())
        
        logger.info(f"Sistema iniciado con {len(self.components)} componentes")
        
    async def stop(self):
        """Detener todos los componentes y el sistema."""
        # Cancelar tareas de monitoreo
        if self.monitor_task:
            self.monitor_task.cancel()
        if self.checkpoint_task:
            self.checkpoint_task.cancel()
            
        # Detener componentes
        stop_tasks = [comp.stop() for comp in self.components.values()]
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        logger.info(f"Sistema detenido. Estadísticas finales: {self.stats}")
        
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas completas del sistema."""
        runtime = time.time() - self.start_time
        stats = self.stats.copy()
        stats["uptime"] = runtime
        stats["components"] = len(self.components)
        stats["mode"] = self.mode.value
        stats["health"] = self.system_health
        
        # Estadísticas agregadas
        total_local_events = sum(len(comp.local_events) for comp in self.components.values())
        total_external_events = sum(len(comp.external_events) for comp in self.components.values())
        
        stats["events"] = {
            "total_local": total_local_events,
            "total_external": total_external_events
        }
        
        # Agregar estadísticas de latencia
        if self._latency_history:
            avg_latency = sum(l[1] for l in self._latency_history) / len(self._latency_history)
            stats["avg_latency"] = avg_latency
        
        # Agregar estadísticas de componentes
        component_stats = {}
        for cid, comp in self.components.items():
            component_stats[cid] = comp.get_stats()
            
        stats["components_detail"] = component_stats
        return stats

# Componente de prueba para el test con soporte de batch
class TestComponent(ComponentAPI):
    """Componente de prueba con soporte para procesamiento por lotes."""
    
    def __init__(self, id: str, essential: bool = False, fail_rate: float = 0.0):
        """
        Inicializar componente de prueba.
        
        Args:
            id: Identificador del componente
            essential: Si es un componente esencial
            fail_rate: Tasa de fallos aleatorios (0.0-1.0)
        """
        super().__init__(id, essential)
        self.fail_rate = fail_rate
        
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """
        Procesar solicitud con fallos simulados.
        
        Args:
            request_type: Tipo de solicitud
            data: Datos de la solicitud
            source: Origen de la solicitud
            
        Returns:
            Resultado de la solicitud
        """
        # Simular fallo aleatorio
        if random.random() < self.fail_rate or "fail" in data:
            await asyncio.sleep(random.uniform(0.05, 0.2))  # Simular latencia antes de fallar
            raise Exception(f"Fallo simulado en {self.id}.{request_type}")
            
        # Operaciones específicas
        if request_type == "ping":
            await asyncio.sleep(0.01)  # Simular procesamiento
            return f"pong from {self.id}"
        elif request_type == "get_state":
            key = data.get("key")
            return self.state.get(key, {"status": "not_found", "key": key})
        elif request_type == "set_state":
            key = data.get("key")
            value = data.get("value")
            self.state[key] = value
            return {"status": "stored", "key": key}
        elif request_type == "test_latency":
            # Simular operación con latencia
            delay = data.get("delay", 0.1)
            await asyncio.sleep(delay)
            return {"latency": delay, "status": "success"}
        elif request_type == "fail":
            # Solicitud para fallar a propósito
            self.active = False
            raise Exception(f"Fallo forzado en {self.id}")
        
        return None
        
    async def _handle_local_event(self, event_type: str, data: Dict[str, Any], source: str):
        """Implementación para manejar eventos locales."""
        # Simular fallo aleatorio
        if random.random() < self.fail_rate:
            raise Exception(f"Fallo simulado procesando evento local en {self.id}")
            
        # Procesar evento específico
        if event_type == "update_state":
            key = data.get("key")
            value = data.get("value")
            if key and value is not None:
                self.state[key] = value
    
    async def _handle_external_event(self, event_type: str, data: Dict[str, Any], source: str):
        """Implementación para manejar eventos externos."""
        # Simular fallo aleatorio
        if random.random() < self.fail_rate:
            raise Exception(f"Fallo simulado procesando evento externo en {self.id}")
            
        # Procesar evento específico
        if event_type == "notification":
            # Simplemente registrarlo para la prueba
            pass
            
    async def _handle_local_batch(self, batch: List[Tuple[str, Dict[str, Any], str]]):
        """Implementación optimizada para manejar lotes de eventos locales."""
        # Simular fallo aleatorio
        if random.random() < self.fail_rate * 0.8:  # Reducir probabilidad para batch
            raise Exception(f"Fallo simulado procesando batch local en {self.id}")
        
        # Agrupar operaciones update_state por clave
        updates = {}
        for event_type, data, source in batch:
            if event_type == "update_state":
                key = data.get("key")
                value = data.get("value")
                if key and value is not None:
                    updates[key] = value
        
        # Aplicar actualizaciones en una sola operación
        for key, value in updates.items():
            self.state[key] = value

# Prueba extrema optimizada para verificar la resiliencia
async def test_resiliencia_extrema():
    """
    Ejecutar prueba extrema optimizada para verificar todas las características
    de resiliencia bajo condiciones extremas.
    
    Esta versión añade:
    - Mayor carga (3000 eventos locales + 200 externos)
    - Más componentes (30 en total)
    - Mayor tasa de fallos (hasta 30%)
    - Latencias extremas (hasta 2s)
    - Fallos en cascada simulados
    """
    logger.info("=== INICIANDO PRUEBA EXTREMA DE RESILIENCIA OPTIMIZADA V2 ===")
    start_time = time.time()
    
    # Crear coordinador
    coordinator = HybridCoordinator()
    
    # Registrar componentes para la prueba (usar más componentes para estresar)
    for i in range(30):  # 30 componentes en total
        # Diferentes tasas de fallo para simular componentes poco confiables
        fail_rate = random.uniform(0.0, 0.3)  # Entre 0% y 30% de fallo
        # Marcar algunos como esenciales
        essential = i < 6  # Los primeros 6 son esenciales
        component = TestComponent(f"component_{i}", essential=essential, fail_rate=fail_rate)
        coordinator.register_component(f"component_{i}", component)
    
    # Iniciar sistema
    await coordinator.start()
    
    try:
        # 1. PRUEBA DE ALTA CARGA
        logger.info("=== Prueba de Alta Carga ===")
        start_test = time.time()
        
        # Generar 3000 eventos locales concurrentes
        local_tasks = [
            coordinator.emit_local(
                f"event_{i}", 
                {"id": i, "timestamp": time.time()}, 
                "test_system",
                # Asignar prioridades variadas
                priority=EventPriority(random.randint(0, 3))
            )
            for i in range(3000)
        ]
        
        # Generar 200 eventos externos concurrentes
        external_tasks = [
            coordinator.emit_external(
                f"ext_event_{i}", 
                {"id": i, "timestamp": time.time()}, 
                "test_system",
                # Asignar prioridades variadas
                priority=EventPriority(random.randint(0, 3))
            )
            for i in range(200)
        ]
        
        # Ejecutar en paralelo
        await asyncio.gather(*(local_tasks + external_tasks))
        
        # Dar tiempo mínimo para procesar (mucho menos que antes)
        await asyncio.sleep(0.3)
        
        # Calcular resultados de alta carga
        high_load_duration = time.time() - start_test
        
        # Resultados procesados (cálculo rápido)
        total_processed = sum(comp.stats["processed_events"] 
                             for comp in coordinator.components.values())
        
        logger.info(f"Prueba de alta carga completada en {high_load_duration:.2f}s")
        logger.info(f"Eventos emitidos: 3200, Procesados: {total_processed}")
        
        # 2. PRUEBA DE FALLOS MASIVOS
        logger.info("=== Prueba de Fallos Masivos ===")
        start_test = time.time()
        
        # Forzar fallos en 12 componentes no esenciales (40%)
        components_to_fail = [f"component_{i}" for i in range(6, 18)]
        
        logger.info(f"Forzando fallo en {len(components_to_fail)} componentes")
        
        # Forzar fallos
        fail_tasks = [
            coordinator.request(cid, "fail", {}, "test_system")
            for cid in components_to_fail
        ]
        
        await asyncio.gather(*fail_tasks, return_exceptions=True)
        
        # Esperar a que el sistema detecte fallos y active recuperación
        await asyncio.sleep(0.3)
        
        # 3. PRUEBA DE LATENCIAS EXTREMAS
        logger.info("=== Prueba de Latencias Extremas ===")
        
        # Realizar solicitudes con diferentes latencias
        latency_results = []
        for latency in [0.05, 0.2, 0.5, 1.0, 1.5, 2.0]:
            component_id = f"component_{random.randint(18, 29)}"  # Usar componentes sanos
            start_op = time.time()
            operation_result = await coordinator.request(
                component_id, 
                "test_latency", 
                {"delay": latency},
                "test_system"
            )
            end_op = time.time()
            
            latency_results.append({
                "target": component_id,
                "requested_latency": latency,
                "actual_latency": end_op - start_op,
                "success": operation_result is not None
            })
        
        # Contar éxitos en prueba de latencia
        latency_success = sum(1 for r in latency_results if r["success"])
        latency_total = len(latency_results)
        
        logger.info(f"Prueba de latencias completada: {latency_success}/{latency_total} exitosas")
        
        # 4. VERIFICACIÓN DE RECUPERACIÓN TRAS FALLOS
        logger.info("=== Verificación de Recuperación ===")
        
        # Esperar a que el sistema intente recuperar componentes
        await asyncio.sleep(0.5)
        
        # Contar componentes recuperados
        recovered_count = coordinator.stats["recoveries"]
        
        # Verificar estado final
        active_components = sum(1 for comp in coordinator.components.values() if comp.active)
        final_mode = coordinator.mode
        
        # Verificar métricas
        logger.info(f"Componentes activos después de prueba: {active_components}/30")
        logger.info(f"Componentes recuperados: {recovered_count}")
        logger.info(f"Modo final del sistema: {final_mode.value}")
        logger.info(f"Salud del sistema: {coordinator.system_health:.2f}%")
        
        # 5. CÁLCULO DE TASA DE ÉXITO GLOBAL
        total_duration = time.time() - start_time
        system_stats = coordinator.get_stats()
        
        # Tasa de procesamiento de eventos
        total_events_sent = 3200  # 3000 locales + 200 externos
        events_processed = total_processed
        event_process_rate = (events_processed / (total_events_sent * len(coordinator.components))) * 100
        
        # Tasa de recuperación
        recovery_rate = (recovered_count / len(components_to_fail)) * 100 if components_to_fail else 100
        
        # Tasa de éxito de latencia
        latency_success_rate = (latency_success / latency_total) * 100 if latency_total else 100
        
        # Tasa de salud del sistema
        health_rate = coordinator.system_health
        
        # Tasa global (promedio ponderado)
        global_success_rate = (
            0.4 * event_process_rate +   # 40% peso a procesamiento
            0.3 * recovery_rate +        # 30% peso a recuperación
            0.2 * latency_success_rate + # 20% peso a latencia
            0.1 * health_rate            # 10% peso a salud general
        )
        
        # Resultados finales
        logger.info("\n=== RESUMEN DE PRUEBA EXTREMA OPTIMIZADA V2 ===")
        logger.info(f"Duración total: {total_duration:.2f}s")
        logger.info(f"Tasa de procesamiento de eventos: {event_process_rate:.2f}%")
        logger.info(f"Tasa de recuperación: {recovery_rate:.2f}%")
        logger.info(f"Tasa de éxito con latencia: {latency_success_rate:.2f}%")
        logger.info(f"Salud del sistema: {health_rate:.2f}%")
        logger.info(f"Tasa de éxito global: {global_success_rate:.2f}%")
        logger.info(f"API calls: {system_stats['api_calls']}, "
                   f"Local events: {system_stats['local_events']}, "
                   f"External events: {system_stats['external_events']}")
        logger.info(f"Fallos: {system_stats['failures']}, "
                   f"Recuperaciones: {system_stats['recoveries']}, "
                   f"Timeouts: {system_stats['timeouts']}")
        logger.info(f"Modo final del sistema: {system_stats['mode']}")
        
        return {
            "duration": total_duration,
            "event_process_rate": event_process_rate,
            "recovery_rate": recovery_rate,
            "latency_success_rate": latency_success_rate,
            "health_rate": health_rate,
            "global_success_rate": global_success_rate,
            "stats": system_stats
        }
    
    finally:
        # Detener sistema
        await coordinator.stop()
        logger.info("Sistema detenido")

# Código para ejecutar la prueba
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ejecutar prueba
    asyncio.run(test_resiliencia_extrema())