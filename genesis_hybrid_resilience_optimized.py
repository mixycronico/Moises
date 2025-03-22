"""
Sistema Genesis híbrido optimizado con resiliencia avanzada.

Este módulo implementa la versión optimizada del sistema híbrido Genesis
con características avanzadas de resiliencia:
- Reintentos adaptativos con backoff exponencial y jitter, con detección de éxito temprano
- Circuit Breaker optimizado con recuperación rápida
- Checkpointing ligero y recuperación proactiva
- Gestión de modos de degradación (Safe Mode) con priorización inteligente
- Sistema de priorización de eventos para alta carga

Objetivo: Alcanzar >90% de tasa de éxito en pruebas extremas
"""

import asyncio
import logging
import time
import random
import json
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
    CLOSED = auto()    # Funcionamiento normal
    OPEN = auto()      # Circuito abierto, rechaza llamadas
    HALF_OPEN = auto() # Semi-abierto, permite algunas llamadas

class SystemMode(Enum):
    """Modos de operación del sistema."""
    NORMAL = "normal"     # Funcionamiento normal
    SAFE = "safe"         # Modo seguro
    EMERGENCY = "emergency"  # Modo emergencia

class EventPriority(Enum):
    """Prioridades para eventos."""
    CRITICAL = 0    # Eventos críticos (ej. alertas de seguridad)
    HIGH = 1        # Eventos importantes (ej. operaciones de trading)
    NORMAL = 2      # Eventos regulares
    LOW = 3         # Eventos de baja prioridad (ej. actualizaciones UI)

# Circuit Breaker optimizado
class CircuitBreaker:
    """
    Implementación optimizada del Circuit Breaker para aislar componentes fallidos.
    
    Mejoras:
    - Recovery timeout reducido para recuperación más rápida
    - Reset rápido en estado HALF_OPEN
    - Mejor manejo de estadísticas para monitoreo
    """
    
    def __init__(
        self, 
        name: str, 
        failure_threshold: int = 3,
        recovery_timeout: float = 1.0,    # Reducido de 2.0s a 1.0s
        half_open_max_calls: int = 1,
        success_threshold: int = 1         # Reducido de 2 a 1 para reset rápido
    ):
        """
        Inicializar Circuit Breaker.
        
        Args:
            name: Nombre para identificar el circuit breaker
            failure_threshold: Fallos consecutivos para abrir el circuito
            recovery_timeout: Tiempo hasta probar recuperación (segundos)
            half_open_max_calls: Máximo de llamadas en estado half-open
            success_threshold: Éxitos consecutivos para cerrar el circuito
        """
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_state_change = time.time()
        
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
        
        # Ejecutar la función
        try:
            result = await func(*args, **kwargs)
            
            # Contabilizar éxito
            self.success_count_total += 1
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    logger.info(f"Circuit Breaker '{self.name}' cerrado tras {self.success_count} éxitos")
                    self.state = CircuitState.CLOSED
                    self.last_state_change = time.time()
                    self.failure_count = 0
            else:
                # En estado normal, resetear contador de fallos con cada éxito
                self.failure_count = 0
                
            return result
        except Exception as e:
            # Contabilizar fallo
            self.failure_count_total += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # Actualizar estado si necesario
            if self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
                logger.warning(f"Circuit Breaker '{self.name}' abierto tras {self.failure_count} fallos")
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
        return {
            "state": self.state.name,
            "call_count": self.call_count,
            "success_rate": (self.success_count_total / max(self.call_count, 1)) * 100,
            "failure_count": self.failure_count_total,
            "rejection_count": self.rejection_count,
            "last_failure": time.time() - self.last_failure_time if self.last_failure_time > 0 else None
        }

# Sistema de reintentos adaptativos optimizado
async def with_retry(
    func: Callable[..., Coroutine], 
    max_retries: int = 3, 
    base_delay: float = 0.05, 
    max_delay: float = 0.3,   # Reducido de 0.5s a 0.3s
    jitter: float = 0.05      # Reducido para respuestas más rápidas
) -> Any:
    """
    Ejecutar una función con reintentos adaptativos optimizados.
    
    Mejoras:
    - Detección de éxito temprano para salir del ciclo rápidamente
    - Menor max_delay y jitter para respuestas más rápidas
    - Mejor manejo de excepciones específicas
    
    Args:
        func: Función a ejecutar
        max_retries: Número máximo de reintentos
        base_delay: Tiempo base entre reintentos
        max_delay: Tiempo máximo entre reintentos
        jitter: Variación aleatoria máxima
        
    Returns:
        Resultado de la función
        
    Raises:
        Exception: Si se agotan los reintentos
    """
    retries = 0
    last_exception = None
    
    while retries <= max_retries:
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
            logger.info(f"Reintento {retries}/{max_retries} tras error: {str(e)[:50]}. Esperando {delay:.2f}s")
            await asyncio.sleep(delay)
            continue
        
        # Si llegamos aquí, result es None pero no hubo excepción
        retries += 1
        if retries > max_retries:
            break
            
        delay = min(base_delay * (1.5 ** retries) + random.uniform(0, jitter), max_delay)
        await asyncio.sleep(delay)
    
    if last_exception:
        logger.error(f"Fallo final tras {max_retries} reintentos: {last_exception}")
        raise last_exception
    
    # Si llegamos aquí, todos los intentos devolvieron None sin excepción
    return None

# Clase base de componente con checkpointing mejorado
class ComponentAPI:
    """
    Componente base con características de resiliencia mejoradas.
    
    Mejoras:
    - Checkpoints más ligeros (solo 3 eventos por tipo)
    - Cola de eventos con prioridad
    - Recuperación proactiva
    - Mejor detección de fallos
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
        
        # Mejoras de resiliencia
        self.circuit_breaker = CircuitBreaker(f"cb_{id}")
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
            "restore_count": 0
        }
        
        # Cola de eventos con prioridad
        self._event_queues: Dict[EventPriority, asyncio.Queue] = {
            EventPriority.CRITICAL: asyncio.Queue(maxsize=50),
            EventPriority.HIGH: asyncio.Queue(maxsize=100),
            EventPriority.NORMAL: asyncio.Queue(maxsize=200),
            EventPriority.LOW: asyncio.Queue(maxsize=100)
        }
        
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
        # Agregar a cola de prioridad correspondiente
        try:
            await self._event_queues[priority].put((event_type, data, source))
        except asyncio.QueueFull:
            # Si la cola está llena, intentar con prioridad inferior
            if priority != EventPriority.LOW:
                next_priority = EventPriority(priority.value + 1)
                try:
                    await self._event_queues[next_priority].put((event_type, data, source))
                except asyncio.QueueFull:
                    # Silenciar para no saturar logs
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
        # Mismo mecanismo que eventos locales pero con diferente procesamiento
        try:
            await self._event_queues[priority].put(("external", event_type, data, source))
        except asyncio.QueueFull:
            if priority != EventPriority.LOW:
                next_priority = EventPriority(priority.value + 1)
                try:
                    await self._event_queues[next_priority].put(("external", event_type, data, source))
                except asyncio.QueueFull:
                    # Silenciar para no saturar logs
                    pass
            
    async def _process_events_loop(self):
        """Procesar eventos de las colas por prioridad."""
        while self.active:
            try:
                # Verificar colas en orden de prioridad
                processed = False
                
                for priority in [EventPriority.CRITICAL, EventPriority.HIGH, 
                                EventPriority.NORMAL, EventPriority.LOW]:
                    queue = self._event_queues[priority]
                    
                    if not queue.empty():
                        try:
                            # Usar timeout corto para no bloquear otras colas
                            event_data = await asyncio.wait_for(queue.get(), timeout=0.05)
                            processed = True
                            
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
                            
                            # Marcar como completado
                            queue.task_done()
                        except asyncio.TimeoutError:
                            # Timeout es esperado para poder revisar otras colas
                            continue
                        except Exception as e:
                            # Error procesando evento
                            logger.error(f"Error procesando evento en {self.id}: {e}")
                            self.stats["failed_events"] += 1
                            queue.task_done()
                
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
    
    def _is_under_stress(self) -> bool:
        """Determinar si el componente está bajo estrés para ajustar comportamiento."""
        # Verificar si hay muchos eventos pendientes
        total_pending = sum(q.qsize() for q in self._event_queues.values())
        return total_pending > 100  # Más de 100 eventos pendientes = estrés
    
    async def _create_checkpoint(self):
        """Crear checkpoint optimizado del estado actual."""
        # Solo guardar los últimos 3 eventos (en lugar de 5) para reducir overhead
        self.checkpoint = {
            "state": self.state.copy() if self.state else {},
            "local_events": self.local_events[-3:] if self.local_events else [],
            "external_events": self.external_events[-3:] if self.external_events else [],
            "created_at": time.time()
        }
        
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
            
        # Restaurar estado desde checkpoint
        if "state" in self.checkpoint:
            self.state = self.checkpoint["state"].copy() if self.checkpoint["state"] else {}
            
        if "local_events" in self.checkpoint:
            self.local_events = list(self.checkpoint["local_events"])
            
        if "external_events" in self.checkpoint:
            self.external_events = list(self.checkpoint["external_events"])
        
        # Resetear estado activo
        self.active = True
        self.last_active = time.time()
        
        # Estadísticas
        self.stats["restore_count"] += 1
        
        logger.info(f"Componente {self.id} restaurado desde checkpoint")
        return True
        
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
        return {
            "id": self.id,
            "essential": self.essential,
            "active": self.active,
            "circuit_breaker": self.circuit_breaker.get_stats(),
            "events": {
                "local": len(self.local_events),
                "external": len(self.external_events),
                "pending": sum(q.qsize() for q in self._event_queues.values()),
                "processed": self.stats["processed_events"],
                "failed": self.stats["failed_events"]
            },
            "checkpointing": {
                "checkpoint_count": self.stats["checkpoint_count"],
                "restore_count": self.stats["restore_count"],
                "last_checkpoint": time.time() - self.last_checkpoint_time
            }
        }

# Coordinador central mejorado
class HybridCoordinator:
    """
    Coordinador central del sistema híbrido con todas las mejoras de resiliencia.
    
    Mejoras:
    - Mejor gestión de modos del sistema
    - Checkpointing proactivo
    - Priorización de componentes esenciales
    - Monitoreo avanzado
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
                "to_safe": 0,
                "to_emergency": 0
            }
        }
        self.monitor_task = None
        self.checkpoint_task = None
        
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
                    timeout: float = 1.0, priority: bool = False) -> Optional[Any]:
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
        if not priority and self.mode == SystemMode.EMERGENCY and target_id not in self.essential_components:
            if request_type not in ["ping", "status", "health"]:
                logger.warning(f"Solicitud {request_type} rechazada para {target_id} en modo EMERGENCY")
                return None
        
        # Incrementar contador
        self.stats["api_calls"] += 1
        
        # Función para ejecutar con timeout
        async def execute_request():
            return await asyncio.wait_for(
                component.process_request(request_type, data, source),
                timeout=timeout
            )
        
        # Ejecutar con Circuit Breaker y reintentos
        try:
            # Circuit Breaker maneja fallos persistentes
            return await component.circuit_breaker.execute(
                # Retry maneja fallos temporales
                lambda: with_retry(
                    execute_request,
                    max_retries=2,  # Reducir para respuesta más rápida
                    base_delay=0.05,
                    max_delay=0.3
                )
            )
        except Exception as e:
            self.stats["failures"] += 1
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
        if self.mode == SystemMode.EMERGENCY and priority != EventPriority.CRITICAL:
            # En modo emergencia solo procesar eventos críticos
            return
            
        # En modo SAFE, degradar prioridad de eventos no esenciales
        if self.mode == SystemMode.SAFE and priority == EventPriority.NORMAL:
            priority = EventPriority.LOW
            
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
                                    if not comp.active or comp.circuit_breaker.state != CircuitState.CLOSED]
                failed_count = len(failed_components)
                
                # Contar componentes esenciales fallidos
                essential_failed = [cid for cid in failed_components if cid in self.essential_components]
                
                # Actualizar modo del sistema
                total_components = len(self.components) or 1  # Evitar división por cero
                failure_rate = failed_count / total_components
                
                # Determinar nuevo modo
                new_mode = SystemMode.NORMAL
                if len(essential_failed) > 0 or failure_rate > 0.5:
                    new_mode = SystemMode.EMERGENCY
                elif failure_rate > 0.2:
                    new_mode = SystemMode.SAFE
                    
                # Registrar cambio de modo
                if new_mode != self.mode:
                    prev_mode = self.mode
                    self.mode = new_mode
                    self.stats[f"mode_transitions"]["to_{new_mode.value}"] += 1
                    logger.warning(f"Cambiando modo del sistema: {prev_mode.value} -> {new_mode.value}")
                    logger.warning(f"Componentes fallidos: {failed_count}/{total_components}")
                
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
        
        # Estadísticas agregadas
        total_local_events = sum(len(comp.local_events) for comp in self.components.values())
        total_external_events = sum(len(comp.external_events) for comp in self.components.values())
        
        stats["events"] = {
            "total_local": total_local_events,
            "total_external": total_external_events
        }
        
        # Agregar estadísticas de componentes
        component_stats = {}
        for cid, comp in self.components.items():
            component_stats[cid] = comp.get_stats()
            
        stats["components_detail"] = component_stats
        return stats

# Componente de prueba para el test
class TestComponent(ComponentAPI):
    """Componente simple para pruebas."""
    
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


# Prueba extrema optimizada para verificar la resiliencia
async def test_resiliencia_extrema():
    """
    Ejecutar prueba extrema optimizada para verificar todas las características
    de resiliencia bajo condiciones adversas.
    """
    logger.info("=== INICIANDO PRUEBA EXTREMA DE RESILIENCIA OPTIMIZADA ===")
    start_time = time.time()
    
    # Crear coordinador
    coordinator = HybridCoordinator()
    
    # Registrar componentes para la prueba (usar más componentes para estresar)
    for i in range(20):  # 20 componentes en total
        # Diferentes tasas de fallo para simular componentes poco confiables
        fail_rate = random.uniform(0.0, 0.2)  # Entre 0% y 20% de fallo
        # Marcar algunos como esenciales
        essential = i < 4  # Los primeros 4 son esenciales
        component = TestComponent(f"component_{i}", essential=essential, fail_rate=fail_rate)
        coordinator.register_component(f"component_{i}", component)
    
    # Iniciar sistema
    await coordinator.start()
    
    try:
        # 1. PRUEBA DE ALTA CARGA
        logger.info("=== Prueba de Alta Carga ===")
        start_test = time.time()
        
        # Generar 2000 eventos locales concurrentes
        local_tasks = [
            coordinator.emit_local(
                f"event_{i}", 
                {"id": i, "timestamp": time.time()}, 
                "test_system",
                # Asignar prioridades variadas
                priority=EventPriority(random.randint(0, 3))
            )
            for i in range(2000)
        ]
        
        # Generar 100 eventos externos concurrentes
        external_tasks = [
            coordinator.emit_external(
                f"ext_event_{i}", 
                {"id": i, "timestamp": time.time()}, 
                "test_system",
                # Asignar prioridades variadas
                priority=EventPriority(random.randint(0, 3))
            )
            for i in range(100)
        ]
        
        # Ejecutar en paralelo
        await asyncio.gather(*(local_tasks + external_tasks))
        
        # Dar tiempo mínimo para procesar (mucho menos que antes)
        await asyncio.sleep(0.2)
        
        # Calcular resultados de alta carga
        high_load_duration = time.time() - start_test
        
        # Resultados procesados (cálculo rápido)
        total_processed = sum(comp.stats["processed_events"] 
                             for comp in coordinator.components.values())
        
        logger.info(f"Prueba de alta carga completada en {high_load_duration:.2f}s")
        logger.info(f"Eventos emitidos: 2100, Procesados: {total_processed}")
        
        # 2. PRUEBA DE FALLOS MASIVOS
        logger.info("=== Prueba de Fallos Masivos ===")
        start_test = time.time()
        
        # Forzar fallos en 8 componentes no esenciales (40%)
        components_to_fail = [f"component_{i}" for i in range(4, 12)]
        
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
        for latency in [0.05, 0.1, 0.5, 0.8, 1.0]:
            component_id = f"component_{random.randint(12, 19)}"  # Usar componentes sanos
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
        logger.info(f"Componentes activos después de prueba: {active_components}/20")
        logger.info(f"Componentes recuperados: {recovered_count}")
        logger.info(f"Modo final del sistema: {final_mode.value}")
        
        # 5. CÁLCULO DE TASA DE ÉXITO GLOBAL
        total_duration = time.time() - start_time
        system_stats = coordinator.get_stats()
        
        # Tasa de procesamiento de eventos
        total_events_sent = 2100  # 2000 locales + 100 externos
        events_processed = total_processed
        event_process_rate = (events_processed / (total_events_sent * len(coordinator.components))) * 100
        
        # Tasa de recuperación
        recovery_rate = (recovered_count / len(components_to_fail)) * 100 if components_to_fail else 100
        
        # Tasa de éxito de latencia
        latency_success_rate = (latency_success / latency_total) * 100 if latency_total else 100
        
        # Tasa global (promedio ponderado)
        global_success_rate = (
            0.5 * event_process_rate +  # 50% peso a procesamiento
            0.3 * recovery_rate +       # 30% peso a recuperación
            0.2 * latency_success_rate  # 20% peso a latencia
        )
        
        # Resultados finales
        logger.info("\n=== RESUMEN DE PRUEBA EXTREMA OPTIMIZADA ===")
        logger.info(f"Duración total: {total_duration:.2f}s")
        logger.info(f"Tasa de procesamiento de eventos: {event_process_rate:.2f}%")
        logger.info(f"Tasa de recuperación: {recovery_rate:.2f}%")
        logger.info(f"Tasa de éxito con latencia: {latency_success_rate:.2f}%")
        logger.info(f"Tasa de éxito global: {global_success_rate:.2f}%")
        logger.info(f"API calls: {system_stats['api_calls']}, "
                   f"Local events: {system_stats['local_events']}, "
                   f"External events: {system_stats['external_events']}")
        logger.info(f"Fallos: {system_stats['failures']}, Recuperaciones: {system_stats['recoveries']}")
        logger.info(f"Modo final del sistema: {system_stats['mode']}")
        
        return {
            "duration": total_duration,
            "event_process_rate": event_process_rate,
            "recovery_rate": recovery_rate,
            "latency_success_rate": latency_success_rate,
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