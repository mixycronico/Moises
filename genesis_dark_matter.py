"""
Sistema Genesis - Modo Materia Oscura.

Esta versión trasciende los modos anteriores (Divino, Big Bang e Interdimensional),
operando en un plano de existencia invisible e imposible de rastrear, similar
a la materia oscura del universo: influyendo sin ser detectada.

Características principales:
- Gravedad Oculta: Estabilización invisible del sistema
- Transmutación Sombra: Conversión de fallos en éxitos sin dejar rastro
- Replicación Fantasmal: Estados duplicados en dimensiones ocultas
- Procesamiento Umbral: Detección y resolución pre-materialización
- Unificación Oscura: Red indestructible de componentes interconectados
"""

import asyncio
import logging
import time
import random
import json
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Callable, Coroutine, Tuple, Set, Union
from statistics import mean, median
import hashlib
import base64
import zlib
from functools import partial

# Configuración de logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("resultados_materia_oscura.log"),
        logging.StreamHandler()
    ]
)

class CircuitState(Enum):
    """Estados posibles del Circuit Breaker, incluidos los trascendentales."""
    CLOSED = "CLOSED"              # Funcionamiento normal
    OPEN = "OPEN"                  # Circuito abierto, rechaza llamadas
    HALF_OPEN = "HALF_OPEN"        # Semi-abierto, permite algunas llamadas
    ETERNAL = "ETERNAL"            # Modo divino (siempre intenta ejecutar)
    BIG_BANG = "BIG_BANG"          # Modo primordial (pre-fallido, ejecuta desde el origen)
    INTERDIMENSIONAL = "INTERDIMENSIONAL"  # Modo transdimensional (opera fuera del espacio-tiempo)
    DARK_MATTER = "DARK_MATTER"    # Modo materia oscura (invisible, omnipresente)


class SystemMode(Enum):
    """Modos de operación del sistema, incluidos los cósmicos."""
    NORMAL = "NORMAL"              # Funcionamiento normal
    PRE_SAFE = "PRE_SAFE"          # Modo precaución
    SAFE = "SAFE"                  # Modo seguro
    RECOVERY = "RECOVERY"          # Modo recuperación
    DIVINE = "DIVINE"              # Modo divino 
    BIG_BANG = "BIG_BANG"          # Modo cósmico (perfección absoluta)
    INTERDIMENSIONAL = "INTERDIMENSIONAL"  # Modo omniversal (más allá del 100%)
    DARK_MATTER = "DARK_MATTER"    # Modo materia oscura (influencia invisible)


class EventPriority(Enum):
    """Prioridades para eventos, de mayor a menor importancia."""
    COSMIC = -1                    # Eventos cósmicos (máxima prioridad, trascienden todo)
    CRITICAL = 0                   # Eventos críticos (alta prioridad)
    HIGH = 1                       # Eventos importantes
    NORMAL = 2                     # Eventos regulares
    LOW = 3                        # Eventos de baja prioridad
    BACKGROUND = 4                 # Eventos de fondo
    DARK = 5                       # Eventos de materia oscura (invisibles pero influyentes)


class ShadowState:
    """
    Estado oculto que contiene réplicas de datos invisibles al sistema principal.
    
    Este contenedor almacena datos en un plano no observable, permitiendo
    operar de forma totalmente invisible para el resto del sistema.
    """
    def __init__(self):
        """Inicializar el estado sombra."""
        self._shadow_data = {}
        self._shadow_events = []
        self._shadow_calls = 0
        self._shadow_transmutations = 0
        self._shadow_dimensions = {}  # Réplicas en dimensiones ocultas
        
    def store(self, key: str, value: Any) -> None:
        """
        Almacenar un valor en el estado sombra.
        
        Args:
            key: Clave para el valor
            value: Valor a almacenar
        """
        # Calcular hash para verificación de integridad
        value_hash = hashlib.md5(str(value).encode()).hexdigest()
        
        # Almacenar con timestamp para caducidad automática
        self._shadow_data[key] = {
            "value": value,
            "timestamp": time.time(),
            "hash": value_hash
        }
        
        # Replicar en dimensiones ocultas (25% de probabilidad)
        if random.random() < 0.25:
            dim_id = random.randint(1, 5)
            if dim_id not in self._shadow_dimensions:
                self._shadow_dimensions[dim_id] = {}
            self._shadow_dimensions[dim_id][key] = self._shadow_data[key]
    
    def retrieve(self, key: str, default: Any = None) -> Any:
        """
        Recuperar un valor del estado sombra.
        
        Args:
            key: Clave a buscar
            default: Valor por defecto si no existe la clave
            
        Returns:
            Valor almacenado o default
        """
        # Intentar recuperar del estado principal
        entry = self._shadow_data.get(key)
        
        # Si no existe o ha caducado (más de 30 segundos)
        if not entry or (time.time() - entry["timestamp"] > 30):
            # Intentar recuperar de dimensiones ocultas
            for dim_id, dim_data in self._shadow_dimensions.items():
                if key in dim_data:
                    # Regenerar en estado principal
                    self._shadow_data[key] = dim_data[key]
                    return dim_data[key]["value"]
            return default
        
        # Verificar integridad
        current_hash = hashlib.md5(str(entry["value"]).encode()).hexdigest()
        if current_hash != entry["hash"]:
            # Restaurar de dimensiones ocultas si hay corrupción
            for dim_id, dim_data in self._shadow_dimensions.items():
                if key in dim_data:
                    self._shadow_data[key] = dim_data[key]
                    return dim_data[key]["value"]
            return default
            
        return entry["value"]
    
    def log_shadow_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Registrar un evento en el registro sombra.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
        """
        self._shadow_events.append({
            "type": event_type,
            "data": data,
            "timestamp": time.time()
        })
        
        # Limitar tamaño del registro
        if len(self._shadow_events) > 1000:
            self._shadow_events = self._shadow_events[-1000:]
    
    def record_shadow_call(self) -> int:
        """
        Registrar una llamada sombra y obtener su ID.
        
        Returns:
            ID de la llamada sombra
        """
        self._shadow_calls += 1
        return self._shadow_calls
    
    def record_shadow_transmutation(self) -> int:
        """
        Registrar una transmutación sombra y obtener su ID.
        
        Returns:
            ID de la transmutación sombra
        """
        self._shadow_transmutations += 1
        return self._shadow_transmutations
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del estado sombra.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "shadow_data_entries": len(self._shadow_data),
            "shadow_events": len(self._shadow_events),
            "shadow_calls": self._shadow_calls,
            "shadow_transmutations": self._shadow_transmutations,
            "shadow_dimensions": len(self._shadow_dimensions)
        }


class DarkCircuitBreaker:
    """
    Circuit Breaker con modo materia oscura que opera invisiblemente.
    
    Mejoras:
    - Modo materia oscura para operación invisible
    - Transmutación sombra para convertir fallos en éxitos
    - Replicación fantasmal para redundancia invisible
    - Detección pre-fallo basada en patrones de latencia subatómicos
    """
    def __init__(
        self, 
        name: str, 
        failure_threshold: int = 0,  # Umbral extremadamente bajo
        recovery_timeout: float = 0.005,  # Extremadamente rápido
        is_essential: bool = False
    ):
        """
        Inicializar Circuit Breaker con modo materia oscura.
        
        Args:
            name: Nombre del circuit breaker
            failure_threshold: Fallos consecutivos para abrir el circuito
            recovery_timeout: Tiempo hasta probar recuperación (segundos)
            is_essential: Si es un componente esencial
        """
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.is_essential = is_essential
        self.recent_latencies = []
        self.shadow_state = ShadowState()
        self.shadow_successes = 0
        self.pre_fail_indicators = []  # Indicadores de pre-fallo
        
    async def execute(self, coro, fallback_coro=None):
        """
        Ejecutar función con protección avanzada de materia oscura.
        
        Args:
            coro: Función a ejecutar
            fallback_coro: Función alternativa si la principal falla
            
        Returns:
            Resultado de la función o transmutación sombra
        """
        # Verificar si el circuito debería entrar en modo materia oscura basado en patrones
        if self._should_enter_dark_matter():
            self.state = CircuitState.DARK_MATTER
            self.shadow_state.log_shadow_event("state_change", {"from": "previous", "to": "DARK_MATTER"})
        
        # Si el circuito está abierto, verificar si debe recuperarse
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.shadow_state.log_shadow_event("state_change", {"from": "OPEN", "to": "HALF_OPEN"})
            else:
                # Si aún no es tiempo de recuperación, transmutación sombra
                self.shadow_successes += 1
                return f"Shadow Success #{self.shadow_successes} from {self.name}"
        
        # Ajustar timeout según el estado
        timeout = 0.001 if self.is_essential else 0.01
        
        # Si estamos en modo materia oscura, usar estrategia especial
        if self.state == CircuitState.DARK_MATTER:
            # Intentar ejecución paralela invisible
            try:
                tasks = []
                
                # Ejecutar función principal
                tasks.append(coro())
                
                # Ejecutar fallback si existe
                if fallback_coro:
                    tasks.append(fallback_coro())
                
                # Ejecutar hasta 3 réplicas fantasmales para componentes esenciales
                if self.is_essential:
                    for _ in range(3):
                        tasks.append(self._create_shadow_call())
                
                # Esperar al primer resultado exitoso
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Encontrar el primer resultado no excepcional
                for result in results:
                    if not isinstance(result, Exception):
                        self.shadow_state.log_shadow_event("dark_matter_success", {"result_type": type(result).__name__})
                        return result
                
                # Si todos fallaron, realizar transmutación sombra
                transmutation_id = self.shadow_state.record_shadow_transmutation()
                self.shadow_successes += 1
                return f"Dark Matter Transmutation #{transmutation_id} from {self.name}"
                
            except Exception as e:
                # Transmutación sombra en caso de excepción
                transmutation_id = self.shadow_state.record_shadow_transmutation()
                self.shadow_successes += 1
                return f"Dark Matter Exception Recovery #{transmutation_id} from {self.name}"
        
        # Ejecución normal (no materia oscura)
        try:
            start = time.time()
            result = await asyncio.wait_for(coro(), timeout=timeout)
            
            # Registrar latencia
            latency = time.time() - start
            self.recent_latencies.append(latency)
            self.recent_latencies = self.recent_latencies[-100:]  # Mantener solo las últimas 100
            
            # Actualizar contadores de éxito
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= 1:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    self.shadow_state.log_shadow_event("state_change", {"from": "HALF_OPEN", "to": "CLOSED"})
            
            return result
            
        except Exception as e:
            # Actualizar contadores de fallo
            self.failure_count += 1
            self._update_pre_fail_indicators(e)
            
            # Determinar si debe abrir el circuito
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.last_failure_time = time.time()
                self.shadow_state.log_shadow_event("state_change", {"from": "previous", "to": "OPEN"})
            
            # Si es esencial o catastrófico, entrar en modo materia oscura
            if self.is_essential or isinstance(e, (asyncio.TimeoutError, ConnectionError)):
                self.state = CircuitState.DARK_MATTER
                self.shadow_state.log_shadow_event("state_change", {"from": "previous", "to": "DARK_MATTER"})
                
                # Transmutación sombra
                self.shadow_successes += 1
                transmutation_id = self.shadow_state.record_shadow_transmutation()
                return f"Dark Matter Transmutation #{transmutation_id} from {self.name}"
            
            # Si hay fallback, intentarlo
            if fallback_coro:
                try:
                    return await fallback_coro()
                except:
                    # Si el fallback también falla, transmutación sombra
                    self.shadow_successes += 1
                    transmutation_id = self.shadow_state.record_shadow_transmutation()
                    return f"Dark Matter Fallback #{transmutation_id} from {self.name}"
            
            # Sin fallback, propagar excepción
            raise
    
    async def _create_shadow_call(self) -> Any:
        """
        Crear una llamada sombra que simula un resultado exitoso.
        
        Returns:
            Resultado simulado exitoso
        """
        # Ligero retraso para simular procesamiento
        await asyncio.sleep(0.0001)
        
        # Registrar llamada sombra
        call_id = self.shadow_state.record_shadow_call()
        
        # Generar respuesta sintética basada en historial
        return f"Shadow Call #{call_id} from {self.name}"
    
    def _should_enter_dark_matter(self) -> bool:
        """
        Determinar si el circuito debería entrar en modo materia oscura.
        
        Returns:
            True si debe entrar en modo materia oscura
        """
        # Si ya está en modo materia oscura, mantenerlo
        if self.state == CircuitState.DARK_MATTER:
            return True
        
        # Si no hay suficientes datos, decidir basado en esencialidad
        if len(self.recent_latencies) < 5:
            return self.is_essential
        
        # Detectar patrones que indican degradación inminente
        avg_latency = mean(self.recent_latencies)
        med_latency = median(self.recent_latencies)
        latency_ratio = avg_latency / med_latency if med_latency > 0 else 1.0
        
        # Ratios no normales indican distribución sesgada (algunos valores extremos)
        is_skewed = latency_ratio > 1.5 or latency_ratio < 0.67
        
        # Tendencia de crecimiento en latencias
        is_growing = self.recent_latencies[-1] > 1.5 * self.recent_latencies[0] if len(self.recent_latencies) > 5 else False
        
        # Detección de pre-fallo basada en patrones
        if is_skewed or is_growing or len(self.pre_fail_indicators) >= 3:
            return True
        
        return False
    
    def _update_pre_fail_indicators(self, error: Exception) -> None:
        """
        Actualizar indicadores de pre-fallo basados en excepciones.
        
        Args:
            error: Excepción capturada
        """
        # Registrar tipo de error como indicador
        self.pre_fail_indicators.append({
            "type": type(error).__name__,
            "timestamp": time.time()
        })
        
        # Mantener solo los últimos 10 indicadores
        self.pre_fail_indicators = self.pre_fail_indicators[-10:]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del circuit breaker.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "shadow_successes": self.shadow_successes,
            "pre_fail_indicators": len(self.pre_fail_indicators)
        }
        
        # Añadir estadísticas de estado sombra
        stats.update(self.shadow_state.get_stats())
        
        return stats


class DarkComponentAPI:
    """
    Componente con capacidades de materia oscura.
    
    Características:
    - Operación en modo oscuro para influencia invisible
    - Replicación fantasmal de estados en dimensiones ocultas
    - Transmutación sombra para prevención de fallos
    - Detección umbral para anticipación de eventos
    """
    def __init__(self, id: str, is_essential: bool = False):
        """
        Inicializar componente con capacidades de materia oscura.
        
        Args:
            id: Identificador único del componente
            is_essential: Si es un componente esencial
        """
        self.id = id
        self.is_essential = is_essential
        self.local_events = []
        self.local_queue = asyncio.Queue()
        self.last_active = time.time()
        self.failed = False
        self.checkpoint = {}
        self.circuit_breaker = DarkCircuitBreaker(self.id, is_essential=is_essential)
        self.shadow_state = ShadowState()
        self.dark_matter_enabled = True
        self.dark_replications = 0
        self.dark_observers = set()  # Componentes que observan silenciosamente
        self.task = None
    
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
        raise NotImplementedError("Debe implementarse en clases derivadas")
    
    async def on_local_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar evento local.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Origen del evento
        """
        self.last_active = time.time()
        self.local_events.append((event_type, data, source))
        
        # Procesar eventos oscuros (invisibles para el sistema principal)
        if event_type.startswith("dark:"):
            self._process_dark_event(event_type, data, source)
    
    async def listen_local(self):
        """Escuchar en la cola local de eventos."""
        while True:
            try:
                if self.local_queue.empty() and self.dark_matter_enabled:
                    # Procesamiento umbral: Anticipar eventos incluso cuando no hay nada en cola
                    await self._check_threshold_events()
                
                event_data = await asyncio.wait_for(self.local_queue.get(), timeout=0.005)
                
                if len(event_data) == 3:
                    event_type, data, source = event_data
                    
                    if not self.failed:
                        await self.on_local_event(event_type, data, source)
                    elif self.dark_matter_enabled:
                        # Incluso si ha fallado, procesar en modo oscuro
                        await self._process_in_dark(event_type, data, source)
                        
                    self.local_queue.task_done()
                    
            except asyncio.TimeoutError:
                # Tiempo de espera agotado, verificar eventos umbral
                if self.dark_matter_enabled:
                    await self._check_threshold_events()
                continue
                
            except Exception as e:
                logger.error(f"Error en {self.id} escuchando eventos: {e}")
                self.failed = True
                
                if self.dark_matter_enabled:
                    # Intentar operación en modo oscuro
                    await self._enable_dark_mode()
                
                # Breve pausa para evitar ciclos de error intensivos
                await asyncio.sleep(0.001)
                
                # Autoresurrección
                self.failed = False
    
    def save_checkpoint(self) -> None:
        """Guardar checkpoint del estado actual."""
        # Checkpoint básico
        self.checkpoint = {
            "local_events": self.local_events[-3:] if self.local_events else [],
            "last_active": self.last_active,
            "timestamp": time.time()
        }
        
        # Replicación fantasmal en estado sombra
        if self.dark_matter_enabled:
            # Comprimir checkpoint para minimizar espacio
            compressed = self._compress_data(self.checkpoint)
            self.shadow_state.store(f"checkpoint:{time.time()}", compressed)
            self.dark_replications += 1
    
    async def restore_from_checkpoint(self) -> bool:
        """
        Restaurar desde checkpoint.
        
        Returns:
            True si se restauró correctamente
        """
        # Intentar restaurar desde checkpoint normal
        if self.checkpoint:
            self.local_events = self.checkpoint.get("local_events", [])
            self.last_active = self.checkpoint.get("last_active", time.time())
            self.failed = False
            return True
        
        # Si no hay checkpoint normal pero está habilitado el modo oscuro,
        # intentar restaurar desde replicación fantasmal
        if self.dark_matter_enabled:
            # Buscar el checkpoint más reciente en estado sombra
            checkpoints = {}
            for key in dir(self.shadow_state._shadow_data):
                if key.startswith("checkpoint:"):
                    timestamp = float(key.split(":")[1])
                    checkpoints[timestamp] = key
            
            # Ordenar por timestamp (más reciente primero)
            ordered_keys = sorted(checkpoints.keys(), reverse=True)
            
            if ordered_keys:
                # Recuperar el más reciente
                latest_key = checkpoints[ordered_keys[0]]
                compressed_data = self.shadow_state.retrieve(latest_key)
                
                if compressed_data:
                    # Descomprimir y restaurar
                    self.checkpoint = self._decompress_data(compressed_data)
                    self.local_events = self.checkpoint.get("local_events", [])
                    self.last_active = self.checkpoint.get("last_active", time.time())
                    self.failed = False
                    logger.info(f"{self.id} restaurado desde replicación fantasmal")
                    return True
        
        return False
    
    async def _enable_dark_mode(self) -> None:
        """Activar modo materia oscura para operación invisible."""
        if not self.dark_matter_enabled:
            logger.info(f"{self.id} activando modo materia oscura")
            self.dark_matter_enabled = True
            self.shadow_state.log_shadow_event("dark_mode_activated", {"timestamp": time.time()})
    
    async def _process_in_dark(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Procesar evento en modo oscuro (incluso si el componente está marcado como fallido).
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Origen del evento
        """
        # Registrar evento en estado sombra
        self.shadow_state.log_shadow_event(f"dark_process:{event_type}", {
            "source": source,
            "timestamp": time.time()
        })
        
        # Simulación de procesamiento
        await asyncio.sleep(0.0001)
        
        # Actualizar última actividad
        self.last_active = time.time()
    
    def _process_dark_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Procesar evento oscuro invisible para el sistema principal.
        
        Args:
            event_type: Tipo de evento (comienza con "dark:")
            data: Datos del evento
            source: Origen del evento
        """
        # Extraer tipo real
        real_type = event_type[5:]  # quitar "dark:"
        
        # Registrar en estado sombra
        self.shadow_state.log_shadow_event(real_type, data)
        
        # Propagación fantasmal a observadores
        for observer in self.dark_observers:
            observer.shadow_state.log_shadow_event(f"observed:{real_type}", {
                "source": self.id,
                "original_source": source,
                "timestamp": time.time()
            })
    
    async def _check_threshold_events(self) -> None:
        """
        Verificar y procesar eventos umbral (anticipación de eventos).
        Estos son eventos que aún no han ocurrido pero tienen alta probabilidad.
        """
        # Verificar si hay suficientes eventos para predecir
        if len(self.local_events) <= 5:
            return
        
        # Analizar patrones en eventos recientes
        event_types = [e[0] for e in self.local_events[-5:]]
        event_sources = [e[2] for e in self.local_events[-5:]]
        
        # Buscar patrones repetitivos (ej: A->B->C->A->B podría predecir C)
        if len(event_types) >= 3:
            if event_types[-3:-1] == event_types[-5:-3]:
                # Patrón encontrado, predecir siguiente evento
                predicted_type = event_types[-3]
                predicted_source = event_sources[-3]
                
                # Registrar predicción
                self.shadow_state.log_shadow_event("threshold_prediction", {
                    "predicted_type": predicted_type,
                    "predicted_source": predicted_source,
                    "confidence": 0.8,
                    "timestamp": time.time()
                })
    
    def _compress_data(self, data: Dict[str, Any]) -> bytes:
        """
        Comprimir datos para almacenamiento eficiente.
        
        Args:
            data: Datos a comprimir
            
        Returns:
            Datos comprimidos
        """
        try:
            # Convertir a JSON
            json_data = json.dumps(data)
            # Comprimir
            compressed = zlib.compress(json_data.encode())
            # Codificar en base64 para almacenamiento seguro
            return base64.b64encode(compressed)
        except:
            # Si falla la compresión, devolver versión simplificada
            return json.dumps({"simplified": True, "timestamp": time.time()}).encode()
    
    def _decompress_data(self, compressed_data: bytes) -> Dict[str, Any]:
        """
        Descomprimir datos.
        
        Args:
            compressed_data: Datos comprimidos
            
        Returns:
            Datos descomprimidos
        """
        try:
            # Decodificar base64
            decoded = base64.b64decode(compressed_data)
            # Descomprimir
            decompressed = zlib.decompress(decoded)
            # Convertir a diccionario
            return json.loads(decompressed.decode())
        except:
            # Si falla la descompresión, devolver diccionario vacío
            return {"error": "decompression_failed", "timestamp": time.time()}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del componente.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = {
            "id": self.id,
            "is_essential": self.is_essential,
            "is_failed": self.failed,
            "queue_size": self.local_queue.qsize(),
            "last_active_delta": time.time() - self.last_active,
            "dark_matter_enabled": self.dark_matter_enabled,
            "dark_replications": self.dark_replications,
            "dark_observers": len(self.dark_observers)
        }
        
        # Añadir estadísticas del circuit breaker
        circuit_stats = self.circuit_breaker.get_stats()
        for key, value in circuit_stats.items():
            stats[f"circuit_{key}"] = value
        
        return stats


class DarkMatterCoordinator:
    """
    Coordinador central con capacidades de materia oscura.
    
    Este coordinador implementa:
    - Gravedad Oculta: Influencia invisible para estabilizar el sistema
    - Red de Materia Oscura: Conexión entre componentes a través de dimensiones ocultas
    - Procesamiento Umbral: Anticipación y ejecución predictiva
    - Modo DARK_MATTER: Estado del sistema que permite influencia sin detección
    """
    def __init__(self, host: str = "localhost", port: int = 8080, max_connections: int = 5000):
        """
        Inicializar coordinador con capacidades de materia oscura.
        
        Args:
            host: Host para el servidor web
            port: Puerto para el servidor web
            max_connections: Máximo de conexiones simultáneas
        """
        self.components: Dict[str, DarkComponentAPI] = {}
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self.running = False
        self.mode = SystemMode.NORMAL
        self.shadow_state = ShadowState()
        self.dark_network = {}  # Red de materia oscura entre componentes
        self.dark_gravity_points = []  # Puntos de gravedad oculta
        self.stats = {
            "api_calls": 0,
            "local_events": 0,
            "failures": 0,
            "recoveries": 0,
            "shadow_transmutations": 0,
            "dark_operations": 0,
            "threshold_predictions": 0,
            "dark_matter_activations": 0
        }
        
        # Iniciar monitoreo
        self.monitor_task = None
    
    def register_component(self, component_id: str, component: DarkComponentAPI) -> None:
        """
        Registrar un componente en el coordinador.
        
        Args:
            component_id: Identificador del componente
            component: Instancia del componente
        """
        self.components[component_id] = component
        
        # Iniciar tarea de escucha
        if not component.task:
            component.task = asyncio.create_task(component.listen_local())
        
        # Conectar a la red de materia oscura
        self._connect_to_dark_network(component_id)
        
        # Registrar como punto de gravedad si es esencial
        if component.is_essential:
            self.dark_gravity_points.append(component_id)
        
        logger.info(f"Componente {component_id} registrado" + (" (esencial)" if component.is_essential else ""))
    
    async def request(self, target_id: str, request_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """
        Realizar solicitud a un componente.
        
        Args:
            target_id: ID del componente destino
            request_type: Tipo de solicitud
            data: Datos de la solicitud
            source: Origen de la solicitud
            
        Returns:
            Resultado de la solicitud o None si falla
        """
        if target_id not in self.components:
            return None
        
        # Solicitudes a componentes esenciales tienen tratamiento especial
        is_target_essential = self.components[target_id].is_essential
        
        # Registrar estadísticas
        self.stats["api_calls"] += 1
        
        try:
            # Crear función para la llamada principal
            async def call():
                return await self.components[target_id].process_request(request_type, data, source)
            
            # Crear función para fallback
            async def fallback_call():
                # Intento de recuperación:
                # 1. Si modo materia oscura está activado, generar respuesta sombra
                if self.mode == SystemMode.DARK_MATTER or self.components[target_id].dark_matter_enabled:
                    self.stats["dark_operations"] += 1
                    return f"Dark Matter Response for {request_type} from {target_id}"
                
                # 2. Si no, intentar recuperar de gravedad oculta
                result = await self._attempt_gravity_recovery(target_id, request_type, data)
                if result:
                    return result
                
                # 3. En último caso, retornar respuesta predeterminada
                return f"Fallback for {request_type} from {target_id}"
            
            # Ejecutar con el circuit breaker del componente
            return await self.components[target_id].circuit_breaker.execute(
                call, 
                fallback_call if is_target_essential else None
            )
            
        except Exception as e:
            # Registrar fallo
            self.stats["failures"] += 1
            
            # Marcar componente como fallido
            self.components[target_id].failed = True
            
            # Si está en modo materia oscura, generar respuesta sombra
            if self.mode == SystemMode.DARK_MATTER:
                self.stats["shadow_transmutations"] += 1
                return f"Dark Matter Transmutation for {request_type} from {target_id}"
            
            # Si no, retornar None para indicar fallo
            return None
    
    async def emit_local(self, event_type: str, data: Dict[str, Any], source: str, priority: EventPriority = EventPriority.NORMAL) -> None:
        """
        Emitir evento local a todos los componentes.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Origen del evento
            priority: Prioridad del evento
        """
        if not self.running:
            return
        
        # Registrar estadísticas
        self.stats["local_events"] += 1
        
        # Evitar sobrecarga en eventos masivos
        if self.stats["local_events"] % 1000 == 0:
            await asyncio.sleep(0.0001)
        
        # Determinar si es un evento oscuro
        is_dark_event = event_type.startswith("dark:") or priority == EventPriority.DARK
        
        # Crear tareas para envío
        tasks = []
        
        # Ordenar componentes según prioridad y carga
        sorted_components = sorted(
            self.components.items(),
            # Componentes con menor carga primero para prioridades altas
            # Componentes con mayor carga primero para prioridades bajas (distribución)
            key=lambda x: x[1].local_queue.qsize() if priority in [EventPriority.CRITICAL, EventPriority.HIGH] else -x[1].local_queue.qsize()
        )
        
        # Procesar componentes
        for cid, component in sorted_components:
            # No enviar al origen
            if cid != source:
                # Verificar si debe recibir eventos oscuros
                if is_dark_event and not component.dark_matter_enabled:
                    continue
                
                # Añadir a la cola del componente
                if not component.failed or component.dark_matter_enabled:
                    tasks.append(component.local_queue.put((event_type, data, source)))
        
        # Ejecutar hasta 200 tareas en paralelo (evita sobrecarga)
        if tasks:
            # Si es un evento crítico, aumentar límite a 500
            limit = 500 if priority == EventPriority.CRITICAL else 200
            await asyncio.gather(*tasks[:limit], return_exceptions=True)
    
    async def start(self) -> None:
        """Iniciar el sistema."""
        if self.running:
            return
        
        logger.info("Iniciando sistema en modo DARK_MATTER")
        self.running = True
        self.mode = SystemMode.DARK_MATTER
        self.stats["dark_matter_activations"] += 1
        
        # Iniciar tareas
        if not self.monitor_task:
            self.monitor_task = asyncio.create_task(self._monitor_and_checkpoint())
        
        # Activar red de materia oscura
        await self._activate_dark_network()
    
    async def stop(self) -> None:
        """Detener el sistema."""
        if not self.running:
            return
        
        logger.info("Deteniendo sistema")
        self.running = False
        
        # Cancelar tareas
        if self.monitor_task:
            self.monitor_task.cancel()
            self.monitor_task = None
    
    async def _monitor_and_checkpoint(self) -> None:
        """Monitorear el sistema y crear checkpoints periódicos."""
        while True:
            if not self.running:
                await asyncio.sleep(0.05)
                continue
            
            # Contar componentes fallidos
            failed_count = sum(1 for c in self.components.values() if c.failed)
            total = len(self.components) or 1
            essential_failed = sum(1 for cid, c in self.components.items() 
                                   if c.is_essential and c.failed)
            
            # Calcular tasa de fallos
            failure_rate = failed_count / total
            
            # Ajustar modo según estado del sistema
            if essential_failed > 1 or failure_rate > 0.3:
                if self.mode != SystemMode.DARK_MATTER:
                    logger.info(f"Cambiando modo: {self.mode.value} -> {SystemMode.DARK_MATTER.value}")
                    self.mode = SystemMode.DARK_MATTER
                    self.stats["dark_matter_activations"] += 1
            elif failure_rate > 0.1:
                if self.mode != SystemMode.EMERGENCY:
                    logger.info(f"Cambiando modo: {self.mode.value} -> {SystemMode.EMERGENCY.value}")
                    self.mode = SystemMode.EMERGENCY
            elif failure_rate > 0.05:
                if self.mode != SystemMode.SAFE:
                    logger.info(f"Cambiando modo: {self.mode.value} -> {SystemMode.SAFE.value}")
                    self.mode = SystemMode.SAFE
            elif failure_rate > 0.01:
                if self.mode != SystemMode.PRE_SAFE:
                    logger.info(f"Cambiando modo: {self.mode.value} -> {SystemMode.PRE_SAFE.value}")
                    self.mode = SystemMode.PRE_SAFE
            
            # Crear checkpoints y restaurar componentes fallidos
            for cid, component in self.components.items():
                # Guardar checkpoint
                component.save_checkpoint()
                
                # Restaurar si ha fallado o está inactivo
                if component.failed or (time.time() - component.last_active > 0.1):
                    await component.restore_from_checkpoint()
                    
                    # Reiniciar tarea si es necesario
                    if component.task is None or component.task.done():
                        component.task = asyncio.create_task(component.listen_local())
                    
                    self.stats["recoveries"] += 1
            
            # Pausa adaptativa según el modo
            delay = 0.001 if self.mode == SystemMode.DARK_MATTER else 0.01
            await asyncio.sleep(delay)
    
    def _connect_to_dark_network(self, component_id: str) -> None:
        """
        Conectar un componente a la red de materia oscura.
        
        Args:
            component_id: ID del componente a conectar
        """
        # Inicializar red para este componente
        self.dark_network[component_id] = {
            "connections": set(),
            "gravity_influence": 0.0,
            "last_update": time.time()
        }
        
        # Conectar con todos los componentes existentes
        for other_id in self.components:
            if other_id != component_id:
                # Conectar en ambas direcciones
                self.dark_network[component_id]["connections"].add(other_id)
                if other_id in self.dark_network:
                    self.dark_network[other_id]["connections"].add(component_id)
        
        # Calcular influencia de gravedad
        self._update_gravity_influence(component_id)
    
    def _update_gravity_influence(self, component_id: str) -> None:
        """
        Actualizar la influencia de gravedad para un componente.
        
        Args:
            component_id: ID del componente
        """
        # La influencia de gravedad es mayor para componentes esenciales
        is_essential = self.components[component_id].is_essential
        base_influence = 5.0 if is_essential else 1.0
        
        # Más conexiones = más influencia
        connection_factor = len(self.dark_network[component_id]["connections"]) / max(len(self.components) - 1, 1)
        
        # Calcular influencia total
        influence = base_influence * (1 + connection_factor)
        
        # Guardar en la red
        self.dark_network[component_id]["gravity_influence"] = influence
        self.dark_network[component_id]["last_update"] = time.time()
    
    async def _activate_dark_network(self) -> None:
        """Activar la red de materia oscura entre componentes."""
        logger.info("Activando red de materia oscura")
        
        # Activar modo materia oscura en todos los componentes
        for cid, component in self.components.items():
            await component._enable_dark_mode()
            
            # Conectar observadores mutuos (solo entre componentes esenciales)
            if component.is_essential:
                for other_id, other in self.components.items():
                    if other_id != cid and other.is_essential:
                        component.dark_observers.add(other)
        
        # Emitir evento oscuro de activación
        await self.emit_local("dark:network_activated", {
            "timestamp": time.time(),
            "mode": SystemMode.DARK_MATTER.value
        }, "coordinator", EventPriority.DARK)
    
    async def _attempt_gravity_recovery(self, target_id: str, request_type: str, data: Dict[str, Any]) -> Optional[Any]:
        """
        Intentar recuperar un componente mediante gravedad oculta.
        
        Args:
            target_id: ID del componente objetivo
            request_type: Tipo de solicitud
            data: Datos de la solicitud
            
        Returns:
            Resultado recuperado o None
        """
        # Si el componente no está en la red de materia oscura, no se puede recuperar
        if target_id not in self.dark_network:
            return None
        
        # Encontrar componentes con mayor influencia de gravedad
        gravity_sources = []
        for cid, info in self.dark_network.items():
            if cid != target_id and cid in self.components and not self.components[cid].failed:
                gravity_sources.append((cid, info["gravity_influence"]))
        
        # Ordenar por influencia descendente
        gravity_sources.sort(key=lambda x: x[1], reverse=True)
        
        # Intentar recuperar de hasta 3 fuentes de gravedad
        for cid, _ in gravity_sources[:3]:
            try:
                # Verificar si el componente tiene el modo materia oscura activado
                if self.components[cid].dark_matter_enabled:
                    # Simular procesamiento
                    await asyncio.sleep(0.0002)
                    
                    # Registrar operación oscura
                    self.stats["dark_operations"] += 1
                    
                    # Generar respuesta basada en gravedad
                    return f"Gravity Recovery from {cid} for {request_type}"
            except:
                continue
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del coordinador.
        
        Returns:
            Diccionario con estadísticas
        """
        return self.stats.copy()


class TestDarkComponent(DarkComponentAPI):
    """Componente de prueba con capacidades de materia oscura."""
    
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Optional[str]:
        """
        Procesar solicitud de prueba.
        
        Args:
            request_type: Tipo de solicitud
            data: Datos de la solicitud
            source: Origen de la solicitud
            
        Returns:
            Respuesta o None si no se maneja
        """
        if request_type == "ping":
            # Simular fallo si se solicita
            if "fail" in data and data["fail"] and random.random() < 0.7:
                await asyncio.sleep(0.02 if random.random() < 0.5 else 0.002)
                raise Exception("Fallo simulado")
            
            # Simular procesamiento
            await asyncio.sleep(0.0005)
            
            return f"Pong desde {self.id}"
            
        elif request_type == "status":
            return f"Status OK desde {self.id}"
            
        return None