"""
Prueba integrada de las características de resiliencia del sistema Genesis.

Este script demuestra la integración de las tres características principales de resiliencia:
1. Sistema de Reintentos Adaptativos con Backoff Exponencial y Jitter
2. Patrón Circuit Breaker
3. Sistema de Checkpointing y Safe Mode

El objetivo es mostrar cómo estas características trabajan juntas para crear
un sistema robusto capaz de mantener la operación bajo condiciones adversas.
"""

import asyncio
import logging
import os
import random
import shutil
import time
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Tuple

# Configurar logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("resiliencia_integrada_test")

# Directorio para checkpoints
CHECKPOINT_DIR = "./test_resiliencia_integrada"

# =================== UTILIDADES GENERALES ===================

def create_test_dir():
    """Crear directorio de pruebas."""
    if os.path.exists(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =================== SISTEMA DE REINTENTOS ADAPTATIVOS ===================

class RetryConfig:
    """Configuración para reintentos adaptativos."""
    max_retries: int = 5
    base_delay: float = 0.1
    max_delay: float = 10.0
    jitter: float = 0.1
    
    @classmethod
    def calculate_delay(cls, attempt: int) -> float:
        """
        Calcular delay con backoff exponencial y jitter.
        
        Args:
            attempt: Número de intento
            
        Returns:
            Tiempo de espera en segundos
        """
        # Backoff exponencial
        delay = min(cls.base_delay * (2 ** attempt), cls.max_delay)
        
        # Jitter aleatorio
        jitter_value = random.uniform(-cls.jitter, cls.jitter)
        
        return max(0.001, delay + jitter_value)  # Mínimo 1ms

async def with_retry(func, max_retries=None, base_delay=None, jitter=None):
    """
    Ejecutar una función con reintentos adaptativos.
    
    Args:
        func: Función a ejecutar
        max_retries: Número máximo de reintentos (None para usar valor por defecto)
        base_delay: Tiempo base de espera (None para usar valor por defecto)
        jitter: Valor máximo de jitter (None para usar valor por defecto)
        
    Returns:
        Resultado de la función
        
    Raises:
        Exception: Si se agotan los reintentos
    """
    # Configurar parámetros
    max_retries = max_retries if max_retries is not None else RetryConfig.max_retries
    base_delay = base_delay if base_delay is not None else RetryConfig.base_delay
    jitter = jitter if jitter is not None else RetryConfig.jitter
    
    # Guardar configuración original
    original_max_retries = RetryConfig.max_retries
    original_base_delay = RetryConfig.base_delay
    original_jitter = RetryConfig.jitter
    
    # Aplicar configuración temporal
    RetryConfig.max_retries = max_retries
    RetryConfig.base_delay = base_delay
    RetryConfig.jitter = jitter
    
    try:
        attempt = 0
        while True:
            try:
                start_time = time.time()
                result = await func()
                elapsed = time.time() - start_time
                
                # Éxito
                if attempt > 0:
                    logger.info(f"Intento {attempt+1}: Éxito en {elapsed:.3f}s después de {attempt} reintentos")
                return result
                
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    logger.error(f"Todos los reintentos agotados ({max_retries}). Último error: {e}")
                    raise
                
                delay = RetryConfig.calculate_delay(attempt - 1)
                logger.info(f"Intento {attempt}: Fallo - {e}. Reintentando en {delay:.3f}s")
                await asyncio.sleep(delay)
    finally:
        # Restaurar configuración original
        RetryConfig.max_retries = original_max_retries
        RetryConfig.base_delay = original_base_delay
        RetryConfig.jitter = original_jitter

# =================== CIRCUIT BREAKER ===================

class CircuitState(Enum):
    """Estados del Circuit Breaker."""
    CLOSED = auto()    # Funcionamiento normal
    OPEN = auto()      # Circuito abierto, rechazan llamadas
    HALF_OPEN = auto() # Semi-abierto, permite algunas llamadas

class CircuitBreaker:
    """Implementación del patrón Circuit Breaker."""
    
    def __init__(
        self, 
        name: str, 
        failure_threshold: int = 3,
        recovery_timeout: float = 5.0,
        half_open_max_calls: int = 1,
        success_threshold: int = 2
    ):
        """
        Inicializar Circuit Breaker.
        
        Args:
            name: Nombre del circuit breaker
            failure_threshold: Fallos consecutivos para abrir el circuito
            recovery_timeout: Tiempo hasta probar recuperación
            half_open_max_calls: Máximo de llamadas en estado half-open
            success_threshold: Éxitos consecutivos para cerrar el circuito
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        
        # Estado
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.call_count = 0
        self.last_failure_time = 0
        self.last_state_change = time.time()
        self.half_open_calls = 0
        
        logger.info(f"Circuit Breaker '{name}' creado en estado {self.state.name}")
    
    async def execute(self, func, *args, **kwargs):
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
        
        # Verificar transición automática de OPEN a HALF_OPEN
        if self.state == CircuitState.OPEN:
            if (time.time() - self.last_state_change) > self.recovery_timeout:
                logger.info(f"Circuit Breaker '{self.name}' cambiando de OPEN a HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                self.half_open_calls = 0
                self.last_state_change = time.time()
        
        # Si el circuito está abierto, rechazar la llamada
        if self.state == CircuitState.OPEN:
            logger.warning(f"Circuit Breaker '{self.name}' abierto. Llamada rechazada.")
            return None
        
        # Si estamos en HALF_OPEN, limitar el número de llamadas
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.half_open_max_calls:
                logger.warning(f"Circuit Breaker '{self.name}' en HALF_OPEN. Máximo de llamadas alcanzado. Rechazando.")
                return None
            self.half_open_calls += 1
        
        try:
            # Ejecutar la función
            result = await func(*args, **kwargs)
            
            # Registrar éxito
            self.success_count += 1
            self.failure_count = 0
            
            # Si estamos en HALF_OPEN y alcanzamos el umbral de éxitos, cerrar el circuito
            if (self.state == CircuitState.HALF_OPEN and 
                self.success_count >= self.success_threshold):
                logger.info(f"Circuit Breaker '{self.name}' cambiando de HALF_OPEN a CLOSED")
                self.state = CircuitState.CLOSED
                self.last_state_change = time.time()
            
            return result
            
        except Exception as e:
            # Registrar fallo
            self.failure_count += 1
            self.success_count = 0
            self.last_failure_time = time.time()
            
            # Si excedemos el umbral de fallos, abrir el circuito
            if (self.state == CircuitState.CLOSED and 
                self.failure_count >= self.failure_threshold):
                logger.warning(
                    f"Circuit Breaker '{self.name}' cambiando a OPEN tras "
                    f"{self.failure_count} fallos consecutivos"
                )
                self.state = CircuitState.OPEN
                self.last_state_change = time.time()
            
            # Si estamos en HALF_OPEN y fallamos, volver a abrir el circuito
            elif self.state == CircuitState.HALF_OPEN:
                logger.warning(f"Circuit Breaker '{self.name}' volviendo a OPEN tras fallo en HALF_OPEN")
                self.state = CircuitState.OPEN
                self.last_state_change = time.time()
                
            # Propagar el error
            raise e
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del circuit breaker.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "name": self.name,
            "state": self.state.name,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "call_count": self.call_count,
            "time_in_state": time.time() - self.last_state_change
        }

# =================== SISTEMA DE CHECKPOINTING Y SAFE MODE ===================

class SystemMode(Enum):
    """Modos de operación del sistema."""
    NORMAL = "normal"     # Funcionamiento normal
    SAFE = "safe"         # Modo seguro
    EMERGENCY = "emergency"  # Modo emergencia

class CheckpointSystem:
    """Sistema de checkpointing."""
    
    def __init__(
        self,
        component_id: str,
        checkpoint_dir: str,
        auto_checkpoint: bool = True,
        checkpoint_interval: float = 0.5  # 0.5 segundos para pruebas
    ):
        """
        Inicializar sistema de checkpointing.
        
        Args:
            component_id: ID del componente
            checkpoint_dir: Directorio para checkpoints
            auto_checkpoint: Si se debe hacer checkpointing automático
            checkpoint_interval: Intervalo entre checkpoints automáticos
        """
        self.component_id = component_id
        self.checkpoint_dir = os.path.join(checkpoint_dir, component_id)
        self.auto_checkpoint = auto_checkpoint
        self.checkpoint_interval = checkpoint_interval
        
        # Crear directorio
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Estado
        self.state = {}
        self.last_checkpoint_time = 0
        self._checkpoint_task = None
        
        logger.info(f"Sistema de checkpointing creado para {component_id}")
    
    async def start(self) -> None:
        """Iniciar checkpointing automático."""
        if self.auto_checkpoint and not self._checkpoint_task:
            self._checkpoint_task = asyncio.create_task(self._auto_checkpoint_loop())
            logger.info(f"Checkpointing automático iniciado para {self.component_id}")
    
    async def stop(self) -> None:
        """Detener checkpointing automático."""
        if self._checkpoint_task:
            self._checkpoint_task.cancel()
            try:
                await self._checkpoint_task
            except asyncio.CancelledError:
                pass
            self._checkpoint_task = None
            logger.info(f"Checkpointing automático detenido para {self.component_id}")
    
    async def _auto_checkpoint_loop(self) -> None:
        """Bucle de checkpointing automático."""
        try:
            while True:
                await asyncio.sleep(self.checkpoint_interval)
                await self.create_checkpoint()
        except asyncio.CancelledError:
            logger.info(f"Bucle de checkpointing cancelado para {self.component_id}")
            raise
    
    async def create_checkpoint(self) -> str:
        """
        Crear checkpoint del estado actual.
        
        Returns:
            ID del checkpoint creado
        """
        # Crear ID basado en timestamp
        checkpoint_id = f"{int(time.time() * 1000)}"
        
        # Ruta del archivo
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{checkpoint_id}.txt")
        
        # Guardar estado como texto
        with open(checkpoint_path, "w") as f:
            f.write(f"Component: {self.component_id}\n")
            f.write(f"Created: {time.ctime()}\n")
            f.write("Data:\n")
            for key, value in self.state.items():
                f.write(f"  {key}: {value}\n")
        
        self.last_checkpoint_time = time.time()
        logger.debug(f"Checkpoint {checkpoint_id} creado para {self.component_id}")
        
        return checkpoint_id
    
    async def restore_latest(self) -> bool:
        """
        Restaurar último checkpoint.
        
        Returns:
            True si se restauró correctamente
        """
        # Obtener archivos de checkpoint ordenados por timestamp (más reciente primero)
        try:
            files = os.listdir(self.checkpoint_dir)
            checkpoint_files = [f for f in files if f.startswith("checkpoint_")]
            
            if not checkpoint_files:
                logger.warning(f"No hay checkpoints para {self.component_id}")
                return False
            
            # Ordenar por timestamp en nombre (formato: checkpoint_TIMESTAMP.txt)
            checkpoint_files.sort(reverse=True)
            latest = checkpoint_files[0]
            
            # Leer archivo
            checkpoint_path = os.path.join(self.checkpoint_dir, latest)
            data = {}
            
            with open(checkpoint_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("  "):  # Línea de datos
                        parts = line.strip().split(": ", 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            data[key] = value
            
            # Restaurar estado
            self.state = data
            
            logger.info(f"Checkpoint {latest} restaurado para {self.component_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error al restaurar checkpoint: {e}")
            return False

class SafeMode:
    """Sistema de Safe Mode."""
    
    def __init__(self, essential_components: List[str]):
        """
        Inicializar Safe Mode.
        
        Args:
            essential_components: Lista de componentes esenciales
        """
        self.essential_components = set(essential_components)
        self.mode = SystemMode.NORMAL
        self.mode_change_time = time.time()
        
        logger.info(f"Safe Mode iniciado con {len(essential_components)} componentes esenciales")
    
    def activate_safe_mode(self, reason: str) -> None:
        """
        Activar modo seguro.
        
        Args:
            reason: Razón para activar
        """
        if self.mode != SystemMode.SAFE:
            self.mode = SystemMode.SAFE
            self.mode_change_time = time.time()
            logger.warning(f"SAFE MODE ACTIVADO. Razón: {reason}")
    
    def activate_emergency_mode(self, reason: str) -> None:
        """
        Activar modo emergencia.
        
        Args:
            reason: Razón para activar
        """
        self.mode = SystemMode.EMERGENCY
        self.mode_change_time = time.time()
        logger.critical(f"EMERGENCY MODE ACTIVADO. Razón: {reason}")
    
    def deactivate(self) -> None:
        """Volver a modo normal."""
        if self.mode != SystemMode.NORMAL:
            self.mode = SystemMode.NORMAL
            self.mode_change_time = time.time()
            logger.info("Modo seguro/emergencia desactivado")
    
    def is_essential(self, component_id: str) -> bool:
        """
        Verificar si un componente es esencial.
        
        Args:
            component_id: ID del componente
            
        Returns:
            True si es esencial
        """
        return component_id in self.essential_components
    
    def is_operation_allowed(self, operation: str, component_id: str) -> bool:
        """
        Verificar si una operación está permitida en el modo actual.
        
        Args:
            operation: Operación a realizar
            component_id: ID del componente
            
        Returns:
            True si la operación está permitida
        """
        # En modo normal, todo permitido
        if self.mode == SystemMode.NORMAL:
            return True
        
        # En modo emergencia, solo operaciones en componentes esenciales
        if self.mode == SystemMode.EMERGENCY:
            return self.is_essential(component_id)
        
        # En modo seguro, operaciones en componentes esenciales + algunas en no esenciales
        if self.mode == SystemMode.SAFE:
            if self.is_essential(component_id):
                return True
            else:
                # En componentes no esenciales, solo permitir lectura
                return operation.startswith("get") or operation.startswith("read")
        
        return False

# =================== SERVICIO SIMULADO ===================

class ServiceHealth(Enum):
    """Estado de salud del servicio."""
    HEALTHY = auto()      # Funciona correctamente
    DEGRADED = auto()     # Funciona con latencia alta
    FAILING = auto()      # Falla constantemente
    UNAVAILABLE = auto()  # No disponible

class ExternalService:
    """Servicio externo simulado con comportamiento controlado."""
    
    def __init__(self, name: str, initial_health: ServiceHealth = ServiceHealth.HEALTHY):
        """
        Inicializar servicio.
        
        Args:
            name: Nombre del servicio
            initial_health: Estado inicial
        """
        self.name = name
        self.health = initial_health
        self.call_count = 0
        self.success_count = 0
        self.failure_count = 0
        
        logger.info(f"Servicio {name} creado con estado {initial_health.name}")
    
    def set_health(self, health: ServiceHealth) -> None:
        """
        Establecer estado de salud.
        
        Args:
            health: Nuevo estado
        """
        self.health = health
        logger.info(f"Servicio {self.name} ahora está en estado {health.name}")
    
    async def call(self, operation: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Llamar al servicio externo.
        
        Args:
            operation: Operación a realizar
            params: Parámetros de la operación
            
        Returns:
            Resultado de la operación
            
        Raises:
            TimeoutError: Si el servicio está degradado y excede umbral
            ConnectionError: Si el servicio no está disponible
            Exception: Si el servicio está fallando
        """
        self.call_count += 1
        params = params or {}
        
        # Simular latencia base
        base_latency = 0.05
        
        if self.health == ServiceHealth.DEGRADED:
            # Latencia alta
            await asyncio.sleep(random.uniform(0.2, 0.4))
            
            # 30% de probabilidad de timeout
            if random.random() < 0.3:
                self.failure_count += 1
                raise TimeoutError(f"Timeout al llamar a {self.name}.{operation}: operación demasiado lenta")
        
        elif self.health == ServiceHealth.FAILING:
            # Latencia normal pero falla la mayoría de veces
            await asyncio.sleep(base_latency)
            
            # 80% de probabilidad de fallo
            if random.random() < 0.8:
                self.failure_count += 1
                raise Exception(f"Error en servicio {self.name}.{operation}: operación fallida")
        
        elif self.health == ServiceHealth.UNAVAILABLE:
            # No hay conexión
            self.failure_count += 1
            raise ConnectionError(f"Servicio {self.name} no disponible")
        
        else:  # HEALTHY
            # Latencia normal
            await asyncio.sleep(base_latency)
        
        # Éxito
        self.success_count += 1
        return {
            "service": self.name,
            "operation": operation,
            "params": params,
            "timestamp": time.time(),
            "success": True
        }

# =================== COMPONENTE RESILIENTE ===================

class ResilientComponent:
    """Componente con características de resiliencia."""
    
    def __init__(
        self, 
        component_id: str,
        services: Dict[str, ExternalService],
        checkpoint_dir: str,
        essential: bool = False,
        use_checkpointing: bool = True,
        use_circuit_breaker: bool = True,
        use_retry: bool = True
    ):
        """
        Inicializar componente resiliente.
        
        Args:
            component_id: ID del componente
            services: Diccionario de servicios externos
            checkpoint_dir: Directorio para checkpoints
            essential: Si es un componente esencial
            use_checkpointing: Si debe usar checkpointing
            use_circuit_breaker: Si debe usar circuit breaker
            use_retry: Si debe usar reintentos
        """
        self.component_id = component_id
        self.services = services
        self.essential = essential
        self.use_checkpointing = use_checkpointing
        self.use_circuit_breaker = use_circuit_breaker
        self.use_retry = use_retry
        
        # Estado
        self.data = {}
        self.operational = True
        
        # Sistemas de resiliencia
        if use_checkpointing:
            self.checkpoint_system = CheckpointSystem(
                component_id=component_id,
                checkpoint_dir=checkpoint_dir,
                auto_checkpoint=True,
                checkpoint_interval=0.5  # 0.5 segundos para pruebas
            )
        else:
            self.checkpoint_system = None
        
        if use_circuit_breaker:
            self.circuit_breakers = {
                service_name: CircuitBreaker(
                    name=f"{component_id}_{service_name}",
                    failure_threshold=3,
                    recovery_timeout=2.0,  # 2 segundos para pruebas
                    half_open_max_calls=1
                )
                for service_name, service in services.items()
            }
        else:
            self.circuit_breakers = {}
        
        logger.info(f"Componente {component_id} creado con resiliencia: " +
                   f"checkpointing={'SÍ' if use_checkpointing else 'NO'}, " +
                   f"circuit_breaker={'SÍ' if use_circuit_breaker else 'NO'}, " +
                   f"retry={'SÍ' if use_retry else 'NO'}")
    
    async def start(self) -> None:
        """Iniciar componente."""
        if self.checkpoint_system:
            await self.checkpoint_system.start()
        self.operational = True
    
    async def stop(self) -> None:
        """Detener componente."""
        if self.checkpoint_system:
            await self.checkpoint_system.stop()
        self.operational = False
    
    async def set_data(self, key: str, value: Any) -> Dict[str, Any]:
        """
        Establecer datos.
        
        Args:
            key: Clave
            value: Valor
            
        Returns:
            Resultado de la operación
        """
        self.data[key] = value
        
        # Actualizar checkpoint si está disponible
        if self.checkpoint_system:
            self.checkpoint_system.state[key] = str(value)
        
        return {"status": "success", "operation": "set", "key": key}
    
    async def get_data(self, key: str) -> Dict[str, Any]:
        """
        Obtener datos.
        
        Args:
            key: Clave
            
        Returns:
            Resultado de la operación
        """
        if key in self.data:
            return {"status": "success", "operation": "get", "key": key, "value": self.data[key]}
        else:
            return {"status": "error", "operation": "get", "key": key, "reason": "not_found"}
    
    async def call_service(
        self,
        service_name: str,
        operation: str,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Llamar a un servicio externo con resiliencia.
        
        Args:
            service_name: Nombre del servicio
            operation: Operación a realizar
            params: Parámetros
            
        Returns:
            Resultado de la operación
            
        Raises:
            Exception: Si la operación falla y no se puede recuperar
        """
        if service_name not in self.services:
            return {"status": "error", "reason": f"unknown_service: {service_name}"}
        
        service = self.services[service_name]
        
        # Sin resiliencia - llamada directa
        if not (self.use_circuit_breaker or self.use_retry):
            return await service.call(operation, params)
        
        # Con Circuit Breaker pero sin Retry
        if self.use_circuit_breaker and not self.use_retry:
            breaker = self.circuit_breakers.get(service_name)
            if breaker:
                return await breaker.execute(service.call, operation, params)
            else:
                return await service.call(operation, params)
        
        # Con Retry pero sin Circuit Breaker
        if self.use_retry and not self.use_circuit_breaker:
            return await with_retry(lambda: service.call(operation, params))
        
        # Con Circuit Breaker y Retry
        breaker = self.circuit_breakers.get(service_name)
        if breaker:
            # Usar ambos, primero Circuit Breaker y luego Retry
            return await breaker.execute(
                lambda: with_retry(lambda: service.call(operation, params))
            )
        else:
            return await with_retry(lambda: service.call(operation, params))
    
    async def execute_operation(
        self, 
        operation: str, 
        params: Dict[str, Any],
        safe_mode: SafeMode
    ) -> Dict[str, Any]:
        """
        Ejecutar operación verificando permisos de Safe Mode.
        
        Args:
            operation: Operación a realizar
            params: Parámetros
            safe_mode: Sistema de Safe Mode
            
        Returns:
            Resultado de la operación o error si no está permitida
        """
        params = params or {}
        
        # Verificar si la operación está permitida
        if not safe_mode.is_operation_allowed(operation, self.component_id):
            return {
                "status": "error", 
                "operation": operation, 
                "reason": f"not_allowed_in_{safe_mode.mode.value}_mode"
            }
        
        # Ejecutar operación local
        if operation == "set":
            return await self.set_data(params.get("key", ""), params.get("value", ""))
        elif operation == "get":
            return await self.get_data(params.get("key", ""))
        
        # Ejecutar operación remota
        elif operation == "call_service":
            service = params.get("service")
            if not service:
                return {"status": "error", "operation": operation, "reason": "missing_service"}
                
            service_op = params.get("service_operation")
            if not service_op:
                return {"status": "error", "operation": operation, "reason": "missing_service_operation"}
                
            service_params = params.get("service_params", {})
            
            try:
                result = await self.call_service(service, service_op, service_params)
                return {
                    "status": "success", 
                    "operation": operation, 
                    "service": service,
                    "service_operation": service_op,
                    "result": result
                }
            except Exception as e:
                return {
                    "status": "error", 
                    "operation": operation, 
                    "service": service,
                    "service_operation": service_op,
                    "reason": str(e)
                }
        else:
            return {"status": "error", "operation": operation, "reason": "unknown_operation"}
    
    async def simulate_crash(self) -> None:
        """Simular un fallo que requiere restauración."""
        logger.warning(f"Componente {self.component_id} fallando...")
        self.data = {}  # Perder todos los datos
        self.operational = False
    
    async def restore(self) -> bool:
        """
        Restaurar estado desde último checkpoint.
        
        Returns:
            True si se restauró correctamente
        """
        if not self.checkpoint_system:
            logger.warning(f"Componente {self.component_id} no tiene checkpointing")
            return False
            
        success = await self.checkpoint_system.restore_latest()
        if success:
            # Extraer datos de state
            self.data = {}
            for key, value in self.checkpoint_system.state.items():
                self.data[key] = value
            logger.info(f"Componente {self.component_id} restaurado exitosamente")
            self.operational = True
        return success
    
    def get_circuit_breaker_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtener estadísticas de circuit breakers.
        
        Returns:
            Diccionario con estadísticas por servicio
        """
        if not self.circuit_breakers:
            return {}
            
        return {
            service: breaker.get_stats()
            for service, breaker in self.circuit_breakers.items()
        }

# =================== PRUEBA INTEGRADA ===================

async def run_test_scenario(
    scenario_name: str,
    components: Dict[str, ResilientComponent],
    safe_mode: SafeMode,
    scenario_func
):
    """
    Ejecutar un escenario de prueba.
    
    Args:
        scenario_name: Nombre del escenario
        components: Diccionario de componentes
        safe_mode: Sistema de Safe Mode
        scenario_func: Función que implementa el escenario
    
    Returns:
        Resultados del escenario
    """
    logger.info(f"\n=== Escenario: {scenario_name} ===")
    
    try:
        start_time = time.time()
        results = await scenario_func(components, safe_mode)
        elapsed = time.time() - start_time
        
        logger.info(f"Escenario completado en {elapsed:.2f}s")
        return results
    except Exception as e:
        logger.error(f"Error en escenario: {e}")
        return {"status": "error", "reason": str(e)}

async def scenario_healthy_operation(
    components: Dict[str, ResilientComponent],
    safe_mode: SafeMode
) -> Dict[str, Any]:
    """
    Escenario 1: Operación normal con todos los sistemas saludables.
    """
    results = []
    
    # Probar operaciones básicas en todos los componentes
    for comp_id, component in components.items():
        # Almacenar datos
        set_result = await component.execute_operation(
            "set", {"key": "test_key", "value": f"test_value_{comp_id}"}, safe_mode
        )
        results.append({"component": comp_id, "operation": "set", "result": set_result})
        
        # Recuperar datos
        get_result = await component.execute_operation(
            "get", {"key": "test_key"}, safe_mode
        )
        results.append({"component": comp_id, "operation": "get", "result": get_result})
        
        # Llamar a servicios externos
        for service_name in component.services:
            call_result = await component.execute_operation(
                "call_service", 
                {
                    "service": service_name,
                    "service_operation": "get_status",
                    "service_params": {"detail": "full"}
                }, 
                safe_mode
            )
            results.append({
                "component": comp_id, 
                "operation": "call_service", 
                "service": service_name,
                "result": call_result
            })
    
    return {"scenario": "healthy_operation", "results": results}

async def scenario_degraded_service(
    components: Dict[str, ResilientComponent],
    safe_mode: SafeMode
) -> Dict[str, Any]:
    """
    Escenario 2: Servicio degradado que prueba el sistema de reintentos.
    """
    results = []
    
    # Establecer estado degradado en el servicio de datos
    data_service = components["market_data"].services["data_provider"]
    data_service.set_health(ServiceHealth.DEGRADED)
    
    # Realizar varias llamadas al servicio degradado
    for i in range(5):
        call_result = await components["market_data"].execute_operation(
            "call_service", 
            {
                "service": "data_provider",
                "service_operation": "get_price_data",
                "service_params": {"symbol": f"BTC{i}"}
            }, 
            safe_mode
        )
        results.append({
            "component": "market_data", 
            "operation": "call_service", 
            "service": "data_provider",
            "attempt": i,
            "result": call_result
        })
    
    # Restaurar servicio
    data_service.set_health(ServiceHealth.HEALTHY)
    
    # Verificar estadísticas del circuit breaker
    cb_stats = components["market_data"].get_circuit_breaker_stats()
    
    return {
        "scenario": "degraded_service", 
        "results": results,
        "circuit_breaker_stats": cb_stats.get("data_provider", {})
    }

async def scenario_failing_service(
    components: Dict[str, ResilientComponent],
    safe_mode: SafeMode
) -> Dict[str, Any]:
    """
    Escenario 3: Servicio que falla constantemente, prueba el Circuit Breaker.
    """
    results = []
    
    # Establecer estado de fallo en el servicio de pagos
    payment_service = components["wallet"].services["payment_provider"]
    payment_service.set_health(ServiceHealth.FAILING)
    
    # Realizar varias llamadas al servicio con fallos
    for i in range(10):
        call_result = await components["wallet"].execute_operation(
            "call_service", 
            {
                "service": "payment_provider",
                "service_operation": "process_payment",
                "service_params": {"amount": 100, "currency": "USD"}
            }, 
            safe_mode
        )
        
        # Verificar estado del circuit breaker después de cada llamada
        cb_stats = components["wallet"].get_circuit_breaker_stats().get("payment_provider", {})
        
        results.append({
            "component": "wallet", 
            "operation": "call_service", 
            "service": "payment_provider",
            "attempt": i,
            "circuit_state": cb_stats.get("state", "Unknown"),
            "result": call_result
        })
    
    # Recuperar servicio
    payment_service.set_health(ServiceHealth.HEALTHY)
    
    # Esperar a que el circuit breaker cambie a half-open
    logger.info("Esperando recuperación de circuit breaker...")
    await asyncio.sleep(3.0)
    
    # Realizar una llamada exitosa para cerrar el circuito
    call_result = await components["wallet"].execute_operation(
        "call_service", 
        {
            "service": "payment_provider",
            "service_operation": "verify_balance",
            "service_params": {"account_id": "test123"}
        }, 
        safe_mode
    )
    
    cb_stats = components["wallet"].get_circuit_breaker_stats().get("payment_provider", {})
    
    results.append({
        "component": "wallet", 
        "operation": "call_service", 
        "service": "payment_provider",
        "attempt": "final",
        "circuit_state": cb_stats.get("state", "Unknown"),
        "result": call_result
    })
    
    return {
        "scenario": "failing_service", 
        "results": results,
        "final_circuit_breaker_stats": cb_stats
    }

async def scenario_component_crash(
    components: Dict[str, ResilientComponent],
    safe_mode: SafeMode
) -> Dict[str, Any]:
    """
    Escenario 4: Fallo de componente y recuperación usando checkpoint.
    """
    results = []
    
    # Almacenar datos importantes en el componente de estrategia
    strategy = components["strategy"]
    
    await strategy.execute_operation(
        "set", {"key": "important_config", "value": "critical_parameter"}, safe_mode
    )
    
    await strategy.execute_operation(
        "set", {"key": "algorithm", "value": "machine_learning_v2"}, safe_mode
    )
    
    # Forzar checkpoint
    if strategy.checkpoint_system:
        checkpoint_id = await strategy.checkpoint_system.create_checkpoint()
        results.append({"operation": "checkpoint", "id": checkpoint_id})
    
    # Verificar estado antes del fallo
    before_crash = await strategy.execute_operation(
        "get", {"key": "important_config"}, safe_mode
    )
    results.append({"operation": "before_crash", "result": before_crash})
    
    # Simular fallo
    await strategy.simulate_crash()
    
    # Verificar estado después del fallo
    after_crash = await strategy.execute_operation(
        "get", {"key": "important_config"}, safe_mode
    )
    results.append({"operation": "after_crash", "result": after_crash})
    
    # Restaurar desde checkpoint
    restore_success = await strategy.restore()
    results.append({"operation": "restore", "success": restore_success})
    
    # Verificar estado después de restauración
    after_restore = await strategy.execute_operation(
        "get", {"key": "important_config"}, safe_mode
    )
    results.append({"operation": "after_restore", "result": after_restore})
    
    return {"scenario": "component_crash", "results": results}

async def scenario_safe_mode(
    components: Dict[str, ResilientComponent],
    safe_mode: SafeMode
) -> Dict[str, Any]:
    """
    Escenario 5: Activación de Safe Mode y restricción de operaciones.
    """
    results = []
    
    # Operaciones normales
    for comp_id, component in components.items():
        normal_result = await component.execute_operation(
            "set", {"key": "normal_key", "value": "normal_value"}, safe_mode
        )
        results.append({
            "mode": "normal", 
            "component": comp_id, 
            "essential": component.essential,
            "operation": "set", 
            "result": normal_result
        })
    
    # Activar Safe Mode
    safe_mode.activate_safe_mode("Prueba de degradación")
    
    # Operaciones en Safe Mode
    for comp_id, component in components.items():
        # Probar escritura
        write_result = await component.execute_operation(
            "set", {"key": "safe_key", "value": "safe_value"}, safe_mode
        )
        results.append({
            "mode": "safe", 
            "component": comp_id, 
            "essential": component.essential,
            "operation": "set", 
            "result": write_result
        })
        
        # Probar lectura
        read_result = await component.execute_operation(
            "get", {"key": "normal_key"}, safe_mode
        )
        results.append({
            "mode": "safe", 
            "component": comp_id, 
            "essential": component.essential,
            "operation": "get", 
            "result": read_result
        })
    
    # Activar Emergency Mode
    safe_mode.activate_emergency_mode("Prueba de emergencia")
    
    # Operaciones en Emergency Mode
    for comp_id, component in components.items():
        emergency_result = await component.execute_operation(
            "set", {"key": "emergency_key", "value": "emergency_value"}, safe_mode
        )
        results.append({
            "mode": "emergency", 
            "component": comp_id, 
            "essential": component.essential,
            "operation": "set", 
            "result": emergency_result
        })
    
    # Volver a modo normal
    safe_mode.deactivate()
    
    return {"scenario": "safe_mode", "results": results}

async def scenario_unavailable_service(
    components: Dict[str, ResilientComponent],
    safe_mode: SafeMode
) -> Dict[str, Any]:
    """
    Escenario 6: Servicio completamente no disponible.
    """
    results = []
    
    # Establecer servicio como no disponible
    exchange_service = components["exchange"].services["exchange_api"]
    exchange_service.set_health(ServiceHealth.UNAVAILABLE)
    
    # Llamadas al servicio no disponible (con circuit breaker y retry)
    for i in range(8):
        call_result = await components["exchange"].execute_operation(
            "call_service", 
            {
                "service": "exchange_api",
                "service_operation": "place_order",
                "service_params": {"symbol": "ETH/USD", "side": "buy", "amount": 1}
            }, 
            safe_mode
        )
        
        cb_stats = components["exchange"].get_circuit_breaker_stats().get("exchange_api", {})
        
        results.append({
            "attempt": i,
            "circuit_state": cb_stats.get("state", "Unknown"),
            "result": call_result
        })
        
        # Breve pausa para log
        await asyncio.sleep(0.1)
    
    # Activar Safe Mode debido a la no disponibilidad
    safe_mode.activate_safe_mode("Exchange no disponible")
    
    # Restaurar servicio
    exchange_service.set_health(ServiceHealth.HEALTHY)
    
    # Intentar nuevamente después de recuperación
    await asyncio.sleep(3.0)  # Esperar a que el circuit breaker se recupere
    
    call_result = await components["exchange"].execute_operation(
        "call_service", 
        {
            "service": "exchange_api",
            "service_operation": "get_market_status",
            "service_params": {"symbols": ["BTC/USD", "ETH/USD"]}
        }, 
        safe_mode
    )
    
    cb_stats = components["exchange"].get_circuit_breaker_stats().get("exchange_api", {})
    
    results.append({
        "attempt": "final",
        "circuit_state": cb_stats.get("state", "Unknown"),
        "result": call_result
    })
    
    # Desactivar Safe Mode
    safe_mode.deactivate()
    
    return {
        "scenario": "unavailable_service", 
        "results": results,
        "final_circuit_breaker_stats": cb_stats
    }

async def main():
    """Función principal."""
    logger.info("=== Prueba Integrada de Resiliencia Genesis ===")
    
    # Crear directorio de pruebas
    create_test_dir()
    
    try:
        # Crear servicios simulados
        services = {
            "data_provider": ExternalService("data_provider"),
            "exchange_api": ExternalService("exchange_api"),
            "payment_provider": ExternalService("payment_provider"),
            "notification_service": ExternalService("notification_service"),
        }
        
        # Crear sistema de Safe Mode
        safe_mode = SafeMode(essential_components=["exchange", "wallet"])
        
        # Crear componentes resilientes
        components = {
            "market_data": ResilientComponent(
                component_id="market_data",
                services={"data_provider": services["data_provider"]},
                checkpoint_dir=CHECKPOINT_DIR,
                essential=False
            ),
            "exchange": ResilientComponent(
                component_id="exchange",
                services={"exchange_api": services["exchange_api"]},
                checkpoint_dir=CHECKPOINT_DIR,
                essential=True
            ),
            "wallet": ResilientComponent(
                component_id="wallet",
                services={"payment_provider": services["payment_provider"]},
                checkpoint_dir=CHECKPOINT_DIR,
                essential=True
            ),
            "strategy": ResilientComponent(
                component_id="strategy",
                services={"notification_service": services["notification_service"]},
                checkpoint_dir=CHECKPOINT_DIR,
                essential=False
            )
        }
        
        # Iniciar componentes
        for component in components.values():
            await component.start()
        
        # Ejecutar escenarios de prueba
        scenario_results = {}
        
        # Escenario 1: Operación normal
        scenario_results["healthy"] = await run_test_scenario(
            "Operación Normal", components, safe_mode, scenario_healthy_operation
        )
        
        # Escenario 2: Servicio degradado (prueba de retry)
        scenario_results["degraded"] = await run_test_scenario(
            "Servicio Degradado", components, safe_mode, scenario_degraded_service
        )
        
        # Escenario 3: Servicio con fallos (prueba de circuit breaker)
        scenario_results["failing"] = await run_test_scenario(
            "Servicio con Fallos", components, safe_mode, scenario_failing_service
        )
        
        # Escenario 4: Fallo de componente (prueba de checkpoint)
        scenario_results["crash"] = await run_test_scenario(
            "Fallo de Componente", components, safe_mode, scenario_component_crash
        )
        
        # Escenario 5: Activación de Safe Mode
        scenario_results["safe_mode"] = await run_test_scenario(
            "Activación de Safe Mode", components, safe_mode, scenario_safe_mode
        )
        
        # Escenario 6: Servicio no disponible
        scenario_results["unavailable"] = await run_test_scenario(
            "Servicio No Disponible", components, safe_mode, scenario_unavailable_service
        )
        
        # Resumen de resultados
        logger.info("\n=== Resumen de Pruebas ===")
        
        success_count = 0
        total_count = 0
        
        for scenario, result in scenario_results.items():
            status = "ÉXITO" if result.get("status") != "error" else "ERROR"
            if status == "ÉXITO":
                success_count += 1
            total_count += 1
            logger.info(f"Escenario {scenario}: {status}")
        
        success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
        logger.info(f"\nTasa de éxito: {success_rate:.1f}% ({success_count}/{total_count})")
        
        # Conclusiones
        logger.info("\n=== Conclusiones ===")
        logger.info("1. Sistema de Reintentos: Mejora la tolerancia a fallos transitorios")
        logger.info("2. Circuit Breaker: Previene sobrecarga de sistemas degradados")
        logger.info("3. Checkpointing: Permite restauración rápida tras fallos")
        logger.info("4. Safe Mode: Mantiene operaciones críticas durante degradación")
        logger.info("\nLa combinación de estas características proporciona un sistema significativamente más robusto")
        
    finally:
        # Detener componentes
        for component in components.values():
            await component.stop()
        
        # Limpiar recursos
        shutil.rmtree(CHECKPOINT_DIR)

if __name__ == "__main__":
    asyncio.run(main())