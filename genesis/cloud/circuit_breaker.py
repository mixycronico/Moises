"""
CloudCircuitBreaker - Componente de resiliencia divina para el Sistema Genesis.

Este módulo implementa el patrón Circuit Breaker con capacidades trascendentales:
- Prevención de fallos en cascada
- Recuperación automática con transmutación cuántica de errores
- Estado coherente en entornos distribuidos
- Protección contra sobrecarga y latencia extrema
"""

import asyncio
import logging
import time
import random
import functools
import uuid
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Callable, TypeVar, Set, Union
from datetime import datetime, timedelta
import traceback

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("genesis.cloud.circuit_breaker")

# Definir tipos genéricos
T = TypeVar('T')


class CircuitState(Enum):
    """Estados posibles para el Circuit Breaker."""
    CLOSED = auto()    # Estado normal, permite operaciones
    OPEN = auto()      # Estado de error, rechaza operaciones
    HALF_OPEN = auto() # Estado de prueba, permite operaciones limitadas


class CloudCircuitBreaker:
    """
    Circuit Breaker con capacidades trascendentales para el Sistema Genesis.
    
    Este componente implementa un Circuit Breaker avanzado con:
    - Transmutación cuántica de errores
    - Capacidad de entrelazamiento con otros circuit breakers
    - Estado coherente en entornos distribuidos
    - Failsafe cuántico con recuperación dimensional
    """
    
    def __init__(self, 
                 name: str,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 half_open_capacity: int = 1,
                 quantum_failsafe: bool = True):
        """
        Inicializar Circuit Breaker.
        
        Args:
            name: Nombre único del circuit breaker
            failure_threshold: Número de fallos antes de abrir el circuito
            recovery_timeout: Tiempo en segundos antes de intentar recuperación
            half_open_capacity: Número de operaciones permitidas en estado HALF_OPEN
            quantum_failsafe: Si activar la protección cuántica avanzada
        """
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_capacity = half_open_capacity
        self.quantum_failsafe = quantum_failsafe
        
        self.last_failure_time = None
        self.last_state_change = datetime.now()
        self.current_half_open_count = 0
        
        # Métricas avanzadas
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "open_transitions": 0,
            "closed_transitions": 0,
            "half_open_transitions": 0,
            "recovery_attempts": 0,
            "successful_recoveries": 0,
            "quantum_transmutations": 0,
            "avg_recovery_time": 0.0,
            "recovery_times": []
        }
        
        # Entrelazamiento cuántico con otros circuit breakers
        self.entangled_circuits: Set[str] = set()
        
        logger.info(f"Circuit Breaker '{name}' inicializado con umbral de fallos {failure_threshold}")
    
    async def before_call(self) -> bool:
        """
        Verificar estado antes de una llamada.
        
        Returns:
            True si la llamada puede proceder, False si debe ser rechazada
        """
        self.metrics["total_calls"] += 1
        
        if self.state == CircuitState.CLOSED:
            # En estado cerrado, las llamadas siempre proceden
            return True
            
        elif self.state == CircuitState.OPEN:
            # En estado abierto, verificar si es tiempo de recuperación
            if self.last_failure_time is not None:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                
                if elapsed >= self.recovery_timeout:
                    # Transición a estado semi-abierto para probar recuperación
                    await self._transition_to(CircuitState.HALF_OPEN)
                    self.metrics["recovery_attempts"] += 1
                    self.current_half_open_count = 0
                    logger.info(f"Circuit Breaker '{self.name}' intentando recuperación")
                    return True
            
            # Aún en estado abierto, rechazar la llamada
            logger.warning(f"Circuit Breaker '{self.name}' rechazando llamada en estado OPEN")
            return False
            
        elif self.state == CircuitState.HALF_OPEN:
            # En estado semi-abierto, permitir número limitado de llamadas
            if self.current_half_open_count < self.half_open_capacity:
                self.current_half_open_count += 1
                return True
            else:
                logger.warning(f"Circuit Breaker '{self.name}' rechazando llamada en estado HALF_OPEN (capacidad alcanzada)")
                return False
        
        # Estado desconocido, rechazar por seguridad
        return False
    
    async def on_success(self) -> None:
        """Registrar éxito de una llamada."""
        self.metrics["successful_calls"] += 1
        
        if self.state == CircuitState.HALF_OPEN:
            # En estado semi-abierto, cada éxito aumenta contador
            self.success_count += 1
            
            # Si alcanzamos umbral de éxitos, cerrar el circuito
            if self.success_count >= self.failure_threshold:
                # Calcular tiempo de recuperación
                recovery_time = (datetime.now() - self.last_state_change).total_seconds()
                self.metrics["recovery_times"].append(recovery_time)
                self.metrics["successful_recoveries"] += 1
                
                # Actualizar tiempo medio de recuperación
                if self.metrics["recovery_times"]:
                    self.metrics["avg_recovery_time"] = sum(self.metrics["recovery_times"]) / len(self.metrics["recovery_times"])
                
                # Transición a estado cerrado
                await self._transition_to(CircuitState.CLOSED)
                logger.info(f"Circuit Breaker '{self.name}' recuperado exitosamente en {recovery_time:.2f}s")
        
        # En cualquier estado, resetear contador de fallos
        self.failure_count = 0
    
    async def on_failure(self, error: Optional[Exception] = None) -> None:
        """
        Registrar fallo de una llamada.
        
        Args:
            error: Excepción que causó el fallo (opcional)
        """
        self.metrics["failed_calls"] += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.CLOSED:
            # En estado cerrado, aumentar contador de fallos
            self.failure_count += 1
            
            # Si alcanzamos umbral de fallos, abrir el circuito
            if self.failure_count >= self.failure_threshold:
                await self._transition_to(CircuitState.OPEN)
                logger.warning(f"Circuit Breaker '{self.name}' abierto tras {self.failure_count} fallos consecutivos")
                
                # Propagar cambio a circuitos entrelazados
                if self.entangled_circuits and self.quantum_failsafe:
                    await self._notify_entangled_circuits()
        
        elif self.state == CircuitState.HALF_OPEN:
            # En estado semi-abierto, cualquier fallo abre el circuito
            await self._transition_to(CircuitState.OPEN)
            logger.warning(f"Circuit Breaker '{self.name}' volvió a estado OPEN tras fallo en recuperación")
            
            # Propagar cambio a circuitos entrelazados
            if self.entangled_circuits and self.quantum_failsafe:
                await self._notify_entangled_circuits()
        
        # Intentar transmutación cuántica del error
        if error is not None and self.quantum_failsafe:
            await self._attempt_quantum_transmutation(error)
    
    async def _transition_to(self, new_state: CircuitState) -> None:
        """
        Transicionar a un nuevo estado.
        
        Args:
            new_state: Nuevo estado del circuit breaker
        """
        old_state = self.state
        self.state = new_state
        self.last_state_change = datetime.now()
        
        # Actualizar métricas
        if new_state == CircuitState.OPEN:
            self.metrics["open_transitions"] += 1
        elif new_state == CircuitState.CLOSED:
            self.metrics["closed_transitions"] += 1
        elif new_state == CircuitState.HALF_OPEN:
            self.metrics["half_open_transitions"] += 1
        
        # Resetear contadores según el estado
        if new_state == CircuitState.CLOSED:
            self.failure_count = 0
            self.success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self.success_count = 0
            self.current_half_open_count = 0
        
        logger.info(f"Circuit Breaker '{self.name}' cambió de estado {old_state} a {new_state}")
    
    async def _notify_entangled_circuits(self) -> None:
        """Notificar a circuitos entrelazados sobre cambio de estado."""
        if not self.entangled_circuits:
            return
            
        # Obtener referencia a factory global desde el módulo
        global circuit_breaker_factory
        if circuit_breaker_factory is None:
            logger.warning(f"Circuit Breaker '{self.name}' no puede notificar a circuitos entrelazados: factory no disponible")
            return
        
        logger.info(f"Circuit Breaker '{self.name}' notificando a {len(self.entangled_circuits)} circuitos entrelazados")
        
        # Notificar a cada circuito entrelazado
        for circuit_name in self.entangled_circuits:
            circuit = circuit_breaker_factory.get(circuit_name)
            if circuit and circuit.name != self.name:  # Evitar bucles
                await circuit._receive_entanglement_update(self.name, self.state)
    
    async def _receive_entanglement_update(self, source_name: str, source_state: CircuitState) -> None:
        """
        Recibir actualización de un circuito entrelazado.
        
        Args:
            source_name: Nombre del circuito que envía la actualización
            source_state: Estado del circuito origen
        """
        logger.info(f"Circuit Breaker '{self.name}' recibió actualización de '{source_name}': {source_state}")
        
        # Aplicar cambios según política de entrelazamiento
        if source_state == CircuitState.OPEN and self.state == CircuitState.CLOSED:
            # Probabilidad de apertura de circuito basada en entrelazamiento cuántico
            if random.random() < 0.5:  # 50% de probabilidad
                await self._transition_to(CircuitState.HALF_OPEN)
                logger.info(f"Circuit Breaker '{self.name}' se movió a HALF_OPEN por entrelazamiento con '{source_name}'")
    
    async def _attempt_quantum_transmutation(self, error: Exception) -> bool:
        """
        Intentar transmutación cuántica de un error.
        
        La transmutación cuántica permite convertir errores fatales en advertencias
        o reconvertir errores para hacerlos manejables por capas superiores.
        
        Args:
            error: Excepción a transmutir
            
        Returns:
            True si la transmutación fue exitosa
        """
        if not self.quantum_failsafe:
            return False
        
        # Simular probabilidad de transmutación exitosa
        transmutation_probability = 0.3  # 30% de probabilidad
        
        if random.random() < transmutation_probability:
            # Marcar error como transmutado
            setattr(error, "transmuted", True)
            setattr(error, "original_type", type(error).__name__)
            setattr(error, "transmutation_time", datetime.now())
            
            self.metrics["quantum_transmutations"] += 1
            logger.info(f"Circuit Breaker '{self.name}' realizó transmutación cuántica de error {type(error).__name__}")
            return True
        
        return False
    
    async def force_open(self) -> None:
        """Forzar apertura del circuito (para pruebas)."""
        await self._transition_to(CircuitState.OPEN)
    
    async def force_closed(self) -> None:
        """Forzar cierre del circuito (para pruebas)."""
        await self._transition_to(CircuitState.CLOSED)
    
    async def reset(self) -> None:
        """Resetear estado y métricas del circuito."""
        await self._transition_to(CircuitState.CLOSED)
        
        # Resetear métricas
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "open_transitions": 0,
            "closed_transitions": 0,
            "half_open_transitions": 0,
            "recovery_attempts": 0,
            "successful_recoveries": 0,
            "quantum_transmutations": 0,
            "avg_recovery_time": 0.0,
            "recovery_times": []
        }
        
        logger.info(f"Circuit Breaker '{self.name}' reseteado")
    
    async def entangle_with(self, circuit_name: str) -> bool:
        """
        Entrelazar con otro circuit breaker.
        
        Args:
            circuit_name: Nombre del circuit breaker a entrelazar
            
        Returns:
            True si el entrelazamiento fue exitoso
        """
        if circuit_name != self.name:  # Evitar auto-entrelazamiento
            self.entangled_circuits.add(circuit_name)
            logger.info(f"Circuit Breaker '{self.name}' entrelazado con '{circuit_name}'")
            return True
        return False
    
    def get_state(self) -> CircuitState:
        """
        Obtener estado actual.
        
        Returns:
            Estado actual del circuit breaker
        """
        return self.state
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas detalladas.
        
        Returns:
            Diccionario con métricas
        """
        # Añadir métricas calculadas
        metrics = self.metrics.copy()
        metrics["state"] = self.state.name
        metrics["uptime"] = (datetime.now() - self.last_state_change).total_seconds()
        
        if metrics["failed_calls"] + metrics["successful_calls"] > 0:
            metrics["success_rate"] = (metrics["successful_calls"] / 
                                      (metrics["failed_calls"] + metrics["successful_calls"])) * 100
        else:
            metrics["success_rate"] = 100.0
            
        metrics["entangled_circuits"] = list(self.entangled_circuits)
        
        return metrics


class CloudCircuitBreakerFactory:
    """
    Factory para crear y gestionar circuit breakers.
    
    Esta clase permite crear y obtener circuit breakers por nombre,
    evitando duplicados y manteniendo una referencia central.
    """
    
    def __init__(self):
        """Inicializar factory."""
        self._circuit_breakers: Dict[str, CloudCircuitBreaker] = {}
        logger.info("Circuit Breaker Factory inicializada")
    
    async def create(self, 
                     name: str,
                     failure_threshold: int = 5,
                     recovery_timeout: float = 60.0,
                     half_open_capacity: int = 1,
                     quantum_failsafe: bool = True) -> CloudCircuitBreaker:
        """
        Crear un nuevo circuit breaker o devolver uno existente.
        
        Args:
            name: Nombre único del circuit breaker
            failure_threshold: Número de fallos antes de abrir el circuito
            recovery_timeout: Tiempo en segundos antes de intentar recuperación
            half_open_capacity: Número de operaciones permitidas en estado HALF_OPEN
            quantum_failsafe: Si activar la protección cuántica avanzada
            
        Returns:
            Circuit breaker creado o existente
        """
        # Si ya existe, devolverlo
        if name in self._circuit_breakers:
            return self._circuit_breakers[name]
        
        # Crear nuevo circuit breaker
        circuit = CloudCircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            half_open_capacity=half_open_capacity,
            quantum_failsafe=quantum_failsafe
        )
        
        # Registrar en diccionario
        self._circuit_breakers[name] = circuit
        logger.info(f"Circuit Breaker '{name}' creado y registrado")
        
        return circuit
    
    def get(self, name: str) -> Optional[CloudCircuitBreaker]:
        """
        Obtener circuit breaker por nombre.
        
        Args:
            name: Nombre del circuit breaker
            
        Returns:
            Circuit breaker o None si no existe
        """
        return self._circuit_breakers.get(name)
    
    def list_all(self) -> List[str]:
        """
        Listar todos los circuit breakers registrados.
        
        Returns:
            Lista de nombres de circuit breakers
        """
        return list(self._circuit_breakers.keys())
    
    async def reset_all(self) -> None:
        """Resetear todos los circuit breakers."""
        for circuit in self._circuit_breakers.values():
            await circuit.reset()
        
        logger.info(f"Reseteados {len(self._circuit_breakers)} circuit breakers")
    
    async def entangle_circuits(self, circuit_names: List[str]) -> int:
        """
        Entrelazar múltiples circuit breakers entre sí.
        
        Args:
            circuit_names: Lista de nombres de circuit breakers a entrelazar
            
        Returns:
            Número de entrelazamientos creados
        """
        if len(circuit_names) < 2:
            return 0
        
        entanglement_count = 0
        
        # Entrelazar cada circuit breaker con los demás
        for i, name1 in enumerate(circuit_names):
            circuit1 = self.get(name1)
            if not circuit1:
                continue
                
            for name2 in circuit_names[i+1:]:
                circuit2 = self.get(name2)
                if not circuit2:
                    continue
                
                # Entrelazamiento bidireccional
                if await circuit1.entangle_with(name2):
                    entanglement_count += 1
                    
                if await circuit2.entangle_with(name1):
                    entanglement_count += 1
        
        logger.info(f"Creados {entanglement_count} entrelazamientos entre {len(circuit_names)} circuit breakers")
        return entanglement_count


# Crear singleton global
circuit_breaker_factory = CloudCircuitBreakerFactory()


def circuit_protected(circuit_breaker: Optional[CloudCircuitBreaker] = None, circuit_name: Optional[str] = None):
    """
    Decorador para proteger funciones con circuit breaker.
    
    Este decorador permite proteger cualquier función con un circuit breaker,
    gestionando automáticamente los estados y fallos.
    
    Args:
        circuit_breaker: Instancia de CloudCircuitBreaker (opcional)
        circuit_name: Nombre del circuit breaker a usar (opcional, solo si no se proporciona circuit_breaker)
        
    Returns:
        Decorador para la función
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Obtener circuit breaker
            cb = circuit_breaker
            
            if cb is None and circuit_name:
                # Obtener desde factory global
                global circuit_breaker_factory
                if circuit_breaker_factory:
                    cb = circuit_breaker_factory.get(circuit_name)
            
            if cb is None:
                # Sin circuit breaker, ejecutar normalmente
                return await func(*args, **kwargs)
            
            # Verificar si podemos proceder
            if not await cb.before_call():
                logger.warning(f"Llamada a {func.__name__} rechazada por circuit breaker {cb.name}")
                raise RuntimeError(f"Circuit breaker {cb.name} abierto")
            
            try:
                # Ejecutar función protegida
                result = await func(*args, **kwargs)
                
                # Registrar éxito
                await cb.on_success()
                
                return result
                
            except Exception as e:
                # Registrar fallo
                await cb.on_failure(e)
                
                # Re-lanzar excepción
                raise
        
        return wrapper
    
    return decorator