#!/usr/bin/env python3
"""
CloudLoadBalancer Ultra-Divino.

Este módulo implementa un balanceador de carga avanzado adaptado para entornos cloud,
permitiendo distribuir operaciones entre múltiples nodos de forma inteligente
y resiliente, con capacidades de auto-escalado y afinidad de sesión.

Características principales:
- Distribución inteligente basada en carga, capacidad y rendimiento
- Algoritmos múltiples: round-robin, weighted, least-connections, etc.
- Auto-escalado basado en métricas y predicciones
- Manejo de afinidad de sesión y operaciones relacionadas
- Integración con circuit breaker y checkpoints distribuidos
- Monitoreo detallado de métricas
"""

import os
import sys
import json
import logging
import time
import asyncio
import random
import hashlib
import uuid
from enum import Enum, auto
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set, Type, TypeVar

# Importación segura de componentes relacionados
try:
    from .circuit_breaker import (
        CircuitState, CloudCircuitBreaker, circuit_breaker_factory,
        circuit_protected
    )
    from .distributed_checkpoint import (
        checkpoint_manager, checkpoint_state,
        DistributedCheckpointManager
    )
    _HAS_CLOUD_DEPENDENCIES = True
except ImportError:
    _HAS_CLOUD_DEPENDENCIES = False
    logging.warning("Componentes cloud relacionados no disponibles. Funcionalidad limitada.")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("genesis.cloud.load_balancer")


# Tipos genéricos
T = TypeVar('T')


class BalancerAlgorithm(Enum):
    """Algoritmos de balanceo de carga."""
    ROUND_ROBIN = auto()            # Distribución equitativa secuencial
    LEAST_CONNECTIONS = auto()      # Nodo con menos conexiones activas
    WEIGHTED = auto()               # Basado en pesos configurados
    RESPONSE_TIME = auto()          # Basado en tiempo de respuesta
    RESOURCE_BASED = auto()         # Basado en uso de recursos (CPU/memoria)
    PREDICTIVE = auto()             # Basado en predicciones de carga futura
    QUANTUM = auto()                # Modo cuántico para distribución óptima


class ScalingPolicy(Enum):
    """Políticas de escalado automático."""
    NONE = auto()                  # Sin escalado automático
    THRESHOLD = auto()             # Basado en umbrales (e.g., CPU > 80%)
    PREDICTIVE = auto()            # Basado en predicciones de tráfico
    SCHEDULE = auto()              # Basado en horarios pre-configurados
    ADAPTIVE = auto()              # Adaptativo basado en patrones


class BalancerState(Enum):
    """Estados posibles del balanceador."""
    INITIALIZING = auto()          # Inicializando
    ACTIVE = auto()                # Activo y distribuyendo
    SCALING = auto()               # Escalando (añadiendo/quitando nodos)
    DEGRADED = auto()              # Funcionando en modo degradado
    MAINTENANCE = auto()           # En mantenimiento
    CRITICAL = auto()              # Estado crítico (pocos nodos disponibles)
    FAILED = auto()                # Fallido (sin nodos disponibles)


class SessionAffinityMode(Enum):
    """Modos de afinidad de sesión."""
    NONE = auto()                  # Sin afinidad de sesión
    IP_BASED = auto()              # Basado en IP
    COOKIE = auto()                # Basado en cookie
    TOKEN = auto()                 # Basado en token
    HEADER = auto()                # Basado en cabeceras HTTP
    CONSISTENT_HASH = auto()       # Hash consistente


class NodeHealthStatus(Enum):
    """Estados de salud de un nodo."""
    HEALTHY = auto()               # Funcionando correctamente
    DEGRADED = auto()              # Funcionando con problemas
    UNHEALTHY = auto()             # No disponible
    STARTING = auto()              # Iniciando
    STOPPING = auto()              # Deteniendo
    MAINTENANCE = auto()           # En mantenimiento
    UNKNOWN = auto()               # Estado desconocido


class CloudNode:
    """
    Representación de un nodo en el sistema de balanceo de carga.
    
    Un nodo puede ser una instancia, container, función serverless, etc.
    """
    
    def __init__(self, 
                 node_id: str,
                 host: str,
                 port: int,
                 weight: float = 1.0,
                 max_connections: int = 100):
        """
        Inicializar nodo.
        
        Args:
            node_id: Identificador único del nodo
            host: Hostname o IP del nodo
            port: Puerto del nodo
            weight: Peso para algoritmos ponderados (1.0 = peso estándar)
            max_connections: Máximo de conexiones simultáneas
        """
        self.node_id = node_id
        self.host = host
        self.port = port
        self.weight = weight
        self.max_connections = max_connections
        
        # Estado del nodo
        self.health_status = NodeHealthStatus.UNKNOWN
        self.active_connections = 0
        self.total_connections = 0
        self.last_connection_time = 0
        self.last_health_check = 0
        self.consecutive_failures = 0
        
        # Métricas de rendimiento
        self.metrics = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "response_times": [],
            "error_rate": 0.0,
            "throughput": 0.0,
            "last_updated": time.time()
        }
    
    async def health_check(self) -> bool:
        """
        Realizar verificación de salud del nodo.
        
        Returns:
            True si el nodo está saludable
        """
        # Simulación para demo. En un sistema real, se verificaría la salud del nodo
        # mediante una petición HTTP, ping, o similar.
        try:
            # Simular comprobación
            await asyncio.sleep(0.01)
            
            # Probabilidad de fallo basada en la carga
            fail_probability = min(self.active_connections / self.max_connections, 0.8)
            if random.random() < fail_probability * 0.1:  # Reducida para demo
                self.consecutive_failures += 1
                
                if self.consecutive_failures > 3:
                    self.health_status = NodeHealthStatus.UNHEALTHY
                elif self.consecutive_failures > 1:
                    self.health_status = NodeHealthStatus.DEGRADED
                else:
                    self.health_status = NodeHealthStatus.HEALTHY
            else:
                self.consecutive_failures = max(0, self.consecutive_failures - 1)
                self.health_status = NodeHealthStatus.HEALTHY
            
            self.last_health_check = time.time()
            return self.health_status == NodeHealthStatus.HEALTHY
        
        except Exception as e:
            logger.error(f"Error en health check del nodo {self.node_id}: {e}")
            self.consecutive_failures += 1
            self.health_status = NodeHealthStatus.UNHEALTHY
            self.last_health_check = time.time()
            return False
    
    async def update_metrics(self) -> None:
        """Actualizar métricas del nodo."""
        # Simulación para demo. En un sistema real, se obtendrían métricas del nodo
        # mediante API, agente, o similar.
        try:
            # Simular obtención de métricas
            await asyncio.sleep(0.005)
            
            # CPU simulado basado en conexiones
            cpu_load = min(0.1 + (self.active_connections / self.max_connections) * 0.9, 1.0)
            memory_load = min(0.1 + (self.total_connections % 100) / 100 * 0.5, 0.95)
            
            # Fluctuación aleatoria para simular variación
            cpu_load *= random.uniform(0.8, 1.2)
            memory_load *= random.uniform(0.9, 1.1)
            
            # Limitar a rango válido
            cpu_load = max(0.0, min(1.0, cpu_load))
            memory_load = max(0.0, min(1.0, memory_load))
            
            # Actualizar métricas
            self.metrics["cpu_usage"] = cpu_load
            self.metrics["memory_usage"] = memory_load
            
            # Tiempo de respuesta simulado (mejor con menos carga)
            response_time = 0.01 + cpu_load * 0.1 * random.uniform(0.8, 1.2)
            self.metrics["response_times"].append(response_time)
            
            # Mantener solo últimas 100 mediciones
            if len(self.metrics["response_times"]) > 100:
                self.metrics["response_times"] = self.metrics["response_times"][-100:]
            
            # Throughput simulado (ops/segundo)
            self.metrics["throughput"] = (self.active_connections / (response_time + 0.01)) * random.uniform(0.9, 1.1)
            
            # Error rate simulado
            self.metrics["error_rate"] = (self.consecutive_failures / 10) * random.uniform(0.8, 1.2)
            self.metrics["last_updated"] = time.time()
        
        except Exception as e:
            logger.error(f"Error al actualizar métricas del nodo {self.node_id}: {e}")
    
    def get_avg_response_time(self) -> float:
        """
        Obtener tiempo de respuesta promedio.
        
        Returns:
            Tiempo promedio en segundos
        """
        if not self.metrics["response_times"]:
            return 0.0
        return sum(self.metrics["response_times"]) / len(self.metrics["response_times"])
    
    def get_load_factor(self) -> float:
        """
        Obtener factor de carga combinado.
        
        Returns:
            Factor de carga (0-1)
        """
        conn_factor = self.active_connections / self.max_connections if self.max_connections > 0 else 1.0
        return (conn_factor * 0.4 + 
                self.metrics["cpu_usage"] * 0.3 + 
                self.metrics["memory_usage"] * 0.2 + 
                self.metrics["error_rate"] * 0.1)
    
    def is_available(self) -> bool:
        """
        Verificar si el nodo está disponible para nuevas conexiones.
        
        Returns:
            True si está disponible
        """
        return (self.health_status in [NodeHealthStatus.HEALTHY, NodeHealthStatus.DEGRADED] and 
                self.active_connections < self.max_connections)
    
    def connection_started(self) -> None:
        """Registrar inicio de conexión."""
        self.active_connections += 1
        self.total_connections += 1
        self.last_connection_time = time.time()
    
    def connection_finished(self) -> None:
        """Registrar fin de conexión."""
        self.active_connections = max(0, self.active_connections - 1)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir a diccionario para serialización.
        
        Returns:
            Diccionario con los datos del nodo
        """
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "weight": self.weight,
            "max_connections": self.max_connections,
            "health_status": self.health_status.name,
            "active_connections": self.active_connections,
            "total_connections": self.total_connections,
            "last_connection_time": self.last_connection_time,
            "last_health_check": self.last_health_check,
            "consecutive_failures": self.consecutive_failures,
            "metrics": {
                "cpu_usage": self.metrics["cpu_usage"],
                "memory_usage": self.metrics["memory_usage"],
                "avg_response_time": self.get_avg_response_time(),
                "error_rate": self.metrics["error_rate"],
                "throughput": self.metrics["throughput"],
                "last_updated": self.metrics["last_updated"]
            }
        }


class SessionStore:
    """
    Almacén de información de sesión para afinidad.
    
    Permite mantener coherencia en operaciones relacionadas
    asegurando que se dirijan al mismo nodo.
    """
    
    def __init__(self, mode: SessionAffinityMode):
        """
        Inicializar almacén de sesiones.
        
        Args:
            mode: Modo de afinidad de sesión
        """
        self.mode = mode
        self.sessions: Dict[str, str] = {}  # session_id -> node_id
        self.last_access: Dict[str, float] = {}  # session_id -> timestamp
        self.session_timeout = 600  # 10 minutos
    
    def get_node_for_session(self, session_key: str) -> Optional[str]:
        """
        Obtener nodo asignado a una sesión.
        
        Args:
            session_key: Clave de sesión (IP, token, etc.)
            
        Returns:
            ID del nodo o None si no hay asignación
        """
        if session_key not in self.sessions:
            return None
        
        # Actualizar timestamp de último acceso
        self.last_access[session_key] = time.time()
        return self.sessions[session_key]
    
    def assign_session(self, session_key: str, node_id: str) -> None:
        """
        Asignar sesión a un nodo.
        
        Args:
            session_key: Clave de sesión (IP, token, etc.)
            node_id: ID del nodo asignado
        """
        self.sessions[session_key] = node_id
        self.last_access[session_key] = time.time()
    
    def remove_session(self, session_key: str) -> None:
        """
        Eliminar asignación de sesión.
        
        Args:
            session_key: Clave de sesión a eliminar
        """
        if session_key in self.sessions:
            self.sessions.pop(session_key)
            self.last_access.pop(session_key, None)
    
    def cleanup_expired_sessions(self) -> int:
        """
        Limpiar sesiones expiradas.
        
        Returns:
            Número de sesiones eliminadas
        """
        now = time.time()
        expired_keys = []
        
        for key, last_time in self.last_access.items():
            if now - last_time > self.session_timeout:
                expired_keys.append(key)
        
        for key in expired_keys:
            self.remove_session(key)
        
        return len(expired_keys)
    
    def get_session_counts(self) -> Dict[str, int]:
        """
        Obtener número de sesiones por nodo.
        
        Returns:
            Diccionario con conteo de sesiones por nodo
        """
        counts: Dict[str, int] = {}
        
        for node_id in self.sessions.values():
            if node_id not in counts:
                counts[node_id] = 0
            counts[node_id] += 1
        
        return counts


class CloudLoadBalancer:
    """
    Balanceador de carga adaptado para entornos cloud.
    
    Implementa distribución inteligente de operaciones entre múltiples nodos,
    con soporte para diversos algoritmos, auto-escalado y afinidad de sesión.
    """
    
    def __init__(self, 
                 name: str, 
                 algorithm: BalancerAlgorithm = BalancerAlgorithm.ROUND_ROBIN,
                 scaling_policy: ScalingPolicy = ScalingPolicy.NONE,
                 session_affinity: SessionAffinityMode = SessionAffinityMode.NONE,
                 health_check_interval: float = 10.0,
                 metrics_interval: float = 5.0):
        """
        Inicializar balanceador de carga.
        
        Args:
            name: Nombre identificativo del balanceador
            algorithm: Algoritmo de balanceo a utilizar
            scaling_policy: Política de escalado automático
            session_affinity: Modo de afinidad de sesión
            health_check_interval: Intervalo entre verificaciones de salud (segundos)
            metrics_interval: Intervalo entre actualizaciones de métricas (segundos)
        """
        self.name = name
        self.algorithm = algorithm
        self.scaling_policy = scaling_policy
        self.session_affinity = session_affinity
        self.health_check_interval = health_check_interval
        self.metrics_interval = metrics_interval
        
        # Estado del balanceador
        self.state = BalancerState.INITIALIZING
        self.nodes: Dict[str, CloudNode] = {}
        self.healthy_nodes: Set[str] = set()
        self.session_store = SessionStore(session_affinity)
        
        # Para algoritmo round robin
        self._rr_index = 0
        
        # Tareas de fondo
        self._health_check_task = None
        self._metrics_task = None
        self._scaling_task = None
        self._initialized = False
        
        # Scaling settings
        self.scaling_settings = {
            "min_nodes": 1,
            "max_nodes": 10,
            "cpu_threshold_up": 0.7,
            "cpu_threshold_down": 0.3,
            "scale_up_cooldown": 60,  # segundos
            "scale_down_cooldown": 300,  # segundos
            "last_scale_up": 0,
            "last_scale_down": 0
        }
        
        # Métricas globales
        self.metrics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "avg_response_time": 0.0,
            "request_rate": 0.0,
            "error_rate": 0.0,
            "last_error": "",
            "last_updated": time.time()
        }
        
        # Para cálculo de tasa de operaciones
        self._last_total_operations = 0
        
        logger.info(f"CloudLoadBalancer '{name}' inicializado con algoritmo {algorithm.name}")
    
    async def initialize(self) -> bool:
        """
        Inicializar balanceador y tareas de monitoreo.
        
        Returns:
            True si se inicializó correctamente
        """
        if self._initialized:
            return True
        
        try:
            # Iniciar tareas de monitoreo
            self._health_check_task = asyncio.create_task(self._run_health_checks())
            self._metrics_task = asyncio.create_task(self._run_metrics_updates())
            
            # Iniciar tarea de escalado si corresponde
            if self.scaling_policy != ScalingPolicy.NONE:
                self._scaling_task = asyncio.create_task(self._run_scaling_decisions())
            
            self._initialized = True
            self.state = BalancerState.ACTIVE
            
            logger.info(f"Balanceador '{self.name}' inicializado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error al inicializar balanceador '{self.name}': {e}")
            self.state = BalancerState.FAILED
            return False
    
    async def shutdown(self) -> None:
        """Detener todas las tareas y liberar recursos."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._metrics_task:
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass
        
        if self._scaling_task:
            self._scaling_task.cancel()
            try:
                await self._scaling_task
            except asyncio.CancelledError:
                pass
        
        self._initialized = False
        logger.info(f"Balanceador '{self.name}' detenido correctamente")
    
    async def _run_health_checks(self) -> None:
        """Tarea periódica para verificar salud de los nodos."""
        try:
            while True:
                await self._check_all_nodes_health()
                await asyncio.sleep(self.health_check_interval)
        except asyncio.CancelledError:
            logger.info(f"Tarea de health check del balanceador '{self.name}' cancelada")
        except Exception as e:
            logger.error(f"Error en tarea de health check del balanceador '{self.name}': {e}")
    
    async def _run_metrics_updates(self) -> None:
        """Tarea periódica para actualizar métricas de los nodos."""
        try:
            while True:
                await self._update_all_nodes_metrics()
                await asyncio.sleep(self.metrics_interval)
        except asyncio.CancelledError:
            logger.info(f"Tarea de métricas del balanceador '{self.name}' cancelada")
        except Exception as e:
            logger.error(f"Error en tarea de métricas del balanceador '{self.name}': {e}")
    
    async def _run_scaling_decisions(self) -> None:
        """Tarea periódica para decisiones de escalado automático."""
        try:
            while True:
                await self._make_scaling_decisions()
                await asyncio.sleep(30)  # Verificar cada 30 segundos
        except asyncio.CancelledError:
            logger.info(f"Tarea de escalado del balanceador '{self.name}' cancelada")
        except Exception as e:
            logger.error(f"Error en tarea de escalado del balanceador '{self.name}': {e}")
    
    async def _check_all_nodes_health(self) -> None:
        """Verificar salud de todos los nodos."""
        if not self.nodes:
            return
        
        healthy_count = 0
        unhealthy_count = 0
        
        # Limpiar conjunto actual de nodos saludables
        self.healthy_nodes.clear()
        
        # Verificar cada nodo
        for node_id, node in self.nodes.items():
            is_healthy = await node.health_check()
            
            if is_healthy:
                self.healthy_nodes.add(node_id)
                healthy_count += 1
            else:
                unhealthy_count += 1
        
        # Actualizar estado del balanceador según la disponibilidad
        if not self.healthy_nodes:
            self.state = BalancerState.FAILED
            logger.critical(f"Balanceador '{self.name}' en estado FAILED: No hay nodos saludables")
        elif len(self.healthy_nodes) < len(self.nodes) * 0.5:
            self.state = BalancerState.CRITICAL
            logger.warning(f"Balanceador '{self.name}' en estado CRITICAL: Solo {len(self.healthy_nodes)}/{len(self.nodes)} nodos saludables")
        elif len(self.healthy_nodes) < len(self.nodes):
            self.state = BalancerState.DEGRADED
            logger.info(f"Balanceador '{self.name}' en estado DEGRADED: {len(self.healthy_nodes)}/{len(self.nodes)} nodos saludables")
        else:
            if self.state not in [BalancerState.ACTIVE, BalancerState.SCALING, BalancerState.MAINTENANCE]:
                self.state = BalancerState.ACTIVE
                logger.info(f"Balanceador '{self.name}' en estado ACTIVE: Todos los nodos saludables")
        
        logger.debug(f"Health check completado: {healthy_count} saludables, {unhealthy_count} no saludables")
    
    async def _update_all_nodes_metrics(self) -> None:
        """Actualizar métricas de todos los nodos."""
        if not self.nodes:
            return
        
        # Actualizar métricas de cada nodo
        for node in self.nodes.values():
            await node.update_metrics()
        
        # Actualizar métricas globales
        total_operations = self.metrics["total_operations"]
        if total_operations > 0:
            self.metrics["error_rate"] = self.metrics["failed_operations"] / total_operations
        
        # Calcular tiempo de respuesta promedio global
        total_resp_time = 0.0
        resp_count = 0
        
        for node in self.nodes.values():
            if node.metrics["response_times"]:
                total_resp_time += sum(node.metrics["response_times"])
                resp_count += len(node.metrics["response_times"])
        
        if resp_count > 0:
            self.metrics["avg_response_time"] = total_resp_time / resp_count
        
        # Calcular tasa de solicitudes
        now = time.time()
        elapsed = now - self.metrics["last_updated"]
        if elapsed > 0:
            # Cambio en operaciones por segundo
            self.metrics["request_rate"] = (total_operations - self._last_total_operations) / elapsed
            self._last_total_operations = total_operations
        
        self.metrics["last_updated"] = now
    
    async def _make_scaling_decisions(self) -> None:
        """Realizar decisiones de escalado automático."""
        if self.scaling_policy == ScalingPolicy.NONE or not self.nodes:
            return
        
        now = time.time()
        total_nodes = len(self.nodes)
        
        # Verificar si está en período de enfriamiento
        can_scale_up = now - self.scaling_settings["last_scale_up"] >= self.scaling_settings["scale_up_cooldown"]
        can_scale_down = now - self.scaling_settings["last_scale_down"] >= self.scaling_settings["scale_down_cooldown"]
        
        # Políticas basadas en umbrales
        if self.scaling_policy == ScalingPolicy.THRESHOLD:
            # Calcular CPU promedio
            avg_cpu = sum(node.metrics["cpu_usage"] for node in self.nodes.values()) / total_nodes
            
            # Escalar hacia arriba
            if (can_scale_up and 
                avg_cpu >= self.scaling_settings["cpu_threshold_up"] and 
                total_nodes < self.scaling_settings["max_nodes"]):
                
                logger.info(f"Iniciando scale-up basado en CPU ({avg_cpu:.2f} >= {self.scaling_settings['cpu_threshold_up']:.2f})")
                await self._scale_up()
                self.scaling_settings["last_scale_up"] = now
            
            # Escalar hacia abajo
            elif (can_scale_down and 
                 avg_cpu <= self.scaling_settings["cpu_threshold_down"] and 
                 total_nodes > self.scaling_settings["min_nodes"]):
                
                logger.info(f"Iniciando scale-down basado en CPU ({avg_cpu:.2f} <= {self.scaling_settings['cpu_threshold_down']:.2f})")
                await self._scale_down()
                self.scaling_settings["last_scale_down"] = now
        
        # Políticas predictivas (basadas en tendencias)
        elif self.scaling_policy == ScalingPolicy.PREDICTIVE:
            # Analizar tendencia de tasa de solicitudes
            # Aquí se podría implementar un algoritmo predictivo real,
            # pero para simplificar usamos la tasa actual y una tendencia simulada
            
            current_rate = self.metrics["request_rate"]
            trend_factor = random.uniform(0.8, 1.2)  # Simulación de tendencia
            predicted_rate = current_rate * trend_factor
            
            # Calcular nodos necesarios basado en la tasa actual
            target_nodes = max(1, min(self.scaling_settings["max_nodes"], 
                                    int(predicted_rate / 10) + 1))  # 10 ops/s por nodo
            
            if can_scale_up and target_nodes > total_nodes:
                logger.info(f"Iniciando scale-up predictivo ({predicted_rate:.2f} ops/s predicho)")
                await self._scale_up(target_nodes - total_nodes)
                self.scaling_settings["last_scale_up"] = now
            
            elif can_scale_down and target_nodes < total_nodes:
                logger.info(f"Iniciando scale-down predictivo ({predicted_rate:.2f} ops/s predicho)")
                await self._scale_down(total_nodes - target_nodes)
                self.scaling_settings["last_scale_down"] = now
        
        # Políticas basadas en horario
        elif self.scaling_policy == ScalingPolicy.SCHEDULE:
            # Implementación básica basada en hora del día
            hour_of_day = datetime.now().hour
            
            # Escalar para horas pico (9AM-6PM)
            if 9 <= hour_of_day <= 18:
                target_nodes = self.scaling_settings["max_nodes"]
            else:
                target_nodes = self.scaling_settings["min_nodes"]
            
            if can_scale_up and target_nodes > total_nodes:
                logger.info(f"Iniciando scale-up programado (hora: {hour_of_day})")
                await self._scale_up(target_nodes - total_nodes)
                self.scaling_settings["last_scale_up"] = now
            
            elif can_scale_down and target_nodes < total_nodes:
                logger.info(f"Iniciando scale-down programado (hora: {hour_of_day})")
                await self._scale_down(total_nodes - target_nodes)
                self.scaling_settings["last_scale_down"] = now
    
    async def _scale_up(self, count: int = 1) -> bool:
        """
        Escalar añadiendo nuevos nodos.
        
        Args:
            count: Número de nodos a añadir
            
        Returns:
            True si el escalado fue exitoso
        """
        if not self._initialized:
            return False
        
        try:
            # Actualizar estado
            previous_state = self.state
            self.state = BalancerState.SCALING
            
            # Generar nuevos nodos
            for _ in range(count):
                node_id = f"node_{self.name}_{str(uuid.uuid4())[:8]}"
                
                # Simular creación de nodo (en un sistema real se invocaría API cloud)
                port = 8080 + len(self.nodes)
                node = CloudNode(
                    node_id=node_id,
                    host="127.0.0.1",  # Local para demo
                    port=port,
                    weight=1.0,
                    max_connections=100
                )
                
                # Inicialmente en estado starting
                node.health_status = NodeHealthStatus.STARTING
                
                # Añadir al pool
                self.nodes[node_id] = node
                logger.info(f"Nodo {node_id} añadido a '{self.name}' (escalado up)")
                
                # Simular tiempo de inicio
                await asyncio.sleep(0.5)
                
                # Actualizar estado a saludable
                node.health_status = NodeHealthStatus.HEALTHY
                self.healthy_nodes.add(node_id)
            
            # Restaurar estado previo (o ACTIVE si todos los nodos están saludables)
            if len(self.healthy_nodes) == len(self.nodes):
                self.state = BalancerState.ACTIVE
            else:
                self.state = previous_state
            
            logger.info(f"Escalado up completado: {count} nodos añadidos a '{self.name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error en escalado up de '{self.name}': {e}")
            return False
    
    async def _scale_down(self, count: int = 1) -> bool:
        """
        Escalar eliminando nodos.
        
        Args:
            count: Número de nodos a eliminar
            
        Returns:
            True si el escalado fue exitoso
        """
        if not self._initialized or len(self.nodes) <= self.scaling_settings["min_nodes"]:
            return False
        
        try:
            # Actualizar estado
            previous_state = self.state
            self.state = BalancerState.SCALING
            
            # Limitar a la cantidad disponible
            count = min(count, len(self.nodes) - self.scaling_settings["min_nodes"])
            
            # Seleccionar nodos a eliminar (los menos utilizados)
            nodes_by_load = sorted(
                list(self.nodes.items()),
                key=lambda x: x[1].get_load_factor()
            )
            
            # Eliminar nodos seleccionados
            for i in range(count):
                if i >= len(nodes_by_load):
                    break
                
                node_id, node = nodes_by_load[i]
                
                # Marcar como en proceso de apagado
                node.health_status = NodeHealthStatus.STOPPING
                if node_id in self.healthy_nodes:
                    self.healthy_nodes.remove(node_id)
                
                # Simular apagado gradual
                await asyncio.sleep(0.2)
                
                # Eliminar nodo
                self.nodes.pop(node_id)
                logger.info(f"Nodo {node_id} eliminado de '{self.name}' (escalado down)")
            
            # Actualizar sesiones que apuntaban a nodos eliminados
            await self._reassign_orphaned_sessions()
            
            # Restaurar estado previo (o ACTIVE si todos los nodos están saludables)
            if len(self.healthy_nodes) == len(self.nodes):
                self.state = BalancerState.ACTIVE
            else:
                self.state = previous_state
            
            logger.info(f"Escalado down completado: {count} nodos eliminados de '{self.name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error en escalado down de '{self.name}': {e}")
            return False
    
    async def _reassign_orphaned_sessions(self) -> None:
        """Reasignar sesiones huérfanas tras eliminar nodos."""
        if self.session_affinity == SessionAffinityMode.NONE:
            return
        
        # Identificar sesiones que apuntan a nodos eliminados
        orphaned_sessions = []
        
        for session_key, node_id in self.session_store.sessions.items():
            if node_id not in self.nodes:
                orphaned_sessions.append(session_key)
        
        # Reasignar cada sesión huérfana al próximo nodo disponible
        for session_key in orphaned_sessions:
            # Seleccionar un nodo usando el algoritmo actual
            selected_node = await self._select_node(session_key)
            
            if selected_node:
                # Actualizar asignación
                self.session_store.assign_session(session_key, selected_node)
                logger.debug(f"Sesión {session_key} reasignada a nodo {selected_node}")
            else:
                # Si no hay nodos disponibles, eliminar la sesión
                self.session_store.remove_session(session_key)
                logger.debug(f"Sesión {session_key} eliminada (no hay nodos disponibles)")
    
    async def add_node(self, node: CloudNode) -> bool:
        """
        Añadir un nodo al balanceador.
        
        Args:
            node: Nodo a añadir
            
        Returns:
            True si se añadió correctamente
        """
        if node.node_id in self.nodes:
            logger.warning(f"Nodo {node.node_id} ya existe en '{self.name}'")
            return False
        
        self.nodes[node.node_id] = node
        
        # Verificar salud del nuevo nodo
        is_healthy = await node.health_check()
        if is_healthy:
            self.healthy_nodes.add(node.node_id)
        
        logger.info(f"Nodo {node.node_id} añadido a '{self.name}'")
        return True
    
    async def remove_node(self, node_id: str) -> bool:
        """
        Eliminar un nodo del balanceador.
        
        Args:
            node_id: ID del nodo a eliminar
            
        Returns:
            True si se eliminó correctamente
        """
        if node_id not in self.nodes:
            logger.warning(f"Nodo {node_id} no existe en '{self.name}'")
            return False
        
        # Eliminar del registro
        self.nodes.pop(node_id)
        if node_id in self.healthy_nodes:
            self.healthy_nodes.remove(node_id)
        
        # Reasignar sesiones
        await self._reassign_orphaned_sessions()
        
        logger.info(f"Nodo {node_id} eliminado de '{self.name}'")
        return True
    
    async def select_node(self, session_key: Optional[str] = None) -> Optional[CloudNode]:
        """
        Seleccionar un nodo según el algoritmo configurado.
        
        Args:
            session_key: Clave de sesión para afinidad (opcional)
            
        Returns:
            Nodo seleccionado o None si no hay disponibles
        """
        node_id = await self._select_node(session_key)
        return self.nodes.get(node_id) if node_id else None
    
    async def _select_node(self, session_key: Optional[str] = None) -> Optional[str]:
        """
        Implementación interna de selección de nodo.
        
        Args:
            session_key: Clave de sesión para afinidad (opcional)
            
        Returns:
            ID del nodo seleccionado o None si no hay disponibles
        """
        # Si no hay nodos saludables, no podemos seleccionar
        if not self.healthy_nodes:
            return None
        
        # Si tenemos afinidad de sesión y se proporciona clave, verificar asignación existente
        if self.session_affinity != SessionAffinityMode.NONE and session_key:
            assigned_node = self.session_store.get_node_for_session(session_key)
            
            # Si ya hay asignación y el nodo está disponible, usarlo
            if assigned_node and assigned_node in self.healthy_nodes:
                return assigned_node
        
        # Si llegamos aquí, necesitamos seleccionar un nuevo nodo
        
        # Convertir set a lista para indexación
        available_nodes = list(self.healthy_nodes)
        selected_node = None
        
        # Aplicar algoritmo configurado
        if self.algorithm == BalancerAlgorithm.ROUND_ROBIN:
            # Round Robin: distribuir equitativamente en secuencia
            self._rr_index = (self._rr_index + 1) % len(available_nodes)
            selected_node = available_nodes[self._rr_index]
            
        elif self.algorithm == BalancerAlgorithm.LEAST_CONNECTIONS:
            # Least Connections: nodo con menos conexiones activas
            selected_node = min(
                available_nodes,
                key=lambda node_id: self.nodes[node_id].active_connections
            )
            
        elif self.algorithm == BalancerAlgorithm.WEIGHTED:
            # Weighted: distribución basada en pesos configurados
            total_weight = sum(self.nodes[node_id].weight for node_id in available_nodes)
            if total_weight <= 0:
                # Si todos los pesos son 0, usar selección aleatoria
                selected_node = random.choice(available_nodes)
            else:
                # Selección ponderada
                choice = random.uniform(0, total_weight)
                cumulative = 0
                for node_id in available_nodes:
                    cumulative += self.nodes[node_id].weight
                    if choice <= cumulative:
                        selected_node = node_id
                        break
            
        elif self.algorithm == BalancerAlgorithm.RESPONSE_TIME:
            # Response Time: nodo con mejor tiempo de respuesta
            selected_node = min(
                available_nodes,
                key=lambda node_id: self.nodes[node_id].get_avg_response_time()
            )
            
        elif self.algorithm == BalancerAlgorithm.RESOURCE_BASED:
            # Resource Based: nodo con menor carga de recursos
            selected_node = min(
                available_nodes,
                key=lambda node_id: self.nodes[node_id].get_load_factor()
            )
            
        elif self.algorithm == BalancerAlgorithm.PREDICTIVE:
            # Predictive: similar a Resource Based pero con predicción
            # Para simplificar, usamos un factor aleatorio para simular predicción
            selected_node = min(
                available_nodes,
                key=lambda node_id: self.nodes[node_id].get_load_factor() * random.uniform(0.8, 1.2)
            )
            
        elif self.algorithm == BalancerAlgorithm.QUANTUM:
            # Quantum: distribución óptima considerando todos los factores
            
            # Generar puntuación compuesta para cada nodo
            node_scores = {}
            for node_id in available_nodes:
                node = self.nodes[node_id]
                
                # Combinar múltiples factores con pesos diferentes
                score = (
                    (1.0 - node.active_connections / node.max_connections) * 0.3 +
                    (1.0 - node.metrics["cpu_usage"]) * 0.2 +
                    (1.0 - node.metrics["memory_usage"]) * 0.15 +
                    (1.0 - node.get_avg_response_time() / 1.0) * 0.25 +  # normalizado a 1 segundo
                    (1.0 - node.metrics["error_rate"]) * 0.1
                )
                
                node_scores[node_id] = score
            
            # Seleccionar el mejor nodo
            if node_scores:
                selected_node = max(node_scores.items(), key=lambda x: x[1])[0]
        
        else:
            # Fallback a selección aleatoria
            selected_node = random.choice(available_nodes)
        
        # Si tenemos afinidad de sesión y se seleccionó un nodo, guardar asignación
        if self.session_affinity != SessionAffinityMode.NONE and session_key and selected_node:
            self.session_store.assign_session(session_key, selected_node)
        
        return selected_node
    
    async def execute_operation(self, 
                              operation: Callable[..., T], 
                              session_key: Optional[str] = None,
                              *args, **kwargs) -> Tuple[Optional[T], Optional[CloudNode]]:
        """
        Ejecutar operación en un nodo seleccionado.
        
        Args:
            operation: Función a ejecutar
            session_key: Clave de sesión para afinidad (opcional)
            *args: Argumentos para la operación
            **kwargs: Argumentos de palabra clave para la operación
            
        Returns:
            Tupla (resultado, nodo) o (None, None) si falló
        """
        # Seleccionar nodo
        node = await self.select_node(session_key)
        if not node:
            logger.warning(f"No hay nodos disponibles en '{self.name}' para ejecutar operación")
            self.metrics["failed_operations"] += 1
            self.metrics["last_error"] = "No hay nodos disponibles"
            return None, None
        
        # Registrar inicio de operación
        node.connection_started()
        self.metrics["total_operations"] += 1
        start_time = time.time()
        
        try:
            # Ejecutar operación
            # En un sistema real, aquí se invocaría la operación en el nodo seleccionado
            # mediante RPC, HTTP, etc. Para demo, ejecutamos localmente.
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)
            
            # Registrar éxito
            self.metrics["successful_operations"] += 1
            
            # Actualizar estadísticas de tiempo de respuesta
            response_time = time.time() - start_time
            node.metrics["response_times"].append(response_time)
            
            return result, node
            
        except Exception as e:
            # Registrar error
            logger.error(f"Error al ejecutar operación en nodo {node.node_id}: {e}")
            self.metrics["failed_operations"] += 1
            self.metrics["last_error"] = str(e)
            
            # Marcar fallo en el nodo
            node.consecutive_failures += 1
            
            return None, node
            
        finally:
            # Registrar fin de operación
            node.connection_finished()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado actual del balanceador.
        
        Returns:
            Diccionario con estado detallado
        """
        return {
            "name": self.name,
            "state": self.state.name,
            "algorithm": self.algorithm.name,
            "scaling_policy": self.scaling_policy.name,
            "session_affinity": self.session_affinity.name,
            "nodes_total": len(self.nodes),
            "nodes_healthy": len(self.healthy_nodes),
            "metrics": self.metrics,
            "scaling_settings": self.scaling_settings
        }
    
    def get_nodes_status(self) -> List[Dict[str, Any]]:
        """
        Obtener estado detallado de todos los nodos.
        
        Returns:
            Lista de diccionarios con estado de cada nodo
        """
        return [node.to_dict() for node in self.nodes.values()]


class CloudLoadBalancerManager:
    """
    Gestor centralizado de balanceadores de carga.
    
    Permite crear, obtener y gestionar múltiples balanceadores,
    así como acceder a métricas globales del sistema.
    """
    
    def __init__(self):
        """Inicializar gestor de balanceadores."""
        self.balancers: Dict[str, CloudLoadBalancer] = {}
        self._initialized = False
        self._maintenance_task = None
    
    async def initialize(self) -> bool:
        """
        Inicializar gestor y tareas de mantenimiento.
        
        Returns:
            True si se inicializó correctamente
        """
        if self._initialized:
            return True
        
        try:
            # Iniciar tarea de mantenimiento
            self._maintenance_task = asyncio.create_task(self._run_maintenance())
            self._initialized = True
            
            logger.info("CloudLoadBalancerManager inicializado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error al inicializar CloudLoadBalancerManager: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Detener todas las tareas y liberar recursos."""
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
        
        # Detener todos los balanceadores
        for balancer in self.balancers.values():
            await balancer.shutdown()
        
        self._initialized = False
        logger.info("CloudLoadBalancerManager detenido correctamente")
    
    async def _run_maintenance(self) -> None:
        """Tarea periódica para mantenimiento de balanceadores."""
        try:
            while True:
                # Limpiar sesiones expiradas
                for balancer in self.balancers.values():
                    expired_count = balancer.session_store.cleanup_expired_sessions()
                    if expired_count > 0:
                        logger.debug(f"Eliminadas {expired_count} sesiones expiradas de '{balancer.name}'")
                
                # Otras tareas de mantenimiento futuras podrían agregarse aquí
                
                await asyncio.sleep(60)  # Ejecutar cada minuto
        except asyncio.CancelledError:
            logger.info("Tarea de mantenimiento de CloudLoadBalancerManager cancelada")
        except Exception as e:
            logger.error(f"Error en tarea de mantenimiento de CloudLoadBalancerManager: {e}")
    
    async def create_balancer(self, 
                            name: str, 
                            algorithm: BalancerAlgorithm = BalancerAlgorithm.ROUND_ROBIN,
                            scaling_policy: ScalingPolicy = ScalingPolicy.NONE,
                            session_affinity: SessionAffinityMode = SessionAffinityMode.NONE,
                            initial_nodes: List[CloudNode] = None) -> Optional[CloudLoadBalancer]:
        """
        Crear un nuevo balanceador.
        
        Args:
            name: Nombre identificativo del balanceador
            algorithm: Algoritmo de balanceo a utilizar
            scaling_policy: Política de escalado automático
            session_affinity: Modo de afinidad de sesión
            initial_nodes: Lista de nodos iniciales (opcional)
            
        Returns:
            Balanceador creado o None si falló
        """
        # Verificar si ya existe un balanceador con ese nombre
        if name in self.balancers:
            logger.warning(f"Balanceador '{name}' ya existe")
            return self.balancers[name]
        
        try:
            # Crear balanceador
            balancer = CloudLoadBalancer(
                name=name,
                algorithm=algorithm,
                scaling_policy=scaling_policy,
                session_affinity=session_affinity
            )
            
            # Inicializar
            await balancer.initialize()
            
            # Añadir nodos iniciales si se proporcionan
            if initial_nodes:
                for node in initial_nodes:
                    await balancer.add_node(node)
            
            # Registrar en el gestor
            self.balancers[name] = balancer
            
            logger.info(f"Balanceador '{name}' creado correctamente")
            return balancer
            
        except Exception as e:
            logger.error(f"Error al crear balanceador '{name}': {e}")
            return None
    
    def get_balancer(self, name: str) -> Optional[CloudLoadBalancer]:
        """
        Obtener un balanceador por nombre.
        
        Args:
            name: Nombre del balanceador
            
        Returns:
            Balanceador o None si no existe
        """
        return self.balancers.get(name)
    
    async def delete_balancer(self, name: str) -> bool:
        """
        Eliminar un balanceador.
        
        Args:
            name: Nombre del balanceador
            
        Returns:
            True si se eliminó correctamente
        """
        if name not in self.balancers:
            logger.warning(f"Balanceador '{name}' no existe para eliminación")
            return False
        
        try:
            # Obtener balanceador
            balancer = self.balancers[name]
            
            # Detener tareas
            await balancer.shutdown()
            
            # Eliminar del registro
            self.balancers.pop(name)
            
            logger.info(f"Balanceador '{name}' eliminado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error al eliminar balanceador '{name}': {e}")
            return False
    
    def get_all_balancers(self) -> Dict[str, CloudLoadBalancer]:
        """
        Obtener todos los balanceadores registrados.
        
        Returns:
            Diccionario de balanceadores por nombre
        """
        return self.balancers.copy()
    
    def get_balancers_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtener estado de todos los balanceadores.
        
        Returns:
            Diccionario con estado detallado de cada balanceador
        """
        return {name: balancer.get_status() for name, balancer in self.balancers.items()}
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas globales de todos los balanceadores.
        
        Returns:
            Diccionario con métricas agregadas
        """
        total_operations = 0
        successful_operations = 0
        failed_operations = 0
        avg_response_times = []
        error_rates = []
        total_nodes = 0
        healthy_nodes = 0
        
        # Agregar métricas de todos los balanceadores
        for balancer in self.balancers.values():
            total_operations += balancer.metrics["total_operations"]
            successful_operations += balancer.metrics["successful_operations"]
            failed_operations += balancer.metrics["failed_operations"]
            
            if balancer.metrics["avg_response_time"] > 0:
                avg_response_times.append(balancer.metrics["avg_response_time"])
            
            if balancer.metrics["error_rate"] >= 0:
                error_rates.append(balancer.metrics["error_rate"])
            
            total_nodes += len(balancer.nodes)
            healthy_nodes += len(balancer.healthy_nodes)
        
        # Calcular promedios
        avg_response_time = sum(avg_response_times) / len(avg_response_times) if avg_response_times else 0
        avg_error_rate = sum(error_rates) / len(error_rates) if error_rates else 0
        
        return {
            "total_balancers": len(self.balancers),
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "failed_operations": failed_operations,
            "success_rate": successful_operations / total_operations if total_operations > 0 else 1.0,
            "avg_response_time": avg_response_time,
            "avg_error_rate": avg_error_rate,
            "total_nodes": total_nodes,
            "healthy_nodes": healthy_nodes,
            "health_rate": healthy_nodes / total_nodes if total_nodes > 0 else 1.0,
            "timestamp": time.time()
        }


# Singleton global para acceso desde cualquier parte del código
load_balancer_manager = CloudLoadBalancerManager()


# Decorador para distribuir funciones a través del balanceador
def distributed(balancer_name: str, session_key_func: Optional[Callable] = None):
    """
    Decorador para distribuir funciones a través de un balanceador de carga.
    
    Args:
        balancer_name: Nombre del balanceador a usar
        session_key_func: Función opcional para extraer clave de sesión de los argumentos
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Obtener balanceador
            balancer = load_balancer_manager.get_balancer(balancer_name)
            if not balancer:
                raise ValueError(f"Balanceador '{balancer_name}' no existe")
            
            # Determinar clave de sesión si se proporcionó función
            session_key = None
            if session_key_func:
                session_key = session_key_func(*args, **kwargs)
            
            # Ejecutar operación a través del balanceador
            result, node = await balancer.execute_operation(func, session_key, *args, **kwargs)
            return result
        
        return wrapper
    
    return decorator


# Integración con Circuit Breaker
def distributed_with_circuit_breaker(balancer_name: str, circuit_name: str, session_key_func: Optional[Callable] = None, **cb_args):
    """
    Decorador que combina distribución y circuit breaker.
    
    Args:
        balancer_name: Nombre del balanceador a usar
        circuit_name: Nombre del circuit breaker a usar
        session_key_func: Función opcional para extraer clave de sesión
        **cb_args: Argumentos adicionales para el circuit breaker
    """
    def decorator(func):
        @circuit_protected(circuit_name, **cb_args)
        async def protected_func(*args, **kwargs):
            # Obtener balanceador
            balancer = load_balancer_manager.get_balancer(balancer_name)
            if not balancer:
                raise ValueError(f"Balanceador '{balancer_name}' no existe")
            
            # Determinar clave de sesión si se proporcionó función
            session_key = None
            if session_key_func:
                session_key = session_key_func(*args, **kwargs)
            
            # Ejecutar operación a través del balanceador
            result, node = await balancer.execute_operation(func, session_key, *args, **kwargs)
            
            if result is None:
                raise Exception(f"Operación falló en balanceador '{balancer_name}'")
                
            return result
        
        return protected_func
    
    return decorator


# Integración con Checkpoint Manager
def distributed_with_checkpoint(balancer_name: str, component_id: str, session_key_func: Optional[Callable] = None):
    """
    Decorador que combina distribución y checkpoint automático.
    
    Args:
        balancer_name: Nombre del balanceador a usar
        component_id: ID del componente para checkpoint
        session_key_func: Función opcional para extraer clave de sesión
    """
    def decorator(func):
        @checkpoint_state(component_id=component_id, tags=["distributed", balancer_name])
        async def checkpoint_func(*args, **kwargs):
            # Obtener balanceador
            balancer = load_balancer_manager.get_balancer(balancer_name)
            if not balancer:
                raise ValueError(f"Balanceador '{balancer_name}' no existe")
            
            # Determinar clave de sesión si se proporcionó función
            session_key = None
            if session_key_func:
                session_key = session_key_func(*args, **kwargs)
            
            # Ejecutar operación a través del balanceador
            result, node = await balancer.execute_operation(func, session_key, *args, **kwargs)
            return result
        
        return checkpoint_func
    
    return decorator


# Integración completa: Circuit Breaker + Checkpoint + Distribución
def ultra_resilient(balancer_name: str, 
                   circuit_name: str,
                   component_id: str, 
                   session_key_func: Optional[Callable] = None,
                   **cb_args):
    """
    Decorador que combina todas las capacidades para resiliencia máxima.
    
    Args:
        balancer_name: Nombre del balanceador a usar
        circuit_name: Nombre del circuit breaker a usar
        component_id: ID del componente para checkpoint
        session_key_func: Función opcional para extraer clave de sesión
        **cb_args: Argumentos adicionales para el circuit breaker
    """
    def decorator(func):
        # Aplicar Circuit Breaker
        @circuit_protected(circuit_name, **cb_args)
        # Aplicar Checkpoint
        @checkpoint_state(component_id=component_id, tags=["ultra_resilient", balancer_name, circuit_name])
        async def ultra_protected_func(*args, **kwargs):
            # Obtener balanceador
            balancer = load_balancer_manager.get_balancer(balancer_name)
            if not balancer:
                raise ValueError(f"Balanceador '{balancer_name}' no existe")
            
            # Determinar clave de sesión si se proporcionó función
            session_key = None
            if session_key_func:
                session_key = session_key_func(*args, **kwargs)
            
            # Ejecutar operación a través del balanceador
            result, node = await balancer.execute_operation(func, session_key, *args, **kwargs)
            
            if result is None:
                raise Exception(f"Operación falló en balanceador '{balancer_name}'")
                
            return result
        
        return ultra_protected_func
    
    return decorator


# Para pruebas si se ejecuta este archivo directamente
if __name__ == "__main__":
    async def run_demo():
        print("\n=== DEMOSTRACIÓN DEL CLOUDLOADBALANCER ===\n")
        
        # Inicializar gestor
        manager = CloudLoadBalancerManager()
        await manager.initialize()
        
        # Crear algunos nodos
        nodes = [
            CloudNode(f"node_{i}", "127.0.0.1", 8080 + i, random.uniform(0.5, 1.5))
            for i in range(5)
        ]
        
        # Crear balanceador
        balancer = await manager.create_balancer(
            name="demo_balancer",
            algorithm=BalancerAlgorithm.WEIGHTED,
            scaling_policy=ScalingPolicy.THRESHOLD,
            initial_nodes=nodes
        )
        
        if not balancer:
            print("Error al crear balanceador")
            return
        
        # Verificar estado inicial
        print("Estado inicial:")
        status = balancer.get_status()
        print(f"  Nombre: {status['name']}")
        print(f"  Estado: {status['state']}")
        print(f"  Algoritmo: {status['algorithm']}")
        print(f"  Nodos totales: {status['nodes_total']}")
        print(f"  Nodos saludables: {status['nodes_healthy']}")
        
        # Función de ejemplo para distribución
        async def example_operation(data):
            # Simular operación
            await asyncio.sleep(random.uniform(0.01, 0.1))
            return {"result": f"Processed: {data}", "timestamp": time.time()}
        
        # Ejecutar operaciones
        print("\nEjecutando operaciones distribuidas...")
        results = []
        
        for i in range(20):
            result, node = await balancer.execute_operation(
                example_operation,
                session_key=f"user_{i % 5}",  # 5 usuarios simulados
                data=f"Data_{i}"
            )
            
            if result:
                node_info = f"{node.node_id}" if node else "unknown"
                results.append((i, node_info, result))
                print(f"  Operación {i}: Éxito - Nodo: {node_info}")
            else:
                print(f"  Operación {i}: Error")
        
        # Prueba de escalado automático
        print("\nSimulando carga elevada para activar escalado...")
        
        # Simular CPU alta
        for node in balancer.nodes.values():
            node.metrics["cpu_usage"] = random.uniform(0.75, 0.95)
            node.active_connections = int(node.max_connections * 0.8)
        
        # Forzar decisión de escalado
        await balancer._make_scaling_decisions()
        
        # Verificar estado después de escalado
        status_after = balancer.get_status()
        print(f"\nEstado después de escalado:")
        print(f"  Nodos totales: {status_after['nodes_total']}")
        print(f"  Nodos saludables: {status_after['nodes_healthy']}")
        
        # Marcar algunos nodos como no saludables
        print("\nSimulando fallo en algunos nodos...")
        for node_id in list(balancer.healthy_nodes)[:2]:
            balancer.nodes[node_id].health_status = NodeHealthStatus.UNHEALTHY
            balancer.healthy_nodes.remove(node_id)
        
        # Verificar estado con nodos fallidos
        await balancer._check_all_nodes_health()
        status_degraded = balancer.get_status()
        print(f"\nEstado con nodos fallidos:")
        print(f"  Estado: {status_degraded['state']}")
        print(f"  Nodos totales: {status_degraded['nodes_total']}")
        print(f"  Nodos saludables: {status_degraded['nodes_healthy']}")
        
        # Ejecutar más operaciones para ver distribución con nodos fallidos
        print("\nEjecutando operaciones con nodos fallidos...")
        for i in range(10):
            result, node = await balancer.execute_operation(
                example_operation,
                session_key=f"user_{i % 5}",
                data=f"DataAfterFailure_{i}"
            )
            
            if result:
                node_info = f"{node.node_id}" if node else "unknown"
                print(f"  Operación {i}: Éxito - Nodo: {node_info}")
            else:
                print(f"  Operación {i}: Error")
        
        # Prueba de escalado hacia abajo
        print("\nSimulando baja carga para activar escalado hacia abajo...")
        
        # Simular CPU baja en nodos saludables
        for node_id in balancer.healthy_nodes:
            balancer.nodes[node_id].metrics["cpu_usage"] = random.uniform(0.1, 0.25)
            balancer.nodes[node_id].active_connections = int(node.max_connections * 0.1)
        
        # Evitar período de enfriamiento
        balancer.scaling_settings["last_scale_down"] = 0
        
        # Forzar decisión de escalado
        await balancer._make_scaling_decisions()
        
        # Verificar estado final
        final_status = balancer.get_status()
        print(f"\nEstado final:")
        print(f"  Estado: {final_status['state']}")
        print(f"  Nodos totales: {final_status['nodes_total']}")
        print(f"  Nodos saludables: {final_status['nodes_healthy']}")
        
        # Ejemplo del decorador distributed
        print("\nPrueba de decorador 'distributed':")
        
        @distributed("demo_balancer")
        async def decorated_function(value):
            await asyncio.sleep(0.05)
            return f"Decorated result: {value}"
        
        result = await decorated_function("test_value")
        print(f"  Resultado: {result}")
        
        # Limpiar recursos
        await manager.shutdown()
        
        print("\n=== DEMOSTRACIÓN COMPLETADA ===\n")
    
    # Ejecutar demo
    asyncio.run(run_demo())