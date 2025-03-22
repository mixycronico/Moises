"""
Pruebas de escenarios extremos para el core del sistema Genesis.

Este módulo contiene pruebas que simulan situaciones extremas como:
- Caídas de red prolongadas
- Cambios rápidos en la configuración del sistema
- Operación continua bajo alta carga
- Situaciones de recursos limitados
- Fallas en cascada de múltiples componentes
"""

import pytest
import asyncio
import logging
import time
import random
from typing import Dict, Any, List, Optional, Tuple, Set
from unittest.mock import patch, MagicMock

from genesis.core.component import Component
from genesis.core.engine_non_blocking import EngineNonBlocking

# Configurar logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class NetworkSimulatorComponent(Component):
    """Componente que simula condiciones de red variables."""
    
    def __init__(self, name: str, initial_state: str = "online"):
        """
        Inicializar componente simulador de red.
        
        Args:
            name: Nombre del componente
            initial_state: Estado inicial de la red ("online", "offline", "unstable")
        """
        super().__init__(name)
        self.events: List[Dict[str, Any]] = []
        self.state = initial_state
        self.processed_online = 0
        self.processed_offline = 0
        self.processed_unstable = 0
        self.latency_ms = 0
        self.packet_loss = 0.0
    
    async def start(self) -> None:
        """Iniciar componente."""
        logger.info(f"Iniciando componente simulador de red {self.name} en estado {self.state}")
    
    async def stop(self) -> None:
        """Detener componente."""
        logger.info(f"Deteniendo componente simulador de red {self.name}")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """
        Manejar un evento, simulando condiciones de red.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Opcional, respuesta al evento
            
        Raises:
            ConnectionError: Si la red está "offline" o falla por inestabilidad
        """
        # Procesar comandos para cambiar el estado de red
        if event_type == "network_control":
            if "state" in data:
                old_state = self.state
                self.state = data["state"]
                logger.info(f"Cambiando estado de red de {old_state} a {self.state}")
            
            if "latency_ms" in data:
                self.latency_ms = data["latency_ms"]
                logger.info(f"Estableciendo latencia de red a {self.latency_ms}ms")
            
            if "packet_loss" in data:
                self.packet_loss = data["packet_loss"]
                logger.info(f"Estableciendo pérdida de paquetes a {self.packet_loss*100}%")
            
            return {"previous_state": old_state if 'old_state' in locals() else self.state}
        
        # Registrar evento
        self.events.append({
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": time.time(),
            "network_state": self.state
        })
        
        # Simular comportamiento según el estado de la red
        if self.state == "offline":
            self.processed_offline += 1
            raise ConnectionError(f"Red no disponible (estado: {self.state})")
        
        elif self.state == "unstable":
            self.processed_unstable += 1
            
            # Simular latencia variable
            latency = self.latency_ms * (0.5 + random.random())
            if latency > 0:
                await asyncio.sleep(latency / 1000.0)  # Convertir ms a segundos
            
            # Simular pérdida de paquetes
            if random.random() < self.packet_loss:
                raise ConnectionError(f"Paquete perdido (pérdida: {self.packet_loss*100}%)")
        
        else:  # online
            self.processed_online += 1
            
            # Simular latencia fija
            if self.latency_ms > 0:
                await asyncio.sleep(self.latency_ms / 1000.0)  # Convertir ms a segundos
        
        return {"network_state": self.state}
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas del componente.
        
        Returns:
            Diccionario con métricas
        """
        total = self.processed_online + self.processed_offline + self.processed_unstable
        return {
            "total_processed": total,
            "processed_online": self.processed_online,
            "processed_offline": self.processed_offline,
            "processed_unstable": self.processed_unstable,
            "current_state": self.state,
            "current_latency_ms": self.latency_ms,
            "current_packet_loss": self.packet_loss
        }


class ResourceMonitorComponent(Component):
    """Componente que simula el monitoreo y la limitación de recursos."""
    
    def __init__(self, name: str, 
                 max_memory_mb: int = 100, 
                 max_cpu_percent: int = 80,
                 initial_memory_mb: int = 0,
                 initial_cpu_percent: int = 0):
        """
        Inicializar componente monitor de recursos.
        
        Args:
            name: Nombre del componente
            max_memory_mb: Límite máximo de memoria en MB
            max_cpu_percent: Límite máximo de CPU en porcentaje
            initial_memory_mb: Memoria inicial utilizada en MB
            initial_cpu_percent: CPU inicial utilizado en porcentaje
        """
        super().__init__(name)
        self.events: List[Dict[str, Any]] = []
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.memory_mb = initial_memory_mb
        self.cpu_percent = initial_cpu_percent
        self.memory_warnings = 0
        self.cpu_warnings = 0
        self.resource_errors = 0
    
    async def start(self) -> None:
        """Iniciar componente."""
        logger.info(f"Iniciando componente monitor de recursos {self.name}")
        logger.info(f"Límites: Memoria={self.max_memory_mb}MB, CPU={self.max_cpu_percent}%")
    
    async def stop(self) -> None:
        """Detener componente."""
        logger.info(f"Deteniendo componente monitor de recursos {self.name}")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """
        Manejar un evento, simulando consumo y monitoreo de recursos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Opcional, respuesta al evento
            
        Raises:
            MemoryError: Si se excede el límite de memoria
            Exception: Si se excede el límite de CPU
        """
        # Registrar evento
        self.events.append({
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": time.time(),
            "memory_mb": self.memory_mb,
            "cpu_percent": self.cpu_percent
        })
        
        # Procesar comandos para controlar recursos
        if event_type == "resource_control":
            if "memory_mb" in data:
                self.memory_mb = data["memory_mb"]
            if "cpu_percent" in data:
                self.cpu_percent = data["cpu_percent"]
            
            return {"memory_mb": self.memory_mb, "cpu_percent": self.cpu_percent}
        
        # Para otros tipos de eventos, simular consumo de recursos
        if "resource_impact" in data:
            impact = data["resource_impact"]
            if "memory_mb" in impact:
                self.memory_mb += impact["memory_mb"]
            if "cpu_percent" in impact:
                self.cpu_percent += impact["cpu_percent"]
        
        # Verificar límites
        warnings = []
        if self.memory_mb > 0.8 * self.max_memory_mb:
            self.memory_warnings += 1
            warnings.append(f"Memoria alta: {self.memory_mb}/{self.max_memory_mb}MB")
        
        if self.cpu_percent > 0.8 * self.max_cpu_percent:
            self.cpu_warnings += 1
            warnings.append(f"CPU alto: {self.cpu_percent}/{self.max_cpu_percent}%")
        
        # Fallar si se exceden los límites
        if self.memory_mb > self.max_memory_mb:
            self.resource_errors += 1
            raise MemoryError(f"Memoria excedida: {self.memory_mb}MB (máx: {self.max_memory_mb}MB)")
        
        if self.cpu_percent > self.max_cpu_percent:
            self.resource_errors += 1
            raise Exception(f"CPU excedido: {self.cpu_percent}% (máx: {self.max_cpu_percent}%)")
        
        return {
            "memory_mb": self.memory_mb,
            "cpu_percent": self.cpu_percent,
            "warnings": warnings
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas del componente.
        
        Returns:
            Diccionario con métricas
        """
        return {
            "processed_events": len(self.events),
            "memory_mb": self.memory_mb,
            "cpu_percent": self.cpu_percent,
            "memory_warnings": self.memory_warnings,
            "cpu_warnings": self.cpu_warnings,
            "resource_errors": self.resource_errors,
            "memory_usage_percent": (self.memory_mb / self.max_memory_mb) * 100 if self.max_memory_mb > 0 else 0,
            "cpu_usage_percent": self.cpu_percent
        }


class ConfigChangeComponent(Component):
    """Componente que simula cambios rápidos de configuración."""
    
    def __init__(self, name: str):
        """Inicializar componente de cambio de configuración."""
        super().__init__(name)
        self.events: List[Dict[str, Any]] = []
        self.config: Dict[str, Any] = {}
        self.config_changes = 0
        self.reconfigurations = 0
        self.config_errors = 0
    
    async def start(self) -> None:
        """Iniciar componente."""
        logger.info(f"Iniciando componente de cambio de configuración {self.name}")
        self.config = {
            "feature_flags": {
                "feature_a": True,
                "feature_b": False,
                "feature_c": True
            },
            "thresholds": {
                "warning": 80,
                "error": 95,
                "critical": 99
            },
            "timeouts": {
                "short": 1000,
                "medium": 5000,
                "long": 30000
            }
        }
    
    async def stop(self) -> None:
        """Detener componente."""
        logger.info(f"Deteniendo componente de cambio de configuración {self.name}")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """
        Manejar un evento, aplicando cambios de configuración si es necesario.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Opcional, respuesta al evento
        """
        # Registrar evento con la configuración actual
        event_record = {
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": time.time(),
            "config_snapshot": self.config.copy()
        }
        self.events.append(event_record)
        
        # Procesar comandos de cambio de configuración
        if event_type == "config_update":
            old_config = self.config.copy()
            
            try:
                path = data.get("path", "")
                value = data.get("value")
                operation = data.get("operation", "set")
                
                if not path:
                    if operation == "set":
                        # Reemplazar toda la configuración
                        if "config" in data:
                            self.config = data["config"]
                            self.reconfigurations += 1
                    elif operation == "reset":
                        # Restaurar configuración por defecto
                        await self.start()
                        self.reconfigurations += 1
                else:
                    # Actualizar una parte específica de la configuración
                    parts = path.split(".")
                    target = self.config
                    
                    # Navegar hasta el penúltimo nivel
                    for i in range(len(parts) - 1):
                        if parts[i] not in target:
                            target[parts[i]] = {}
                        target = target[parts[i]]
                    
                    # Aplicar la operación
                    last_key = parts[-1]
                    if operation == "set":
                        target[last_key] = value
                    elif operation == "delete":
                        if last_key in target:
                            del target[last_key]
                    elif operation == "increment" and last_key in target:
                        if isinstance(target[last_key], (int, float)):
                            target[last_key] += value
                    elif operation == "toggle" and last_key in target:
                        if isinstance(target[last_key], bool):
                            target[last_key] = not target[last_key]
                    
                    self.config_changes += 1
                
                # Simular un breve período de recarga de configuración
                await asyncio.sleep(0.05)
                
                return {
                    "success": True,
                    "previous_config": old_config,
                    "new_config": self.config
                }
            
            except Exception as e:
                self.config_errors += 1
                logger.error(f"Error al actualizar configuración: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
        
        # Para otros eventos, usar la configuración actual para determinar el comportamiento
        result = {"config_applied": False}
        
        if "check_feature" in data:
            feature = data["check_feature"]
            if feature in self.config.get("feature_flags", {}):
                result["feature_enabled"] = self.config["feature_flags"][feature]
                result["config_applied"] = True
        
        if "check_threshold" in data:
            threshold = data["check_threshold"]
            if threshold in self.config.get("thresholds", {}):
                result["threshold_value"] = self.config["thresholds"][threshold]
                result["config_applied"] = True
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas del componente.
        
        Returns:
            Diccionario con métricas
        """
        return {
            "processed_events": len(self.events),
            "config_changes": self.config_changes,
            "reconfigurations": self.reconfigurations,
            "config_errors": self.config_errors,
            "current_config_size": len(str(self.config)),
            "feature_count": len(self.config.get("feature_flags", {}))
        }


class CascadeFailureComponent(Component):
    """Componente que simula fallos en cascada entre componentes dependientes."""
    
    def __init__(self, name: str, dependencies: Optional[List[str]] = None, fail_threshold: int = 3):
        """
        Inicializar componente de fallo en cascada.
        
        Args:
            name: Nombre del componente
            dependencies: Lista de nombres de componentes de los que depende
            fail_threshold: Número de fallos de dependencias antes de fallar
        """
        super().__init__(name)
        self.events: List[Dict[str, Any]] = []
        self.dependencies = dependencies or []
        self.fail_threshold = fail_threshold
        self.dependency_failures: Dict[str, int] = {dep: 0 for dep in self.dependencies}
        self.healthy = True
        self.cascaded_failures = 0
        self.processed_healthy = 0
        self.processed_unhealthy = 0
    
    async def start(self) -> None:
        """Iniciar componente."""
        logger.info(f"Iniciando componente de fallo en cascada {self.name}")
        logger.info(f"Dependencias: {self.dependencies}")
    
    async def stop(self) -> None:
        """Detener componente."""
        logger.info(f"Deteniendo componente de fallo en cascada {self.name}")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """
        Manejar un evento, considerando fallos en cascada.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Opcional, respuesta al evento
            
        Raises:
            Exception: Si el componente no está sano o si debe fallar en cascada
        """
        # Registrar evento
        self.events.append({
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": time.time(),
            "healthy": self.healthy
        })
        
        # Manejar eventos de dependencia
        if event_type == "dependency_status":
            if "component" in data and "status" in data:
                component = data["component"]
                status = data["status"]
                
                if component in self.dependency_failures:
                    if status == "failure":
                        self.dependency_failures[component] += 1
                        logger.info(f"{self.name}: Registrado fallo de dependencia {component} "
                                  f"(total: {self.dependency_failures[component]})")
                    elif status == "recovery":
                        self.dependency_failures[component] = 0
                        logger.info(f"{self.name}: Registrada recuperación de dependencia {component}")
                
                # Verificar si se alcanzó el umbral de fallos en alguna dependencia
                failed_dependencies = [dep for dep, count in self.dependency_failures.items() 
                                     if count >= self.fail_threshold]
                
                if failed_dependencies and self.healthy:
                    self.healthy = False
                    self.cascaded_failures += 1
                    logger.warning(f"{self.name}: Fallo en cascada debido a dependencias: {failed_dependencies}")
                
                elif not failed_dependencies and not self.healthy:
                    self.healthy = True
                    logger.info(f"{self.name}: Recuperado de fallo en cascada")
                
                return {
                    "component": self.name,
                    "healthy": self.healthy,
                    "failed_dependencies": failed_dependencies
                }
        
        # Para recuperación explícita
        if event_type == "recovery_command" and not self.healthy:
            self.healthy = True
            for dep in self.dependency_failures:
                self.dependency_failures[dep] = 0
            logger.info(f"{self.name}: Recuperado por comando explícito")
            return {"recovered": True}
        
        # Fallar si no está sano
        if not self.healthy:
            self.processed_unhealthy += 1
            raise Exception(f"Componente {self.name} no está sano debido a fallos en cascada")
        
        # Procesar normalmente si está sano
        self.processed_healthy += 1
        return {"processed": True, "component": self.name}
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas del componente.
        
        Returns:
            Diccionario con métricas
        """
        return {
            "total_processed": self.processed_healthy + self.processed_unhealthy,
            "processed_healthy": self.processed_healthy,
            "processed_unhealthy": self.processed_unhealthy,
            "cascaded_failures": self.cascaded_failures,
            "current_health": "healthy" if self.healthy else "unhealthy",
            "dependency_failures": self.dependency_failures.copy()
        }


@pytest.mark.asyncio
async def test_network_disruption():
    """Prueba el sistema durante disrupciones de red simuladas."""
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)  # El parámetro component_timeout no existe en esta clase
    
    # Crear componentes
    network = NetworkSimulatorComponent("network", initial_state="online")
    normal_comp = NetworkSimulatorComponent("normal", initial_state="online")
    
    # Registrar componentes
    engine.register_component(network)
    engine.register_component(normal_comp)
    
    # Iniciar motor
    await engine.start()
    
    # Enviar eventos con red en línea
    for i in range(5):
        await engine.emit_event(f"test_event_{i}", {"id": i}, "test")
        await asyncio.sleep(0.1)
    
    # Cambiar estado de red a "offline"
    await engine.emit_event("network_control", {"state": "offline"}, "test")
    await asyncio.sleep(0.2)
    
    # Enviar eventos durante la desconexión
    for i in range(5):
        await engine.emit_event(f"offline_event_{i}", {"id": i}, "test")
        await asyncio.sleep(0.1)
    
    # Restablecer conexión
    await engine.emit_event("network_control", {"state": "online"}, "test")
    await asyncio.sleep(0.2)
    
    # Enviar eventos después de recuperarse
    for i in range(5):
        await engine.emit_event(f"recovery_event_{i}", {"id": i}, "test")
        await asyncio.sleep(0.1)
    
    # Simular red inestable con latencia y pérdida de paquetes
    await engine.emit_event("network_control", 
                          {"state": "unstable", "latency_ms": 200, "packet_loss": 0.3}, 
                          "test")
    await asyncio.sleep(0.2)
    
    # Enviar eventos durante la inestabilidad
    for i in range(5):
        await engine.emit_event(f"unstable_event_{i}", {"id": i}, "test")
        await asyncio.sleep(0.3)  # Esperar más debido a la latencia simulada
    
    # Verificar métricas
    network_metrics = network.get_metrics()
    normal_metrics = normal_comp.get_metrics()
    
    logger.info(f"Métricas de red: {network_metrics}")
    logger.info(f"Métricas de componente normal: {normal_metrics}")
    
    # El componente de red debería haber procesado todos los eventos en diferentes estados
    assert network_metrics["processed_online"] > 0, "Debería haber eventos procesados en estado online"
    assert network_metrics["processed_offline"] > 0, "Debería haber eventos procesados en estado offline"
    assert network_metrics["processed_unstable"] > 0, "Debería haber eventos procesados en estado inestable"
    
    # El componente normal debería haberse mantenido online todo el tiempo
    assert normal_metrics["processed_online"] > 0, "El componente normal debería haber procesado eventos online"
    assert normal_metrics["processed_offline"] == 0, "El componente normal no debería tener eventos offline"
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
async def test_resource_exhaustion():
    """Prueba el sistema bajo condiciones de recursos limitados simulados."""
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componentes
    resource_monitor = ResourceMonitorComponent("resource_monitor", 
                                               max_memory_mb=1000, 
                                               max_cpu_percent=90,
                                               initial_memory_mb=200,
                                               initial_cpu_percent=20)
    
    normal_comp = CascadeFailureComponent("normal", dependencies=["resource_monitor"])
    
    # Registrar componentes
    engine.register_component(resource_monitor)
    engine.register_component(normal_comp)
    
    # Iniciar motor
    await engine.start()
    
    # Enviar eventos con uso normal de recursos
    for i in range(5):
        await engine.emit_event(f"normal_event_{i}", 
                              {"id": i, "resource_impact": {"memory_mb": 10, "cpu_percent": 2}}, 
                              "test")
        await asyncio.sleep(0.1)
    
    # Verificar que los recursos han aumentado pero están dentro de límites
    metrics = resource_monitor.get_metrics()
    assert 200 < metrics["memory_mb"] < 300, "La memoria debería haber aumentado gradualmente"
    assert 20 < metrics["cpu_percent"] < 40, "El CPU debería haber aumentado gradualmente"
    
    # Aumentar el uso de recursos drásticamente (pero por debajo del límite)
    await engine.emit_event("resource_control", 
                          {"memory_mb": 700, "cpu_percent": 70}, 
                          "test")
    await asyncio.sleep(0.1)
    
    # Verificar que los recursos están en nivel de advertencia pero no de error
    metrics = resource_monitor.get_metrics()
    assert metrics["memory_warnings"] > 0, "Debería haber advertencias de memoria"
    assert metrics["cpu_warnings"] > 0, "Debería haber advertencias de CPU"
    assert metrics["resource_errors"] == 0, "No debería haber errores de recursos aún"
    
    # Enviar eventos que deberían procesarse a pesar de recursos altos
    for i in range(5):
        await engine.emit_event(f"high_resource_event_{i}", 
                              {"id": i, "resource_impact": {"memory_mb": 5, "cpu_percent": 1}}, 
                              "test")
        await asyncio.sleep(0.1)
    
    # Exceder los límites de recursos
    await engine.emit_event("resource_control", 
                          {"memory_mb": 1100, "cpu_percent": 95}, 
                          "test")
    await asyncio.sleep(0.1)
    
    # Enviar eventos que deberían causar errores de recursos
    for i in range(5):
        await engine.emit_event(f"resource_error_event_{i}", 
                              {"id": i}, 
                              "test")
        await asyncio.sleep(0.1)
    
    # Verificar que se registraron errores de recursos
    metrics = resource_monitor.get_metrics()
    assert metrics["resource_errors"] > 0, "Debería haber errores por recursos excedidos"
    
    # Restaurar recursos a niveles normales
    await engine.emit_event("resource_control", 
                          {"memory_mb": 300, "cpu_percent": 30}, 
                          "test")
    await asyncio.sleep(0.1)
    
    # Enviar eventos después de la recuperación
    for i in range(5):
        await engine.emit_event(f"recovery_event_{i}", 
                              {"id": i, "resource_impact": {"memory_mb": 5, "cpu_percent": 1}}, 
                              "test")
        await asyncio.sleep(0.1)
    
    # Verificar métricas finales
    metrics = resource_monitor.get_metrics()
    logger.info(f"Métricas finales de recursos: {metrics}")
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
async def test_rapid_config_changes():
    """Prueba el sistema con cambios rápidos de configuración."""
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componente de configuración
    config_comp = ConfigChangeComponent("config")
    
    # Registrar componente
    engine.register_component(config_comp)
    
    # Iniciar motor
    await engine.start()
    
    # Enviar eventos para verificar la configuración inicial
    await engine.emit_event("check_config", {"check_feature": "feature_a"}, "test")
    await asyncio.sleep(0.1)
    
    # Realizar una serie de cambios rápidos de configuración
    config_changes = [
        {"path": "feature_flags.feature_a", "operation": "toggle", "value": None},
        {"path": "thresholds.warning", "operation": "set", "value": 90},
        {"path": "timeouts.short", "operation": "increment", "value": 500},
        {"path": "feature_flags.feature_d", "operation": "set", "value": True},
        {"path": "feature_flags.feature_b", "operation": "delete", "value": None}
    ]
    
    # Aplicar cambios rápidamente
    for i, change in enumerate(config_changes):
        await engine.emit_event("config_update", change, "test")
        await asyncio.sleep(0.01)  # Cambios muy rápidos
    
    # Enviar eventos para verificar la configuración después de los cambios
    await engine.emit_event("check_config", {"check_feature": "feature_a"}, "test")
    await engine.emit_event("check_config", {"check_feature": "feature_d"}, "test")
    await engine.emit_event("check_config", {"check_threshold": "warning"}, "test")
    await asyncio.sleep(0.1)
    
    # Reemplazar toda la configuración
    new_config = {
        "feature_flags": {
            "feature_x": True,
            "feature_y": False,
        },
        "thresholds": {
            "low": 20,
            "high": 80
        },
        "new_section": {
            "enabled": True,
            "value": 42
        }
    }
    
    await engine.emit_event("config_update", {"operation": "set", "config": new_config}, "test")
    await asyncio.sleep(0.1)
    
    # Verificar que la configuración se actualizó completamente
    await engine.emit_event("check_config", {"check_feature": "feature_x"}, "test")
    await engine.emit_event("check_config", {"check_threshold": "high"}, "test")
    await asyncio.sleep(0.1)
    
    # Verificar métricas
    metrics = config_comp.get_metrics()
    logger.info(f"Métricas de configuración: {metrics}")
    
    assert metrics["config_changes"] >= 5, "Deberían haberse registrado al menos 5 cambios de configuración"
    assert metrics["reconfigurations"] >= 1, "Debería haberse registrado al menos una reconfiguración completa"
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
async def test_cascading_failures():
    """Prueba el sistema con fallos en cascada entre componentes dependientes."""
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componentes con dependencias en cascada
    comp_a = CascadeFailureComponent("comp_a", dependencies=[])
    comp_b = CascadeFailureComponent("comp_b", dependencies=["comp_a"])
    comp_c = CascadeFailureComponent("comp_c", dependencies=["comp_b"])
    comp_d = CascadeFailureComponent("comp_d", dependencies=["comp_b", "comp_c"])
    comp_e = CascadeFailureComponent("comp_e", dependencies=["comp_a"])
    
    # Registrar componentes
    for comp in [comp_a, comp_b, comp_c, comp_d, comp_e]:
        engine.register_component(comp)
    
    # Iniciar motor
    await engine.start()
    
    # Enviar eventos normales a todos los componentes
    for i in range(3):
        await engine.emit_event(f"normal_event_{i}", {"id": i}, "test")
        await asyncio.sleep(0.1)
    
    # Verificar que todos los componentes procesaron los eventos correctamente
    for comp in [comp_a, comp_b, comp_c, comp_d, comp_e]:
        assert comp.processed_healthy == 3, f"El componente {comp.name} debería haber procesado 3 eventos en estado sano"
    
    # Simular fallos en el componente A
    for i in range(3):  # 3 fallos, que debería alcanzar el umbral
        await engine.emit_event("dependency_status", {"component": "comp_a", "status": "failure"}, "test")
        await asyncio.sleep(0.1)
    
    # Verificar que el componente B falló en cascada debido a A
    assert not comp_b.healthy, "El componente B debería haber fallado en cascada debido a fallos en A"
    
    # Verificar que el componente C falló en cascada debido a B
    assert not comp_c.healthy, "El componente C debería haber fallado en cascada debido a fallos en B"
    
    # Verificar que el componente D falló en cascada debido a B y C
    assert not comp_d.healthy, "El componente D debería haber fallado en cascada debido a fallos en B y C"
    
    # Verificar que el componente E falló en cascada debido a A
    assert not comp_e.healthy, "El componente E debería haber fallado en cascada debido a fallos en A"
    
    # Enviar eventos durante el estado de fallo
    for i in range(3):
        await engine.emit_event(f"failure_event_{i}", {"id": i}, "test")
        await asyncio.sleep(0.1)
    
    # Recuperar el componente A
    await engine.emit_event("dependency_status", {"component": "comp_a", "status": "recovery"}, "test")
    await asyncio.sleep(0.1)
    
    # La recuperación no es automática en todos los componentes de la cascada
    # Recuperar el componente B explícitamente
    await engine.emit_event("recovery_command", {}, "comp_b")
    await asyncio.sleep(0.1)
    
    # Recuperar el componente C explícitamente
    await engine.emit_event("recovery_command", {}, "comp_c")
    await asyncio.sleep(0.1)
    
    # Recuperar el componente D explícitamente
    await engine.emit_event("recovery_command", {}, "comp_d")
    await asyncio.sleep(0.1)
    
    # Recuperar el componente E explícitamente
    await engine.emit_event("recovery_command", {}, "comp_e")
    await asyncio.sleep(0.1)
    
    # Verificar que todos los componentes están sanos de nuevo
    for comp in [comp_a, comp_b, comp_c, comp_d, comp_e]:
        assert comp.healthy, f"El componente {comp.name} debería estar sano después de la recuperación"
    
    # Enviar eventos después de la recuperación
    for i in range(3):
        await engine.emit_event(f"recovery_event_{i}", {"id": i}, "test")
        await asyncio.sleep(0.1)
    
    # Verificar métricas finales
    for comp in [comp_a, comp_b, comp_c, comp_d, comp_e]:
        metrics = comp.get_metrics()
        logger.info(f"Métricas de {comp.name}: {metrics}")
        assert metrics["cascaded_failures"] > 0, f"El componente {comp.name} debería haber registrado fallos en cascada"
        assert metrics["processed_healthy"] >= 6, f"El componente {comp.name} debería haber procesado al menos 6 eventos en estado sano"
        assert metrics["processed_unhealthy"] > 0, f"El componente {comp.name} debería haber registrado eventos durante el estado no sano"
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
async def test_recovery_time_under_load():
    """Prueba el tiempo de recuperación del sistema bajo carga continua."""
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componentes para diferentes aspectos del sistema
    network = NetworkSimulatorComponent("network", initial_state="online")
    resource = ResourceMonitorComponent("resource", max_memory_mb=1000, max_cpu_percent=90)
    config = ConfigChangeComponent("config")
    
    # Registrar componentes
    engine.register_component(network)
    engine.register_component(resource)
    engine.register_component(config)
    
    # Iniciar motor
    await engine.start()
    
    # Función para enviar eventos de carga continua
    async def send_load_events(duration_seconds: float, interval: float = 0.05):
        end_time = time.time() + duration_seconds
        count = 0
        while time.time() < end_time:
            await engine.emit_event("load_event", {"id": count, "timestamp": time.time()}, "test")
            count += 1
            await asyncio.sleep(interval)
        return count
    
    # Enviar carga inicial para establecer una línea base
    baseline_events = await send_load_events(1.0)
    logger.info(f"Carga inicial: {baseline_events} eventos en 1 segundo")
    
    # Simular disrupción de red
    await engine.emit_event("network_control", {"state": "unstable", "latency_ms": 50, "packet_loss": 0.2}, "test")
    
    # Aumentar el uso de recursos
    await engine.emit_event("resource_control", {"memory_mb": 800, "cpu_percent": 75}, "test")
    
    # Cambiar la configuración
    await engine.emit_event("config_update", {"path": "thresholds.warning", "operation": "set", "value": 95}, "test")
    
    # Medir el rendimiento bajo condiciones adversas
    disruption_start = time.time()
    disruption_events = await send_load_events(2.0)
    disruption_duration = time.time() - disruption_start
    disruption_rate = disruption_events / disruption_duration
    
    logger.info(f"Durante la disrupción: {disruption_events} eventos en {disruption_duration:.2f} segundos")
    logger.info(f"Tasa durante la disrupción: {disruption_rate:.2f} eventos/segundo")
    
    # Iniciar recuperación
    recovery_start = time.time()
    
    # Restaurar la red
    await engine.emit_event("network_control", {"state": "online", "latency_ms": 0, "packet_loss": 0.0}, "test")
    
    # Restaurar recursos
    await engine.emit_event("resource_control", {"memory_mb": 200, "cpu_percent": 20}, "test")
    
    # Restaurar configuración
    await engine.emit_event("config_update", {"operation": "reset"}, "test")
    
    # Esperar un poco para que el sistema se estabilice
    await asyncio.sleep(0.5)
    
    # Medir el rendimiento después de la recuperación
    recovery_events = await send_load_events(1.0)
    recovery_time = time.time() - recovery_start
    recovery_rate = recovery_events / 1.0
    
    logger.info(f"Después de la recuperación: {recovery_events} eventos en 1 segundo")
    logger.info(f"Tasa después de la recuperación: {recovery_rate:.2f} eventos/segundo")
    logger.info(f"Tiempo de recuperación: {recovery_time:.2f} segundos")
    
    # Verificar que el rendimiento después de la recuperación es similar al inicial
    recovery_ratio = recovery_rate / (baseline_events / 1.0)
    logger.info(f"Ratio de recuperación: {recovery_ratio:.2f} (>0.8 es bueno)")
    
    assert recovery_ratio > 0.8, "El rendimiento después de la recuperación debería ser al menos el 80% del rendimiento inicial"
    
    # Verificar métricas finales de los componentes
    network_metrics = network.get_metrics()
    resource_metrics = resource.get_metrics()
    config_metrics = config.get_metrics()
    
    logger.info(f"Métricas de red: {network_metrics}")
    logger.info(f"Métricas de recursos: {resource_metrics}")
    logger.info(f"Métricas de configuración: {config_metrics}")
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
async def test_long_running_operation():
    """Prueba el sistema durante una operación prolongada con eventos periódicos."""
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componentes
    network = NetworkSimulatorComponent("network", initial_state="online")
    resource = ResourceMonitorComponent("resource", max_memory_mb=1000, max_cpu_percent=90)
    
    # Registrar componentes
    engine.register_component(network)
    engine.register_component(resource)
    
    # Iniciar motor
    await engine.start()
    
    # Configuración de la prueba
    duration_seconds = 5  # Duración total de la prueba
    check_interval = 0.5  # Intervalo para verificaciones periódicas
    event_interval = 0.1  # Intervalo para enviar eventos regulares
    
    # Iniciar contadores
    start_time = time.time()
    events_sent = 0
    checks_performed = 0
    
    # Ajustes aleatorios de la red y recursos
    async def perform_random_adjustment():
        # Alternar entre condiciones normales y degradadas
        if random.random() < 0.5:
            # Condiciones degradadas
            latency = random.randint(10, 100)
            packet_loss = random.random() * 0.2
            memory_increase = random.randint(10, 50)
            cpu_increase = random.randint(5, 15)
            
            await engine.emit_event("network_control", 
                                  {"state": "unstable", "latency_ms": latency, "packet_loss": packet_loss}, 
                                  "test")
            
            current_metrics = resource.get_metrics()
            await engine.emit_event("resource_control", 
                                  {"memory_mb": current_metrics["memory_mb"] + memory_increase, 
                                   "cpu_percent": current_metrics["cpu_percent"] + cpu_increase}, 
                                  "test")
            
            return "degraded"
        else:
            # Condiciones normales
            await engine.emit_event("network_control", 
                                  {"state": "online", "latency_ms": 0, "packet_loss": 0.0}, 
                                  "test")
            
            await engine.emit_event("resource_control", 
                                  {"memory_mb": 200, "cpu_percent": 20}, 
                                  "test")
            
            return "normal"
    
    # Ejecutar prueba de larga duración
    logger.info(f"Iniciando prueba de larga duración por {duration_seconds} segundos")
    
    last_check_time = start_time
    last_event_time = start_time
    conditions = []
    
    while time.time() - start_time < duration_seconds:
        current_time = time.time()
        
        # Enviar eventos regulares
        if current_time - last_event_time >= event_interval:
            await engine.emit_event("regular_event", 
                                  {"id": events_sent, "timestamp": current_time}, 
                                  "test")
            events_sent += 1
            last_event_time = current_time
        
        # Realizar verificaciones periódicas y ajustes aleatorios
        if current_time - last_check_time >= check_interval:
            condition = await perform_random_adjustment()
            conditions.append(condition)
            
            network_metrics = network.get_metrics()
            resource_metrics = resource.get_metrics()
            
            logger.info(f"Verificación #{checks_performed + 1} - Condición: {condition}")
            logger.info(f"  Red: {network_metrics['current_state']}, "
                      f"Latencia: {network_metrics['current_latency_ms']}ms, "
                      f"Pérdida: {network_metrics['current_packet_loss']:.2f}")
            logger.info(f"  Recursos: Memoria: {resource_metrics['memory_mb']}MB, "
                      f"CPU: {resource_metrics['cpu_percent']}%")
            
            checks_performed += 1
            last_check_time = current_time
        
        # Pequeña pausa para no saturar la CPU
        await asyncio.sleep(0.01)
    
    # Calcular estadísticas finales
    end_time = time.time()
    total_duration = end_time - start_time
    
    network_metrics = network.get_metrics()
    resource_metrics = resource.get_metrics()
    
    degraded_count = conditions.count("degraded")
    normal_count = conditions.count("normal")
    
    # Mostrar resultados
    logger.info(f"Prueba de larga duración completada en {total_duration:.2f} segundos")
    logger.info(f"Eventos enviados: {events_sent}")
    logger.info(f"Verificaciones realizadas: {checks_performed}")
    logger.info(f"Condiciones: Normal: {normal_count}, Degradado: {degraded_count}")
    logger.info(f"Eventos procesados en red: {network_metrics['total_processed']}")
    logger.info(f"Eventos procesados en recursos: {resource_metrics['processed_events']}")
    
    # Verificaciones finales
    assert events_sent > 0, "Deberían haberse enviado eventos"
    assert checks_performed > 0, "Deberían haberse realizado verificaciones"
    assert network_metrics['total_processed'] > 0, "La red debería haber procesado eventos"
    assert resource_metrics['processed_events'] > 0, "El monitor de recursos debería haber procesado eventos"
    
    # Detener motor
    await engine.stop()