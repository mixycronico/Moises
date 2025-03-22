"""
Tests para el monitor de componentes optimizado.

Este módulo prueba el funcionamiento del monitor de componentes optimizado
que utiliza colas dedicadas para evitar deadlocks y gestionar componentes problemáticos.
"""

import asyncio
import logging
import pytest
import time
from typing import Dict, Any, List, Optional

from genesis.core.base import Component
from genesis.core.component_monitor_optimized import OptimizedComponentMonitor
from genesis.core.engine_dedicated_queues import DedicatedQueueEngine

# Configurar logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestComponent(Component):
    """Componente para pruebas del monitor optimizado."""
    
    def __init__(self, name: str, always_healthy: bool = True):
        """
        Inicializar componente de prueba.
        
        Args:
            name: Nombre del componente
            always_healthy: Si es False, el componente reportará que no está saludable
        """
        super().__init__(name)
        self.events_received = []
        self.always_healthy = always_healthy
        self.dependencies = []
        self.should_timeout = False
        
    async def start(self) -> None:
        """Iniciar componente."""
        self.running = True
        logger.debug(f"Componente {self.name} iniciado")
        
    async def stop(self) -> None:
        """Detener componente."""
        self.running = False
        logger.debug(f"Componente {self.name} detenido")
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """
        Manejar eventos recibidos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Origen del evento
            
        Returns:
            Respuesta opcional al evento
        """
        # Registrar evento
        self.events_received.append((event_type, data, source))
        logger.debug(f"Componente {self.name} recibió evento {event_type} de {source}")
        
        # Simular timeout si está configurado
        if self.should_timeout:
            logger.debug(f"Componente {self.name} simulando timeout")
            await asyncio.sleep(10.0)  # Dormir por tiempo largo para forzar timeout
        
        # Responder según el tipo de evento
        if event_type == "check_status":
            return {
                "component": self.name,
                "healthy": self.always_healthy
            }
            
        elif event_type == "dependency.status_changed":
            # Actualizar estado de dependencia
            component_id = data.get("component_id")
            status = data.get("status", False)
            if component_id in self.dependencies:
                logger.debug(f"Componente {self.name} actualizando estado de dependencia {component_id} a {status}")
            return {
                "component": self.name,
                "dependency_updated": True,
                "component_id": component_id,
                "status": status
            }
            
        elif event_type == "set_health":
            self.always_healthy = data.get("healthy", True)
            return {
                "component": self.name,
                "health_updated": True,
                "healthy": self.always_healthy
            }
            
        elif event_type == "set_timeout":
            self.should_timeout = data.get("timeout", False)
            return {
                "component": self.name,
                "timeout_updated": True,
                "should_timeout": self.should_timeout
            }
            
        # Para otros eventos
        return {
            "component": self.name,
            "processed": True
        }
        
    def add_dependency(self, component_id: str) -> None:
        """
        Añadir una dependencia a este componente.
        
        Args:
            component_id: ID del componente del que depende
        """
        if component_id not in self.dependencies:
            self.dependencies.append(component_id)

@pytest.fixture
async def optimized_engine():
    """Fixture que proporciona un motor con colas dedicadas."""
    # Crear motor en modo test
    engine = DedicatedQueueEngine(test_mode=True)
    
    yield engine
    
    # Limpieza
    if engine.running:
        await engine.stop()

@pytest.fixture
async def optimized_monitor_setup():
    """Fixture que proporciona un motor con monitor optimizado."""
    # Crear motor en modo test
    engine = DedicatedQueueEngine(test_mode=True)
    
    # Crear monitor optimizado con intervalos pequeños para pruebas
    monitor = OptimizedComponentMonitor(
        name="test_monitor",
        check_interval=0.2,
        max_failures=2,
        recovery_interval=0.3,
        test_mode=True
    )
    
    # Registrar y empezar a usar el motor
    await engine.register_component(monitor)
    await engine.start()
    
    yield engine, monitor
    
    # Limpieza
    if engine.running:
        await engine.stop()

@pytest.mark.asyncio
async def test_optimized_monitor_initialization(optimized_monitor_setup):
    """Probar inicialización del monitor optimizado."""
    engine, monitor = optimized_monitor_setup
    
    # Verificar que el monitor está registrado
    assert monitor.name in engine.components
    
    # Verificar estado inicial
    assert monitor.running
    assert not monitor.isolated_components
    assert not monitor.health_status
    assert isinstance(monitor.component_metadata, dict)

@pytest.mark.asyncio
async def test_optimized_check_healthy_component(optimized_monitor_setup):
    """Probar verificación de un componente saludable."""
    engine, monitor = optimized_monitor_setup
    
    # Crear componente saludable
    comp_a = TestComponent("comp_a", always_healthy=True)
    
    # Registrar en el motor
    await engine.register_component(comp_a)
    
    # Verificar componente
    result = await monitor._check_component_health("comp_a")
    
    # Verificar resultado
    assert result["component_id"] == "comp_a"
    assert result["healthy"] is True
    assert "comp_a" in monitor.health_status
    assert monitor.health_status["comp_a"] is True
    assert "comp_a" in monitor.failure_counts
    assert monitor.failure_counts["comp_a"] == 0

@pytest.mark.asyncio
async def test_optimized_check_unhealthy_component(optimized_monitor_setup):
    """Probar verificación de un componente no saludable."""
    engine, monitor = optimized_monitor_setup
    
    # Crear componente no saludable
    comp_b = TestComponent("comp_b", always_healthy=False)
    
    # Registrar en el motor
    await engine.register_component(comp_b)
    
    # Verificar componente
    result = await monitor._check_component_health("comp_b")
    
    # Verificar resultado
    assert result["component_id"] == "comp_b"
    assert result["healthy"] is False
    assert "comp_b" in monitor.health_status
    assert monitor.health_status["comp_b"] is False
    assert "comp_b" in monitor.failure_counts
    assert monitor.failure_counts["comp_b"] > 0

@pytest.mark.asyncio
async def test_optimized_isolation_after_failures(optimized_monitor_setup):
    """Probar aislamiento automático después de varios fallos."""
    engine, monitor = optimized_monitor_setup
    
    # Crear componente no saludable
    comp_c = TestComponent("comp_c", always_healthy=False)
    
    # Registrar en el motor
    await engine.register_component(comp_c)
    
    # Verificar componente varias veces
    for _ in range(monitor.max_failures + 1):
        await monitor._check_component_health("comp_c")
        # Esperar para que se procese
        await asyncio.sleep(0.1)
    
    # Verificar que fue aislado
    assert "comp_c" in monitor.isolated_components

@pytest.mark.asyncio
async def test_optimized_manual_isolation(optimized_monitor_setup):
    """Probar aislamiento manual de un componente."""
    engine, monitor = optimized_monitor_setup
    
    # Crear componente
    comp_d = TestComponent("comp_d", always_healthy=True)
    
    # Registrar en el motor
    await engine.register_component(comp_d)
    
    # Aislar manualmente
    await monitor._isolate_component("comp_d", "Aislamiento manual para prueba")
    
    # Verificar aislamiento
    assert "comp_d" in monitor.isolated_components
    assert "comp_d" in monitor.component_metadata
    assert "isolation_reason" in monitor.component_metadata["comp_d"]
    assert monitor.component_metadata["comp_d"]["isolation_reason"] == "Aislamiento manual para prueba"

@pytest.mark.asyncio
async def test_optimized_dependency_notification(optimized_monitor_setup):
    """Probar notificación de dependencias entre componentes."""
    engine, monitor = optimized_monitor_setup
    
    # Crear componentes con dependencia
    comp_e = TestComponent("comp_e", always_healthy=True)
    comp_f = TestComponent("comp_f", always_healthy=True)
    
    # Establecer dependencia
    comp_f.add_dependency("comp_e")
    
    # Registrar en el motor
    await engine.register_component(comp_e)
    await engine.register_component(comp_f)
    
    # Aislar componente del que depende otro
    await monitor._isolate_component("comp_e", "Aislamiento para probar dependencias")
    
    # Esperar para que se procese
    await asyncio.sleep(0.2)
    
    # Verificar que el componente dependiente recibió la notificación
    dependency_events = [
        event for event in comp_f.events_received 
        if event[0] == "dependency.status_changed" and event[1].get("component_id") == "comp_e"
    ]
    
    assert len(dependency_events) > 0
    assert dependency_events[0][1]["status"] is False

@pytest.mark.asyncio
async def test_optimized_recovery(optimized_monitor_setup):
    """Probar recuperación de un componente aislado."""
    engine, monitor = optimized_monitor_setup
    
    # Crear componente inicialmente no saludable
    comp_g = TestComponent("comp_g", always_healthy=False)
    
    # Registrar en el motor
    await engine.register_component(comp_g)
    
    # Aislar manualmente
    await monitor._isolate_component("comp_g", "Aislamiento para prueba de recuperación")
    
    # Verificar aislamiento
    assert "comp_g" in monitor.isolated_components
    
    # Hacer que el componente sea saludable
    comp_g.always_healthy = True
    
    # Intentar recuperar
    success = await monitor._attempt_recovery("comp_g")
    
    # Verificar resultado
    assert success is True
    assert "comp_g" not in monitor.isolated_components
    assert "comp_g" in monitor.component_metadata
    assert "recovery_time" in monitor.component_metadata["comp_g"]

@pytest.mark.asyncio
async def test_optimized_monitor_handle_events(optimized_monitor_setup):
    """Probar API del monitor a través de eventos."""
    engine, monitor = optimized_monitor_setup
    
    # Crear componente
    comp_h = TestComponent("comp_h", always_healthy=True)
    
    # Registrar en el motor
    await engine.register_component(comp_h)
    
    # Verificar API de verificación de componente
    response = await monitor.handle_event(
        "check_component",
        {"component_id": "comp_h"},
        "test"
    )
    
    assert response["component_id"] == "comp_h"
    assert response["healthy"] is True
    
    # Probar API de aislamiento
    response = await monitor.handle_event(
        "isolate_component",
        {"component_id": "comp_h", "reason": "Prueba de API"},
        "test"
    )
    
    assert response["component_id"] == "comp_h"
    assert response["isolated"] is True
    assert "comp_h" in monitor.isolated_components
    
    # Probar API de recuperación
    comp_h.always_healthy = True
    
    response = await monitor.handle_event(
        "recover_component",
        {"component_id": "comp_h"},
        "test"
    )
    
    assert response["component_id"] == "comp_h"
    assert response["recovered"] is True
    assert "comp_h" not in monitor.isolated_components
    
    # Probar API de informe de salud
    response = await monitor.handle_event(
        "get_health_report",
        {},
        "test"
    )
    
    assert "component_status" in response
    assert "isolated_components" in response
    assert "component_metadata" in response
    assert "comp_h" in response["component_status"]