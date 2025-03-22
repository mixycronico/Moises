"""
Pruebas unitarias para verificar la prevención de fallos en cascada.

Este módulo prueba las soluciones implementadas para prevenir fallos en cascada,
incluyendo el uso del ComponentMonitor y componentes con conciencia de dependencias.
"""

import asyncio
import logging
import pytest
import time
from typing import Dict, List, Any, Optional, Set

from genesis.core.engine_non_blocking import EngineNonBlocking
from genesis.core.component import Component
from genesis.core.component_monitor import ComponentMonitor
from tests.utils.timeout_helpers import (
    emit_with_timeout,
    safe_get_response,
    cleanup_engine
)

# Configurar logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DependentComponent(Component):
    """Componente que depende de otros componentes."""
    
    def __init__(self, name: str, dependencies: Optional[List[str]] = None):
        """
        Inicializar componente con dependencias.
        
        Args:
            name: Nombre del componente
            dependencies: Lista de dependencias (nombres de componentes)
        """
        super().__init__(name)
        self.dependencies = dependencies or []
        self.healthy = True
        self.dependency_status = {dep: True for dep in self.dependencies}
        self.event_count = 0
        self.handled_events: List[Dict[str, Any]] = []
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """
        Manejar eventos recibidos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Respuesta opcional
        """
        self.event_count += 1
        self.handled_events.append({
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": time.time()
        })
        
        # Verificación de estado
        if event_type == "check_status":
            return {
                "component": self.name,
                "healthy": self.healthy,
                "dependencies": self.dependency_status
            }
            
        # Actualización de dependencia
        elif event_type == "dependency_status_change":
            dep_id = data.get("dependency_id")
            status = data.get("status", False)
            
            if dep_id in self.dependencies:
                # Actualizar estado de la dependencia
                self.dependency_status[dep_id] = status
                
                # Actualizar estado propio basado en dependencias
                previous_health = self.healthy
                self.healthy = all(self.dependency_status.values())
                
                logger.debug(f"{self.name}: Dependencia {dep_id} cambió a {status}, salud actualizada a {self.healthy}")
                
                return {
                    "component": self.name,
                    "dependency": dep_id,
                    "dependency_status": status,
                    "healthy": self.healthy,
                    "previous_health": previous_health
                }
                
        # Establecer salud del componente (para pruebas)
        elif event_type == "set_health":
            previous = self.healthy
            self.healthy = data.get("healthy", True)
            
            logger.debug(f"{self.name}: Salud establecida a {self.healthy} (era {previous})")
            
            return {
                "component": self.name,
                "previous_health": previous,
                "current_health": self.healthy
            }
            
        # Actualización directa de dependencia (para pruebas)
        elif event_type == "dependency_update":
            dep_name = data.get("dependency")
            dep_status = data.get("status", False)
            
            if dep_name in self.dependencies:
                # Actualizar estado de la dependencia
                self.dependency_status[dep_name] = dep_status
                
                # Actualizar estado propio basado en dependencias
                previous_health = self.healthy
                self.healthy = all(self.dependency_status.values())
                
                logger.debug(f"{self.name}: Dependencia {dep_name} actualizada manualmente a {dep_status}, salud actualizada a {self.healthy}")
                
                return {
                    "component": self.name,
                    "dependency": dep_name,
                    "dependency_status": dep_status,
                    "healthy": self.healthy,
                    "previous_health": previous_health
                }
        
        return {"processed": True}

class AutoRecoveringComponent(DependentComponent):
    """Componente que se recupera automáticamente cuando sus dependencias vuelven a estar sanas."""
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """
        Manejar eventos con recuperación automática.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Respuesta opcional
        """
        # Comportamiento base
        result = await super().handle_event(event_type, data, source)
        
        # Auto-recuperación
        if (event_type == "dependency_status_change" or event_type == "dependency_update") and not self.healthy:
            # Verificar si todas las dependencias están sanas
            if all(self.dependency_status.values()):
                # Auto-recuperación
                previous = self.healthy
                self.healthy = True
                
                logger.debug(f"{self.name}: Auto-recuperación activada. Salud actualizada de {previous} a {self.healthy}")
                
                if result is None:
                    result = {}
                    
                result["auto_recovered"] = True
                result["healthy"] = self.healthy
        
        return result

@pytest.fixture
async def engine_fixture():
    """
    Fixture que proporciona un motor para pruebas.
    
    Returns:
        Motor inicializado
    """
    engine = EngineNonBlocking(test_mode=True)
    yield engine
    await cleanup_engine(engine)

@pytest.fixture
async def monitor_fixture(engine_fixture):
    """
    Fixture que proporciona un monitor de componentes.
    
    Args:
        engine_fixture: Fixture de motor
        
    Returns:
        Monitor inicializado
    """
    engine = engine_fixture
    monitor = ComponentMonitor(
        name="test_monitor",
        check_interval=0.5,
        max_failures=2,
        recovery_interval=1.0
    )
    await engine.register_component(monitor)
    await monitor.start()
    
    yield monitor

@pytest.mark.asyncio
async def test_cascading_failure_basic(engine_fixture):
    """Verificar que un fallo se propaga correctamente a sus dependientes."""
    engine = engine_fixture
    
    # Crear componentes con dependencias en cadena
    comp_a = DependentComponent("comp_a")
    comp_b = DependentComponent("comp_b", dependencies=["comp_a"])
    comp_c = DependentComponent("comp_c", dependencies=["comp_b"])
    
    # Registrar componentes
    await engine.register_component(comp_a)
    await engine.register_component(comp_b)
    await engine.register_component(comp_c)
    
    # Verificar estado inicial
    assert comp_a.healthy
    assert comp_b.healthy
    assert comp_c.healthy
    
    # Fallar componente A
    await emit_with_timeout(
        engine, "set_health", {"healthy": False}, "comp_a", timeout=1.0
    )
    
    # Notificar a B sobre fallo en A
    await emit_with_timeout(
        engine, "dependency_update", {"dependency": "comp_a", "status": False}, "comp_b", timeout=1.0
    )
    
    # Notificar a C sobre fallo en B
    await emit_with_timeout(
        engine, "dependency_update", {"dependency": "comp_b", "status": False}, "comp_c", timeout=1.0
    )
    
    # Verificar propagación
    assert not comp_a.healthy
    assert not comp_b.healthy
    assert not comp_c.healthy
    
    # Verificar que los eventos fueron recibidos
    assert any(event["type"] == "set_health" for event in comp_a.handled_events)
    assert any(event["type"] == "dependency_update" for event in comp_b.handled_events)
    assert any(event["type"] == "dependency_update" for event in comp_c.handled_events)

@pytest.mark.asyncio
async def test_cascading_failure_partial(engine_fixture):
    """Verificar que los fallos sólo afectan a componentes dependientes, no a todos."""
    engine = engine_fixture
    
    # Crear componentes con algunas dependencias
    comp_a = DependentComponent("comp_a")
    comp_b = DependentComponent("comp_b", dependencies=["comp_a"])
    comp_c = DependentComponent("comp_c")  # No depende de nadie
    comp_d = DependentComponent("comp_d", dependencies=["comp_c"])  # Depende de C, no de A o B
    
    # Registrar componentes
    await engine.register_component(comp_a)
    await engine.register_component(comp_b)
    await engine.register_component(comp_c)
    await engine.register_component(comp_d)
    
    # Verificar estado inicial
    assert comp_a.healthy
    assert comp_b.healthy
    assert comp_c.healthy
    assert comp_d.healthy
    
    # Fallar componente A
    await emit_with_timeout(
        engine, "set_health", {"healthy": False}, "comp_a", timeout=1.0
    )
    
    # Notificar a B sobre fallo en A
    await emit_with_timeout(
        engine, "dependency_update", {"dependency": "comp_a", "status": False}, "comp_b", timeout=1.0
    )
    
    # Verificar propagación parcial
    assert not comp_a.healthy  # Fallado
    assert not comp_b.healthy  # Afectado por dependencia
    assert comp_c.healthy     # No afectado, independiente
    assert comp_d.healthy     # No afectado, depende de C
    
    # Ahora fallar C para verificar propagación a D
    await emit_with_timeout(
        engine, "set_health", {"healthy": False}, "comp_c", timeout=1.0
    )
    
    # Notificar a D sobre fallo en C
    await emit_with_timeout(
        engine, "dependency_update", {"dependency": "comp_c", "status": False}, "comp_d", timeout=1.0
    )
    
    # Verificar
    assert not comp_c.healthy
    assert not comp_d.healthy
    assert not comp_a.healthy  # Sigue fallado
    assert not comp_b.healthy  # Sigue afectado

@pytest.mark.asyncio
async def test_cascading_failure_recovery(engine_fixture):
    """Verificar la recuperación automática de componentes tras la recuperación de sus dependencias."""
    engine = engine_fixture
    
    # Crear componentes con recuperación automática
    comp_a = DependentComponent("comp_a")
    comp_b = AutoRecoveringComponent("comp_b", dependencies=["comp_a"])
    comp_c = AutoRecoveringComponent("comp_c", dependencies=["comp_b"])
    
    # Registrar componentes
    await engine.register_component(comp_a)
    await engine.register_component(comp_b)
    await engine.register_component(comp_c)
    
    # Verificar estado inicial
    assert comp_a.healthy
    assert comp_b.healthy
    assert comp_c.healthy
    
    # Fallar componente A
    await emit_with_timeout(
        engine, "set_health", {"healthy": False}, "comp_a", timeout=1.0
    )
    
    # Notificar a B sobre fallo en A
    await emit_with_timeout(
        engine, "dependency_update", {"dependency": "comp_a", "status": False}, "comp_b", timeout=1.0
    )
    
    # Notificar a C sobre fallo en B
    await emit_with_timeout(
        engine, "dependency_update", {"dependency": "comp_b", "status": False}, "comp_c", timeout=1.0
    )
    
    # Verificar propagación
    assert not comp_a.healthy
    assert not comp_b.healthy
    assert not comp_c.healthy
    
    # Recuperar componente A
    await emit_with_timeout(
        engine, "set_health", {"healthy": True}, "comp_a", timeout=1.0
    )
    
    # Notificar a B sobre recuperación de A
    await emit_with_timeout(
        engine, "dependency_update", {"dependency": "comp_a", "status": True}, "comp_b", timeout=1.0
    )
    
    # Verificar auto-recuperación de B
    assert comp_a.healthy
    assert comp_b.healthy  # Auto-recuperado
    assert not comp_c.healthy  # Todavía no actualizado
    
    # Notificar a C sobre recuperación de B
    await emit_with_timeout(
        engine, "dependency_update", {"dependency": "comp_b", "status": True}, "comp_c", timeout=1.0
    )
    
    # Verificar auto-recuperación de C
    assert comp_a.healthy
    assert comp_b.healthy
    assert comp_c.healthy  # Auto-recuperado

@pytest.mark.asyncio
async def test_monitor_prevents_cascading_failures(engine_fixture, monitor_fixture):
    """Verificar que el monitor de componentes previene fallos en cascada mediante aislamiento."""
    engine = engine_fixture
    monitor = monitor_fixture
    
    # Crear componentes con dependencias en cadena
    comp_a = DependentComponent("comp_a")
    comp_b = DependentComponent("comp_b", dependencies=["comp_a"])
    comp_c = DependentComponent("comp_c", dependencies=["comp_b"])
    
    # Registrar componentes
    await engine.register_component(comp_a)
    await engine.register_component(comp_b)
    await engine.register_component(comp_c)
    
    # Esperar que el monitor registre los componentes
    await asyncio.sleep(1.0)
    
    # Verificar estado inicial
    assert comp_a.healthy
    assert comp_b.healthy
    assert comp_c.healthy
    
    # Fallar componente A
    await emit_with_timeout(
        engine, "set_health", {"healthy": False}, "comp_a", timeout=1.0
    )
    
    # Esperar a que el monitor detecte y aísle
    await asyncio.sleep(2.5)  # Más largo que max_failures * check_interval
    
    # Verificar que A está aislado
    assert "comp_a" in monitor.isolated_components
    
    # Verificar que B y C fueron notificados
    assert not comp_b.healthy
    assert not comp_c.healthy
    
    # Verificar que el monitor notificó correctamente
    assert any(
        event["type"] == "dependency_status_change" and 
        event["data"].get("dependency_id") == "comp_a" and 
        event["data"].get("status") is False
        for event in comp_b.handled_events
    )
    
    # Recuperar componente A
    await emit_with_timeout(
        engine, "set_health", {"healthy": True}, "comp_a", timeout=1.0
    )
    
    # Intentar recuperar A manualmente
    await emit_with_timeout(
        engine, "recover_component", {"component_id": "comp_a"}, 
        "test_monitor", timeout=1.0
    )
    
    # Esperar a que el monitor actualice
    await asyncio.sleep(1.0)
    
    # Verificar que A ya no está aislado
    assert "comp_a" not in monitor.isolated_components
    assert comp_a.healthy
    
    # Componentes dependientes deben ser notificados
    assert any(
        event["type"] == "dependency_status_change" and 
        event["data"].get("dependency_id") == "comp_a" and 
        event["data"].get("status") is True
        for event in comp_b.handled_events
    )

@pytest.mark.asyncio
async def test_circuit_breaker_pattern(engine_fixture, monitor_fixture):
    """Verificar que el monitor implementa correctamente el patrón Circuit Breaker."""
    engine = engine_fixture
    monitor = monitor_fixture
    
    # Crear un componente que fallará
    failing_component = DependentComponent("failing_component")
    await engine.register_component(failing_component)
    
    # Esperar que el monitor registre el componente
    await asyncio.sleep(1.0)
    
    # Ciclo 1: Fallar y recuperar
    # Fallar componente
    await emit_with_timeout(
        engine, "set_health", {"healthy": False}, "failing_component", timeout=1.0
    )
    
    # Esperar a que el monitor detecte y aísle
    await asyncio.sleep(2.5)
    
    # Verificar aislamiento (Circuit Open)
    assert "failing_component" in monitor.isolated_components
    
    # Recuperar componente
    await emit_with_timeout(
        engine, "set_health", {"healthy": True}, "failing_component", timeout=1.0
    )
    
    # Esperar a que el monitor recupere
    await asyncio.sleep(1.5)
    
    # Verificar recuperación (Circuit Closed)
    assert "failing_component" not in monitor.isolated_components
    
    # Ciclo 2: Fallar y recuperar para verificar consistencia
    # Fallar componente nuevamente
    await emit_with_timeout(
        engine, "set_health", {"healthy": False}, "failing_component", timeout=1.0
    )
    
    # Esperar a que el monitor detecte y aísle
    await asyncio.sleep(2.5)
    
    # Verificar aislamiento (Circuit Open)
    assert "failing_component" in monitor.isolated_components
    
    # Recuperar componente
    await emit_with_timeout(
        engine, "set_health", {"healthy": True}, "failing_component", timeout=1.0
    )
    
    # Esperar a que el monitor recupere
    await asyncio.sleep(1.5)
    
    # Verificar recuperación (Circuit Closed)
    assert "failing_component" not in monitor.isolated_components
    
    # Probar aislamiento manual
    await emit_with_timeout(
        engine, "isolate_component", 
        {"component_id": "failing_component", "reason": "Aislamiento manual"}, 
        "test_monitor", timeout=1.0
    )
    
    # Verificar aislamiento (Circuit Open)
    assert "failing_component" in monitor.isolated_components