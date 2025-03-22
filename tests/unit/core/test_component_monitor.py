"""
Pruebas unitarias para el monitor de componentes.

Este módulo prueba el funcionamiento del ComponentMonitor,
incluyendo la detección de componentes no saludables, aislamiento
y recuperación automática.
"""

import asyncio
import logging
import pytest
import time
from typing import Dict, Any, Optional, List

from genesis.core.engine_non_blocking import EngineNonBlocking
from genesis.core.component import Component
from genesis.core.component_monitor import ComponentMonitor
from tests.utils.timeout_helpers import safe_get_response, emit_with_timeout, cleanup_engine

# Configurar logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestComponent(Component):
    """Componente para probar el monitor."""
    
    def __init__(self, name: str, dependencies: Optional[List[str]] = None):
        """
        Inicializar componente de prueba.
        
        Args:
            name: Nombre del componente
            dependencies: Lista de dependencias (nombres de componentes)
        """
        super().__init__(name)
        self.dependencies = dependencies or []
        self.healthy = True
        self.dependency_status = {dep: True for dep in self.dependencies}
        self.event_count = 0
        self.last_event = None
        self.respond_to_checks = True
        
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
        self.last_event = (event_type, data, source)
        
        # Verificación de estado
        if event_type == "check_status":
            if not self.respond_to_checks:
                # Simular un componente bloqueado que no responde
                await asyncio.sleep(10.0)
                
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
            
            return {
                "component": self.name,
                "previous_health": previous,
                "current_health": self.healthy
            }
            
        # Configurar si responde a verificaciones
        elif event_type == "set_response_behavior":
            self.respond_to_checks = data.get("respond", True)
            
            return {
                "component": self.name,
                "respond_to_checks": self.respond_to_checks
            }
        
        return {"processed": True}

class BlockingComponent(Component):
    """Componente que se bloquea intencionalmente."""
    
    def __init__(self, name: str):
        """
        Inicializar componente que se bloquea.
        
        Args:
            name: Nombre del componente
        """
        super().__init__(name)
        self.blocking = False
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """
        Manejar eventos, bloqueándose si está configurado para hacerlo.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Respuesta opcional
        """
        # Activar/desactivar bloqueo
        if event_type == "set_blocking":
            self.blocking = data.get("blocking", False)
            return {"component": self.name, "blocking": self.blocking}
            
        # Si está en modo bloqueo, se bloquea indefinidamente
        if self.blocking and event_type == "check_status":
            await asyncio.sleep(10)  # Simular bloqueo largo
            
        # Respuesta normal
        if event_type == "check_status":
            return {"component": self.name, "healthy": not self.blocking}
            
        return {"processed": True}

@pytest.fixture
async def engine_with_monitor():
    """
    Fixture que proporciona un motor con monitor de componentes.
    
    Returns:
        Tupla (engine, monitor)
    """
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear monitor con intervalos cortos para pruebas
    monitor = ComponentMonitor(
        name="test_monitor",
        check_interval=0.5,  # Intervalos cortos para pruebas
        max_failures=2,      # Solo 2 fallos para aislar
        recovery_interval=1.0  # Intervalo de recuperación corto
    )
    
    # Registrar monitor
    await engine.register_component(monitor)
    
    # Iniciar el monitor
    await monitor.start()
    
    yield engine, monitor
    
    # Limpiar recursos
    await cleanup_engine(engine)

@pytest.mark.asyncio
async def test_monitor_initialization(engine_with_monitor):
    """Probar que el monitor se inicializa correctamente."""
    engine, monitor = engine_with_monitor
    
    # Verificar que el monitor está registrado
    assert "test_monitor" in engine.components
    
    # Verificar estado inicial
    assert monitor.running
    assert not monitor.isolated_components
    assert not monitor.health_status
    assert not monitor.failure_counts
    
    # Verificar que responde a verificaciones de estado
    response = await emit_with_timeout(
        engine, "check_status", {}, "test_monitor", timeout=1.0
    )
    
    # Verificar respuesta
    assert safe_get_response(response, "component") == "test_monitor"
    assert safe_get_response(response, "healthy") is True

@pytest.mark.asyncio
async def test_monitor_detects_healthy_component(engine_with_monitor):
    """Probar que el monitor detecta un componente saludable."""
    engine, monitor = engine_with_monitor
    
    # Crear y registrar componente saludable
    comp_a = TestComponent("comp_a")
    await engine.register_component(comp_a)
    
    # Esperar a que el monitor verifique el componente
    await asyncio.sleep(1.0)
    
    # Verificar que el monitor conoce el estado del componente
    assert "comp_a" in monitor.health_status
    assert monitor.health_status["comp_a"] is True
    assert "comp_a" not in monitor.isolated_components

@pytest.mark.asyncio
async def test_monitor_detects_unhealthy_component(engine_with_monitor):
    """Probar que el monitor detecta un componente no saludable."""
    engine, monitor = engine_with_monitor
    
    # Crear y registrar componente
    comp_a = TestComponent("comp_a")
    await engine.register_component(comp_a)
    
    # Establecer como no saludable
    await emit_with_timeout(
        engine, "set_health", {"healthy": False}, "comp_a", timeout=1.0
    )
    
    # Esperar a que el monitor verifique el componente varias veces
    await asyncio.sleep(2.0)
    
    # Verificar que el monitor detecta que no está saludable
    assert "comp_a" in monitor.health_status
    assert monitor.health_status["comp_a"] is False
    assert monitor.failure_counts["comp_a"] >= 1

@pytest.mark.asyncio
async def test_monitor_isolates_component_after_max_failures(engine_with_monitor):
    """Probar que el monitor aísla un componente después de varios fallos."""
    engine, monitor = engine_with_monitor
    
    # Crear y registrar componente
    comp_a = TestComponent("comp_a")
    await engine.register_component(comp_a)
    
    # Establecer como no saludable
    await emit_with_timeout(
        engine, "set_health", {"healthy": False}, "comp_a", timeout=1.0
    )
    
    # Esperar a que el monitor verifique el componente varias veces
    # (suficientes para alcanzar max_failures)
    await asyncio.sleep(2.5)  # Más largo que max_failures * check_interval
    
    # Verificar que el componente está aislado
    assert "comp_a" in monitor.isolated_components
    assert "comp_a" in monitor.health_status
    assert monitor.health_status["comp_a"] is False
    assert monitor.failure_counts["comp_a"] >= monitor.max_failures

@pytest.mark.asyncio
async def test_monitor_handles_unresponsive_component(engine_with_monitor):
    """Probar que el monitor maneja componentes que no responden (timeout)."""
    engine, monitor = engine_with_monitor
    
    # Crear y registrar componente que no responderá
    comp_a = TestComponent("comp_a")
    await engine.register_component(comp_a)
    
    # Configurar para que no responda a verificaciones
    await emit_with_timeout(
        engine, "set_response_behavior", {"respond": False}, "comp_a", timeout=1.0
    )
    
    # Esperar a que el monitor verifique e intente aislar
    await asyncio.sleep(2.5)  # Más largo que max_failures * check_interval
    
    # Verificar que el componente está aislado por no responder
    assert "comp_a" in monitor.isolated_components
    assert "comp_a" in monitor.health_status
    assert monitor.health_status["comp_a"] is False
    assert monitor.failure_counts["comp_a"] >= monitor.max_failures

@pytest.mark.asyncio
async def test_monitor_recovers_isolated_component(engine_with_monitor):
    """Probar que el monitor recupera un componente aislado cuando vuelve a estar saludable."""
    engine, monitor = engine_with_monitor
    
    # Crear y registrar componente
    comp_a = TestComponent("comp_a")
    await engine.register_component(comp_a)
    
    # Establecer como no saludable para que sea aislado
    await emit_with_timeout(
        engine, "set_health", {"healthy": False}, "comp_a", timeout=1.0
    )
    
    # Esperar a que el componente sea aislado
    await asyncio.sleep(2.5)
    assert "comp_a" in monitor.isolated_components
    
    # Ahora establecer como saludable
    await emit_with_timeout(
        engine, "set_health", {"healthy": True}, "comp_a", timeout=1.0
    )
    
    # Esperar a que el monitor intente recuperar el componente
    await asyncio.sleep(1.5)  # Más largo que recovery_interval
    
    # Verificar que el componente ya no está aislado
    assert "comp_a" not in monitor.isolated_components
    assert "comp_a" in monitor.health_status
    assert monitor.health_status["comp_a"] is True
    assert monitor.failure_counts["comp_a"] == 0

@pytest.mark.asyncio
async def test_monitor_isolates_blocking_component(engine_with_monitor):
    """Probar que el monitor aísla un componente que se bloquea."""
    engine, monitor = engine_with_monitor
    
    # Crear y registrar un componente que se bloquea
    blocker = BlockingComponent("blocker")
    await engine.register_component(blocker)
    
    # Configurar para que se bloquee
    await emit_with_timeout(
        engine, "set_blocking", {"blocking": True}, "blocker", timeout=1.0
    )
    
    # Esperar a que el monitor lo detecte e intente aislarlo
    await asyncio.sleep(3.0)
    
    # Verificar que el componente está aislado
    assert "blocker" in monitor.isolated_components
    assert "blocker" in monitor.health_status
    assert monitor.health_status["blocker"] is False

@pytest.mark.asyncio
async def test_monitor_notifies_dependencies(engine_with_monitor):
    """Probar que el monitor notifica a los componentes dependientes cuando un componente es aislado."""
    engine, monitor = engine_with_monitor
    
    # Crear componentes con dependencias
    comp_a = TestComponent("comp_a")
    comp_b = TestComponent("comp_b", dependencies=["comp_a"])
    
    # Registrar componentes
    await engine.register_component(comp_a)
    await engine.register_component(comp_b)
    
    # Esperar que el monitor registre los componentes
    await asyncio.sleep(1.0)
    
    # Establecer comp_a como no saludable para que sea aislado
    await emit_with_timeout(
        engine, "set_health", {"healthy": False}, "comp_a", timeout=1.0
    )
    
    # Esperar a que sea aislado y notifique
    await asyncio.sleep(2.5)
    
    # Verificar que comp_a está aislado
    assert "comp_a" in monitor.isolated_components
    
    # Verificar que comp_b ha sido notificado
    # (puede tardar un poco, así que esperamos y comprobamos)
    max_attempts = 5
    for _ in range(max_attempts):
        status = await emit_with_timeout(
            engine, "check_status", {}, "comp_b", timeout=1.0
        )
        
        # Verificar si comp_b ha actualizado su estado
        dep_status = safe_get_response(status, "dependencies", {})
        if dep_status.get("comp_a") is False:
            break
            
        await asyncio.sleep(0.5)
    else:
        pytest.fail("comp_b no fue notificado del aislamiento de comp_a")
    
    # Verificar que comp_b también está no saludable
    status = await emit_with_timeout(
        engine, "check_status", {}, "comp_b", timeout=1.0
    )
    assert safe_get_response(status, "healthy") is False

@pytest.mark.asyncio
async def test_monitor_health_report(engine_with_monitor):
    """Probar que el monitor genera informes de salud completos."""
    engine, monitor = engine_with_monitor
    
    # Crear varios componentes
    comp_a = TestComponent("comp_a")
    comp_b = TestComponent("comp_b", dependencies=["comp_a"])
    comp_c = TestComponent("comp_c")
    
    # Registrar componentes
    await engine.register_component(comp_a)
    await engine.register_component(comp_b)
    await engine.register_component(comp_c)
    
    # Esperar que el monitor registre los componentes
    await asyncio.sleep(1.0)
    
    # Establecer comp_a como no saludable
    await emit_with_timeout(
        engine, "set_health", {"healthy": False}, "comp_a", timeout=1.0
    )
    
    # Esperar a que sea detectado
    await asyncio.sleep(1.0)
    
    # Solicitar informe de salud
    report = await emit_with_timeout(
        engine, "get_health_report", {}, "test_monitor", timeout=1.0
    )
    
    # Verificar que el informe contiene información completa
    assert safe_get_response(report, "monitor") == "test_monitor"
    assert isinstance(safe_get_response(report, "timestamp"), float)
    assert isinstance(safe_get_response(report, "component_status"), dict)
    assert "comp_a" in safe_get_response(report, "component_status", {})
    assert "comp_b" in safe_get_response(report, "component_status", {})
    assert "comp_c" in safe_get_response(report, "component_status", {})
    
    # Verificar contadores de fallos
    failure_counts = safe_get_response(report, "failure_counts", {})
    assert "comp_a" in failure_counts
    assert failure_counts["comp_a"] > 0
    
    # Esperar a que comp_a sea aislado
    await asyncio.sleep(1.5)
    
    # Solicitar nuevo informe para verificar aislamiento
    report = await emit_with_timeout(
        engine, "get_health_report", {}, "test_monitor", timeout=1.0
    )
    
    # Verificar componentes aislados
    isolated = safe_get_response(report, "isolated_components", [])
    assert "comp_a" in isolated

@pytest.mark.asyncio
async def test_manual_component_operations(engine_with_monitor):
    """Probar operaciones manuales sobre componentes (aislamiento, recuperación)."""
    engine, monitor = engine_with_monitor
    
    # Crear componente
    comp_a = TestComponent("comp_a")
    await engine.register_component(comp_a)
    
    # Esperar que el monitor registre el componente
    await asyncio.sleep(1.0)
    
    # Aislar manualmente
    isolation_response = await emit_with_timeout(
        engine, "isolate_component", {"component_id": "comp_a", "reason": "Aislamiento manual"}, 
        "test_monitor", timeout=1.0
    )
    
    # Verificar respuesta
    assert safe_get_response(isolation_response, "component_id") == "comp_a"
    assert safe_get_response(isolation_response, "isolated") is True
    
    # Verificar que está aislado
    assert "comp_a" in monitor.isolated_components
    
    # Intentar recuperar manualmente
    recovery_response = await emit_with_timeout(
        engine, "recover_component", {"component_id": "comp_a"}, 
        "test_monitor", timeout=1.0
    )
    
    # Verificar respuesta
    assert safe_get_response(recovery_response, "component_id") == "comp_a"
    assert safe_get_response(recovery_response, "recovered") is True
    
    # Verificar que ya no está aislado
    assert "comp_a" not in monitor.isolated_components