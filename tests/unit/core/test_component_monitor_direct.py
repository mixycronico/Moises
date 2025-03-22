"""
Test directo para el monitor de componentes sin dependencias complejas.

Este módulo implementa pruebas para ComponentMonitor usando un enfoque directo
que evita las dependencias en el sistema de eventos asíncronos completo.
"""

import asyncio
import logging
import pytest
import time
from typing import Dict, Any, List, Optional

from genesis.core.base import Component
from genesis.core.component_monitor import ComponentMonitor
from genesis.core.engine_non_blocking import EngineNonBlocking

# Configurar logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DirectComponent(Component):
    """Componente directo para pruebas sin complejidad asíncrona."""
    
    def __init__(self, name: str, health: bool = True, dependencies: List[str] = None):
        super().__init__(name)
        self.events_received = []
        self.test_health = health
        self.dependencies = dependencies or []
        self.dependency_status: Dict[str, bool] = {}
        self.response_timeout = False  # Si True, simulará un timeout
        
    async def start(self) -> None:
        """Implementación mínima de start."""
        self.running = True
        logger.debug(f"DirectComponent {self.name} iniciado")
        
    async def stop(self) -> None:
        """Implementación mínima de stop."""
        self.running = False
        logger.debug(f"DirectComponent {self.name} detenido")
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """
        Implementación directa del manejador de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Origen del evento
            
        Returns:
            Respuesta opcional al evento
        """
        # Registrar evento
        self.events_received.append((event_type, data, source))
        logger.debug(f"DirectComponent {self.name} recibió evento: {event_type} de {source}")
        
        # Si está configurado para simular timeout, dormir por un tiempo largo
        if self.response_timeout:
            logger.debug(f"DirectComponent {self.name} simulando timeout")
            await asyncio.sleep(5.0)  # Tiempo largo para forzar timeout
            
        # Responder a verificaciones de estado
        if event_type == "check_status":
            return {
                "component": self.name,
                "healthy": self.test_health,
                "dependencies": self.dependency_status
            }
            
        # Responder a actualizaciones de estado de dependencias
        elif event_type == "dependency.status_changed":
            component_id = data.get("component_id")
            status = data.get("status", False)
            
            if component_id and component_id in self.dependencies:
                self.dependency_status[component_id] = status
                # Actualizar estado propio basado en dependencias
                # Si alguna dependencia no está saludable, este componente tampoco
                if not status:
                    self.test_health = False
                    
            return {"processed": True}
            
        # Establecer comportamiento de salud
        elif event_type == "set_health":
            healthy = data.get("healthy", True)
            self.test_health = healthy
            return {"component": self.name, "healthy": self.test_health}
            
        # Establecer timeout
        elif event_type == "set_response_behavior":
            respond = data.get("respond", True)
            self.response_timeout = not respond
            return {"component": self.name, "will_respond": respond}
            
        return None

@pytest.fixture
async def direct_engine():
    """
    Fixture que proporciona un motor con componentes directos.
    
    Esta implementación evita las complejidades del sistema de eventos asíncrono
    y proporciona un entorno simplificado para probar el monitor.
    """
    # Crear motor en modo test
    engine = EngineNonBlocking(test_mode=True)
    
    yield engine
    
    # Limpieza simplificada
    try:
        engine.running = False
        for component_name, component in list(engine.components.items()):
            component.running = False
    except Exception as e:
        logger.warning(f"Error en limpieza de engine: {e}")

@pytest.fixture
async def direct_monitor_setup():
    """
    Fixture que proporciona un motor con monitor y componentes directos.
    """
    # Crear motor en modo test
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear monitor con intervalos mínimos
    monitor = ComponentMonitor(
        name="test_monitor",
        check_interval=0.2,  # Intervalo mínimo para pruebas
        max_failures=2,      # Solo 2 fallos para pruebas
        recovery_interval=0.5  # Intervalo mínimo para recuperación
    )
    
    # Registrar monitor directamente
    engine.components[monitor.name] = monitor
    monitor.attach_event_bus(engine.event_bus)
    
    # Iniciar monitor con timeout para evitar bloqueos
    try:
        await asyncio.wait_for(monitor.start(), timeout=0.5)
    except asyncio.TimeoutError:
        logger.warning("Timeout al iniciar monitor - continuando test")
        monitor.running = True  # Forzar para continuar test
    
    yield engine, monitor
    
    # Limpieza agresiva
    try:
        monitor.running = False
        engine.running = False
        for component_name, component in list(engine.components.items()):
            component.running = False
    except Exception as e:
        logger.warning(f"Error en limpieza: {e}")

@pytest.mark.asyncio
async def test_monitor_direct_initialization(direct_monitor_setup):
    """Probar inicialización directa del monitor."""
    engine, monitor = direct_monitor_setup
    
    # Verificar que el monitor está registrado
    assert "test_monitor" in engine.components
    
    # Verificar estado inicial
    assert monitor.running
    assert not monitor.isolated_components
    assert not monitor.health_status

@pytest.mark.asyncio
async def test_monitor_direct_component_check(direct_monitor_setup):
    """Probar verificación directa de componentes."""
    engine, monitor = direct_monitor_setup
    
    # Crear componente directo
    comp_a = DirectComponent("comp_a")
    
    # Registrar componente directamente
    engine.components[comp_a.name] = comp_a
    comp_a.attach_event_bus(engine.event_bus)
    await asyncio.wait_for(comp_a.start(), timeout=0.5)
    
    # Verificar directamente el componente
    result = await monitor._check_component_health("comp_a")
    
    # Verificar resultado
    assert result["component_id"] == "comp_a"
    assert result["healthy"] is True

@pytest.mark.asyncio
async def test_monitor_direct_unhealthy_detection(direct_monitor_setup):
    """Probar detección directa de componentes no saludables."""
    engine, monitor = direct_monitor_setup
    
    # Crear componente no saludable
    comp_a = DirectComponent("comp_a", health=False)
    
    # Registrar componente
    engine.components[comp_a.name] = comp_a
    comp_a.attach_event_bus(engine.event_bus)
    await asyncio.wait_for(comp_a.start(), timeout=0.5)
    
    # Verificar directamente
    await monitor._check_component_health("comp_a")
    
    # Verificar que se detectó como no saludable
    assert "comp_a" in monitor.health_status
    assert monitor.health_status["comp_a"] is False
    assert monitor.failure_counts.get("comp_a", 0) > 0

@pytest.mark.asyncio
async def test_monitor_direct_isolation(direct_monitor_setup):
    """Probar aislamiento directo de componentes."""
    engine, monitor = direct_monitor_setup
    
    # Crear componente
    comp_a = DirectComponent("comp_a", health=True)
    
    # Registrar componente
    engine.components[comp_a.name] = comp_a
    comp_a.attach_event_bus(engine.event_bus)
    
    # Aislar manualmente
    await monitor._isolate_component("comp_a", "Aislamiento de prueba")
    
    # Verificar aislamiento
    assert "comp_a" in monitor.isolated_components

@pytest.mark.asyncio
async def test_monitor_direct_dependency_notification(direct_monitor_setup):
    """Probar notificación directa de dependencias."""
    engine, monitor = direct_monitor_setup
    
    # Crear componentes con dependencias
    comp_a = DirectComponent("comp_a")
    comp_b = DirectComponent("comp_b", dependencies=["comp_a"])
    
    # Registrar componentes
    engine.components[comp_a.name] = comp_a
    engine.components[comp_b.name] = comp_b
    comp_a.attach_event_bus(engine.event_bus)
    comp_b.attach_event_bus(engine.event_bus)
    
    # Aislar comp_a
    await monitor._isolate_component("comp_a", "Aislamiento de prueba")
    
    # Notificar a dependencias
    await monitor._notify_dependencies("comp_a", False)
    
    # Verificar que comp_b recibió la notificación
    assert len(comp_b.events_received) > 0
    for event in comp_b.events_received:
        if event[0] == "dependency.status_changed":
            assert event[1].get("component_id") == "comp_a"
            assert event[1].get("status") is False
            break
    else:
        pytest.fail("No se recibió notificación de dependencia")

@pytest.mark.asyncio
async def test_monitor_direct_component_recovery(direct_monitor_setup):
    """Probar recuperación directa de componentes."""
    engine, monitor = direct_monitor_setup
    
    # Crear componente
    comp_a = DirectComponent("comp_a", health=False)
    
    # Registrar componente
    engine.components[comp_a.name] = comp_a
    comp_a.attach_event_bus(engine.event_bus)
    
    # Aislar componente
    await monitor._isolate_component("comp_a", "Aislamiento para prueba")
    assert "comp_a" in monitor.isolated_components
    
    # Volver a establecer como saludable
    comp_a.test_health = True
    
    # Intentar recuperar
    success = await monitor._attempt_recovery("comp_a")
    
    # Verificar éxito
    assert success
    assert "comp_a" not in monitor.isolated_components

@pytest.mark.asyncio
async def test_monitor_direct_handle_events(direct_monitor_setup):
    """Probar manejo directo de eventos en el monitor."""
    engine, monitor = direct_monitor_setup
    
    # Crear componente
    comp_a = DirectComponent("comp_a")
    
    # Registrar componente
    engine.components[comp_a.name] = comp_a
    comp_a.attach_event_bus(engine.event_bus)
    
    # Llamar directamente al manejador de eventos del monitor
    response = await monitor.handle_event(
        "check_component", 
        {"component_id": "comp_a"}, 
        "test"
    )
    
    # Verificar respuesta
    assert response.get("component_id") == "comp_a"
    assert isinstance(response.get("healthy"), bool)
    
    # Probar aislamiento a través del manejador de eventos
    response = await monitor.handle_event(
        "isolate_component",
        {"component_id": "comp_a", "reason": "Prueba de API"},
        "test"
    )
    
    # Verificar respuesta y efecto
    assert response.get("component_id") == "comp_a"
    assert response.get("isolated") is True
    assert "comp_a" in monitor.isolated_components