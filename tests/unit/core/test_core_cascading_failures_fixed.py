"""
Pruebas para validar la correcta implementación de mecanismos de prevención
de fallos en cascada en el sistema Genesis.

Este módulo implementa pruebas que verifican que los componentes manejen
correctamente fallos en cascada, utilizando las técnicas de manejo defensivo
de respuestas y timeouts para garantizar resultados consistentes.
"""

import pytest
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional

# Importar utilidades para timeout
from tests.utils.timeout_helpers import (
    emit_with_timeout,
    check_component_status,
    run_test_with_timing,
    safe_get_response,
    cleanup_engine
)

# Importar clases necesarias
from genesis.core.engine_non_blocking import EngineNonBlocking
from genesis.core.component import Component

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DependentComponent(Component):
    """
    Componente que depende de otros componentes y responde a sus cambios de estado.
    
    Este componente implementa el patrón de fallo en cascada, donde un componente
    falla automáticamente cuando sus dependencias no están saludables.
    """
    
    def __init__(self, name: str, dependencies: Optional[List[str]] = None):
        """
        Inicializar componente con dependencias.
        
        Args:
            name: Nombre del componente
            dependencies: Lista de nombres de componentes de los que depende
        """
        super().__init__(name)
        self.dependencies = dependencies or []
        self.healthy = True
        self.dependency_status = {dep: True for dep in self.dependencies}
        
    async def start(self) -> None:
        """Iniciar el componente."""
        logger.debug(f"Componente {self.name} iniciado")
        
    async def stop(self) -> None:
        """Detener el componente."""
        logger.debug(f"Componente {self.name} detenido")
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """
        Manejar eventos para este componente.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Respuesta al evento, si corresponde
        """
        # Verificación de estado
        if event_type == "check_status":
            return {
                "component": self.name,
                "healthy": self.healthy,
                "dependencies": self.dependency_status
            }
            
        # Cambiar estado de salud
        elif event_type == "set_health":
            previous = self.healthy
            self.healthy = data.get("healthy", True)
            logger.info(f"Componente {self.name}: salud cambiada de {previous} a {self.healthy}")
            return {"component": self.name, "healthy": self.healthy, "previous": previous}
            
        # Notificación de cambio de estado de dependencia
        elif event_type == "dependency_update":
            dep_name = data.get("dependency")
            dep_status = data.get("status")
            
            if dep_name in self.dependencies:
                # Actualizar estado de la dependencia
                self.dependency_status[dep_name] = dep_status
                
                # Actualizar estado propio basado en dependencias
                previous_health = self.healthy
                self.healthy = all(self.dependency_status.values())
                
                logger.info(f"Componente {self.name}: actualizado estado de dependencia {dep_name} a {dep_status}; "
                           f"salud propia cambiada de {previous_health} a {self.healthy}")
                
                return {
                    "component": self.name,
                    "dependency": dep_name,
                    "dependency_status": dep_status,
                    "healthy": self.healthy,
                    "previous_health": previous_health
                }
        
        # Para otros eventos, respuesta genérica
        return {"component": self.name, "processed": True, "event_type": event_type}

# Fixture para motor con limpieza adecuada
@pytest.fixture
async def engine_fixture():
    """Proporcionar un motor de eventos con limpieza adecuada."""
    engine = EngineNonBlocking(test_mode=True)
    yield engine
    await cleanup_engine(engine)

@pytest.mark.asyncio
async def test_cascading_failure_basic(engine_fixture):
    """
    Verificar que un fallo en un componente se propague correctamente a sus dependientes.
    
    Esta prueba crea una cadena simple de dependencias A -> B -> C y verifica que
    un fallo en A se propague correctamente a B y C.
    """
    engine = engine_fixture
    
    # Crear componentes con cadena de dependencias
    comp_a = DependentComponent("comp_a")
    comp_b = DependentComponent("comp_b", dependencies=["comp_a"])
    comp_c = DependentComponent("comp_c", dependencies=["comp_b"])
    
    # Registrar componentes
    await engine.register_component(comp_a)
    await engine.register_component(comp_b)
    await engine.register_component(comp_c)
    
    # FASE 1: Verificar estado inicial
    logger.info("FASE 1: Verificando estado inicial")
    
    # Usar helpers seguros para obtener estado
    status_a = await check_component_status(engine, "comp_a")
    status_b = await check_component_status(engine, "comp_b")
    status_c = await check_component_status(engine, "comp_c")
    
    # Verificaciones iniciales con manejo defensivo
    assert status_a.get("healthy", False), "comp_a debería estar sano al inicio"
    assert status_b.get("healthy", False), "comp_b debería estar sano al inicio"
    assert status_c.get("healthy", False), "comp_c debería estar sano al inicio"
    
    # FASE 2: Provocar fallo en componente A
    logger.info("FASE 2: Provocando fallo en componente A")
    
    # Cambiar estado de A con manejo de timeout
    response_a = await emit_with_timeout(
        engine, "set_health", {"healthy": False}, "comp_a", timeout=2.0
    )
    
    # Verificar respuesta
    assert safe_get_response(response_a, "healthy") is False, "Cambio de estado en A no confirmado"
    
    # FASE 3: Propagar fallo a través de notificaciones
    logger.info("FASE 3: Propagando fallo a través de notificaciones")
    
    # Notificar a B sobre fallo en A
    notify_b = await emit_with_timeout(
        engine, "dependency_update", {"dependency": "comp_a", "status": False}, "comp_b", timeout=2.0
    )
    
    # Notificar a C sobre fallo en B
    notify_c = await emit_with_timeout(
        engine, "dependency_update", {"dependency": "comp_b", "status": False}, "comp_c", timeout=2.0
    )
    
    # FASE 4: Verificar propagación de fallos
    logger.info("FASE 4: Verificando propagación de fallos")
    
    # Verificar estado después del fallo, con manejo defensivo
    status_a_after = await check_component_status(engine, "comp_a")
    status_b_after = await check_component_status(engine, "comp_b")
    status_c_after = await check_component_status(engine, "comp_c")
    
    # Verificar estado con safe_get_response para evitar errores
    is_a_healthy = safe_get_response(status_a_after, "healthy", default=True)
    is_b_healthy = safe_get_response(status_b_after, "healthy", default=True)
    is_c_healthy = safe_get_response(status_c_after, "healthy", default=True)
    
    # Verificar propagación
    assert not is_a_healthy, "comp_a debería estar no-sano después del fallo"
    assert not is_b_healthy, "comp_b debería estar no-sano debido a dependencia de A"
    assert not is_c_healthy, "comp_c debería estar no-sano debido a dependencia de B"
    
    # FASE 5: Recuperación
    logger.info("FASE 5: Recuperando componentes")
    
    # Restaurar A
    recovery_a = await emit_with_timeout(
        engine, "set_health", {"healthy": True}, "comp_a", timeout=2.0
    )
    
    # Propagar recuperación
    recover_b = await emit_with_timeout(
        engine, "dependency_update", {"dependency": "comp_a", "status": True}, "comp_b", timeout=2.0
    )
    
    recover_c = await emit_with_timeout(
        engine, "dependency_update", {"dependency": "comp_b", "status": True}, "comp_c", timeout=2.0
    )
    
    # FASE 6: Verificar recuperación
    logger.info("FASE 6: Verificando recuperación")
    
    # Verificar estado final
    final_a = await check_component_status(engine, "comp_a")
    final_b = await check_component_status(engine, "comp_b")
    final_c = await check_component_status(engine, "comp_c")
    
    # Usar manejo defensivo
    is_a_recovered = safe_get_response(final_a, "healthy", default=False)
    is_b_recovered = safe_get_response(final_b, "healthy", default=False)
    is_c_recovered = safe_get_response(final_c, "healthy", default=False)
    
    # Verificar recuperación
    assert is_a_recovered, "comp_a debería estar sano después de recuperación"
    assert is_b_recovered, "comp_b debería estar sano después de recuperación"
    assert is_c_recovered, "comp_c debería estar sano después de recuperación"

@pytest.mark.asyncio
async def test_cascading_failure_partial(engine_fixture):
    """
    Verificar que los fallos no afecten a componentes independientes.
    
    Esta prueba crea componentes A -> B y C (independiente) y verifica
    que un fallo en A afecte a B pero no a C.
    """
    engine = engine_fixture
    
    # Función de prueba específica
    async def run_test(engine):
        # Crear componentes
        comp_a = DependentComponent("comp_a")
        comp_b = DependentComponent("comp_b", dependencies=["comp_a"])
        comp_c = DependentComponent("comp_c")  # Sin dependencias
        
        # Registrar componentes
        await engine.register_component(comp_a)
        await engine.register_component(comp_b)
        await engine.register_component(comp_c)
        
        # Verificar estado inicial
        logger.info("Verificando estado inicial")
        assert safe_get_response(await check_component_status(engine, "comp_a"), "healthy", False)
        assert safe_get_response(await check_component_status(engine, "comp_b"), "healthy", False)
        assert safe_get_response(await check_component_status(engine, "comp_c"), "healthy", False)
        
        # Fallar componente A
        logger.info("Fallando componente A")
        await emit_with_timeout(engine, "set_health", {"healthy": False}, "comp_a", timeout=2.0)
        
        # Notificar a B sobre fallo en A
        await emit_with_timeout(
            engine, "dependency_update", {"dependency": "comp_a", "status": False}, "comp_b", timeout=2.0
        )
        
        # Verificar propagación
        status_a = await check_component_status(engine, "comp_a")
        status_b = await check_component_status(engine, "comp_b")
        status_c = await check_component_status(engine, "comp_c")
        
        # Aserciones usando safe_get_response
        assert not safe_get_response(status_a, "healthy", True), "A debería estar no-sano"
        assert not safe_get_response(status_b, "healthy", True), "B debería estar no-sano (dependencia de A)"
        assert safe_get_response(status_c, "healthy", False), "C debería seguir sano (independiente)"
        
        return True  # Indicar prueba exitosa
    
    # Ejecutar prueba con medición de tiempo
    result = await run_test_with_timing(engine, "test_cascading_failure_partial", run_test)
    assert result, "La prueba falló"

@pytest.mark.asyncio
async def test_cascading_failure_recovery(engine_fixture):
    """
    Verificar la recuperación automática de componentes tras la recuperación de sus dependencias.
    
    Esta prueba simula un escenario donde un componente se recupera automáticamente
    cuando sus dependencias vuelven a estar saludables.
    """
    engine = engine_fixture
    
    # Crear componente con recuperación automática
    class AutoRecoveringComponent(DependentComponent):
        """Componente que se recupera automáticamente cuando sus dependencias están sanas."""
        
        async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
            # Comportamiento base
            result = await super().handle_event(event_type, data, source)
            
            # Recuperación automática si las dependencias están bien
            if event_type == "dependency_update" and not self.healthy:
                # Verificar si todas las dependencias están sanas
                if all(self.dependency_status.values()):
                    # Auto-recuperación
                    self.healthy = True
                    logger.info(f"Componente {self.name}: recuperación automática activada")
            
            return result
    
    # Crear componentes
    comp_a = DependentComponent("comp_a")
    comp_b = AutoRecoveringComponent("comp_b", dependencies=["comp_a"])
    
    # Registrar componentes
    await engine.register_component(comp_a)
    await engine.register_component(comp_b)
    
    # Verificar estado inicial
    assert safe_get_response(await check_component_status(engine, "comp_a"), "healthy", False)
    assert safe_get_response(await check_component_status(engine, "comp_b"), "healthy", False)
    
    # Fallar componente A
    await emit_with_timeout(engine, "set_health", {"healthy": False}, "comp_a", timeout=2.0)
    
    # Notificar a B sobre fallo en A
    await emit_with_timeout(
        engine, "dependency_update", {"dependency": "comp_a", "status": False}, "comp_b", timeout=2.0
    )
    
    # Verificar propagación
    assert not safe_get_response(await check_component_status(engine, "comp_a"), "healthy", True)
    assert not safe_get_response(await check_component_status(engine, "comp_b"), "healthy", True)
    
    # Recuperar A
    await emit_with_timeout(engine, "set_health", {"healthy": True}, "comp_a", timeout=2.0)
    
    # Notificar a B sobre recuperación de A
    await emit_with_timeout(
        engine, "dependency_update", {"dependency": "comp_a", "status": True}, "comp_b", timeout=2.0
    )
    
    # Verificar auto-recuperación de B
    assert safe_get_response(await check_component_status(engine, "comp_a"), "healthy", False)
    assert safe_get_response(await check_component_status(engine, "comp_b"), "healthy", False), "B debería haberse recuperado automáticamente"

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])