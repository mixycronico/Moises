"""
Pruebas para escenarios extremos del motor de eventos, versión optimizada.

Este módulo contiene pruebas optimizadas para verificar el comportamiento del 
motor de eventos en situaciones extremas, como fallos en cascada y condiciones
de carrera. Implementa timeouts y monitoreo de rendimiento para evitar bloqueos.
"""

import asyncio
import logging
import pytest
import time
from typing import Dict, Any, List, Optional

# Importamos las utilidades para timeout que creamos
from tests.utils.timeout_helpers import (
    emit_with_timeout,
    check_component_status,
    run_test_with_timing,
    cleanup_engine
)

# Configuración de logging para pruebas
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Nos aseguramos que pytest detecte correctamente las pruebas asíncronas
pytestmark = pytest.mark.asyncio

# Importamos las clases necesarias para las pruebas
# (Asumimos que estos imports existen en el archivo original)
from genesis.core.engine_non_blocking import EngineNonBlocking
from genesis.core.component import Component

class CascadeFailureComponent(Component):
    """
    Componente optimizado que demuestra fallos en cascada.
    
    Este componente falla automáticamente cuando cualquiera de sus 
    dependencias falla, y se recupera cuando todas sus dependencias
    están saludables.
    """
    
    def __init__(self, name: str, dependencies: Optional[List[str]] = None):
        """
        Inicializar el componente de fallo en cascada.
        
        Args:
            name: Nombre del componente
            dependencies: Lista de IDs de componentes de los que depende (opcional)
        """
        super().__init__(name)
        self.dependencies = dependencies if dependencies is not None else []
        self.healthy = True
        self.dependency_status = {dep: True for dep in self.dependencies}
        self.recovery_delay = 0.5  # Tiempo de recuperación reducido para optimizar las pruebas
        
    async def start(self) -> None:
        """Iniciar el componente."""
        logger.info(f"Componente {self.name} iniciado")
        self.healthy = True
        
    async def stop(self) -> None:
        """Detener el componente."""
        logger.info(f"Componente {self.name} detenido")
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """
        Manejar eventos para el componente.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Respuesta al evento, si corresponde
        """
        if event_type == "check_status":
            return {"healthy": self.healthy, "dependencies": self.dependency_status}
            
        elif event_type == "set_health":
            previous_health = self.healthy
            self.healthy = data.get("healthy", True)
            logger.info(f"Estado de salud de {self.name} cambiado de {previous_health} a {self.healthy}")
            return {"healthy": self.healthy, "previous": previous_health}
            
        elif event_type == "dependency_status_change":
            dep_id = data.get("dependency_id", "")
            status = data.get("status", True)
            
            if dep_id and dep_id in self.dependencies:
                self.dependency_status[dep_id] = status
                
                # Si alguna dependencia no está saludable, el componente tampoco lo está
                if not all(self.dependency_status.values()):
                    self.healthy = False
                    logger.info(f"{self.name} marcado como no saludable debido a dependencias")
                    
                # Si todas las dependencias están saludables, programar recuperación
                elif all(self.dependency_status.values()) and not self.healthy:
                    logger.info(f"{self.name} iniciando recuperación...")
                    # La recuperación es asíncrona pero no bloquea el manejo del evento
                    asyncio.create_task(self._recover())
                    
                return {
                    "dependency_id": dep_id,
                    "status": status,
                    "component_health": self.healthy,
                    "dependency_status": self.dependency_status
                }
        
        return None
    
    async def _recover(self) -> None:
        """
        Proceso interno de recuperación del componente.
        
        Esta función simula un tiempo de recuperación tras un fallo.
        """
        await asyncio.sleep(self.recovery_delay)
        self.healthy = True
        logger.info(f"{self.name} recuperado completamente")


@pytest.fixture
async def engine_fixture():
    """
    Fixture para proporcionar un motor de eventos con limpieza adecuada.
    
    Este fixture garantiza que los recursos se limpien correctamente
    entre pruebas, evitando problemas de recursos colgados o condiciones de carrera.
    """
    engine = EngineNonBlocking()
    yield engine
    await cleanup_engine(engine)


async def test_cascading_failures_optimized(engine_fixture):
    """
    Prueba optimizada para verificar los fallos en cascada entre componentes.
    
    Esta prueba verifica que cuando un componente falla, los componentes
    que dependen de él también fallen, pero los componentes independientes
    sigan funcionando normalmente.
    """
    engine = engine_fixture
    
    # FASE 1: Configuración
    logger.info("FASE 1: Configuración de la prueba")
    
    # Registrar los componentes
    comp_a = CascadeFailureComponent("comp_a")
    comp_b = CascadeFailureComponent("comp_b", dependencies=["comp_a"])
    
    await engine.register_component(comp_a)
    await engine.register_component(comp_b)
    
    # Verificar estado inicial
    status_a = await check_component_status(engine, "comp_a")
    status_b = await check_component_status(engine, "comp_b")
    
    assert status_a["healthy"], "Componente A debería estar saludable al inicio"
    assert status_b["healthy"], "Componente B debería estar saludable al inicio"
    
    # FASE 2: Provocar fallo en A
    logger.info("FASE 2: Provocando fallo en componente A")
    
    # Medir el tiempo que tarda esta operación
    start_time = time.time()
    
    # Establecer componente A como no saludable (con timeout)
    response = await emit_with_timeout(
        engine, "set_health", {"healthy": False}, "comp_a", timeout=2.0
    )
    
    logger.info(f"Respuesta al cambiar salud de A: {response} "
               f"(tardó {time.time() - start_time:.3f}s)")
    
    # Notificar cambio de estado a los dependientes (con timeout)
    notify_response = await emit_with_timeout(
        engine, 
        "dependency_status_change", 
        {"dependency_id": "comp_a", "status": False}, 
        "comp_b",
        timeout=2.0
    )
    
    logger.info(f"Respuesta a la notificación: {notify_response}")
    
    # Pequeña pausa para permitir que se propague el fallo
    await asyncio.sleep(0.1)
    
    # FASE 3: Verificar estado después del fallo
    logger.info("FASE 3: Verificando estado después del fallo")
    
    # Verificar A con timeout
    start_time = time.time()
    resp_a = await check_component_status(engine, "comp_a")
    logger.info(f"Estado A: {resp_a} (tardó {time.time() - start_time:.3f}s)")
    
    # Verificar B con timeout
    start_time = time.time()
    resp_b = await check_component_status(engine, "comp_b")
    logger.info(f"Estado B: {resp_b} (tardó {time.time() - start_time:.3f}s)")
    
    # Aserciones
    assert not resp_a["healthy"], "A debería estar no-sano después del fallo"
    assert not resp_b["healthy"], "B debería estar no-sano debido a la dependencia"
    
    # FASE 4: Recuperación de A
    logger.info("FASE 4: Recuperando componente A")
    
    # Restablecer A como saludable (con timeout)
    recovery_resp = await emit_with_timeout(
        engine, "set_health", {"healthy": True}, "comp_a", timeout=2.0
    )
    
    logger.info(f"Respuesta de recuperación de A: {recovery_resp}")
    
    # Notificar cambio a los dependientes (con timeout)
    notify_recovery = await emit_with_timeout(
        engine, 
        "dependency_status_change", 
        {"dependency_id": "comp_a", "status": True}, 
        "comp_b",
        timeout=2.0
    )
    
    logger.info(f"Respuesta a notificación de recuperación: {notify_recovery}")
    
    # Esperar un tiempo para la recuperación automática de B
    # (tiempo ligeramente mayor que el recovery_delay definido en el componente)
    await asyncio.sleep(0.7)
    
    # FASE 5: Verificar recuperación
    logger.info("FASE 5: Verificando recuperación")
    
    # Verificar estado de componentes después de recuperación (con timeout)
    resp_a_recovery = await check_component_status(engine, "comp_a")
    resp_b_recovery = await check_component_status(engine, "comp_b")
    
    logger.info(f"Estado A tras recuperación: {resp_a_recovery}")
    logger.info(f"Estado B tras recuperación: {resp_b_recovery}")
    
    # Aserciones finales
    assert resp_a_recovery["healthy"], "A debería estar sano después de recuperación"
    assert resp_b_recovery["healthy"], "B debería recuperarse automáticamente"
    
    logger.info("Prueba completada exitosamente")


async def test_multiple_component_failure_optimized(engine_fixture):
    """
    Prueba optimizada para verificar fallo y recuperación de múltiples componentes.
    
    Esta prueba crea una cadena de dependencias y verifica que los fallos
    se propaguen correctamente y que la recuperación restaure todos los componentes.
    """
    # Esta prueba utilizará la función run_test_with_timing para medir el rendimiento
    async def run_multiple_failure_test(engine):
        # Crear y registrar componentes en cadena: A <- B <- C <- D
        components = {}
        component_ids = ["comp_a", "comp_b", "comp_c", "comp_d"]
        
        # Crear A (sin dependencias)
        components["comp_a"] = CascadeFailureComponent("comp_a")
        await engine.register_component(components["comp_a"])
        
        # Crear B, C, D con dependencias en cadena
        for i in range(1, 4):
            curr_id = component_ids[i]
            prev_id = component_ids[i-1]
            components[curr_id] = CascadeFailureComponent(curr_id, dependencies=[prev_id])
            await engine.register_component(components[curr_id])
        
        # Verificar estados iniciales
        for comp_id in component_ids:
            status = await check_component_status(engine, comp_id)
            assert status["healthy"], f"{comp_id} debería estar sano inicialmente"
        
        # Provocar fallo en A (con timeout)
        await emit_with_timeout(engine, "set_health", {"healthy": False}, "comp_a", timeout=2.0)
        
        # Propagar fallo a través de la cadena
        for i in range(1, 4):
            prev_id = component_ids[i-1]
            curr_id = component_ids[i]
            await emit_with_timeout(
                engine,
                "dependency_status_change",
                {"dependency_id": prev_id, "status": False},
                curr_id,
                timeout=2.0
            )
        
        # Pequeña pausa para permitir propagación
        await asyncio.sleep(0.1)
        
        # Verificar que todos los componentes han fallado
        for comp_id in component_ids:
            status = await check_component_status(engine, comp_id)
            assert not status["healthy"], f"{comp_id} debería estar no-sano"
        
        # Recuperar A (con timeout)
        await emit_with_timeout(engine, "set_health", {"healthy": True}, "comp_a", timeout=2.0)
        
        # Propagar recuperación
        for i in range(1, 4):
            prev_id = component_ids[i-1]
            curr_id = component_ids[i]
            await emit_with_timeout(
                engine,
                "dependency_status_change",
                {"dependency_id": prev_id, "status": True},
                curr_id,
                timeout=2.0
            )
        
        # Esperar recuperación
        await asyncio.sleep(0.7)
        
        # Verificar que todos se recuperaron
        for comp_id in component_ids:
            status = await check_component_status(engine, comp_id)
            assert status["healthy"], f"{comp_id} debería estar recuperado"
            
        return True  # Prueba completada con éxito
    
    # Ejecutar la prueba con medición de tiempo
    result = await run_test_with_timing(
        engine_fixture, 
        "test_multiple_component_failure", 
        run_multiple_failure_test
    )
    
    assert result, "La prueba debería completarse exitosamente"
    
    
if __name__ == "__main__":
    # Para poder ejecutar este archivo directamente
    import pytest
    pytest.main(["-xvs", __file__])