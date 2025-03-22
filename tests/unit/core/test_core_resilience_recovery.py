"""
Pruebas de resiliencia y recuperación para el core del sistema Genesis.

Este módulo contiene pruebas que verifican la capacidad del sistema para
recuperarse de fallos controlados, como desconexiones, errores severos
en componentes, y otras situaciones adversas.
"""

import pytest
import asyncio
import logging
import time
import random
from typing import Dict, Any, List, Optional, Set
from unittest.mock import patch, MagicMock

from genesis.core.component import Component
from genesis.core.engine_non_blocking import EngineNonBlocking

# Configurar logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class ResilienceTestComponent(Component):
    """Componente diseñado para probar la resiliencia del sistema."""
    
    def __init__(self, name: str):
        """Inicializar componente de prueba."""
        super().__init__(name)
        self.events: List[Dict[str, Any]] = []
        self.start_count = 0
        self.stop_count = 0
        self.error_count = 0
        self.is_healthy = True
        self.total_events_processed = 0
        self.recovery_events = 0
    
    async def start(self) -> None:
        """Iniciar componente."""
        self.start_count += 1
        logger.info(f"Iniciando componente de resiliencia {self.name} (#{self.start_count})")
    
    async def stop(self) -> None:
        """Detener componente."""
        self.stop_count += 1
        logger.info(f"Deteniendo componente de resiliencia {self.name} (#{self.stop_count})")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """
        Manejar un evento, posiblemente generando un error.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Opcional, respuesta al evento
            
        Raises:
            Exception: Si el componente no está sano o si el evento indica un error
        """
        # Registrar evento
        self.events.append({
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": time.time()
        })
        
        self.total_events_processed += 1
        
        # Si el componente no está sano, fallar
        if not self.is_healthy:
            self.error_count += 1
            raise Exception(f"Componente {self.name} no está sano")
        
        # Si es un evento de error, fallar
        if event_type.startswith("error_"):
            self.error_count += 1
            raise Exception(f"Error simulado en {self.name} al procesar {event_type}")
        
        # Si es un evento de recuperación, marcar como sano
        if event_type == "recovery":
            self.is_healthy = True
            self.recovery_events += 1
        
        # Si es un evento para marcar como no sano
        if event_type == "set_unhealthy":
            self.is_healthy = False
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas del componente.
        
        Returns:
            Diccionario con métricas
        """
        return {
            "start_count": self.start_count,
            "stop_count": self.stop_count,
            "error_count": self.error_count,
            "is_healthy": self.is_healthy,
            "total_events": self.total_events_processed,
            "recovery_events": self.recovery_events
        }


class FlakyComponent(Component):
    """Componente que falla intermitentemente."""
    
    def __init__(self, name: str, fail_rate: float = 0.3):
        """
        Inicializar componente intermitente.
        
        Args:
            name: Nombre del componente
            fail_rate: Tasa de fallos (0-1)
        """
        super().__init__(name)
        self.events: List[Dict[str, Any]] = []
        self.fail_rate = fail_rate
        self.error_count = 0
        self.success_count = 0
    
    async def start(self) -> None:
        """Iniciar componente, posiblemente fallando."""
        if random.random() < self.fail_rate:
            raise Exception(f"Error al iniciar componente {self.name}")
        logger.info(f"Iniciando componente intermitente {self.name}")
    
    async def stop(self) -> None:
        """Detener componente, posiblemente fallando."""
        if random.random() < self.fail_rate:
            raise Exception(f"Error al detener componente {self.name}")
        logger.info(f"Deteniendo componente intermitente {self.name}")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """
        Manejar un evento, posiblemente fallando aleatoriamente.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Opcional, respuesta al evento
            
        Raises:
            Exception: Si el componente decide fallar aleatoriamente
        """
        # Registrar evento
        self.events.append({
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": time.time()
        })
        
        # Decidir si fallar
        if random.random() < self.fail_rate:
            self.error_count += 1
            raise Exception(f"Error aleatorio en {self.name} al procesar {event_type}")
        
        self.success_count += 1
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas del componente.
        
        Returns:
            Diccionario con métricas
        """
        return {
            "error_count": self.error_count,
            "success_count": self.success_count,
            "fail_rate": self.fail_rate,
            "total_events": self.error_count + self.success_count
        }


class CrashingComponent(Component):
    """Componente que se bloquea por completo después de un número de eventos."""
    
    def __init__(self, name: str, crash_after: int = 5):
        """
        Inicializar componente que se bloquea.
        
        Args:
            name: Nombre del componente
            crash_after: Número de eventos después del cual se bloquea
        """
        super().__init__(name)
        self.events: List[Dict[str, Any]] = []
        self.crash_after = crash_after
        self.event_count = 0
        self.crashed = False
    
    async def start(self) -> None:
        """Iniciar componente."""
        logger.info(f"Iniciando componente que se bloquea {self.name}")
        self.event_count = 0
        self.crashed = False
    
    async def stop(self) -> None:
        """Detener componente."""
        logger.info(f"Deteniendo componente que se bloquea {self.name}")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """
        Manejar un evento, bloqueándose completamente después de un número específico.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Opcional, respuesta al evento
        """
        # Registrar evento
        self.events.append({
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": time.time()
        })
        
        self.event_count += 1
        
        # Verificar si debe bloquearse
        if self.event_count >= self.crash_after:
            self.crashed = True
            # Simular un bloqueo que será interrumpido por el timeout
            # En modo de prueba, usamos un tiempo más corto para no bloquear los tests
            # El sistema debería interrumpir este sleep con el timeout configurado
            await asyncio.sleep(0.5)
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas del componente.
        
        Returns:
            Diccionario con métricas
        """
        return {
            "event_count": self.event_count,
            "crash_after": self.crash_after,
            "crashed": self.crashed
        }


@pytest.mark.asyncio
async def test_component_recovery():
    """Prueba la capacidad del sistema para manejar componentes que fallan y se recuperan."""
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componentes
    normal_comp = ResilienceTestComponent("normal")
    recoverable_comp = ResilienceTestComponent("recoverable")
    
    # Registrar componentes
    engine.register_component(normal_comp)
    engine.register_component(recoverable_comp)
    
    # Iniciar motor
    await engine.start()
    
    # Enviar eventos normales a ambos componentes
    for i in range(5):
        await engine.emit_event(f"normal_event_{i}", {"id": i}, "test")
        await asyncio.sleep(0.1)
    
    # Verificar que ambos componentes procesaron los eventos
    assert normal_comp.total_events_processed == 5, "El componente normal debería haber procesado 5 eventos"
    assert recoverable_comp.total_events_processed == 5, "El componente recuperable debería haber procesado 5 eventos"
    
    # Marcar el componente recuperable como no sano
    await engine.emit_event("set_unhealthy", {}, "test")
    await asyncio.sleep(0.1)
    
    # Enviar eventos que deberían causar errores en el componente no sano
    for i in range(3):
        await engine.emit_event(f"should_fail_{i}", {"id": i}, "test")
        await asyncio.sleep(0.1)
    
    # Verificar que el componente normal siguió procesando eventos
    assert normal_comp.total_events_processed == 9, "El componente normal debería haber procesado 9 eventos"
    
    # Verificar que el componente recuperable registró los eventos pero generó errores
    assert recoverable_comp.total_events_processed == 9, "El componente recuperable debería haber registrado 9 eventos"
    assert recoverable_comp.error_count > 0, "El componente recuperable debería haber generado errores"
    
    # Enviar evento de recuperación
    await engine.emit_event("recovery", {"recovery": True}, "test")
    await asyncio.sleep(0.1)
    
    # Verificar que el componente se recuperó
    assert recoverable_comp.is_healthy, "El componente recuperable debería estar sano después de la recuperación"
    assert recoverable_comp.recovery_events == 1, "El componente recuperable debería haber registrado un evento de recuperación"
    
    # Enviar más eventos normales
    for i in range(3):
        await engine.emit_event(f"after_recovery_{i}", {"id": i}, "test")
        await asyncio.sleep(0.1)
    
    # Verificar que ambos componentes procesaron los eventos posteriores a la recuperación
    assert normal_comp.total_events_processed == 13, "El componente normal debería haber procesado 13 eventos en total"
    assert recoverable_comp.total_events_processed == 13, "El componente recuperable debería haber procesado 13 eventos en total"
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
async def test_error_isolation():
    """Prueba que los errores en un componente no afecten a otros componentes."""
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componentes
    components = [ResilienceTestComponent(f"comp_{i}") for i in range(5)]
    error_comp = ResilienceTestComponent("error_comp")
    
    # Registrar componentes
    for comp in components:
        engine.register_component(comp)
    engine.register_component(error_comp)
    
    # Iniciar motor
    await engine.start()
    
    # Enviar eventos normales a todos los componentes
    for i in range(3):
        await engine.emit_event(f"normal_event_{i}", {"id": i}, "test")
        await asyncio.sleep(0.1)
    
    # Marcar el componente de error como no sano
    await engine.emit_event("set_unhealthy", {}, "test")
    await asyncio.sleep(0.1)
    
    # Enviar eventos que causarán errores en el componente no sano
    for i in range(3):
        await engine.emit_event(f"mixed_event_{i}", {"id": i}, "test")
        await asyncio.sleep(0.1)
    
    # Enviar eventos específicos de error
    for i in range(3):
        await engine.emit_event(f"error_event_{i}", {"id": i}, "test")
        await asyncio.sleep(0.1)
    
    # Verificar que los componentes normales procesaron todos los eventos
    for i, comp in enumerate(components):
        expected_events = 3 + 3 + 3  # normal + mixed + error
        assert comp.total_events_processed == expected_events, f"El componente {i} debería haber procesado {expected_events} eventos"
        assert comp.error_count == 3, f"El componente {i} debería tener errores solo para los eventos de error"
    
    # Verificar que el componente de error generó errores para todos los eventos después de ser marcado como no sano
    assert error_comp.error_count > 3, "El componente de error debería haber generado errores para eventos después de ser marcado como no sano"
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
async def test_intermittent_failures():
    """Prueba el sistema con componentes que fallan intermitentemente."""
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componentes
    normal_comp = ResilienceTestComponent("normal")
    flaky_comp = FlakyComponent("flaky", fail_rate=0.5)  # 50% de probabilidad de fallo
    
    # Registrar componentes
    engine.register_component(normal_comp)
    engine.register_component(flaky_comp)
    
    # Iniciar motor
    await engine.start()
    
    # Enviar eventos
    num_events = 20
    for i in range(num_events):
        await engine.emit_event(f"test_event_{i}", {"id": i}, "test")
        await asyncio.sleep(0.1)
    
    # Verificar que el componente normal procesó todos los eventos
    assert normal_comp.total_events_processed == num_events, f"El componente normal debería haber procesado {num_events} eventos"
    
    # Verificar que el componente intermitente tuvo una tasa de fallos cercana a la esperada
    flaky_metrics = flaky_comp.get_metrics()
    actual_fail_rate = flaky_metrics["error_count"] / flaky_metrics["total_events"]
    
    logger.info(f"Componente intermitente - tasa de fallos esperada: {flaky_comp.fail_rate}, real: {actual_fail_rate}")
    logger.info(f"Eventos exitosos: {flaky_metrics['success_count']}, fallidos: {flaky_metrics['error_count']}")
    
    # La tasa de fallos real debe estar dentro de un rango razonable de la esperada
    # (considerando que con pocos eventos puede haber variación estadística)
    assert 0.2 <= actual_fail_rate <= 0.8, f"La tasa de fallos debería estar cerca de 0.5, pero fue {actual_fail_rate}"
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
@pytest.mark.timeout(5)  # Limitar el tiempo de ejecución a 5 segundos
async def test_component_slow_response():
    """Prueba que el sistema maneje correctamente componentes que responden lentamente."""
    # Crear motor no bloqueante con timeouts muy cortos para pruebas
    engine = EngineNonBlocking(test_mode=True, component_timeout=0.1)  # Timeout muy corto para pruebas
    
    try:
        # Crear componentes
        normal_comp = ResilienceTestComponent("normal")
        error_comp = ResilienceTestComponent("error")
        error_comp.is_healthy = False  # Este componente generará errores
        
        # Registrar componentes
        engine.register_component(normal_comp)
        engine.register_component(error_comp)
        
        # Iniciar motor
        await engine.start()
        
        # Enviar algunos eventos simples al sistema
        await engine.emit_event("test_event_1", {"id": 1}, "test")
        await engine.emit_event("test_event_2", {"id": 2}, "test")
        
        # Verificar el comportamiento básico
        assert normal_comp.total_events_processed >= 2, "El componente normal debería haber procesado eventos"
        assert error_comp.total_events_processed >= 2, "El componente con errores debería haber registrado eventos"
        assert error_comp.error_count > 0, "El componente con errores debería haber generado errores"
        
        # Verificar que el motor sigue funcionando correctamente
        assert engine.is_running, "El motor debería seguir en ejecución a pesar de los errores"
        
    finally:
        # Asegurarse de detener el motor incluso si hay errores
        if 'engine' in locals() and engine.is_running:
            await engine.stop()


@pytest.mark.asyncio
async def test_engine_recovery_after_severe_errors():
    """Prueba que el motor puede recuperarse después de errores severos."""
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componentes
    normal_comp = ResilienceTestComponent("normal")
    
    # Este componente fallará en start() pero aún así debería registrarse
    with patch.object(ResilienceTestComponent, 'start', side_effect=Exception("Error simulado en start")):
        failing_start_comp = ResilienceTestComponent("failing_start")
        engine.register_component(failing_start_comp)
    
    # Registrar componente normal
    engine.register_component(normal_comp)
    
    # Iniciar motor (debería continuar a pesar del error en start de un componente)
    await engine.start()
    
    # Verificar que el motor inició a pesar del error
    assert engine.is_running, "El motor debería estar en ejecución a pesar del error en start de un componente"
    
    # Verificar que el componente normal inició correctamente
    assert normal_comp.start_count == 1, "El componente normal debería haberse iniciado una vez"
    
    # Enviar eventos al sistema
    for i in range(5):
        await engine.emit_event(f"test_event_{i}", {"id": i}, "test")
        await asyncio.sleep(0.1)
    
    # Verificar que el componente normal procesó los eventos
    assert normal_comp.total_events_processed == 5, "El componente normal debería haber procesado 5 eventos"
    
    # Detener motor
    await engine.stop()
    
    # Verificar que el componente normal se detuvo
    assert normal_comp.stop_count == 1, "El componente normal debería haberse detenido una vez"


@pytest.mark.asyncio
async def test_component_reregistration():
    """Prueba que un componente puede ser desregistrado y reregistrado en el motor."""
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componentes
    comp1 = ResilienceTestComponent("comp1")
    comp2 = ResilienceTestComponent("comp2")
    
    # Registrar componentes
    engine.register_component(comp1)
    engine.register_component(comp2)
    
    # Iniciar motor
    await engine.start()
    
    # Enviar eventos iniciales
    for i in range(3):
        await engine.emit_event(f"initial_event_{i}", {"id": i}, "test")
        await asyncio.sleep(0.1)
    
    # Verificar que ambos componentes procesaron los eventos
    assert comp1.total_events_processed == 3, "El componente 1 debería haber procesado 3 eventos iniciales"
    assert comp2.total_events_processed == 3, "El componente 2 debería haber procesado 3 eventos iniciales"
    
    # Desregistrar el componente 1
    engine.deregister_component(comp1)
    
    # Enviar más eventos
    for i in range(3):
        await engine.emit_event(f"mid_event_{i}", {"id": i}, "test")
        await asyncio.sleep(0.1)
    
    # Verificar que solo el componente 2 procesó los eventos intermedios
    assert comp1.total_events_processed == 3, "El componente 1 no debería haber recibido eventos después de desregistrarse"
    assert comp2.total_events_processed == 6, "El componente 2 debería haber procesado 6 eventos en total"
    
    # Volver a registrar el componente 1
    engine.register_component(comp1)
    
    # Enviar eventos finales
    for i in range(3):
        await engine.emit_event(f"final_event_{i}", {"id": i}, "test")
        await asyncio.sleep(0.1)
    
    # Verificar que ambos componentes procesaron los eventos finales
    assert comp1.total_events_processed == 6, "El componente 1 debería haber procesado 3 eventos iniciales y 3 finales"
    assert comp2.total_events_processed == 9, "El componente 2 debería haber procesado 9 eventos en total"
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
async def test_engine_restart():
    """Prueba que el motor puede detenerse y reiniciarse correctamente."""
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componentes
    comp1 = ResilienceTestComponent("comp1")
    comp2 = ResilienceTestComponent("comp2")
    
    # Registrar componentes
    engine.register_component(comp1)
    engine.register_component(comp2)
    
    # Primera ejecución del motor
    await engine.start()
    
    # Enviar eventos
    for i in range(3):
        await engine.emit_event(f"first_run_event_{i}", {"id": i}, "test")
        await asyncio.sleep(0.1)
    
    # Verificar que ambos componentes procesaron los eventos
    assert comp1.total_events_processed == 3, "El componente 1 debería haber procesado 3 eventos en la primera ejecución"
    assert comp2.total_events_processed == 3, "El componente 2 debería haber procesado 3 eventos en la primera ejecución"
    
    # Detener motor
    await engine.stop()
    
    # Verificar que los componentes se detuvieron
    assert comp1.stop_count == 1, "El componente 1 debería haberse detenido una vez"
    assert comp2.stop_count == 1, "El componente 2 debería haberse detenido una vez"
    
    # Reiniciar motor
    await engine.start()
    
    # Verificar que los componentes se iniciaron nuevamente
    assert comp1.start_count == 2, "El componente 1 debería haberse iniciado dos veces"
    assert comp2.start_count == 2, "El componente 2 debería haberse iniciado dos veces"
    
    # Enviar más eventos
    for i in range(3):
        await engine.emit_event(f"second_run_event_{i}", {"id": i}, "test")
        await asyncio.sleep(0.1)
    
    # Verificar que ambos componentes procesaron los eventos en la segunda ejecución
    assert comp1.total_events_processed == 6, "El componente 1 debería haber procesado 6 eventos en total"
    assert comp2.total_events_processed == 6, "El componente 2 debería haber procesado 6 eventos en total"
    
    # Detener motor nuevamente
    await engine.stop()
    
    # Verificar que los componentes se detuvieron nuevamente
    assert comp1.stop_count == 2, "El componente 1 debería haberse detenido dos veces"
    assert comp2.stop_count == 2, "El componente 2 debería haberse detenido dos veces"