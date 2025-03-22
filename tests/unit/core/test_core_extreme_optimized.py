"""
Versión optimizada de pruebas extremas para el core de Genesis.

Este módulo implementa pruebas para escenarios extremos usando
el motor con timeouts configurables para mejor tolerancia.
"""

import pytest
import asyncio
import logging
import time
import random
from typing import Dict, Any, List, Optional, Set

from genesis.core.component import Component
from genesis.core.engine_configurable import ConfigurableTimeoutEngine

# Configurar logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class ExtremeTestComponent(Component):
    """Componente para pruebas extremas con timeout configurable."""
    
    def __init__(self, name: str, 
                 fail_on_start: bool = False, 
                 fail_on_stop: bool = False,
                 start_delay: float = 0.0,
                 stop_delay: float = 0.0,
                 event_delay: float = 0.0,
                 fail_probability: float = 0.0,
                 block_after_events: int = 0,
                 block_time: float = 0.5):
        """
        Inicializar componente de prueba de extremos.
        
        Args:
            name: Nombre del componente
            fail_on_start: Si debe fallar al iniciar
            fail_on_stop: Si debe fallar al detener
            start_delay: Tiempo de retardo al iniciar (segundos)
            stop_delay: Tiempo de retardo al detener (segundos)
            event_delay: Tiempo de retardo al procesar eventos (segundos)
            fail_probability: Probabilidad de falla al procesar eventos (0-1)
            block_after_events: Número de eventos tras los cuales se bloquea
            block_time: Tiempo de bloqueo (segundos)
        """
        super().__init__(name)
        self.events = []
        self.fail_on_start = fail_on_start
        self.fail_on_stop = fail_on_stop
        self.start_delay = start_delay
        self.stop_delay = stop_delay
        self.event_delay = event_delay
        self.fail_probability = fail_probability
        self.block_after_events = block_after_events
        self.block_time = block_time
        
        # Contadores
        self.start_attempts = 0
        self.stop_attempts = 0
        self.processed_count = 0
        self.failed_count = 0
        self.blocked_count = 0
        self.started = False
        self.stopped = False
    
    async def start(self) -> None:
        """Iniciar componente con posible retardo o fallo."""
        self.start_attempts += 1
        logger.info(f"Iniciando componente {self.name} (intento {self.start_attempts})")
        
        # Simular retardo
        if self.start_delay > 0:
            logger.debug(f"Componente {self.name} retrasando inicio por {self.start_delay}s")
            await asyncio.sleep(self.start_delay)
        
        # Simular fallo
        if self.fail_on_start:
            logger.warning(f"Componente {self.name} fallando intencionalmente al iniciar")
            raise RuntimeError(f"Fallo simulado al iniciar {self.name}")
        
        self.started = True
    
    async def stop(self) -> None:
        """Detener componente con posible retardo o fallo."""
        self.stop_attempts += 1
        logger.info(f"Deteniendo componente {self.name} (intento {self.stop_attempts})")
        
        # Simular retardo
        if self.stop_delay > 0:
            logger.debug(f"Componente {self.name} retrasando detención por {self.stop_delay}s")
            await asyncio.sleep(self.stop_delay)
        
        # Simular fallo
        if self.fail_on_stop:
            logger.warning(f"Componente {self.name} fallando intencionalmente al detener")
            raise RuntimeError(f"Fallo simulado al detener {self.name}")
        
        self.stopped = True
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """
        Manejar un evento con posible retardo, bloqueo o fallo.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Respuesta opcional
            
        Raises:
            RuntimeError: Si el componente está configurado para fallar
        """
        start_time = time.time()
        
        # Registrar evento
        self.events.append({
            "type": event_type,
            "data": data.copy() if data else {},  # Crear copia para evitar problemas
            "source": source,
            "received_at": start_time
        })
        
        self.processed_count += 1
        
        # Simular retardo normal
        if self.event_delay > 0:
            await asyncio.sleep(self.event_delay)
        
        # Simular bloqueo después de cierto número de eventos
        if self.block_after_events > 0 and self.processed_count >= self.block_after_events:
            self.blocked_count += 1
            logger.warning(f"Componente {self.name} bloqueándose por {self.block_time}s")
            await asyncio.sleep(self.block_time)
        
        # Simular fallo aleatorio
        if random.random() < self.fail_probability:
            self.failed_count += 1
            logger.warning(f"Componente {self.name} fallando intencionalmente al procesar {event_type}")
            raise RuntimeError(f"Fallo simulado al procesar evento {event_type} en {self.name}")
        
        # Calcular tiempo de procesamiento
        processing_time = time.time() - start_time
        
        return {
            "component": self.name,
            "processed_count": self.processed_count,
            "processing_time": processing_time
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del componente.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "name": self.name,
            "events_received": len(self.events),
            "events_processed": self.processed_count,
            "start_attempts": self.start_attempts,
            "stop_attempts": self.stop_attempts,
            "failed_count": self.failed_count,
            "blocked_count": self.blocked_count,
            "started": self.started,
            "stopped": self.stopped
        }


@pytest.mark.asyncio
async def test_extreme_conditions_optimized():
    """
    Prueba optimizada de condiciones extremas usando el motor configurable.
    
    Esta prueba crea varios componentes con comportamientos extremos y
    verifica que el sistema puede manejarlos con timeouts configurables.
    """
    # Crear motor configurable con timeouts mayores a los predeterminados
    engine = ConfigurableTimeoutEngine(
        test_mode=True,
        component_start_timeout=1.0,  # 1 segundo para iniciar componentes
        component_stop_timeout=1.0,   # 1 segundo para detener componentes
        event_timeout=1.0,           # 1 segundo para emitir eventos
        component_event_timeout=1.0   # 1 segundo para manejadores de eventos
    )
    
    # Habilitar recuperación avanzada
    engine.enable_advanced_recovery(True)
    
    # Crear varios componentes con comportamientos extremos
    components = [
        # Componente normal para referencia
        ExtremeTestComponent(
            name="normal",
            event_delay=0.01  # Pequeño retardo para simular trabajo real
        ),
        
        # Componente que falla al iniciar
        ExtremeTestComponent(
            name="fail_start", 
            fail_on_start=True
        ),
        
        # Componente lento al iniciar pero eventualmente funciona
        ExtremeTestComponent(
            name="slow_start",
            start_delay=0.8  # Justo por debajo del timeout
        ),
        
        # Componente que a veces falla al procesar eventos
        ExtremeTestComponent(
            name="flaky",
            fail_probability=0.3  # 30% de probabilidad de fallo
        ),
        
        # Componente que se bloquea después de cierto número de eventos
        ExtremeTestComponent(
            name="blocking",
            block_after_events=5,
            block_time=0.8  # Justo por debajo del timeout
        ),
        
        # Componente extremadamente lento para procesar eventos
        ExtremeTestComponent(
            name="very_slow",
            event_delay=0.5  # Significativo pero debajo del timeout
        )
    ]
    
    # Registrar todos los componentes
    for comp in components:
        engine.register_component(comp)
    
    # Iniciar el motor (debería manejar el componente que falla al iniciar)
    await engine.start()
    
    # Verificar que el motor está funcionando a pesar de los componentes problemáticos
    assert engine.running, "El motor debería estar ejecutándose incluso con componentes fallidos"
    
    # Revisar estadísticas de timeouts
    stats = engine.get_timeout_stats()
    logger.info(f"Estadísticas de timeouts después de iniciar: {stats}")
    
    # Enviar varios eventos al sistema
    num_events = 10
    for i in range(num_events):
        event_data = {"id": i, "value": f"test_{i}"}
        await engine.emit_event(f"test_event_{i % 5}", event_data, "test")
        # Pausa pequeña para evitar sobrecarga
        await asyncio.sleep(0.05)
    
    # Enviar un evento que causará bloqueo en el componente configurado para bloquearse
    await engine.emit_event("trigger_block", {"trigger": True}, "test")
    await asyncio.sleep(0.1)
    
    # Revisar estadísticas de timeouts después de los eventos
    stats = engine.get_timeout_stats()
    logger.info(f"Estadísticas de timeouts después de eventos: {stats}")
    
    # Verificar que los componentes recibieron eventos según lo esperado
    for comp in components:
        comp_stats = comp.get_stats()
        logger.info(f"Estadísticas del componente {comp.name}: {comp_stats}")
        
        # Verificar que el componente normal procesó todos los eventos
        if comp.name == "normal":
            # El componente normal debería haber procesado todos los eventos
            assert comp_stats["events_processed"] >= num_events, \
                f"El componente normal debería haber procesado al menos {num_events} eventos"
        
        # Verificar que el componente que falla al iniciar no procesó eventos
        if comp.name == "fail_start":
            assert not comp_stats["started"], \
                "El componente que falla al iniciar no debería estar iniciado"
            assert comp_stats["events_processed"] == 0, \
                "El componente que falla al iniciar no debería haber procesado eventos"
        
        # Verificar que el componente lento al iniciar procesó eventos
        if comp.name == "slow_start":
            assert comp_stats["started"], \
                "El componente lento al iniciar debería haberse iniciado"
            assert comp_stats["events_processed"] > 0, \
                "El componente lento al iniciar debería haber procesado eventos"
        
        # Verificar que el componente intermitente tiene registros de fallos
        if comp.name == "flaky":
            assert comp_stats["failed_count"] > 0, \
                "El componente intermitente debería haber registrado fallos"
        
        # Verificar que el componente que se bloquea llegó a bloquearse
        if comp.name == "blocking":
            assert comp_stats["blocked_count"] > 0, \
                "El componente que se bloquea debería haberse bloqueado al menos una vez"
    
    # Detener el motor
    await engine.stop()
    
    # Verificar que el motor se detuvo correctamente
    assert not engine.running, "El motor debería estar detenido"
    
    # Revisar estadísticas finales de timeouts
    stats = engine.get_timeout_stats()
    logger.info(f"Estadísticas finales de timeouts: {stats}")


@pytest.mark.asyncio
async def test_extreme_load_optimized():
    """
    Prueba optimizada de carga extrema usando el motor configurable.
    
    Esta prueba envía un gran número de eventos en ráfagas para
    verificar que el sistema puede manejar alta carga sin bloquearse.
    """
    # Crear motor configurable con timeouts más largos
    engine = ConfigurableTimeoutEngine(
        test_mode=True,
        component_start_timeout=1.0,
        component_stop_timeout=1.0,
        event_timeout=1.0,
        component_event_timeout=1.0
    )
    
    # Habilitar recuperación avanzada
    engine.enable_advanced_recovery(True)
    
    # Crear componentes con diferentes características de rendimiento
    components = [
        # Componente rápido
        ExtremeTestComponent(
            name="fast",
            event_delay=0.001  # Muy rápido
        ),
        
        # Componente medio
        ExtremeTestComponent(
            name="medium",
            event_delay=0.01  # Moderadamente rápido
        ),
        
        # Componente lento
        ExtremeTestComponent(
            name="slow",
            event_delay=0.05  # Bastante lento
        ),
        
        # Componente muy lento pero por debajo del timeout
        ExtremeTestComponent(
            name="very_slow",
            event_delay=0.2  # Muy lento pero debajo del timeout
        ),
        
        # Componente con fallo aleatorio (baja probabilidad)
        ExtremeTestComponent(
            name="flaky",
            event_delay=0.01,
            fail_probability=0.05  # 5% de probabilidad de fallo
        )
    ]
    
    # Registrar todos los componentes
    for comp in components:
        engine.register_component(comp)
    
    # Iniciar el motor
    await engine.start()
    
    # Configuración de carga extrema
    num_bursts = 3            # Número de ráfagas
    events_per_burst = 10     # Eventos por ráfaga
    burst_interval = 0.2      # Intervalo entre ráfagas
    
    # Enviar eventos en ráfagas
    total_events = 0
    for burst in range(num_bursts):
        logger.info(f"Enviando ráfaga {burst+1}/{num_bursts} con {events_per_burst} eventos")
        
        # Enviar ráfaga de eventos
        for i in range(events_per_burst):
            event_data = {
                "id": total_events,
                "burst": burst,
                "index": i,
                "timestamp": time.time()
            }
            await engine.emit_event(f"burst_event_{i % 5}", event_data, "test")
            total_events += 1
            
            # Pequeña pausa para simular eventos reales
            await asyncio.sleep(0.01)
        
        # Esperar intervalo entre ráfagas
        if burst < num_bursts - 1:
            await asyncio.sleep(burst_interval)
    
    # Esperar un tiempo para que se procesen los eventos
    # Calcular tiempo de espera basado en el componente más lento
    slowest_delay = max(comp.event_delay for comp in components)
    wait_time = slowest_delay * events_per_burst * 0.5  # Factor de seguridad
    logger.info(f"Esperando {wait_time:.2f}s para procesamiento...")
    await asyncio.sleep(wait_time)
    
    # Revisar estadísticas de timeouts
    stats = engine.get_timeout_stats()
    logger.info(f"Estadísticas de timeouts: {stats}")
    
    # Verificar que los componentes procesaron eventos según su velocidad
    for comp in components:
        comp_stats = comp.get_stats()
        logger.info(f"Estadísticas del componente {comp.name}: {comp_stats}")
        
        # El componente más rápido debería haber procesado todos los eventos
        if comp.name == "fast":
            assert comp_stats["events_processed"] >= total_events, \
                f"El componente rápido debería haber procesado al menos {total_events} eventos"
        
        # Todos los componentes deberían haber procesado al menos algunos eventos
        assert comp_stats["events_processed"] > 0, \
            f"El componente {comp.name} debería haber procesado al menos algunos eventos"
    
    # Ajustar timeouts basado en estadísticas
    engine.adjust_timeouts_based_on_stats()
    
    # Detener el motor
    await engine.stop()
    
    # Verificar que el motor se detuvo correctamente
    assert not engine.running, "El motor debería estar detenido"