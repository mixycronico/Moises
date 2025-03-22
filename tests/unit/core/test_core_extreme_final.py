"""
Prueba básica final que verifica el comportamiento extremo del sistema.

Esta prueba se centra en confirmar que el core puede manejar adecuadamente
situaciones extremas como fallos, bloqueos y alto volumen de eventos.
"""

import pytest
import asyncio
import logging
import time
import random
from typing import Dict, Any, List, Optional, Set, Tuple
from unittest.mock import patch

from genesis.core.component import Component
from genesis.core.engine_non_blocking import EngineNonBlocking

# Configurar logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class ExtremeComponent(Component):
    """Componente para pruebas extremas."""
    
    def __init__(self, name: str, 
                 fail_on_start: bool = False, 
                 fail_on_stop: bool = False,
                 block_after_events: int = 0,
                 fail_probability: float = 0):
        """
        Inicializar componente para pruebas extremas.
        
        Args:
            name: Nombre del componente
            fail_on_start: Si es True, el componente fallará al iniciar
            fail_on_stop: Si es True, el componente fallará al detener
            block_after_events: Número de eventos después del cual el componente se bloqueará
            fail_probability: Probabilidad de fallo al procesar un evento
        """
        super().__init__(name)
        self.events: List[Dict[str, Any]] = []
        self.fail_on_start = fail_on_start
        self.fail_on_stop = fail_on_stop
        self.block_after_events = block_after_events
        self.fail_probability = fail_probability
        self.processed_ids = set()
        self.start_attempts = 0
        self.stop_attempts = 0
        self.failure_count = 0
        self.block_time = 0.3  # Tiempo de bloqueo (en segundos)
    
    async def start(self) -> None:
        """Iniciar componente, posiblemente fallando."""
        self.start_attempts += 1
        logger.info(f"Iniciando componente extremo {self.name} (intento #{self.start_attempts})")
        
        if self.fail_on_start:
            raise RuntimeError(f"Error simulado al iniciar componente {self.name}")
    
    async def stop(self) -> None:
        """Detener componente, posiblemente fallando."""
        self.stop_attempts += 1
        logger.info(f"Deteniendo componente extremo {self.name} (intento #{self.stop_attempts})")
        
        if self.fail_on_stop:
            raise RuntimeError(f"Error simulado al detener componente {self.name}")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """
        Manejar un evento, posiblemente fallando o bloqueándose.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Opcional, respuesta al evento
            
        Raises:
            Exception: Si el componente está configurado para fallar
        """
        # Registrar evento
        event_record = {
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": time.time()
        }
        self.events.append(event_record)
        
        # Registrar ID si existe
        if "id" in data:
            self.processed_ids.add(data["id"])
        
        # Verificar si debe fallar aleatoriamente
        if random.random() < self.fail_probability:
            self.failure_count += 1
            raise RuntimeError(f"Error aleatorio procesando evento {event_type} en {self.name}")
        
        # Verificar si debe bloquearse
        if self.block_after_events > 0 and len(self.events) >= self.block_after_events:
            # Bloquear por un tiempo suficiente para activar timeout en test_mode
            logger.warning(f"Componente {self.name} bloqueándose por {self.block_time}s después de {len(self.events)} eventos")
            await asyncio.sleep(self.block_time)
        
        return {"processed_by": self.name, "event_count": len(self.events)}
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas del componente.
        
        Returns:
            Diccionario con métricas
        """
        return {
            "event_count": len(self.events),
            "unique_ids": len(self.processed_ids),
            "start_attempts": self.start_attempts,
            "stop_attempts": self.stop_attempts,
            "failure_count": self.failure_count
        }


@pytest.mark.asyncio
async def test_core_resilience_with_failing_components():
    """
    Prueba que verifica que el motor puede manejar componentes que fallan.
    
    Este test verifica:
    1. Componentes que fallan al iniciar
    2. Componentes que fallan al procesar eventos
    3. Componentes que se bloquean por largo tiempo
    """
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componentes con diferentes problemas
    standard_comp = ExtremeComponent("standard")
    fail_start_comp = ExtremeComponent("fail_start", fail_on_start=True)
    flaky_comp = ExtremeComponent("flaky", fail_probability=0.3)
    blocking_comp = ExtremeComponent("blocking", block_after_events=5)
    
    # Registrar componentes
    engine.register_component(standard_comp)
    engine.register_component(fail_start_comp)
    engine.register_component(flaky_comp)
    engine.register_component(blocking_comp)
    
    # Iniciar motor (debería continuar incluso con un componente que falla al iniciar)
    await engine.start()
    
    # Verificar que el motor inició correctamente
    assert engine.running, "El motor debería estar en ejecución a pesar de tener un componente que falló al iniciar"
    
    # Verificar que el componente normal inició correctamente
    assert standard_comp.start_attempts == 1, "El componente estándar debería haberse iniciado"
    
    # Verificar que el componente problemático intentó iniciarse
    assert fail_start_comp.start_attempts >= 1, "El componente con fallo en start debería haber intentado iniciarse"
    
    # Enviar algunos eventos al sistema
    num_events = 10
    for i in range(num_events):
        event_data = {"id": i, "value": f"test_{i}"}
        await engine.emit_event(f"test_event_{i}", event_data, "test")
        await asyncio.sleep(0.05)  # Pequeña pausa
    
    # Esperar un tiempo para el procesamiento
    await asyncio.sleep(0.5)
    
    # Verificar que el componente estándar procesó todos los eventos
    standard_metrics = standard_comp.get_metrics()
    assert standard_metrics["unique_ids"] == num_events, f"El componente estándar debería haber procesado {num_events} eventos únicos, procesó {standard_metrics['unique_ids']}"
    
    # Verificar que el componente intermitente procesó eventos (con algunos fallos)
    flaky_metrics = flaky_comp.get_metrics()
    assert flaky_metrics["event_count"] > 0, "El componente intermitente debería haber procesado al menos algunos eventos"
    assert flaky_metrics["failure_count"] > 0, "El componente intermitente debería haber registrado algunos fallos"
    
    # Verificar que el componente con bloqueo procesó algunos eventos hasta bloquearse
    blocking_metrics = blocking_comp.get_metrics()
    assert blocking_metrics["event_count"] >= blocking_comp.block_after_events, "El componente con bloqueo debería haber alcanzado su umbral de bloqueo"
    
    # Detener el motor
    await engine.stop()
    
    # Verificar que el componente normal se detuvo
    assert standard_comp.stop_attempts == 1, "El componente estándar debería haberse detenido"


@pytest.mark.asyncio
async def test_core_recover_from_component_crash():
    """
    Prueba que el sistema se recupera correctamente después de un fallo severo.
    
    Esta prueba simula una situación donde un componente "crítico" falla
    catastroficamente, pero el sistema debe poder recuperarse y continuar
    funcionando con los componentes restantes.
    """
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componentes normales que deben seguir funcionando
    comp_a = ExtremeComponent("comp_a")
    comp_b = ExtremeComponent("comp_b")
    
    # Crear componente que fallará catastróficamente
    crashing_comp = ExtremeComponent("crasher", block_after_events=3, fail_probability=1.0)
    
    # Registrar componentes
    engine.register_component(comp_a)
    engine.register_component(comp_b)
    engine.register_component(crashing_comp)
    
    # Iniciar motor
    await engine.start()
    
    # Enviar eventos iniciales (menos de los necesarios para que falle el componente)
    for i in range(2):
        await engine.emit_event(f"initial_event_{i}", {"id": i}, "test")
        await asyncio.sleep(0.05)
    
    # Verificar que todos los componentes procesaron los eventos iniciales
    assert comp_a.get_metrics()["event_count"] == 2, "El componente A debería haber procesado 2 eventos iniciales"
    assert comp_b.get_metrics()["event_count"] == 2, "El componente B debería haber procesado 2 eventos iniciales"
    assert crashing_comp.get_metrics()["event_count"] == 2, "El componente crasher debería haber procesado 2 eventos iniciales"
    
    # Enviar evento que hará que el componente crasher falle
    await engine.emit_event("crash_trigger", {"id": 99, "crash": True}, "test")
    await asyncio.sleep(0.1)
    
    # Verificar que el componente crasher recibió el evento
    assert crashing_comp.get_metrics()["event_count"] == 3, "El componente crasher debería haber recibido el evento de trigger"
    
    # Enviar más eventos después del fallo
    for i in range(5):
        await engine.emit_event(f"post_crash_event_{i}", {"id": 100 + i}, "test")
        await asyncio.sleep(0.05)
    
    # Esperar que se procesen los eventos
    await asyncio.sleep(0.2)
    
    # Verificar que los componentes normales siguieron funcionando
    assert comp_a.get_metrics()["event_count"] == 8, "El componente A debería haber procesado todos los eventos"
    assert comp_b.get_metrics()["event_count"] == 8, "El componente B debería haber procesado todos los eventos"
    
    # Detener motor
    await engine.stop()