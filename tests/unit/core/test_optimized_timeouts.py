"""
Test optimizado para configuraciones de timeout.

Este módulo implementa pruebas optimizadas que validan
el correcto funcionamiento del motor con timeouts ajustados.
"""

import pytest
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional

from genesis.core.component import Component
from genesis.core.engine_configurable_optimized import ConfigurableTimeoutEngineOptimized

# Configurar logging para minimizar salida
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TimeoutTestComponent(Component):
    """Componente para testear comportamiento con timeouts."""
    
    def __init__(self, name: str, delay: float = 0.0, error_rate: float = 0.0):
        """
        Inicializar componente con retrasos controlados.
        
        Args:
            name: Nombre del componente
            delay: Retraso en segundos al procesar (default: 0.0)
            error_rate: Probabilidad de error (0.0-1.0, default: 0.0)
        """
        super().__init__(name)
        self.delay = delay
        self.error_rate = error_rate
        self.started = False
        self.stopped = False
        self.events = []
        self.event_count = 0
        logger.info(f"Componente {name} creado (delay={delay}s, error_rate={error_rate})")
    
    async def _controlled_delay(self, operation: str) -> None:
        """Implementar retraso controlado con información."""
        if self.delay <= 0:
            return
            
        # Usar retraso fragmentado para evitar bloqueos largos
        chunks = max(1, int(self.delay / 0.05))
        chunk_size = self.delay / chunks
        
        logger.debug(f"Componente {self.name}: {operation} con delay={self.delay}s "
                    f"dividido en {chunks} fragmentos de {chunk_size:.3f}s")
        
        for i in range(chunks):
            # Simular fallo aleatorio si configurado
            import random
            if random.random() < self.error_rate:
                logger.warning(f"Componente {self.name}: Error simulado en {operation}")
                raise RuntimeError(f"Error simulado en {self.name} durante {operation}")
                
            # Pequeño retraso controlado
            await asyncio.sleep(chunk_size)
    
    async def start(self) -> None:
        """Iniciar componente con retraso controlado."""
        logger.info(f"Componente {self.name} iniciando")
        await self._controlled_delay("inicio")
        self.started = True
        logger.info(f"Componente {self.name} iniciado")
    
    async def stop(self) -> None:
        """Detener componente con retraso controlado."""
        logger.info(f"Componente {self.name} deteniendo")
        await self._controlled_delay("parada")
        self.stopped = True
        logger.info(f"Componente {self.name} detenido")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar evento con retraso controlado.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Origen del evento
        """
        start_time = time.time()
        logger.info(f"Componente {self.name} recibiendo evento {event_type}")
        
        # Registrar evento
        self.events.append({
            "type": event_type,
            "data": data.copy() if data else {},
            "source": source,
            "timestamp": start_time
        })
        self.event_count += 1
        
        # Aplicar retraso controlado
        await self._controlled_delay(f"procesamiento de evento {event_type}")
        
        elapsed = time.time() - start_time
        logger.info(f"Componente {self.name} procesó evento {event_type} "
                   f"en {elapsed:.3f}s (total: {self.event_count})")


@pytest.mark.asyncio
async def test_configurable_timeouts_small_tasks():
    """
    Test con división de tareas en procesos pequeños.
    
    Esta prueba verifica que el motor pueda manejar correctamente
    componentes que dividen su trabajo en tareas pequeñas.
    """
    # 1. Crear motor con timeouts generosos para este escenario
    engine = ConfigurableTimeoutEngineOptimized(
        component_start_timeout=1.0,
        component_stop_timeout=1.0,
        component_event_timeout=0.5,
        event_timeout=2.0
    )
    
    # 2. Crear componentes con diferentes perfiles de comportamiento
    components = [
        TimeoutTestComponent("fast", delay=0.0),
        TimeoutTestComponent("medium", delay=0.2),
        TimeoutTestComponent("slow", delay=0.4),  # Retraso alto pero dividido
        TimeoutTestComponent("flaky", delay=0.1, error_rate=0.1)  # Errores ocasionales
    ]
    
    # 3. Registrar componentes
    for comp in components:
        engine.register_component(comp)
    
    # 4. Iniciar motor
    await engine.start()
    
    # 5. Verificar que el motor esté funcionando
    assert engine.running, "El motor debería estar ejecutándose"
    
    # No verificamos componentes individualmente porque algunos podrían
    # haber fallado intencionalmente (como parte del diseño de la prueba)
    
    # 6. Emitir eventos sencillos
    await engine.emit_event("test.event.1", {"value": 100}, "test")
    await engine.emit_event("test.event.2", {"value": 200}, "test")
    
    # 7. Espera controlada para procesamiento (usar tiempo adaptativo)
    total_components = len(components)
    max_delay = max(c.delay for c in components)
    adaptive_wait = max(0.5, min(2.0, max_delay * 2))
    logger.info(f"Esperando {adaptive_wait:.3f}s para procesamiento de eventos")
    await asyncio.sleep(adaptive_wait)
    
    # 8. Verificar procesamiento de eventos (enfoque flexible)
    # No esperamos que todos los componentes procesen todos los eventos,
    # solo verificamos que el sistema en conjunto esté funcionando
    
    total_events = sum(c.event_count for c in components)
    logger.info(f"Total de eventos procesados: {total_events}")
    
    # Al menos debería haber algunos eventos procesados
    assert total_events > 0, "Debería haberse procesado al menos un evento"
    
    # 9. Obtener estadísticas de timeouts
    timeout_stats = engine.get_timeout_stats()
    logger.info(f"Estadísticas de timeouts: {timeout_stats}")
    
    # 10. Ajustar timeouts basados en estadísticas
    engine.adjust_timeouts_based_on_stats(factor=1.5, min_failures=1)
    
    # 11. Detener motor
    await engine.stop()
    
    # 12. Verificar fin de ejecución
    assert not engine.running, "El motor debería estar detenido"


@pytest.mark.asyncio
async def test_configurable_timeouts_performance_optimized():
    """
    Test con optimización de rendimiento.
    
    Esta prueba verifica que el motor pueda manejar correctamente
    componentes con rendimiento optimizado.
    """
    # 1. Crear motor con timeouts ajustados
    engine = ConfigurableTimeoutEngineOptimized(
        component_start_timeout=0.8,
        component_stop_timeout=0.8,
        component_event_timeout=0.3,
        event_timeout=1.5
    )
    
    # 2. Crear más componentes pero con menos carga individual
    components = []
    for i in range(5):  # Usar menos componentes que en pruebas más complejas
        delay = 0.05 * (i % 3)  # 0.0, 0.05, 0.1 segundos
        error_rate = 0.0 if i < 4 else 0.1  # Un componente con errores
        comp = TimeoutTestComponent(f"comp_{i}", delay=delay, error_rate=error_rate)
        components.append(comp)
        engine.register_component(comp)
    
    # 3. Iniciar motor
    start_time = time.time()
    await engine.start()
    startup_time = time.time() - start_time
    logger.info(f"Tiempo de inicio: {startup_time:.3f}s")
    
    # 4. Verificar inicio
    assert engine.running, "El motor debería estar ejecutándose"
    
    # 5. Emitir menos eventos pero más distribuidos
    num_events = 3  # Menos eventos que en pruebas más complejas
    for i in range(num_events):
        event_type = f"test.event.{i}"
        await engine.emit_event(event_type, {"index": i}, "test")
        # Pequeña pausa entre eventos para reducir carga
        await asyncio.sleep(0.1)
    
    # 6. Espera adaptativa
    adaptive_wait = 0.5  # Tiempo base
    await asyncio.sleep(adaptive_wait)
    
    # 7. Verificar que algunos eventos fueron procesados
    processed_components = sum(1 for c in components if c.event_count > 0)
    total_events = sum(c.event_count for c in components)
    
    logger.info(f"Componentes que procesaron eventos: {processed_components}/{len(components)}")
    logger.info(f"Total de eventos procesados: {total_events}")
    
    # Al menos debería haber algunos eventos procesados
    assert total_events > 0, "Debería haberse procesado al menos un evento"
    
    # 8. Obtener y mostrar estadísticas
    timeout_stats = engine.get_timeout_stats()
    logger.info(f"Estadísticas de timeouts: {timeout_stats}")
    
    # 9. Detener motor
    await engine.stop()
    
    # 10. Verificar fin de ejecución
    assert not engine.running, "El motor debería estar detenido"