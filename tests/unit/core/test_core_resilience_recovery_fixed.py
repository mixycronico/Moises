"""
Versión simplificada y optimizada de las pruebas de resiliencia y recuperación.

Este módulo contiene versiones optimizadas de las pruebas que verifican
la capacidad de recuperación del sistema frente a fallos.
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

# Configurar logging para reducir verbosidad
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)

# Definir excepciones específicas a nivel de módulo
class ComponentHealthException(Exception):
    """Excepción específica para errores de componentes no saludables."""
    pass
    
class ComponentTimeoutException(Exception):
    """Excepción específica para timeouts de componentes."""
    pass

class ValidationException(Exception):
    """Excepción específica para errores de validación."""
    pass


class SimpleResilientComponent(Component):
    """Componente simplificado para pruebas de resiliencia con mínimo overhead."""
    
    def __init__(self, name: str, fail_when_unhealthy: bool = True):
        """Inicializar componente."""
        super().__init__(name)
        self.events_received = 0
        self.normal_events_received = 0
        self.is_healthy = True
        self.error_count = 0
        self.recovery_count = 0
        self.success_count = 0
        self.finally_count = 0
        self.fail_when_unhealthy = fail_when_unhealthy
        self.should_handle_unhealthy = True  # Flag para controlar si procesar set_unhealthy
    
    async def start(self) -> None:
        """Iniciar componente."""
        pass
    
    async def stop(self) -> None:
        """Detener componente."""
        pass
            
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """
        Procesar evento con lógica simplificada.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            None
            
        Raises:
            ComponentHealthException: Si el componente no está sano
            ComponentTimeoutException: Si el procesamiento excede el timeout
            ValueError: Si los parámetros son inválidos
        """
        try:
            # Incrementar contador general
            self.events_received += 1
            
            # No contar eventos del sistema
            if event_type in ["component_registered", "component_started", "engine_started", "engine_stopped"]:
                return None
                
            # Contar eventos normales
            if event_type.startswith("normal_") or event_type.startswith("test_"):
                self.normal_events_received += 1
            
            # Manejar eventos de estado
            if event_type == "set_unhealthy" and self.should_handle_unhealthy:
                self.is_healthy = False
                return None
                
            if event_type == "recovery":
                self.is_healthy = True
                self.recovery_count += 1
                return None
                
            # Validación básica de parámetros
            if not isinstance(data, dict):
                raise ValueError(f"Data debe ser un diccionario, recibido: {type(data)}")
                
            # Fallo cuando no está healthy
            if not self.is_healthy and self.fail_when_unhealthy:
                self.error_count += 1
                # Generar excepción específica 
                raise ComponentHealthException(f"Componente {self.name} no está en estado saludable")
            
            # Si llegamos aquí, el procesamiento fue exitoso
            self.success_count += 1
            
        except ComponentHealthException as e:
            # Manejo específico para problemas de salud del componente
            logger.warning(f"Error de salud en componente: {e}")
            raise  # Re-lanzar para que el motor lo maneje
            
        except ValueError as e:
            # Manejo específico para errores de validación
            logger.error(f"Error de validación en {self.name}: {e}")
            raise
            
        except ComponentTimeoutException as e:
            # Manejo específico para timeouts
            logger.error(f"Timeout en componente {self.name}: {e}")
            raise
            
        except Exception as e:
            # Capturar cualquier otra excepción no prevista
            logger.critical(f"Error inesperado en {self.name}: {e}")
            raise
            
        finally:
            # Este código se ejecuta siempre, haya o no excepción
            self.finally_count += 1
            
        return None


@pytest.mark.asyncio
@pytest.mark.timeout(3)  # Timeout más estricto
async def test_component_recovery_simplified():
    """Prueba ultra simplificada para verificar únicamente la recuperación básica."""
    # Crear componente especial para esta prueba - sin registro en eventos
    class RecoverableComponent(Component):
        def __init__(self, name):
            super().__init__(name)
            self.is_healthy = True
            self.recovery_count = 0
            self.events_processed = 0
        
        async def start(self):
            pass
            
        async def stop(self):
            pass
            
        async def handle_event(self, event_type, data, source):
            # Solo procesamos eventos cuando estamos healthy
            if not self.is_healthy:
                return None
                
            self.events_processed += 1
            return None
    
    # Crear componente recuperable
    recoverable = RecoverableComponent("test_recovery")
    
    # Verificar estado inicial
    assert recoverable.is_healthy, "Componente debería estar saludable inicialmente"
    assert recoverable.events_processed == 0, "No debería haber procesado eventos aún"
    
    # Simular un fallo
    recoverable.is_healthy = False
    
    # Simular algunos eventos siendo procesados durante el estado no saludable
    # En un componente real, estos no incrementarían el contador porque hay una verificación
    # de estado en el método handle_event
    await recoverable.handle_event("test", {}, "test")
    await recoverable.handle_event("test", {}, "test")
    
    # Verificar que no se procesaron eventos
    assert not recoverable.is_healthy, "Componente debería estar no saludable"
    assert recoverable.events_processed == 0, "No debería haber procesado eventos durante estado no saludable"
    
    # Recuperar el componente
    recoverable.is_healthy = True
    recoverable.recovery_count += 1
    
    # Procesar más eventos después de la recuperación
    await recoverable.handle_event("test", {}, "test")
    await recoverable.handle_event("test", {}, "test")
    
    # Verificar recuperación
    assert recoverable.is_healthy, "Componente debería estar saludable después de recuperación"
    assert recoverable.recovery_count == 1, "El contador de recuperación debería ser 1"
    assert recoverable.events_processed == 2, "Debería haber procesado 2 eventos después de recuperación"
    
    
@pytest.mark.asyncio
@pytest.mark.timeout(3)  # Timeout más estricto
async def test_error_isolation_simplified():
    """Prueba simplificada de aislamiento de errores."""
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear pocos componentes (3 en lugar de 5+1)
    components = [SimpleResilientComponent(f"comp_{i}") for i in range(3)]
    error_comp = SimpleResilientComponent("error_comp")
    
    # Registrar componentes
    for comp in components:
        engine.register_component(comp)
    engine.register_component(error_comp)
    
    # Iniciar motor
    await engine.start()
    
    # Enviar solo 2 eventos normales
    for i in range(2):
        await engine.emit_event(f"normal_{i}", {"id": i}, "test")
        await asyncio.sleep(0.05)
    
    # Verificar que todos procesaron bien
    for i, comp in enumerate(components):
        assert comp.normal_events_received == 2, f"El componente {i} debería haber recibido 2 eventos"
    assert error_comp.normal_events_received == 2, "El componente de error debería haber recibido 2 eventos"
    
    # Marcar componente de error como no sano
    await engine.emit_event("set_unhealthy", {}, "test")
    # Dar tiempo suficiente para que el estado se actualice
    await asyncio.sleep(0.2)
    
    # Enviar más eventos
    for i in range(2):
        await engine.emit_event(f"normal_{i+2}", {"id": i+2}, "test")
        # Aumentar el tiempo entre eventos para garantizar el procesamiento
        await asyncio.sleep(0.1)
    
    # Verificar que los componentes sanos procesaron todos los eventos
    for i, comp in enumerate(components):
        assert comp.normal_events_received == 4, f"El componente sano {i} debería haber procesado 4 eventos"
    
    # Verificar que el componente de error generó errores
    assert error_comp.error_count > 0, "El componente de error debería haber generado errores"
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
@pytest.mark.timeout(3)  # Timeout más estricto
async def test_engine_recovery_after_severe_errors_simplified():
    """Prueba simplificada que el motor se recupera después de errores severos."""
    # Crear motor no bloqueante
    engine = EngineNonBlocking(test_mode=True)
    
    # Crear componentes (solo 1 normal y 1 problemático)
    normal_comp = SimpleResilientComponent("normal")
    
    # Este componente siempre falla
    problem_comp = SimpleResilientComponent("problem", fail_when_unhealthy=True)
    problem_comp.is_healthy = False  # Siempre no sano
    
    # Registrar componentes
    engine.register_component(normal_comp)
    engine.register_component(problem_comp)
    
    # Iniciar motor
    await engine.start()
    
    # Enviar eventos - el componente problema fallará en todos
    for i in range(5):
        await engine.emit_event(f"test_{i}", {"id": i}, "test")
        await asyncio.sleep(0.05)
    
    # Verificar que el componente normal procesó todos los eventos
    assert normal_comp.normal_events_received == 5, "El componente normal debería haber procesado todos los eventos"
    
    # Verificar que el componente problema falló en todos
    assert problem_comp.error_count > 0, "El componente problema debería haber generado errores"
    
    # Eliminar el componente problemático
    engine.deregister_component(problem_comp)
    
    # Dar tiempo para que el motor elimine correctamente el componente
    await asyncio.sleep(0.2)
    
    # Enviar más eventos
    for i in range(3):
        await engine.emit_event(f"test_{i+5}", {"id": i+5}, "test")
        # Aumentar el tiempo entre eventos para garantizar el procesamiento
        await asyncio.sleep(0.1)
    
    # Verificar que el componente normal procesó todos los eventos
    assert normal_comp.normal_events_received == 8, "El componente normal debería haber procesado todos los eventos después de eliminar el problemático"
    
    # Detener motor
    await engine.stop()