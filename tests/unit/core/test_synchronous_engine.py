"""
Pruebas unitarias para el motor síncrono del sistema Genesis.

Este módulo implementa pruebas para verificar el funcionamiento del motor síncrono,
que evita deadlocks mediante un bucle de actualización centralizado.
"""

import pytest
import time
import threading
from typing import Dict, Any, List, Optional

from genesis.core.synchronous_engine import SynchronousEngine

# Componente de prueba
class TestComponent:
    """Componente simple para pruebas."""
    
    def __init__(self, component_id: str):
        """Inicializar componente con ID."""
        self.id = component_id
        self.events = []
        self.started = False
        self.stopped = False
        self.updated = False
        self.update_count = 0
        
    def start(self):
        """Iniciar componente."""
        self.started = True
        return None  # Retorno explícito para compatibilidad
        
    def stop(self):
        """Detener componente."""
        self.stopped = True
        return None  # Retorno explícito para compatibilidad
        
    def update(self):
        """Actualizar componente."""
        self.updated = True
        self.update_count += 1
        
    def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """Manejar evento, registrándolo."""
        self.events.append((event_type, data, source))
        
        # Si se solicita respuesta, proporcionar una
        if data and "_response_to" in data:
            return {
                "result": f"Response from {self.id} to {event_type}",
                "component_id": self.id
            }
        return None

# Componente que simula procesos lentos
class SlowComponent(TestComponent):
    """Componente que realiza operaciones lentas."""
    
    def __init__(self, component_id: str, delay: float = 0.05):
        """Inicializar componente con retardo."""
        super().__init__(component_id)
        self.delay = delay
        
    def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """Manejar evento con retardo."""
        time.sleep(self.delay)  # Simular operación lenta
        return super().handle_event(event_type, data, source)
        
    def update(self):
        """Actualizar con retardo."""
        time.sleep(self.delay / 2)  # Mitad del retardo para updates
        super().update()

# Componente que simula fallos
class FailingComponent(TestComponent):
    """Componente que falla en ciertas condiciones."""
    
    def __init__(self, component_id: str, fail_events: List[str] = None):
        """Inicializar componente con eventos que fallan."""
        super().__init__(component_id)
        self.fail_events = fail_events or ["fail_test"]
        self.fail_updates = False
        self.fail_count = 0
        
    def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """Manejar evento, fallando si corresponde."""
        if event_type in self.fail_events:
            self.fail_count += 1
            raise ValueError(f"Fallo simulado en {self.id} para evento {event_type}")
        return super().handle_event(event_type, data, source)
        
    def update(self):
        """Actualizar, fallando si corresponde."""
        if self.fail_updates:
            self.fail_count += 1
            raise ValueError(f"Fallo simulado en update de {self.id}")
        super().update()

# Pruebas para el motor síncrono
def test_engine_initialization():
    """Probar inicialización básica del motor."""
    engine = SynchronousEngine(tick_rate=0.01)
    assert not engine.running
    assert len(engine.components) == 0
    assert len(engine.event_buffer) == 0

def test_component_registration():
    """Probar registro de componentes."""
    engine = SynchronousEngine(tick_rate=0.01)
    
    # Registrar componentes
    comp_a = TestComponent("a")
    comp_b = TestComponent("b")
    
    engine.register_component("a", comp_a)
    engine.register_component("b", comp_b)
    
    # Verificar registro
    assert len(engine.components) == 2
    assert "a" in engine.components
    assert "b" in engine.components
    assert engine.components["a"] == comp_a
    assert engine.components["b"] == comp_b
    
def test_component_with_dependencies():
    """Probar registro de componentes con dependencias."""
    engine = SynchronousEngine(tick_rate=0.01)
    
    # Registrar componentes con dependencias
    comp_a = TestComponent("a")
    comp_b = TestComponent("b")
    comp_c = TestComponent("c")
    
    engine.register_component("a", comp_a)
    engine.register_component("b", comp_b, depends_on=["a"])
    engine.register_component("c", comp_c, depends_on=["a", "b"])
    
    # Verificar dependencias
    assert "a" in engine.component_dependencies["b"]
    assert "a" in engine.component_dependencies["c"]
    assert "b" in engine.component_dependencies["c"]
    
def test_component_priorities():
    """Probar prioridades de componentes."""
    engine = SynchronousEngine(tick_rate=0.01)
    
    # Registrar componentes con prioridades
    comp_a = TestComponent("a")
    comp_b = TestComponent("b")
    comp_c = TestComponent("c")
    
    engine.register_component("a", comp_a, priority=30)  # Menor = mayor prioridad
    engine.register_component("b", comp_b, priority=20)  # Mayor prioridad que a
    engine.register_component("c", comp_c, priority=10)  # Mayor prioridad que b
    
    # Verificar orden
    order = engine._get_component_order()
    
    # Las prioridades deberían determinar el orden (menor primero)
    assert order.index("c") < order.index("b")
    assert order.index("b") < order.index("a")
    
def test_component_start_stop():
    """Probar inicio y detención de componentes."""
    engine = SynchronousEngine(tick_rate=0.01)
    
    # Registrar componentes 
    comp_a = TestComponent("a")
    comp_b = TestComponent("b")
    
    engine.register_component("a", comp_a)
    engine.register_component("b", comp_b)
    
    # Iniciar (threaded=False para que bloquee y sea predecible en tests)
    # Iniciar en un hilo para no bloquear
    def run_engine():
        engine.start(threaded=False)
        
    thread = threading.Thread(target=run_engine)
    thread.daemon = True
    thread.start()
    
    # Esperar a que inicie
    time.sleep(0.1)
    
    # Verificar que está en ejecución
    assert engine.running
    assert comp_a.started
    assert comp_b.started
    
    # Detener
    engine.stop()
    
    # Esperar a que se detenga
    thread.join(timeout=1.0)
    
    # Verificar que se detuvo
    assert not engine.running
    assert comp_a.stopped
    assert comp_b.stopped

def test_event_emission():
    """Probar emisión y recepción de eventos."""
    engine = SynchronousEngine(tick_rate=0.01)
    
    # Registrar componentes
    comp_a = TestComponent("a")
    comp_b = TestComponent("b")
    
    engine.register_component("a", comp_a)
    engine.register_component("b", comp_b)
    
    # Iniciar en un hilo
    engine.start(threaded=True)
    
    # Esperar a que inicie
    time.sleep(0.1)
    
    # Emitir evento
    test_data = {"value": 42, "timestamp": time.time()}
    engine.emit("test_event", test_data, "a")
    
    # Esperar procesamiento
    time.sleep(0.1)
    
    # Verificar que b recibió el evento (a no, porque es la fuente)
    assert len(comp_a.events) == 0
    assert len(comp_b.events) == 1
    assert comp_b.events[0][0] == "test_event"
    assert comp_b.events[0][1]["value"] == 42
    assert comp_b.events[0][2] == "a"
    
    # Detener
    engine.stop()
    
def test_event_with_response():
    """Probar eventos con respuesta."""
    engine = SynchronousEngine(tick_rate=0.01)
    
    # Registrar componentes
    comp_a = TestComponent("a")
    comp_b = TestComponent("b")
    comp_c = TestComponent("c")
    
    engine.register_component("a", comp_a)
    engine.register_component("b", comp_b)
    engine.register_component("c", comp_c)
    
    # Iniciar en un hilo
    engine.start(threaded=True)
    
    # Esperar a que inicie
    time.sleep(0.1)
    
    # Emitir evento con respuesta
    responses = engine.emit_with_response("query_event", {"param": "test"}, "a")
    
    # Verificar respuestas
    assert len(responses) == 2  # Debe haber respuestas de b y c
    
    # Extraer componentes que respondieron
    responders = [r["component"] for r in responses]
    assert "b" in responders
    assert "c" in responders
    
    # Verificar contenido
    for response in responses:
        assert "component" in response
        assert "response" in response
        assert "result" in response["response"]
        assert "component_id" in response["response"]
        
    # Detener
    engine.stop()
    
def test_slow_component_handling():
    """Probar manejo de componentes lentos."""
    engine = SynchronousEngine(tick_rate=0.01)
    
    # Registrar componentes, incluyendo uno lento
    comp_a = TestComponent("a")
    comp_slow = SlowComponent("slow", delay=0.05)  # 50ms de retardo
    
    engine.register_component("a", comp_a)
    engine.register_component("slow", comp_slow)
    
    # Iniciar en un hilo
    engine.start(threaded=True)
    
    # Esperar a que inicie
    time.sleep(0.1)
    
    # Emitir varios eventos para estresar el sistema
    for i in range(10):
        engine.emit(f"test_event_{i}", {"sequence": i}, "a")
        
    # Esperar procesamiento
    time.sleep(0.5)
    
    # Verificar que el componente lento procesó eventos
    assert len(comp_slow.events) > 0
    
    # Obtener estado
    status = engine.get_status()
    
    # Verificar que hubo ticks
    assert status["tick_count"] > 0
    
    # Detener
    engine.stop()
    
def test_error_handling():
    """Probar manejo de errores en componentes."""
    engine = SynchronousEngine(tick_rate=0.01)
    
    # Registrar un componente que falla
    comp_normal = TestComponent("normal")
    comp_failing = FailingComponent("failing", fail_events=["fail_test"])
    
    engine.register_component("normal", comp_normal)
    engine.register_component("failing", comp_failing)
    
    # Iniciar en un hilo
    engine.start(threaded=True)
    
    # Esperar a que inicie
    time.sleep(0.1)
    
    # Emitir evento que causará fallo
    engine.emit("fail_test", {"should_fail": True}, "normal")
    
    # Emitir evento normal para verificar que se sigue procesando
    engine.emit("normal_event", {"should_process": True}, "normal")
    
    # Esperar procesamiento
    time.sleep(0.2)
    
    # Verificar que el componente que falla registró el fallo
    assert comp_failing.fail_count > 0
    
    # Verificar que también procesó eventos normales
    normal_events = [e for e in comp_failing.events if e[0] == "normal_event"]
    assert len(normal_events) == 1
    
    # Detener
    engine.stop()
    
def test_component_update():
    """Probar actualización periódica de componentes."""
    engine = SynchronousEngine(tick_rate=0.01)
    
    # Registrar componentes
    comp_a = TestComponent("a")
    
    engine.register_component("a", comp_a)
    
    # Iniciar en un hilo
    engine.start(threaded=True)
    
    # Esperar varias actualizaciones
    time.sleep(0.1)  # Debería haber ~10 ticks
    
    # Verificar que se llamó a update
    assert comp_a.updated
    assert comp_a.update_count > 0
    
    # Detener
    engine.stop()
    
def test_component_health_monitoring():
    """Probar monitoreo de estado de componentes."""
    engine = SynchronousEngine(tick_rate=0.01)
    
    # Registrar componentes, incluyendo uno que falla
    comp_normal = TestComponent("normal")
    comp_failing = FailingComponent("failing", fail_events=["health_test"])
    
    engine.register_component("normal", comp_normal)
    engine.register_component("failing", comp_failing)
    
    # Configurar un callback para cambios de estado
    status_changes = []
    def status_callback(component_id, metadata):
        status_changes.append((component_id, metadata["healthy"]))
    
    engine.register_status_callback(status_callback)
    
    # Iniciar en un hilo
    engine.start(threaded=True)
    
    # Esperar a que inicie
    time.sleep(0.1)
    
    # Provocar varios fallos
    for _ in range(10):
        engine.emit("health_test", {}, "normal")
        time.sleep(0.02)
        
    # Esperar a que se detecte el problema
    time.sleep(0.2)
    
    # Verificar estado de salud
    status = engine.get_status()
    assert status["components"]["failing"]["errors"] > 0
    
    # Verificar callbacks de estado
    failing_status = [s for s in status_changes if s[0] == "failing"]
    assert len(failing_status) > 0
    
    # Detener
    engine.stop()
    
def test_component_restart():
    """Probar reinicio de componentes."""
    engine = SynchronousEngine(tick_rate=0.01)
    
    # Registrar componente
    comp_a = TestComponent("a")
    
    engine.register_component("a", comp_a)
    
    # Iniciar en un hilo
    engine.start(threaded=True)
    
    # Esperar a que inicie
    time.sleep(0.1)
    
    # Verificar estado inicial
    assert comp_a.started
    assert not comp_a.stopped
    
    # Reiniciar componente
    success = engine.restart_component("a")
    
    # Verificar reinicio
    assert success
    assert comp_a.stopped  # Se llamó a stop
    assert comp_a.started  # Se volvió a llamar a start
    
    # Detener
    engine.stop()
    
def test_event_buffer_limits():
    """Probar límites del buffer de eventos."""
    engine = SynchronousEngine(tick_rate=0.01, max_events_per_tick=5)
    
    # Registrar un componente muy lento
    comp_slow = SlowComponent("slow", delay=0.1)  # 100ms, bloqueará el procesamiento
    
    engine.register_component("slow", comp_slow)
    
    # Iniciar en un hilo
    engine.start(threaded=True)
    
    # Esperar a que inicie
    time.sleep(0.1)
    
    # Emitir muchos eventos rápidamente
    for i in range(50):
        engine.emit(f"flood_event_{i}", {"sequence": i}, "test")
        
    # Verificar buffer lleno
    assert len(engine.event_buffer) > 0
    
    # Esperar algo de procesamiento
    time.sleep(0.3)
    
    # Obtener estado
    status = engine.get_status()
    
    # Debe haber ticks y eventos procesados
    assert status["tick_count"] > 0
    assert status["events_processed"] > 0
    
    # Limpiar buffer
    cleared = engine.clear_event_buffer()
    assert cleared >= 0
    
    # Detener
    engine.stop()
    
def test_system_status():
    """Probar obtención de estado del sistema."""
    engine = SynchronousEngine(tick_rate=0.01)
    
    # Registrar componentes
    comp_a = TestComponent("a")
    comp_b = TestComponent("b")
    
    engine.register_component("a", comp_a)
    engine.register_component("b", comp_b, depends_on=["a"])
    
    # Iniciar en un hilo
    engine.start(threaded=True)
    
    # Esperar a que inicie
    time.sleep(0.1)
    
    # Emitir evento
    engine.emit("status_test", {}, "a")
    
    # Esperar procesamiento
    time.sleep(0.1)
    
    # Obtener estado
    status = engine.get_status()
    
    # Verificar campos del estado
    assert "running" in status
    assert "uptime" in status
    assert "tick_count" in status
    assert "events_processed" in status
    assert "components" in status
    
    # Verificar información de componentes
    assert "a" in status["components"]
    assert "b" in status["components"]
    assert "healthy" in status["components"]["a"]
    assert "dependencies" in status["components"]["b"]
    assert "a" in status["components"]["b"]["dependencies"]
    
    # Detener
    engine.stop()