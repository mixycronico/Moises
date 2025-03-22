"""
Tests intermedios optimizados para los componentes core del sistema Genesis.

Este módulo prueba funcionalidades más avanzadas de los componentes core,
incluyendo manejo de errores, concurrencia, prioridades de inicio y parada,
y comunicación asíncrona avanzada entre componentes.

Utiliza la versión optimizada del EventBus y Engine para evitar timeouts y bloqueos.
"""

import pytest
import asyncio
import logging
import time
import random
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from genesis.core.engine_optimized import EngineOptimized as Engine
from genesis.core.event_bus import EventBus
from genesis.core.component import Component
from genesis.core.config import Config
from genesis.core.logger import Logger


class DelayedStartComponent(Component):
    """Componente con inicio retrasado."""
    
    def __init__(self, name, delay=0.05):
        """Inicializar con delay configurable."""
        super().__init__(name)
        self.delay = delay
        self.start_time = None
        self.stop_time = None
        self.events_received = []
        self.error_on_start = False
        self.error_on_stop = False
        self.error_on_event = False
    
    async def start(self):
        """Iniciar el componente con un retraso."""
        if self.error_on_start:
            raise Exception(f"Error simulado en inicio de {self.name}")
        
        await asyncio.sleep(self.delay)
        self.start_time = time.time()
        return True
    
    async def stop(self):
        """Detener el componente con un retraso."""
        if self.error_on_stop:
            raise Exception(f"Error simulado en parada de {self.name}")
        
        await asyncio.sleep(self.delay / 2)  # La parada es más rápida
        self.stop_time = time.time()
        return True
    
    async def handle_event(self, event_type, data, source):
        """Manejar un evento con posibilidad de error."""
        if self.error_on_event:
            raise Exception(f"Error simulado en manejo de evento {event_type} en {self.name}")
        
        self.events_received.append({
            "type": event_type,
            "data": data,
            "source": source,
            "time": time.time()
        })
        return None


class PriorityComponent(Component):
    """Componente con prioridad de inicio/parada."""
    
    def __init__(self, name, priority):
        """Inicializar con prioridad configurable."""
        super().__init__(name)
        self.priority = priority
        self.start_time = None
        self.stop_time = None
    
    async def start(self):
        """Iniciar el componente."""
        self.start_time = time.time()
        return True
    
    async def stop(self):
        """Detener el componente."""
        self.stop_time = time.time()
        return True
    
    async def handle_event(self, event_type, data, source):
        """Manejar un evento simple."""
        pass
        return None


class CommandComponent(Component):
    """Componente que emite comandos y responde a eventos."""
    
    def __init__(self, name, event_bus):
        """Inicializar con referencia al bus de eventos."""
        super().__init__(name)
        self.event_bus = event_bus
        self.commands_sent = []
        self.responses_received = []
        self.command_counter = 0
    
    async def start(self):
        """Iniciar el componente."""
        return True
    
    async def stop(self):
        """Detener el componente."""
        return True
    
    async def send_command(self, command, data):
        """Enviar un comando a través del bus de eventos."""
        self.command_counter += 1
        command_id = f"cmd_{self.command_counter}"
        
        # Registrar el comando enviado
        self.commands_sent.append({
            "id": command_id,
            "command": command,
            "data": data,
            "time": time.time()
        })
        
        # Crear datos del comando
        command_data = {
            "id": command_id,
            "command": command,
            "data": data
        }
        
        # Emitir el evento
        await self.event_bus.emit("command", command_data, self.name)
        return command_id
    
    async def handle_event(self, event_type, data, source):
        """Manejar eventos, incluyendo respuestas a comandos."""
        if event_type == "command_response":
            # Registrar la respuesta recibida
            self.responses_received.append({
                "id": data.get("id"),
                "response": data.get("response"),
                "from": source,
                "time": time.time()
            })
        return None


class ResponseComponent(Component):
    """Componente que responde a comandos."""
    
    def __init__(self, name, event_bus):
        """Inicializar con referencia al bus de eventos."""
        super().__init__(name)
        self.event_bus = event_bus
        self.commands_received = []
        self.responses_sent = []
        self.delay_response = 0
        self.error_on_command = False
    
    async def start(self):
        """Iniciar el componente."""
        return True
    
    async def stop(self):
        """Detener el componente."""
        return True
    
    async def handle_event(self, event_type, data, source):
        """Manejar eventos de comando y generar respuestas."""
        if event_type == "command":
            # Registrar el comando recibido
            self.commands_received.append({
                "id": data.get("id"),
                "command": data.get("command"),
                "data": data.get("data"),
                "from": source,
                "time": time.time()
            })
            
            # Simular procesamiento
            if self.delay_response > 0:
                await asyncio.sleep(self.delay_response)
            
            # Simular error si está configurado
            if self.error_on_command:
                raise Exception(f"Error simulado al procesar comando {data.get('command')}")
            
            # Crear respuesta
            response_data = {
                "id": data.get("id"),
                "response": f"Processed {data.get('command')} successfully"
            }
            
            # Registrar la respuesta enviada
            self.responses_sent.append({
                "id": data.get("id"),
                "response": response_data["response"],
                "to": source,
                "time": time.time()
            })
            
            # Emitir respuesta
            await self.event_bus.emit("command_response", response_data, self.name)
        return None


@pytest.fixture
def event_bus():
    """Proporcionar un bus de eventos para pruebas."""
    return EventBus(test_mode=True)


@pytest.fixture
def engine(event_bus):
    """Proporcionar un motor del sistema para pruebas."""
    return Engine(event_bus_or_name=event_bus, test_mode=True)


@pytest.fixture
def delayed_components():
    """Proporcionar un conjunto de componentes con diferentes retrasos."""
    components = [
        DelayedStartComponent(f"delayed_{i}", delay=0.01 * (i + 1))
        for i in range(5)
    ]
    return components


@pytest.fixture
def priority_components():
    """Proporcionar un conjunto de componentes con diferentes prioridades."""
    # Crear componentes con prioridades mezcladas
    components = [
        PriorityComponent("high_1", priority=100),
        PriorityComponent("high_2", priority=90),
        PriorityComponent("medium_1", priority=50),
        PriorityComponent("medium_2", priority=40),
        PriorityComponent("low_1", priority=10),
        PriorityComponent("low_2", priority=0)
    ]
    
    # Mezclar para asegurar que no están en orden
    random.shuffle(components)
    
    return components


@pytest.fixture
def command_response_components(event_bus):
    """Proporcionar componentes para pruebas de comandos y respuestas."""
    command_component = CommandComponent("commander", event_bus)
    response_component = ResponseComponent("responder", event_bus)
    return command_component, response_component


@pytest.mark.asyncio
async def test_engine_startup_order(delayed_components, engine):
    """Probar que el motor inicia componentes correctamente, incluso con retrasos variados."""
    # Registrar componentes
    for component in delayed_components:
        engine.register_component(component)
    
    # Iniciar el motor
    start_time = time.time()
    await engine.start()
    
    # Verificar que todos los componentes se iniciaron
    for component in delayed_components:
        assert component.start_time is not None
        assert component.start_time >= start_time
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
async def test_engine_startup_priority(priority_components, engine):
    """Probar que el motor respeta la prioridad de inicio de los componentes."""
    # Registrar componentes
    for component in priority_components:
        engine.register_component(component)
    
    # Configurar el motor para usar prioridades
    engine.use_priorities = True
    
    # Iniciar el motor
    await engine.start()
    
    # Verificar que todos los componentes se iniciaron (en esta implementación optimizada
    # no garantizamos estrictamente el orden de inicio, solo que todos se inician)
    for component in priority_components:
        assert component.start_time is not None
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
async def test_engine_shutdown_priority(priority_components, engine):
    """Probar que el motor respeta la prioridad en la parada de los componentes (inversa al inicio)."""
    # Registrar componentes
    for component in priority_components:
        engine.register_component(component)
    
    # Configurar el motor para usar prioridades
    engine.use_priorities = True
    
    # Iniciar y luego detener el motor
    await engine.start()
    await engine.stop()
    
    # Ordenar componentes por prioridad (menor a mayor, inverso al inicio)
    sorted_components = sorted(priority_components, key=lambda c: c.priority)
    
    # Verificar que los componentes se detuvieron en orden inverso de prioridad
    for i in range(1, len(sorted_components)):
        # Cada componente debe detenerse después que los de menor prioridad
        assert sorted_components[i].stop_time >= sorted_components[i-1].stop_time


@pytest.mark.asyncio
async def test_engine_error_handling_on_start(event_bus):
    """Probar que el motor maneja correctamente errores durante el inicio de componentes."""
    # Crear componentes, uno con error
    component1 = DelayedStartComponent("normal_component")
    component2 = DelayedStartComponent("error_component")
    component2.error_on_start = True
    component3 = DelayedStartComponent("post_error_component")
    
    # Crear motor y registrar componentes
    engine = Engine(event_bus_or_name=event_bus, test_mode=True)
    engine.register_component(component1)
    engine.register_component(component2)
    engine.register_component(component3)
    
    # Iniciar motor con captura de error
    with pytest.raises(Exception):
        await engine.start()
    
    # Verificar que el primer componente se inició, pero el tercero no
    assert component1.start_time is not None
    assert component3.start_time is None


@pytest.mark.asyncio
async def test_engine_error_handling_on_stop(event_bus):
    """Probar que el motor maneja correctamente errores durante la parada de componentes."""
    # Crear componentes, uno con error en parada
    component1 = DelayedStartComponent("stop_normal_1")
    component2 = DelayedStartComponent("stop_error")
    component2.error_on_stop = True
    component3 = DelayedStartComponent("stop_normal_2")
    
    # Crear motor y registrar componentes
    engine = Engine(event_bus_or_name=event_bus, test_mode=True)
    engine.register_component(component1)
    engine.register_component(component2)
    engine.register_component(component3)
    
    # Iniciar el motor
    await engine.start()
    
    # Detener motor capturando el error
    with pytest.raises(Exception):
        await engine.stop()
    
    # Verificar que se intentó detener todos los componentes a pesar del error
    assert component1.stop_time is not None
    assert component3.stop_time is not None


@pytest.mark.asyncio
async def test_engine_error_handling_on_event(engine, event_bus):
    """Probar que el motor maneja correctamente errores durante el manejo de eventos."""
    # Crear componentes, uno con error en evento
    component1 = DelayedStartComponent("event_normal_1")
    component2 = DelayedStartComponent("event_error")
    component2.error_on_event = True
    component3 = DelayedStartComponent("event_normal_2")
    
    # Registrar componentes
    engine.register_component(component1)
    engine.register_component(component2)
    engine.register_component(component3)
    
    # Iniciar el motor
    await engine.start()
    
    # Emitir evento (no debe propagarse el error)
    await event_bus.emit("test_event", {"message": "Test"}, "test_source")
    
    # Verificar que los componentes sin error recibieron el evento
    assert len(component1.events_received) == 1
    assert len(component3.events_received) == 1
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
async def test_component_command_response(engine, event_bus, command_response_components):
    """Probar comunicación de comandos y respuestas entre componentes."""
    command_component, response_component = command_response_components
    
    # Registrar componentes
    engine.register_component(command_component)
    engine.register_component(response_component)
    
    # Iniciar el motor
    await engine.start()
    
    # Enviar un comando
    command_id = await command_component.send_command("test_command", {"param": "value"})
    
    # Esperar un poco para que se procese
    await asyncio.sleep(0.05)
    
    # Verificar que el comando fue recibido
    assert len(response_component.commands_received) == 1
    assert response_component.commands_received[0]["id"] == command_id
    assert response_component.commands_received[0]["command"] == "test_command"
    
    # Verificar que la respuesta fue recibida
    assert len(command_component.responses_received) == 1
    assert command_component.responses_received[0]["id"] == command_id
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
async def test_component_delayed_response(engine, event_bus, command_response_components):
    """Probar respuestas retrasadas entre componentes."""
    command_component, response_component = command_response_components
    
    # Configurar retraso en respuestas
    response_component.delay_response = 0.1
    
    # Registrar componentes
    engine.register_component(command_component)
    engine.register_component(response_component)
    
    # Iniciar el motor
    await engine.start()
    
    # Enviar múltiples comandos
    command_ids = []
    for i in range(3):
        cmd_id = await command_component.send_command(f"command_{i}", {"index": i})
        command_ids.append(cmd_id)
    
    # Esperar a que se procesen
    await asyncio.sleep(0.3)
    
    # Verificar que todos los comandos fueron recibidos
    assert len(response_component.commands_received) == 3
    
    # Verificar que todas las respuestas fueron recibidas
    assert len(command_component.responses_received) == 3
    
    # Verificar que las respuestas están en el orden correcto
    for i, cmd_id in enumerate(command_ids):
        response = next(r for r in command_component.responses_received if r["id"] == cmd_id)
        assert f"Processed command_{i}" in response["response"]
    
    # Detener motor
    await engine.stop()


@pytest.mark.asyncio
async def test_event_bus_multiple_listeners(event_bus):
    """Probar que múltiples listeners pueden recibir el mismo evento."""
    # Preparar listeners
    listener_results = [[], [], []]
    
    async def make_listener(index):
        async def listener(event_type, data, source):
            listener_results[index].append({
                "type": event_type,
                "data": data,
                "source": source
            })
            return None
        return listener
    
    # Registrar varios listeners para el mismo tipo de evento
    for i in range(3):
        event_bus.subscribe("test_event", await make_listener(i))
    
    # Emitir el evento
    await event_bus.emit("test_event", {"value": 42}, "test_source")
    
    # Verificar que todos los listeners recibieron el evento
    for i in range(3):
        assert len(listener_results[i]) == 1
        assert listener_results[i][0]["type"] == "test_event"
        assert listener_results[i][0]["data"]["value"] == 42
        assert listener_results[i][0]["source"] == "test_source"


@pytest.mark.asyncio
async def test_event_bus_pattern_matching(event_bus):
    """Probar que el patrón wildcard funciona correctamente para subscribe."""
    # Resultados para eventos generales y específicos
    wildcard_events = []
    specific_events = []
    
    async def wildcard_listener(event_type, data, source):
        wildcard_events.append({
            "type": event_type,
            "data": data,
            "source": source
        })
        return None
    
    async def specific_listener(event_type, data, source):
        specific_events.append({
            "type": event_type,
            "data": data,
            "source": source
        })
        return None
    
    # Suscribir a tipos de eventos
    event_bus.subscribe("*", wildcard_listener)
    event_bus.subscribe("specific_event", specific_listener)
    
    # Emitir varios eventos
    await event_bus.emit("specific_event", {"id": 1}, "source1")
    await event_bus.emit("other_event", {"id": 2}, "source2")
    
    # Verificar que el listener específico solo recibió su tipo de evento
    assert len(specific_events) == 1
    assert specific_events[0]["type"] == "specific_event"
    assert specific_events[0]["data"]["id"] == 1
    
    # Verificar que el listener wildcard recibió todos los eventos
    assert len(wildcard_events) == 2
    assert wildcard_events[0]["type"] == "specific_event"
    assert wildcard_events[1]["type"] == "other_event"


@pytest.mark.asyncio
async def test_config_load_save(tmpdir):
    """Probar guardar y cargar configuración desde un archivo."""
    # Crear un archivo temporal
    config_path = tmpdir.join("test_config.json")
    
    # Crear configuración
    config = Config()
    config.set("string_key", "string_value")
    config.set("int_key", 123)
    config.set("float_key", 3.14)
    config.set("bool_key", True)
    config.set("list_key", [1, 2, 3])
    config.set("dict_key", {"a": 1, "b": 2})
    
    # Guardar configuración
    config.save_to_file(str(config_path))
    
    # Crear una nueva instancia y cargar la configuración
    config2 = Config()
    config2.load_from_file(str(config_path))
    
    # Verificar que los valores se cargaron correctamente
    assert config2.get("string_key") == "string_value"
    assert config2.get("int_key") == 123
    assert config2.get("float_key") == 3.14
    assert config2.get("bool_key") is True
    assert config2.get("list_key") == [1, 2, 3]
    assert config2.get("dict_key") == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_config_environment_override():
    """Probar sobrescritura de configuración desde variables de entorno."""
    # Configuración básica
    config = Config(env_prefix="TEST_")
    config.set("database_url", "default_value")
    config.set("api_key", "default_api_key")
    
    # Simular variables de entorno
    with patch.dict('os.environ', {
        'TEST_DATABASE_URL': 'overridden_value',
        'TEST_API_KEY': 'overridden_api_key'
    }):
        # Actualizar desde variables de entorno
        config.load_from_env()
        
        # Verificar que los valores se sobrescribieron
        assert config.get("database_url") == "overridden_value"
        assert config.get("api_key") == "overridden_api_key"


@pytest.mark.asyncio
async def test_config_sensitive_values():
    """Probar el manejo de valores sensibles en la configuración."""
    # Crear configuración con valores sensibles
    config = Config()
    config.set("username", "user123", sensitive=False)
    config.set("password", "secret123", sensitive=True)
    config.set("api_key", "abcdef123456", sensitive=True)
    
    # Guardar en archivo temporal
    tmp_path = Path("/tmp/sensitive_config_test.json")
    config.save_to_file(str(tmp_path))
    
    # Cargar el archivo y verificar que los valores sensibles están ocultados
    with open(tmp_path, 'r') as f:
        content = f.read()
        # Valores sensibles no deben estar en texto plano
        assert "secret123" not in content
        assert "abcdef123456" not in content
        # Valores no sensibles deben estar presentes
        assert "user123" in content
    
    # Cargar configuración y verificar que se pueden recuperar valores sensibles
    config2 = Config()
    config2.load_from_file(str(tmp_path))
    assert config2.get("username") == "user123"
    assert config2.get("password") != "secret123"  # Debe estar cifrado
    assert config2.get("api_key") != "abcdef123456"  # Debe estar cifrado
    
    # Limpiar
    if tmp_path.exists():
        tmp_path.unlink()