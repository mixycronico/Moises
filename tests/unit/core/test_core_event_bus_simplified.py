"""
Tests simplificados para el EventBus del sistema Genesis.

Este módulo contiene versiones altamente simplificadas de las pruebas
para el EventBus, centradas en la funcionalidad básica sin pruebas de concurrencia.
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock

from genesis.core.event_bus import EventBus


@pytest.fixture
def event_bus():
    """Proporcionar un bus de eventos para pruebas."""
    bus = EventBus(test_mode=True)  # Usar modo test para evitar crear tareas en background
    return bus


@pytest.mark.asyncio
async def test_register_and_emit_basic(event_bus):
    """Prueba básica de registro y emisión de eventos."""
    # Variable para rastrear si el evento fue recibido
    event_received = False
    event_data = None
    
    # Función de callback simple
    async def test_handler(event_type, data, source):
        nonlocal event_received, event_data
        event_received = True
        event_data = data
    
    # Registrar handler
    event_bus.register_listener("test_event", test_handler)
    
    # Emitir evento
    await event_bus.emit("test_event", {"value": "test"}, "test_source")
    
    # Verificar recepción
    assert event_received
    assert event_data["value"] == "test"


@pytest.mark.asyncio
async def test_wildcard_subscription(event_bus):
    """Prueba la suscripción comodín a todos los eventos."""
    # Contador para rastrear eventos
    event_count = 0
    
    # Handler para todos los eventos
    async def wildcard_handler(event_type, data, source):
        nonlocal event_count
        event_count += 1
    
    # Registrar con comodín
    event_bus.register_listener("*", wildcard_handler)
    
    # Emitir varios eventos de diferentes tipos
    await event_bus.emit("event1", {}, "test")
    await event_bus.emit("event2", {}, "test")
    await event_bus.emit("event3", {}, "test")
    
    # Verificar que se recibieron todos
    assert event_count == 3


@pytest.mark.asyncio
async def test_unregister_listener(event_bus):
    """Prueba dar de baja a un listener."""
    # Contador para rastrear eventos
    event_count = 0
    
    # Handler simple
    async def test_handler(event_type, data, source):
        nonlocal event_count
        event_count += 1
    
    # Registrar handler
    event_bus.register_listener("test_event", test_handler)
    
    # Emitir evento - debe ser capturado
    await event_bus.emit("test_event", {}, "test")
    assert event_count == 1
    
    # Dar de baja al handler
    event_bus.unregister_listener("test_event", test_handler)
    
    # Emitir de nuevo - no debe incrementar el contador
    await event_bus.emit("test_event", {}, "test")
    assert event_count == 1  # Sigue siendo 1


@pytest.mark.asyncio
async def test_one_time_listener(event_bus):
    """Prueba listener de un solo uso."""
    # Contador para rastrear eventos
    event_count = 0
    
    # Handler simple
    async def test_handler(event_type, data, source):
        nonlocal event_count
        event_count += 1
    
    # Registrar como one-time listener usando el método nativo
    event_bus.subscribe_once("test_event", test_handler)
    
    # Emitir evento dos veces
    await event_bus.emit("test_event", {}, "test")
    await event_bus.emit("test_event", {}, "test")
    
    # Verificar que solo se contó una vez
    assert event_count == 1


@pytest.mark.asyncio
async def test_pattern_matching_simplified(event_bus):
    """Versión simplificada de prueba de coincidencia de patrones."""
    # Contadores para diferentes patrones
    specific_count = 0
    wildcard_count = 0
    
    # Handlers simples
    async def specific_handler(event_type, data, source):
        nonlocal specific_count
        specific_count += 1
    
    async def wildcard_handler(event_type, data, source):
        nonlocal wildcard_count
        wildcard_count += 1
    
    # Registrar handlers
    event_bus.register_listener("test.specific", specific_handler)
    event_bus.register_listener("test.*", wildcard_handler)
    
    # Emitir eventos
    await event_bus.emit("test.specific", {}, "test")
    await event_bus.emit("test.other", {}, "test")
    
    # Verificar contadores
    assert specific_count == 1
    assert wildcard_count == 2  # Debe capturar ambos eventos