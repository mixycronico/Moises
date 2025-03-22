"""
EventBus minimalista para pruebas.

Esta implementación ultra simplificada del EventBus está diseñada específicamente
para pruebas, eliminando la complejidad de la versión completa y enfocándose
solo en la funcionalidad básica.
"""

import asyncio
import logging
from typing import Dict, List, Callable, Any, Awaitable

# Tipo para manejadores de eventos
EventHandler = Callable[[str, Dict[str, Any], str], Awaitable[Any]]

class EventBusMinimal:
    """
    EventBus minimalista para pruebas.
    
    Características:
    - Procesa eventos sincrónicamente (para facilitar pruebas)
    - Maneja errores en los handlers sin propagar excepciones
    - No tiene colas ni procesamiento asíncrono
    - Sin pattern matching ni funciones avanzadas
    """
    
    def __init__(self):
        """Inicializar el bus de eventos minimalista."""
        self.subscribers: Dict[str, List[EventHandler]] = {}
        self.running = True
        self.logger = logging.getLogger("genesis.event_bus_minimal")
    
    async def start(self):
        """Iniciar el bus de eventos."""
        self.running = True
    
    async def stop(self):
        """Detener el bus de eventos."""
        self.running = False
    
    def subscribe(self, event_type: str, handler: EventHandler):
        """
        Suscribir un manejador a un tipo de evento.
        
        Args:
            event_type: Tipo de evento
            handler: Función manejadora
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        if handler not in self.subscribers[event_type]:
            self.subscribers[event_type].append(handler)
    
    def unsubscribe(self, event_type: str, handler: EventHandler):
        """
        Desuscribir un manejador de un tipo de evento.
        
        Args:
            event_type: Tipo de evento
            handler: Función manejadora
        """
        if event_type in self.subscribers and handler in self.subscribers[event_type]:
            self.subscribers[event_type].remove(handler)
    
    async def emit(self, event_type: str, data: Dict[str, Any], source: str):
        """
        Emitir un evento.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
        """
        if not self.running:
            self.logger.warning(f"Evento {event_type} ignorado, bus no está en ejecución")
            return
        
        # Recopilar handlers para este evento
        handlers = []
        if event_type in self.subscribers:
            handlers = self.subscribers[event_type]
        
        # Ejecutar handlers
        for handler in handlers:
            try:
                await handler(event_type, data, source)
            except Exception as e:
                self.logger.error(f"Error en manejador de eventos: {e}")
                # No propagar la excepción para que otros handlers se ejecuten