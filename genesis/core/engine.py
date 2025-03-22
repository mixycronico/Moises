"""
Main engine of the Genesis trading system.

The engine is responsible for initializing, coordinating, and managing all system
components. It handles the startup and shutdown sequences and provides the main
entry point for system operation.
"""

import asyncio
import signal
import logging
import time
from typing import Dict, List, Optional, Any, Type

from genesis.config.settings import settings
from genesis.core.base import Component
from genesis.core.event_bus import EventBus
from genesis.utils.logger import setup_logging


class Engine:
    """
    Main engine of the Genesis trading system.
    
    The engine initializes and coordinates all components, manages the system
    lifecycle, and provides the main entry point for system operation.
    """
    
    def __init__(self, event_bus_or_name=None):
        """
        Initialize the engine.
        
        Args:
            event_bus_or_name: Either an EventBus instance or a name string for the engine
                               For backwards compatibility with testing code
        """
        # Handle both EventBus instance or string name for backwards compatibility
        if isinstance(event_bus_or_name, EventBus):
            self.name = "engine"
            self.event_bus = event_bus_or_name
        else:
            self.name = event_bus_or_name if isinstance(event_bus_or_name, str) else "engine"
            self.event_bus = EventBus()
            
        self.logger = setup_logging(self.name, level=settings.get('log_level', 'INFO'))
        self.components: Dict[str, Component] = {}
        self.running = False
        self._shutdown_event = asyncio.Event()
        self.started_at = None
        
        # Atributos adicionales para compatibilidad con pruebas
        self.use_priorities = settings.get('use_priorities', False)
        self.operation_timeout = settings.get('operation_timeout', 30.0)
        
    @property
    def is_running(self):
        """Alias para running, para compatibilidad con pruebas."""
        return self.running
    
    def register_component(self, component: Component) -> None:
        """
        Register a component with the engine.
        
        Args:
            component: Component instance to register
        """
        if component.name in self.components:
            self.logger.warning(f"Component with name '{component.name}' already registered, replacing")
        
        self.components[component.name] = component
        component.attach_event_bus(self.event_bus)
        
        # Registrar el componente para que reciba todos los eventos
        # Esto permite que handle_event se llame para cada evento
        self.event_bus.subscribe("*", component.handle_event)
        
        self.logger.debug(f"Registered component: {component.name}")
    
    def get_component(self, name: str) -> Optional[Component]:
        """
        Get a component by name.
        
        Args:
            name: Name of the component to retrieve
            
        Returns:
            The component instance or None if not found
        """
        return self.components.get(name)
    
    async def start(self) -> None:
        """Start the engine and all registered components."""
        if self.running:
            self.logger.warning("Engine already running")
            return
        
        self.logger.info("Starting Genesis engine")
        
        # Setup signal handlers with shutdown timeout
        shutdown_timeout = float(settings.get('shutdown_timeout', 10.0))
        self._setup_signal_handlers(shutdown_timeout=shutdown_timeout)
        
        # Start event bus first
        await self.event_bus.start()
        
        # Start all components
        for name, component in self.components.items():
            try:
                self.logger.info(f"Starting component: {name}")
                await component.start()
            except Exception as e:
                self.logger.error(f"Error starting component {name}: {e}")
                # Continue with other components
        
        self.running = True
        self.started_at = time.time()
        self.logger.info("Genesis engine started")
        
        # Emit system started event
        await self.event_bus.emit(
            "system.started",
            {"components": list(self.components.keys())},
            "engine"
        )
    
    async def stop(self, timeout: Optional[float] = None) -> None:
        """
        Stop the engine and all registered components.
        
        Args:
            timeout: Maximum time (in seconds) to wait for components to stop
        """
        if not self.running:
            self.logger.warning("Engine not running")
            return
        
        self.logger.info("Stopping Genesis engine")
        
        # Stop components in reverse order
        component_names = list(self.components.keys())
        for name in reversed(component_names):
            try:
                self.logger.info(f"Stopping component: {name}")
                if timeout:
                    try:
                        await asyncio.wait_for(self.components[name].stop(), timeout)
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Stopping component {name} timed out after {timeout} seconds")
                else:
                    await self.components[name].stop()
            except Exception as e:
                self.logger.error(f"Error stopping component {name}: {e}")
        
        # Stop event bus last
        await self.event_bus.stop()
        
        self.running = False
        self.logger.info("Genesis engine stopped")
    
    def _setup_signal_handlers(self, shutdown_timeout: float = 10.0) -> None:
        """
        Set up signal handlers for graceful shutdown.
        
        Args:
            shutdown_timeout: Maximum time to wait for components to shut down
        """
        loop = asyncio.get_event_loop()
        
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda sig=sig, timeout=shutdown_timeout: asyncio.create_task(self._shutdown(sig, timeout))
            )
    
    async def _shutdown(self, sig: signal.Signals, timeout: float = 10.0) -> None:
        """
        Handle shutdown signal.
        
        Args:
            sig: Signal that triggered the shutdown
            timeout: Maximum time to wait for components to shut down
        """
        self.logger.info(f"Received exit signal {sig.name}")
        await self.stop(timeout=timeout)
        self._shutdown_event.set()
    
    async def run_forever(self) -> None:
        """Run the engine until a shutdown signal is received."""
        await self.start()
        await self._shutdown_event.wait()
