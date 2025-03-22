"""
Main engine of the Genesis trading system.

The engine is responsible for initializing, coordinating, and managing all system
components. It handles the startup and shutdown sequences and provides the main
entry point for system operation.
"""

import asyncio
import signal
import logging
import sys
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
    
    def __init__(self, event_bus_or_name=None, test_mode=False):
        """
        Initialize the engine.
        
        Args:
            event_bus_or_name: Either an EventBus instance or a name string for the engine
                               For backwards compatibility with testing code
            test_mode: If True, operate in test mode with direct event delivery
        """
        # Handle both EventBus instance or string name for backwards compatibility
        if isinstance(event_bus_or_name, EventBus):
            self.name = "engine"
            self.event_bus = event_bus_or_name
        else:
            self.name = event_bus_or_name if isinstance(event_bus_or_name, str) else "engine"
            self.event_bus = EventBus(test_mode=test_mode)
            
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
    
    def register_component(self, component: Component, priority: int = 50) -> None:
        """
        Register a component with the engine.
        
        Args:
            component: Component instance to register
            priority: Priority level for event handling and startup
                    (higher values = higher priority)
            
        Raises:
            ValueError: If a component with the same name is already registered
        """
        if component.name in self.components:
            raise ValueError(f"Component with name '{component.name}' already registered")
        
        self.components[component.name] = component
        # Almacenar la prioridad para usarla en startup
        self.operation_priorities = getattr(self, 'operation_priorities', {})
        self.operation_priorities[component.name] = priority
        
        component.attach_event_bus(self.event_bus)
        
        # Registrar el componente para que reciba todos los eventos
        # Esto permite que handle_event se llame para cada evento
        self.event_bus.subscribe("*", component.handle_event, priority=priority)
        
        self.logger.debug(f"Registered component: {component.name}")
        
        # Si el motor ya está ejecutándose, iniciar también el componente
        if self.running and hasattr(component, 'start'):
            asyncio.create_task(self._start_component(component))
    
    def get_component(self, name: str) -> Optional[Component]:
        """
        Get a component by name.
        
        Args:
            name: Name of the component to retrieve
            
        Returns:
            The component instance or None if not found
        """
        return self.components.get(name)
        
    def get_all_components(self) -> List[Component]:
        """
        Get all registered components.
        
        Returns:
            List of all component instances
        """
        return list(self.components.values())
        
    def deregister_component(self, name: str) -> None:
        """
        Unregister a component from the engine.
        
        Args:
            name: Name of the component to unregister
            
        Raises:
            ValueError: If component with given name is not registered
        """
        if name not in self.components:
            raise ValueError(f"Component with name '{name}' not registered")
            
        component = self.components[name]
        
        # Si el motor está ejecutándose, detener primero el componente
        if self.running and hasattr(component, 'stop'):
            # Para pruebas, ejecutar y esperar síncronamente
            # Esto asegura que el componente esté detenido cuando termina el test
            loop = asyncio.get_event_loop()
            if hasattr(sys, '_called_from_test') or self.event_bus.test_mode:
                try:
                    # Para pruebas, se ejecuta síncronamente para evitar race conditions
                    loop.run_until_complete(component.stop())
                except RuntimeError:
                    # Si el loop ya está ejecutándose, crear una tarea
                    asyncio.create_task(component.stop())
            else:
                # Para entorno de producción, ejecutar de forma asíncrona
                asyncio.create_task(component.stop())
            
        # Eliminar el componente de la lista
        del self.components[name]
        
        # Eliminar suscripción a eventos globales
        self.event_bus.unsubscribe("*", component.handle_event)
        
        self.logger.debug(f"Unregistered component: {name}")
    
    async def start(self) -> None:
        """Start the engine and all registered components."""
        if self.running:
            self.logger.warning("Engine already running")
            return
        
        self.logger.info("Starting Genesis engine")
        
        # Setup signal handlers with shutdown timeout
        shutdown_timeout = float(settings.get('shutdown_timeout', 10.0))
        self._setup_signal_handlers(shutdown_timeout=shutdown_timeout)
        
        # Start event bus first with timeout para evitar bloqueos
        try:
            # Usar un timeout más corto en modo prueba
            timeout = 1.0 if self.event_bus.test_mode or hasattr(sys, '_called_from_test') else 5.0
            await asyncio.wait_for(self.event_bus.start(), timeout=timeout)
        except asyncio.TimeoutError:
            self.logger.warning("Timeout al iniciar event_bus, continuando de todos modos")
        except Exception as e:
            self.logger.error(f"Error iniciando event_bus: {e}, continuando de todos modos")
        
        # Ordenar componentes por prioridad (mayor prioridad primero)
        priorities = getattr(self, 'operation_priorities', {})
        ordered_components = sorted(
            self.components.items(),
            key=lambda x: priorities.get(x[0], 50),
            reverse=True  # Mayor prioridad primero
        )
        
        # Start all components in priority order
        start_tasks = []
        for name, component in ordered_components:
            # Crear tarea para iniciar el componente (permitirá ejecutar en paralelo con timeout)
            start_task = asyncio.create_task(self._start_component_with_timeout(name, component))
            start_tasks.append(start_task)
        
        # Esperar a que todos los componentes se inicien con un tiempo máximo total
        total_start_timeout = 3.0 if self.event_bus.test_mode or hasattr(sys, '_called_from_test') else 30.0
        try:
            await asyncio.wait_for(asyncio.gather(*start_tasks, return_exceptions=True), timeout=total_start_timeout)
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout global al iniciar componentes ({total_start_timeout}s)")
        
        # Marcar como iniciado aunque algunos componentes hayan fallado
        self.running = True
        self.started_at = time.time()
        self.logger.info("Genesis engine started")
        
        # Emit system started event con timeout para evitar bloqueos
        try:
            event_timeout = 1.0 if self.event_bus.test_mode or hasattr(sys, '_called_from_test') else 5.0
            await asyncio.wait_for(
                self.event_bus.emit(
                    "system.started",
                    {"components": list(self.components.keys())},
                    "engine"
                ),
                timeout=event_timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning("Timeout al emitir evento system.started")
        except Exception as e:
            self.logger.error(f"Error al emitir evento system.started: {e}")
    
    async def _start_component_with_timeout(self, name, component):
        """Helper interno para iniciar un componente con timeout."""
        try:
            priorities = getattr(self, 'operation_priorities', {})
            self.logger.info(f"Starting component: {name} (priority: {priorities.get(name, 50)})")
            
            # Usar timeout en modo prueba para evitar bloqueos
            timeout = 1.0 if self.event_bus.test_mode or hasattr(sys, '_called_from_test') else 5.0
            try:
                await asyncio.wait_for(component.start(), timeout=timeout)
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout al iniciar componente {name} (max: {timeout}s)")
            except Exception as e:
                self.logger.error(f"Error iniciando componente {name}: {e}")
        except Exception as e:
            self.logger.error(f"Error inesperado al iniciar componente {name}: {e}")
    
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
        
        # Ordenar componentes por prioridad (menor prioridad primero para detener)
        priorities = getattr(self, 'operation_priorities', {})
        ordered_components = sorted(
            self.components.items(),
            key=lambda x: priorities.get(x[0], 50)  # Menor prioridad primero
        )
        
        # Determinar el timeout basado en el modo de ejecución
        default_timeout = 1.0 if self.event_bus.test_mode or hasattr(sys, '_called_from_test') else 5.0
        actual_timeout = timeout if timeout is not None else default_timeout
        
        # Crear tareas para detener componentes en paralelo con timeout individual
        stop_tasks = []
        for name, component in ordered_components:
            stop_task = asyncio.create_task(self._stop_component_with_timeout(name, component, actual_timeout))
            stop_tasks.append(stop_task)
        
        # Esperar a que todos los componentes se detengan con un timeout global
        global_timeout = 3.0 if self.event_bus.test_mode or hasattr(sys, '_called_from_test') else 30.0
        try:
            await asyncio.wait_for(asyncio.gather(*stop_tasks, return_exceptions=True), timeout=global_timeout)
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout global al detener componentes ({global_timeout}s)")
        
        # Detener el event bus con timeout
        try:
            bus_timeout = 1.0 if self.event_bus.test_mode or hasattr(sys, '_called_from_test') else 5.0
            await asyncio.wait_for(self.event_bus.stop(), timeout=bus_timeout)
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout al detener event_bus ({bus_timeout}s)")
        except Exception as e:
            self.logger.error(f"Error al detener event_bus: {e}")
        
        self.running = False
        self.logger.info("Genesis engine stopped")
    
    async def _stop_component_with_timeout(self, name, component, timeout):
        """Helper interno para detener un componente con timeout."""
        try:
            priorities = getattr(self, 'operation_priorities', {})
            self.logger.info(f"Stopping component: {name} (priority: {priorities.get(name, 50)})")
            
            try:
                await asyncio.wait_for(component.stop(), timeout)
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout al detener componente {name} (max: {timeout}s)")
            except Exception as e:
                self.logger.error(f"Error deteniendo componente {name}: {e}")
        except Exception as e:
            self.logger.error(f"Error inesperado al detener componente {name}: {e}")
    
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
        
    async def _start_component(self, component: Component) -> None:
        """
        Helper interno para iniciar un componente durante la ejecución.
        
        Args:
            component: Componente a iniciar
        """
        try:
            priorities = getattr(self, 'operation_priorities', {})
            priority = priorities.get(component.name, 50)
            self.logger.info(f"Starting component at runtime: {component.name} (priority: {priority})")
            
            # Usar timeout en modo prueba para evitar bloqueos
            if getattr(self, 'event_bus', None) and (self.event_bus.test_mode or hasattr(sys, '_called_from_test')):
                try:
                    await asyncio.wait_for(component.start(), timeout=1.0)
                except asyncio.TimeoutError:
                    self.logger.warning(f"Timeout al iniciar componente {component.name}")
            else:
                await component.start()
        except Exception as e:
            self.logger.error(f"Error starting component {component.name}: {e}")
