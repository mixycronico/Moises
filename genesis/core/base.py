"""
Base module with abstract base classes for the Genesis system components.
"""

import abc
import asyncio
from typing import Dict, Any, Optional, List

from genesis.core.event_bus import EventBus


class Component(abc.ABC):
    """Abstract base class for all system components."""
    
    def __init__(self, name: str):
        """Initialize component with a name."""
        self.name = name
        self.event_bus: Optional[EventBus] = None
        self.running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None
    
    def attach_event_bus(self, event_bus: EventBus) -> None:
        """Attach a reference to the event bus."""
        self.event_bus = event_bus
    
    @abc.abstractmethod
    async def start(self) -> None:
        """Start the component."""
        self.running = True
        self.loop = asyncio.get_event_loop()
    
    @abc.abstractmethod
    async def stop(self) -> None:
        """Stop the component."""
        self.running = False
    
    async def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event to the event bus."""
        if not self.event_bus:
            raise RuntimeError(f"Component {self.name} not attached to event bus")
        
        await self.event_bus.emit(event_type, data, self.name)
    
    @abc.abstractmethod
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Handle an event from the event bus."""
        pass


class Module(Component):
    """Base class for modules that need to process events in the background."""
    
    def __init__(self, name: str):
        """Initialize the module."""
        super().__init__(name)
        self.tasks: List[asyncio.Task] = []
    
    async def start(self) -> None:
        """Start the module and its background tasks."""
        await super().start()
        process_task = asyncio.create_task(self._process_loop())
        self.tasks.append(process_task)
    
    async def stop(self) -> None:
        """Stop the module and its background tasks."""
        await super().stop()
        for task in self.tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.tasks.clear()
    
    @abc.abstractmethod
    async def _process_loop(self) -> None:
        """Background processing loop implementation."""
        pass
