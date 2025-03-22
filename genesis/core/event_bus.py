"""
Event bus implementation for inter-module communication.

The event bus allows decoupled communication between system components using
an asynchronous publish-subscribe pattern.
"""

import asyncio
from typing import Dict, Set, Callable, Any, Awaitable, Optional

# Type definition for event handlers
EventHandler = Callable[[str, Dict[str, Any], str], Awaitable[None]]


class EventBus:
    """
    Asynchronous event bus for inter-module communication.
    
    The event bus enables publish-subscribe messaging between components
    in the system without direct dependencies, facilitating a modular design.
    """
    
    def __init__(self):
        """Initialize the event bus."""
        self.subscribers: Dict[str, Set[EventHandler]] = {}
        self.running = False
        self.queue: Optional[asyncio.Queue] = None
        self.process_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the event bus."""
        if self.running:
            return
        
        self.running = True
        self.queue = asyncio.Queue()
        self.process_task = asyncio.create_task(self._process_events())
    
    async def stop(self) -> None:
        """Stop the event bus."""
        if not self.running:
            return
        
        self.running = False
        
        if self.process_task:
            self.process_task.cancel()
            try:
                await self.process_task
            except asyncio.CancelledError:
                pass
            self.process_task = None
    
    async def _process_events(self) -> None:
        """Process events from the queue."""
        if not self.queue:
            raise RuntimeError("Event queue not initialized")
        
        while self.running:
            try:
                event_type, data, source = await self.queue.get()
                
                if event_type in self.subscribers:
                    for handler in self.subscribers[event_type]:
                        try:
                            await handler(event_type, data, source)
                        except Exception as e:
                            print(f"Error in event handler: {e}")
                
                # Process 'all' event subscribers
                if '*' in self.subscribers:
                    for handler in self.subscribers['*']:
                        try:
                            await handler(event_type, data, source)
                        except Exception as e:
                            print(f"Error in wildcard event handler: {e}")
                
                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error processing event: {e}")
    
    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """
        Subscribe to an event type.
        
        Args:
            event_type: The event type to subscribe to, or '*' for all events
            handler: Async callback function to handle the event
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = set()
        
        self.subscribers[event_type].add(handler)
    
    # Alias for backwards compatibility with tests
    register_listener = subscribe
    
    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: The event type to unsubscribe from
            handler: The handler to remove
        """
        if event_type in self.subscribers and handler in self.subscribers[event_type]:
            self.subscribers[event_type].remove(handler)
            
            # Clean up empty subscriber sets
            if not self.subscribers[event_type]:
                del self.subscribers[event_type]
    
    # Alias for backwards compatibility with tests
    unregister_listener = unsubscribe
    
    async def emit(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Emit an event to subscribers.
        
        Args:
            event_type: Type of the event
            data: Event data
            source: Source component of the event
        """
        if not self.running or not self.queue:
            raise RuntimeError("Event bus not started")
        
        await self.queue.put((event_type, data, source))
