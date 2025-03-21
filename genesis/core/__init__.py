"""
Core module of the Genesis trading system.

This module contains the central components that coordinate the system's operation,
including the main engine and the event bus for inter-module communication.
"""

from genesis.core.base import Component
from genesis.core.engine import Engine
from genesis.core.event_bus import EventBus

__all__ = [
    "Component",
    "Engine",
    "EventBus"
]
