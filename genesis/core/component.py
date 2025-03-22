"""
Re-export of Component class from the base module
for backwards compatibility with existing code.
"""

from genesis.core.base import Component

# Re-export for backwards compatibility
__all__ = ['Component']