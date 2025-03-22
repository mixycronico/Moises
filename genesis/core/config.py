"""
Re-export of the Settings class as Config for backwards compatibility.

This module ensures that existing code that imports Config from genesis.core.config
can continue to function correctly while the actual implementation has moved to
genesis.config.settings.
"""

from genesis.config.settings import Settings

# Re-export Settings as Config for backwards compatibility
class Config(Settings):
    """
    Backwards compatibility wrapper for Settings class.
    
    Extends the Settings class to ensure compatibility with older code
    that expects a Config class located in genesis.core.config.
    """
    pass

# For direct imports when a function expects Config
__all__ = ['Config']