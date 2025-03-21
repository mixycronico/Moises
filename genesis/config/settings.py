"""
Settings module handling system-wide configuration.

This module provides access to system configuration from environment variables,
configuration files, and default values.
"""

import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path

# Default settings that can be overridden
DEFAULT_SETTINGS = {
    "log_level": "INFO",
    "database": {
        "host": os.environ.get("PGHOST", "localhost"),
        "port": os.environ.get("PGPORT", "5432"),
        "user": os.environ.get("PGUSER", "postgres"),
        "password": os.environ.get("PGPASSWORD", ""),
        "database": os.environ.get("PGDATABASE", "genesis"),
        "connection_string": os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/genesis")
    },
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "debug": True
    },
    "exchanges": {
        "default_exchange": "binance",
        "connection_timeout": 30,
        "rate_limit": {
            "max_requests": 10,
            "timeframe": 1  # seconds
        }
    },
    "trading": {
        "trading_enabled": False,  # Safety default
        "default_symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        "interval": "1h",
        "max_open_trades": 5,
        "stake_amount": 100.0,
        "dry_run": True  # Don't execute real trades by default
    },
    "risk": {
        "max_risk_per_trade": 0.02,  # 2% of portfolio
        "max_risk_total": 0.2,  # 20% of portfolio
        "stop_loss_pct": 0.05,  # 5% stop loss
        "risk_free_rate": 0.02  # 2% risk-free rate for Sharpe calculation
    },
    "strategies": {
        "default_strategy": "trend_following"
    }
}


class Settings:
    """
    Settings class to manage application configuration.
    
    Loads settings from environment variables, configuration files,
    and falls back to default values.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize Settings with optional configuration file path.
        
        Args:
            config_path: Path to a JSON configuration file
        """
        self._settings = DEFAULT_SETTINGS.copy()
        self._load_from_file(config_path)
        self._load_from_env()
        
    def _load_from_file(self, config_path: Optional[Path]) -> None:
        """Load settings from a configuration file if provided."""
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    file_settings = json.load(f)
                    self._settings = self._deep_update(self._settings, file_settings)
            except Exception as e:
                print(f"Error loading configuration file: {e}")
    
    def _load_from_env(self) -> None:
        """Override settings from environment variables."""
        # Handle known environment mappings
        if os.environ.get("LOG_LEVEL"):
            self._settings["log_level"] = os.environ.get("LOG_LEVEL")
            
        # Database settings from environment
        if os.environ.get("DATABASE_URL"):
            self._settings["database"]["connection_string"] = os.environ.get("DATABASE_URL")
            
        # Trading settings
        if os.environ.get("TRADING_ENABLED"):
            self._settings["trading"]["trading_enabled"] = os.environ.get("TRADING_ENABLED").lower() == "true"
            
        if os.environ.get("DRY_RUN"):
            self._settings["trading"]["dry_run"] = os.environ.get("DRY_RUN").lower() == "true"
            
        if os.environ.get("STAKE_AMOUNT"):
            self._settings["trading"]["stake_amount"] = float(os.environ.get("STAKE_AMOUNT"))
    
    @staticmethod
    def _deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively update a dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                d[k] = Settings._deep_update(d[k], v)
            else:
                d[k] = v
        return d
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value by key with optional default."""
        keys = key.split(".")
        value = self._settings
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def get_all(self) -> Dict[str, Any]:
        """Get a copy of all settings."""
        return self._settings.copy()


# Create a global instance for easy importing
settings = Settings()
