"""
Exchanges module for interacting with cryptocurrency exchanges.

This module provides abstractions for trading on various cryptocurrency exchanges
through CCXT.
"""

from genesis.exchanges.api_client import APIClient
from genesis.exchanges.base import Exchange
from genesis.exchanges.ccxt_wrapper import CCXTExchange
from genesis.exchanges.exchange_selector import ExchangeSelector
from genesis.exchanges.manager import ExchangeManager

__all__ = [
    "APIClient",
    "Exchange",
    "CCXTExchange",
    "ExchangeSelector",
    "ExchangeManager"
]
