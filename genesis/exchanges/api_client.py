"""
Cliente API para interactuar con diversos exchanges.

Este módulo proporciona una interfaz unificada para interactuar con
múltiples exchanges a través de la biblioteca CCXT.
"""

import ccxt
import time
import logging
from typing import Dict, Any, Optional

class APIClient:
    """Cliente universal para múltiples exchanges usando CCXT."""
    
    def __init__(self, exchange_name, config):
        """
        Inicializar el cliente de API.
        
        Args:
            exchange_name: Nombre del exchange (debe ser compatible con CCXT)
            config: Configuración con credenciales y parámetros
        """
        self.name = exchange_name
        self.api_key = config.get("api_key")
        self.api_secret = config.get("api_secret")
        self.fees = config.get("fees", 0.001)
        self.priority = config.get("priority", 1)

        try:
            exchange_class = getattr(ccxt, exchange_name)
            self.client = exchange_class({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'timeout': 10000
            })
            logging.info(f"[{self.name}] Cliente inicializado con éxito.")
        except Exception as e:
            logging.error(f"[{self.name}] Error al iniciar cliente CCXT: {e}")
            self.client = None

    def get_order_book(self, trading_pair):
        """
        Obtener el libro de órdenes para un par de trading.
        
        Args:
            trading_pair: Par de trading (ej: 'BTC/USDT')
            
        Returns:
            Diccionario con mejores precios de compra y venta, o None si hay error
        """
        try:
            order_book = self.client.fetch_order_book(trading_pair)
            bid = order_book['bids'][0][0] if order_book['bids'] else 0
            ask = order_book['asks'][0][0] if order_book['asks'] else 0
            return {'bid': bid, 'ask': ask}
        except Exception as e:
            logging.warning(f"[{self.name}] Error al obtener order book: {e}")
            return None

    def place_order(self, trading_pair, side, amount, price=None):
        """
        Colocar una orden en el exchange.
        
        Args:
            trading_pair: Par de trading (ej: 'BTC/USDT')
            side: Dirección de la orden ('buy' o 'sell')
            amount: Cantidad a comprar/vender
            price: Precio límite (opcional, None para orden de mercado)
            
        Returns:
            Diccionario con estado de la orden y detalles
        """
        try:
            if price:
                order = self.client.create_limit_order(trading_pair, side, amount, price)
            else:
                order = self.client.create_market_order(trading_pair, side, amount)
            return {
                "status": "ok",
                "order_id": order.get("id"),
                "exchange": self.name
            }
        except Exception as e:
            logging.error(f"[{self.name}] Error al colocar orden: {e}")
            return {"status": "error", "message": str(e)}

    def get_balance(self, currency="USDT"):
        """
        Obtener el balance para una moneda específica.
        
        Args:
            currency: Símbolo de la moneda (ej: 'USDT', 'BTC')
            
        Returns:
            Diccionario con el balance
        """
        try:
            balances = self.client.fetch_balance()
            total = balances['total'].get(currency, 0)
            return {currency: total}
        except Exception as e:
            logging.warning(f"[{self.name}] Error al obtener balance: {e}")
            return {currency: 0}

    def get_latency(self):
        """
        Medir la latencia de conexión al exchange.
        
        Returns:
            Latencia en milisegundos
        """
        try:
            start = time.time()
            self.client.fetch_time()
            end = time.time()
            return round((end - start) * 1000, 2)  # ms
        except Exception as e:
            logging.warning(f"[{self.name}] Error midiendo latencia: {e}")
            return 9999

    def get_trading_fees(self, trading_pair):
        """
        Obtener comisiones de trading para un par.
        
        Args:
            trading_pair: Par de trading
            
        Returns:
            Comisión como porcentaje (0.001 = 0.1%)
        """
        return self.fees

    def get_exchange_rate(self, from_asset, to_asset):
        """
        Obtener tipo de cambio entre dos activos.
        
        Args:
            from_asset: Activo de origen
            to_asset: Activo de destino
            
        Returns:
            Tasa de cambio
        """
        try:
            ticker = self.client.fetch_ticker(f"{from_asset}/{to_asset}")
            return ticker.get("last", 0)
        except Exception as e:
            logging.warning(f"[{self.name}] Error al obtener tasa {from_asset}/{to_asset}: {e}")
            return 0