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
            
            # Configuración básica
            exchange_config = {
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'timeout': 10000
            }
            
            # Añadir configuración específica para testnet si está disponible
            if config.get("testnet", False):
                if exchange_name == 'binance':
                    exchange_config['options'] = {'defaultType': 'future'}
                    exchange_config['urls'] = {
                        'api': {
                            'public': 'https://testnet.binancefuture.com/fapi/v1',
                            'private': 'https://testnet.binancefuture.com/fapi/v1',
                            'fapiPublic': 'https://testnet.binancefuture.com/fapi/v1',
                            'fapiPrivate': 'https://testnet.binancefuture.com/fapi/v1'
                        },
                        'test': {
                            'public': 'https://testnet.binancefuture.com/fapi/v1',
                            'private': 'https://testnet.binancefuture.com/fapi/v1'
                        }
                    }
                    logging.info(f"[{self.name}] Configurado para usar Binance Testnet")
                elif exchange_name == 'kucoin':
                    exchange_config['urls'] = {'api': 'https://openapi-sandbox.kucoin.com'}
                    logging.info(f"[{self.name}] Configurado para usar KuCoin Sandbox")
                else:
                    logging.warning(f"[{self.name}] Testnet no disponible para {exchange_name}")
            
            self.client = exchange_class(exchange_config)
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
            
    def fetch_historical_data(self, symbol, timeframe='1h', limit=500, since=None):
        """
        Obtener datos históricos del exchange.
        
        Args:
            symbol: Par de trading (ej: 'BTC/USDT')
            timeframe: Marco temporal ('1m', '5m', '15m', '1h', '4h', '1d', etc.)
            limit: Cantidad máxima de velas a obtener
            since: Timestamp desde donde obtener datos (opcional)
            
        Returns:
            Lista de velas OHLCV o None si hay error
        """
        try:
            # Asegurarse de que el símbolo sea válido para el exchange
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"  # Añadir USDT por defecto si no se especifica
                
            logging.info(f"[{self.name}] Obteniendo datos históricos para {symbol}, timeframe {timeframe}")
            
            # Obtener datos históricos
            ohlcv_data = self.client.fetch_ohlcv(symbol, timeframe, limit=limit, since=since)
            
            # Convertir a formato adecuado para el sistema
            formatted_data = []
            for candle in ohlcv_data:
                timestamp, open_price, high, low, close, volume = candle
                formatted_data.append({
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume
                })
                
            logging.info(f"[{self.name}] Obtenidos {len(formatted_data)} registros históricos")
            return formatted_data
        except Exception as e:
            logging.error(f"[{self.name}] Error al obtener datos históricos: {e}")
            return None