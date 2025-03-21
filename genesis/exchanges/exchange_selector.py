"""
Selección inteligente de exchanges para el sistema Genesis.

Este módulo proporciona funcionalidades para seleccionar el exchange óptimo
para una operación basándose en factores como liquidez, comisiones y latencia.
"""

import logging
import asyncio
from typing import Dict, Optional, Tuple, Any

from genesis.exchanges.api_client import APIClient

class ExchangeSelector:
    """Selecciona el mejor exchange basado en liquidez, comisiones y latencia."""

    def __init__(self, exchanges: Dict[str, APIClient]):
        """
        Inicializar el selector de exchanges.
        
        Args:
            exchanges: Diccionario de clientes API de exchanges
        """
        self.exchanges = exchanges
        self.logger = logging.getLogger(__name__)

    async def get_best_exchange(self, trading_pair: str) -> Optional[str]:
        """
        Selecciona el mejor exchange para un trading pair específico.
        
        Args:
            trading_pair: Par de trading (ej: 'BTC/USDT')
            
        Returns:
            Nombre del mejor exchange o None si no hay disponibles
        """
        scores = {}
        tasks = [self.evaluate_exchange(name, client, trading_pair) for name, client in self.exchanges.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                self.logger.warning(f"Error evaluando exchange: {result}")
                continue
                
            name, score = result
            if score:
                scores[name] = score

        return max(scores, key=scores.get) if scores else None

    async def evaluate_exchange(self, exchange_name: str, client: APIClient, trading_pair: str) -> Tuple[str, Optional[float]]:
        """
        Evaluar un exchange específico para un par de trading.
        
        Args:
            exchange_name: Nombre del exchange
            client: Cliente API del exchange
            trading_pair: Par de trading
            
        Returns:
            Tupla (nombre_exchange, puntuación) o (nombre_exchange, None) si hay error
        """
        try:
            # Convertir los métodos síncronos a asíncronos para compatibilidad
            loop = asyncio.get_event_loop()
            
            order_book_future = loop.run_in_executor(None, lambda: client.get_order_book(trading_pair))
            latency_future = loop.run_in_executor(None, client.get_latency)
            fees_future = loop.run_in_executor(None, lambda: client.get_trading_fees(trading_pair))
            
            order_book = await order_book_future
            latency = await latency_future
            fees = await fees_future
            
            if not order_book:
                return exchange_name, None
            
            spread = order_book["ask"] - order_book["bid"]
            liquidity = 1 / spread if spread > 0 else 0
            priority = client.priority
            
            # Normalizar y combinar factores para obtener una puntuación
            # Mayor liquidez, menores fees, menor latencia = mayor puntuación
            score = (liquidity / (fees * latency)) * priority if fees and latency else 0
            
            self.logger.debug(
                f"{exchange_name}: liquidity={liquidity:.6f}, fees={fees:.6f}, "
                f"latency={latency:.2f}ms, priority={priority}, score={score:.2f}"
            )
            
            return exchange_name, score
        except Exception as e:
            self.logger.error(f"Error evaluando {exchange_name}: {e}")
            return exchange_name, None