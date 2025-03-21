"""
Estrategia de cruce de medias móviles para trading.

Este módulo implementa una estrategia de trading basada en el cruce
de medias móviles simples (SMA).
"""
import logging
from typing import Dict, Any

import pandas as pd

from genesis.strategies.base import Strategy, SignalType

class MovingAverageCrossoverStrategy(Strategy):
    """
    Estrategia basada en el cruce de medias móviles.
    
    Esta estrategia genera señales de compra cuando la media móvil rápida (corto plazo)
    cruza por encima de la media móvil lenta (largo plazo), y señales de venta cuando
    cruza por debajo.
    """
    
    def __init__(
        self, 
        name: str = "ma_crossover",
        fast_period: int = 20, 
        slow_period: int = 50,
        signal_column: str = "close"
    ):
        """
        Inicializar la estrategia.
        
        Args:
            name: Nombre de la estrategia
            fast_period: Período para la media móvil rápida
            slow_period: Período para la media móvil lenta
            signal_column: Columna sobre la que calcular las medias móviles
        """
        super().__init__(name)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_column = signal_column
        self.logger = logging.getLogger(__name__)
        
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generar señales de trading basadas en el cruce de medias móviles.
        
        Args:
            symbol: Símbolo de trading
            data: DataFrame con datos OHLCV
            
        Returns:
            Diccionario con la señal generada
        """
        # Verificar que hay suficientes datos
        if len(data) < self.slow_period:
            self.logger.warning(f"Datos insuficientes para calcular SMA: {len(data)} < {self.slow_period}")
            return {"signal_type": SignalType.HOLD, "symbol": symbol}
        
        # Calcular medias móviles
        data_copy = data.copy()
        data_copy['sma_fast'] = data_copy[self.signal_column].rolling(window=self.fast_period).mean()
        data_copy['sma_slow'] = data_copy[self.signal_column].rolling(window=self.slow_period).mean()
        
        # Identificar cruces
        data_copy['sma_cross'] = 0
        
        # Cruce alcista: SMA rápida cruza por encima de la SMA lenta
        data_copy.loc[(data_copy['sma_fast'] > data_copy['sma_slow']) & 
                      (data_copy['sma_fast'].shift(1) <= data_copy['sma_slow'].shift(1)), 'sma_cross'] = 1
        
        # Cruce bajista: SMA rápida cruza por debajo de la SMA lenta
        data_copy.loc[(data_copy['sma_fast'] < data_copy['sma_slow']) & 
                      (data_copy['sma_fast'].shift(1) >= data_copy['sma_slow'].shift(1)), 'sma_cross'] = -1
        
        # Obtener la última señal
        current_price = data_copy[self.signal_column].iloc[-1]
        current_cross = data_copy['sma_cross'].iloc[-1]
        
        # Generar señal según el cruce
        if current_cross == 1:
            self.logger.info(f"Señal de COMPRA para {symbol} a {current_price} (cruce de SMA)")
            return {
                "signal_type": SignalType.BUY,
                "symbol": symbol,
                "price": current_price,
                "timestamp": data_copy.index[-1],
                "indicators": {
                    "sma_fast": data_copy['sma_fast'].iloc[-1],
                    "sma_slow": data_copy['sma_slow'].iloc[-1]
                }
            }
        elif current_cross == -1:
            self.logger.info(f"Señal de VENTA para {symbol} a {current_price} (cruce de SMA)")
            return {
                "signal_type": SignalType.SELL,
                "symbol": symbol,
                "price": current_price,
                "timestamp": data_copy.index[-1],
                "indicators": {
                    "sma_fast": data_copy['sma_fast'].iloc[-1],
                    "sma_slow": data_copy['sma_slow'].iloc[-1]
                }
            }
        else:
            return {
                "signal_type": SignalType.HOLD,
                "symbol": symbol,
                "price": current_price,
                "timestamp": data_copy.index[-1],
                "indicators": {
                    "sma_fast": data_copy['sma_fast'].iloc[-1],
                    "sma_slow": data_copy['sma_slow'].iloc[-1]
                }
            }
            
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcular los indicadores necesarios para la estrategia.
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            DataFrame con indicadores añadidos
        """
        df_copy = df.copy()
        
        # Calcular medias móviles
        df_copy['sma_fast'] = df_copy[self.signal_column].rolling(window=self.fast_period).mean()
        df_copy['sma_slow'] = df_copy[self.signal_column].rolling(window=self.slow_period).mean()
        
        return df_copy