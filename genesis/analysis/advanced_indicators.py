"""
Indicadores técnicos avanzados para análisis de mercado.

Este módulo implementa indicadores técnicos avanzados como Ichimoku Kinko Hyo,
Bollinger Bands dinámicas, Market Profile, y otros indicadores adaptados para
trading de criptomonedas.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class AdvancedIndicators:
    """
    Clase para calcular y aplicar indicadores técnicos avanzados.
    
    Proporciona funciones para calcular Ichimoku, Bollinger Bands adaptativas,
    Market Profile y otros indicadores técnicos avanzados que pueden usarse
    como features para modelos de ML/RL.
    """
    
    def __init__(self):
        """Inicializar calculador de indicadores avanzados."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("AdvancedIndicators inicializado")
    
    def add_ichimoku(self, 
                     df: pd.DataFrame, 
                     tenkan_period: int = 9, 
                     kijun_period: int = 26, 
                     senkou_span_b_period: int = 52,
                     displacement: int = 26) -> pd.DataFrame:
        """
        Calcular y añadir Ichimoku Kinko Hyo al DataFrame.
        
        Args:
            df: DataFrame con datos OHLCV
            tenkan_period: Período para Tenkan-sen (conversión)
            kijun_period: Período para Kijun-sen (base)
            senkou_span_b_period: Período para Senkou Span B
            displacement: Desplazamiento para Senkou Span
            
        Returns:
            DataFrame con indicadores Ichimoku añadidos
        """
        # Verificar columnas requeridas
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Columna requerida no encontrada: {col}")
        
        # Crear copia para no modificar el original
        df_result = df.copy()
        
        # Tenkan-sen (línea de conversión): (máx(9) + mín(9)) / 2
        high_tenkan = df_result['high'].rolling(window=tenkan_period).max()
        low_tenkan = df_result['low'].rolling(window=tenkan_period).min()
        df_result['tenkan_sen'] = (high_tenkan + low_tenkan) / 2
        
        # Kijun-sen (línea base): (máx(26) + mín(26)) / 2
        high_kijun = df_result['high'].rolling(window=kijun_period).max()
        low_kijun = df_result['low'].rolling(window=kijun_period).min()
        df_result['kijun_sen'] = (high_kijun + low_kijun) / 2
        
        # Senkou Span A (línea adelantada A): (Tenkan-sen + Kijun-sen) / 2 desplazada 26 períodos
        df_result['senkou_span_a'] = ((df_result['tenkan_sen'] + df_result['kijun_sen']) / 2).shift(displacement)
        
        # Senkou Span B (línea adelantada B): (máx(52) + mín(52)) / 2 desplazada 26 períodos
        high_senkou = df_result['high'].rolling(window=senkou_span_b_period).max()
        low_senkou = df_result['low'].rolling(window=senkou_span_b_period).min()
        df_result['senkou_span_b'] = ((high_senkou + low_senkou) / 2).shift(displacement)
        
        # Chikou Span (línea rezagada): Precio de cierre desplazado -26 períodos
        df_result['chikou_span'] = df_result['close'].shift(-displacement)
        
        # Añadir señales de Ichimoku
        df_result = self._add_ichimoku_signals(df_result)
        
        return df_result
    
    def _add_ichimoku_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añadir señales basadas en Ichimoku.
        
        Args:
            df: DataFrame con indicadores Ichimoku
            
        Returns:
            DataFrame con señales añadidas
        """
        # Crear copia para no modificar el original
        df_result = df.copy()
        
        # Señal TK Cross (Tenkan/Kijun Cross)
        df_result['tk_cross'] = 0
        # TK Cross bullish: Tenkan cruza por encima de Kijun
        tk_cross_bullish = (df_result['tenkan_sen'] > df_result['kijun_sen']) & (df_result['tenkan_sen'].shift(1) <= df_result['kijun_sen'].shift(1))
        # TK Cross bearish: Tenkan cruza por debajo de Kijun
        tk_cross_bearish = (df_result['tenkan_sen'] < df_result['kijun_sen']) & (df_result['tenkan_sen'].shift(1) >= df_result['kijun_sen'].shift(1))
        
        df_result.loc[tk_cross_bullish, 'tk_cross'] = 1  # Bullish
        df_result.loc[tk_cross_bearish, 'tk_cross'] = -1  # Bearish
        
        # Señal de Kumo (nube)
        df_result['kumo_color'] = 0  # 1: bullish (verde), -1: bearish (rojo)
        df_result.loc[df_result['senkou_span_a'] > df_result['senkou_span_b'], 'kumo_color'] = 1
        df_result.loc[df_result['senkou_span_a'] < df_result['senkou_span_b'], 'kumo_color'] = -1
        
        # Precio sobre/bajo el Kumo
        df_result['price_vs_kumo'] = 0
        # Precio sobre el Kumo: bullish
        price_above_kumo = (df_result['close'] > df_result['senkou_span_a']) & (df_result['close'] > df_result['senkou_span_b'])
        # Precio bajo el Kumo: bearish
        price_below_kumo = (df_result['close'] < df_result['senkou_span_a']) & (df_result['close'] < df_result['senkou_span_b'])
        # Precio dentro del Kumo: neutral
        
        df_result.loc[price_above_kumo, 'price_vs_kumo'] = 1
        df_result.loc[price_below_kumo, 'price_vs_kumo'] = -1
        
        # Señal combinada de Ichimoku (suma de todas las señales)
        df_result['ichimoku_signal'] = df_result['tk_cross'] + df_result['kumo_color'] + df_result['price_vs_kumo']
        
        return df_result
    
    def add_dynamic_bollinger_bands(self, 
                                   df: pd.DataFrame, 
                                   window: int = 20, 
                                   num_std_dev: float = 2.0,
                                   adaptive_std: bool = True,
                                   use_atr: bool = True,
                                   atr_window: int = 14) -> pd.DataFrame:
        """
        Calcular y añadir Bollinger Bands dinámicas al DataFrame.
        
        Args:
            df: DataFrame con datos OHLCV
            window: Periodo para la media móvil
            num_std_dev: Número de desviaciones estándar
            adaptive_std: Si es True, ajusta las bandas según volatilidad
            use_atr: Si es True, utiliza ATR para ajustar bandas
            atr_window: Periodo para ATR
            
        Returns:
            DataFrame con Bollinger Bands dinámicas añadidas
        """
        # Verificar columnas requeridas
        required_columns = ['close']
        if use_atr:
            required_columns.extend(['high', 'low'])
        
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Columna requerida no encontrada: {col}")
        
        # Crear copia para no modificar el original
        df_result = df.copy()
        
        # Calcular media móvil
        df_result['bb_ma'] = df_result['close'].rolling(window=window).mean()
        
        # Calcular desviación estándar
        df_result['bb_std'] = df_result['close'].rolling(window=window).std()
        
        # Ajustar desviación estándar si se solicita
        if adaptive_std:
            # Calcular volatilidad histórica (HV)
            df_result['log_returns'] = np.log(df_result['close'] / df_result['close'].shift(1))
            df_result['hv'] = df_result['log_returns'].rolling(window=window).std() * np.sqrt(252)  # Anualizada
            
            # Ajustar num_std_dev según la volatilidad relativa
            # Más volatilidad = bandas más anchas
            volatility_ratio = df_result['hv'] / df_result['hv'].rolling(window=window*5).mean()
            # Limitar el ratio para evitar bandas extremas
            volatility_ratio = volatility_ratio.clip(0.5, 2.0)
            
            # Ajustar num_std_dev
            df_result['bb_std_dev_multiplier'] = num_std_dev * volatility_ratio
        else:
            df_result['bb_std_dev_multiplier'] = num_std_dev
        
        # Usar ATR para ajustar bandas si se solicita
        if use_atr:
            # Calcular True Range
            tr1 = df_result['high'] - df_result['low']
            tr2 = abs(df_result['high'] - df_result['close'].shift(1))
            tr3 = abs(df_result['low'] - df_result['close'].shift(1))
            
            df_result['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df_result['atr'] = df_result['tr'].rolling(window=atr_window).mean()
            
            # Ajustar bandas con ATR
            atr_factor = df_result['atr'] / df_result['close']
            # Normalizar ATR para evitar valores extremos
            atr_factor = atr_factor / atr_factor.rolling(window=window*5).mean()
            atr_factor = atr_factor.clip(0.5, 2.0)
            
            # Combinar con la desviación estándar
            if adaptive_std:
                df_result['bb_width_multiplier'] = (df_result['bb_std_dev_multiplier'] + atr_factor) / 2
            else:
                df_result['bb_width_multiplier'] = num_std_dev * atr_factor
        else:
            df_result['bb_width_multiplier'] = df_result['bb_std_dev_multiplier']
        
        # Calcular bandas
        df_result['bb_upper'] = df_result['bb_ma'] + df_result['bb_std'] * df_result['bb_width_multiplier']
        df_result['bb_lower'] = df_result['bb_ma'] - df_result['bb_std'] * df_result['bb_width_multiplier']
        
        # Añadir indicadores de banda
        df_result['bb_width'] = (df_result['bb_upper'] - df_result['bb_lower']) / df_result['bb_ma']
        df_result['bb_pct_b'] = (df_result['close'] - df_result['bb_lower']) / (df_result['bb_upper'] - df_result['bb_lower'])
        
        # Añadir señales de Bollinger
        df_result = self._add_bollinger_signals(df_result)
        
        return df_result
    
    def _add_bollinger_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añadir señales basadas en Bollinger Bands.
        
        Args:
            df: DataFrame con Bollinger Bands
            
        Returns:
            DataFrame con señales añadidas
        """
        # Crear copia para no modificar el original
        df_result = df.copy()
        
        # Señal de banda: 1 (sobre la banda superior), 0 (dentro de bandas), -1 (bajo la banda inferior)
        df_result['bb_signal'] = 0
        df_result.loc[df_result['close'] > df_result['bb_upper'], 'bb_signal'] = 1
        df_result.loc[df_result['close'] < df_result['bb_lower'], 'bb_signal'] = -1
        
        # Señal de compresión/expansión: 1 (compresión), -1 (expansión)
        df_result['bb_squeeze'] = 0
        # Calcular banda histórica para comparar
        bb_width_avg = df_result['bb_width'].rolling(window=50).mean()
        
        # Detectar compresión (bandas estrechas)
        compression = df_result['bb_width'] < bb_width_avg * 0.85
        # Detectar expansión (bandas anchas)
        expansion = df_result['bb_width'] > bb_width_avg * 1.15
        
        df_result.loc[compression, 'bb_squeeze'] = 1
        df_result.loc[expansion, 'bb_squeeze'] = -1
        
        # Señal de reversión a la media
        df_result['bb_mean_reversion'] = 0
        # Sobreventa extrema (potencial compra)
        oversold = (df_result['bb_pct_b'] < 0.05) & (df_result['bb_pct_b'].shift(1) < 0.05)
        # Sobrecompra extrema (potencial venta)
        overbought = (df_result['bb_pct_b'] > 0.95) & (df_result['bb_pct_b'].shift(1) > 0.95)
        
        df_result.loc[oversold, 'bb_mean_reversion'] = 1
        df_result.loc[overbought, 'bb_mean_reversion'] = -1
        
        return df_result
    
    def add_market_profile(self, 
                          df: pd.DataFrame, 
                          window: int = 20,
                          num_bins: int = 10,
                          use_volume: bool = True) -> pd.DataFrame:
        """
        Calcular y añadir Market Profile (Value Area) al DataFrame.
        
        Args:
            df: DataFrame con datos OHLCV
            window: Periodo para el análisis de Market Profile
            num_bins: Número de niveles de precio para el histograma
            use_volume: Si es True, pondera los niveles por volumen
            
        Returns:
            DataFrame con indicadores de Market Profile añadidos
        """
        # Verificar columnas requeridas
        required_columns = ['high', 'low', 'close']
        if use_volume:
            required_columns.append('volume')
        
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Columna requerida no encontrada: {col}")
        
        # Crear copia para no modificar el original
        df_result = df.copy()
        
        # Para cada fila, calcular el Value Area de los 'window' periodos anteriores
        value_areas = []
        
        for i in range(len(df_result)):
            if i < window:
                # No hay suficientes datos históricos
                value_areas.append({'poc': np.nan, 'vah': np.nan, 'val': np.nan, 'value_area_ratio': np.nan})
                continue
            
            # Obtener datos de la ventana
            window_data = df_result.iloc[i-window:i]
            
            # Calcular POC (Point of Control) y Value Area
            poc, vah, val, value_area_ratio = self._calculate_value_area(window_data, num_bins, use_volume)
            
            value_areas.append({
                'poc': poc,
                'vah': vah,
                'val': val,
                'value_area_ratio': value_area_ratio
            })
        
        # Añadir al DataFrame
        df_result['mp_poc'] = [va['poc'] for va in value_areas]
        df_result['mp_vah'] = [va['vah'] for va in value_areas]
        df_result['mp_val'] = [va['val'] for va in value_areas]
        df_result['mp_va_ratio'] = [va['value_area_ratio'] for va in value_areas]
        
        # Calcular distancia del precio al POC
        df_result['mp_poc_distance'] = (df_result['close'] - df_result['mp_poc']) / df_result['mp_poc']
        
        # Indicar si el precio está dentro del Value Area
        df_result['mp_in_value_area'] = 0
        in_value_area = (df_result['close'] >= df_result['mp_val']) & (df_result['close'] <= df_result['mp_vah'])
        df_result.loc[in_value_area, 'mp_in_value_area'] = 1
        
        return df_result
    
    def _calculate_value_area(self, 
                            df: pd.DataFrame, 
                            num_bins: int = 10,
                            use_volume: bool = True) -> Tuple[float, float, float, float]:
        """
        Calcular Value Area a partir de datos históricos.
        
        Args:
            df: DataFrame con datos históricos
            num_bins: Número de niveles de precio
            use_volume: Si es True, pondera los niveles por volumen
            
        Returns:
            Tupla (POC, VAH, VAL, Value Area Ratio)
        """
        # Determinar rango de precios
        price_high = df['high'].max()
        price_low = df['low'].min()
        
        if price_high == price_low:
            return price_high, price_high, price_high, 1.0
        
        # Crear bins de precio
        price_bins = np.linspace(price_low, price_high, num_bins + 1)
        bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
        
        # Calcular histograma de precios
        if use_volume:
            # Ponderar por volumen
            hist_values = np.zeros(num_bins)
            
            for i, row in df.iterrows():
                # Determinar en qué bins caen los precios de la vela
                for j in range(num_bins):
                    if row['low'] <= price_bins[j+1] and row['high'] >= price_bins[j]:
                        # Estimar qué porcentaje de la vela cae en este bin
                        overlap = min(row['high'], price_bins[j+1]) - max(row['low'], price_bins[j])
                        ratio = overlap / (row['high'] - row['low']) if row['high'] > row['low'] else 1.0
                        hist_values[j] += row['volume'] * ratio
        else:
            # Sin ponderar por volumen, solo contar ocurrencias
            hist_values = np.zeros(num_bins)
            
            for i, row in df.iterrows():
                for j in range(num_bins):
                    if row['low'] <= price_bins[j+1] and row['high'] >= price_bins[j]:
                        hist_values[j] += 1
        
        if hist_values.sum() == 0:
            return np.nan, np.nan, np.nan, np.nan
        
        # Encontrar POC (Point of Control) - nivel con más volumen/ocurrencias
        poc_idx = np.argmax(hist_values)
        poc = bin_centers[poc_idx]
        
        # Calcular Value Area (70% del volumen total alrededor del POC)
        total_value = hist_values.sum()
        target_value = total_value * 0.7
        
        # Comenzar desde el POC y expandir hacia arriba y abajo
        current_value = hist_values[poc_idx]
        
        lower_idx = poc_idx
        upper_idx = poc_idx
        
        while current_value < target_value and (lower_idx > 0 or upper_idx < num_bins - 1):
            # Expandir hacia arriba o abajo según qué lado tiene más volumen
            lower_val = hist_values[lower_idx - 1] if lower_idx > 0 else 0
            upper_val = hist_values[upper_idx + 1] if upper_idx < num_bins - 1 else 0
            
            if lower_val > upper_val:
                lower_idx -= 1
                current_value += lower_val
            else:
                upper_idx += 1
                current_value += upper_val
        
        # Value Area High y Low
        vah = price_bins[upper_idx + 1]
        val = price_bins[lower_idx]
        
        # Value Area Ratio (qué porcentaje del rango total representa)
        value_area_ratio = (vah - val) / (price_high - price_low)
        
        return poc, vah, val, value_area_ratio
    
    def add_hurst_exponent(self, 
                          df: pd.DataFrame, 
                          column: str = 'close',
                          min_window: int = 10,
                          max_window: int = 100,
                          num_windows: int = 5,
                          window: int = 50) -> pd.DataFrame:
        """
        Calcular y añadir el exponente de Hurst al DataFrame.
        
        El exponente de Hurst mide la tendencia y mean-reversion:
        - H > 0.5: Serie con tendencia
        - H = 0.5: Serie aleatoria (random walk)
        - H < 0.5: Serie con mean-reversion
        
        Args:
            df: DataFrame con datos
            column: Columna con los precios
            min_window: Ventana mínima para calcular el rango reescalado
            max_window: Ventana máxima para calcular el rango reescalado
            num_windows: Número de ventanas entre min_window y max_window
            window: Ventana para calcular el exponente de Hurst
            
        Returns:
            DataFrame con el exponente de Hurst añadido
        """
        # Verificar columna requerida
        if column not in df.columns:
            raise ValueError(f"Columna no encontrada: {column}")
        
        # Crear copia para no modificar el original
        df_result = df.copy()
        
        # Lista de ventanas para calcular el rango reescalado
        windows = np.logspace(np.log10(min_window), np.log10(max_window), num_windows, dtype=int)
        windows = np.unique(windows)  # Eliminar duplicados
        
        # Calcular exponente de Hurst para cada ventana de 'window' datos
        hurst_values = []
        
        for i in range(len(df_result)):
            if i < window:
                # No hay suficientes datos históricos
                hurst_values.append(np.nan)
                continue
            
            # Obtener datos de la ventana
            price_series = df_result[column].iloc[i-window:i].values
            
            # Calcular exponente de Hurst
            hurst = self._calculate_hurst(price_series, windows)
            hurst_values.append(hurst)
        
        # Añadir al DataFrame
        df_result['hurst'] = hurst_values
        
        # Añadir señales basadas en Hurst
        df_result['market_type'] = 0  # 0: aleatorio, 1: tendencia, -1: mean-reversion
        df_result.loc[df_result['hurst'] > 0.6, 'market_type'] = 1
        df_result.loc[df_result['hurst'] < 0.4, 'market_type'] = -1
        
        return df_result
    
    def _calculate_hurst(self, price_series: np.ndarray, windows: np.ndarray) -> float:
        """
        Calcular exponente de Hurst para una serie de precios.
        
        Args:
            price_series: Serie de precios
            windows: Lista de ventanas para calcular el rango reescalado
            
        Returns:
            Exponente de Hurst
        """
        # Calcular retornos
        returns = np.diff(np.log(price_series))
        
        if len(returns) < max(windows):
            return np.nan
        
        # Calcular rango reescalado para cada ventana
        rs_values = []
        
        for w in windows:
            if len(returns) < w:
                continue
                
            # Número de bloques que caben en la serie
            num_blocks = int(len(returns) / w)
            
            if num_blocks < 1:
                continue
            
            # Calcular rango reescalado para cada bloque y promediar
            rs_by_block = []
            
            for i in range(num_blocks):
                block = returns[i*w:(i+1)*w]
                
                # Desviación acumulada respecto a la media
                mean_block = np.mean(block)
                cum_dev = np.cumsum(block - mean_block)
                
                # Rango
                r = np.max(cum_dev) - np.min(cum_dev)
                
                # Desviación estándar
                s = np.std(block)
                
                if s == 0:
                    continue
                
                # Rango reescalado
                rs = r / s
                rs_by_block.append(rs)
            
            if not rs_by_block:
                continue
                
            # Promedio de R/S para esta ventana
            rs_values.append((w, np.mean(rs_by_block)))
        
        if len(rs_values) < 2:
            return np.nan
        
        # Ajuste lineal en escala logarítmica
        log_windows = np.log10([item[0] for item in rs_values])
        log_rs = np.log10([item[1] for item in rs_values])
        
        # Pendiente = exponente de Hurst
        if len(log_windows) < 2:
            return np.nan
            
        hurst = np.polyfit(log_windows, log_rs, 1)[0]
        
        return hurst
    
    def add_market_making_indicators(self, 
                                   df: pd.DataFrame, 
                                   spread_window: int = 20,
                                   volume_window: int = 20,
                                   volatility_window: int = 20) -> pd.DataFrame:
        """
        Añadir indicadores útiles para estrategias de market making.
        
        Args:
            df: DataFrame con datos OHLCV
            spread_window: Ventana para calcular el spread medio
            volume_window: Ventana para calcular tendencias de volumen
            volatility_window: Ventana para calcular volatilidad
            
        Returns:
            DataFrame con indicadores de market making
        """
        # Verificar columnas requeridas
        required_columns = ['high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Columna requerida no encontrada: {col}")
        
        # Crear copia para no modificar el original
        df_result = df.copy()
        
        # Calcular spread histórico (High-Low) / Close
        df_result['spread'] = (df_result['high'] - df_result['low']) / df_result['close']
        
        # Spread medio según ventana
        df_result['avg_spread'] = df_result['spread'].rolling(window=spread_window).mean()
        
        # Ratio spread actual / spread medio
        df_result['spread_ratio'] = df_result['spread'] / df_result['avg_spread']
        
        # Volumen normalizado
        df_result['norm_volume'] = df_result['volume'] / df_result['volume'].rolling(window=volume_window).mean()
        
        # Volatilidad (desviación estándar de retornos)
        df_result['log_returns'] = np.log(df_result['close'] / df_result['close'].shift(1))
        df_result['volatility'] = df_result['log_returns'].rolling(window=volatility_window).std()
        
        # Volatilidad normalizada (actual vs promedio)
        df_result['norm_volatility'] = df_result['volatility'] / df_result['volatility'].rolling(window=volatility_window*2).mean()
        
        # Señal de spread óptimo para market making
        # Un spread óptimo se basa en volatilidad y volumen
        df_result['optimal_spread'] = df_result['avg_spread'] * df_result['norm_volatility'] * np.sqrt(df_result['norm_volume'])
        
        # Señal de actividad de market making
        df_result['mm_activity'] = ((df_result['high'] - df_result['low']) / df_result['avg_spread']) * df_result['norm_volume']
        
        # Señal de dirección probable del precio basada en desequilibrio de órdenes
        # (simulado aquí sin datos de orderbook)
        df_result['mm_pressure'] = 0
        df_result.loc[(df_result['close'] > df_result['open']) & (df_result['norm_volume'] > 1), 'mm_pressure'] = 1  # Presión alcista
        df_result.loc[(df_result['close'] < df_result['open']) & (df_result['norm_volume'] > 1), 'mm_pressure'] = -1  # Presión bajista
        
        return df_result
    
    def add_all_indicators(self, 
                          df: pd.DataFrame, 
                          add_ichimoku: bool = True,
                          add_bollinger: bool = True,
                          add_market_profile: bool = True,
                          add_hurst: bool = True,
                          add_market_making: bool = True) -> pd.DataFrame:
        """
        Añadir todos los indicadores avanzados al DataFrame.
        
        Args:
            df: DataFrame con datos OHLCV
            add_ichimoku: Si es True, añade indicadores Ichimoku
            add_bollinger: Si es True, añade Bollinger Bands dinámicas
            add_market_profile: Si es True, añade Market Profile
            add_hurst: Si es True, añade exponente de Hurst
            add_market_making: Si es True, añade indicadores de market making
            
        Returns:
            DataFrame con todos los indicadores seleccionados
        """
        result_df = df.copy()
        
        if add_ichimoku:
            result_df = self.add_ichimoku(result_df)
        
        if add_bollinger:
            result_df = self.add_dynamic_bollinger_bands(result_df)
        
        if add_market_profile:
            result_df = self.add_market_profile(result_df)
        
        if add_hurst:
            result_df = self.add_hurst_exponent(result_df)
        
        if add_market_making:
            result_df = self.add_market_making_indicators(result_df)
        
        return result_df