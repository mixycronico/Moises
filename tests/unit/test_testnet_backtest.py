"""
Pruebas unitarias para el backtesting usando datos históricos de Binance Testnet.

Este módulo prueba la funcionalidad de backtesting del sistema Genesis
utilizando datos históricos reales descargados de Binance Testnet.
"""

import os
import sys
import pytest
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("testnet_backtest")

# Asegurarse que podemos importar los módulos de Genesis
sys.path.insert(0, os.getcwd())

# Importar componentes del sistema
from genesis.strategies.base import Strategy, SignalType
from genesis.backtesting.engine import BacktestEngine
from genesis.risk.position_sizer import PositionSizer
from genesis.data.market_data import MarketDataManager


class TestnetDataAdapter:
    """
    Adaptador para cargar datos de Binance Testnet desde archivos CSV
    y convertirlos al formato que espera el sistema de backtesting.
    """
    
    def __init__(self, data_dir="./data/testnet"):
        """
        Inicializar el adaptador.
        
        Args:
            data_dir: Directorio donde se encuentran los datos de Testnet
        """
        self.data_dir = Path(data_dir)
        self.cached_data = {}
        
    def list_available_symbols(self):
        """
        Listar los símbolos disponibles en los archivos de datos.
        
        Returns:
            Lista de símbolos disponibles
        """
        symbols = set()
        for file_path in self.data_dir.glob("*.csv"):
            parts = file_path.stem.split("_")
            if len(parts) >= 2:
                symbol = f"{parts[0]}/{parts[1]}"
                symbols.add(symbol)
        
        return list(symbols)
    
    def list_available_timeframes(self, symbol):
        """
        Listar los timeframes disponibles para un símbolo.
        
        Args:
            symbol: Símbolo de trading
            
        Returns:
            Lista de timeframes disponibles
        """
        timeframes = set()
        symbol_clean = symbol.replace("/", "_")
        
        for file_path in self.data_dir.glob(f"{symbol_clean}_*.csv"):
            parts = file_path.stem.split("_")
            if len(parts) >= 3:
                timeframe = parts[2]
                timeframes.add(timeframe)
        
        return list(timeframes)
    
    async def load_market_data(self, symbol, timeframe):
        """
        Cargar datos de mercado desde archivo CSV.
        
        Args:
            symbol: Símbolo de trading
            timeframe: Marco temporal
            
        Returns:
            DataFrame con datos de mercado
        """
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self.cached_data:
            return self.cached_data[cache_key]
        
        symbol_clean = symbol.replace("/", "_")
        file_path = self.data_dir / f"{symbol_clean}_{timeframe}.csv"
        
        if not file_path.exists():
            logger.warning(f"No se encontró archivo para {symbol} - {timeframe}")
            return pd.DataFrame()
        
        try:
            # Cargar datos desde el archivo CSV
            df = pd.read_csv(file_path)
            
            # Asegurarse que las columnas están en el formato esperado
            if 'datetime' in df.columns:
                df['date'] = pd.to_datetime(df['datetime'])
                df.set_index('date', inplace=True)
            
            # Convertir a nombres de columnas estándar si es necesario
            rename_map = {
                'open': 'open', 
                'high': 'high', 
                'low': 'low', 
                'close': 'close', 
                'volume': 'volume'
            }
            df.rename(columns={col: rename_map[col] for col in rename_map if col in df.columns}, inplace=True)
            
            # Guardar en caché para futuras llamadas
            self.cached_data[cache_key] = df
            
            logger.info(f"Cargados {len(df)} registros para {symbol} - {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error al cargar datos para {symbol} - {timeframe}: {e}")
            return pd.DataFrame()


class SimpleMAStrategy(Strategy):
    """
    Estrategia simple de cruce de medias móviles para pruebas.
    """
    
    def __init__(self, name="simple_ma_strategy", fast_period=10, slow_period=30):
        """
        Inicializar la estrategia.
        
        Args:
            name: Nombre de la estrategia
            fast_period: Período para la media móvil rápida
            slow_period: Período para la media móvil lenta
        """
        super().__init__(name)
        self.fast_period = fast_period
        self.slow_period = slow_period
        
    async def generate_signal(self, symbol, data):
        """
        Generar señal de trading basada en el cruce de medias móviles.
        
        Args:
            symbol: Símbolo de trading
            data: DataFrame con datos OHLCV
            
        Returns:
            Señal de trading (compra, venta, mantener)
        """
        if data.empty:
            return {"signal": SignalType.HOLD, "symbol": symbol}
        
        # Calcular medias móviles si no existen
        if "ma_fast" not in data.columns:
            data["ma_fast"] = data["close"].rolling(window=self.fast_period).mean()
        if "ma_slow" not in data.columns:
            data["ma_slow"] = data["close"].rolling(window=self.slow_period).mean()
        
        # Obtener los valores más recientes
        last_row = data.iloc[-1]
        prev_row = data.iloc[-2] if len(data) > 1 else None
        
        # Verificar si hay un cruce
        if prev_row is not None:
            # Cruce alcista: la MA rápida cruza por encima de la MA lenta
            if (prev_row["ma_fast"] <= prev_row["ma_slow"]) and (last_row["ma_fast"] > last_row["ma_slow"]):
                return {"signal": SignalType.BUY, "symbol": symbol}
                
            # Cruce bajista: la MA rápida cruza por debajo de la MA lenta
            elif (prev_row["ma_fast"] >= prev_row["ma_slow"]) and (last_row["ma_fast"] < last_row["ma_slow"]):
                return {"signal": SignalType.SELL, "symbol": symbol}
        
        # Sin cruce, mantener
        return {"signal": SignalType.HOLD, "symbol": symbol}


class TestnetBacktestManager:
    """
    Gestor para ejecutar backtests con datos de Binance Testnet.
    """
    
    def __init__(self):
        """Inicializar el gestor de backtests."""
        self.data_adapter = TestnetDataAdapter()
        self.backtest_engine = BacktestEngine()
        
    async def run_simple_backtest(self, symbol, timeframe, strategy, start_date=None, end_date=None):
        """
        Ejecutar un backtest simple con una estrategia.
        
        Args:
            symbol: Símbolo de trading
            timeframe: Marco temporal
            strategy: Estrategia a probar
            start_date: Fecha de inicio (opcional)
            end_date: Fecha de fin (opcional)
            
        Returns:
            Resultados del backtest
        """
        # Cargar datos históricos
        data = await self.data_adapter.load_market_data(symbol, timeframe)
        if data.empty:
            logger.error(f"No se pudieron cargar datos para {symbol} - {timeframe}")
            return None
        
        # Filtrar por fechas si se especifican
        if start_date:
            start_date = pd.to_datetime(start_date)
            data = data[data.index >= start_date]
            
        if end_date:
            end_date = pd.to_datetime(end_date)
            data = data[data.index <= end_date]
            
        # Verificar que haya suficientes datos
        if len(data) < 50:
            logger.warning(f"Pocos datos para {symbol} - {timeframe}: {len(data)} registros")
        
        # Calcular indicadores
        params = {
            "sma_fast_period": getattr(strategy, "fast_period", 10),
            "sma_slow_period": getattr(strategy, "slow_period", 30)
        }
        data_with_indicators = await self.backtest_engine.calculate_indicators(data, params)
        
        # Generar señales
        data_with_signals = await self.backtest_engine.run_strategy(strategy, data_with_indicators, params)
        
        # Simular trading
        results = await self.backtest_engine.simulate_trading(data_with_signals)
        
        return results


@pytest.fixture
def testnet_data_adapter():
    """Fixture que proporciona un adaptador de datos de Testnet."""
    return TestnetDataAdapter()


@pytest.fixture
def simple_ma_strategy():
    """Fixture que proporciona una estrategia de cruce de medias móviles."""
    return SimpleMAStrategy(fast_period=5, slow_period=20)


@pytest.fixture
def backtest_manager():
    """Fixture que proporciona un gestor de backtests."""
    return TestnetBacktestManager()


@pytest.mark.asyncio
async def test_list_available_symbols(testnet_data_adapter):
    """Probar que se pueden listar los símbolos disponibles."""
    symbols = testnet_data_adapter.list_available_symbols()
    assert len(symbols) > 0
    assert "BTC/USDT" in symbols
    logger.info(f"Símbolos disponibles: {symbols}")


@pytest.mark.asyncio
async def test_list_available_timeframes(testnet_data_adapter):
    """Probar que se pueden listar los timeframes disponibles para un símbolo."""
    timeframes = testnet_data_adapter.list_available_timeframes("BTC/USDT")
    assert len(timeframes) > 0
    logger.info(f"Timeframes disponibles para BTC/USDT: {timeframes}")


@pytest.mark.asyncio
async def test_load_market_data(testnet_data_adapter):
    """Probar que se pueden cargar datos de mercado desde archivos CSV."""
    data = await testnet_data_adapter.load_market_data("BTC/USDT", "1h")
    assert not data.empty
    assert "open" in data.columns
    assert "high" in data.columns
    assert "low" in data.columns
    assert "close" in data.columns
    assert "volume" in data.columns
    logger.info(f"Cargados {len(data)} registros para BTC/USDT - 1h")


@pytest.mark.asyncio
async def test_simple_ma_strategy(simple_ma_strategy):
    """Probar que la estrategia de cruce de medias móviles funciona correctamente."""
    # Crear datos de prueba
    dates = pd.date_range("2025-01-01", periods=50, freq="1h")
    data = pd.DataFrame({
        "open": np.random.normal(40000, 1000, 50),
        "high": np.random.normal(40500, 1000, 50),
        "low": np.random.normal(39500, 1000, 50),
        "close": np.random.normal(40000, 1000, 50),
        "volume": np.random.normal(10, 5, 50),
    }, index=dates)
    
    # Calcular medias móviles
    data["ma_fast"] = data["close"].rolling(window=5).mean()
    data["ma_slow"] = data["close"].rolling(window=20).mean()
    
    # Generar señal
    signal = await simple_ma_strategy.generate_signal("BTC/USDT", data)
    
    # Verificar que la señal es válida
    assert "signal" in signal
    assert signal["signal"] in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
    assert signal["symbol"] == "BTC/USDT"
    
    logger.info(f"Señal generada: {signal['signal']}")


@pytest.mark.asyncio
async def test_backtest_with_testnet_data(backtest_manager, simple_ma_strategy):
    """Probar que se puede ejecutar un backtest con datos reales de Testnet."""
    
    # Ejecutar backtest para BTC/USDT - 1h
    results = await backtest_manager.run_simple_backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        strategy=simple_ma_strategy
    )
    
    # Verificar que hay resultados
    assert results is not None
    
    # Imprimir métricas clave
    logger.info(f"Resultado del backtest para BTC/USDT - 1h con estrategia de MA simple:")
    
    if "total_return" in results:
        logger.info(f"Retorno total: {results['total_return'] * 100:.2f}%")
        
    if "sharpe_ratio" in results:
        logger.info(f"Ratio de Sharpe: {results['sharpe_ratio']:.2f}")
        
    if "max_drawdown" in results:
        logger.info(f"Máximo drawdown: {results['max_drawdown'] * 100:.2f}%")
        
    if "total_trades" in results:
        logger.info(f"Número total de trades: {results['total_trades']}")
        
    if "win_rate" in results:
        logger.info(f"Tasa de acierto: {results['win_rate'] * 100:.2f}%")


@pytest.mark.asyncio
async def test_backtest_with_multiple_timeframes(backtest_manager, simple_ma_strategy):
    """Probar que se puede ejecutar un backtest con múltiples timeframes."""
    timeframes = ["1h", "4h", "1d"]
    
    for tf in timeframes:
        # Ejecutar backtest para cada timeframe
        results = await backtest_manager.run_simple_backtest(
            symbol="BTC/USDT",
            timeframe=tf,
            strategy=simple_ma_strategy
        )
        
        # Verificar que hay resultados
        if results:
            logger.info(f"Resultado del backtest para BTC/USDT - {tf}:")
            logger.info(f"Retorno total: {results.get('total_return', 0) * 100:.2f}%")
            logger.info(f"Número total de trades: {results.get('total_trades', 0)}")
        else:
            logger.warning(f"No se pudieron obtener resultados para BTC/USDT - {tf}")


if __name__ == "__main__":
    logging.info("Ejecutando pruebas de backtesting con datos de Binance Testnet")
    pytest.main(["-xvs", __file__])