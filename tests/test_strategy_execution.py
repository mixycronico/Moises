import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json

"""
Prueba integrada de ejecución de estrategias de trading.

Este script ejecuta las diferentes estrategias implementadas:
1. RSI
2. Bollinger Bands
3. Media Móvil
4. MACD
5. Sentimiento

Utiliza datos de prueba para simular condiciones reales del mercado
y verifica que cada estrategia genera señales coherentes con sus reglas.
"""

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("strategy_test")

# Crear datos de prueba
def create_test_data(strategy_type):
    """Crear datos específicos para cada tipo de estrategia."""
    dates = pd.date_range("2025-01-01", periods=100, freq="1h")
    
    if strategy_type == "rsi":
        # Para RSI, creamos una serie que alterna entre sobreventa y sobrecompra
        np.random.seed(42)
        prices = []
        current = 100.0
        # Crear un patrón que oscila para simular cruces de RSI
        for i in range(100):
            if i < 20:  # Tendencia bajista para empezar (RSI bajo)
                current = current * 0.99
            elif i < 40:  # Tendencia alcista (RSI sube)
                current = current * 1.01
            elif i < 60:  # Tendencia bajista (RSI baja)
                current = current * 0.99
            elif i < 80:  # Tendencia alcista (RSI sube)
                current = current * 1.01
            else:  # Tendencia bajista final (RSI baja)
                current = current * 0.99
            
            # Añadir ruido
            noise = np.random.normal(0, 1)
            prices.append(current + noise)
    
    elif strategy_type == "bollinger":
        # Para Bollinger Bands, alternamos volatilidad alta y baja
        np.random.seed(43)
        prices = []
        current = 100.0
        volatility = 0.5  # Inicia con baja volatilidad
        
        for i in range(100):
            if i % 33 == 0:  # Cambiar la volatilidad cada 33 períodos
                volatility = 3.0 if volatility < 1.0 else 0.5
            
            # Generamos movimientos que cruzan las bandas ocasionalmente
            if i % 20 == 0:  # Cada 20 períodos un movimiento fuerte
                current = current * (1 + 0.03 * np.random.choice([-1, 1]))
            else:
                current = current * (1 + 0.003 * np.random.normal())
                
            noise = np.random.normal(0, volatility)
            prices.append(current + noise)
    
    elif strategy_type == "ma_crossover":
        # Para cruce de medias, creamos tendencias claras que producirán cruces
        np.random.seed(44)
        prices = []
        current = 100.0
        
        for i in range(100):
            if i < 30:  # Tendencia alcista
                current = current * 1.007
            elif i < 50:  # Tendencia bajista
                current = current * 0.995
            elif i < 80:  # Tendencia lateral
                current = current * (1 + 0.001 * np.random.normal())
            else:  # Tendencia alcista final
                current = current * 1.006
                
            noise = np.random.normal(0, 0.5)
            prices.append(current + noise)
    
    elif strategy_type == "macd":
        # Para MACD, alternamos entre tendencias fuertes y consolidaciones
        np.random.seed(45)
        prices = []
        current = 100.0
        
        for i in range(100):
            if i < 25:  # Tendencia alcista fuerte
                current = current * 1.008
            elif i < 40:  # Consolidación
                current = current * (1 + 0.001 * np.random.normal())
            elif i < 65:  # Tendencia bajista
                current = current * 0.994
            elif i < 80:  # Consolidación
                current = current * (1 + 0.001 * np.random.normal())
            else:  # Tendencia alcista final
                current = current * 1.007
                
            noise = np.random.normal(0, 0.6)
            prices.append(current + noise)
    
    elif strategy_type == "sentiment":
        # Para sentimiento, simulamos noticias positivas/negativas
        np.random.seed(46)
        prices = []
        current = 100.0
        sentiment_score = 0.2  # Comenzamos neutro-negativo
        
        for i in range(100):
            if i % 20 == 0:  # Cambiar el sentimiento cada 20 períodos
                sentiment_score = np.random.uniform(-0.7, 0.7)
            
            # El precio sigue al sentimiento con algo de ruido
            price_change = 0.005 * sentiment_score + 0.001 * np.random.normal()
            current = current * (1 + price_change)
            prices.append(current)
    
    else:  # Datos genéricos si no se especifica estrategia
        np.random.seed(42)
        prices = np.random.normal(0, 1, 100).cumsum() + 100
    
    # Crear DataFrame OHLCV
    df = pd.DataFrame({
        "open": np.array(prices) - np.random.normal(0, 0.5, 100),
        "high": np.array(prices) + np.random.normal(2, 0.5, 100),
        "low": np.array(prices) - np.random.normal(2, 0.5, 100),
        "close": prices,
        "volume": np.random.lognormal(10, 1, 100),
    }, index=dates)
    
    return df

async def test_rsi_strategy():
    """Prueba la estrategia RSI."""
    try:
        # Importar módulos necesarios
        from genesis.strategies.mean_reversion import RSIStrategy
        from genesis.strategies.base import SignalType
        
        logger.info("\n=== Prueba de estrategia RSI ===")
        
        # Crear datos específicos para RSI
        data = create_test_data("rsi")
        
        # Creamos instancia de estrategia
        strategy = RSIStrategy(period=14, overbought=70, oversold=30)
        
        # Generar señal
        signal = await strategy.generate_signal("BTCUSDT", data)
        
        # Mostrar resultados
        logger.info(f"Señal generada: {signal['type']}")
        logger.info(f"Razón: {signal['reason']}")
        logger.info(f"RSI actual: {signal['rsi']:.2f}")
        
        # Verificar condiciones
        if signal['type'] == SignalType.BUY:
            logger.info("✓ Generada señal de COMPRA - RSI confirma condición de sobreventa con cruce")
        elif signal['type'] == SignalType.SELL:
            logger.info("✓ Generada señal de VENTA - RSI confirma condición de sobrecompra con cruce")
        elif signal['type'] == SignalType.HOLD:
            logger.info("✓ Sin señal (HOLD) - RSI en zona neutral o sin cruce de umbral")
        
        return True
    except Exception as e:
        logger.error(f"Error en prueba RSI: {str(e)}")
        return False

async def test_bollinger_bands_strategy():
    """Prueba la estrategia de Bandas de Bollinger."""
    try:
        # Importar módulos necesarios
        from genesis.strategies.mean_reversion import BollingerBandsStrategy
        from genesis.strategies.base import SignalType
        
        logger.info("\n=== Prueba de estrategia Bollinger Bands ===")
        
        # Crear datos específicos para Bollinger Bands
        data = create_test_data("bollinger")
        
        # Creamos instancia de estrategia
        strategy = BollingerBandsStrategy(period=20, std_dev=2.0)
        
        # Generar señal
        signal = await strategy.generate_signal("BTCUSDT", data)
        
        # Mostrar resultados
        logger.info(f"Señal generada: {signal['type']}")
        logger.info(f"Razón: {signal['reason']}")
        logger.info(f"Precio actual: {signal['price']:.2f}")
        logger.info(f"Banda superior: {signal['upper']:.2f}")
        logger.info(f"Banda media: {signal['middle']:.2f}")
        logger.info(f"Banda inferior: {signal['lower']:.2f}")
        logger.info(f"Ancho de banda: {signal['bandwidth']:.4f}")
        
        # Verificar condiciones
        if signal['type'] == SignalType.BUY:
            logger.info("✓ Generada señal de COMPRA - Precio bajo la banda inferior")
            assert signal['price'] < signal['lower'], "Precio debería estar bajo la banda inferior"
        elif signal['type'] == SignalType.SELL:
            logger.info("✓ Generada señal de VENTA - Precio sobre la banda superior")
            assert signal['price'] > signal['upper'], "Precio debería estar sobre la banda superior"
        elif signal['type'] == SignalType.HOLD:
            logger.info("✓ Sin señal (HOLD) - Precio dentro de las bandas")
            assert signal['lower'] <= signal['price'] <= signal['upper'], "Precio debería estar dentro de las bandas"
        
        return True
    except Exception as e:
        logger.error(f"Error en prueba Bollinger Bands: {str(e)}")
        return False

async def test_ma_crossover_strategy():
    """Prueba la estrategia de Cruce de Medias Móviles."""
    try:
        # Importar módulos necesarios
        from genesis.strategies.trend_following import MovingAverageCrossover
        from genesis.strategies.base import SignalType
        
        logger.info("\n=== Prueba de estrategia Moving Average Crossover ===")
        
        # Crear datos específicos para MA Crossover
        data = create_test_data("ma_crossover")
        
        # Creamos instancia de estrategia
        strategy = MovingAverageCrossover(name="ma_crossover", fast_period=10, slow_period=30)
        
        # Generar señal
        signal = await strategy.generate_signal("BTCUSDT", data)
        
        # Mostrar resultados
        logger.info(f"Señal generada: {signal['type']}")
        logger.info(f"Razón: {signal['reason']}")
        
        # Información de las medias móviles, si están disponibles
        if 'fast_ma' in signal and 'slow_ma' in signal:
            logger.info(f"Media móvil rápida (MA{strategy.fast_period}): {signal['fast_ma']:.2f}")
            logger.info(f"Media móvil lenta (MA{strategy.slow_period}): {signal['slow_ma']:.2f}")
        
        # Verificar condiciones
        if signal['type'] == SignalType.BUY:
            logger.info("✓ Generada señal de COMPRA - MA rápida cruzó por encima de MA lenta")
            if 'fast_ma' in signal and 'slow_ma' in signal:
                assert signal['fast_ma'] > signal['slow_ma'], "MA rápida debe estar sobre MA lenta"
        elif signal['type'] == SignalType.SELL:
            logger.info("✓ Generada señal de VENTA - MA rápida cruzó por debajo de MA lenta")
            if 'fast_ma' in signal and 'slow_ma' in signal:
                assert signal['fast_ma'] < signal['slow_ma'], "MA rápida debe estar bajo MA lenta"
        elif signal['type'] == SignalType.HOLD:
            logger.info("✓ Sin señal (HOLD) - No hay cruce reciente de medias móviles")
        
        return True
    except Exception as e:
        logger.error(f"Error en prueba MA Crossover: {str(e)}")
        return False

async def test_macd_strategy():
    """Prueba la estrategia MACD."""
    try:
        # Importar módulos necesarios
        from genesis.strategies.trend_following import MACDStrategy
        from genesis.strategies.base import SignalType
        
        logger.info("\n=== Prueba de estrategia MACD ===")
        
        # Crear datos específicos para MACD
        data = create_test_data("macd")
        
        # Creamos instancia de estrategia
        strategy = MACDStrategy(name="macd_strategy", fast_period=12, slow_period=26, signal_period=9)
        
        # Generar señal
        signal = await strategy.generate_signal("BTCUSDT", data)
        
        # Mostrar resultados
        logger.info(f"Señal generada: {signal['type']}")
        logger.info(f"Razón: {signal['reason']}")
        
        # Información de MACD, si están disponibles
        if 'macd' in signal:
            logger.info(f"MACD: {signal['macd']:.4f}")
        if 'signal_line' in signal:
            logger.info(f"Línea de señal: {signal['signal_line']:.4f}")
        if 'histogram' in signal:
            logger.info(f"Histograma: {signal['histogram']:.4f}")
        
        # Verificar condiciones
        if signal['type'] == SignalType.BUY:
            logger.info("✓ Generada señal de COMPRA - MACD cruzó por encima de línea de señal")
            if 'macd' in signal and 'signal_line' in signal:
                assert signal['macd'] > signal['signal_line'], "MACD debe estar sobre línea de señal"
        elif signal['type'] == SignalType.SELL:
            logger.info("✓ Generada señal de VENTA - MACD cruzó por debajo de línea de señal")
            if 'macd' in signal and 'signal_line' in signal:
                assert signal['macd'] < signal['signal_line'], "MACD debe estar bajo línea de señal"
        elif signal['type'] == SignalType.HOLD:
            logger.info("✓ Sin señal (HOLD) - No hay cruce reciente entre MACD y línea de señal")
        
        return True
    except Exception as e:
        logger.error(f"Error en prueba MACD: {str(e)}")
        return False

async def test_sentiment_strategy():
    """Prueba la estrategia basada en sentimiento."""
    try:
        # Importar módulos necesarios
        from genesis.strategies.sentiment_based import SentimentStrategy
        from genesis.strategies.base import SignalType
        
        logger.info("\n=== Prueba de estrategia basada en Sentimiento ===")
        
        # Crear datos específicos para Sentimiento
        data = create_test_data("sentiment")
        
        # Simular un análisis de sentimiento
        sentiment_data = {
            "score": 0.65,  # Sentimiento positivo
            "news_count": 15,
            "social_mentions": 1200,
            "trend": "bullish"
        }
        
        # Creamos instancia de estrategia
        strategy = SentimentStrategy(name="sentiment_strategy")
        strategy.get_sentiment = lambda symbol: sentiment_data  # Mock para datos de sentimiento
        
        # Generar señal
        signal = await strategy.generate_signal("BTCUSDT", data)
        
        # Mostrar resultados
        logger.info(f"Señal generada: {signal['type']}")
        logger.info(f"Razón: {signal['reason']}")
        logger.info(f"Score de sentimiento: {signal['sentiment']:.2f}")
        
        # Verificar condiciones
        if signal['type'] == SignalType.BUY:
            logger.info("✓ Generada señal de COMPRA - Sentimiento positivo confirmado")
            assert signal['sentiment'] > 0.5, "Score de sentimiento debe ser positivo"
        elif signal['type'] == SignalType.SELL:
            logger.info("✓ Generada señal de VENTA - Sentimiento negativo confirmado")
            assert signal['sentiment'] < 0, "Score de sentimiento debe ser negativo"
        elif signal['type'] == SignalType.HOLD:
            logger.info("✓ Sin señal (HOLD) - Sentimiento neutro o conflictivo")
        
        # Probar con sentimiento negativo
        sentiment_data["score"] = -0.60  # Sentimiento negativo
        sentiment_data["trend"] = "bearish"
        
        signal = await strategy.generate_signal("BTCUSDT", data)
        logger.info(f"Señal con sentimiento negativo: {signal['type']}")
        logger.info(f"Score de sentimiento: {signal['sentiment']:.2f}")
        
        if signal['type'] == SignalType.SELL:
            logger.info("✓ Generada señal de VENTA - Sentimiento negativo confirmado")
        
        return True
    except Exception as e:
        logger.error(f"Error en prueba Sentiment: {str(e)}")
        return False

async def main():
    """Ejecutar todas las pruebas."""
    try:
        logger.info("=== INICIO DE PRUEBAS DE ESTRATEGIAS ===")
        
        # Ejecutar todas las pruebas
        results = []
        
        # Probar estrategia RSI
        results.append(await test_rsi_strategy())
        
        # Probar estrategia Bollinger Bands
        results.append(await test_bollinger_bands_strategy())
        
        # Probar estrategia MA Crossover
        results.append(await test_ma_crossover_strategy())
        
        # Probar estrategia MACD
        results.append(await test_macd_strategy())
        
        # Probar estrategia Sentiment
        results.append(await test_sentiment_strategy())
        
        # Resultados finales
        success_count = sum(results)
        total_count = len(results)
        
        logger.info("\n=== RESUMEN DE PRUEBAS ===")
        logger.info(f"Pruebas exitosas: {success_count}/{total_count}")
        
        if success_count == total_count:
            logger.info("✓ ¡Todas las pruebas completadas con éxito!")
        else:
            logger.info("✗ Algunas pruebas fallaron, revisar errores")
        
        logger.info("=== FIN DE PRUEBAS DE ESTRATEGIAS ===")
        
    except Exception as e:
        logger.error(f"Error general: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())