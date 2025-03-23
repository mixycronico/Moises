"""
Script para probar la inicialización de la estrategia ReinforcementEnsemble.
"""
import asyncio
import logging
import json

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_strategy')

async def test_strategy():
    """Probar inicialización y funcionamiento básico de la estrategia."""
    try:
        # Importar la función para obtener estrategias
        from genesis.strategies.advanced import get_advanced_strategy
        
        # Configuración básica
        config = {
            'name': 'Test Strategy',
            'symbols': ['BTC/USDT', 'ETH/USDT'],
            'use_deepseek': True,
            'deepseek_intelligence_factor': 1.0
        }
        
        # Crear instancia de la estrategia
        logger.info("Creando instancia de estrategia...")
        strategy = get_advanced_strategy('reinforcement_ensemble', config)
        
        if not strategy:
            logger.error("Error: No se pudo crear la estrategia")
            return
        
        logger.info(f"Estrategia creada: {strategy.name}")
        
        # Inicializar la estrategia
        logger.info("Inicializando estrategia...")
        success = await strategy.initialize()
        logger.info(f"Inicialización exitosa: {success}")
        
        if success:
            # Probar generación de señal con datos básicos
            logger.info("Probando generación de señal...")
            
            # Datos simulados de mercado
            ohlcv_data = {
                'open': [100.0] * 50,
                'high': [105.0] * 50,
                'low': [95.0] * 50,
                'close': [102.0] * 50,
                'volume': [1000.0] * 50
            }
            
            market_data = {
                'ohlcv': ohlcv_data,
                'news': []
            }
            
            # Generar señal
            signal = await strategy.generate_signal('BTC/USDT', market_data)
            
            logger.info(f"Señal generada: {signal.get('signal', 'NONE')}")
            logger.info(f"Confianza: {signal.get('confidence', 0.0)}")
            logger.info(f"Razón: {signal.get('reason', 'N/A')}")
            
    except Exception as e:
        logger.error(f"Error durante la prueba: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Ejecutar prueba
    asyncio.run(test_strategy())