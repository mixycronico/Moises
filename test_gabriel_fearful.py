"""
Prueba de integración del estado FEARFUL de Gabriel

Este script prueba la implementación del estado FEARFUL al 100% 
en el motor de comportamiento humano Gabriel.

Verifica:
1. Cambio de estado a FEARFUL
2. Rechazo de compras con confianza < 100%
3. Reducción de tamaño de operaciones (50% compras, 120% ventas)
4. Persistencia en estado de miedo
"""

import asyncio
import logging
from datetime import datetime
import sys

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("test_gabriel_fearful")

async def main():
    """Prueba la implementación del estado FEARFUL."""
    try:
        # Importaciones dentro de la función para evitar problemas de inicialización
        from genesis.trading.gabriel_adapter import GabrielAdapter
        from genesis.trading.human_behavior_engine import EmotionalState

        # Crear instancia del adaptador Gabriel
        logger.info("Creando instancia de GabrielAdapter...")
        adapter = GabrielAdapter()
        
        # Inicializar el adaptador
        await adapter.initialize()
        
        # Verificar estado inicial
        estado_inicial = adapter.emotional_state
        logger.info(f"Estado emocional inicial: {estado_inicial.name}")
        
        # Cambiar a estado FEARFUL
        logger.info("Cambiando a estado FEARFUL...")
        adapter.set_fearful_state("prueba_fearful_100pct")
        
        # Verificar que se haya cambiado correctamente
        estado_fearful = adapter.emotional_state
        logger.info(f"Estado emocional después de set_fearful_state: {estado_fearful.name}")
        
        if estado_fearful != EmotionalState.FEARFUL:
            logger.error(f"ERROR: El estado no cambió a FEARFUL, sigue en {estado_fearful.name}")
            return
            
        # Prueba 1: Verificar rechazo de compras
        logger.info("\n=== PRUEBA 1: Rechazo de compras con confianza < 100% ===")
        # Niveles de confianza a probar
        niveles = [0.5, 0.8, 0.95, 0.99, 1.0]
        
        for nivel in niveles:
            # Probar validación directa
            trade_params = {
                "symbol": "BTC/USDT",
                "side": "buy",
                "price": 50000.0,
                "amount": 0.1,
                "confidence": nivel
            }
            
            # Validar directamente la operación
            valida, razon = await adapter.validate_trade(trade_params)
            
            logger.info(f"Confianza {nivel:.2f}: {'APROBADA' if valida else 'RECHAZADA'} - {razon or 'aprobada'}")
            
        # Prueba 2: Verificar ajuste de tamaño
        logger.info("\n=== PRUEBA 2: Ajuste de tamaño de operaciones ===")
        
        # Tamaño base para pruebas
        tamano_base = 100.0
        
        # Probar compra
        tamano_compra = await adapter.adjust_order_size(
            base_size=tamano_base,
            confidence=0.8,
            is_buy=True
        )
        
        # Probar venta
        tamano_venta = await adapter.adjust_order_size(
            base_size=tamano_base,
            confidence=0.8,
            is_buy=False
        )
        
        logger.info(f"Tamaño original: {tamano_base}")
        logger.info(f"Tamaño ajustado para compra: {tamano_compra:.2f} ({tamano_compra/tamano_base*100:.0f}% del original)")
        logger.info(f"Tamaño ajustado para venta: {tamano_venta:.2f} ({tamano_venta/tamano_base*100:.0f}% del original)")
        
        # Verificar persistencia del estado
        logger.info("\n=== PRUEBA 3: Persistencia del estado FEARFUL ===")
        # Intentar cambiar mediante eventos normales
        await adapter.gabriel.hear("opportunity", 0.7)
        
        # Verificar si cambió
        estado_despues = adapter.emotional_state
        logger.info(f"Estado después de evento 'oportunidad': {estado_despues.name}")
        
        logger.info("Pruebas completadas.")
        
    except Exception as e:
        logger.error(f"Error en prueba: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ejecutar la prueba asíncrona
    asyncio.run(main())