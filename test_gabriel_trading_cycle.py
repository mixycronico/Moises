"""
Prueba de ciclo de trading completo con Gabriel en estado FEARFUL

Este script ejecuta un ciclo de trading completo simulado para
validar el comportamiento del estado FEARFUL en un escenario realista.
"""

import asyncio
import logging
from datetime import datetime
import sys
import random

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("test_gabriel_trading_cycle")

class SimulatedExchange:
    """Simulador básico de exchange para pruebas."""
    
    def __init__(self):
        self.price = 50000.0
        self.open_orders = {}
        self.order_counter = 1
        
    async def get_price(self, symbol):
        """Obtener precio actual simulado."""
        # Variar ligeramente el precio
        variation = random.uniform(-0.005, 0.005)
        self.price *= (1 + variation)
        return self.price
        
    async def place_order(self, symbol, side, price, amount):
        """Colocar orden simulada."""
        order_id = f"order_{self.order_counter}"
        self.order_counter += 1
        
        self.open_orders[order_id] = {
            "symbol": symbol,
            "side": side,
            "price": price,
            "amount": amount,
            "status": "open",
            "created_at": datetime.now()
        }
        
        return {
            "id": order_id,
            "status": "open"
        }
        
    async def cancel_order(self, order_id):
        """Cancelar orden simulada."""
        if order_id in self.open_orders:
            self.open_orders[order_id]["status"] = "canceled"
            return True
        return False
    
    async def get_order(self, order_id):
        """Obtener información de orden simulada."""
        if order_id in self.open_orders:
            return self.open_orders[order_id]
        return None

async def run_trading_cycle():
    """Ejecutar un ciclo de trading simulado con Gabriel."""
    try:
        # Importaciones dentro de la función para evitar problemas de inicialización
        from genesis.trading.gabriel_adapter import GabrielAdapter
        from genesis.trading.human_behavior_engine import EmotionalState
        
        # Crear simulador de exchange
        exchange = SimulatedExchange()
        
        # Crear instancia del adaptador Gabriel
        logger.info("Creando instancia de GabrielAdapter...")
        adapter = GabrielAdapter()
        
        # Inicializar el adaptador
        await adapter.initialize()
        
        # Verificar estado inicial
        logger.info(f"Estado emocional inicial: {adapter.emotional_state.name}")
        
        # Ejecutar ciclo de trading normal (sin miedo)
        logger.info("\n=== CICLO DE TRADING NORMAL ===")
        await execute_trading_cycle(adapter, exchange, "BTC/USDT", is_fearful=False)
        
        # Cambiar a estado FEARFUL
        logger.info("\n=== ACTIVANDO ESTADO FEARFUL ===")
        adapter.set_fearful_state("ciclo_trading_fearful")
        logger.info(f"Estado emocional: {adapter.emotional_state.name}")
        
        # Ejecutar ciclo de trading en estado fearful
        logger.info("\n=== CICLO DE TRADING EN ESTADO FEARFUL ===")
        await execute_trading_cycle(adapter, exchange, "BTC/USDT", is_fearful=True)
        
        logger.info("Prueba completada exitosamente.")
        
    except Exception as e:
        logger.error(f"Error en prueba: {str(e)}")
        import traceback
        traceback.print_exc()

async def execute_trading_cycle(adapter, exchange, symbol, is_fearful=False):
    """
    Ejecutar un ciclo completo de trading utilizando Gabriel.
    
    Args:
        adapter: Instancia de GabrielAdapter
        exchange: Simulador de exchange
        symbol: Símbolo a operar
        is_fearful: Si estamos en estado de miedo
    """
    try:
        # 1. Obtener precio actual
        price = await exchange.get_price(symbol)
        logger.info(f"Precio actual de {symbol}: ${price:.2f}")
        
        # 2. Analizar señal de entrada
        signal_strength = 0.85  # Señal normal
        
        if is_fearful:
            # Probar también con señal máxima
            signal_strengths = [0.85, 1.0]
            
            for strength in signal_strengths:
                # Evaluar si entrar
                should_enter, reason = await adapter.should_enter_trade(
                    signal_strength=strength,
                    market_context={"market_sentiment": "neutral", "volatility": 0.5}
                )
                
                logger.info(f"¿Entrar con señal {strength:.2f}? {should_enter} - {reason}")
                
                if should_enter:
                    # Si aprueba, probar el resto del ciclo con esta señal
                    signal_strength = strength
                    break
        else:
            # Evaluar si entrar (modo normal)
            should_enter, reason = await adapter.should_enter_trade(
                signal_strength=signal_strength,
                market_context={"market_sentiment": "neutral", "volatility": 0.5}
            )
            
            logger.info(f"¿Entrar? {should_enter} - {reason}")
        
        # Si no debemos entrar, terminar ciclo
        if not should_enter:
            logger.info("Ciclo de trading finalizado (sin entrada)")
            return
            
        # 3. Calcular tamaño de operación
        base_size = 0.1  # BTC
        
        # Ajustar tamaño según estado emocional
        adjusted_size = await adapter.adjust_order_size(
            base_size=base_size,
            confidence=signal_strength,
            is_buy=True
        )
        
        logger.info(f"Tamaño base: {base_size} BTC")
        logger.info(f"Tamaño ajustado: {adjusted_size:.4f} BTC ({adjusted_size/base_size*100:.0f}% del base)")
        
        # 4. Colocar orden
        order = await exchange.place_order(
            symbol=symbol,
            side="buy",
            price=price,
            amount=adjusted_size
        )
        
        logger.info(f"Orden colocada: ID {order['id']}")
        
        # 5. Simular cambio de precio favorable
        price_change = 0.02  # 2% de ganancia
        new_price = price * (1 + price_change)
        logger.info(f"Nuevo precio simulado: ${new_price:.2f} ({price_change*100:.1f}% de cambio)")
        
        # 6. Decidir si salir
        should_exit, exit_reason = await adapter.should_exit_trade(
            profit_percent=price_change*100,
            time_in_trade_hours=0.5,
            price_momentum=0.001
        )
        
        logger.info(f"¿Salir? {should_exit} - {exit_reason}")
        
        # 7. Si decidimos salir, colocar orden de venta
        if should_exit:
            # Calcular tamaño para venta
            sell_size = await adapter.adjust_order_size(
                base_size=adjusted_size,  # Usamos el tamaño de compra como base
                confidence=0.9,
                is_buy=False  # Esto es una venta
            )
            
            logger.info(f"Tamaño de venta ajustado: {sell_size:.4f} BTC ({sell_size/adjusted_size*100:.0f}% del tamaño de compra)")
            
            # Colocar orden de venta
            sell_order = await exchange.place_order(
                symbol=symbol,
                side="sell",
                price=new_price,
                amount=sell_size
            )
            
            logger.info(f"Orden de venta colocada: ID {sell_order['id']}")
        
        logger.info("Ciclo de trading completado")
        
    except Exception as e:
        logger.error(f"Error en ciclo de trading: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ejecutar la prueba asíncrona
    asyncio.run(run_trading_cycle())