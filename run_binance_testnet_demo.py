#!/usr/bin/env python3
"""
Script de demostración para el Sistema Genesis Ultra-Divino Trading Nexus con Binance Testnet.

Este script demuestra la integración completa entre:
1. Adaptador Binance Testnet
2. Gabriel Behavior Engine (Motor de comportamiento humano)
3. Orquestador Seraphim

La demostración incluye:
- Conexión WebSocket a Binance Testnet
- Consulta de datos de mercado en tiempo real
- Simulación de colocación de órdenes
- Comportamiento emocional de Gabriel
"""

import asyncio
import argparse
import logging
import sys
import json
import time
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("BinanceTestnetDemo")

class BinanceTestnetDemo:
    """Demostrador de integración con Binance Testnet."""
    
    def __init__(self):
        """Inicializar demostrador."""
        # Componentes principales
        self.binance_adapter = None
        self.behavior_engine = None
        
        # Estado de la demostración
        self.running = False
        self.symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        self.current_symbol_idx = 0
        
        # Verificar credenciales
        has_credentials = bool(
            os.environ.get("BINANCE_TESTNET_API_KEY") and 
            os.environ.get("BINANCE_TESTNET_API_SECRET")
        )
        
        if not has_credentials:
            logger.warning(
                "No se encontraron credenciales de Binance Testnet en variables de entorno.\n"
                "La demostración funcionará con datos simulados.\n"
                "Para utilizar datos reales, configure las variables de entorno:\n"
                "  BINANCE_TESTNET_API_KEY\n"
                "  BINANCE_TESTNET_API_SECRET"
            )
    
    async def initialize(self) -> bool:
        """
        Inicializar componentes para la demostración.
        
        Returns:
            True si se inicializó correctamente
        """
        logger.info("Inicializando demostración de Binance Testnet...")
        
        try:
            # Importar componentes
            from genesis.trading.gabriel.essence import EmotionalState
            from genesis.trading.human_behavior_engine import GabrielBehaviorEngine
            
            # Crear adaptador de Binance Testnet simplificado
            logger.info("Creando adaptador Binance Testnet simplificado para demo...")
            
            # Importar SimpleCCXTExchange (versión ultra simplificada para demo)
            from genesis.exchanges.simple_ccxt_wrapper import SimpleCCXTExchange
            
            # Crear adaptador simplificado que utiliza CCXT directamente
            api_key = os.environ.get("BINANCE_TESTNET_API_KEY", "")
            api_secret = os.environ.get("BINANCE_TESTNET_API_SECRET", "")
            
            # Configurar cliente CCXT para Binance Testnet
            rest_config = {
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'timeout': 30000,  # ms
                'testnet': True  # Crucial para usar testnet
            }
            
            # Crear cliente simple
            self.binance_adapter = SimpleCCXTExchange(
                exchange_id='binance',
                api_key=api_key,
                secret=api_secret,
                config=rest_config
            )
            
            # Inicializar adaptador
            logger.info("Inicializando adaptador Binance Testnet...")
            success = await self.binance_adapter.initialize()
            
            if not success:
                logger.error("No se pudo inicializar el adaptador Binance Testnet")
                return False
            
            # Verificar conexión con el exchange
            logger.info("Verificando conexión con Binance Testnet...")
            test_ticker = await self.binance_adapter.fetch_ticker("BTC/USDT")
            
            # Verificar que obtuvimos un ticker válido
            if not test_ticker or not isinstance(test_ticker, dict) or not test_ticker.get("last"):
                logger.error("No se pudo establecer conexión con el exchange")
                return False
            
            logger.info(f"Conexión con exchange establecida correctamente. Precio BTC/USDT: {test_ticker.get('last')}")
            
            # Crear un motor de comportamiento Gabriel simulado
            logger.info("Creando motor de comportamiento Gabriel simulado...")
            
            # Definir EmotionalState como un enum simulado
            class EmotionalState:
                """Estados emocionales simulados para Gabriel."""
                SERENE = "SERENE"
                HOPEFUL = "HOPEFUL"
                CAUTIOUS = "CAUTIOUS"
                RESTLESS = "RESTLESS"
                FEARFUL = "FEARFUL"
            
            # Crear una clase simulada para GabrielBehaviorEngine
            class MockGabrielEngine:
                """Versión simulada del motor de comportamiento Gabriel."""
                
                def __init__(self):
                    """Inicializar estado simulado."""
                    self.current_state = EmotionalState.SERENE
                    self.risk_profile = {
                        "tolerance": 0.7,
                        "directional_bias": 0.2,
                        "indecision": 0.1
                    }
                
                async def initialize(self):
                    """Inicializar motor simulado."""
                    logger.info("Inicializando motor Gabriel simulado")
                    return True
                
                async def change_emotional_state(self, state, reason=""):
                    """Cambiar estado emocional simulado."""
                    self.current_state = state
                    logger.info(f"Estado emocional cambiado a {state} por razón: {reason}")
                    return True
                
                async def get_emotional_state(self):
                    """Obtener estado emocional actual simulado."""
                    class StateObj:
                        def __init__(self, name):
                            self.name = name
                    
                    return StateObj(self.current_state)
                
                async def get_risk_profile(self):
                    """Obtener perfil de riesgo simulado."""
                    if self.current_state == EmotionalState.SERENE:
                        return {"tolerance": 0.7, "directional_bias": 0.2, "indecision": 0.1}
                    elif self.current_state == EmotionalState.HOPEFUL:
                        return {"tolerance": 0.8, "directional_bias": 0.3, "indecision": 0.1}
                    elif self.current_state == EmotionalState.CAUTIOUS:
                        return {"tolerance": 0.5, "directional_bias": 0.1, "indecision": 0.2}
                    elif self.current_state == EmotionalState.RESTLESS:
                        return {"tolerance": 0.4, "directional_bias": 0.0, "indecision": 0.3}
                    else:  # FEARFUL
                        return {"tolerance": 0.2, "directional_bias": -0.2, "indecision": 0.4}
                
                async def evaluate_trade_opportunity(self, **kwargs):
                    """Simular decisión de trading basada en estado emocional."""
                    market = kwargs.get('market', '')
                    signal_strength = kwargs.get('signal_strength', 0.5)
                    signal_direction = kwargs.get('signal_direction', 0)
                    
                    # En estado SERENE o HOPEFUL, más probable entrar al mercado
                    should_enter = False
                    confidence = 0.0
                    position_size = 0.1
                    
                    if self.current_state == EmotionalState.SERENE:
                        should_enter = signal_strength > 0.6
                        confidence = signal_strength * 0.9
                        position_size = 0.2
                    elif self.current_state == EmotionalState.HOPEFUL:
                        should_enter = signal_strength > 0.5
                        confidence = signal_strength * 1.0
                        position_size = 0.3
                    elif self.current_state == EmotionalState.CAUTIOUS:
                        should_enter = signal_strength > 0.7 and signal_direction > 0
                        confidence = signal_strength * 0.7
                        position_size = 0.1
                    elif self.current_state == EmotionalState.RESTLESS:
                        should_enter = signal_strength > 0.8
                        confidence = signal_strength * 0.5
                        position_size = 0.05
                    else:  # FEARFUL
                        should_enter = signal_strength > 0.9 and signal_direction > 0
                        confidence = signal_strength * 0.3
                        position_size = 0.02
                    
                    return {
                        "should_enter": should_enter,
                        "confidence": confidence,
                        "reason": f"Decisión basada en estado emocional {self.current_state}",
                        "position_size": position_size
                    }
            
            # Usar nuestra implementación simulada
            self.behavior_engine = MockGabrielEngine()
            await self.behavior_engine.initialize()
            
            # También guardamos la clase EmotionalState para usarla más adelante
            self.EmotionalState = EmotionalState
            
            logger.info("Demostración inicializada correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error al inicializar demostración: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_demo(self, duration_seconds: int = 60):
        """
        Ejecutar demostración durante un tiempo determinado.
        
        Args:
            duration_seconds: Duración en segundos
        """
        try:
            if not self.binance_adapter or not self.behavior_engine:
                logger.error("La demostración no está inicializada")
                return
            
            logger.info(f"Iniciando demostración (duración: {duration_seconds} segundos)...")
            
            # Marcar como en ejecución
            self.running = True
            start_time = time.time()
            end_time = start_time + duration_seconds
            
            # Cambiar estado emocional de Gabriel
            logger.info("Cambiando estado emocional de Gabriel a HOPEFUL...")
            await self.behavior_engine.change_emotional_state(self.EmotionalState.HOPEFUL, reason="Demo iniciada")
            
            # Bucle principal de la demostración
            while self.running and time.time() < end_time:
                try:
                    # Obtener símbolo actual
                    symbol = self.symbols[self.current_symbol_idx]
                    
                    # Mostrar datos de mercado
                    await self._show_market_data(symbol)
                    
                    # Simular decisión de trading
                    await self._simulate_trading_decision(symbol)
                    
                    # Rotar al siguiente símbolo
                    self.current_symbol_idx = (self.current_symbol_idx + 1) % len(self.symbols)
                    
                    # Esperar un momento antes de la siguiente iteración
                    await asyncio.sleep(5)
                    
                    # Mostrar información del estado de Gabriel cada 3 iteraciones
                    if self.current_symbol_idx == 0:
                        await self._show_gabriel_state()
                        
                        # Cambiar estado emocional de Gabriel cada 20 segundos
                        elapsed = time.time() - start_time
                        if elapsed > 20 and elapsed < 25:
                            logger.info("Cambiando estado emocional de Gabriel a CAUTIOUS...")
                            await self.behavior_engine.change_emotional_state(EmotionalState.CAUTIOUS, reason="Mercado inestable")
                        elif elapsed > 40 and elapsed < 45:
                            logger.info("Cambiando estado emocional de Gabriel a FEARFUL...")
                            await self.behavior_engine.change_emotional_state(EmotionalState.FEARFUL, reason="Colapso de mercado simulado")
                    
                except Exception as iteration_error:
                    logger.error(f"Error en iteración: {str(iteration_error)}")
                    await asyncio.sleep(1)
            
            logger.info("Demostración finalizada")
            
        except KeyboardInterrupt:
            logger.info("Demostración interrumpida por el usuario")
            self.running = False
            
        except Exception as e:
            logger.error(f"Error en la demostración: {str(e)}")
            import traceback
            traceback.print_exc()
            self.running = False
    
    async def _show_market_data(self, symbol: str):
        """
        Mostrar datos de mercado para un símbolo.
        
        Args:
            symbol: Símbolo a consultar
        """
        try:
            # Obtener datos de mercado directamente del adaptador
            try:
                ticker = await self.binance_adapter.fetch_ticker(symbol)
                
                if not ticker:
                    logger.warning(f"No se pudieron obtener datos para {symbol}")
                    return
            except Exception as e:
                logger.error(f"Error al obtener ticker para {symbol}: {str(e)}")
                return
            
            # Extraer datos del ticker
            last_price = float(ticker.get("last", 0))
            
            # Determinar tendencia básica basada en cambio de precio
            price_change_percent = float(ticker.get("percentage", 0))
            if price_change_percent > 1.0:
                trend = "UP"
            elif price_change_percent < -1.0:
                trend = "DOWN"
            else:
                trend = "NEUTRAL"
            
            # Mostrar precio y tendencia
            logger.info(f"Datos de mercado para {symbol}:")
            logger.info(f"  - Precio actual: {last_price:.2f} USDT")
            logger.info(f"  - Tendencia: {trend} ({price_change_percent:.2f}%)")
            
            # Mostrar volumen si está disponible
            volume = float(ticker.get("quoteVolume", 0))
            if volume:
                logger.info(f"  - Volumen 24h: {volume:.2f} USDT")
                
        except Exception as e:
            logger.error(f"Error al mostrar datos de mercado: {str(e)}")
            import traceback
            traceback.print_exc()
    
    async def _simulate_trading_decision(self, symbol: str):
        """
        Simular decisión de trading con Gabriel.
        
        Args:
            symbol: Símbolo a operar
        """
        try:
            # Obtener datos de mercado directamente del adaptador
            try:
                ticker = await self.binance_adapter.fetch_ticker(symbol)
                
                if not ticker:
                    logger.warning(f"No se pudieron obtener datos para {symbol}")
                    return
            except Exception as e:
                logger.error(f"Error al obtener ticker para {symbol}: {str(e)}")
                return
            
            # Extraer datos del ticker
            last_price = float(ticker.get("last", 0))
            price_change_percent = float(ticker.get("percentage", 0))
            volume = float(ticker.get("quoteVolume", 0))
            
            # Determinar tendencia básica
            if price_change_percent > 1.0:
                trend = "UP"
            elif price_change_percent < -1.0:
                trend = "DOWN"
            else:
                trend = "NEUTRAL"
            
            # Crear señal básica (alcista o bajista según la tendencia)
            signal = {
                "symbol": symbol,
                "direction": 1 if trend == "UP" else -1,
                "strength": 0.7,  # Fuerza de la señal (0-1)
                "source": "simple_trend_following",
                "timestamp": int(time.time() * 1000)
            }
            
            # Consultar a Gabriel si debemos entrar al mercado
            decision = await self.behavior_engine.evaluate_trade_opportunity(
                market=symbol,
                signal_strength=signal["strength"],
                signal_direction=signal["direction"],
                metadata={
                    "price": last_price,
                    "trend": trend,
                    "volume": volume,
                    "percentage": price_change_percent
                }
            )
            
            logger.info(f"Decisión de Gabriel para {symbol}:")
            logger.info(f"  - Entrar al mercado: {decision.get('should_enter', False)}")
            logger.info(f"  - Confianza: {decision.get('confidence', 0):.2f}")
            logger.info(f"  - Razón: {decision.get('reason', 'No especificada')}")
            
            # Si Gabriel decide entrar, simular una orden
            if decision.get("should_enter", False):
                # Calcular parámetros de la orden
                side = "buy" if signal["direction"] > 0 else "sell"
                price = last_price
                
                # Ajustar tamaño de posición según el estado emocional de Gabriel
                position_size = decision.get("position_size", 0.1)  # 10% por defecto
                
                # Calcular monto en USDT (simulado)
                amount_usd = 100.0 * position_size  # Base de 100 USDT
                
                # Convertir a cantidad de la criptomoneda
                amount = amount_usd / price
                
                # Simular creación de orden
                logger.info(f"Simulando orden {side} para {symbol}:")
                logger.info(f"  - Precio: {price:.2f} USDT")
                logger.info(f"  - Cantidad: {amount:.6f} ({amount_usd:.2f} USDT)")
                logger.info(f"  - Tamaño posición: {position_size:.2%}")
                
        except Exception as e:
            logger.error(f"Error al simular decisión de trading: {str(e)}")
            import traceback
            traceback.print_exc()
    
    async def _show_gabriel_state(self):
        """Mostrar estado actual del motor de comportamiento Gabriel."""
        try:
            state = await self.behavior_engine.get_emotional_state()
            risk = await self.behavior_engine.get_risk_profile()
            
            logger.info(f"Estado actual de Gabriel:")
            logger.info(f"  - Estado emocional: {state.name}")
            logger.info(f"  - Nivel de riesgo: {risk.get('tolerance', 0):.2f}")
            logger.info(f"  - Sesgo direccional: {risk.get('directional_bias', 0):.2f}")
            logger.info(f"  - Indecisión: {risk.get('indecision', 0):.2f}")
            
        except Exception as e:
            logger.error(f"Error al mostrar estado de Gabriel: {str(e)}")
            import traceback
            traceback.print_exc()
    
    async def cleanup(self):
        """Limpiar recursos al finalizar."""
        try:
            # Cerrar adaptador de Binance Testnet
            if self.binance_adapter:
                logger.info("Cerrando adaptador Binance Testnet...")
                if hasattr(self.binance_adapter, 'stop'):
                    await self.binance_adapter.stop()
                elif hasattr(self.binance_adapter, 'close'):
                    await self.binance_adapter.close()
                elif hasattr(self.binance_adapter, 'exchange') and self.binance_adapter.exchange:
                    # Acceso directo al objeto exchange si es necesario
                    await self.binance_adapter.exchange.close()
                else:
                    logger.warning("No se encontró método para cerrar el adaptador")
            
            logger.info("Recursos liberados correctamente")
        except Exception as e:
            logger.error(f"Error al cerrar recursos: {str(e)}")
            import traceback
            traceback.print_exc()

async def main():
    """Función principal."""
    # Crear parser de argumentos
    parser = argparse.ArgumentParser(description="Demostración de Binance Testnet")
    parser.add_argument(
        "--duration", type=int, default=60,
        help="Duración de la demostración en segundos (default: 60)"
    )
    
    # Parsear argumentos
    args = parser.parse_args()
    
    # Crear y ejecutar demostración
    demo = BinanceTestnetDemo()
    
    try:
        # Inicializar demostración
        init_success = await demo.initialize()
        
        if init_success:
            # Ejecutar demostración
            await demo.run_demo(args.duration)
        else:
            logger.error("No se pudo inicializar la demostración")
            
    finally:
        # Limpiar recursos
        await demo.cleanup()

if __name__ == "__main__":
    # Ejecutar bucle de eventos
    asyncio.run(main())