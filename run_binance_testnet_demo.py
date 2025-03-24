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
        self.orchestrator = None
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
            from genesis.exchanges.adapter_factory import ExchangeAdapterFactory, AdapterType
            from genesis.trading.human_behavior_engine import GabrielBehaviorEngine
            
            # Crear adaptador de Binance Testnet directamente
            logger.info("Creando adaptador Binance Testnet...")
            factory = ExchangeAdapterFactory()
            self.binance_adapter = await factory.create_adapter(
                adapter_type=AdapterType.BINANCE_TESTNET,
                exchange_id="binance"
            )
            
            # Inicializar adaptador
            logger.info("Inicializando adaptador Binance Testnet...")
            await self.binance_adapter.initialize()
            
            # Verificar conexión con el exchange
            logger.info("Verificando conexión con Binance Testnet...")
            exchange_state = await self.binance_adapter.get_state()
            
            if exchange_state.get("status") != "CONNECTED":
                logger.error("No se pudo establecer conexión con el exchange")
                return False
            
            logger.info("Conexión con exchange establecida correctamente")
            
            # Crear motor de comportamiento Gabriel directamente
            logger.info("Creando motor de comportamiento Gabriel...")
            self.behavior_engine = GabrielBehaviorEngine()
            await self.behavior_engine.initialize()
            
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
            await self.behavior_engine.set_emotional_state("OPTIMISTIC", "Demo iniciada")
            
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
                            await self.behavior_engine.set_emotional_state("CAUTIOUS", "Mercado inestable")
                        elif elapsed > 40 and elapsed < 45:
                            logger.info("Cambiando estado emocional de Gabriel a FEARFUL...")
                            await self.behavior_engine.set_emotional_state("FEARFUL", "Colapso de mercado simulado")
                    
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
            # Obtener datos de mercado a través del orquestador
            market_data = await self.orchestrator.get_market_data(symbol)
            
            if not market_data.get("success", False):
                logger.warning(f"No se pudieron obtener datos para {symbol}: {market_data.get('error', 'Unknown error')}")
                return
            
            # Mostrar precio y tendencia
            last_price = market_data.get("last_price", 0)
            trend = market_data.get("trend", "NEUTRAL")
            
            logger.info(f"Datos de mercado para {symbol}:")
            logger.info(f"  - Precio actual: {last_price:.2f} USDT")
            logger.info(f"  - Tendencia: {trend}")
            
            # Mostrar volumen si está disponible
            volume = market_data.get("volume", 0)
            if volume:
                logger.info(f"  - Volumen 24h: {volume:.2f}")
                
        except Exception as e:
            logger.error(f"Error al mostrar datos de mercado: {str(e)}")
    
    async def _simulate_trading_decision(self, symbol: str):
        """
        Simular decisión de trading con Gabriel.
        
        Args:
            symbol: Símbolo a operar
        """
        try:
            # Obtener datos de mercado
            market_data = await self.orchestrator.get_market_data(symbol)
            
            if not market_data.get("success", False):
                logger.warning(f"No se pudieron obtener datos para {symbol}")
                return
            
            # Crear señal básica (alcista o bajista según la tendencia)
            signal = {
                "symbol": symbol,
                "direction": 1 if market_data.get("trend") == "UP" else -1,
                "strength": 0.7,  # Fuerza de la señal (0-1)
                "source": "simple_trend_following",
                "timestamp": int(time.time() * 1000)
            }
            
            # Consultar a Gabriel si debemos entrar al mercado
            decision = await self.behavior_engine.should_enter(
                market=symbol,
                signal_strength=signal["strength"],
                signal_direction=signal["direction"],
                metadata={
                    "price": market_data.get("last_price", 0),
                    "trend": market_data.get("trend"),
                    "volume": market_data.get("volume", 0)
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
                price = market_data.get("last_price", 0)
                
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
    
    async def _show_gabriel_state(self):
        """Mostrar estado actual del motor de comportamiento Gabriel."""
        try:
            state = self.behavior_engine.get_state()
            
            logger.info(f"Estado actual de Gabriel:")
            logger.info(f"  - Estado emocional: {state.get('emotional_state', 'UNKNOWN')}")
            logger.info(f"  - Nivel de riesgo: {state.get('risk_tolerance', 0):.2f}")
            logger.info(f"  - Sesgo direccional: {state.get('directional_bias', 0):.2f}")
            logger.info(f"  - Indecisión: {state.get('indecision', 0):.2f}")
            
        except Exception as e:
            logger.error(f"Error al mostrar estado de Gabriel: {str(e)}")
    
    async def cleanup(self):
        """Limpiar recursos al finalizar."""
        if self.orchestrator:
            logger.info("Cerrando orquestador...")
            await self.orchestrator.shutdown()
            logger.info("Recursos liberados correctamente")

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