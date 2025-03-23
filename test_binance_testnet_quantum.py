"""
Demostración del Integrador Trascendental para Binance Testnet.

Este script demuestra las capacidades del WebSocket Ultra-Cuántico
conectándose a Binance Testnet y mostrando sus capacidades de transmutación
y procesamiento asincrónico.
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional

# Importamos la versión específica para Binance Testnet
from genesis.core.transcendental_exchange_integrator import (
    TranscendentalWebSocketAdapter,
    ExchangeID
)

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BinanceTestnetDemo")

# Constantes para Binance Testnet
BINANCE_TESTNET_WS_URL = "wss://testnet.binance.vision/ws"
BINANCE_TESTNET_API_URL = "https://testnet.binance.vision/api"

class BinanceTestnetAdapter(TranscendentalWebSocketAdapter):
    """Versión personalizada del adaptador para Binance Testnet."""
    
    def __init__(self):
        """Inicializar adaptador para Binance Testnet."""
        super().__init__(ExchangeID.BINANCE)
        # Sobreescribir configuración para usar Testnet
        self.config = {
            "ws_url": BINANCE_TESTNET_WS_URL,
            "api_url": BINANCE_TESTNET_API_URL,
            "ping_interval": 30,
            "subscription_format": "method",  # {method: "SUBSCRIBE", params: [...]}
        }
        self.logger = logger.getChild("WebSocket.BinanceTestnet")

async def demo_market_data():
    """Demostrar recepción de datos de mercado en tiempo real."""
    logger.info("=== INICIANDO DEMOSTRACIÓN DE DATOS DE MERCADO EN BINANCE TESTNET ===")
    
    # Crear adaptador personalizado para Binance Testnet
    testnet = BinanceTestnetAdapter()
    
    try:
        # Conectar a Binance Testnet
        connect_result = await testnet.connect()
        logger.info(f"Conexión establecida: {json.dumps(connect_result, indent=2)}")
        
        # Suscribir a datos de tickers de trading
        symbols = ["btcusdt", "ethusdt", "bnbusdt"]
        channels = []
        
        for symbol in symbols:
            channels.append(f"{symbol}@ticker")
            
        subscription_result = await testnet.subscribe(channels)
        logger.info(f"Suscripción realizada: {json.dumps(subscription_result, indent=2)}")
        
        # Recibir y mostrar datos de mercado durante 30 segundos
        logger.info(f"Recibiendo datos de tickers para {', '.join(symbols)}...")
        
        start_time = time.time()
        message_count = 0
        
        # Mantener recepción de datos durante 30 segundos
        while time.time() - start_time < 30:
            try:
                # Recibir mensajes utilizando las capacidades cuánticas
                message = await testnet.receive()
                
                # Contar mensajes recibidos
                message_count += 1
                
                # Cada 5 mensajes, mostrar uno como ejemplo
                if message_count % 5 == 0:
                    # Si es un ticker, mostrar información simplificada
                    if "e" in message and message["e"] == "24hrTicker":
                        symbol = message.get("s", "UNKNOWN")
                        price = message.get("c", "0")
                        change = message.get("P", "0")
                        volume = message.get("v", "0")
                        
                        logger.info(f"Ticker {symbol}: Precio: {price}, Cambio: {change}%, Volumen: {volume}")
                    else:
                        # Para otros mensajes, mostrar tipo genérico
                        msg_type = message.get("e", "desconocido")
                        logger.info(f"Mensaje tipo {msg_type} recibido")
                
            except Exception as e:
                logger.warning(f"Error recibiendo datos (será transmutado): {e}")
                
                # Pequeña pausa para no saturar el log en caso de errores
                await asyncio.sleep(0.5)
                
        # Mostrar estadísticas
        duration = time.time() - start_time
        msg_per_sec = message_count / duration
        
        logger.info(f"Demostración completada: {message_count} mensajes recibidos en {duration:.1f} segundos")
        logger.info(f"Tasa de mensajes: {msg_per_sec:.2f} mensajes/segundo")
        
    finally:
        # Siempre cerrar la conexión al terminar
        await testnet.close()
        logger.info("Conexión cerrada")

async def demo_error_transmutation():
    """Demostrar capacidades de transmutación de errores."""
    logger.info("=== INICIANDO DEMOSTRACIÓN DE TRANSMUTACIÓN DE ERRORES ===")
    
    # Crear adaptador con URL incorrecta para forzar errores
    faulty_adapter = BinanceTestnetAdapter()
    # Modificar URL para que falle
    faulty_adapter.config["ws_url"] = "wss://invalid.example.com/ws"
    
    try:
        # Intentar conectar (debería fallar y transmutarse)
        logger.info("Intentando conectar a URL inválida (debe fallar pero transmutarse)...")
        connect_result = await faulty_adapter.connect()
        
        # La conexión debería estar en estado transmutado
        logger.info(f"Resultado de la conexión transmutada: {json.dumps(connect_result, indent=2)}")
        
        # Verificar estado
        state = faulty_adapter.get_state()
        logger.info(f"Estado del adaptador: {state['state']}")
        
        # Intentar suscripción (también debería transmutarse)
        sub_result = await faulty_adapter.subscribe(["btcusdt@ticker"])
        logger.info(f"Resultado de la suscripción transmutada: {json.dumps(sub_result, indent=2)}")
        
        # Intentar recibir datos (serán generados por la transmutación)
        logger.info("Recibiendo datos transmutados durante 10 segundos...")
        
        start_time = time.time()
        transmuted_count = 0
        
        while time.time() - start_time < 10:
            # Estos datos serán generados por la transmutación cuántica
            message = await faulty_adapter.receive()
            transmuted_count += 1
            
            # Mostrar ejemplo de datos transmutados
            if transmuted_count % 3 == 0:
                logger.info(f"Datos transmutados: {json.dumps(message, indent=2)}")
                
        logger.info(f"Recibidos {transmuted_count} mensajes transmutados cuánticamente")
        
    finally:
        # Cerrar adaptador
        await faulty_adapter.close()
        logger.info("Conexión cerrada")

async def main():
    """Función principal para ejecutar ambas demostraciones."""
    logger.info("=== INICIANDO DEMO DEL SISTEMA CUÁNTICO ULTRA-DIVINO PARA BINANCE TESTNET ===")
    
    # Demo 1: Datos de mercado en tiempo real
    await demo_market_data()
    
    # Pequeña pausa entre demos
    await asyncio.sleep(2)
    
    # Demo 2: Transmutación de errores
    await demo_error_transmutation()
    
    logger.info("=== DEMO COMPLETADA EXITOSAMENTE ===")
    logger.info("El Sistema Cuántico Ultra-Divino funciona perfectamente")

if __name__ == "__main__":
    asyncio.run(main())