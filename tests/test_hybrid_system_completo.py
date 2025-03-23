"""
Prueba completa del Sistema Genesis Híbrido con WebSocket y API Trascendental.

Este test evalúa la integración de todos los componentes del sistema híbrido:
- WebSocket Externo para intercambio de datos con exchanges
- EventBus Trascendental para comunicación interna
- Conector WebSocket para integración entre ambos
- API Trascendental para interfaz externa

El objetivo es verificar que todo el sistema funciona correctamente como una unidad.
"""

import asyncio
import logging
import json
import time
import sys
import os
from typing import Dict, Any, List, Optional

# Asegurar que el directorio raíz está en el path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("Genesis.SystemTest")

# Importar componentes del sistema
from genesis.core.transcendental_external_websocket import TranscendentalExternalWebSocket
from genesis.core.exchange_websocket_connector import ExchangeWebSocketConnector
from genesis.core.transcendental_event_bus import TranscendentalEventBus, EventPriority, SystemMode
from transcendental_ws_adapter import TranscendentalWebSocketAdapter, TranscendentalAPI

# Componentes de prueba
class MarketDataProcessor:
    """Componente para procesar datos de mercado."""
    
    def __init__(self, component_id: str = "market_data_processor"):
        self.component_id = component_id
        self.logger = logging.getLogger(f"Genesis.{component_id}")
        self.received_events = []
        self.processed_data = {}
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Manejar eventos del EventBus."""
        self.logger.info(f"Recibido evento {event_type} de {source}")
        self.received_events.append({
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": time.time()
        })
        
        # Procesar según tipo
        if event_type.startswith("market_data"):
            symbol = data.get("_symbol", data.get("symbol", "unknown"))
            if symbol not in self.processed_data:
                self.processed_data[symbol] = []
                
            self.processed_data[symbol].append(data)
            
            # Solo mantener los últimos 10 datos por símbolo
            if len(self.processed_data[symbol]) > 10:
                self.processed_data[symbol] = self.processed_data[symbol][-10:]
                
        return {
            "success": True,
            "processed_by": self.component_id,
            "timestamp": time.time()
        }
        
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del procesador."""
        return {
            "events_received": len(self.received_events),
            "symbols_processed": len(self.processed_data),
            "symbols": list(self.processed_data.keys())
        }

class TradingStrategy:
    """Estrategia de trading para pruebas."""
    
    def __init__(self, component_id: str = "trading_strategy"):
        self.component_id = component_id
        self.logger = logging.getLogger(f"Genesis.{component_id}")
        self.market_data = {}
        self.signals = []
        self.is_active = False
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Manejar eventos del EventBus."""
        self.logger.info(f"Estrategia recibió evento {event_type} de {source}")
        
        # Procesar datos de mercado
        if event_type.startswith("market_data"):
            symbol = data.get("_symbol", data.get("symbol", "unknown"))
            if symbol not in self.market_data:
                self.market_data[symbol] = {}
                
            # Actualizar datos
            self.market_data[symbol].update({
                "price": data.get("price", 0.0),
                "timestamp": data.get("timestamp", time.time()),
                "last_update": time.time()
            })
            
            # Generar señal si estamos activos
            if self.is_active and "price" in data:
                signal = {
                    "symbol": symbol,
                    "price": data.get("price", 0.0),
                    "action": "BUY" if hash(str(time.time())) % 2 == 0 else "SELL",
                    "confidence": 0.7 + (time.time() % 0.3),
                    "timestamp": time.time()
                }
                self.signals.append(signal)
                self.logger.info(f"Generada señal: {signal['action']} {symbol} @ {signal['price']}")
                
        # Activar/desactivar estrategia
        elif event_type == "strategy_control":
            if "action" in data:
                if data["action"] == "start":
                    self.is_active = True
                    self.logger.info("Estrategia activada")
                elif data["action"] == "stop":
                    self.is_active = False
                    self.logger.info("Estrategia desactivada")
                    
        return {
            "success": True,
            "processed_by": self.component_id,
            "timestamp": time.time()
        }
        
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de la estrategia."""
        return {
            "symbols_tracked": len(self.market_data),
            "signals_generated": len(self.signals),
            "is_active": self.is_active,
            "last_signals": self.signals[-3:] if self.signals else []
        }

class OrderManager:
    """Gestor de órdenes para pruebas."""
    
    def __init__(self, component_id: str = "order_manager"):
        self.component_id = component_id
        self.logger = logging.getLogger(f"Genesis.{component_id}")
        self.orders = []
        self.pending_orders = []
        self.completed_orders = []
        self.exchange_connector = None
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Manejar eventos del EventBus."""
        self.logger.info(f"Gestor de órdenes recibió evento {event_type} de {source}")
        
        # Procesar señales de trading
        if event_type == "trade_signal" and self.exchange_connector:
            # Crear orden a partir de la señal
            symbol = data.get("symbol", "")
            price = data.get("price", 0.0)
            action = data.get("action", "")
            
            if symbol and price and action in ["BUY", "SELL"]:
                order = {
                    "id": f"order_{int(time.time() * 1000)}_{len(self.orders)}",
                    "symbol": symbol,
                    "price": price,
                    "size": 0.01,  # Tamaño fijo para pruebas
                    "side": action,
                    "status": "PENDING",
                    "created_at": time.time()
                }
                self.orders.append(order)
                self.pending_orders.append(order)
                self.logger.info(f"Creada orden: {order['side']} {order['size']} {symbol} @ {price}")
                
                # Simular envío a exchange
                # En un sistema real, aquí enviaríamos la orden al exchange
                # mediante el exchange_connector
                
        return {
            "success": True,
            "processed_by": self.component_id,
            "timestamp": time.time()
        }
        
    def set_exchange_connector(self, connector: ExchangeWebSocketConnector):
        """Establecer conector de exchange."""
        self.exchange_connector = connector
        
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del gestor de órdenes."""
        return {
            "total_orders": len(self.orders),
            "pending_orders": len(self.pending_orders),
            "completed_orders": len(self.completed_orders)
        }

async def setup_system():
    """Configurar el sistema completo."""
    logger.info("=== INICIANDO CONFIGURACIÓN DEL SISTEMA ===")
    
    # Crear componentes principales
    event_bus = TranscendentalEventBus()
    ws_adapter = TranscendentalWebSocketAdapter()
    api = TranscendentalAPI(ws_adapter=ws_adapter)
    exchange_ws = TranscendentalExternalWebSocket("binance", testnet=True)
    connector = ExchangeWebSocketConnector(event_bus=event_bus)
    
    # Crear componentes de negocio
    market_processor = MarketDataProcessor()
    trading_strategy = TradingStrategy()
    order_manager = OrderManager()
    order_manager.set_exchange_connector(connector)
    
    # Iniciar componentes
    await event_bus.start()
    await ws_adapter.start()
    await api.initialize(intensity=10.0)
    await exchange_ws.connect()
    await connector.initialize(event_bus)
    await connector.register_exchange(exchange_ws)
    
    # Registrar componentes en el EventBus
    await event_bus.register_component(
        market_processor.component_id,
        market_processor.handle_event,
        is_essential=True
    )
    await event_bus.register_component(
        trading_strategy.component_id,
        trading_strategy.handle_event
    )
    await event_bus.register_component(
        order_manager.component_id,
        order_manager.handle_event
    )
    
    # Suscribir componentes a eventos
    await event_bus.subscribe(market_processor.component_id, ["market_data"])
    await event_bus.subscribe(trading_strategy.component_id, ["market_data", "strategy_control"])
    await event_bus.subscribe(order_manager.component_id, ["trade_signal"])
    
    # Suscribir a símbolos en el exchange
    await connector.subscribe("BTC/USDT", ["ticker"])
    await connector.subscribe("ETH/USDT", ["ticker"])
    
    # Establecer modo trascendental para máxima resiliencia
    event_bus.set_mode(SystemMode.TRANSCENDENTAL)
    
    logger.info("=== SISTEMA CONFIGURADO CORRECTAMENTE ===")
    
    # Devolver todos los componentes para uso en tests
    return {
        "event_bus": event_bus,
        "ws_adapter": ws_adapter,
        "api": api,
        "exchange_ws": exchange_ws,
        "connector": connector,
        "market_processor": market_processor,
        "trading_strategy": trading_strategy,
        "order_manager": order_manager
    }

async def run_system_test():
    """Ejecutar prueba completa del sistema."""
    # Configurar sistema
    components = await setup_system()
    
    event_bus = components["event_bus"]
    connector = components["connector"]
    market_processor = components["market_processor"]
    trading_strategy = components["trading_strategy"]
    order_manager = components["order_manager"]
    
    logger.info("=== INICIANDO PRUEBA DEL SISTEMA ===")
    
    # Simular actividad del sistema durante 5 segundos
    logger.info("Simulando actividad del sistema durante 5 segundos...")
    
    # Activar estrategia
    await event_bus.emit_local(
        "strategy_control",
        {"action": "start"},
        "test",
        EventPriority.HIGH
    )
    
    # Esperar a que el sistema procese algunos datos
    for i in range(5):
        # Obtener datos de mercado manualmente cada segundo
        btc_ticker = await connector.get_market_data("BTC/USDT", "ticker")
        eth_ticker = await connector.get_market_data("ETH/USDT", "ticker")
        
        logger.info(f"BTC/USDT: ${btc_ticker['price']:.2f}, ETH/USDT: ${eth_ticker['price']:.2f}")
        
        # Esperar 1 segundo
        await asyncio.sleep(1)
        
        # Cada 2 segundos, emitir una señal de trading manualmente
        if i % 2 == 0:
            signal = {
                "symbol": "BTC/USDT",
                "price": btc_ticker['price'],
                "action": "BUY" if i % 4 == 0 else "SELL",
                "confidence": 0.85,
                "timestamp": time.time()
            }
            
            await event_bus.emit_local(
                "trade_signal",
                signal,
                "test",
                EventPriority.CRITICAL
            )
            
            logger.info(f"Emitida señal manual: {signal['action']} {signal['symbol']} @ {signal['price']}")
    
    # Desactivar estrategia
    await event_bus.emit_local(
        "strategy_control",
        {"action": "stop"},
        "test",
        EventPriority.HIGH
    )
    
    # Obtener estadísticas finales
    mp_stats = market_processor.get_stats()
    ts_stats = trading_strategy.get_stats()
    om_stats = order_manager.get_stats()
    eb_stats = event_bus.get_stats()
    conn_stats = connector.get_stats()
    
    # Recopilar resultados
    results = {
        "market_processor": mp_stats,
        "trading_strategy": ts_stats,
        "order_manager": om_stats,
        "event_bus": eb_stats,
        "connector": conn_stats,
        "test_duration": 5,
        "timestamp": time.time()
    }
    
    # Mostrar resultados
    logger.info("=== RESULTADOS DE LA PRUEBA ===")
    logger.info(f"Eventos procesados: {eb_stats['events_processed']}")
    logger.info(f"Símbolos procesados: {mp_stats['symbols_processed']}")
    logger.info(f"Señales generadas: {ts_stats['signals_generated']}")
    logger.info(f"Órdenes creadas: {om_stats['total_orders']}")
    
    # Limpiar
    logger.info("Limpiando recursos...")
    
    # Detener componentes en orden inverso
    await components["connector"].shutdown()
    await components["exchange_ws"].disconnect()
    await components["event_bus"].stop()
    
    logger.info("=== PRUEBA FINALIZADA ===")
    
    return results

async def main():
    """Función principal."""
    start_time = time.time()
    
    try:
        # Ejecutar prueba
        results = await run_system_test()
        
        # Guardar resultados
        with open("resultados_hybrid_system.json", "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Resultados guardados en resultados_hybrid_system.json")
        
    except Exception as e:
        logger.error(f"Error durante la prueba: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        total_time = time.time() - start_time
        logger.info(f"Tiempo total: {total_time:.2f} segundos")

if __name__ == "__main__":
    asyncio.run(main())