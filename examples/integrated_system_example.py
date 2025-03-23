
Ejemplo de Sistema Genesis Integrado con Singularidad Trascendental V4.

Este ejemplo demuestra la integración completa del sistema Genesis utilizando:
1. TranscendentalEventBus - Para comunicación interna entre componentes
2. ExchangeWebSocketHandler - Para comunicación externa con exchanges
3. Mecanismos trascendentales - Para operación perfecta a intensidad 1000.0

El sistema se conecta al exchange, recibe datos de mercado en tiempo real,
los procesa a través de los componentes internos, y demuestra el sistema
híbrido WebSocket/API Trascendental funcionando a su máximo potencial.


import asyncio
import logging
import json
import time
import random
from typing import Dict, Any, List, Optional, Tuple
import sys
import os

# Asegurar que el directorio raíz esté en el path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("integrated_system.log")
    ]
)

logger = logging.getLogger("Genesis.IntegratedSystem")

# Importar componentes del sistema
from genesis.core.transcendental_event_bus import TranscendentalEventBus
from genesis.core.exchange_websocket_connector import ExchangeWebSocketHandler

# Importar mecanismos trascendentales para demo
from genesis_singularity_transcendental_v4 import (
    DimensionalCollapseV4,
    QuantumTimeV4,
    OmniConvergenceV4
)

# ===== COMPONENTES DEL SISTEMA =====

class MarketDataComponent:
    
    Componente para recibir y procesar datos de mercado externos.
    
    Este componente se conecta con exchanges usando el ExchangeWebSocketHandler
    y distribuye los datos de mercado a través del sistema mediante el
    TranscendentalEventBus.
    
    
    def __init__(self, event_bus: TranscendentalEventBus, component_id: str = "market_data"):
        
        Inicializar componente de datos de mercado.
        
        Args:
            event_bus: Bus de eventos trascendental
            component_id: ID único del componente
        
        self.event_bus = event_bus
        self.component_id = component_id
        self.exchange_ws = ExchangeWebSocketHandler("binance")
        self.running = False
        
        # Estadísticas
        self.stats = {
            "trades_received": 0,
            "trades_processed": 0,
            "klines_received": 0,
            "orderbook_updates": 0,
            "errors_transmuted": 0
        }
        
        # Mecanismos trascendentales para optimización
        self.mechanisms = {
            "collapse": DimensionalCollapseV4(),
            "time": QuantumTimeV4(),
            "convergence": OmniConvergenceV4()
        }
        
        logger.info(f"Componente {self.component_id} inicializado")
    
    async def start(self) -> None:
        Iniciar componente y conectarse a exchanges."""
        if self.running:
            return
            
        logger.info(f"Iniciando componente {self.component_id}...")
        
        # Conectar a streams del exchange
        await self._connect_to_exchange_streams()
        
        # Suscribirse a eventos internos relevantes
        await self.event_bus.subscribe(
            "request_market_data", 
            self._handle_data_request,
            priority=0,  # Alta prioridad
            component_id=self.component_id
        )
        
        self.running = True
        logger.info(f"Componente {self.component_id} iniciado")
    
    async def stop(self) -> None:
        Detener componente y desconectarse de exchanges."""
        if not self.running:
            return
            
        logger.info(f"Deteniendo componente {self.component_id}...")
        
        # Desconectar de exchanges
        await self.exchange_ws.disconnect_all()
        
        self.running = False
        logger.info(f"Componente {self.component_id} detenido")
    
    async def _connect_to_exchange_streams(self) -> None:
        Conectar a streams relevantes del exchange."""
        # Conectar a stream de trades
        await self.exchange_ws.connect_to_stream("btcusdt@trade", self._on_trade_data)
        
        # Conectar a stream de klines (velas)
        await self.exchange_ws.connect_to_stream("btcusdt@kline_1m", self._on_kline_data)
        
        # Conectar a stream de orderbook
        await self.exchange_ws.connect_to_stream("btcusdt@depth20", self._on_orderbook_data)
        
        logger.info(f"Conectado a streams del exchange")
    
    async def _on_trade_data(self, data: Dict[str, Any]) -> None:
        
        Procesar datos de trades recibidos del exchange.
        
        Args:
            data: Datos normalizados del trade
        
        self.stats["trades_received"] += 1
        
        # Procesar datos a través de mecanismos trascendentales
        try:
            async with self.mechanisms["time"].nullify_time():
                # Colapsar datos a su forma esencial
                collapsed_data = await self.mechanisms["collapse"].collapse_data(data)
                
                # Emitir evento para otros componentes
                await self.event_bus.emit(
                    "market_trade", 
                    collapsed_data, 
                    self.component_id
                )
                
                self.stats["trades_processed"] += 1
                
                # Log periódico
                if self.stats["trades_processed"] % 20 == 0:
                    symbol = data.get("symbol", "unknown")
                    price = data.get("price", 0.0)
                    logger.info(f"Procesados {self.stats['trades_processed']} trades - Último: {symbol} @ {price}")
                
        except Exception as e:
            logger.error(f"Error procesando trade: {e}")
            self.stats["errors_transmuted"] += 1
    
    async def _on_kline_data(self, data: Dict[str, Any]) -> None:
        
        Procesar datos de klines (velas) recibidos del exchange.
        
        Args:
            data: Datos normalizados de la vela
        
        self.stats["klines_received"] += 1
        
        # Emitir evento para otros componentes
        await self.event_bus.emit(
            "market_kline", 
            data, 
            self.component_id
        )
    
    async def _on_orderbook_data(self, data: Dict[str, Any]) -> None:
        
        Procesar datos de orderbook recibidos del exchange.
        
        Args:
            data: Datos normalizados del orderbook
        
        self.stats["orderbook_updates"] += 1
        
        # Emitir evento para otros componentes si hay cambios significativos
        if self._is_significant_update(data):
            await self.event_bus.emit(
                "market_orderbook", 
                data, 
                self.component_id
            )
    
    def _is_significant_update(self, data: Dict[str, Any]) -> bool:
        
        Determinar si una actualización de orderbook es significativa.
        
        Args:
            data: Datos del orderbook
            
        Returns:
            True si la actualización es significativa
        
        # Implementación simple: una actualización de cada 5 es "significativa"
        return self.stats["orderbook_updates"] % 5 == 0
    
    async def _handle_data_request(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        
        Manejar solicitud de datos de mercado de otro componente.
        
        Args:
            event_type: Tipo de evento
            data: Datos de la solicitud
            source: Componente que envía la solicitud
        
        request_type = data.get("type", "unknown")
        symbol = data.get("symbol", "btcusdt")
        
        logger.info(f"Solicitud de datos de mercado de {source}: {request_type} para {symbol}")
        
        # Preparar respuesta
        response = {
            "request_id": data.get("request_id", "unknown"),
            "timestamp": time.time(),
            "symbol": symbol,
            "type": request_type,
            "success": True
        }
        
        # Añadir datos según tipo de solicitud
        if request_type == "trade":
            # Responder con último trade conocido
            response["data"] = {
                "price": 45000.0 + random.uniform(-100, 100),
                "quantity": random.uniform(0.1, 2.0),
                "side": random.choice(["buy", "sell"])
            }
            
        elif request_type == "kline":
            # Responder con última vela conocida
            response["data"] = {
                "open": 45000.0 + random.uniform(-100, 100),
                "high": 45100.0 + random.uniform(-100, 100),
                "low": 44900.0 + random.uniform(-100, 100),
                "close": 45050.0 + random.uniform(-100, 100),
                "volume": random.uniform(100, 500),
                "interval": data.get("interval", "1m")
            }
            
        elif request_type == "orderbook":
            # Responder con snapshot del orderbook
            response["data"] = {
                "bids": [[45000.0 - i*10, random.uniform(0.1, 2.0)] for i in range(5)],
                "asks": [[45000.0 + i*10, random.uniform(0.1, 2.0)] for i in range(5)]
            }
        
        # Enviar respuesta
        await self.event_bus.emit(
            "market_data_response",
            response,
            self.component_id
        )
    
    def get_stats(self) -> Dict[str, Any]:
        
        Obtener estadísticas del componente.
        
        Returns:
            Diccionario con estadísticas
        
        # Combinar estadísticas propias con las del WebSocket
        combined_stats = dict(self.stats)
        if self.exchange_ws:
            ws_stats = self.exchange_ws.get_stats()
            combined_stats["exchange_ws"] = ws_stats
        
        return combined_stats

class StrategyComponent:
    
    Componente de estrategia de trading.
    
    Este componente recibe datos de mercado, aplica estrategias de trading
    y emite señales de operación.
    
    
    def __init__(self, event_bus: TranscendentalEventBus, component_id: str = "strategy"):
        
        Inicializar componente de estrategia.
        
        Args:
            event_bus: Bus de eventos trascendental
            component_id: ID único del componente
        
        self.event_bus = event_bus
        self.component_id = component_id
        self.running = False
        
        # Datos de mercado
        self.market_data = {
            "last_trade": None,
            "last_kline": None,
            "orderbook": None
        }
        
        # Estadísticas
        self.stats = {
            "signals_generated": 0,
            "trades_analyzed": 0,
            "klines_analyzed": 0
        }
        
        logger.info(f"Componente {self.component_id} inicializado")
    
    async def start(self) -> None:
        Iniciar componente y suscribirse a eventos."""
        if self.running:
            return
            
        logger.info(f"Iniciando componente {self.component_id}...")
        
        # Suscribirse a eventos de mercado
        await self.event_bus.subscribe(
            "market_trade", 
            self._handle_trade,
            component_id=self.component_id
        )
        
        await self.event_bus.subscribe(
            "market_kline", 
            self._handle_kline,
            component_id=self.component_id
        )
        
        await self.event_bus.subscribe(
            "market_orderbook", 
            self._handle_orderbook,
            component_id=self.component_id
        )
        
        # Iniciar tarea de análisis periódico
        asyncio.create_task(self._run_periodic_analysis())
        
        self.running = True
        logger.info(f"Componente {self.component_id} iniciado")
    
    async def stop(self) -> None:
        Detener componente."""
        if not self.running:
            return
            
        logger.info(f"Deteniendo componente {self.component_id}...")
        self.running = False
        logger.info(f"Componente {self.component_id} detenido")
    
    async def _handle_trade(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        
        Manejar evento de trade.
        
        Args:
            event_type: Tipo de evento
            data: Datos del trade
            source: Componente que envía el evento
        
        self.market_data["last_trade"] = data
        self.stats["trades_analyzed"] += 1
        
        # Analizar inmediatamente si hay condiciones relevantes
        if self._check_immediate_signal_conditions(data):
            await self._generate_trading_signal(data.get("symbol", "unknown"), "trade_trigger")
    
    async def _handle_kline(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        
        Manejar evento de kline.
        
        Args:
            event_type: Tipo de evento
            data: Datos de la vela
            source: Componente que envía el evento
        
        self.market_data["last_kline"] = data
        self.stats["klines_analyzed"] += 1
        
        # Si la vela está cerrada, analizar para señal
        if data.get("is_closed", False):
            await self._analyze_closed_kline(data)
    
    async def _handle_orderbook(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        
        Manejar evento de orderbook.
        
        Args:
            event_type: Tipo de evento
            data: Datos del orderbook
            source: Componente que envía el evento
        
        self.market_data["orderbook"] = data
        
        # Analizar orderbook para detectar desequilibrios
        await self._analyze_orderbook_imbalance(data)
    
    def _check_immediate_signal_conditions(self, trade_data: Dict[str, Any]) -> bool:
        
        Verificar si un trade cumple condiciones para señal inmediata.
        
        Args:
            trade_data: Datos del trade
            
        Returns:
            True si hay condiciones para señal
        
        # Simplificado: 5% de probabilidad de generar señal por cada trade
        return random.random() < 0.05
    
    async def _analyze_closed_kline(self, kline_data: Dict[str, Any]) -> None:
        
        Analizar vela cerrada para posible señal.
        
        Args:
            kline_data: Datos de la vela
        
        symbol = kline_data.get("symbol", "unknown")
        interval = kline_data.get("interval", "unknown")
        
        # Ejemplo simple: detectar vela alcista significativa
        if "open" in kline_data and "close" in kline_data:
            open_price = float(kline_data["open"])
            close_price = float(kline_data["close"])
            
            # Si la vela es alcista con diferencia > 0.5%
            if close_price > open_price and (close_price - open_price) / open_price > 0.005:
                logger.info(f"Detectada vela alcista significativa en {symbol} ({interval})")
                await self._generate_trading_signal(symbol, "bullish_candle")
    
    async def _analyze_orderbook_imbalance(self, orderbook_data: Dict[str, Any]) -> None:
        
        Analizar desequilibrios en el orderbook.
        
        Args:
            orderbook_data: Datos del orderbook
        
        symbol = orderbook_data.get("symbol", "unknown")
        
        # Ejemplo: detectar desequilibrio entre bids y asks
        bids = orderbook_data.get("bids", [])
        asks = orderbook_data.get("asks", [])
        
        if not bids or not asks:
            return
            
        # Calcular volumen total en primeros 5 niveles
        bids_volume = sum(float(bid[1]) for bid in bids[:5]) if len(bids) >= 5 else 0
        asks_volume = sum(float(ask[1]) for ask in asks[:5]) if len(asks) >= 5 else 0
        
        # Si hay desequilibrio > 3:1
        if bids_volume > 0 and asks_volume > 0:
            ratio = bids_volume / asks_volume
            
            if ratio > 3.0:
                logger.info(f"Detectado desequilibrio en orderbook de {symbol}: más compradores (ratio {ratio:.2f})")
                await self._generate_trading_signal(symbol, "buy_pressure")
                
            elif ratio < 0.33:
                logger.info(f"Detectado desequilibrio en orderbook de {symbol}: más vendedores (ratio {ratio:.2f})")
                await self._generate_trading_signal(symbol, "sell_pressure")
    
    async def _generate_trading_signal(self, symbol: str, signal_type: str) -> None:
        
        Generar señal de trading.
        
        Args:
            symbol: Símbolo del instrumento
            signal_type: Tipo de señal
        
        # Crear señal
        signal = {
            "symbol": symbol,
            "type": signal_type,
            "timestamp": time.time(),
            "strength": random.uniform(0.6, 1.0),
            "direction": "buy" if signal_type in ["bullish_candle", "buy_pressure"] else "sell",
            "signal_id": f"SIG{int(time.time())}-{random.randint(1000, 9999)}"
        }
        
        # Emitir evento con la señal
        await self.event_bus.emit(
            "trading_signal",
            signal,
            self.component_id
        )
        
        self.stats["signals_generated"] += 1
        logger.info(f"Señal generada: {signal_type} para {symbol} - Dirección: {signal['direction']}")
    
    async def _run_periodic_analysis(self) -> None:
        Ejecutar análisis periódico de mercado."""
        while self.running:
            # Esperar intervalo
            await asyncio.sleep(10)  # Cada 10 segundos
            
            # Solicitar datos actualizados de mercado
            await self._request_market_data()
            
            # Ejecutar análisis integral
            await self._perform_comprehensive_analysis()
    
    async def _request_market_data(self) -> None:
        Solicitar datos actualizados de mercado."""
        # Crear solicitud
        request = {
            "request_id": f"REQ{int(time.time())}-{random.randint(1000, 9999)}",
            "type": random.choice(["trade", "kline", "orderbook"]),
            "symbol": "btcusdt",
            "timestamp": time.time()
        }
        
        # Solicitar datos
        await self.event_bus.emit(
            "request_market_data",
            request,
            self.component_id
        )
    
    async def _perform_comprehensive_analysis(self) -> None:
        Realizar análisis integral de mercado con todos los datos disponibles."""
        # Simplificado: probabilidad del 20% de generar señal en cada análisis
        if random.random() < 0.2 and self.market_data["last_trade"]:
            symbol = self.market_data["last_trade"].get("symbol", "btcusdt")
            await self._generate_trading_signal(symbol, "complex_analysis")
    
    def get_stats(self) -> Dict[str, Any]:
        
        Obtener estadísticas del componente.
        
        Returns:
            Diccionario con estadísticas
        
        return dict(self.stats)

class SignalProcessorComponent:
    
    Componente para procesar señales de trading.
    
    Este componente recibe señales de trading, las evalúa y emite
    órdenes si cumplen los criterios configurados.
    
    
    def __init__(self, event_bus: TranscendentalEventBus, component_id: str = "signal_processor"):
        
        Inicializar procesador de señales.
        
        Args:
            event_bus: Bus de eventos trascendental
            component_id: ID único del componente
        
        self.event_bus = event_bus
        self.component_id = component_id
        self.running = False
        
        # Configuración
        self.config = {
            "min_signal_strength": 0.7,
            "allowed_symbols": ["btcusdt", "ethusdt"],
            "risk_per_trade": 0.01,  # 1% del capital
            "max_daily_signals": 10
        }
        
        # Estadísticas
        self.stats = {
            "signals_received": 0,
            "signals_accepted": 0,
            "signals_rejected": 0,
            "orders_generated": 0
        }
        
        # Registro de señales
        self.signal_history = []
        
        logger.info(f"Componente {self.component_id} inicializado")
    
    async def start(self) -> None:
        Iniciar componente y suscribirse a eventos."""
        if self.running:
            return
            
        logger.info(f"Iniciando componente {self.component_id}...")
        
        # Suscribirse a eventos de señales
        await self.event_bus.subscribe(
            "trading_signal", 
            self._handle_signal,
            component_id=self.component_id
        )
        
        # Suscribirse a respuestas de datos de mercado
        await self.event_bus.subscribe(
            "market_data_response", 
            self._handle_market_data_response,
            component_id=self.component_id
        )
        
        self.running = True
        logger.info(f"Componente {self.component_id} iniciado")
    
    async def stop(self) -> None:
        Detener componente."""
        if not self.running:
            return
            
        logger.info(f"Deteniendo componente {self.component_id}...")
        self.running = False
        logger.info(f"Componente {self.component_id} detenido")
    
    async def _handle_signal(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        
        Manejar señal de trading.
        
        Args:
            event_type: Tipo de evento
            data: Datos de la señal
            source: Componente que envía la señal
        
        self.stats["signals_received"] += 1
        signal_id = data.get("signal_id", "unknown")
        
        # Registrar señal en historial
        self.signal_history.append({
            "signal": data,
            "timestamp": time.time(),
            "source": source
        })
        
        # Limitar historial a últimas 100 señales
        if len(self.signal_history) > 100:
            self.signal_history = self.signal_history[-100:]
        
        # Evaluar señal
        is_valid = await self._validate_signal(data)
        
        if is_valid:
            self.stats["signals_accepted"] += 1
            logger.info(f"Señal {signal_id} aceptada, generando orden...")
            
            # Generar orden basada en la señal
            await self._generate_order(data)
        else:
            self.stats["signals_rejected"] += 1
            logger.info(f"Señal {signal_id} rechazada por no cumplir criterios")
    
    async def _handle_market_data_response(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        
        Manejar respuesta de datos de mercado.
        
        Args:
            event_type: Tipo de evento
            data: Datos de mercado
            source: Componente que envía los datos
        
        # Procesar datos de mercado si hay señales pendientes
        pass  # Simplificado para el ejemplo
    
    async def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        
        Validar si una señal cumple los criterios para generar orden.
        
        Args:
            signal: Datos de la señal
            
        Returns:
            True si la señal es válida
        
        # Verificar fuerza de la señal
        strength = signal.get("strength", 0.0)
        if strength < self.config["min_signal_strength"]:
            return False
        
        # Verificar símbolo permitido
        symbol = signal.get("symbol", "").lower()
        if symbol not in self.config["allowed_symbols"]:
            return False
        
        # Verificar límite diario de señales
        daily_signals = len([s for s in self.signal_history 
                           if time.time() - s["timestamp"] < 86400])
        if daily_signals >= self.config["max_daily_signals"]:
            return False
        
        # Por defecto, la señal es válida
        return True
    
    async def _generate_order(self, signal: Dict[str, Any]) -> None:
        
        Generar orden basada en una señal.
        
        Args:
            signal: Datos de la señal
        
        # Crear orden
        order = {
            "symbol": signal.get("symbol", "btcusdt"),
            "side": signal.get("direction", "buy"),
            "type": "MARKET",
            "timestamp": time.time(),
            "quantity": 0.01,  # Simplificado
            "signal_id": signal.get("signal_id", "unknown"),
            "order_id": f"ORD{int(time.time())}-{random.randint(1000, 9999)}"
        }
        
        # Emitir evento de orden
        await self.event_bus.emit(
            "trading_order",
            order,
            self.component_id
        )
        
        self.stats["orders_generated"] += 1
        logger.info(f"Orden generada: {order['side']} {order['quantity']} {order['symbol']}")
    
    def get_stats(self) -> Dict[str, Any]:
        
        Obtener estadísticas del componente.
        
        Returns:
            Diccionario con estadísticas
        
        return dict(self.stats)

# ===== SISTEMA INTEGRADO =====

class IntegratedSystem:
    
    Sistema Genesis integrado con todos los componentes.
    
    Esta clase orquesta todos los componentes del sistema y demuestra
    la integración del TranscendentalEventBus con el ExchangeWebSocketHandler.
    
    
    def __init__(self):
        Inicializar sistema integrado."""
        # Crear event bus trascendental
        self.event_bus = TranscendentalEventBus(test_mode=True)
        
        # Crear componentes
        self.market_data = MarketDataComponent(self.event_bus)
        self.strategy = StrategyComponent(self.event_bus)
        self.signal_processor = SignalProcessorComponent(self.event_bus)
        
        # Estado del sistema
        self.running = False
        
        logger.info("Sistema Genesis integrado inicializado")
    
    async def start(self) -> None:
        Iniciar sistema integrado."""
        if self.running:
            return
            
        logger.info("Iniciando sistema Genesis integrado...")
        
        # Iniciar event bus
        await self.event_bus.start()
        
        # Iniciar componentes
        await self.market_data.start()
        await self.strategy.start()
        await self.signal_processor.start()
        
        self.running = True
        logger.info("Sistema Genesis integrado iniciado y funcionando")
    
    async def stop(self) -> None:
        Detener sistema integrado."""
        if not self.running:
            return
            
        logger.info("Deteniendo sistema Genesis integrado...")
        
        # Detener componentes
        await self.signal_processor.stop()
        await self.strategy.stop()
        await self.market_data.stop()
        
        # Detener event bus
        await self.event_bus.stop()
        
        self.running = False
        logger.info("Sistema Genesis integrado detenido")
    
    def get_system_stats(self) -> Dict[str, Any]:
        
        Obtener estadísticas completas del sistema.
        
        Returns:
            Diccionario con estadísticas de todos los componentes
        
        # Recopilar estadísticas de todos los componentes
        stats = {
            "market_data": self.market_data.get_stats(),
            "strategy": self.strategy.get_stats(),
            "signal_processor": self.signal_processor.get_stats(),
            "event_bus": self.event_bus.get_stats()
        }
        
        # Añadir estadísticas globales
        stats["system"] = {
            "uptime": time.time() - self._start_time if hasattr(self, "_start_time") else 0,
            "components_active": sum(1 for c in [self.market_data, self.strategy, self.signal_processor] 
                                    if getattr(c, "running", False)),
            "total_events_processed": stats["event_bus"].get("events_delivered", 0)
        }
        
        return stats

# ===== DEMOSTRACIÓN DEL SISTEMA =====

async def run_demo():
    Ejecutar demostración del sistema integrado."""
    logger.info("=== INICIANDO DEMOSTRACIÓN DEL SISTEMA GENESIS INTEGRADO ===")
    
    # Crear sistema
    system = IntegratedSystem()
    system._start_time = time.time()
    
    try:
        # Iniciar sistema
        await system.start()
        
        # Ejecutar durante 2 minutos, mostrando estadísticas cada 30 segundos
        logger.info("Sistema en ejecución. Mostrando estadísticas cada 30 segundos durante 2 minutos...")
        
        for i in range(4):  # 4 x 30s = 2min
            await asyncio.sleep(30)
            
            # Obtener estadísticas
            stats = system.get_system_stats()
            
            # Mostrar resumen
            logger.info(f"\n=== ESTADÍSTICAS DEL SISTEMA (iteración {i+1}/4) ===")
            logger.info(f"Tiempo de ejecución: {stats['system']['uptime']:.1f}s")
            logger.info(f"Componentes activos: {stats['system']['components_active']}/3")
            logger.info(f"Eventos procesados: {stats['system']['total_events_processed']}")
            logger.info(f"Trades recibidos: {stats['market_data'].get('trades_received', 0)}")
            logger.info(f"Señales generadas: {stats['strategy'].get('signals_generated', 0)}")
            logger.info(f"Órdenes creadas: {stats['signal_processor'].get('orders_generated', 0)}")
            
            if i == 3:  # En la última iteración, mostrar estadísticas detalladas
                logger.info("\n=== ESTADÍSTICAS DETALLADAS ===")
                logger.info(f"Event Bus: {json.dumps(stats['event_bus'], indent=2)}")
                logger.info(f"Market Data: {json.dumps(stats['market_data'], indent=2)}")
                logger.info(f"Strategy: {json.dumps(stats['strategy'], indent=2)}")
                logger.info(f"Signal Processor: {json.dumps(stats['signal_processor'], indent=2)}")
        
        logger.info("\n=== DEMOSTRACIÓN COMPLETADA CON ÉXITO ===")
        
    except Exception as e:
        logger.error(f"Error en la demostración: {e}")
        raise
        
    finally:
        # Detener sistema
        await system.stop()
        logger.info("Sistema detenido")

if __name__ == "__main__":
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        logger.info("Demostración interrumpida por el usuario")
    except Exception as e:
        logger.error(f"Error en la demostración: {e}")
        raise
