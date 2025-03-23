"""
Prueba de Integración del WebSocket Trascendental en Sistema Genesis.

Este script realiza pruebas avanzadas de integración del Sistema Genesis
con WebSocket Trascendental, verificando:
1. Integración perfecta entre TranscendentalEventBus y ExchangeWebSocketHandler
2. Flujo de datos desde exchanges externos hasta componentes internos
3. Rendimiento bajo intensidad extrema (1000.0)
4. Transmutación de errores en operaciones exitosas
5. Resiliencia del sistema completo integrado
"""

import asyncio
import logging
import json
import time
import random
from typing import Dict, Any, List, Optional, Tuple
import sys
import os
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Asegurar que el directorio raíz esté en el path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("test_integracion_transcendental.log")
    ]
)

logger = logging.getLogger("Genesis.TestIntegracion")

# Importar componentes del sistema
from genesis.core.transcendental_event_bus import TranscendentalEventBus
from genesis.core.exchange_websocket_connector import ExchangeWebSocketHandler
from genesis_singularity_transcendental_v4 import QuantumTimeV4

# Configuración global
DEFAULT_TEST_DURATION = 60  # segundos
DEFAULT_INTENSITY = 1000.0
DEFAULT_PARALLEL_STREAMS = 5
DEFAULT_MESSAGE_RATE = 100  # mensajes por segundo

class TestStats:
    """Estadísticas de prueba."""
    
    def __init__(self):
        """Inicializar estadísticas."""
        self.start_time = time.time()
        self.events_received = 0
        self.events_processed = 0
        self.errors = 0
        self.errors_transmuted = 0
        self.max_latency = 0.0
        self.min_latency = float('inf')
        self.total_latency = 0.0
        self.success_rate = 100.0
        
    def update_latency(self, latency: float) -> None:
        """
        Actualizar estadísticas de latencia.
        
        Args:
            latency: Tiempo de latencia en segundos
        """
        self.max_latency = max(self.max_latency, latency)
        self.min_latency = min(self.min_latency, latency)
        self.total_latency += latency
    
    def get_avg_latency(self) -> float:
        """
        Obtener latencia promedio.
        
        Returns:
            Latencia promedio en segundos
        """
        if self.events_processed == 0:
            return 0.0
        return self.total_latency / self.events_processed
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen de estadísticas.
        
        Returns:
            Diccionario con resumen de estadísticas
        """
        elapsed = time.time() - self.start_time
        return {
            "duration": elapsed,
            "events_received": self.events_received,
            "events_processed": self.events_processed,
            "errors": self.errors,
            "errors_transmuted": self.errors_transmuted,
            "rate": self.events_processed / elapsed if elapsed > 0 else 0,
            "success_rate": self.success_rate,
            "latency": {
                "min": self.min_latency if self.min_latency != float('inf') else 0,
                "max": self.max_latency,
                "avg": self.get_avg_latency()
            }
        }

class MarketDataSimulator:
    """
    Simulador de datos de mercado para pruebas.
    
    Este componente simula datos de mercado como si vinieran de un exchange,
    pero generados localmente para pruebas intensivas.
    """
    
    def __init__(self, exchange_ws: ExchangeWebSocketHandler, event_bus: TranscendentalEventBus, intensity: float = 1000.0):
        """
        Inicializar simulador.
        
        Args:
            exchange_ws: Manejador de WebSocket para exchanges
            event_bus: Bus de eventos trascendental
            intensity: Intensidad de la prueba (1.0 - 1000.0)
        """
        self.exchange_ws = exchange_ws
        self.event_bus = event_bus
        self.intensity = intensity
        self.running = False
        self.stats = TestStats()
        
        # Precomputar símbolos para pruebas
        self.symbols = ["btcusdt", "ethusdt", "bnbusdt", "xrpusdt", "adausdt", 
                       "solusdt", "dogeusdt", "dotusdt", "maticusdt", "linkusdt"]
        
        # Quantum Time para operaciones fuera del tiempo lineal
        self.quantum_time = QuantumTimeV4()
        
        logger.info(f"Simulador de datos de mercado inicializado con intensidad {intensity}")
    
    async def start(self, rate: int = 100, duration: int = 60) -> None:
        """
        Iniciar simulación de datos.
        
        Args:
            rate: Tasa de mensajes por segundo
            duration: Duración en segundos
        """
        if self.running:
            return
            
        logger.info(f"Iniciando simulación de datos a {rate} msgs/s durante {duration}s...")
        self.running = True
        self.stats = TestStats()  # Reiniciar estadísticas
        
        # Calcular intervalo entre mensajes
        interval = 1.0 / rate if rate > 0 else 0.01
        
        # Programar finalización
        end_time = time.time() + duration
        
        # Simular datos a la tasa especificada
        sent_count = 0
        
        async with self.quantum_time.nullify_time():
            while self.running and time.time() < end_time:
                # Generar y enviar mensaje
                symbol = random.choice(self.symbols)
                message_type = random.choice(["trade", "kline", "orderbook"])
                
                try:
                    # Generar datos simulados según tipo
                    if message_type == "trade":
                        await self._generate_trade(symbol)
                    elif message_type == "kline":
                        await self._generate_kline(symbol)
                    else:
                        await self._generate_orderbook(symbol)
                    
                    sent_count += 1
                    
                    # Log periódico
                    if sent_count % 1000 == 0:
                        elapsed = time.time() - self.stats.start_time
                        rate = sent_count / elapsed if elapsed > 0 else 0
                        logger.info(f"Generados {sent_count} mensajes ({rate:.2f} msgs/s)")
                    
                except Exception as e:
                    logger.error(f"Error generando mensaje: {e}")
                    self.stats.errors += 1
                
                # Esperar intervalo (adaptado por intensidad)
                await asyncio.sleep(interval / min(self.intensity, 100.0))
        
        # Finalizar
        self.running = False
        logger.info(f"Simulación finalizada. Generados {sent_count} mensajes.")
    
    async def stop(self) -> None:
        """Detener simulación."""
        self.running = False
        logger.info("Simulación detenida")
    
    async def _generate_trade(self, symbol: str) -> None:
        """
        Generar datos de trade simulados.
        
        Args:
            symbol: Símbolo del instrumento
        """
        # Generar precio base según símbolo
        base_price = {
            "btcusdt": 45000.0,
            "ethusdt": 3000.0,
            "bnbusdt": 400.0,
            "xrpusdt": 1.0,
            "adausdt": 1.2,
            "solusdt": 100.0,
            "dogeusdt": 0.1,
            "dotusdt": 20.0,
            "maticusdt": 1.5,
            "linkusdt": 15.0
        }.get(symbol, 10.0)
        
        # Añadir variación aleatoria (-0.5% a +0.5%)
        variation = random.uniform(-0.005, 0.005)
        price = base_price * (1 + variation)
        
        # Generar cantidad
        quantity = random.uniform(0.01, 10.0)
        
        # Crear datos de trade
        trade_data = {
            "e": "trade",                      # Evento
            "s": symbol.upper(),               # Símbolo
            "p": str(price),                   # Precio
            "q": str(quantity),                # Cantidad
            "T": int(time.time() * 1000),      # Timestamp
            "m": random.choice([True, False]), # Market maker
            "t": int(time.time() * 1000) + random.randint(1, 1000000)  # Trade ID
        }
        
        # Emitir a través del event bus como si viniera del exchange
        start_time = time.time()
        
        # Simular recepción del exchange
        await self._process_simulated_exchange_data(symbol, "trade", trade_data)
        
        # Actualizar estadísticas
        latency = time.time() - start_time
        self.stats.update_latency(latency)
        self.stats.events_processed += 1
    
    async def _generate_kline(self, symbol: str) -> None:
        """
        Generar datos de kline (vela) simulados.
        
        Args:
            symbol: Símbolo del instrumento
        """
        # Generar precio base según símbolo
        base_price = {
            "btcusdt": 45000.0,
            "ethusdt": 3000.0,
            "bnbusdt": 400.0,
            "xrpusdt": 1.0,
            "adausdt": 1.2,
            "solusdt": 100.0,
            "dogeusdt": 0.1,
            "dotusdt": 20.0,
            "maticusdt": 1.5,
            "linkusdt": 15.0
        }.get(symbol, 10.0)
        
        # Generar variaciones para OHLC
        open_var = random.uniform(-0.01, 0.01)
        high_var = random.uniform(0, 0.02)
        low_var = random.uniform(-0.02, 0)
        close_var = random.uniform(-0.01, 0.01)
        
        # Calcular precios
        open_price = base_price * (1 + open_var)
        high_price = base_price * (1 + max(open_var, close_var) + high_var)
        low_price = base_price * (1 + min(open_var, close_var) + low_var)
        close_price = base_price * (1 + close_var)
        
        # Asegurar que low <= open, close <= high
        low_price = min(low_price, open_price, close_price)
        high_price = max(high_price, open_price, close_price)
        
        # Crear datos de kline
        kline_data = {
            "e": "kline",                 # Evento
            "s": symbol.upper(),          # Símbolo
            "k": {
                "t": int(time.time() * 1000) - 60000,  # Inicio
                "T": int(time.time() * 1000),          # Fin
                "s": symbol.upper(),                   # Símbolo
                "i": "1m",                             # Intervalo
                "o": str(open_price),                  # Open
                "h": str(high_price),                  # High
                "l": str(low_price),                   # Low
                "c": str(close_price),                 # Close
                "v": str(random.uniform(100, 1000)),   # Volumen
                "x": random.choice([True, False])      # Cerrada
            }
        }
        
        # Emitir a través del event bus como si viniera del exchange
        start_time = time.time()
        
        # Simular recepción del exchange
        await self._process_simulated_exchange_data(symbol, "kline", kline_data)
        
        # Actualizar estadísticas
        latency = time.time() - start_time
        self.stats.update_latency(latency)
        self.stats.events_processed += 1
    
    async def _generate_orderbook(self, symbol: str) -> None:
        """
        Generar datos de orderbook simulados.
        
        Args:
            symbol: Símbolo del instrumento
        """
        # Generar precio base según símbolo
        base_price = {
            "btcusdt": 45000.0,
            "ethusdt": 3000.0,
            "bnbusdt": 400.0,
            "xrpusdt": 1.0,
            "adausdt": 1.2,
            "solusdt": 100.0,
            "dogeusdt": 0.1,
            "dotusdt": 20.0,
            "maticusdt": 1.5,
            "linkusdt": 15.0
        }.get(symbol, 10.0)
        
        # Generar bids (compra) - 10 niveles por debajo del precio base
        bids = []
        for i in range(10):
            price = base_price * (1 - 0.001 * (i + 1))
            quantity = random.uniform(0.1, 10.0)
            bids.append([str(price), str(quantity)])
        
        # Generar asks (venta) - 10 niveles por encima del precio base
        asks = []
        for i in range(10):
            price = base_price * (1 + 0.001 * (i + 1))
            quantity = random.uniform(0.1, 10.0)
            asks.append([str(price), str(quantity)])
        
        # Crear datos de orderbook
        orderbook_data = {
            "e": "depthUpdate",        # Evento
            "s": symbol.upper(),       # Símbolo
            "E": int(time.time() * 1000),  # Timestamp
            "b": bids,                 # Bids
            "a": asks                  # Asks
        }
        
        # Emitir a través del event bus como si viniera del exchange
        start_time = time.time()
        
        # Simular recepción del exchange
        await self._process_simulated_exchange_data(symbol, "depth", orderbook_data)
        
        # Actualizar estadísticas
        latency = time.time() - start_time
        self.stats.update_latency(latency)
        self.stats.events_processed += 1
    
    async def _process_simulated_exchange_data(self, symbol: str, data_type: str, data: Dict[str, Any]) -> None:
        """
        Procesar datos simulados como si vinieran de un exchange.
        
        Args:
            symbol: Símbolo del instrumento
            data_type: Tipo de datos (trade, kline, depth)
            data: Datos simulados
        """
        # Crear mensaje para el bus de eventos
        event_type = f"market_{data_type}"
        event_data = {
            "symbol": symbol,
            "type": data_type,
            "data": data,
            "timestamp": time.time(),
            "source": "exchange_simulator"
        }
        
        # Emitir evento
        try:
            await self.event_bus.emit(event_type, event_data, "exchange_simulator")
            self.stats.events_received += 1
        except Exception as e:
            logger.error(f"Error emitiendo evento: {e}")
            self.stats.errors += 1

class IntegrationTester:
    """
    Probador de integración del sistema WebSocket Trascendental.
    
    Este componente orquesta pruebas completas del sistema integrado.
    """
    
    def __init__(self, intensity: float = 1000.0):
        """
        Inicializar probador.
        
        Args:
            intensity: Intensidad de la prueba (1.0 - 1000.0)
        """
        self.intensity = intensity
        self.event_bus = None
        self.exchange_ws = None
        self.simulator = None
        self.component_stats = {}
        
        logger.info(f"Probador de integración inicializado con intensidad {intensity}")
    
    async def setup(self) -> None:
        """Configurar componentes para la prueba."""
        logger.info("Configurando componentes para prueba de integración...")
        
        # Crear TranscendentalEventBus
        self.event_bus = TranscendentalEventBus(test_mode=True)
        await self.event_bus.start()
        
        # Crear ExchangeWebSocketHandler
        self.exchange_ws = ExchangeWebSocketHandler("binance")
        
        # Crear simulador de datos
        self.simulator = MarketDataSimulator(self.exchange_ws, self.event_bus, self.intensity)
        
        # Conectar componentes mediante suscripciones
        await self._setup_event_listeners()
        
        logger.info("Componentes configurados correctamente")
    
    async def _setup_event_listeners(self) -> None:
        """Configurar listeners de eventos para la prueba."""
        # Escuchar eventos de trades
        await self.event_bus.subscribe(
            "market_trade", 
            self._handle_trade_event,
            component_id="test_integration"
        )
        
        # Escuchar eventos de klines
        await self.event_bus.subscribe(
            "market_kline", 
            self._handle_kline_event,
            component_id="test_integration"
        )
        
        # Escuchar eventos de orderbook
        await self.event_bus.subscribe(
            "market_orderbook", 
            self._handle_orderbook_event,
            component_id="test_integration"
        )
    
    async def _handle_trade_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Manejar evento de trade."""
        self.component_stats.setdefault("trades", 0)
        self.component_stats["trades"] += 1
    
    async def _handle_kline_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Manejar evento de kline."""
        self.component_stats.setdefault("klines", 0)
        self.component_stats["klines"] += 1
    
    async def _handle_orderbook_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Manejar evento de orderbook."""
        self.component_stats.setdefault("orderbooks", 0)
        self.component_stats["orderbooks"] += 1
    
    async def run_test(self, rate: int = 100, duration: int = 60, parallel_streams: int = 5) -> Dict[str, Any]:
        """
        Ejecutar prueba de integración.
        
        Args:
            rate: Tasa de mensajes por segundo
            duration: Duración en segundos
            parallel_streams: Número de streams paralelos
            
        Returns:
            Resultados de la prueba
        """
        logger.info(f"Iniciando prueba de integración: {rate} msgs/s, {duration}s, {parallel_streams} streams")
        
        try:
            # Iniciar simulación de datos
            simulation_task = asyncio.create_task(
                self.simulator.start(rate=rate, duration=duration)
            )
            
            # Emular streams paralelos adicionales
            parallel_tasks = []
            for i in range(parallel_streams - 1):  # -1 porque ya tenemos uno activo
                task = asyncio.create_task(self._simulate_parallel_stream(i, duration))
                parallel_tasks.append(task)
            
            # Esperar a que termine la simulación principal
            await simulation_task
            
            # Esperar a que terminen los streams paralelos
            if parallel_tasks:
                await asyncio.gather(*parallel_tasks)
            
            # Recopilar resultados
            simulator_stats = self.simulator.stats.get_summary()
            event_bus_stats = self.event_bus.get_stats()
            
            # Calcular tasa de éxito
            if simulator_stats["events_received"] > 0:
                success_rate = (self.component_stats.get("trades", 0) +
                               self.component_stats.get("klines", 0) +
                               self.component_stats.get("orderbooks", 0)) / simulator_stats["events_received"] * 100
            else:
                success_rate = 0
            
            # Preparar resultados
            results = {
                "intensity": self.intensity,
                "duration": duration,
                "message_rate": rate,
                "parallel_streams": parallel_streams,
                "events_generated": simulator_stats["events_processed"],
                "events_received": simulator_stats["events_received"],
                "component_events_processed": sum(self.component_stats.values()),
                "success_rate": success_rate,
                "latency": simulator_stats["latency"],
                "errors": simulator_stats["errors"],
                "event_bus_stats": {
                    "events_emitted": event_bus_stats.get("events_emitted", 0),
                    "events_delivered": event_bus_stats.get("events_delivered", 0),
                    "errors_transmuted": event_bus_stats.get("errors_transmuted", 0)
                },
                "component_stats": dict(self.component_stats)
            }
            
            logger.info(f"Prueba completada. Tasa de éxito: {success_rate:.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"Error en prueba de integración: {e}")
            raise
    
    async def _simulate_parallel_stream(self, stream_id: int, duration: int) -> None:
        """
        Simular stream paralelo para carga adicional.
        
        Args:
            stream_id: ID del stream
            duration: Duración en segundos
        """
        logger.info(f"Iniciando stream paralelo {stream_id}")
        
        event_count = 0
        end_time = time.time() + duration
        
        try:
            while time.time() < end_time:
                # Emitir evento simple
                event_type = f"parallel_stream_{stream_id}"
                event_data = {
                    "stream_id": stream_id,
                    "timestamp": time.time(),
                    "counter": event_count
                }
                
                await self.event_bus.emit(event_type, event_data, f"parallel_stream_{stream_id}")
                event_count += 1
                
                # Breve pausa proporcional a intensidad
                await asyncio.sleep(0.1 / min(self.intensity / 10, 100))
                
        except Exception as e:
            logger.error(f"Error en stream paralelo {stream_id}: {e}")
        
        logger.info(f"Stream paralelo {stream_id} finalizado. Eventos enviados: {event_count}")
    
    async def teardown(self) -> None:
        """Limpiar recursos después de la prueba."""
        logger.info("Limpiando recursos...")
        
        # Detener simulador
        if self.simulator:
            await self.simulator.stop()
        
        # Detener event bus
        if self.event_bus:
            await self.event_bus.stop()
        
        logger.info("Recursos liberados")

async def run_integration_test(intensity: float, rate: int, duration: int, parallel_streams: int) -> Dict[str, Any]:
    """
    Ejecutar una única prueba de integración.
    
    Args:
        intensity: Intensidad de la prueba
        rate: Tasa de mensajes por segundo
        duration: Duración en segundos
        parallel_streams: Número de streams paralelos
        
    Returns:
        Resultados de la prueba
    """
    # Crear y configurar tester
    tester = IntegrationTester(intensity=intensity)
    await tester.setup()
    
    try:
        # Ejecutar prueba
        results = await tester.run_test(
            rate=rate,
            duration=duration,
            parallel_streams=parallel_streams
        )
        return results
    finally:
        # Limpiar recursos
        await tester.teardown()

def save_results(results: Dict[str, Any], filename: str = "resultados_integracion_websocket.json") -> None:
    """
    Guardar resultados en archivo JSON.
    
    Args:
        results: Resultados a guardar
        filename: Nombre del archivo
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Resultados guardados en {filename}")
    except Exception as e:
        logger.error(f"Error guardando resultados: {e}")

def print_results_summary(results: Dict[str, Any]) -> None:
    """
    Imprimir resumen de resultados.
    
    Args:
        results: Resultados de la prueba
    """
    print("\n" + "="*80)
    print(f"RESUMEN DE PRUEBA DE INTEGRACIÓN WEBSOCKET TRASCENDENTAL")
    print("="*80)
    print(f"Intensidad: {results['intensity']:.1f}")
    print(f"Duración: {results['duration']:.1f} segundos")
    print(f"Tasa de mensajes: {results['message_rate']} msgs/s")
    print(f"Streams paralelos: {results['parallel_streams']}")
    print("-"*80)
    print(f"Eventos generados: {results['events_generated']}")
    print(f"Eventos recibidos por event bus: {results['events_received']}")
    print(f"Eventos procesados por componentes: {results['component_events_processed']}")
    print(f"Tasa de éxito: {results['success_rate']:.2f}%")
    print("-"*80)
    print(f"Latencia mínima: {results['latency']['min']*1000:.2f} ms")
    print(f"Latencia máxima: {results['latency']['max']*1000:.2f} ms")
    print(f"Latencia promedio: {results['latency']['avg']*1000:.2f} ms")
    print("-"*80)
    print(f"Errores: {results['errors']}")
    print(f"Errores transmutados: {results['event_bus_stats']['errors_transmuted']}")
    print("="*80)
    
    # Calcular tasas finales
    msgs_per_sec = results['events_generated'] / results['duration'] if results['duration'] > 0 else 0
    print(f"Rendimiento efectivo: {msgs_per_sec:.2f} mensajes por segundo")
    print("="*80)

async def run_test_suite(intensities: List[float], parallel_streams: int = 5, duration: int = 60) -> Dict[str, Any]:
    """
    Ejecutar suite completa de pruebas con diferentes intensidades.
    
    Args:
        intensities: Lista de intensidades a probar
        parallel_streams: Número de streams paralelos
        duration: Duración de cada prueba en segundos
        
    Returns:
        Resultados completos de todas las pruebas
    """
    results = {
        "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": []
    }
    
    for intensity in intensities:
        # Calcular tasa de mensajes según intensidad
        rate = int(DEFAULT_MESSAGE_RATE * min(intensity, 10))
        
        logger.info(f"Ejecutando prueba con intensidad {intensity:.1f}, tasa {rate} msgs/s")
        
        # Ejecutar prueba
        test_result = await run_integration_test(
            intensity=intensity,
            rate=rate,
            duration=duration,
            parallel_streams=parallel_streams
        )
        
        # Añadir a resultados
        results["tests"].append(test_result)
        
        # Imprimir resumen
        print_results_summary(test_result)
        
        # Breve pausa entre pruebas
        await asyncio.sleep(5)
    
    # Añadir resumen global
    results["summary"] = {
        "total_tests": len(results["tests"]),
        "average_success_rate": sum(t["success_rate"] for t in results["tests"]) / len(results["tests"]) if results["tests"] else 0,
        "max_throughput": max(t["events_generated"] / t["duration"] for t in results["tests"]) if results["tests"] else 0
    }
    
    return results

def main():
    """Función principal."""
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Prueba de integración del WebSocket Trascendental")
    parser.add_argument("--intensity", type=float, default=DEFAULT_INTENSITY,
                        help=f"Intensidad de la prueba (1.0 - 1000.0, por defecto: {DEFAULT_INTENSITY})")
    parser.add_argument("--duration", type=int, default=DEFAULT_TEST_DURATION,
                        help=f"Duración de la prueba en segundos (por defecto: {DEFAULT_TEST_DURATION})")
    parser.add_argument("--rate", type=int, default=DEFAULT_MESSAGE_RATE,
                        help=f"Tasa de mensajes por segundo (por defecto: {DEFAULT_MESSAGE_RATE})")
    parser.add_argument("--streams", type=int, default=DEFAULT_PARALLEL_STREAMS,
                        help=f"Número de streams paralelos (por defecto: {DEFAULT_PARALLEL_STREAMS})")
    parser.add_argument("--suite", action="store_true",
                        help="Ejecutar suite completa de pruebas con diferentes intensidades")
    
    args = parser.parse_args()
    
    try:
        if args.suite:
            # Ejecutar suite completa
            intensities = [1.0, 10.0, 100.0, 1000.0]
            results = asyncio.run(run_test_suite(
                intensities=intensities,
                parallel_streams=args.streams,
                duration=args.duration
            ))
            
            # Guardar resultados
            save_results(results, "resultados_suite_integracion_websocket.json")
            
            # Imprimir resumen final
            print("\n" + "="*80)
            print(f"RESUMEN FINAL DE SUITE DE PRUEBAS")
            print("="*80)
            print(f"Pruebas ejecutadas: {results['summary']['total_tests']}")
            print(f"Tasa de éxito promedio: {results['summary']['average_success_rate']:.2f}%")
            print(f"Rendimiento máximo: {results['summary']['max_throughput']:.2f} msgs/s")
            print("="*80)
            
        else:
            # Ejecutar prueba individual
            results = asyncio.run(run_integration_test(
                intensity=args.intensity,
                rate=args.rate,
                duration=args.duration,
                parallel_streams=args.streams
            ))
            
            # Guardar y mostrar resultados
            save_results(results)
            print_results_summary(results)
        
    except KeyboardInterrupt:
        logger.info("Prueba interrumpida por el usuario")
        
    except Exception as e:
        logger.error(f"Error en prueba: {e}")
        raise

if __name__ == "__main__":
    main()
"""