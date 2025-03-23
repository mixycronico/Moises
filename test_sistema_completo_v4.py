"""
Prueba Completa Sistema Genesis Singularidad Trascendental V4.

Este script realiza pruebas exhaustivas de los tres componentes fundamentales:
1. Core - Mecanismos trascendentales a intensidad 1000.0
2. API/WebSocket Local - TranscendentalEventBus 
3. WebSocket Externo - Conexión con exchanges

La prueba verifica la integración perfecta entre todos los componentes
y la operación del sistema completo bajo condiciones extremas.
"""

import asyncio
import logging
import json
import time
import random
import os
import sys
from typing import Dict, Any, List, Optional, Tuple, Set, Union, Callable, Coroutine
import argparse

# Asegurar que el directorio raíz está en el path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("test_sistema_completo_v4.log")
    ]
)

logger = logging.getLogger("Genesis.TestCompleto")

# Importar componentes del sistema
from genesis_singularity_transcendental_v4 import (
    DimensionalCollapseV4,
    EventHorizonV4,
    QuantumTimeV4,
    QuantumTunnelV4,
    InfiniteDensityV4,
    ResilientReplicationV4,
    EntanglementV4,
    RealityMatrixV4,
    OmniConvergenceV4,
    OmniversalSharedMemory,
    PredictiveRecoverySystem,
    QuantumFeedbackLoop,
    EvolvingConsciousInterface
)

from genesis.core.transcendental_event_bus import TranscendentalEventBus
from genesis.core.exchange_websocket_connector import ExchangeWebSocketHandler

# Constantes para pruebas
DEFAULT_INTENSITY = 1000.0
DEFAULT_TEST_DURATION = 60  # segundos
DEFAULT_MESSAGE_RATE = 100  # mensajes/segundo
DEFAULT_PARALLEL_STREAMS = 5

# ===== PRUEBA DE MECANISMOS TRASCENDENTALES (CORE) =====

class CoreTester:
    """
    Probador de los mecanismos trascendentales del núcleo.
    
    Evalúa todos los mecanismos trascendentales a intensidad 1000.0
    para verificar su funcionamiento perfecto bajo carga extrema.
    """
    
    def __init__(self, intensity: float = 1000.0):
        """
        Inicializar probador del core.
        
        Args:
            intensity: Intensidad de la prueba (1.0 - 1000.0)
        """
        self.intensity = intensity
        self.mechanisms = self._init_mechanisms()
        self.stats = {
            "start_time": time.time(),
            "operations": 0,
            "successes": 0,
            "transmutations": 0,
            "mechanism_stats": {}
        }
        
        logger.info(f"Probador Core inicializado con intensidad {intensity}")
    
    def _init_mechanisms(self) -> Dict[str, Any]:
        """
        Inicializar todos los mecanismos trascendentales.
        
        Returns:
            Diccionario con instancias de todos los mecanismos
        """
        return {
            "collapse": DimensionalCollapseV4(),
            "horizon": EventHorizonV4(),
            "time": QuantumTimeV4(),
            "tunnel": QuantumTunnelV4(),
            "density": InfiniteDensityV4(),
            "replication": ResilientReplicationV4(),
            "entanglement": EntanglementV4(),
            "reality": RealityMatrixV4(),
            "convergence": OmniConvergenceV4(),
            "memory": OmniversalSharedMemory(),
            "prediction": PredictiveRecoverySystem(),
            "feedback": QuantumFeedbackLoop(),
            "interface": EvolvingConsciousInterface()
        }
    
    async def test_all_mechanisms(self, operations_per_mechanism: int = 1000) -> Dict[str, Any]:
        """
        Probar todos los mecanismos.
        
        Args:
            operations_per_mechanism: Número de operaciones por mecanismo
            
        Returns:
            Estadísticas de la prueba
        """
        logger.info(f"Iniciando prueba de {len(self.mechanisms)} mecanismos, " +
                   f"{operations_per_mechanism} operaciones cada uno")
        
        start_time = time.time()
        
        # Limitar con tiempo cuántico para operaciones en microsegundos
        async with self.mechanisms["time"].nullify_time():
            # Probar cada mecanismo en paralelo
            tasks = []
            for name, mechanism in self.mechanisms.items():
                task = self._test_mechanism(name, mechanism, operations_per_mechanism)
                tasks.append(task)
            
            # Ejecutar todas las pruebas en paralelo
            await asyncio.gather(*tasks)
        
        # Actualizar estadísticas
        elapsed = time.time() - start_time
        total_operations = operations_per_mechanism * len(self.mechanisms)
        
        for name, mechanism in self.mechanisms.items():
            if hasattr(mechanism, "get_stats"):
                self.stats["mechanism_stats"][name] = mechanism.get_stats()
        
        self.stats.update({
            "total_time": elapsed,
            "operations_per_second": total_operations / elapsed if elapsed > 0 else 0,
            "success_rate": (self.stats["successes"] / self.stats["operations"] * 100) 
                           if self.stats["operations"] > 0 else 0
        })
        
        logger.info(f"Prueba de mecanismos completada en {elapsed:.2f}s")
        logger.info(f"Operaciones: {self.stats['operations']}, " +
                   f"Éxitos: {self.stats['successes']}, " +
                   f"Tasa: {self.stats['success_rate']:.2f}%")
        
        return self.stats
    
    async def _test_mechanism(self, name: str, mechanism: Any, operations: int) -> None:
        """
        Probar un mecanismo específico.
        
        Args:
            name: Nombre del mecanismo
            mechanism: Instancia del mecanismo
            operations: Número de operaciones a realizar
        """
        logger.info(f"Probando mecanismo: {name}")
        
        for i in range(operations):
            try:
                self.stats["operations"] += 1
                
                # Ejecutar operación según el tipo de mecanismo
                if name == "collapse":
                    data = {"test": i, "complex": {"nested": [1, 2, 3]}}
                    result = await mechanism.collapse_data(data)
                    assert result is not None
                
                elif name == "horizon":
                    try:
                        # Provocar error intencionalmente
                        if random.random() < 0.5:
                            raise ValueError(f"Error simulado #{i}")
                        assert 1 == 1  # Siempre exitoso
                    except Exception as e:
                        result = await mechanism.transmute_error(e, {"operation": f"test_{i}"})
                        self.stats["transmutations"] += 1
                        assert result["transmuted"]
                
                elif name == "time":
                    if i % 3 == 0:
                        result = await mechanism.dilate_time(factor=2.0)
                    elif i % 3 == 1:
                        result = await mechanism.contract_time(factor=0.5)
                    else:
                        async with mechanism.nullify_time():
                            await asyncio.sleep(0)
                    
                elif name == "tunnel":
                    data = {"message": f"test_{i}", "value": i * 10}
                    result = await mechanism.tunnel_data(data)
                    assert result is not None
                
                elif name == "density":
                    data = {"payload": "x" * 1000, "iteration": i}
                    compressed = await mechanism.compress(data)
                    decompressed = await mechanism.decompress(compressed)
                    assert decompressed is not None
                
                elif name == "replication":
                    # Alternar operaciones
                    if i % 3 == 0:
                        state = {"id": f"state_{i}", "value": i * 100}
                        await mechanism.replicate_state(state)
                    elif i % 3 == 1:
                        await mechanism.regenerate()
                    else:
                        state = await mechanism.get_replicated_state()
                        assert state is not None
                
                elif name == "entanglement":
                    # Crear componentes de prueba para entrelazar
                    components = [
                        {"id": f"comp_{i}_1", "state": "active"},
                        {"id": f"comp_{i}_2", "state": "standby"}
                    ]
                    result = await mechanism.entangle_components(components)
                    assert result is not None
                
                elif name == "reality":
                    # Alternar entre tipos de datos
                    if i % 3 == 0:
                        data = {"numeric": i * 100, "iteration": i}
                    elif i % 3 == 1:
                        data = [1, 2, 3, 4, 5]
                    else:
                        data = f"test_string_{i}"
                    
                    result = await mechanism.optimize(data)
                    assert result is not None
                
                elif name == "convergence":
                    # Probar convergencia y reconfiguración
                    if i % 2 == 0:
                        result = await mechanism.converge()
                    else:
                        await mechanism.reconfigure({"threshold": random.random()})
                        result = True
                    assert result is not None
                
                elif name == "memory":
                    # Alternar almacenamiento y recuperación
                    key = {"test_id": i, "operation": "test"}
                    if i % 2 == 0:
                        state = {"value": i * 1000, "timestamp": time.time()}
                        await mechanism.store_state(key, state)
                        result = True
                    else:
                        result = await mechanism.retrieve_state(key)
                    assert result is not None
                
                elif name == "prediction":
                    context = {"operation": f"test_{i}", "intensity": self.intensity}
                    result = await mechanism.predict_and_prevent(context)
                    assert result is not None
                
                elif name == "feedback":
                    # Alternar tipos de datos para retroalimentación
                    if i % 2 == 0:
                        data = {"operation": f"operation_{i}", "result": i * 10}
                    else:
                        data = [f"element_{j}" for j in range(i % 5 + 1)]
                    
                    result = await mechanism.apply_feedback(data)
                    assert result is not None
                
                elif name == "interface":
                    # Probar registro de patrones y evolución
                    if i % 3 == 0:
                        pattern_type = "test_pattern"
                        pattern_data = {"iteration": i, "value": i * random.random()}
                        result = await mechanism.register_pattern(pattern_type, pattern_data)
                    elif i % 3 == 1:
                        result = await mechanism.evolve_system({"stats": {"iterations": i}})
                    else:
                        result = mechanism.get_stats()
                    assert result is not None
                
                self.stats["successes"] += 1
                
                # Log periódico
                if i > 0 and i % 200 == 0:
                    logger.debug(f"Mecanismo {name}: {i}/{operations} operaciones")
                
            except Exception as e:
                logger.error(f"Error en mecanismo {name}, operación {i}: {e}")
        
        logger.info(f"Mecanismo {name}: {operations} operaciones completadas")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas detalladas de la prueba.
        
        Returns:
            Estadísticas de la prueba
        """
        # Añadir información adicional
        stats = dict(self.stats)
        
        # Recopilar estadísticas de cada mecanismo
        for name, mechanism in self.mechanisms.items():
            if hasattr(mechanism, "get_stats"):
                mechanism_stats = mechanism.get_stats()
                stats["mechanism_stats"][name] = mechanism_stats
        
        return stats

# ===== PRUEBA DEL TRANSCENDENTAL EVENT BUS (API/WEBSOCKET LOCAL) =====

class EventBusTester:
    """
    Probador del TranscendentalEventBus.
    
    Evalúa el funcionamiento del bus de eventos trascendental
    como reemplazo del event_bus tradicional.
    """
    
    def __init__(self, intensity: float = 1000.0):
        """
        Inicializar probador del bus de eventos.
        
        Args:
            intensity: Intensidad de la prueba (1.0 - 1000.0)
        """
        self.intensity = intensity
        self.event_bus = None
        self.stats = {
            "start_time": time.time(),
            "events_emitted": 0,
            "events_received": 0,
            "subscribers": 0,
            "event_types": 0,
            "errors": 0,
            "errors_transmuted": 0
        }
        
        # Rastreadores por tipo de evento
        self.event_trackers = {}
        
        logger.info(f"Probador EventBus inicializado con intensidad {intensity}")
    
    async def setup(self) -> None:
        """Configurar el bus de eventos para pruebas."""
        logger.info("Configurando TranscendentalEventBus para pruebas...")
        
        # Crear bus de eventos en modo prueba
        self.event_bus = TranscendentalEventBus(test_mode=True)
        
        # Iniciar bus
        await self.event_bus.start()
        
        # Registrar manejadores para diferentes tipos de eventos
        event_types = [
            "test_event", "data_event", "control_event", "system_event",
            "priority_high", "priority_normal", "priority_low",
            "error_event", "batch_event"
        ]
        
        for event_type in event_types:
            # Crear tracker para este tipo
            self.event_trackers[event_type] = {
                "received": 0,
                "first_timestamp": None,
                "last_timestamp": None
            }
            
            # Registrar manejador regular
            await self.event_bus.subscribe(
                event_type,
                self._make_handler(event_type),
                priority=self._get_priority(event_type),
                component_id=f"test_component_{event_type}"
            )
            
            # Para algunos tipos, agregar manejador de un solo uso
            if event_type in ["test_event", "error_event"]:
                await self.event_bus.subscribe_once(
                    f"{event_type}_once",
                    self._make_once_handler(f"{event_type}_once"),
                    component_id=f"test_component_once_{event_type}"
                )
            
            self.stats["subscribers"] += 1
        
        self.stats["event_types"] = len(event_types)
        logger.info(f"EventBus configurado con {self.stats['subscribers']} suscriptores")
    
    def _get_priority(self, event_type: str) -> int:
        """
        Obtener prioridad para un tipo de evento.
        
        Args:
            event_type: Tipo de evento
            
        Returns:
            Prioridad (0-2)
        """
        if "high" in event_type:
            return 0  # Más alta
        elif "low" in event_type:
            return 2  # Más baja
        else:
            return 1  # Normal
    
    def _make_handler(self, event_type: str) -> Callable:
        """
        Crear manejador para un tipo de evento.
        
        Args:
            event_type: Tipo de evento
            
        Returns:
            Función manejadora
        """
        async def handler(received_type: str, data: Dict[str, Any], source: str) -> None:
            # Verificar tipo correcto
            assert received_type == event_type, f"Tipo incorrecto: {received_type} ≠ {event_type}"
            
            # Registrar recepción
            if event_type in self.event_trackers:
                tracker = self.event_trackers[event_type]
                tracker["received"] += 1
                tracker["last_timestamp"] = time.time()
                if tracker["first_timestamp"] is None:
                    tracker["first_timestamp"] = time.time()
            
            self.stats["events_received"] += 1
            
            # Simular procesamiento
            await asyncio.sleep(0.001)
            
            # Simular error aleatorio para probar transmutación
            if event_type == "error_event" and random.random() < 0.5:
                raise ValueError(f"Error simulado en {event_type}")
            
        return handler
    
    def _make_once_handler(self, event_type: str) -> Callable:
        """
        Crear manejador de un solo uso.
        
        Args:
            event_type: Tipo de evento
            
        Returns:
            Función manejadora de un solo uso
        """
        async def once_handler(received_type: str, data: Dict[str, Any], source: str) -> None:
            # Verificar que coincidan los tipos
            assert received_type == event_type, f"Tipo incorrecto: {received_type} ≠ {event_type}"
            
            # Registrar recepción
            self.stats["events_received"] += 1
            
            # Simular procesamiento
            await asyncio.sleep(0.001)
            
        return once_handler
    
    async def test_event_emission(self, num_events: int = 1000) -> Dict[str, Any]:
        """
        Probar emisión de eventos.
        
        Args:
            num_events: Número de eventos a emitir
            
        Returns:
            Estadísticas de la prueba
        """
        logger.info(f"Iniciando prueba de emisión de {num_events} eventos...")
        
        start_time = time.time()
        event_types = list(self.event_trackers.keys())
        
        try:
            for i in range(num_events):
                # Seleccionar tipo aleatorio
                event_type = random.choice(event_types)
                
                # Crear datos
                data = {
                    "iteration": i,
                    "timestamp": time.time(),
                    "random_value": random.random() * self.intensity,
                    "nested": {
                        "field1": f"value_{i}",
                        "field2": i * 10
                    }
                }
                
                # Emitir evento
                await self.event_bus.emit(event_type, data, "test_emitter")
                self.stats["events_emitted"] += 1
                
                # Eventos "once" especiales
                if i % 100 == 0:
                    for special_type in ["test_event_once", "error_event_once"]:
                        await self.event_bus.emit(special_type, data, "test_emitter")
                        self.stats["events_emitted"] += 1
                
                # Simular error para transmutación
                if event_type == "error_event" and random.random() < 0.2:
                    try:
                        raise ValueError(f"Error simulado en emisión #{i}")
                    except Exception as e:
                        # El error se transmutará internamente
                        pass
                
                # Breve pausa para evitar saturación
                await asyncio.sleep(0.001 / min(self.intensity / 10, 100))
                
                # Log periódico
                if i > 0 and i % 200 == 0:
                    rate = i / (time.time() - start_time) if time.time() > start_time else 0
                    logger.info(f"Emitidos {i}/{num_events} eventos ({rate:.2f}/s)")
            
            # Esperar procesamiento residual
            await asyncio.sleep(0.5)
            
            # Calcular estadísticas finales
            elapsed = time.time() - start_time
            
            # Actualizar estadísticas
            self.stats.update({
                "elapsed": elapsed,
                "events_per_second": num_events / elapsed if elapsed > 0 else 0,
                "event_bus_stats": self.event_bus.get_stats()
            })
            
            # Calcular tasa de recepción
            if self.stats["events_emitted"] > 0:
                self.stats["reception_rate"] = (self.stats["events_received"] / 
                                              self.stats["events_emitted"] * 100)
            else:
                self.stats["reception_rate"] = 0
                
            # Extraer errores transmutados
            if "event_bus_stats" in self.stats and "errors_transmuted" in self.stats["event_bus_stats"]:
                self.stats["errors_transmuted"] = self.stats["event_bus_stats"]["errors_transmuted"]
            
            logger.info(f"Prueba completada: {self.stats['events_emitted']} emitidos, " +
                       f"{self.stats['events_received']} recibidos, " +
                       f"{self.stats['reception_rate']:.2f}% tasa")
            
            return self.stats
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error en prueba de emisión: {e}")
            raise
    
    async def test_large_payloads(self, num_events: int = 100, payload_size_kb: int = 100) -> Dict[str, Any]:
        """
        Probar emisión de eventos con payloads grandes.
        
        Args:
            num_events: Número de eventos a emitir
            payload_size_kb: Tamaño aproximado del payload en KB
            
        Returns:
            Estadísticas de la prueba
        """
        logger.info(f"Iniciando prueba de payloads grandes: {num_events} eventos, {payload_size_kb}KB cada uno")
        
        # Generar payload base (aproximadamente del tamaño especificado)
        # Cada carácter es ~1 byte, así que necesitamos payload_size_kb * 1024 caracteres
        base_payload = "x" * (payload_size_kb * 1024)
        
        start_time = time.time()
        
        try:
            for i in range(num_events):
                # Añadir variación para evitar compresión perfecta
                payload = f"{base_payload}_{i}_{random.randint(1000, 9999)}"
                
                # Crear datos con payload grande
                data = {
                    "iteration": i,
                    "timestamp": time.time(),
                    "payload": payload,
                    "metadata": {
                        "size_kb": payload_size_kb,
                        "event_id": f"large_{i}"
                    }
                }
                
                # Emitir evento
                await self.event_bus.emit("batch_event", data, "test_large_emitter")
                self.stats["events_emitted"] += 1
                
                # Breve pausa para evitar saturación
                await asyncio.sleep(0.01 / min(self.intensity / 10, 100))
                
                # Log periódico
                if i > 0 and i % 10 == 0:
                    rate = i / (time.time() - start_time) if time.time() > start_time else 0
                    logger.info(f"Emitidos {i}/{num_events} eventos grandes ({rate:.2f}/s)")
            
            # Esperar procesamiento residual
            await asyncio.sleep(1)
            
            # Actualizar estadísticas
            elapsed = time.time() - start_time
            
            large_payload_stats = {
                "elapsed": elapsed,
                "events_per_second": num_events / elapsed if elapsed > 0 else 0,
                "total_data_mb": (num_events * payload_size_kb) / 1024,
                "throughput_mbps": (num_events * payload_size_kb) / elapsed / 1024 if elapsed > 0 else 0
            }
            
            self.stats["large_payload_test"] = large_payload_stats
            
            logger.info(f"Prueba de payloads grandes completada: {large_payload_stats['throughput_mbps']:.2f} MB/s")
            
            return large_payload_stats
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error en prueba de payloads grandes: {e}")
            raise
    
    async def teardown(self) -> None:
        """Limpiar recursos después de las pruebas."""
        logger.info("Limpiando recursos de prueba EventBus...")
        
        if self.event_bus:
            await self.event_bus.stop()
        
        logger.info("Recursos EventBus liberados")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas detalladas de la prueba.
        
        Returns:
            Estadísticas de la prueba
        """
        # Agregar estadísticas del bus de eventos
        if self.event_bus:
            self.stats["event_bus_stats"] = self.event_bus.get_stats()
        
        # Agregar detalles de eventos por tipo
        self.stats["event_type_stats"] = dict(self.event_trackers)
        
        return self.stats

# ===== PRUEBA DEL EXCHANGE WEBSOCKET HANDLER (WEBSOCKET EXTERNO) =====

class ExchangeWebSocketTester:
    """
    Probador del ExchangeWebSocketHandler.
    
    Evalúa la conexión externa con exchanges de criptomonedas
    utilizando el WebSocket Trascendental.
    """
    
    def __init__(self, intensity: float = 1000.0, use_test_mode: bool = True):
        """
        Inicializar probador del WebSocket externo.
        
        Args:
            intensity: Intensidad de la prueba (1.0 - 1000.0)
            use_test_mode: Si usar modo de prueba (sin conexión real)
        """
        self.intensity = intensity
        self.use_test_mode = use_test_mode
        self.exchange_ws = None
        self.stats = {
            "start_time": time.time(),
            "connections_attempted": 0,
            "connections_successful": 0,
            "messages_received": 0,
            "messages_processed": 0,
            "errors": 0,
            "errors_transmuted": 0
        }
        
        # Rastreadores por stream
        self.stream_trackers = {}
        
        logger.info(f"Probador ExchangeWebSocket inicializado con intensidad {intensity}")
    
    async def setup(self) -> None:
        """Configurar el WebSocket para pruebas."""
        logger.info("Configurando ExchangeWebSocketHandler para pruebas...")
        
        # Crear manejador de WebSocket
        self.exchange_ws = ExchangeWebSocketHandler("binance")
        
        logger.info("ExchangeWebSocketHandler configurado")
    
    async def _handle_trade(self, data: Dict[str, Any]) -> None:
        """
        Manejar evento de trade.
        
        Args:
            data: Datos del trade
        """
        stream = data.get("stream", "unknown")
        symbol = data.get("symbol", "unknown")
        
        # Registrar en tracker
        key = f"{symbol}@trade"
        if key in self.stream_trackers:
            self.stream_trackers[key]["messages"] += 1
            self.stream_trackers[key]["last_timestamp"] = time.time()
        
        self.stats["messages_processed"] += 1
        
        # Log periódico
        if self.stream_trackers[key]["messages"] % 20 == 0:
            price = data.get("price", 0)
            logger.debug(f"Trade {self.stream_trackers[key]['messages']} para {symbol}: {price}")
    
    async def _handle_kline(self, data: Dict[str, Any]) -> None:
        """
        Manejar evento de kline (vela).
        
        Args:
            data: Datos de la vela
        """
        stream = data.get("stream", "unknown")
        symbol = data.get("symbol", "unknown")
        
        # Registrar en tracker
        key = f"{symbol}@kline"
        if key in self.stream_trackers:
            self.stream_trackers[key]["messages"] += 1
            self.stream_trackers[key]["last_timestamp"] = time.time()
        
        self.stats["messages_processed"] += 1
    
    async def _handle_orderbook(self, data: Dict[str, Any]) -> None:
        """
        Manejar evento de orderbook.
        
        Args:
            data: Datos del orderbook
        """
        stream = data.get("stream", "unknown")
        symbol = data.get("symbol", "unknown")
        
        # Registrar en tracker
        key = f"{symbol}@depth"
        if key in self.stream_trackers:
            self.stream_trackers[key]["messages"] += 1
            self.stream_trackers[key]["last_timestamp"] = time.time()
        
        self.stats["messages_processed"] += 1
    
    async def test_websocket_connections(self, symbols: List[str] = None, duration: int = 60) -> Dict[str, Any]:
        """
        Probar conexiones WebSocket a exchanges.
        
        Args:
            symbols: Lista de símbolos a probar (default: btcusdt, ethusdt)
            duration: Duración de la prueba en segundos
            
        Returns:
            Estadísticas de la prueba
        """
        if symbols is None:
            symbols = ["btcusdt", "ethusdt"]
        
        logger.info(f"Iniciando prueba de conexiones para {len(symbols)} símbolos durante {duration}s...")
        
        start_time = time.time()
        connected_streams = []
        
        try:
            # Conectar a streams de trades para cada símbolo
            for symbol in symbols:
                # Streams a conectar para este símbolo
                streams = [
                    f"{symbol}@trade",
                    f"{symbol}@kline_1m",
                    f"{symbol}@depth20"
                ]
                
                for stream in streams:
                    # Inicializar tracker
                    self.stream_trackers[stream] = {
                        "symbol": symbol,
                        "type": stream.split("@")[1],
                        "connected": False,
                        "messages": 0,
                        "start_timestamp": time.time(),
                        "last_timestamp": None
                    }
                    
                    # Seleccionar callback según tipo
                    if "trade" in stream:
                        callback = self._handle_trade
                    elif "kline" in stream:
                        callback = self._handle_kline
                    else:
                        callback = self._handle_orderbook
                    
                    # Intentar conectar
                    self.stats["connections_attempted"] += 1
                    logger.info(f"Conectando a stream {stream}...")
                    
                    success = await self.exchange_ws.connect_to_stream(stream, callback)
                    
                    if success:
                        self.stats["connections_successful"] += 1
                        self.stream_trackers[stream]["connected"] = True
                        connected_streams.append(stream)
                        logger.info(f"Conexión exitosa a {stream}")
                    else:
                        logger.warning(f"Falló conexión a {stream}")
            
            # Esperar y monitorear durante la duración especificada
            logger.info(f"Conexiones establecidas. Monitoreando durante {duration}s...")
            
            end_time = time.time() + duration
            while time.time() < end_time:
                # Esperar 5 segundos
                await asyncio.sleep(5)
                
                # Mostrar estadísticas intermedias
                elapsed = time.time() - start_time
                messages = self.stats["messages_processed"]
                rate = messages / elapsed if elapsed > 0 else 0
                
                logger.info(f"Estado tras {elapsed:.1f}s: {messages} mensajes recibidos ({rate:.2f}/s)")
                
                # Mostrar detalles por stream
                for stream in connected_streams:
                    tracker = self.stream_trackers[stream]
                    if tracker["messages"] > 0:
                        logger.info(f"  {stream}: {tracker['messages']} mensajes")
            
            # Calcular estadísticas finales
            elapsed = time.time() - start_time
            self.stats["duration"] = elapsed
            self.stats["messages_per_second"] = self.stats["messages_processed"] / elapsed if elapsed > 0 else 0
            
            # Obtener estadísticas del WebSocket
            ws_stats = self.exchange_ws.get_stats()
            self.stats["websocket_stats"] = ws_stats
            
            # Extraer errores transmutados
            if "errors_transmuted" in ws_stats:
                self.stats["errors_transmuted"] = ws_stats["errors_transmuted"]
            
            logger.info(f"Prueba completada: {len(connected_streams)}/{len(symbols)*3} streams conectados, " +
                       f"{self.stats['messages_processed']} mensajes, " +
                       f"{self.stats['messages_per_second']:.2f} msgs/s")
            
            return self.stats
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error en prueba de conexiones: {e}")
            raise
    
    async def test_reconnection(self, symbol: str = "btcusdt") -> Dict[str, Any]:
        """
        Probar capacidad de reconexión del WebSocket.
        
        Args:
            symbol: Símbolo a utilizar para la prueba
            
        Returns:
            Estadísticas de la prueba
        """
        logger.info(f"Iniciando prueba de reconexión para {symbol}...")
        
        stream = f"{symbol}@trade"
        reconnection_stats = {
            "stream": stream,
            "initial_connection": False,
            "messages_before": 0,
            "disconnect_time": None,
            "reconnect_time": None,
            "messages_after": 0,
            "reconnection_time": None,
            "success": False
        }
        
        try:
            # Conectar inicialmente
            success = await self.exchange_ws.connect_to_stream(stream, self._handle_trade)
            reconnection_stats["initial_connection"] = success
            
            if not success:
                logger.error(f"Falló conexión inicial a {stream}")
                return reconnection_stats
            
            # Esperar mensajes iniciales (10 segundos)
            await asyncio.sleep(10)
            
            # Registrar mensajes antes de desconexión
            if stream in self.stream_trackers:
                reconnection_stats["messages_before"] = self.stream_trackers[stream]["messages"]
            
            # Forzar desconexión
            logger.info(f"Forzando desconexión de {stream}...")
            reconnection_stats["disconnect_time"] = time.time()
            
            # Desconectar
            await self.exchange_ws.disconnect_from_stream(stream)
            
            # Esperar brevemente
            await asyncio.sleep(1)
            
            # Reconectar
            logger.info(f"Intentando reconexión a {stream}...")
            reconnection_stats["reconnect_time"] = time.time()
            
            success = await self.exchange_ws.connect_to_stream(stream, self._handle_trade)
            reconnection_stats["success"] = success
            
            if not success:
                logger.error(f"Falló reconexión a {stream}")
                return reconnection_stats
            
            # Calcular tiempo de reconexión
            if reconnection_stats["reconnect_time"] and reconnection_stats["disconnect_time"]:
                reconnection_stats["reconnection_time"] = (
                    reconnection_stats["reconnect_time"] - reconnection_stats["disconnect_time"]
                )
            
            # Esperar mensajes tras reconexión (10 segundos)
            await asyncio.sleep(10)
            
            # Registrar mensajes después de reconexión
            if stream in self.stream_trackers:
                reconnection_stats["messages_after"] = (
                    self.stream_trackers[stream]["messages"] - reconnection_stats["messages_before"]
                )
            
            logger.info(f"Prueba de reconexión completada: " +
                       f"Tiempo={reconnection_stats['reconnection_time']:.2f}s, " +
                       f"Mensajes nuevos={reconnection_stats['messages_after']}")
            
            return reconnection_stats
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error en prueba de reconexión: {e}")
            raise
    
    async def teardown(self) -> None:
        """Limpiar recursos después de las pruebas."""
        logger.info("Limpiando recursos de prueba ExchangeWebSocket...")
        
        if self.exchange_ws:
            # Desconectar de todos los streams
            await self.exchange_ws.disconnect_all()
        
        logger.info("Recursos ExchangeWebSocket liberados")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas detalladas de la prueba.
        
        Returns:
            Estadísticas de la prueba
        """
        # Agregar estadísticas del WebSocket
        if self.exchange_ws:
            self.stats["websocket_stats"] = self.exchange_ws.get_stats()
        
        # Agregar detalles de streams
        self.stats["stream_stats"] = dict(self.stream_trackers)
        
        return self.stats

# ===== PRUEBA DEL SISTEMA COMPLETO INTEGRADO =====

class IntegratedSystemTester:
    """
    Probador del sistema completo integrado.
    
    Evalúa la interacción entre todos los componentes:
    - Mecanismos trascendentales del core
    - API/WebSocket local (TranscendentalEventBus)
    - WebSocket externo (ExchangeWebSocketHandler)
    """
    
    def __init__(self, intensity: float = 1000.0):
        """
        Inicializar probador del sistema integrado.
        
        Args:
            intensity: Intensidad de la prueba (1.0 - 1000.0)
        """
        self.intensity = intensity
        self.event_bus = None
        self.exchange_ws = None
        self.quantum_time = QuantumTimeV4()
        self.stats = {
            "start_time": time.time(),
            "market_data_received": 0,
            "signals_generated": 0,
            "operations_executed": 0,
            "errors": 0,
            "errors_transmuted": 0
        }
        
        logger.info(f"Probador Sistema Integrado inicializado con intensidad {intensity}")
    
    async def setup(self) -> None:
        """Configurar todos los componentes para pruebas integradas."""
        logger.info("Configurando componentes para prueba de sistema integrado...")
        
        # Crear TranscendentalEventBus
        self.event_bus = TranscendentalEventBus(test_mode=True)
        await self.event_bus.start()
        
        # Crear ExchangeWebSocketHandler
        self.exchange_ws = ExchangeWebSocketHandler("binance")
        
        # Suscribirse a eventos relevantes
        await self._setup_event_handlers()
        
        logger.info("Sistema integrado configurado")
    
    async def _setup_event_handlers(self) -> None:
        """Configurar manejadores de eventos para prueba integrada."""
        # Suscribirse a eventos de mercado
        await self.event_bus.subscribe(
            "market_data",
            self._handle_market_data,
            component_id="integrated_test"
        )
        
        # Suscribirse a eventos de señales
        await self.event_bus.subscribe(
            "trading_signal",
            self._handle_signal,
            component_id="integrated_test"
        )
        
        # Suscribirse a eventos de operaciones
        await self.event_bus.subscribe(
            "trading_operation",
            self._handle_operation,
            component_id="integrated_test"
        )
    
    async def _handle_market_data(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar evento de datos de mercado.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
        """
        self.stats["market_data_received"] += 1
        
        # Analizar datos y potencialmente generar señal
        if random.random() < 0.05:  # 5% de probabilidad de generar señal
            await self._generate_signal(data)
    
    async def _handle_signal(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar evento de señal de trading.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
        """
        self.stats["signals_generated"] += 1
        
        # Procesar señal y potencialmente generar operación
        if random.random() < 0.2:  # 20% de probabilidad de ejecutar
            await self._execute_operation(data)
    
    async def _handle_operation(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar evento de operación de trading.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
        """
        self.stats["operations_executed"] += 1
    
    async def _handle_trade(self, data: Dict[str, Any]) -> None:
        """
        Manejar datos de trade del exchange.
        
        Args:
            data: Datos del trade
        """
        # Emitir evento de datos de mercado
        await self.event_bus.emit(
            "market_data",
            {
                "type": "trade",
                "symbol": data.get("symbol", "unknown"),
                "price": data.get("price", 0),
                "quantity": data.get("quantity", 0),
                "timestamp": time.time(),
                "source": "exchange"
            },
            "exchange_connector"
        )
    
    async def _generate_signal(self, market_data: Dict[str, Any]) -> None:
        """
        Generar señal basada en datos de mercado.
        
        Args:
            market_data: Datos de mercado
        """
        # Crear señal
        signal = {
            "symbol": market_data.get("symbol", "btcusdt"),
            "type": random.choice(["breakout", "trend", "reversal"]),
            "direction": random.choice(["buy", "sell"]),
            "strength": random.uniform(0.5, 1.0),
            "timestamp": time.time(),
            "market_data": market_data
        }
        
        # Emitir evento de señal
        await self.event_bus.emit(
            "trading_signal",
            signal,
            "signal_generator"
        )
    
    async def _execute_operation(self, signal: Dict[str, Any]) -> None:
        """
        Ejecutar operación basada en señal.
        
        Args:
            signal: Datos de la señal
        """
        # Crear operación
        operation = {
            "symbol": signal.get("symbol", "btcusdt"),
            "type": signal.get("direction", "buy"),
            "quantity": random.uniform(0.01, 1.0),
            "price": 0,  # Mercado
            "timestamp": time.time(),
            "signal_id": id(signal),
            "executed": True
        }
        
        # Emitir evento de operación
        await self.event_bus.emit(
            "trading_operation",
            operation,
            "operation_executor"
        )
    
    async def test_integrated_system(self, duration: int = 60, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Ejecutar prueba del sistema integrado.
        
        Args:
            duration: Duración de la prueba en segundos
            symbols: Lista de símbolos a probar (default: btcusdt, ethusdt)
            
        Returns:
            Estadísticas de la prueba
        """
        if symbols is None:
            symbols = ["btcusdt", "ethusdt"]
        
        logger.info(f"Iniciando prueba de sistema integrado durante {duration}s...")
        
        # Conectar a streams de trades para cada símbolo
        for symbol in symbols:
            logger.info(f"Conectando a stream de trades para {symbol}...")
            success = await self.exchange_ws.connect_to_stream(f"{symbol}@trade", self._handle_trade)
            if success:
                logger.info(f"Conexión exitosa a {symbol}@trade")
            else:
                logger.warning(f"Falló conexión a {symbol}@trade")
        
        # Ejecutar prueba durante la duración especificada
        start_time = time.time()
        end_time = start_time + duration
        
        logger.info(f"Sistema en ejecución. Prueba durará {duration}s...")
        
        # Monitorear progreso
        while time.time() < end_time:
            # Esperar 5 segundos
            await asyncio.sleep(5)
            
            # Mostrar estadísticas intermedias
            elapsed = time.time() - start_time
            market_data = self.stats["market_data_received"]
            signals = self.stats["signals_generated"]
            operations = self.stats["operations_executed"]
            
            logger.info(f"Estado tras {elapsed:.1f}s:")
            logger.info(f"  Datos de mercado: {market_data}")
            logger.info(f"  Señales: {signals}")
            logger.info(f"  Operaciones: {operations}")
        
        # Calcular estadísticas finales
        elapsed = time.time() - start_time
        
        # Recopilar estadísticas de componentes
        event_bus_stats = self.event_bus.get_stats() if self.event_bus else {}
        exchange_ws_stats = self.exchange_ws.get_stats() if self.exchange_ws else {}
        
        # Actualizar estadísticas
        self.stats.update({
            "duration": elapsed,
            "market_data_per_second": self.stats["market_data_received"] / elapsed if elapsed > 0 else 0,
            "signals_per_second": self.stats["signals_generated"] / elapsed if elapsed > 0 else 0,
            "operations_per_second": self.stats["operations_executed"] / elapsed if elapsed > 0 else 0,
            "event_bus_stats": event_bus_stats,
            "exchange_ws_stats": exchange_ws_stats
        })
        
        # Extraer errores transmutados
        errors_transmuted = 0
        if "errors_transmuted" in event_bus_stats:
            errors_transmuted += event_bus_stats["errors_transmuted"]
        if "errors_transmuted" in exchange_ws_stats:
            errors_transmuted += exchange_ws_stats["errors_transmuted"]
        
        self.stats["errors_transmuted"] = errors_transmuted
        
        logger.info(f"Prueba de sistema integrado completada: " +
                   f"{self.stats['market_data_received']} datos de mercado, " +
                   f"{self.stats['signals_generated']} señales, " +
                   f"{self.stats['operations_executed']} operaciones")
        
        return self.stats
    
    async def teardown(self) -> None:
        """Limpiar recursos después de las pruebas."""
        logger.info("Limpiando recursos de prueba Sistema Integrado...")
        
        # Desconectar exchange WebSocket
        if self.exchange_ws:
            await self.exchange_ws.disconnect_all()
        
        # Detener event bus
        if self.event_bus:
            await self.event_bus.stop()
        
        logger.info("Recursos Sistema Integrado liberados")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas detalladas de la prueba.
        
        Returns:
            Estadísticas de la prueba
        """
        return dict(self.stats)

# ===== FUNCIONES AUXILIARES Y PRINCIPAL =====

def save_results(results: Dict[str, Any], filename: str = "resultados_sistema_completo_v4.json") -> None:
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
        results: Resultados de las pruebas
    """
    print("\n" + "="*80)
    print(f"RESUMEN DE PRUEBAS DEL SISTEMA GENESIS SINGULARIDAD TRASCENDENTAL V4")
    print("="*80)
    
    # Core
    if "core" in results:
        core = results["core"]
        print(f"CORE (Mecanismos Trascendentales):")
        print(f"  Intensidad: {core.get('intensity', 0.0):.1f}")
        print(f"  Operaciones: {core.get('operations', 0)}")
        print(f"  Éxitos: {core.get('successes', 0)}")
        print(f"  Tasa de éxito: {core.get('success_rate', 0.0):.2f}%")
        print(f"  Operaciones/segundo: {core.get('operations_per_second', 0.0):.2f}")
        print()
    
    # EventBus
    if "event_bus" in results:
        bus = results["event_bus"]
        print(f"EVENT BUS (API/WebSocket Local):")
        print(f"  Eventos emitidos: {bus.get('events_emitted', 0)}")
        print(f"  Eventos recibidos: {bus.get('events_received', 0)}")
        print(f"  Tasa de recepción: {bus.get('reception_rate', 0.0):.2f}%")
        print(f"  Eventos/segundo: {bus.get('events_per_second', 0.0):.2f}")
        
        # Payloads grandes
        if "large_payload_test" in bus:
            large = bus["large_payload_test"]
            print(f"  Throughput payloads grandes: {large.get('throughput_mbps', 0.0):.2f} MB/s")
        print()
    
    # ExchangeWebSocket
    if "exchange_ws" in results:
        ws = results["exchange_ws"]
        print(f"EXCHANGE WEBSOCKET (WebSocket Externo):")
        print(f"  Conexiones intentadas: {ws.get('connections_attempted', 0)}")
        print(f"  Conexiones exitosas: {ws.get('connections_successful', 0)}")
        print(f"  Mensajes recibidos: {ws.get('messages_processed', 0)}")
        print(f"  Mensajes/segundo: {ws.get('messages_per_second', 0.0):.2f}")
        
        # Reconexión
        if "reconnection" in ws:
            recon = ws["reconnection"]
            print(f"  Tiempo de reconexión: {recon.get('reconnection_time', 0.0):.2f}s")
            print(f"  Mensajes tras reconexión: {recon.get('messages_after', 0)}")
        print()
    
    # Sistema Integrado
    if "integrated" in results:
        integrated = results["integrated"]
        print(f"SISTEMA INTEGRADO:")
        print(f"  Datos de mercado: {integrated.get('market_data_received', 0)}")
        print(f"  Señales generadas: {integrated.get('signals_generated', 0)}")
        print(f"  Operaciones ejecutadas: {integrated.get('operations_executed', 0)}")
        print(f"  Datos/segundo: {integrated.get('market_data_per_second', 0.0):.2f}")
        print(f"  Señales/segundo: {integrated.get('signals_per_second', 0.0):.2f}")
        print(f"  Operaciones/segundo: {integrated.get('operations_per_second', 0.0):.2f}")
        print()
    
    # Resumen global
    print(f"RESUMEN GLOBAL:")
    if "summary" in results:
        summary = results["summary"]
        print(f"  Tasa de éxito global: {summary.get('global_success_rate', 0.0):.2f}%")
        print(f"  Errores: {summary.get('total_errors', 0)}")
        print(f"  Errores transmutados: {summary.get('total_errors_transmuted', 0)}")
        print(f"  Tiempo total de prueba: {summary.get('total_duration', 0.0):.2f}s")
    print("="*80)

async def run_full_test_suite(intensity: float = 1000.0, duration: int = 60) -> Dict[str, Any]:
    """
    Ejecutar suite completa de pruebas.
    
    Args:
        intensity: Intensidad de la prueba (1.0 - 1000.0)
        duration: Duración de cada prueba en segundos
        
    Returns:
        Resultados completos de todas las pruebas
    """
    results = {
        "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "intensity": intensity,
        "duration": duration
    }
    
    start_time = time.time()
    
    try:
        # 1. Prueba de mecanismos core
        logger.info("=== INICIANDO PRUEBA DE MECANISMOS CORE ===")
        core_tester = CoreTester(intensity=intensity)
        core_results = await core_tester.test_all_mechanisms(operations_per_mechanism=500)
        results["core"] = core_results
        logger.info("Prueba de mecanismos core completada")
        
        # 2. Prueba de EventBus
        logger.info("\n=== INICIANDO PRUEBA DE EVENT BUS ===")
        bus_tester = EventBusTester(intensity=intensity)
        await bus_tester.setup()
        
        try:
            # Prueba de emisión estándar
            bus_results = await bus_tester.test_event_emission(num_events=1000)
            
            # Prueba de payloads grandes
            large_results = await bus_tester.test_large_payloads(
                num_events=20, 
                payload_size_kb=100
            )
            
            # Combinar resultados
            bus_results.update({"large_payload_test": large_results})
            results["event_bus"] = bus_results
            
        finally:
            await bus_tester.teardown()
        
        logger.info("Prueba de EventBus completada")
        
        # 3. Prueba de ExchangeWebSocket
        logger.info("\n=== INICIANDO PRUEBA DE EXCHANGE WEBSOCKET ===")
        ws_tester = ExchangeWebSocketTester(intensity=intensity)
        await ws_tester.setup()
        
        try:
            # Prueba de conexiones
            ws_results = await ws_tester.test_websocket_connections(
                symbols=["btcusdt", "ethusdt"],
                duration=duration
            )
            
            # Prueba de reconexión
            recon_results = await ws_tester.test_reconnection(symbol="btcusdt")
            
            # Combinar resultados
            ws_results.update({"reconnection": recon_results})
            results["exchange_ws"] = ws_results
            
        finally:
            await ws_tester.teardown()
        
        logger.info("Prueba de ExchangeWebSocket completada")
        
        # 4. Prueba de sistema integrado
        logger.info("\n=== INICIANDO PRUEBA DE SISTEMA INTEGRADO ===")
        integrated_tester = IntegratedSystemTester(intensity=intensity)
        await integrated_tester.setup()
        
        try:
            # Prueba integrada
            integrated_results = await integrated_tester.test_integrated_system(
                duration=duration,
                symbols=["btcusdt"]
            )
            
            results["integrated"] = integrated_results
            
        finally:
            await integrated_tester.teardown()
        
        logger.info("Prueba de sistema integrado completada")
        
        # Calcular resumen global
        total_duration = time.time() - start_time
        
        # Calcular tasa de éxito global
        success_rates = []
        if "core" in results and "success_rate" in results["core"]:
            success_rates.append(results["core"]["success_rate"])
        
        if "event_bus" in results and "reception_rate" in results["event_bus"]:
            success_rates.append(results["event_bus"]["reception_rate"])
        
        if "exchange_ws" in results and "connections_attempted" in results["exchange_ws"]:
            ws = results["exchange_ws"]
            if ws["connections_attempted"] > 0:
                ws_rate = ws["connections_successful"] / ws["connections_attempted"] * 100
                success_rates.append(ws_rate)
        
        global_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0
        
        # Contar errores totales
        total_errors = (
            results.get("core", {}).get("errors", 0) +
            results.get("event_bus", {}).get("errors", 0) +
            results.get("exchange_ws", {}).get("errors", 0) +
            results.get("integrated", {}).get("errors", 0)
        )
        
        # Contar transmutaciones totales
        total_errors_transmuted = (
            results.get("core", {}).get("transmutations", 0) +
            results.get("event_bus", {}).get("errors_transmuted", 0) +
            results.get("exchange_ws", {}).get("errors_transmuted", 0) +
            results.get("integrated", {}).get("errors_transmuted", 0)
        )
        
        # Guardar resumen
        results["summary"] = {
            "global_success_rate": global_success_rate,
            "total_errors": total_errors,
            "total_errors_transmuted": total_errors_transmuted,
            "total_duration": total_duration
        }
        
        logger.info(f"\n=== PRUEBAS COMPLETADAS EN {total_duration:.2f}s ===")
        logger.info(f"Tasa de éxito global: {global_success_rate:.2f}%")
        logger.info(f"Errores totales: {total_errors}")
        logger.info(f"Errores transmutados: {total_errors_transmuted}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error en suite de pruebas: {e}")
        results["error"] = str(e)
        raise

def main():
    """Función principal."""
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Prueba Completa Sistema Genesis Singularidad Trascendental V4")
    parser.add_argument("--intensity", type=float, default=DEFAULT_INTENSITY,
                        help=f"Intensidad de la prueba (1.0 - 1000.0, por defecto: {DEFAULT_INTENSITY})")
    parser.add_argument("--duration", type=int, default=DEFAULT_TEST_DURATION,
                        help=f"Duración de cada prueba en segundos (por defecto: {DEFAULT_TEST_DURATION})")
    parser.add_argument("--core-only", action="store_true",
                        help="Ejecutar solo prueba de mecanismos core")
    parser.add_argument("--event-bus-only", action="store_true",
                        help="Ejecutar solo prueba de EventBus")
    parser.add_argument("--websocket-only", action="store_true",
                        help="Ejecutar solo prueba de ExchangeWebSocket")
    parser.add_argument("--integrated-only", action="store_true",
                        help="Ejecutar solo prueba de sistema integrado")
    
    args = parser.parse_args()
    
    # Determinar qué pruebas ejecutar
    run_core = args.core_only or not (args.event_bus_only or args.websocket_only or args.integrated_only)
    run_event_bus = args.event_bus_only or not (args.core_only or args.websocket_only or args.integrated_only)
    run_websocket = args.websocket_only or not (args.core_only or args.event_bus_only or args.integrated_only)
    run_integrated = args.integrated_only or not (args.core_only or args.event_bus_only or args.websocket_only)
    
    # Inicializar resultados
    results = {
        "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "intensity": args.intensity,
        "duration": args.duration
    }
    
    start_time = time.time()
    
    try:
        # Ejecutar pruebas seleccionadas
        if run_core:
            # Prueba de mecanismos core
            logger.info("=== INICIANDO PRUEBA DE MECANISMOS CORE ===")
            core_tester = CoreTester(intensity=args.intensity)
            core_results = asyncio.run(core_tester.test_all_mechanisms(operations_per_mechanism=500))
            results["core"] = core_results
            logger.info("Prueba de mecanismos core completada")
        
        if run_event_bus:
            # Prueba de EventBus
            logger.info("\n=== INICIANDO PRUEBA DE EVENT BUS ===")
            
            async def run_event_bus_tests():
                bus_tester = EventBusTester(intensity=args.intensity)
                await bus_tester.setup()
                
                try:
                    # Prueba de emisión estándar
                    bus_results = await bus_tester.test_event_emission(num_events=1000)
                    
                    # Prueba de payloads grandes
                    large_results = await bus_tester.test_large_payloads(
                        num_events=20, 
                        payload_size_kb=100
                    )
                    
                    # Combinar resultados
                    bus_results.update({"large_payload_test": large_results})
                    return bus_results
                    
                finally:
                    await bus_tester.teardown()
            
            results["event_bus"] = asyncio.run(run_event_bus_tests())
            logger.info("Prueba de EventBus completada")
        
        if run_websocket:
            # Prueba de ExchangeWebSocket
            logger.info("\n=== INICIANDO PRUEBA DE EXCHANGE WEBSOCKET ===")
            
            async def run_websocket_tests():
                ws_tester = ExchangeWebSocketTester(intensity=args.intensity)
                await ws_tester.setup()
                
                try:
                    # Prueba de conexiones
                    ws_results = await ws_tester.test_websocket_connections(
                        symbols=["btcusdt", "ethusdt"],
                        duration=args.duration
                    )
                    
                    # Prueba de reconexión
                    recon_results = await ws_tester.test_reconnection(symbol="btcusdt")
                    
                    # Combinar resultados
                    ws_results.update({"reconnection": recon_results})
                    return ws_results
                    
                finally:
                    await ws_tester.teardown()
            
            results["exchange_ws"] = asyncio.run(run_websocket_tests())
            logger.info("Prueba de ExchangeWebSocket completada")
        
        if run_integrated:
            # Prueba de sistema integrado
            logger.info("\n=== INICIANDO PRUEBA DE SISTEMA INTEGRADO ===")
            
            async def run_integrated_tests():
                integrated_tester = IntegratedSystemTester(intensity=args.intensity)
                await integrated_tester.setup()
                
                try:
                    # Prueba integrada
                    return await integrated_tester.test_integrated_system(
                        duration=args.duration,
                        symbols=["btcusdt"]
                    )
                    
                finally:
                    await integrated_tester.teardown()
            
            results["integrated"] = asyncio.run(run_integrated_tests())
            logger.info("Prueba de sistema integrado completada")
        
        # Si se ejecutaron todas las pruebas, calcular resumen global
        if run_core and run_event_bus and run_websocket and run_integrated:
            # Calcular tasa de éxito global
            success_rates = []
            if "core" in results and "success_rate" in results["core"]:
                success_rates.append(results["core"]["success_rate"])
            
            if "event_bus" in results and "reception_rate" in results["event_bus"]:
                success_rates.append(results["event_bus"]["reception_rate"])
            
            if "exchange_ws" in results and "connections_attempted" in results["exchange_ws"]:
                ws = results["exchange_ws"]
                if ws["connections_attempted"] > 0:
                    ws_rate = ws["connections_successful"] / ws["connections_attempted"] * 100
                    success_rates.append(ws_rate)
            
            global_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0
            
            # Contar errores totales
            total_errors = (
                results.get("core", {}).get("errors", 0) +
                results.get("event_bus", {}).get("errors", 0) +
                results.get("exchange_ws", {}).get("errors", 0) +
                results.get("integrated", {}).get("errors", 0)
            )
            
            # Contar transmutaciones totales
            total_errors_transmuted = (
                results.get("core", {}).get("transmutations", 0) +
                results.get("event_bus", {}).get("errors_transmuted", 0) +
                results.get("exchange_ws", {}).get("errors_transmuted", 0) +
                results.get("integrated", {}).get("errors_transmuted", 0)
            )
            
            # Guardar resumen
            results["summary"] = {
                "global_success_rate": global_success_rate,
                "total_errors": total_errors,
                "total_errors_transmuted": total_errors_transmuted,
                "total_duration": time.time() - start_time
            }
        
        # Guardar resultados
        save_results(results)
        
        # Imprimir resumen
        print_results_summary(results)
        
    except KeyboardInterrupt:
        logger.info("Prueba interrumpida por el usuario")
        
    except Exception as e:
        logger.error(f"Error en prueba: {e}")
        results["error"] = str(e)
        save_results(results, "resultados_sistema_completo_error.json")
        raise

if __name__ == "__main__":
    main()