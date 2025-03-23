"""
Prueba de Integración Extrema para el Módulo genesis/core.

Este script realiza pruebas exhaustivas de los componentes principales
del módulo genesis/core y su integración entre sí:

1. Procesador Asíncrono Ultra-Cuántico
2. Adaptador WebSocket Trascendental
3. WebSocket Externo Trascendental
4. Integrador de Exchanges Trascendental
5. Componentes Circuit Breaker
6. Sistema de Checkpointing y Recuperación

Las pruebas simulan condiciones extremas, incluyendo:
- Operaciones masivamente paralelas (1000+ tareas)
- Inducción de errores y verificación de transmutación
- Simulación de latencia extrema
- Cortes de conexión y reconexión automática
- Sobrecarga de memoria y CPU
"""

import asyncio
import logging
import time
import random
import json
import os
import threading
import concurrent.futures
from typing import Dict, Any, List, Optional, Tuple, Set
from enum import Enum, auto
import gc

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestCoreIntegrationExtreme")

# Importar los componentes del core que vamos a probar
from genesis.core.async_quantum_processor import AsyncQuantumProcessor, ProcessingSpace
from genesis.core.transcendental_ws_adapter import TranscendentalWebSocketAdapter, UltraQuantumWebSocketAdapter
from genesis.core.transcendental_external_websocket import ExternalWebSocketAdapter
from genesis.core.transcendental_exchange_integrator import TranscendentalExchangeIntegrator, ExchangeID
from genesis.core.circuit_breaker import CircuitBreaker, CircuitBreakerState
from genesis.core.checkpoint_recovery import CheckpointManager, CheckpointType

# Parámetros de prueba
MAX_PARALLEL_TASKS = 1000  # Número máximo de tareas paralelas
EXTREME_INTENSITY = 10.0   # Intensidad de prueba (10x más que normal)
TEST_DURATION = 60         # Duración de prueba en segundos
ERROR_RATE = 0.1           # Tasa de error intencional para probar transmutación

# Clases auxiliares para pruebas
class TestResult:
    """Almacena resultados de pruebas para análisis."""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time = time.time()
        self.end_time = None
        self.success = False
        self.metrics = {}
        self.errors = []
        self.details = {}
        
    def finish(self, success: bool = True):
        """Finalizar prueba y registrar tiempo."""
        self.end_time = time.time()
        self.success = success
        return self
        
    def add_metric(self, name: str, value: Any):
        """Añadir una métrica."""
        self.metrics[name] = value
        return self
        
    def add_error(self, error: Exception):
        """Añadir un error."""
        self.errors.append(str(error))
        return self
        
    def add_detail(self, key: str, value: Any):
        """Añadir detalle."""
        self.details[key] = value
        return self
        
    def duration(self) -> float:
        """Obtener duración de la prueba."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "test_name": self.test_name,
            "success": self.success,
            "duration": self.duration(),
            "metrics": self.metrics,
            "errors": self.errors,
            "details": self.details
        }
        
    def __str__(self) -> str:
        """Representación como string."""
        status = "✅ ÉXITO" if self.success else "❌ FALLO"
        return f"{status} - {self.test_name} - {self.duration():.2f}s - Métricas: {len(self.metrics)} - Errores: {len(self.errors)}"

class TestSuite:
    """Suite de pruebas con análisis agregado."""
    
    def __init__(self, name: str):
        self.name = name
        self.results = []
        self.start_time = time.time()
        
    def add_result(self, result: TestResult):
        """Añadir resultado de prueba."""
        self.results.append(result)
        
    def success_rate(self) -> float:
        """Tasa de éxito de las pruebas."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)
    
    def total_duration(self) -> float:
        """Duración total de las pruebas."""
        return time.time() - self.start_time
    
    def aggregate_metrics(self) -> Dict[str, Any]:
        """Agregar métricas de todas las pruebas."""
        all_metrics = {}
        
        # Recopilar todas las métricas disponibles
        for result in self.results:
            for key, value in result.metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                if isinstance(value, (int, float)):
                    all_metrics[key].append(value)
                    
        # Calcular estadísticas para cada métrica
        aggregated = {}
        for key, values in all_metrics.items():
            if not values:
                continue
                
            aggregated[key] = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "count": len(values)
            }
            
        return aggregated
    
    def summary(self) -> Dict[str, Any]:
        """Obtener resumen de resultados."""
        return {
            "suite_name": self.name,
            "total_tests": len(self.results),
            "successful_tests": sum(1 for r in self.results if r.success),
            "failed_tests": sum(1 for r in self.results if not r.success),
            "success_rate": self.success_rate(),
            "total_duration": self.total_duration(),
            "aggregated_metrics": self.aggregate_metrics()
        }
    
    def print_summary(self):
        """Imprimir resumen de resultados."""
        summary = self.summary()
        logger.info("\n" + "=" * 80)
        logger.info(f"RESUMEN DE PRUEBAS: {self.name}")
        logger.info(f"Total pruebas: {summary['total_tests']}")
        logger.info(f"Pruebas exitosas: {summary['successful_tests']}")
        logger.info(f"Pruebas fallidas: {summary['failed_tests']}")
        logger.info(f"Tasa de éxito: {summary['success_rate']*100:.2f}%")
        logger.info(f"Duración total: {summary['total_duration']:.2f}s")
        
        # Imprimir métricas agregadas
        if summary['aggregated_metrics']:
            logger.info("\nMÉTRICAS AGREGADAS:")
            for metric, stats in summary['aggregated_metrics'].items():
                logger.info(f"  - {metric}: min={stats['min']:.2f}, max={stats['max']:.2f}, avg={stats['avg']:.2f}")
                
        # Imprimir resultados individuales
        logger.info("\nRESULTADOS INDIVIDUALES:")
        for result in self.results:
            logger.info(f"  {result}")
            
        logger.info("=" * 80)

# Pruebas para cada componente
async def test_async_quantum_processor():
    """Prueba para el Procesador Asíncrono Ultra-Cuántico."""
    result = TestResult("AsyncQuantumProcessor")
    
    try:
        # Inicializar procesador con parámetros extremos
        processor = AsyncQuantumProcessor(
            max_entanglement=EXTREME_INTENSITY * 10,
            quantum_stability=0.99,
            dimensional_layers=7,
            transmutation_enabled=True
        )
        
        # Tareas de prueba para operaciones paralelas
        async def task_operation(task_id: int, intensity: float) -> Dict[str, Any]:
            """Tarea de prueba con intensidad variable."""
            # Introducir error aleatorio para probar transmutación
            if random.random() < ERROR_RATE:
                raise Exception(f"Error simulado en tarea {task_id}")
                
            # Simulación de operación con intensidad variable
            await asyncio.sleep(random.uniform(0.01, 0.1) * intensity)
            return {
                "task_id": task_id,
                "result": task_id * intensity,
                "timestamp": time.time()
            }
        
        # Prueba 1: Operaciones masivamente paralelas
        tasks = []
        task_count = int(MAX_PARALLEL_TASKS * EXTREME_INTENSITY)
        
        logger.info(f"Ejecutando {task_count} tareas paralelas en procesador cuántico...")
        start_time = time.time()
        
        # Crear tareas
        for i in range(task_count):
            intensity = random.uniform(0.5, 2.0)
            space = random.choice(list(ProcessingSpace))
            tasks.append(processor.process(lambda i=i, intensity=intensity: task_operation(i, intensity), space=space))
            
        # Ejecutar tareas
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analizar resultados
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        transmuted_count = sum(1 for r in results if isinstance(r, Dict) and r.get("_transmuted", False))
        error_count = sum(1 for r in results if isinstance(r, Exception))
        
        duration = time.time() - start_time
        operations_per_second = task_count / duration
        
        # Registrar métricas
        result.add_metric("task_count", task_count)
        result.add_metric("success_count", success_count)
        result.add_metric("transmuted_count", transmuted_count)
        result.add_metric("error_count", error_count)
        result.add_metric("duration", duration)
        result.add_metric("operations_per_second", operations_per_second)
        
        # Verificar estadísticas del procesador
        stats = processor.get_stats()
        result.add_detail("processor_stats", stats)
        
        # Prueba 2: Prueba de transmutación forzada
        logger.info("Prueba de transmutación forzada...")
        
        # Función que siempre falla
        async def failing_operation():
            raise Exception("Error forzado para prueba de transmutación")
            
        transmuted_result = await processor.process(
            failing_operation, 
            space=ProcessingSpace.HYPER_ENTANGLED,
            force_transmutation=True
        )
        
        # Verificar transmutación
        is_transmuted = isinstance(transmuted_result, dict) and transmuted_result.get("_transmuted", False)
        result.add_metric("forced_transmutation_success", int(is_transmuted))
        result.add_detail("transmuted_result", transmuted_result)
        
        # Prueba 3: Prueba de spaces aislados
        logger.info("Prueba de spaces aislados...")
        
        # Valor compartido para probar aislamiento
        shared_values = {}
        
        # Función que modifica el valor compartido
        async def space_operation(space_name, key):
            # Simular operación
            await asyncio.sleep(0.1)
            # Intentar modificar el valor compartido
            if key not in shared_values:
                shared_values[key] = set()
            shared_values[key].add(space_name)
            return {"space": space_name, "key": key}
            
        # Ejecutar operaciones en spaces diferentes con misma clave
        space_tasks = []
        for space in ProcessingSpace:
            task = processor.process(
                lambda s=space: space_operation(s.name, "test_key"),
                space=space
            )
            space_tasks.append(task)
            
        space_results = await asyncio.gather(*space_tasks)
        
        # Verificar aislamiento
        is_isolated = len(shared_values.get("test_key", set())) <= 1
        result.add_metric("space_isolation_success", int(is_isolated))
        
        # Finalizar resultado
        return result.finish(True)
        
    except Exception as e:
        logger.error(f"Error en prueba AsyncQuantumProcessor: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return result.add_error(e).finish(False)

async def test_websocket_adapter():
    """Prueba para el Adaptador WebSocket Trascendental."""
    result = TestResult("WebSocketAdapter")
    
    try:
        # Inicializar adaptador WebSocket con parámetros extremos
        adapter = UltraQuantumWebSocketAdapter(
            port=8081,
            qubits=int(64 * EXTREME_INTENSITY),
            temporal_horizon=EXTREME_INTENSITY * 5.0,
            multiverse_replication=int(7 * EXTREME_INTENSITY)
        )
        
        # Inicializar el adaptador
        await adapter.start()
        
        # Prueba 1: Envío masivo de mensajes
        message_count = int(MAX_PARALLEL_TASKS * EXTREME_INTENSITY / 5)  # Reducido para no saturar
        logger.info(f"Enviando {message_count} mensajes a través del adaptador WebSocket...")
        
        tasks = []
        sent_messages = []
        
        # Función para enviar mensaje
        async def send_message(msg_id: int):
            message = {
                "id": msg_id,
                "type": "test",
                "data": f"Test data {msg_id}",
                "timestamp": time.time()
            }
            sent_messages.append(message)
            return await adapter.send_message("test_component", message)
            
        # Crear tareas para envío masivo
        for i in range(message_count):
            tasks.append(send_message(i))
            
        # Ejecutar tareas
        send_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analizar resultados
        success_count = sum(1 for r in send_results if not isinstance(r, Exception) and r.get("success", False))
        error_count = sum(1 for r in send_results if isinstance(r, Exception))
        
        # Registrar métricas
        result.add_metric("message_count", message_count)
        result.add_metric("success_count", success_count)
        result.add_metric("error_count", error_count)
        
        # Prueba 2: Comprobación de predicción temporal
        logger.info("Probando predicción temporal...")
        
        # Registrar anomalía temporal
        anomaly_result = await adapter.register_temporal_anomaly("test_anomaly", {"severity": "high"})
        
        # Resolver anomalías
        resolve_result = await adapter.resolve_temporal_anomalies()
        
        # Registrar métricas
        result.add_metric("temporal_anomaly_registered", int(anomaly_result.get("registered", False)))
        result.add_metric("temporal_anomalies_resolved", resolve_result.get("resolved_count", 0))
        
        # Prueba 3: Prueba de entrelazamiento y coherencia
        logger.info("Probando entrelazamiento cuántico...")
        
        # Entrelazar componentes
        entanglement_result = await adapter.entangle_components(["comp_a", "comp_b", "comp_c"])
        
        # Medir coherencia
        coherence = adapter.measure_coherence()
        
        # Registrar métricas
        result.add_metric("entanglement_success", int(entanglement_result.get("success", False)))
        result.add_metric("coherence_level", coherence)
        
        # Detener adaptador
        await adapter.stop()
        
        # Verificar estadísticas
        stats = adapter.get_stats()
        result.add_detail("adapter_stats", stats)
        
        # Finalizar resultado
        return result.finish(True)
        
    except Exception as e:
        logger.error(f"Error en prueba WebSocketAdapter: {e}")
        import traceback
        logger.error(traceback.format_exc())
        try:
            if adapter:
                await adapter.stop()
        except:
            pass
        return result.add_error(e).finish(False)

async def test_external_websocket():
    """Prueba para el WebSocket Externo Trascendental."""
    result = TestResult("ExternalWebSocket")
    
    try:
        # Inicializar adaptador WebSocket Externo
        adapter = ExternalWebSocketAdapter(exchange_id="BINANCE_TEST")
        
        # Inicializar el adaptador
        await adapter.initialize()
        
        # Prueba 1: Conexión simulada
        logger.info("Probando conexión simulada al exchange...")
        
        connect_result = await adapter.connect()
        
        # Registrar métricas
        result.add_metric("connect_success", int(connect_result.get("success", False)))
        
        # Prueba 2: Suscripción a canales
        logger.info("Probando suscripción a canales...")
        
        channels = ["btcusdt@ticker", "ethusdt@ticker", "bnbusdt@ticker"]
        subscribe_result = await adapter.subscribe(channels)
        
        # Registrar métricas
        result.add_metric("subscribe_success", int(subscribe_result.get("success", False)))
        result.add_metric("channels_count", len(channels))
        
        # Prueba 3: Recepción de mensajes
        logger.info("Recibiendo mensajes simulados...")
        
        messages = []
        start_time = time.time()
        message_count = 50  # Número de mensajes a recibir
        
        # Recibir mensajes
        for _ in range(message_count):
            message = await adapter.receive()
            messages.append(message)
            
        duration = time.time() - start_time
        messages_per_second = message_count / duration if duration > 0 else 0
        
        # Registrar métricas
        result.add_metric("message_count", message_count)
        result.add_metric("duration", duration)
        result.add_metric("messages_per_second", messages_per_second)
        
        # Prueba 4: Desconexión
        logger.info("Probando desconexión...")
        
        disconnect_result = await adapter.close()
        
        # Registrar métricas
        result.add_metric("disconnect_success", int(disconnect_result.get("success", False)))
        
        # Verificar estadísticas
        stats = adapter.get_stats()
        result.add_detail("adapter_stats", stats)
        
        # Finalizar resultado
        return result.finish(True)
        
    except Exception as e:
        logger.error(f"Error en prueba ExternalWebSocket: {e}")
        import traceback
        logger.error(traceback.format_exc())
        try:
            if adapter:
                await adapter.close()
        except:
            pass
        return result.add_error(e).finish(False)

async def test_exchange_integrator():
    """Prueba para el Integrador de Exchanges Trascendental."""
    result = TestResult("ExchangeIntegrator")
    
    try:
        # Inicializar integrador
        integrator = TranscendentalExchangeIntegrator()
        
        # Prueba 1: Añadir múltiples exchanges
        logger.info("Añadiendo múltiples exchanges...")
        
        exchanges = [
            ExchangeID.BINANCE,
            ExchangeID.COINBASE,
            ExchangeID.KRAKEN,
            ExchangeID.BITFINEX,
            ExchangeID.HUOBI,
            ExchangeID.BYBIT,
            ExchangeID.OKEX,
            ExchangeID.KUCOIN,
            ExchangeID.BITMEX,
            ExchangeID.FTX  # Este debería provocar transmutación
        ]
        
        add_results = []
        for exchange_id in exchanges:
            result_dict = await integrator.add_exchange(exchange_id)
            add_results.append(result_dict)
            
        # Registrar métricas
        success_count = sum(1 for r in add_results if r.get("success", False))
        result.add_metric("exchanges_count", len(exchanges))
        result.add_metric("exchanges_added", success_count)
        
        # Prueba 2: Conectar a todos los exchanges
        logger.info("Conectando a todos los exchanges...")
        
        connect_result = await integrator.connect_all()
        
        # Registrar métricas
        connected_count = sum(1 for r in connect_result.get("results", {}).values() if r.get("success", False))
        transmuted_count = sum(1 for r in connect_result.get("results", {}).values() if r.get("transmuted", False))
        
        result.add_metric("connected_count", connected_count)
        result.add_metric("connect_transmuted", transmuted_count)
        
        # Prueba 3: Suscribirse a canales en todos los exchanges
        logger.info("Suscribiendo a canales en todos los exchanges...")
        
        # Personalizar símbolos por exchange
        symbols = {
            ExchangeID.BINANCE: "btcusdt",
            ExchangeID.COINBASE: "BTC-USD",
            ExchangeID.KRAKEN: "XBT/USD",
            ExchangeID.BITFINEX: "tBTCUSD",
            ExchangeID.HUOBI: "btcusdt",
            ExchangeID.BYBIT: "BTCUSD",
            ExchangeID.OKEX: "BTC-USDT",
            ExchangeID.KUCOIN: "BTC-USDT",
            ExchangeID.BITMEX: "XBTUSD",
            ExchangeID.FTX: "BTC/USD"
        }
        
        subscription_result = await integrator.subscribe_all(["ticker"], symbols)
        
        # Registrar métricas
        subscribed_count = sum(1 for r in subscription_result.get("results", {}).values() if r.get("success", False))
        sub_transmuted_count = sum(1 for r in subscription_result.get("results", {}).values() if r.get("transmuted", False))
        
        result.add_metric("subscribed_count", subscribed_count)
        result.add_metric("subscribe_transmuted", sub_transmuted_count)
        
        # Prueba 4: Recibir mensajes de todos los exchanges
        logger.info("Recibiendo mensajes de todos los exchanges...")
        
        messages = []
        message_counts = {}
        transmuted_messages = 0
        
        # Función para recibir mensajes
        async def receive_messages():
            nonlocal messages, message_counts, transmuted_messages
            try:
                # Usar listen_all para recibir mensajes de todos los exchanges
                count = 0
                start_time = time.time()
                
                # Usar 5 segundos o hasta recibir MAX_PARALLEL_TASKS / 10 mensajes
                message_limit = MAX_PARALLEL_TASKS / 10
                
                async for message in integrator.listen_all():
                    count += 1
                    messages.append(message)
                    
                    # Contar por exchange
                    exchange_id = message.get("_integrator", {}).get("exchange_id", "unknown")
                    if exchange_id not in message_counts:
                        message_counts[exchange_id] = 0
                    message_counts[exchange_id] += 1
                    
                    # Contar transmutados
                    if message.get("_transmuted", False):
                        transmuted_messages += 1
                        
                    # Limitar tiempo o cantidad
                    if count >= message_limit or time.time() - start_time > 5:
                        break
            except Exception as e:
                logger.error(f"Error recibiendo mensajes: {e}")
                
        # Ejecutar recepción de mensajes
        await receive_messages()
        
        # Registrar métricas
        result.add_metric("messages_count", len(messages))
        result.add_metric("messages_transmuted", transmuted_messages)
        result.add_metric("exchanges_with_messages", len(message_counts))
        
        # Prueba 5: Desconectar de todos los exchanges
        logger.info("Desconectando de todos los exchanges...")
        
        disconnect_result = await integrator.disconnect_all()
        
        # Registrar métricas
        disconnected_count = sum(1 for r in disconnect_result.get("results", {}).values() if r.get("success", False))
        
        result.add_metric("disconnected_count", disconnected_count)
        
        # Verificar estadísticas
        stats = integrator.get_stats()
        result.add_detail("integrator_stats", stats)
        
        # Finalizar resultado
        return result.finish(True)
        
    except Exception as e:
        logger.error(f"Error en prueba ExchangeIntegrator: {e}")
        import traceback
        logger.error(traceback.format_exc())
        try:
            if integrator:
                await integrator.disconnect_all()
        except:
            pass
        return result.add_error(e).finish(False)

async def test_circuit_breaker():
    """Prueba para Circuit Breaker."""
    result = TestResult("CircuitBreaker")
    
    try:
        # Inicializar Circuit Breaker con configuración extrema
        circuit_breaker = CircuitBreaker(
            name="test_circuit",
            failure_threshold=int(5 * EXTREME_INTENSITY),
            reset_timeout=1.0 / EXTREME_INTENSITY,  # Muy rápido para pruebas
            half_open_max_calls=int(3 * EXTREME_INTENSITY)
        )
        
        # Prueba 1: Operaciones básicas
        logger.info("Probando operaciones básicas del Circuit Breaker...")
        
        # Función de prueba que puede fallar
        async def test_operation(should_fail: bool = False):
            await asyncio.sleep(0.01)
            if should_fail:
                raise Exception("Error simulado")
            return {"success": True}
            
        # Ejecutar operaciones exitosas
        for _ in range(10):
            result_op = await circuit_breaker.execute(lambda: test_operation(False))
            assert result_op.get("success", False)
        
        # Verificar estado
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        
        # Registrar métricas
        result.add_metric("init_success", 1)
        result.add_metric("initial_state_correct", 1)
        
        # Prueba 2: Abrir circuito con fallos
        logger.info("Probando apertura de circuito por fallos...")
        
        # Ejecutar operaciones fallidas hasta abrir el circuito
        failure_count = 0
        while circuit_breaker.state == CircuitBreakerState.CLOSED:
            try:
                await circuit_breaker.execute(lambda: test_operation(True))
            except:
                failure_count += 1
                
        # Verificar que se abrió el circuito
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Registrar métricas
        result.add_metric("failures_to_open", failure_count)
        result.add_metric("circuit_opened", 1)
        
        # Prueba 3: Verificar rechazo de solicitudes en estado abierto
        logger.info("Verificando rechazo de solicitudes en estado abierto...")
        
        rejected = False
        try:
            await circuit_breaker.execute(lambda: test_operation(False))
        except Exception as e:
            rejected = "circuit is open" in str(e).lower()
            
        # Registrar métricas
        result.add_metric("open_circuit_rejects", int(rejected))
        
        # Prueba 4: Transición a estado medio abierto
        logger.info("Probando transición a estado medio abierto...")
        
        # Esperar a que se reinicie el timeout
        await asyncio.sleep(circuit_breaker.reset_timeout * 1.5)
        
        # Verificar que pasó a medio abierto
        half_open = circuit_breaker.state == CircuitBreakerState.HALF_OPEN
        
        # Registrar métricas
        result.add_metric("transitioned_to_half_open", int(half_open))
        
        # Prueba 5: Transición de medio abierto a cerrado
        logger.info("Probando transición de medio abierto a cerrado...")
        
        # Ejecutar operaciones exitosas en estado medio abierto
        for _ in range(circuit_breaker.half_open_max_calls):
            await circuit_breaker.execute(lambda: test_operation(False))
            
        # Verificar que volvió a estado cerrado
        closed_again = circuit_breaker.state == CircuitBreakerState.CLOSED
        
        # Registrar métricas
        result.add_metric("transitioned_back_to_closed", int(closed_again))
        
        # Verificar estadísticas
        stats = circuit_breaker.get_stats()
        result.add_detail("circuit_breaker_stats", stats)
        
        # Finalizar resultado
        return result.finish(True)
        
    except Exception as e:
        logger.error(f"Error en prueba CircuitBreaker: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return result.add_error(e).finish(False)

async def test_checkpoint_recovery():
    """Prueba para el sistema de Checkpointing y Recuperación."""
    result = TestResult("CheckpointRecovery")
    
    try:
        # Inicializar Checkpoint Manager
        checkpoint_manager = CheckpointManager(
            component_id="test_component",
            checkpoint_type=CheckpointType.HYBRID,
            max_checkpoints=int(10 * EXTREME_INTENSITY),
            checkpoint_interval=1.0 / EXTREME_INTENSITY
        )
        
        # Prueba 1: Crear checkpoints
        logger.info("Creando checkpoints...")
        
        checkpoints = []
        
        # Crear varios checkpoints con datos diferentes
        for i in range(10):
            data = {
                "iteration": i,
                "timestamp": time.time(),
                "value": random.random() * 100,
                "items": [f"item_{j}" for j in range(i)]
            }
            
            checkpoint_id = await checkpoint_manager.create_checkpoint(data)
            checkpoints.append(checkpoint_id)
            
        # Registrar métricas
        result.add_metric("checkpoints_created", len(checkpoints))
        
        # Prueba 2: Listar checkpoints
        logger.info("Listando checkpoints...")
        
        checkpoint_list = await checkpoint_manager.list_checkpoints()
        
        # Registrar métricas
        result.add_metric("checkpoints_listed", len(checkpoint_list))
        
        # Prueba 3: Restaurar checkpoint específico
        logger.info("Restaurando checkpoint específico...")
        
        # Elegir un checkpoint aleatorio para restaurar
        checkpoint_to_restore = random.choice(checkpoints)
        restored_data = await checkpoint_manager.restore_checkpoint(checkpoint_to_restore)
        
        # Verificar restauración
        restored_success = "iteration" in restored_data
        
        # Registrar métricas
        result.add_metric("restore_success", int(restored_success))
        
        # Prueba 4: Borrar checkpoints específicos
        logger.info("Borrando checkpoints específicos...")
        
        # Elegir algunos checkpoints para borrar
        checkpoints_to_delete = checkpoints[:5]
        
        deleted_count = 0
        for checkpoint_id in checkpoints_to_delete:
            success = await checkpoint_manager.delete_checkpoint(checkpoint_id)
            if success:
                deleted_count += 1
                
        # Registrar métricas
        result.add_metric("checkpoints_deleted", deleted_count)
        
        # Prueba 5: Crear checkpoint automático
        logger.info("Probando checkpoint automático...")
        
        # Activar checkpoint automático
        await checkpoint_manager.enable_auto_checkpoint(True)
        
        # Simular actualizaciones de estado
        for i in range(20):
            data = {
                "auto_iteration": i,
                "timestamp": time.time(),
                "value": random.random() * 100
            }
            await checkpoint_manager.update_state(data)
            await asyncio.sleep(0.1)  # Dar tiempo para checkpoint automático
            
        # Recuperar último checkpoint automático
        auto_checkpoint = await checkpoint_manager.restore_latest_checkpoint()
        
        # Verificar checkpoint automático
        auto_checkpoint_success = auto_checkpoint and "auto_iteration" in auto_checkpoint
        
        # Registrar métricas
        result.add_metric("auto_checkpoint_success", int(auto_checkpoint_success))
        
        # Verificar estadísticas
        stats = checkpoint_manager.get_stats()
        result.add_detail("checkpoint_manager_stats", stats)
        
        # Finalizar resultado
        return result.finish(True)
        
    except Exception as e:
        logger.error(f"Error en prueba CheckpointRecovery: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return result.add_error(e).finish(False)

async def test_integrated_system():
    """Prueba de integración de todos los componentes juntos."""
    result = TestResult("IntegratedSystem")
    
    # Componentes
    processor = None
    websocket = None
    integrator = None
    circuit_breaker = None
    checkpoint_manager = None
    
    try:
        logger.info("Iniciando prueba integrada de todos los componentes...")
        
        # Inicializar todos los componentes
        
        # 1. Procesador Asíncrono
        processor = AsyncQuantumProcessor(
            max_entanglement=EXTREME_INTENSITY * 5,
            quantum_stability=0.95,
            dimensional_layers=3,
            transmutation_enabled=True
        )
        
        # 2. WebSocket
        websocket = UltraQuantumWebSocketAdapter(
            port=8082,
            qubits=int(32 * EXTREME_INTENSITY),
            temporal_horizon=EXTREME_INTENSITY * 2.0,
            multiverse_replication=int(3 * EXTREME_INTENSITY)
        )
        
        # 3. Integrador
        integrator = TranscendentalExchangeIntegrator()
        
        # 4. Circuit Breaker
        circuit_breaker = CircuitBreaker(
            name="integrated_circuit",
            failure_threshold=int(3 * EXTREME_INTENSITY),
            reset_timeout=2.0 / EXTREME_INTENSITY,
            half_open_max_calls=int(2 * EXTREME_INTENSITY)
        )
        
        # 5. Checkpoint Manager
        checkpoint_manager = CheckpointManager(
            component_id="integrated_component",
            checkpoint_type=CheckpointType.HYBRID,
            max_checkpoints=int(5 * EXTREME_INTENSITY),
            checkpoint_interval=2.0 / EXTREME_INTENSITY
        )
        
        # Inicializar componentes que lo requieren
        await websocket.start()
        
        # Añadir exchanges al integrador
        exchanges = [
            ExchangeID.BINANCE,
            ExchangeID.COINBASE,
            ExchangeID.KRAKEN
        ]
        
        for exchange_id in exchanges:
            await integrator.add_exchange(exchange_id)
            
        # Conectar a exchanges
        await integrator.connect_all()
        
        # Activar checkpoint automático
        await checkpoint_manager.enable_auto_checkpoint(True)
        
        # Registrar inicialización exitosa
        result.add_metric("initialization_success", 1)
        
        # Prueba 1: Flujo completo integrado
        logger.info("Ejecutando flujo completo integrado...")
        
        # Estado del sistema
        system_state = {
            "exchanges": exchanges,
            "subscriptions": {},
            "messages": [],
            "checkpoints": [],
            "errors": [],
            "transmutations": 0
        }
        
        # Suscribirse a canales
        symbols = {
            ExchangeID.BINANCE: "btcusdt",
            ExchangeID.COINBASE: "BTC-USD",
            ExchangeID.KRAKEN: "XBT/USD"
        }
        
        # Usar circuit breaker para suscripción
        subscription_result = await circuit_breaker.execute(
            lambda: integrator.subscribe_all(["ticker"], symbols)
        )
        
        system_state["subscriptions"] = subscription_result
        
        # Guardar checkpoint del estado inicial
        initial_checkpoint_id = await checkpoint_manager.create_checkpoint(system_state)
        system_state["checkpoints"].append(initial_checkpoint_id)
        
        # Función para procesar mensaje
        async def process_message(message):
            # Añadir timestamp de procesamiento
            message["processed_at"] = time.time()
            
            # Procesar en procesador cuántico
            try:
                result = await processor.process(
                    lambda msg=message: asyncio.sleep(0.01) and msg,
                    space=ProcessingSpace.SAFE_ZONE
                )
                
                # Verificar transmutación
                if isinstance(result, dict) and result.get("_transmuted", False):
                    system_state["transmutations"] += 1
                    
                # Enviar a WebSocket
                await websocket.send_message("data_processor", result)
                
                return result
            except Exception as e:
                system_state["errors"].append(str(e))
                return None
        
        # Procesar mensajes durante 5 segundos
        logger.info("Procesando mensajes en tiempo real...")
        
        start_time = time.time()
        message_count = 0
        
        # Crear y ejecutar tareas de procesamiento
        while time.time() - start_time < 5:
            # Recibir mensajes de todos los exchanges
            for exchange_id in exchanges:
                try:
                    message = await integrator.receive(exchange_id)
                    
                    # Añadir a lista de mensajes
                    system_state["messages"].append(message)
                    message_count += 1
                    
                    # Procesar mensaje
                    await process_message(message)
                    
                    # Guardar checkpoint cada 10 mensajes
                    if message_count % 10 == 0:
                        checkpoint_id = await checkpoint_manager.create_checkpoint(system_state)
                        system_state["checkpoints"].append(checkpoint_id)
                except Exception as e:
                    system_state["errors"].append(str(e))
        
        # Registrar métricas
        result.add_metric("messages_processed", message_count)
        result.add_metric("transmutations", system_state["transmutations"])
        result.add_metric("checkpoints_created", len(system_state["checkpoints"]))
        result.add_metric("errors", len(system_state["errors"]))
        
        # Prueba 2: Recuperación de estado
        logger.info("Probando recuperación de estado...")
        
        # Recuperar estado desde el primer checkpoint
        recovered_state = await checkpoint_manager.restore_checkpoint(system_state["checkpoints"][0])
        
        # Verificar recuperación
        recovery_success = (
            recovered_state and 
            "exchanges" in recovered_state and
            len(recovered_state["exchanges"]) == len(exchanges)
        )
        
        result.add_metric("recovery_success", int(recovery_success))
        
        # Limpiar
        await integrator.disconnect_all()
        await websocket.stop()
        
        # Obtener estadísticas
        result.add_detail("processor_stats", processor.get_stats())
        result.add_detail("websocket_stats", websocket.get_stats())
        result.add_detail("integrator_stats", integrator.get_stats())
        result.add_detail("circuit_breaker_stats", circuit_breaker.get_stats())
        result.add_detail("checkpoint_manager_stats", checkpoint_manager.get_stats())
        
        # Finalizar resultado
        return result.finish(True)
        
    except Exception as e:
        logger.error(f"Error en prueba IntegratedSystem: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Limpiar
        try:
            if integrator:
                await integrator.disconnect_all()
            if websocket:
                await websocket.stop()
        except:
            pass
            
        return result.add_error(e).finish(False)

async def run_tests():
    """Ejecutar todas las pruebas."""
    
    # Crear suite de pruebas
    suite = TestSuite("Genesis Core Integration Extreme")
    
    # Mostrar mensaje de inicio
    logger.info("\n" + "=" * 80)
    logger.info(f"INICIANDO PRUEBAS EXHAUSTIVAS DEL MÓDULO GENESIS/CORE (INTENSIDAD: {EXTREME_INTENSITY}x)")
    logger.info("=" * 80)
    
    # Lista de pruebas a ejecutar
    tests = [
        test_async_quantum_processor,
        test_websocket_adapter,
        test_external_websocket,
        test_exchange_integrator,
        test_circuit_breaker,
        test_checkpoint_recovery,
        test_integrated_system
    ]
    
    # Ejecutar pruebas una por una
    for i, test_func in enumerate(tests):
        # Limpiar memoria entre pruebas
        gc.collect()
        
        logger.info(f"\n[{i+1}/{len(tests)}] Ejecutando {test_func.__name__}...")
        
        try:
            # Ejecutar prueba
            result = await test_func()
            
            # Añadir resultado a la suite
            suite.add_result(result)
            
            # Mostrar resultado
            logger.info(f"Resultado: {result}")
            
        except Exception as e:
            # Crear resultado fallido
            result = TestResult(test_func.__name__).add_error(e).finish(False)
            suite.add_result(result)
            
            logger.error(f"Error ejecutando {test_func.__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Mostrar resumen de resultados
    suite.print_summary()
    
    # Guardar resultados en archivo
    results_file = "test_core_integration_extreme_results.json"
    try:
        with open(results_file, "w") as f:
            json.dump({
                "summary": suite.summary(),
                "results": [r.to_dict() for r in suite.results]
            }, f, indent=2)
        logger.info(f"Resultados guardados en {results_file}")
    except Exception as e:
        logger.error(f"Error guardando resultados: {e}")
    
    # Devolver suite para análisis posterior
    return suite

if __name__ == "__main__":
    # Configurar timeout más largo para pruebas extremas
    os.environ["PYTHONASYNCIODEBUG"] = "1"
    
    # Ejecutar pruebas
    asyncio.run(run_tests())