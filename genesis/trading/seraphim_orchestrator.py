"""
Seraphim Orchestrator - Orquestador Celestial para el Sistema Genesis Ultra-Divino.

Este módulo implementa el orquestador divino que coordina todos los componentes
del Sistema Genesis para la estrategia Seraphim Pool, asegurando una integración
perfecta y sincronización trascendental entre:

- Estrategia Seraphim Pool (comportamiento humano simulado)
- Análisis de Buddha AI (sabiduría divina)
- Clasificador Transcendental (selección de activos)
- Sistema de Gestión de Riesgo Adaptativo (protección celestial)
- Componentes Cloud Divinos (resiliencia absoluta)

El Seraphim Orchestrator representa la cúspide de la evolución arquitectónica
del Sistema Genesis, con capacidades cuánticas que permiten un rendimiento superior
y resiliencia absoluta.

Autor: Genesis AI Assistant
Versión: 1.0.0 (Divina)
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum, auto
import uuid
import random

# Componentes Genesis
from genesis.strategies.seraphim.seraphim_pool import SeraphimPool, SeraphimState, CyclePhase
from genesis.trading.buddha_integrator import BuddhaIntegrator
from genesis.trading.human_behavior_engine import GabrielBehaviorEngine, EmotionalState, RiskTolerance, DecisionStyle
from genesis.trading.codicia_manager import CodiciaManager, OrderType, OrderSide, OrderStatus
from genesis.analysis.transcendental_crypto_classifier import TranscendentalCryptoClassifier
from genesis.cloud.circuit_breaker_v4 import CloudCircuitBreakerV4
from genesis.cloud.distributed_checkpoint_v4 import DistributedCheckpointManagerV4
from genesis.cloud.load_balancer_v4 import CloudLoadBalancerV4
from genesis.db.transcendental_database import TranscendentalDatabase
from genesis.oracle.quantum_oracle import QuantumOracle
from genesis.notifications.alert_manager import AlertManager

# Configuración de logging
logger = logging.getLogger(__name__)

class OrchestratorState(Enum):
    """Estados divinos del orquestador Seraphim."""
    INACTIVE = auto()        # Sin inicializar
    AWAKENING = auto()       # Inicializando componentes
    HARMONIZING = auto()     # Sincronizando componentes
    CONDUCTING = auto()      # Coordinando operaciones activas
    TRANSCENDING = auto()    # Operando en modo divino completo
    MEDITATING = auto()      # Pausa entre ciclos
    ASCENDING = auto()       # Mejorando a nivel superior
    TRANSFORMING = auto()    # Reconfigurando componentes

class SeraphimOrchestrator:
    """
    Orquestador Celestial para el Sistema Genesis Ultra-Divino.
    
    Este orquestador divino implementa la coordinación perfecta entre todos
    los componentes del Sistema Genesis, con capacidades cuánticas y
    resiliencia absoluta.
    """
    
    def __init__(self):
        """Inicializar el Orquestador Seraphim con propiedades divinas."""
        # Estado y configuración
        self.state = OrchestratorState.INACTIVE
        self.instance_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        
        # Componentes integrados
        self.seraphim_strategy: Optional[SeraphimPool] = None
        self.buddha_integrator: Optional[BuddhaIntegrator] = None
        self.behavior_engine: Optional[GabrielBehaviorEngine] = None  # Motor de comportamiento humano
        self.classifier: Optional[TranscendentalCryptoClassifier] = None
        self.circuit_breaker: Optional[CloudCircuitBreakerV4] = None
        self.checkpoint_manager: Optional[DistributedCheckpointManagerV4] = None
        self.load_balancer: Optional[CloudLoadBalancerV4] = None
        self.database: Optional[TranscendentalDatabase] = None
        self.oracle: Optional[QuantumOracle] = None
        self.alert_manager: Optional[AlertManager] = None
        
        # Componente de Exchange (simulado o real)
        self.exchange_adapter = None  # Será configurado externamente
        self.codicia_manager = None  # Portal de extracción de riqueza para trading
        
        # Estado operacional
        self.active_cycle_id: Optional[str] = None
        self.active_cycles_count = 0
        self.completed_cycles_count = 0
        self.total_realized_profit = 0.0
        self.system_health = 1.0  # 0.0-1.0
        self.last_checkpoint_time = None
        
        # Configuración
        self.auto_cycle_enabled = False
        self.cycle_interval = timedelta(hours=24)
        self.health_check_interval = timedelta(minutes=15)
        
        logger.info("Seraphim Orchestrator inicializado en estado INACTIVE")
    
    async def initialize(self) -> bool:
        """
        Inicializar el orquestador y todos sus componentes divinos.
        
        Returns:
            True si la inicialización fue exitosa
        """
        try:
            logger.info("Iniciando inicialización del Seraphim Orchestrator...")
            self.state = OrchestratorState.AWAKENING
            
            # Inicializar componentes
            self.seraphim_strategy = SeraphimPool()
            self.buddha_integrator = BuddhaIntegrator()
            self.behavior_engine = GabrielBehaviorEngine()  # Motor de comportamiento humano
            self.classifier = TranscendentalCryptoClassifier()
            self.circuit_breaker = CloudCircuitBreakerV4()
            self.checkpoint_manager = DistributedCheckpointManagerV4()
            self.load_balancer = CloudLoadBalancerV4()
            self.database = TranscendentalDatabase()
            self.oracle = QuantumOracle()
            self.alert_manager = AlertManager()
            
            # Inicializar cada componente
            
            # Inicializar el motor de comportamiento humano primero, ya que otros componentes lo usan
            await self.behavior_engine.initialize()
            logger.info("Motor de comportamiento humano Gabriel inicializado correctamente")
            
            # Inicializar el resto de componentes
            components_init_results = await asyncio.gather(
                self.seraphim_strategy.initialize(),
                self.buddha_integrator.initialize(),
                self.classifier.initialize(),
                self.circuit_breaker.initialize(),
                self.checkpoint_manager.initialize(),
                self.load_balancer.initialize(),
                self.database.initialize(),
                self.oracle.initialize(),
                self.alert_manager.initialize(),
                return_exceptions=True
            )
            
            # Verificar inicialización exitosa
            any_failed = any(isinstance(res, Exception) for res in components_init_results)
            all_success = all(res is True for res in components_init_results
                            if not isinstance(res, Exception))
            
            if any_failed:
                failed_components = [
                    str(res) for res in components_init_results
                    if isinstance(res, Exception)
                ]
                logger.error(f"Falló inicialización de componentes: {', '.join(failed_components)}")
                self.state = OrchestratorState.INACTIVE
                return False
            
            if not all_success:
                logger.warning("Algunos componentes se inicializaron con advertencias")
                self.system_health = 0.9
            
            # Sincronizar componentes
            self.state = OrchestratorState.HARMONIZING
            sync_success = await self._synchronize_components()
            
            if not sync_success:
                logger.error("Falló sincronización de componentes")
                self.state = OrchestratorState.INACTIVE
                return False
            
            # Crear checkpoint inicial
            self.last_checkpoint_time = datetime.now()
            await self.checkpoint_manager.create_checkpoint(
                f"orchestrator_init_{self.instance_id}",
                {"orchestrator_id": self.instance_id, "state": self.get_state()}
            )
            
            # Activar orquestador
            self.state = OrchestratorState.CONDUCTING
            
            logger.info("Seraphim Orchestrator inicializado correctamente, estado: CONDUCTING")
            return True
            
        except Exception as e:
            logger.error(f"Error al inicializar Seraphim Orchestrator: {str(e)}")
            self.state = OrchestratorState.INACTIVE
            return False
    
    async def _synchronize_components(self) -> bool:
        """
        Sincronizar todos los componentes del sistema.
        
        Returns:
            True si la sincronización fue exitosa
        """
        try:
            # Verificar conexiones entre componentes
            logger.info("Sincronizando componentes del Sistema Genesis...")
            
            # 1. Verificar integración Buddha
            buddha_status = await self.buddha_integrator.get_status()
            if not buddha_status.get("active", False):
                logger.warning(f"Buddha AI no está activo: {buddha_status.get('status', 'unknown')}")
                self.system_health *= 0.9
            
            # 2. Verificar clasificador
            classifier_ready = await self.classifier.is_ready()
            if not classifier_ready:
                logger.warning("Clasificador Trascendental no está listo")
                self.system_health *= 0.9
            
            # 3. Verificar componentes cloud
            cloud_components_status = await asyncio.gather(
                self.circuit_breaker.get_status(),
                self.checkpoint_manager.get_status(),
                self.load_balancer.get_status()
            )
            
            all_cloud_ready = all(
                status.get("status", "") == "ready" 
                for status in cloud_components_status
            )
            
            if not all_cloud_ready:
                logger.warning("No todos los componentes cloud están listos")
                self.system_health *= 0.95
            
            # 4. Verificar oracle
            oracle_ready = await self.oracle.verify_connection()
            if not oracle_ready:
                logger.warning("Oráculo Cuántico no está listo")
                self.system_health *= 0.9
            
            # Registrar componentes en el load balancer
            await self.load_balancer.register_component(
                component_id="seraphim_strategy",
                component_type="strategy",
                capacity=1.0,
                priority=10
            )
            
            # Vincular estrategia con Buddha AI
            await self.seraphim_strategy.buddha_integrator.sync_with_external(
                self.buddha_integrator
            )
            
            # Vincular estrategia con el motor de comportamiento humano Gabriel
            logger.info("Vinculando motor de comportamiento humano Gabriel con estrategia Seraphim...")
            # Crear una instancia del motor de comportamiento en la estrategia si no existe
            if not hasattr(self.seraphim_strategy, 'behavior_engine') or self.seraphim_strategy.behavior_engine is None:
                self.seraphim_strategy.behavior_engine = self.behavior_engine
                logger.info("Motor de comportamiento humano Gabriel asignado a estrategia Seraphim")
            
            logger.info(f"Sincronización completa, salud del sistema: {self.system_health:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error al sincronizar componentes: {str(e)}")
            return False
    
    async def start_trading_cycle(self) -> Dict[str, Any]:
        """
        Iniciar un nuevo ciclo de trading a través de la estrategia Seraphim.
        
        Returns:
            Resultado del inicio del ciclo
        """
        try:
            # Verificar que el orquestador esté activo
            if self.state not in [OrchestratorState.CONDUCTING, OrchestratorState.TRANSCENDING]:
                logger.warning(f"Orquestador no está en estado adecuado: {self.state}")
                return {"success": False, "error": "Orchestrator not in conducting state"}
            
            # Verificar si hay un ciclo activo
            if self.active_cycle_id:
                cycle_status = await self.get_cycle_status(self.active_cycle_id)
                if cycle_status.get("is_active", False):
                    logger.warning(f"Ya hay un ciclo activo: {self.active_cycle_id}")
                    return {"success": False, "error": "Active cycle already exists"}
            
            # Preparar componentes para nuevo ciclo
            preparation_success = await self._prepare_for_new_cycle()
            if not preparation_success:
                logger.error("Falló preparación para nuevo ciclo")
                return {"success": False, "error": "Failed to prepare for new cycle"}
            
            # Iniciar ciclo
            cycle_start_result = await self.seraphim_strategy.start_cycle()
            if not cycle_start_result:
                logger.error("Falló inicio de ciclo")
                return {"success": False, "error": "Failed to start cycle"}
            
            # Registrar ciclo activo
            self.active_cycle_id = self.seraphim_strategy.cycle_id
            self.active_cycles_count += 1
            
            # Crear checkpoint
            await self.checkpoint_manager.create_checkpoint(
                f"cycle_start_{self.active_cycle_id}",
                {"orchestrator_id": self.instance_id, "cycle_id": self.active_cycle_id}
            )
            
            logger.info(f"Ciclo de trading iniciado: {self.active_cycle_id}")
            
            # Entrar en modo trascendental durante ciclo activo
            self.state = OrchestratorState.TRANSCENDING
            
            return {
                "success": True,
                "cycle_id": self.active_cycle_id,
                "start_time": self.seraphim_strategy.cycle_start_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error al iniciar ciclo de trading: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _prepare_for_new_cycle(self) -> bool:
        """
        Preparar sistema para un nuevo ciclo de trading.
        
        Returns:
            True si la preparación fue exitosa
        """
        try:
            # 1. Verificar salud del sistema
            health_check = await self.check_system_health()
            if health_check < 0.8:
                logger.warning(f"Salud del sistema baja para nuevo ciclo: {health_check:.2f}")
                return False
            
            # 2. Verificar clasificación actualizada
            classifier_updated = await self.classifier.update_classification()
            if not classifier_updated:
                logger.warning("No se pudo actualizar clasificación")
                return False
            
            # 3. Consultar al oráculo sobre condiciones
            oracle_prediction = await self.oracle.predict_market_conditions()
            safe_to_proceed = oracle_prediction.get("safe_to_proceed", False)
            
            if not safe_to_proceed:
                danger_reason = oracle_prediction.get("warning", "Unknown danger")
                logger.warning(f"Oráculo advierte no proceder: {danger_reason}")
                return False
            
            # 4. Actualizar análisis de Buddha
            await self.buddha_integrator.refresh_analysis()
            
            # 5. Verificar disponibilidad de exchanges
            exchange_status = await self._verify_exchange_connections()
            if not exchange_status:
                logger.warning("Conexión a exchanges no disponible")
                return False
            
            logger.info("Sistema preparado para nuevo ciclo de trading")
            return True
            
        except Exception as e:
            logger.error(f"Error al preparar para nuevo ciclo: {str(e)}")
            return False
    
    async def _verify_exchange_connections(self) -> bool:
        """
        Verificar conexiones a exchanges utilizando el adaptador configurado.
        
        Returns:
            True si las conexiones están disponibles
        """
        try:
            # Si aún no existe un adaptador de exchange, intentamos crearlo
            if not hasattr(self, 'exchange_adapter') or self.exchange_adapter is None:
                logger.info("No hay adaptador de exchange configurado, comprobando Binance Testnet...")
                
                # Importar fábrica de adaptadores
                from genesis.exchanges.adapter_factory import ExchangeAdapterFactory, AdapterType
                
                # Verificar credenciales de Binance Testnet
                import os
                has_binance_testnet_credentials = bool(
                    os.environ.get("BINANCE_TESTNET_API_KEY") and 
                    os.environ.get("BINANCE_TESTNET_API_SECRET")
                )
                
                if has_binance_testnet_credentials:
                    logger.info("Se encontraron credenciales de Binance Testnet, usando adaptador específico...")
                    
                    # Crear adaptador para Binance Testnet
                    self.exchange_adapter = await ExchangeAdapterFactory.create_adapter(
                        exchange_id="BINANCE",
                        adapter_type=AdapterType.BINANCE_TESTNET
                    )
                    
                    logger.info(f"Adaptador de Binance Testnet creado: {self.exchange_adapter.__class__.__name__}")
                else:
                    logger.info("No se encontraron credenciales de Binance Testnet, usando simulador...")
                    
                    # Crear adaptador simulado por defecto
                    self.exchange_adapter = await ExchangeAdapterFactory.create_adapter(
                        exchange_id="BINANCE",
                        adapter_type=AdapterType.SIMULATED,
                        config={
                            "tick_interval_ms": 500,
                            "volatility_factor": 0.005,
                            "pattern_duration": 120,
                            "enable_failures": False,  # Desactivar fallos para más estabilidad
                            "default_candle_count": 1000  # Velas históricas suficientes
                        }
                    )
                    
                    logger.info(f"Adaptador de exchange simulado creado: {self.exchange_adapter.__class__.__name__}")
                
                # Precargar símbolos comunes
                await self._preload_symbols()
            
            # Inicializar CodiciaManager si aún no existe
            if not hasattr(self, 'codicia_manager') or self.codicia_manager is None:
                logger.info("Inicializando CodiciaManager para gestión de órdenes...")
                # Creamos el CodiciaManager con los componentes necesarios
                self.codicia_manager = CodiciaManager(
                    self.exchange_adapter,
                    self.behavior_engine
                )
                # Inicializar el CodiciaManager para activar el seguimiento de órdenes
                await self.codicia_manager.initialize()
                logger.info("CodiciaManager inicializado y vinculado al exchange adapter")
            
            # Verificar estado del adaptador
            if hasattr(self.exchange_adapter, 'get_state'):
                adapter_state = self.exchange_adapter.get_state()
                is_connected = adapter_state.get('connected', False)
                
                if not is_connected:
                    # Intentar (re)conectar
                    await self.exchange_adapter.connect()
                    # Verificar estado nuevamente
                    adapter_state = self.exchange_adapter.get_state()
                    is_connected = adapter_state.get('connected', False)
                
                logger.info(f"Estado de conexión a exchange: {is_connected}")
                return is_connected
            else:
                # Para adaptadores que no implementan get_state
                return True
                
        except Exception as e:
            logger.error(f"Error al verificar conexiones a exchanges: {str(e)}")
            return False
    
    async def _preload_symbols(self) -> None:
        """Precargar símbolos comunes en el simulador."""
        if self.exchange_adapter is None:
            logger.warning("No hay adaptador de exchange para precargar símbolos")
            return
        
        try:
            # Lista de símbolos comunes
            symbols = [
                "BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT",
                "BNB/USDT", "DOT/USDT", "DOGE/USDT", "AVAX/USDT", "MATIC/USDT"
            ]
            
            # Inicializar cada símbolo en el simulador
            for symbol in symbols:
                # Intentar obtener ticker para inicializar el símbolo
                ticker = await self.exchange_adapter.get_ticker(symbol)
                logger.debug(f"Símbolo precargado: {symbol}, precio: {ticker.get('last', 'N/A')}")
            
            logger.info(f"Precargados {len(symbols)} símbolos en el simulador")
            
        except Exception as e:
            logger.error(f"Error al precargar símbolos: {str(e)}")
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Obtener datos de mercado para un símbolo.
        
        Args:
            symbol: Símbolo a consultar
            
        Returns:
            Datos de mercado actuales
        """
        try:
            if self.exchange_adapter is None:
                logger.warning("No hay adaptador de exchange configurado")
                return {"success": False, "error": "No exchange adapter configured"}
            
            # Obtener datos del ticker
            ticker = await self.exchange_adapter.get_ticker(symbol)
            
            # Obtener últimas velas (OHLCV)
            candles = await self.exchange_adapter.get_candles(
                symbol=symbol,
                timeframe="5m",  # Intervalo de 5 minutos
                limit=20  # Las últimas 20 velas
            )
            
            # Calcular algunas métricas básicas
            prices = [candle[4] for candle in candles]  # El precio de cierre (índice 4)
            avg_price = sum(prices) / len(prices) if prices else 0
            
            # Calcular tendencia simple
            if len(prices) >= 2:
                trend = "UP" if prices[-1] > prices[0] else "DOWN"
            else:
                trend = "NEUTRAL"
            
            return {
                "success": True,
                "symbol": symbol,
                "last_price": ticker.get("last", 0),
                "bid": ticker.get("bid", 0),
                "ask": ticker.get("ask", 0),
                "volume": ticker.get("volume", 0),
                "timestamp": ticker.get("timestamp", int(time.time() * 1000)),
                "avg_price": avg_price,
                "trend": trend,
                "candles": candles[:5]  # Solo incluir las últimas 5 velas para no sobrecargar
            }
            
        except Exception as e:
            logger.error(f"Error al obtener datos de mercado para {symbol}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_symbols(self) -> List[str]:
        """
        Obtener lista de símbolos disponibles.
        
        Returns:
            Lista de símbolos
        """
        try:
            if self.exchange_adapter is None:
                logger.warning("No hay adaptador de exchange configurado")
                return []
            
            # Obtener mercados disponibles
            markets = await self.exchange_adapter.get_markets()
            
            # Extraer símbolos
            symbols = [market.get("symbol") for market in markets if "symbol" in market]
            
            return symbols
            
        except Exception as e:
            logger.error(f"Error al obtener símbolos: {str(e)}")
            return []
            
    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType, 
                          amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Colocar una orden de trading a través del CodiciaManager.
        
        Args:
            symbol: Símbolo del mercado (ej: BTC/USDT)
            side: Lado de la orden (buy/sell)
            order_type: Tipo de orden (market/limit)
            amount: Cantidad
            price: Precio (solo para órdenes limit)
            
        Returns:
            Resultado de la orden
        """
        try:
            # Verificar que el CodiciaManager esté configurado
            if self.codicia_manager is None:
                # Inicializar CodiciaManager si no existe
                await self._verify_exchange_connections()
                
                if self.codicia_manager is None:
                    logger.error("No se pudo inicializar CodiciaManager")
                    return {"success": False, "error": "CodiciaManager not available"}
            
            # Colocar orden a través del CodiciaManager
            # Crear diccionario de parámetros para cumplir con la firma del método
            order_params = {
                "symbol": symbol,
                "side": side.name if isinstance(side, OrderSide) else side,
                "order_type": order_type.name if isinstance(order_type, OrderType) else order_type,
                "amount": amount
            }
            
            # Añadir precio solo si no es None
            if price is not None:
                order_params["price"] = price
                
            order_result = await self.codicia_manager.place_order(order_params)
            
            logger.info(f"Orden colocada: {side.name} {order_type.name} {amount} {symbol} - ID: {order_result.get('order_id', 'unknown')}")
            
            return {
                "success": True,
                "order_id": order_result.get("order_id"),
                "symbol": symbol,
                "side": side.name,
                "type": order_type.name,
                "amount": amount,
                "price": price,
                "status": order_result.get("status", OrderStatus.NEW.name),
                "created_at": order_result.get("created_at", datetime.now().isoformat())
            }
            
        except Exception as e:
            logger.error(f"Error al colocar orden para {symbol}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancelar una orden existente.
        
        Args:
            order_id: Identificador de la orden a cancelar
            
        Returns:
            Resultado de la cancelación
        """
        try:
            # Verificar que el CodiciaManager esté configurado
            if self.codicia_manager is None:
                logger.error("CodiciaManager no inicializado")
                return {"success": False, "error": "CodiciaManager not available"}
            
            # Cancelar orden a través del CodiciaManager
            cancel_result = await self.codicia_manager.cancel_order(order_id)
            
            if cancel_result.get("success", False):
                logger.info(f"Orden cancelada correctamente: {order_id}")
                return {
                    "success": True,
                    "order_id": order_id,
                    "message": "Order cancelled successfully"
                }
            else:
                logger.warning(f"No se pudo cancelar orden {order_id}: {cancel_result.get('error', 'unknown reason')}")
                return {
                    "success": False,
                    "order_id": order_id,
                    "error": cancel_result.get("error", "Failed to cancel order")
                }
            
        except Exception as e:
            logger.error(f"Error al cancelar orden {order_id}: {str(e)}")
            return {"success": False, "order_id": order_id, "error": str(e)}
    
    async def get_orders(self, symbol: Optional[str] = None, status: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtener órdenes existentes.
        
        Args:
            symbol: Filtrar por símbolo (opcional)
            status: Filtrar por estado (optional: "open", "closed", o None para todas)
            
        Returns:
            Resultado con lista de órdenes
        """
        try:
            # Verificar que el CodiciaManager esté configurado
            if self.codicia_manager is None:
                logger.error("CodiciaManager no inicializado")
                return {"success": False, "error": "CodiciaManager not available", "orders": []}
            
            # Obtener órdenes a través del CodiciaManager
            orders = await self.codicia_manager.get_orders(symbol=symbol, status=status)
            
            logger.info(f"Obtenidas {len(orders)} órdenes para {symbol or 'todos los símbolos'}")
            
            return {
                "success": True,
                "count": len(orders),
                "symbol": symbol,
                "status": status,
                "orders": orders
            }
            
        except Exception as e:
            logger.error(f"Error al obtener órdenes: {str(e)}")
            return {"success": False, "error": str(e), "orders": []}
    
    async def process_cycle(self) -> Dict[str, Any]:
        """
        Procesar ciclo activo completo, desde análisis hasta distribución.
        
        Returns:
            Resultado del procesamiento del ciclo
        """
        try:
            # Verificar ciclo activo
            if not self.active_cycle_id:
                logger.warning("No hay ciclo activo para procesar")
                return {"success": False, "error": "No active cycle"}
            
            # Verificar estado adecuado
            cycle_status = await self.get_cycle_status(self.active_cycle_id)
            cycle_phase = cycle_status.get("cycle_phase", "unknown")
            
            # Ejecutar fase apropiada según estado actual
            if cycle_phase == "REVELATION":
                # Fase de análisis de mercado
                result = await self.seraphim_strategy.analyze_market()
                logger.info(f"Análisis de mercado completado: {len(result.get('selected_assets', []))} activos seleccionados")
                
            elif cycle_phase == "EXECUTION":
                # Fase de ejecución de operaciones
                result = await self.seraphim_strategy.execute_trades()
                logger.info(f"Ejecución de operaciones completada: {len(result.get('executed_trades', []))} operaciones")
                
            elif cycle_phase == "GUARDIANSHIP":
                # Fase de monitorización de posiciones
                result = await self.seraphim_strategy.monitor_positions()
                roi = result.get("roi_percentage", 0)
                logger.info(f"Monitorización completada: ROI actual {roi:.2f}%")
                
            elif cycle_phase == "REFLECTION":
                # Fase de evaluación de resultados
                result = await self.seraphim_strategy.evaluate_cycle()
                logger.info("Evaluación de ciclo completada")
                
            elif cycle_phase == "DISTRIBUTION":
                # Fase de distribución de ganancias
                result = await self.seraphim_strategy.distribute_profits()
                # Actualizar contador de ciclos completados
                self.completed_cycles_count += 1
                # Acumular ganancias totales
                profit = result.get("total_profit", 0)
                self.total_realized_profit += profit
                # Actualizar ciclo activo
                self.active_cycle_id = None
                
                logger.info(f"Distribución completada, ganancia: ${profit:.2f}")
                
                # Volver a estado de conducción normal
                self.state = OrchestratorState.CONDUCTING
                
            else:
                # Fase no procesable directamente
                logger.info(f"Fase actual {cycle_phase} no requiere procesamiento directo")
                result = {"success": True, "status": "no_action_required"}
            
            # Crear checkpoint después de cada fase
            await self.checkpoint_manager.create_checkpoint(
                f"cycle_phase_{cycle_phase}_{self.active_cycle_id}",
                {"orchestrator_id": self.instance_id, "cycle_id": self.active_cycle_id}
            )
            
            return {
                "success": True,
                "cycle_id": self.active_cycle_id,
                "phase": cycle_phase,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error al procesar ciclo: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_cycle_status(self, cycle_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtener estado actual del ciclo activo o especificado.
        
        Args:
            cycle_id: ID del ciclo a consultar, o None para ciclo activo
            
        Returns:
            Estado actual del ciclo
        """
        try:
            # Usar ciclo activo si no se especifica ID
            if cycle_id is None:
                cycle_id = self.active_cycle_id
            
            if not cycle_id:
                return {"success": False, "error": "No active cycle"}
            
            # Obtener estado de la estrategia
            strategy_state = self.seraphim_strategy.get_state()
            
            # Verificar si el ciclo está activo
            is_active = (
                strategy_state.get("cycle_id") == cycle_id and
                strategy_state.get("cycle_phase") not in ["REBIRTH", "PREPARATION"]
            )
            
            # Obtener datos adicionales desde base de datos
            cycle_data = await self.database.get_data(f"cycle:{cycle_id}")
            
            return {
                "success": True,
                "cycle_id": cycle_id,
                "is_active": is_active,
                "cycle_phase": strategy_state.get("cycle_phase"),
                "strategy_state": strategy_state.get("state"),
                "cycle_performance": strategy_state.get("cycle_performance", {}),
                "cycle_start_time": strategy_state.get("cycle_start_time"),
                "additional_data": cycle_data
            }
            
        except Exception as e:
            logger.error(f"Error al obtener estado del ciclo {cycle_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def check_system_health(self) -> float:
        """
        Verificar salud completa del sistema.
        
        Returns:
            Puntuación de salud del sistema (0.0-1.0)
        """
        try:
            health_score = 1.0
            
            # 1. Verificar conexiones de componentes
            components_status = await asyncio.gather(
                self.seraphim_strategy.initialize() if not self.seraphim_strategy else True,
                self.buddha_integrator.get_status(),
                self.classifier.is_ready(),
                self.circuit_breaker.get_status(),
                self.checkpoint_manager.get_status(),
                self.load_balancer.get_status(),
                self.database.is_connected(),
                self.oracle.verify_connection(),
                self.alert_manager.is_operational(),
                return_exceptions=True
            )
            
            # Penalizar por cada componente con problemas
            for i, status in enumerate(components_status):
                if isinstance(status, Exception):
                    logger.warning(f"Componente {i} falló verificación: {str(status)}")
                    health_score *= 0.8
                elif status in [False, None] or (isinstance(status, dict) and not status.get("success", False)):
                    logger.warning(f"Componente {i} no está saludable: {status}")
                    health_score *= 0.9
            
            # 2. Verificar errores recientes
            recent_errors = await self._get_recent_errors()
            if recent_errors > 0:
                # Penalizar por errores recientes
                health_score *= max(0.5, 1.0 - (recent_errors / 10))
            
            # 3. Verificar capacidad de recuperación
            recovery_test = await self._perform_recovery_test()
            if not recovery_test:
                health_score *= 0.7
            
            # Actualizar salud del sistema
            self.system_health = health_score
            logger.info(f"Verificación de salud completa: {health_score:.2f}")
            
            return health_score
            
        except Exception as e:
            logger.error(f"Error al verificar salud del sistema: {str(e)}")
            self.system_health = 0.5  # Actualizar como precaución
            return 0.5
    
    async def _get_recent_errors(self) -> int:
        """
        Obtener conteo de errores recientes.
        
        Returns:
            Número de errores recientes
        """
        # Implementación simulada para demo
        return 0
    
    async def _perform_recovery_test(self) -> bool:
        """
        Realizar prueba de recuperación.
        
        Returns:
            True si la recuperación funciona correctamente
        """
        # Implementación simulada para demo
        return True
    
    async def run_autonomous_operation(self, duration_hours: Optional[int] = None) -> Dict[str, Any]:
        """
        Ejecutar operación autónoma por un período especificado.
        
        Args:
            duration_hours: Duración máxima en horas, o None para indefinido
            
        Returns:
            Resultado de la operación autónoma
        """
        try:
            # Configurar operación autónoma
            self.auto_cycle_enabled = True
            
            # Definir tiempo límite si se especificó
            end_time = None
            if duration_hours is not None:
                end_time = datetime.now() + timedelta(hours=duration_hours)
                logger.info(f"Iniciando operación autónoma por {duration_hours} horas")
            else:
                logger.info("Iniciando operación autónoma indefinida")
            
            # Ciclo principal de operación autónoma
            cycles_started = 0
            
            # Para demo, simular solo unos pocos ciclos
            max_simulated_cycles = 3
            
            # En un entorno real, este bucle se ejecutaría en un
            # trabajo en segundo plano o un servicio separado
            while self.auto_cycle_enabled:
                # Verificar si se alcanzó el tiempo límite
                if end_time and datetime.now() >= end_time:
                    logger.info("Alcanzado tiempo límite de operación autónoma")
                    break
                
                # Verificar salud del sistema
                health = await self.check_system_health()
                if health < 0.7:
                    logger.warning(f"Salud del sistema demasiado baja para operación autónoma: {health:.2f}")
                    break
                
                # Iniciar ciclo si no hay uno activo
                if not self.active_cycle_id:
                    # Iniciar nuevo ciclo
                    cycle_start_result = await self.start_trading_cycle()
                    
                    if cycle_start_result.get("success", False):
                        cycles_started += 1
                        logger.info(f"Iniciado ciclo autónomo #{cycles_started}: {self.active_cycle_id}")
                    else:
                        logger.error(f"Error al iniciar ciclo autónomo: {cycle_start_result.get('error', 'Unknown error')}")
                        # Esperar antes de reintentar
                        await asyncio.sleep(600)  # 10 minutos
                        continue
                
                # Procesar ciclo activo
                process_result = await self.process_cycle()
                
                if not process_result.get("success", False):
                    logger.warning(f"Error al procesar ciclo: {process_result.get('error', 'Unknown error')}")
                
                # Simular pausa entre actualizaciones
                await asyncio.sleep(2)  # Reducido para simulación
                
                # Para demo, limitar número de ciclos
                if cycles_started >= max_simulated_cycles:
                    logger.info(f"Completados {cycles_started} ciclos simulados")
                    break
            
            # Desactivar operación autónoma
            self.auto_cycle_enabled = False
            
            return {
                "success": True,
                "cycles_started": cycles_started,
                "cycles_completed": self.completed_cycles_count,
                "total_profit": self.total_realized_profit,
                "duration": str(datetime.now() - self.start_time)
            }
            
        except Exception as e:
            logger.error(f"Error en operación autónoma: {str(e)}")
            self.auto_cycle_enabled = False
            return {"success": False, "error": str(e)}
    
    def stop_autonomous_operation(self) -> Dict[str, Any]:
        """
        Detener operación autónoma.
        
        Returns:
            Resultado de la operación
        """
        self.auto_cycle_enabled = False
        logger.info("Operación autónoma detenida")
        return {"success": True, "status": "autonomous_operation_stopped"}
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """
        Obtener visión general del sistema.
        
        Returns:
            Visión general del sistema
        """
        try:
            # Obtener estado del ciclo activo si existe
            cycle_status = {}
            if self.active_cycle_id:
                cycle_status = await self.get_cycle_status(self.active_cycle_id)
            
            # Estadísticas del sistema
            system_stats = {
                "uptime": str(datetime.now() - self.start_time),
                "health_score": self.system_health,
                "active_cycles_count": self.active_cycles_count,
                "completed_cycles_count": self.completed_cycles_count,
                "total_profit": self.total_realized_profit,
                "auto_cycle_enabled": self.auto_cycle_enabled,
                "orchestrator_state": self.state.name,
                "last_checkpoint": self.last_checkpoint_time.isoformat() if self.last_checkpoint_time else None
            }
            
            # Verificar Buddha
            buddha_status = await self.buddha_integrator.get_status()
            
            # Clasificación actual
            top_cryptos = await self.classifier.get_top_opportunities(limit=5)
            crypto_names = [crypto.get("symbol", "unknown") for crypto in top_cryptos]
            
            return {
                "success": True,
                "system_stats": system_stats,
                "active_cycle": cycle_status if self.active_cycle_id else None,
                "buddha_status": buddha_status.get("status", "unknown"),
                "top_cryptos": crypto_names,
                "oracle_prediction": await self.oracle.get_current_prediction()
            }
            
        except Exception as e:
            logger.error(f"Error al obtener visión general del sistema: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_state(self) -> Dict[str, Any]:
        """
        Obtener estado completo del orquestador para checkpoints.
        
        Returns:
            Estado serializable del orquestador
        """
        return {
            "orchestrator_id": self.instance_id,
            "state": self.state.name,
            "start_time": self.start_time.isoformat(),
            "uptime": str(datetime.now() - self.start_time),
            "active_cycle_id": self.active_cycle_id,
            "active_cycles_count": self.active_cycles_count,
            "completed_cycles_count": self.completed_cycles_count,
            "total_realized_profit": self.total_realized_profit,
            "system_health": self.system_health,
            "auto_cycle_enabled": self.auto_cycle_enabled,
            "human_behavior": self.get_human_behavior() if self.behavior_engine else {}
        }
        
    def get_human_behavior(self) -> Dict[str, Any]:
        """
        Obtener configuración actual del comportamiento humano simulado.
        
        Returns:
            Detalles del comportamiento humano de Gabriel
        """
        if not self.behavior_engine:
            return {"available": False}
            
        try:
            return {
                "available": True,
                "emotional_state": self.behavior_engine.emotional_state.name,
                "risk_tolerance": self.behavior_engine.risk_tolerance.name,
                "decision_style": self.behavior_engine.decision_style.name,
                "current_characteristics": self.behavior_engine.get_current_characteristics()
            }
        except Exception as e:
            logger.error(f"Error al obtener estado del comportamiento humano: {str(e)}")
            return {"available": True, "error": str(e)}
    
    async def randomize_human_behavior(self) -> Dict[str, Any]:
        """
        Aleatorizar las características del comportamiento humano simulado.
        
        Este método cambia aleatoriamente el estado emocional, tolerancia al riesgo
        y estilo de decisión del motor de comportamiento humano Gabriel, lo que influirá
        en todas las decisiones de trading subsiguientes en el sistema.
        
        Returns:
            Resultado de la aleatorización
        """
        if not self.behavior_engine:
            logger.warning("Motor de comportamiento humano no disponible para aleatorizar")
            return {"success": False, "error": "Human behavior engine not available"}
            
        try:
            # Aleatorizar el comportamiento
            new_characteristics = self.behavior_engine.randomize()
            
            # Si la estrategia también tiene una referencia, actualizarla
            if hasattr(self.seraphim_strategy, 'behavior_engine'):
                self.seraphim_strategy.behavior_engine = self.behavior_engine
            
            # Asegurarnos de que el CodiciaManager tenga el motor actualizado
            if hasattr(self, 'codicia_manager') and self.codicia_manager:
                self.codicia_manager.behavior_engine = self.behavior_engine
                logger.debug("Motor de comportamiento actualizado en CodiciaManager")
            
            # Actualizar en cualquier otro componente que utilice el behavior_engine
            if hasattr(self, 'exchange_adapter') and hasattr(self.exchange_adapter, 'behavior_engine'):
                self.exchange_adapter.behavior_engine = self.behavior_engine
                
            logger.info(f"Comportamiento humano aleatorizado: {self.behavior_engine.emotional_state.name} / {self.behavior_engine.risk_tolerance.name}")
            
            return {
                "success": True,
                "human_behavior": new_characteristics,
                "emotional_state": self.behavior_engine.emotional_state.name,
                "risk_tolerance": self.behavior_engine.risk_tolerance.name,
                "decision_style": self.behavior_engine.decision_style.name if hasattr(self.behavior_engine, 'decision_style') else "NEUTRAL"
            }
            
        except Exception as e:
            logger.error(f"Error al aleatorizar comportamiento humano: {str(e)}")
            return {"success": False, "error": str(e)}