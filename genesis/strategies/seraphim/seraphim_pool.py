"""
Estrategia Seraphim Pool - El núcleo divino del Sistema Genesis Ultra.

Esta estrategia trascendental integra comportamiento humano simulado con capacidades
divinas de análisis y procesamiento, implementando el concepto HumanPoolTrader
en su forma más avanzada y celestial.

La estrategia Seraphim Pool representa la cúspide de la evolución del Sistema Genesis,
combinando:
- Comportamiento humano simulado para evitar detección algorítmica
- Sabiduría Buddha para análisis de mercado superior
- Clasificación trascendental para selección perfecta de activos
- Gestión de riesgo adaptativo con múltiples capas de protección
- Sistema de ciclos con capital limitado y distribución equitativa

Autor: Genesis AI Assistant
Versión: 1.0.0 (Divina)
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum, auto
import uuid

# Componentes Genesis
from genesis.strategies.base_strategy import BaseStrategy
from genesis.risk.adaptive_risk_manager import AdaptiveRiskManager
from genesis.trading.codicia_manager import CodiciaManager
from genesis.analysis.transcendental_crypto_classifier import TranscendentalCryptoClassifier
from genesis.accounting.capital_scaling import CapitalScalingManager
from genesis.db.transcendental_database import TranscendentalDatabase
from genesis.cloud.circuit_breaker_v4 import CloudCircuitBreakerV4
from genesis.cloud.distributed_checkpoint_v4 import DistributedCheckpointManagerV4
from genesis.notifications.alert_manager import AlertManager
from genesis.trading.buddha_integrator import BuddhaIntegrator
from genesis.trading.human_behavior_engine import GabrielBehaviorEngine, EmotionalState

# Configuración de logging
logger = logging.getLogger(__name__)

class SeraphimState(Enum):
    """Estados divinos de la estrategia Seraphim."""
    DORMANT = auto()          # En espera de inicialización
    CONTEMPLATING = auto()    # Analizando el mercado
    ILLUMINATED = auto()      # Señal de trading identificada
    ASCENDING = auto()        # Ejecutando operación (entrada)
    TRANSCENDING = auto()     # Manteniendo posición
    DESCENDING = auto()       # Cerrando posición
    REFLECTING = auto()       # Evaluando resultados
    DISTRIBUTING = auto()     # Distribuyendo ganancias
    RESTING = auto()          # Ciclo completado, en pausa

class CyclePhase(Enum):
    """Fases del ciclo celestial de trading."""
    PREPARATION = auto()      # Preparando el ciclo
    REVELATION = auto()       # Revelando oportunidades
    EXECUTION = auto()        # Ejecutando operaciones
    GUARDIANSHIP = auto()     # Protegiendo posiciones
    ASCENSION = auto()        # Completando ciclo con éxito
    REFLECTION = auto()       # Evaluando resultados
    DISTRIBUTION = auto()     # Distribuyendo ganancias
    REBIRTH = auto()          # Preparando nuevo ciclo

class HumanBehaviorPattern(Enum):
    """Patrones de comportamiento humano para la simulación divina."""
    CAUTIOUS = auto()         # Precaución extrema, operaciones limitadas
    BALANCED = auto()         # Equilibrio entre riesgo y recompensa
    OPPORTUNISTIC = auto()    # Aprovechamiento de oportunidades claras
    CONTEMPLATIVE = auto()    # Análisis profundo antes de actuar
    PROTECTIVE = auto()       # Enfoque en protección de capital
    PATIENT = auto()          # Espera de condiciones perfectas

class SeraphimPool(BaseStrategy):
    """
    Estrategia Seraphim Pool - Implementación trascendental del HumanPoolTrader.
    
    Esta estrategia representa la integración perfecta de comportamiento humano
    simulado con capacidades divinas del Sistema Genesis, operando en ciclos
    de trading con capital limitado y distribución equitativa de ganancias.
    """
    
    def __init__(self):
        """Inicializar la estrategia Seraphim Pool con propiedades divinas."""
        super().__init__(name="Seraphim Pool Strategy")
        
        # Componentes integrados
        self.risk_manager: Optional[AdaptiveRiskManager] = None
        self.codicia_manager: Optional[CodiciaManager] = None
        self.classifier: Optional[TranscendentalCryptoClassifier] = None
        self.capital_manager: Optional[CapitalScalingManager] = None
        self.database: Optional[TranscendentalDatabase] = None
        self.circuit_breaker: Optional[CloudCircuitBreakerV4] = None
        self.checkpoint_manager: Optional[DistributedCheckpointManagerV4] = None
        self.alert_manager: Optional[AlertManager] = None
        self.buddha_integrator: Optional[BuddhaIntegrator] = None
        self.behavior_engine: Optional[GabrielBehaviorEngine] = None
        
        # Estado y configuración
        self.state = SeraphimState.DORMANT
        self.cycle_phase = CyclePhase.PREPARATION
        self.cycle_id = str(uuid.uuid4())
        self.current_behavior = HumanBehaviorPattern.BALANCED
        
        # Configuración del ciclo
        self.cycle_capital = 150.0  # Capital por ciclo
        self.cycle_target_return = 0.085  # 8.5% objetivo
        self.cycle_max_loss = 0.02  # 2% pérdida máxima
        self.cycle_duration = timedelta(hours=24)  # Duración máxima
        
        # Datos operativos
        self.cycle_start_time = None
        self.selected_assets = []
        self.asset_allocations = {}
        self.open_positions = {}
        self.cycle_performance = {
            "starting_capital": 0.0,
            "current_capital": 0.0,
            "realized_profit": 0.0,
            "unrealized_profit": 0.0,
            "trades_count": 0,
            "successful_trades": 0,
            "roi_percentage": 0.0
        }
        
        # Participantes del pool (simulado inicialmente)
        self.pool_participants = [
            {"id": "participant_1", "name": "Metatron", "share": 0.2},
            {"id": "participant_2", "name": "Gabriel", "share": 0.2},
            {"id": "participant_3", "name": "Uriel", "share": 0.2},
            {"id": "participant_4", "name": "Rafael", "share": 0.2},
            {"id": "participant_5", "name": "Miguel", "share": 0.2}
        ]
        
        logger.info("Estrategia Seraphim Pool inicializada en estado DORMANT")
    
    async def initialize(self) -> bool:
        """
        Inicializar todos los componentes divinos de la estrategia.
        
        Returns:
            True si la inicialización fue exitosa
        """
        try:
            # Inicializar gestores y componentes
            self.risk_manager = AdaptiveRiskManager()
            self.codicia_manager = CodiciaManager()
            self.classifier = TranscendentalCryptoClassifier()
            self.capital_manager = CapitalScalingManager(base_capital=10000.0)
            self.database = TranscendentalDatabase()
            self.circuit_breaker = CloudCircuitBreakerV4()
            self.checkpoint_manager = DistributedCheckpointManagerV4()
            self.alert_manager = AlertManager()
            self.buddha_integrator = BuddhaIntegrator()
            self.behavior_engine = GabrielBehaviorEngine()
            
            # Inicializar cada componente
            await self.risk_manager.initialize()
            await self.codicia_manager.initialize()
            await self.classifier.initialize()
            await self.capital_manager.initialize()
            await self.database.initialize()
            await self.circuit_breaker.initialize()
            await self.checkpoint_manager.initialize()
            await self.alert_manager.initialize()
            await self.buddha_integrator.initialize()
            
            # Inicializar motor de comportamiento humano
            await self.behavior_engine.initialize()
            
            # Actualizar estado
            self.state = SeraphimState.CONTEMPLATING
            logger.info("Estrategia Seraphim Pool inicializada correctamente, estado: CONTEMPLATING")
            
            # Cargar configuración desde base de datos
            await self._load_configuration()
            
            # Registrar estrategia en el checkpoint manager
            await self.checkpoint_manager.register_strategy(self.name, self.get_state())
            
            return True
        except Exception as e:
            logger.error(f"Error al inicializar estrategia Seraphim Pool: {str(e)}")
            return False
    
    async def _load_configuration(self) -> None:
        """Cargar configuración desde la base de datos trascendental."""
        try:
            config = await self.database.get_data("seraphim_pool_config")
            if config:
                self.cycle_capital = config.get("cycle_capital", self.cycle_capital)
                self.cycle_target_return = config.get("cycle_target_return", self.cycle_target_return)
                self.cycle_max_loss = config.get("cycle_max_loss", self.cycle_max_loss)
                
                # Cargar participantes si existen
                participants = await self.database.get_data("seraphim_pool_participants")
                if participants:
                    self.pool_participants = participants
                
                logger.info(f"Configuración cargada: capital por ciclo ${self.cycle_capital:.2f}, "
                           f"objetivo {self.cycle_target_return*100:.1f}%, "
                           f"pérdida máxima {self.cycle_max_loss*100:.1f}%")
        except Exception as e:
            logger.warning(f"No se pudo cargar configuración, usando valores predeterminados: {str(e)}")
    
    async def start_cycle(self) -> bool:
        """
        Iniciar un nuevo ciclo celestial de trading.
        
        Returns:
            True si el ciclo se inició correctamente
        """
        try:
            # Protección: verificar que no haya ciclo activo
            if self.cycle_phase != CyclePhase.PREPARATION and self.cycle_phase != CyclePhase.REBIRTH:
                logger.warning(f"No se puede iniciar ciclo, fase actual: {self.cycle_phase}")
                return False
            
            # Generar nuevo ID de ciclo
            self.cycle_id = str(uuid.uuid4())
            self.cycle_start_time = datetime.now()
            
            # Restablecer datos del ciclo
            self.cycle_performance = {
                "starting_capital": self.cycle_capital,
                "current_capital": self.cycle_capital,
                "realized_profit": 0.0,
                "unrealized_profit": 0.0,
                "trades_count": 0,
                "successful_trades": 0,
                "roi_percentage": 0.0
            }
            
            # Limpiar datos anteriores
            self.selected_assets = []
            self.asset_allocations = {}
            self.open_positions = {}
            
            # Actualizar fase
            self.cycle_phase = CyclePhase.REVELATION
            
            # Definir comportamiento para este ciclo (simulando patrón humano)
            self.current_behavior = random.choice(list(HumanBehaviorPattern))
            
            # Crear registro en base de datos
            cycle_data = {
                "cycle_id": self.cycle_id,
                "start_time": self.cycle_start_time.isoformat(),
                "capital": self.cycle_capital,
                "target_return": self.cycle_target_return,
                "max_loss": self.cycle_max_loss,
                "behavior": self.current_behavior.name,
                "status": "active"
            }
            await self.database.set_data(f"cycle:{self.cycle_id}", cycle_data)
            
            # Notificar participantes
            await self._notify_cycle_start()
            
            logger.info(f"Ciclo iniciado: ID {self.cycle_id}, "
                       f"capital ${self.cycle_capital:.2f}, "
                       f"comportamiento {self.current_behavior.name}")
            
            # Crear checkpoint
            await self.checkpoint_manager.create_checkpoint(
                f"cycle_start_{self.cycle_id}", 
                {"strategy": self.name, "state": self.get_state()}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error al iniciar ciclo: {str(e)}")
            return False
    
    async def _notify_cycle_start(self) -> None:
        """Notificar a los participantes del inicio del ciclo."""
        try:
            for participant in self.pool_participants:
                notification = {
                    "type": "cycle_start",
                    "recipient": participant["id"],
                    "subject": "Nuevo ciclo de trading iniciado",
                    "message": (f"Un nuevo ciclo de trading ha comenzado con ID {self.cycle_id}. "
                              f"Capital asignado: ${self.cycle_capital:.2f}. "
                              f"Objetivo: {self.cycle_target_return*100:.1f}%.")
                }
                await self.alert_manager.send_notification(notification)
                
            logger.debug(f"Notificaciones de inicio de ciclo enviadas a {len(self.pool_participants)} participantes")
        except Exception as e:
            logger.warning(f"Error al enviar notificaciones de inicio: {str(e)}")
    
    async def analyze_market(self) -> Dict[str, Any]:
        """
        Realizar análisis de mercado divino combinando todos los componentes.
        
        Returns:
            Diccionario con resultados del análisis
        """
        try:
            # Verificar fase correcta
            if self.cycle_phase != CyclePhase.REVELATION:
                logger.warning(f"Fase incorrecta para análisis de mercado: {self.cycle_phase}")
                return {"success": False, "error": "Incorrect cycle phase"}
            
            # Simular comportamiento humano: pausa estratégica
            await self._simulate_human_contemplation()
            
            # 1. Análisis con TranscendentalCryptoClassifier
            classifier_results = await self.classifier.get_top_opportunities(limit=5)
            
            # 2. Análisis con Buddha AI
            buddha_analysis = await self.buddha_integrator.analyze_market_conditions()
            
            # 3. Refinamiento con AdaptiveRiskManager
            risk_assessment = await self.risk_manager.evaluate_market_conditions()
            
            # 4. Aplicar patrón de comportamiento humano
            filtered_assets = await self._apply_human_behavior_filter(
                classifier_results, 
                buddha_analysis,
                risk_assessment
            )
            
            # Almacenar activos seleccionados
            self.selected_assets = filtered_assets
            
            # Crear asignación de capital simulando decisión humana
            await self._allocate_capital_to_assets()
            
            # Actualizar fase
            self.cycle_phase = CyclePhase.EXECUTION
            
            # Registrar resultados en base de datos
            analysis_results = {
                "cycle_id": self.cycle_id,
                "timestamp": datetime.now().isoformat(),
                "selected_assets": self.selected_assets,
                "allocations": self.asset_allocations,
                "buddha_sentiment": buddha_analysis.get("market_sentiment", "neutral"),
                "risk_level": risk_assessment.get("overall_risk", "medium")
            }
            await self.database.set_data(f"analysis:{self.cycle_id}", analysis_results)
            
            logger.info(f"Análisis de mercado completado: {len(self.selected_assets)} activos seleccionados")
            
            return {
                "success": True,
                "selected_assets": self.selected_assets,
                "allocations": self.asset_allocations,
                "buddha_insights": buddha_analysis.get("insights", [])
            }
            
        except Exception as e:
            logger.error(f"Error en análisis de mercado: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _simulate_human_contemplation(self) -> None:
        """Simular contemplación humana con pausa estratégica."""
        # Determinar tiempo de contemplación basado en comportamiento actual
        contemplation_times = {
            HumanBehaviorPattern.CAUTIOUS: (20, 40),       # (min, max) segundos
            HumanBehaviorPattern.BALANCED: (10, 25),
            HumanBehaviorPattern.OPPORTUNISTIC: (5, 15),
            HumanBehaviorPattern.CONTEMPLATIVE: (30, 60),
            HumanBehaviorPattern.PROTECTIVE: (15, 35),
            HumanBehaviorPattern.PATIENT: (25, 50)
        }
        
        # Si tenemos un motor de comportamiento humano, usarlo para el tiempo de contemplación
        if hasattr(self, 'behavior_engine') and self.behavior_engine:
            try:
                # Usar el motor Gabriel para simular retraso humano realista
                await self.behavior_engine.simulate_human_delay("analysis")
                logger.debug(f"Usando motor Gabriel para contemplación humana, estado emocional: {self.behavior_engine.emotional_state.name}")
                return
            except Exception as e:
                logger.warning(f"Error al usar motor Gabriel para contemplación: {e}")
                # Continuar con el método original si falla
        
        # Método original como fallback
        behavior_time = contemplation_times.get(
            self.current_behavior, 
            (10, 30)  # Tiempo predeterminado
        )
        
        # Calcular tiempo aleatorio dentro del rango
        contemplation_time = random.uniform(behavior_time[0], behavior_time[1])
        
        logger.debug(f"Simulando contemplación humana, comportamiento: {self.current_behavior.name}, "
                   f"tiempo: {contemplation_time:.1f} segundos")
        
        # En un entorno real, descomentar la siguiente línea
        # await asyncio.sleep(contemplation_time)
        
        # Actualizar estado durante la contemplación
        self.state = SeraphimState.CONTEMPLATING
    
    async def _apply_human_behavior_filter(
        self, 
        classifier_results: List[Dict[str, Any]],
        buddha_analysis: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Aplicar filtro de comportamiento humano a resultados del clasificador.
        
        Args:
            classifier_results: Resultados del clasificador transcendental
            buddha_analysis: Análisis de Buddha AI
            risk_assessment: Evaluación del gestor de riesgo
            
        Returns:
            Lista filtrada de activos
        """
        # Log para diagnóstico
        logger.debug(f"Aplicando filtro de comportamiento {self.current_behavior.name}")
        logger.debug(f"Activos antes de filtrar: {len(classifier_results)}")
        
        # Obtener sentimiento general del mercado desde Buddha
        market_sentiment = buddha_analysis.get("market_sentiment", "neutral")
        market_risk = risk_assessment.get("overall_risk", "medium")
        
        # Si tenemos un motor de comportamiento humano Gabriel, usarlo para decisiones
        if hasattr(self, 'behavior_engine') and self.behavior_engine:
            try:
                filtered = []
                # Evaluar cada activo usando el motor Gabriel para decisiones humanas realistas
                for asset in classifier_results:
                    # Calcular score de oportunidad basado en múltiples factores
                    opportunity_score = (
                        asset.get("profit_potential", 0.5) * 0.4 +
                        (1 - asset.get("risk_score", 0.5)) * 0.3 +
                        asset.get("trend_strength", 0.5) * 0.2 +
                        asset.get("stability_score", 0.5) * 0.1
                    )
                    
                    # Usar el motor Gabriel para decisión humana realista
                    should_enter, reason = await self.behavior_engine.should_enter_trade(
                        opportunity_score=opportunity_score,
                        asset_data=asset
                    )
                    
                    if should_enter:
                        filtered.append(asset)
                        logger.debug(f"Gabriel aprueba activo {asset.get('symbol')}: {reason}")
                    else:
                        logger.debug(f"Gabriel rechaza activo {asset.get('symbol')}: {reason}")
                
                # Si el motor rechazó todos los activos, usar al menos uno (los humanos raramente rechazan todo)
                if not filtered and classifier_results:
                    best_asset = max(classifier_results, key=lambda x: x.get("profit_potential", 0))
                    filtered.append(best_asset)
                    logger.debug(f"Gabriel finalmente acepta activo {best_asset.get('symbol')} a pesar del rechazo inicial")
                
                # Limitar a máximo 5 activos (simulando limitación cognitiva humana)
                filtered = filtered[:5]
                
                # Log para diagnóstico
                logger.debug(f"Activos después de filtrar con Gabriel: {len(filtered)}")
                symbols = [asset.get("symbol", "unknown") for asset in filtered]
                logger.info(f"Activos seleccionados por Gabriel: {symbols}")
                
                return filtered
            
            except Exception as e:
                logger.warning(f"Error al usar motor Gabriel para filtrar activos: {e}")
                # Continuar con el método original si falla
                logger.debug("Usando método de filtrado alternativo")
        
        # Método original como fallback
        # Comportamientos diferentes llevan a diferentes criterios de filtrado
        if self.current_behavior == HumanBehaviorPattern.CAUTIOUS:
            # Más conservador: solo activos de menor riesgo
            filtered = [asset for asset in classifier_results 
                      if asset.get("risk_score", 1.0) < 0.6
                      and asset.get("volatility", 1.0) < 0.7]
                      
        elif self.current_behavior == HumanBehaviorPattern.PROTECTIVE:
            # Enfocado en protección: solo los más estables
            filtered = [asset for asset in classifier_results 
                      if asset.get("stability_score", 0) > 0.7]
                      
        elif self.current_behavior == HumanBehaviorPattern.OPPORTUNISTIC:
            # Busca oportunidades: enfoque en potencial de ganancia
            filtered = [asset for asset in classifier_results
                      if asset.get("profit_potential", 0) > 0.65]
        
        elif self.current_behavior == HumanBehaviorPattern.CONTEMPLATIVE:
            # Análisis profundo: considerar múltiples factores incluyendo Buddha
            filtered = []
            for asset in classifier_results:
                symbol = asset.get("symbol", "unknown")
                
                # Verificar si Buddha tiene insights específicos para este activo
                has_positive_insight = False
                for insight in buddha_analysis.get("insights", []):
                    if insight.get("symbol") == symbol and insight.get("sentiment") == "positive":
                        has_positive_insight = True
                        break
                
                # Considerar activos con buena estabilidad o insight positivo
                if asset.get("stability_score", 0) > 0.6 or has_positive_insight:
                    filtered.append(asset)
        
        elif self.current_behavior == HumanBehaviorPattern.BALANCED:
            # Equilibrado: considerar riesgo y recompensa por igual
            filtered = [asset for asset in classifier_results
                      if asset.get("risk_reward_ratio", 0) > 1.2]
        
        elif self.current_behavior == HumanBehaviorPattern.PATIENT:
            # Paciente: solo los mejores activos con tendencia clara
            filtered = []
            for asset in classifier_results:
                if (asset.get("trend_strength", 0) > 0.8 and 
                    asset.get("trend_direction", "") in ["up", "down"]):
                    filtered.append(asset)
                
        else:
            # Comportamiento por defecto: usar todos los activos recomendados por el clasificador
            filtered = classifier_results
        
        # Aplicar sentimiento de mercado según Buddha
        if market_sentiment == "bearish" and market_risk == "high":
            # En mercado bajista y riesgo alto, ser más selectivo
            filtered = [asset for asset in filtered
                      if asset.get("bear_market_resilience", 0) > 0.7]
        
        # Simular rechazo subjetivo (elemento humano)
        if len(filtered) > 2:
            # A veces los humanos rechazan una opción por "instinto"
            subjective_rejection_idx = random.randint(0, len(filtered) - 1)
            rejected_asset = filtered.pop(subjective_rejection_idx)
            logger.debug(f"Rechazo subjetivo de activo: {rejected_asset.get('symbol', 'unknown')}")
        
        # Limitar a máximo 5 activos (simulando limitación cognitiva humana)
        filtered = filtered[:5]
        
        # Log para diagnóstico
        logger.debug(f"Activos después de filtrar: {len(filtered)}")
        symbols = [asset.get("symbol", "unknown") for asset in filtered]
        logger.info(f"Activos seleccionados: {symbols}")
        
        return filtered
    
    async def _allocate_capital_to_assets(self) -> None:
        """Asignar capital a los activos seleccionados usando patrón humano."""
        if not self.selected_assets:
            logger.warning("No hay activos seleccionados para asignar capital")
            return
        
        # Diferentes patrones de asignación según comportamiento
        allocations = {}
        remaining_capital = self.cycle_capital
        
        if self.current_behavior == HumanBehaviorPattern.CAUTIOUS:
            # Asignación conservadora: distribuir capital de forma equilibrada
            # pero reservar una parte sin invertir
            reserved_percentage = 0.3  # 30% sin invertir
            investable_capital = self.cycle_capital * (1 - reserved_percentage)
            base_allocation = investable_capital / len(self.selected_assets)
            
            for asset in self.selected_assets:
                symbol = asset.get("symbol", "unknown")
                allocations[symbol] = base_allocation
                remaining_capital -= base_allocation
                
        elif self.current_behavior == HumanBehaviorPattern.OPPORTUNISTIC:
            # Concentrar en mejores oportunidades
            # Ordenar por potencial de ganancia
            sorted_assets = sorted(
                self.selected_assets, 
                key=lambda x: x.get("profit_potential", 0), 
                reverse=True
            )
            
            # Asignar más a los mejores
            allocation_weights = [0.4, 0.3, 0.15, 0.1, 0.05]  # Pesos decrecientes
            for i, asset in enumerate(sorted_assets):
                if i >= len(allocation_weights):
                    break
                    
                symbol = asset.get("symbol", "unknown")
                asset_allocation = self.cycle_capital * allocation_weights[i]
                allocations[symbol] = asset_allocation
                remaining_capital -= asset_allocation
                
        else:  # BALANCED o comportamientos no especializados
            # Asignación relativamente equilibrada con variaciones humanas
            base_allocation = self.cycle_capital / len(self.selected_assets)
            
            for asset in self.selected_assets:
                symbol = asset.get("symbol", "unknown")
                # Variación humana: ligera aleatorización en asignaciones
                variation = random.uniform(0.8, 1.2)
                asset_allocation = base_allocation * variation
                
                # Asegurarse de no exceder el capital
                if asset_allocation > remaining_capital:
                    asset_allocation = remaining_capital
                    
                allocations[symbol] = asset_allocation
                remaining_capital -= asset_allocation
        
        # Guardar asignaciones
        self.asset_allocations = allocations
        
        # El capital no asignado queda como reserva
        logger.info(f"Capital asignado: ${self.cycle_capital - remaining_capital:.2f}, "
                   f"Reserva: ${remaining_capital:.2f}")
    
    async def execute_trades(self) -> Dict[str, Any]:
        """
        Ejecutar operaciones de trading según la estrategia divina.
        
        Returns:
            Diccionario con resultados de las operaciones
        """
        try:
            # Verificar fase correcta
            if self.cycle_phase != CyclePhase.EXECUTION:
                logger.warning(f"Fase incorrecta para ejecución: {self.cycle_phase}")
                return {"success": False, "error": "Incorrect cycle phase"}
            
            # Actualizar estado
            self.state = SeraphimState.ASCENDING
            
            # Simular comportamiento humano al ejecutar
            await self._simulate_human_execution()
            
            # Ejecutar operaciones
            executed_trades = []
            
            for symbol, allocation in self.asset_allocations.items():
                # Consultar precio actual (simulado para demo)
                current_price = await self._get_current_price(symbol)
                
                # Calcular cantidad a comprar
                quantity = allocation / current_price
                
                # Preparar orden
                order_params = {
                    "symbol": symbol,
                    "side": "BUY",
                    "quantity": quantity,
                    "price": current_price,
                    "timestamp": datetime.now().isoformat(),
                    "cycle_id": self.cycle_id
                }
                
                # Validar con el CircuitBreaker
                breaker_result = await self.circuit_breaker.protect_operation(
                    lambda: self._validate_order(order_params),
                    operation_id=f"trade_{symbol}_{self.cycle_id}"
                )
                
                if breaker_result.get("status") == "approved":
                    # Ejecutar orden a través del CodiciaManager
                    order_result = await self.codicia_manager.place_order(order_params)
                    
                    if order_result.get("success", False):
                        # Registrar posición abierta
                        # Información para evaluar cierre de posición con comportamiento FEARFUL
                        now = datetime.now()
                        self.open_positions[symbol] = {
                            "entry_price": current_price,
                            "quantity": quantity,
                            "allocation": allocation,
                            "entry_time": now,
                            "entry_time_iso": now.isoformat(),  # Versión ISO para serialización
                            "order_id": order_result.get("order_id", "unknown"),
                            "recent_price_change": 0,  # Inicializar cambio reciente de precio
                            "last_price": current_price  # Último precio para cálculos futuros
                        }
                        
                        # Actualizar estadísticas
                        self.cycle_performance["trades_count"] += 1
                        
                        # Registrar operación exitosa
                        executed_trades.append({
                            "symbol": symbol,
                            "price": current_price,
                            "quantity": quantity,
                            "allocation": allocation,
                            "status": "executed"
                        })
                        
                        logger.info(f"Orden ejecutada: {symbol}, "
                                  f"precio: ${current_price:.2f}, "
                                  f"cantidad: {quantity:.6f}, "
                                  f"total: ${allocation:.2f}")
                    else:
                        # Registrar fallo
                        executed_trades.append({
                            "symbol": symbol,
                            "status": "failed",
                            "error": order_result.get("error", "Unknown error")
                        })
                        
                        logger.warning(f"Fallo al ejecutar orden: {symbol}, "
                                     f"error: {order_result.get('error', 'Unknown error')}")
                else:
                    # Orden rechazada por CircuitBreaker
                    executed_trades.append({
                        "symbol": symbol,
                        "status": "rejected",
                        "reason": breaker_result.get("reason", "Rejected by CircuitBreaker")
                    })
                    
                    logger.warning(f"Orden rechazada por CircuitBreaker: {symbol}, "
                                 f"razón: {breaker_result.get('reason', 'Unknown')}")
            
            # Actualizar fase si hay posiciones abiertas
            if self.open_positions:
                self.cycle_phase = CyclePhase.GUARDIANSHIP
                self.state = SeraphimState.TRANSCENDING
                
                # Crear checkpoint
                await self.checkpoint_manager.create_checkpoint(
                    f"trades_executed_{self.cycle_id}", 
                    {"strategy": self.name, "state": self.get_state()}
                )
            else:
                # No se pudo abrir ninguna posición, volver a fase de preparación
                self.cycle_phase = CyclePhase.PREPARATION
                self.state = SeraphimState.REFLECTING
                
                logger.warning("No se pudieron abrir posiciones, ciclo cancelado")
            
            # Registrar operaciones en base de datos
            trades_data = {
                "cycle_id": self.cycle_id,
                "timestamp": datetime.now().isoformat(),
                "trades": executed_trades,
                "open_positions": self.open_positions
            }
            await self.database.set_data(f"trades:{self.cycle_id}", trades_data)
            
            return {
                "success": True,
                "executed_trades": executed_trades,
                "open_positions": len(self.open_positions)
            }
            
        except Exception as e:
            logger.error(f"Error al ejecutar operaciones: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _simulate_human_execution(self) -> None:
        """Simular comportamiento humano al ejecutar órdenes."""
        # Si tenemos un motor de comportamiento humano Gabriel, usarlo para la ejecución
        if hasattr(self, 'behavior_engine') and self.behavior_engine:
            try:
                # Usar el motor Gabriel para simular retraso humano realista
                await self.behavior_engine.simulate_human_delay("trade_entry")
                logger.debug(f"Usando motor Gabriel para ejecución humana, estado emocional: {self.behavior_engine.emotional_state.name}")
                
                # Actualizar estado emocional después de entrar en operación
                # Este cambio emocional afectará decisiones futuras
                await self.behavior_engine.update_emotional_state("trade_execution", 1.0)
                logger.debug(f"Estado emocional actualizado a: {self.behavior_engine.emotional_state.name}")
                
                # Determinar el patrón de ejecución basado en el estado emocional
                emotional_state = self.behavior_engine.emotional_state
                
                # Los estados emocionales afectan cómo se ejecutan las operaciones
                if emotional_state == EmotionalState.IMPATIENT:
                    # Impaciente: Ejecuta órdenes rápidamente y con poca deliberación
                    execution_style = "rápido"
                    randomization_factor = 0.1  # Casi sin aleatorización (ejecuta todo junto)
                    logger.debug("Patrón de ejecución impaciente: operaciones rápidas y secuenciales")
                    
                elif emotional_state == EmotionalState.CONFIDENT:
                    # Confiado: Ejecuta decisivamente pero con cierto orden estratégico
                    execution_style = "confiado"
                    randomization_factor = 0.3
                    logger.debug("Patrón de ejecución confiado: operaciones decisivas y estratégicas")
                    
                elif emotional_state == EmotionalState.CAUTIOUS:
                    # Cauteloso: Mucho tiempo entre operaciones, muy espaciadas
                    execution_style = "cauteloso"
                    randomization_factor = 0.9  # Alta aleatorización (espaciamiento)
                    logger.debug("Patrón de ejecución cauteloso: operaciones muy espaciadas y deliberadas")
                    
                elif emotional_state == EmotionalState.ANXIOUS:
                    # Ansioso: Operaciones erráticas, a veces rápidas, a veces lentas
                    execution_style = "ansioso"
                    randomization_factor = 0.7
                    # Aleatoriamente puede decidir no ejecutar algunos activos
                    # Simular duda y decidir no invertir en algunos activos (25% de probabilidad)
                    filtered_allocations = {}
                    for symbol, allocation in self.asset_allocations.items():
                        if random.random() > 0.25:  # 75% de probabilidad de mantener
                            filtered_allocations[symbol] = allocation
                        else:
                            logger.debug(f"Motor Gabriel (ansioso) decidió no invertir en {symbol}")
                    
                    # Ajustar asignaciones eliminando algunos activos
                    if filtered_allocations:  # Asegurarse de no eliminar todos
                        self.asset_allocations = filtered_allocations
                    
                    logger.debug("Patrón de ejecución ansioso: operaciones erráticas con posible omisión")
                
                elif emotional_state == EmotionalState.FEARFUL:
                    # Temeroso: Posible reducción de tamaño de posiciones
                    execution_style = "temeroso"
                    randomization_factor = 0.8
                    
                    # Reducir el tamaño de las posiciones (entre 30% y 60%)
                    reduction_factor = random.uniform(0.4, 0.7)  # Mantendremos entre 40% y 70%
                    for symbol in self.asset_allocations:
                        self.asset_allocations[symbol]["amount"] *= reduction_factor
                        logger.debug(f"Motor Gabriel (temeroso) redujo posición en {symbol} al {int(reduction_factor*100)}%")
                    
                    logger.debug("Patrón de ejecución temeroso: operaciones reducidas en tamaño")
                
                else:
                    # Estado neutro o balanceado
                    execution_style = "balanceado"
                    randomization_factor = 0.5
                    logger.debug("Patrón de ejecución balanceado: operaciones con espaciamiento moderado")
                
                # Registrar decisión en métricas de comportamiento
                execution_metrics = {
                    "emotional_state": emotional_state.name,
                    "execution_style": execution_style,
                    "timestamp": datetime.now().isoformat(),
                    "cycle_id": self.cycle_id,
                    "randomization_factor": randomization_factor,
                    "asset_count": len(self.asset_allocations)
                }
                
                # Podríamos guardar en base de datos para análisis
                # await self.database.set_data(f"behavior_metrics:{self.cycle_id}:{datetime.now().isoformat()}", 
                #                             execution_metrics)
                
                # Aplicar aleatorización según estado emocional
                self.asset_allocations = dict(
                    sorted(
                        self.asset_allocations.items(),
                        key=lambda x: random.random() * randomization_factor
                    )
                )
                return
                
            except Exception as e:
                logger.warning(f"Error al usar motor Gabriel para ejecución: {e}")
                # Continuar con el método original si falla
        
        # Método original como fallback
        # Simular toma de decisiones no instantánea
        execution_delay = random.uniform(5, 15)  # segundos
        logger.debug(f"Simulando toma de decisión humana: {execution_delay:.1f} segundos")
        # await asyncio.sleep(execution_delay)  # Comentado para no bloquear en pruebas
        
        # Simular ejecución no simultánea de todas las órdenes
        self.asset_allocations = dict(
            sorted(
                self.asset_allocations.items(),
                key=lambda x: random.random()  # Orden aleatorio para simular humano
            )
        )
    
    async def _validate_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validar orden antes de enviarla.
        
        Args:
            order_params: Parámetros de la orden
            
        Returns:
            Diccionario con resultado de validación
        """
        symbol = order_params.get("symbol", "unknown")
        allocation = order_params.get("quantity", 0) * order_params.get("price", 0)
        
        # Si tenemos un motor de comportamiento humano Gabriel, usarlo para validación
        if hasattr(self, 'behavior_engine') and self.behavior_engine:
            try:
                # Obtener datos adicionales para validación
                volatility = await self._get_asset_volatility(symbol)
                
                # Preparar contexto para la decisión
                order_context = {
                    "symbol": symbol,
                    "allocation": allocation,
                    "allocation_percentage": allocation / self.cycle_capital * 100 if self.cycle_capital > 0 else 0,
                    "available_capital": self.cycle_performance["current_capital"],
                    "volatility": volatility,
                    "time_of_day": datetime.now().hour,
                    "cycle_progress": (datetime.now() - self.cycle_start_time).total_seconds() / self.cycle_duration.total_seconds() if self.cycle_start_time else 0
                }
                
                # Preparar datos para la validación por el motor Gabriel
                trade_data = {
                    'symbol': symbol,
                    'side': 'buy',  # Por defecto en este contexto
                    'amount': allocation,
                    'price': order_context.get('price', 0.0),
                    'reason': 'señal_seraphim',
                    'confidence': order_context.get('signal_strength', 0.6),
                    'market_data': {
                        'volatility': volatility,
                        'time_of_day': order_context.get('time_of_day', datetime.now().hour),
                        'cycle_progress': order_context.get('cycle_progress', 0.5)
                    },
                    'historical_performance': self.cycle_performance
                }
                
                # Obtener decisión del motor Gabriel
                decision, reason = await self.behavior_engine.validate_trade(trade_data)
                
                # Gabriel puede rechazar operaciones por razones emocionales o intuitivas
                if not decision:
                    logger.debug(f"Gabriel rechazó orden para {symbol}: {reason}")
                    return {
                        "valid": False,
                        "reason": reason
                    }
                
                # Gabriel también podría modificar la orden (ajustar tamaño)
                order_data = {
                    'symbol': symbol,
                    'side': 'buy',
                    'amount': allocation,
                    'price': order_context.get('price', 0.0),
                    'total_value': allocation * order_context.get('price', 0.0),
                    'confidence': order_context.get('signal_strength', 0.6),
                    'available_capital': self.cycle_performance["current_capital"]
                }
                
                # Ajustar tamaño de orden según comportamiento humano
                adjusted_order = await self.behavior_engine.adjust_order_size(order_data)
                
                # Si el tamaño fue ajustado por el motor de comportamiento
                if adjusted_order.get('amount', allocation) != allocation:
                    # Gabriel decidió modificar la orden (cantidad)
                    adjusted_amount = adjusted_order.get('amount', allocation)
                    adjustment_factors = adjusted_order.get('adjustment_factors', {})
                    
                    logger.debug(f"Gabriel modificó orden para {symbol}: {adjustment_factors.get('reason', 'ajuste intuitivo')}")
                    
                    # Actualizar parámetros de la orden
                    order_params["quantity"] = adjusted_amount
                    logger.info(f"Cantidad ajustada para {symbol}: {adjusted_amount} (original: {allocation})")
                
                logger.debug(f"Gabriel aprobó orden para {symbol}")
                return {"valid": True, "params": order_params}
                
            except Exception as e:
                logger.warning(f"Error al usar motor Gabriel para validar orden: {e}")
                # Continuar con el método original si falla
        
        # Método original como fallback
        # Aplicar validaciones simulando criterio humano
        if allocation > self.cycle_capital * 0.5:
            # Un humano raramente pondría más del 50% en un solo activo
            return {
                "valid": False, 
                "reason": f"Allocation too high for {symbol}: ${allocation:.2f}"
            }
        
        # Verificar que no exceda el capital disponible
        if allocation > self.cycle_performance["current_capital"]:
            return {
                "valid": False, 
                "reason": f"Insufficient capital for {symbol}: need ${allocation:.2f}"
            }
        
        # Otras validaciones subjetivas basadas en comportamiento humano
        if self.current_behavior == HumanBehaviorPattern.CAUTIOUS:
            # Más validaciones para comportamiento cauto
            volatility = await self._get_asset_volatility(symbol)
            if volatility > 0.8:  # Escala 0-1
                return {
                    "valid": False, 
                    "reason": f"Volatility too high for {symbol} in cautious mode: {volatility:.2f}"
                }
        
        return {"valid": True, "params": order_params}
    
    async def _get_current_price(self, symbol: str) -> float:
        """
        Obtener precio actual de un activo (simulado para demo).
        
        Args:
            symbol: Símbolo del activo
            
        Returns:
            Precio actual
        """
        # En implementación real, obtener de exchange
        # Para demo, generar precios simulados
        base_prices = {
            "BTC": 50000.0,
            "ETH": 3000.0,
            "SOL": 150.0,
            "ADA": 0.5,
            "DOT": 20.0,
            "AVAX": 30.0,
            "MATIC": 1.5,
            "LINK": 15.0,
            "XRP": 0.6,
            "BNB": 500.0
        }
        
        if symbol in base_prices:
            # Variación aleatoria del ±2%
            variation = random.uniform(0.98, 1.02)
            return base_prices[symbol] * variation
        else:
            # Precio genérico para símbolos desconocidos
            return random.uniform(10.0, 100.0)
    
    async def _get_asset_volatility(self, symbol: str) -> float:
        """
        Obtener volatilidad de un activo (simulado para demo).
        
        Args:
            symbol: Símbolo del activo
            
        Returns:
            Volatilidad en escala 0-1
        """
        # En implementación real, calcular basado en datos históricos
        # Para demo, valores simulados
        volatilities = {
            "BTC": 0.65,
            "ETH": 0.7,
            "SOL": 0.8,
            "ADA": 0.6,
            "DOT": 0.75,
            "AVAX": 0.85,
            "MATIC": 0.75,
            "LINK": 0.6,
            "XRP": 0.5,
            "BNB": 0.55
        }
        
        if symbol in volatilities:
            # Variación aleatoria del ±10%
            variation = random.uniform(0.9, 1.1)
            volatility = volatilities[symbol] * variation
            # Asegurar que está en rango 0-1
            return max(0.0, min(1.0, volatility))
        else:
            # Volatilidad genérica para símbolos desconocidos
            return random.uniform(0.5, 0.9)
    
    async def monitor_positions(self) -> Dict[str, Any]:
        """
        Monitorizar posiciones abiertas con comportamiento humano.
        
        Returns:
            Estado actualizado de posiciones
        """
        try:
            # Verificar fase correcta
            if self.cycle_phase != CyclePhase.GUARDIANSHIP:
                logger.warning(f"Fase incorrecta para monitorización: {self.cycle_phase}")
                return {"success": False, "error": "Incorrect cycle phase"}
            
            # Actualizar estado
            self.state = SeraphimState.TRANSCENDING
            
            # Verificar si es tiempo de finalizar el ciclo
            cycle_elapsed = datetime.now() - self.cycle_start_time
            if cycle_elapsed > self.cycle_duration:
                logger.info(f"Ciclo completó duración máxima: {cycle_elapsed}")
                return await self.close_positions(reason="cycle_duration_reached")
            
            # Monitorizar posiciones
            position_updates = []
            unrealized_profit = 0.0
            
            for symbol, position in self.open_positions.items():
                # Obtener precio actual
                current_price = await self._get_current_price(symbol)
                entry_price = position.get("entry_price", current_price)
                quantity = position.get("quantity", 0)
                
                # Calcular ganancia/pérdida no realizada
                position_value = current_price * quantity
                entry_value = entry_price * quantity
                unrealized_pnl = position_value - entry_value
                unrealized_pnl_pct = (current_price / entry_price - 1) * 100
                
                # Calcular cambio reciente de precio para comportamiento FEARFUL
                last_price = position.get("last_price", current_price)
                recent_price_change = (current_price / last_price - 1) if last_price > 0 else 0
                
                # Actualizar posición
                self.open_positions[symbol].update({
                    "current_price": current_price,
                    "current_value": position_value,
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_pct": unrealized_pnl_pct,
                    "last_price": position.get("current_price", current_price),  # Guardar precio anterior
                    "recent_price_change": recent_price_change  # Tasa de cambio reciente (para comportamiento FEARFUL)
                })
                
                # Acumular ganancia/pérdida no realizada total
                unrealized_profit += unrealized_pnl
                
                # Registrar actualización
                position_updates.append({
                    "symbol": symbol,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "quantity": quantity,
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_pct": unrealized_pnl_pct
                })
                
                # Verificar si se debe cerrar automáticamente por target o stop loss
                close_position = await self._evaluate_position_close(symbol, position)
                if close_position:
                    # En implementación real, esto iniciaría una orden de cierre
                    logger.info(f"Posición {symbol} marcada para cierre automático: {close_position}")
            
            # Actualizar rendimiento del ciclo
            self.cycle_performance["unrealized_profit"] = unrealized_profit
            self.cycle_performance["roi_percentage"] = (
                unrealized_profit / self.cycle_performance["starting_capital"] * 100
                if self.cycle_performance["starting_capital"] > 0 else 0
            )
            
            # Verificar si se alcanzó el objetivo o stop loss del ciclo completo
            cycle_roi = self.cycle_performance["roi_percentage"]
            if cycle_roi >= self.cycle_target_return * 100:
                logger.info(f"Ciclo alcanzó objetivo de retorno: {cycle_roi:.2f}%")
                return await self.close_positions(reason="target_reached")
            
            if cycle_roi <= -self.cycle_max_loss * 100:
                logger.warning(f"Ciclo alcanzó pérdida máxima: {cycle_roi:.2f}%")
                return await self.close_positions(reason="stop_loss_triggered")
            
            # Registrar monitorización en base de datos
            monitoring_data = {
                "cycle_id": self.cycle_id,
                "timestamp": datetime.now().isoformat(),
                "positions": self.open_positions,
                "unrealized_profit": unrealized_profit,
                "roi_percentage": cycle_roi
            }
            await self.database.set_data(f"monitoring:{self.cycle_id}:{int(time.time())}", monitoring_data)
            
            return {
                "success": True,
                "positions": position_updates,
                "unrealized_profit": unrealized_profit,
                "roi_percentage": cycle_roi
            }
            
        except Exception as e:
            logger.error(f"Error al monitorizar posiciones: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _evaluate_position_close(
        self, 
        symbol: str, 
        position: Dict[str, Any]
    ) -> Optional[str]:
        """
        Evaluar si una posición debe cerrarse automáticamente.
        
        Args:
            symbol: Símbolo del activo
            position: Datos de la posición
            
        Returns:
            Razón para cerrar, o None si debe mantenerse
        """
        # Obtener variación de precio
        entry_price = position.get("entry_price", 0)
        current_price = position.get("current_price", entry_price)
        price_change_pct = (current_price / entry_price - 1) * 100
        
        # Si tenemos acceso al motor de comportamiento humano Gabriel, usar sus reglas
        if hasattr(self, 'behavior_engine') and self.behavior_engine:
            # Calcular tasa de cambio reciente para comportamiento FEARFUL
            recent_price_change = position.get("recent_price_change", 0)
            
            # Obtener tiempo de entrada
            entry_time = position.get("entry_time", datetime.now() - timedelta(hours=1))
            
            # Consultar al motor Gabriel si debemos salir
            try:
                # Esto llama a GabrielBehaviorEngine.should_exit_trade que tiene lógica FEARFUL
                should_exit, reason = await self.behavior_engine.should_exit_trade(
                    unrealized_pnl_pct=price_change_pct,
                    asset_data={"symbol": symbol, "type": "crypto"},
                    entry_time=entry_time,
                    price_change_rate=recent_price_change
                )
                
                if should_exit:
                    logger.info(f"Gabriel sugiere cerrar posición {symbol}: {reason}")
                    return reason
            except Exception as e:
                logger.error(f"Error al consultar Gabriel para cierre de posición: {str(e)}")
                # Fallback a comportamiento tradicional si hay error
        
        # Comportamiento tradicional (fallback o complementario)
        # Límites según comportamiento
        if self.current_behavior == HumanBehaviorPattern.CAUTIOUS:
            # Más conservador: take profit y stop loss más ajustados
            take_profit = 3.0  # %
            stop_loss = -2.0   # %
        elif self.current_behavior == HumanBehaviorPattern.OPPORTUNISTIC:
            # Más arriesgado: take profit y stop loss más amplios
            take_profit = 8.0  # %
            stop_loss = -5.0   # %
        else:  # Comportamiento equilibrado
            take_profit = 5.0  # %
            stop_loss = -3.0   # %
        
        # Evaluar condiciones de cierre
        if price_change_pct >= take_profit:
            return "take_profit"
        
        if price_change_pct <= stop_loss:
            return "stop_loss"
        
        # Decisión adicional: tendencia del precio
        trend = await self._analyze_price_trend(symbol)
        if trend == "strongly_bearish" and price_change_pct > 0:
            # Si tenemos ganancia pero tendencia muy bajista, tomar beneficio
            return "protective_close_bearish_trend"
        
        return None
    
    async def _analyze_price_trend(self, symbol: str) -> str:
        """
        Analizar tendencia del precio (simulado para demo).
        
        Args:
            symbol: Símbolo del activo
            
        Returns:
            Clasificación de tendencia
        """
        # En una implementación real, usar datos históricos e indicadores técnicos
        # Para demo, generar resultados simulados
        trend_options = ["strongly_bullish", "bullish", "neutral", "bearish", "strongly_bearish"]
        weights = [0.1, 0.25, 0.3, 0.25, 0.1]  # Distribución de probabilidad
        return random.choices(trend_options, weights=weights)[0]
    
    async def close_positions(self, reason: str = "manual") -> Dict[str, Any]:
        """
        Cerrar todas las posiciones abiertas.
        
        Args:
            reason: Razón para cerrar posiciones
            
        Returns:
            Resultados del cierre
        """
        try:
            # Verificar que hay posiciones para cerrar
            if not self.open_positions:
                logger.warning("No hay posiciones abiertas para cerrar")
                return {"success": False, "error": "No open positions"}
            
            # Actualizar estado
            self.state = SeraphimState.DESCENDING
            
            # Cerrar posiciones
            closing_results = []
            realized_profit = 0.0
            successful_trades = 0
            
            for symbol, position in self.open_positions.items():
                # Obtener precio actual
                current_price = await self._get_current_price(symbol)
                entry_price = position.get("entry_price", current_price)
                quantity = position.get("quantity", 0)
                
                # Preparar orden de cierre
                order_params = {
                    "symbol": symbol,
                    "side": "SELL",
                    "quantity": quantity,
                    "price": current_price,
                    "timestamp": datetime.now().isoformat(),
                    "cycle_id": self.cycle_id,
                    "reason": reason
                }
                
                # Ejecutar orden a través del CodiciaManager
                order_result = await self.codicia_manager.place_order(order_params)
                
                if order_result.get("success", False):
                    # Calcular ganancia/pérdida realizada
                    position_value = current_price * quantity
                    entry_value = entry_price * quantity
                    pnl = position_value - entry_value
                    pnl_pct = (current_price / entry_price - 1) * 100
                    
                    # Determinar si fue exitosa (ganancia)
                    is_successful = pnl > 0
                    if is_successful:
                        successful_trades += 1
                    
                    # Acumular ganancia/pérdida realizada total
                    realized_profit += pnl
                    
                    # Registrar cierre exitoso
                    closing_results.append({
                        "symbol": symbol,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "quantity": quantity,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "successful": is_successful,
                        "status": "closed"
                    })
                    
                    logger.info(f"Posición cerrada: {symbol}, "
                              f"entrada: ${entry_price:.2f}, "
                              f"salida: ${current_price:.2f}, "
                              f"PnL: ${pnl:.2f} ({pnl_pct:.1f}%)")
                else:
                    # Registrar fallo en cierre
                    closing_results.append({
                        "symbol": symbol,
                        "status": "failed",
                        "error": order_result.get("error", "Unknown error")
                    })
                    
                    logger.warning(f"Fallo al cerrar posición: {symbol}, "
                                 f"error: {order_result.get('error', 'Unknown error')}")
            
            # Actualizar estadísticas del ciclo
            self.cycle_performance["realized_profit"] = realized_profit
            self.cycle_performance["current_capital"] = (
                self.cycle_performance["starting_capital"] + realized_profit
            )
            self.cycle_performance["successful_trades"] = successful_trades
            self.cycle_performance["roi_percentage"] = (
                realized_profit / self.cycle_performance["starting_capital"] * 100
                if self.cycle_performance["starting_capital"] > 0 else 0
            )
            
            # Limpiar posiciones
            self.open_positions = {}
            
            # Actualizar fase
            self.cycle_phase = CyclePhase.REFLECTION
            self.state = SeraphimState.REFLECTING
            
            # Crear checkpoint
            await self.checkpoint_manager.create_checkpoint(
                f"positions_closed_{self.cycle_id}", 
                {"strategy": self.name, "state": self.get_state()}
            )
            
            # Registrar resultados en base de datos
            closing_data = {
                "cycle_id": self.cycle_id,
                "timestamp": datetime.now().isoformat(),
                "reason": reason,
                "results": closing_results,
                "realized_profit": realized_profit,
                "roi_percentage": self.cycle_performance["roi_percentage"],
                "successful_trades": successful_trades
            }
            await self.database.set_data(f"closing:{self.cycle_id}", closing_data)
            
            return {
                "success": True,
                "results": closing_results,
                "realized_profit": realized_profit,
                "roi_percentage": self.cycle_performance["roi_percentage"],
                "reason": reason
            }
            
        except Exception as e:
            logger.error(f"Error al cerrar posiciones: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def evaluate_cycle(self) -> Dict[str, Any]:
        """
        Evaluar resultados del ciclo para aprendizaje divino.
        
        Returns:
            Evaluación del ciclo
        """
        try:
            # Verificar fase correcta
            if self.cycle_phase != CyclePhase.REFLECTION:
                logger.warning(f"Fase incorrecta para evaluación: {self.cycle_phase}")
                return {"success": False, "error": "Incorrect cycle phase"}
            
            # Calcular métricas de evaluación
            cycle_duration = datetime.now() - self.cycle_start_time
            duration_hours = cycle_duration.total_seconds() / 3600
            
            # Calcular rendimiento anualizado (muy simplificado)
            roi = self.cycle_performance["roi_percentage"] / 100
            annualized_roi = ((1 + roi) ** (365 * 24 / duration_hours) - 1) * 100 if duration_hours > 0 else 0
            
            # Calcular tasa de éxito
            success_rate = (
                self.cycle_performance["successful_trades"] / self.cycle_performance["trades_count"] * 100
                if self.cycle_performance["trades_count"] > 0 else 0
            )
            
            # Crear evaluación detallada
            evaluation = {
                "cycle_id": self.cycle_id,
                "duration_hours": duration_hours,
                "capital_initial": self.cycle_performance["starting_capital"],
                "capital_final": self.cycle_performance["current_capital"],
                "realized_profit": self.cycle_performance["realized_profit"],
                "roi_percentage": self.cycle_performance["roi_percentage"],
                "annualized_roi": annualized_roi,
                "trades_count": self.cycle_performance["trades_count"],
                "successful_trades": self.cycle_performance["successful_trades"],
                "success_rate": success_rate,
                "behavior": self.current_behavior.name,
                "completed": True,
                "timestamp": datetime.now().isoformat()
            }
            
            # Registrar evaluación en base de datos
            await self.database.set_data(f"evaluation:{self.cycle_id}", evaluation)
            
            # Actualizar fase
            self.cycle_phase = CyclePhase.DISTRIBUTION
            
            logger.info(f"Ciclo evaluado: ROI {self.cycle_performance['roi_percentage']:.2f}%, "
                       f"ganancia ${self.cycle_performance['realized_profit']:.2f}, "
                       f"tasa éxito {success_rate:.1f}%")
                       
            # Notificar a los participantes
            await self._notify_cycle_results(evaluation)
            
            return {
                "success": True,
                "evaluation": evaluation
            }
            
        except Exception as e:
            logger.error(f"Error al evaluar ciclo: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _notify_cycle_results(self, evaluation: Dict[str, Any]) -> None:
        """
        Notificar resultados del ciclo a los participantes.
        
        Args:
            evaluation: Datos de evaluación del ciclo
        """
        try:
            # Preparar mensaje según rendimiento
            roi = evaluation.get("roi_percentage", 0)
            if roi > 0:
                subject = f"Ciclo completado con éxito: +{roi:.2f}%"
                sentiment = "positive"
            else:
                subject = f"Ciclo completado: {roi:.2f}%"
                sentiment = "neutral" if roi >= -1 else "negative"
                
            for participant in self.pool_participants:
                notification = {
                    "type": "cycle_result",
                    "recipient": participant["id"],
                    "subject": subject,
                    "message": (f"El ciclo {self.cycle_id} ha finalizado con un ROI de {roi:.2f}%.\n"
                              f"Operaciones exitosas: {evaluation.get('success_rate', 0):.1f}%\n"
                              f"Ganancia realizada: ${evaluation.get('realized_profit', 0):.2f}"),
                    "sentiment": sentiment
                }
                await self.alert_manager.send_notification(notification)
                
            logger.debug(f"Notificaciones de resultados enviadas a {len(self.pool_participants)} participantes")
        except Exception as e:
            logger.warning(f"Error al enviar notificaciones de resultados: {str(e)}")
    
    async def distribute_profits(self) -> Dict[str, Any]:
        """
        Distribuir ganancias entre los participantes del pool.
        
        Returns:
            Distribución de ganancias
        """
        try:
            # Verificar fase correcta
            if self.cycle_phase != CyclePhase.DISTRIBUTION:
                logger.warning(f"Fase incorrecta para distribución: {self.cycle_phase}")
                return {"success": False, "error": "Incorrect cycle phase"}
            
            # Actualizar estado
            self.state = SeraphimState.DISTRIBUTING
            
            # Obtener ganancia realizada
            profit = self.cycle_performance["realized_profit"]
            
            if profit <= 0:
                logger.info(f"No hay ganancias para distribuir: ${profit:.2f}")
                # Actualizar fase
                self.cycle_phase = CyclePhase.REBIRTH
                self.state = SeraphimState.RESTING
                return {
                    "success": True,
                    "distribution": [],
                    "profit": profit,
                    "message": "No profit to distribute"
                }
            
            # Calcular distribución
            # 85% para participantes, 10% reserva, 5% sistema
            participant_allocation = profit * 0.85
            reserve_allocation = profit * 0.10
            system_allocation = profit * 0.05
            
            # Distribuir entre participantes según sus acciones
            distributions = []
            for participant in self.pool_participants:
                participant_share = participant.get("share", 0)
                amount = participant_allocation * participant_share
                
                distributions.append({
                    "participant_id": participant.get("id"),
                    "participant_name": participant.get("name"),
                    "share": participant_share,
                    "amount": amount
                })
                
                logger.debug(f"Distribución a {participant.get('name')}: ${amount:.2f} "
                           f"({participant_share*100:.1f}% de ${participant_allocation:.2f})")
            
            # Crear registro de distribución
            distribution_record = {
                "cycle_id": self.cycle_id,
                "timestamp": datetime.now().isoformat(),
                "total_profit": profit,
                "participant_allocation": participant_allocation,
                "reserve_allocation": reserve_allocation,
                "system_allocation": system_allocation,
                "distributions": distributions
            }
            
            # Registrar en base de datos
            await self.database.set_data(f"distribution:{self.cycle_id}", distribution_record)
            
            # Crear checkpoint
            await self.checkpoint_manager.create_checkpoint(
                f"profits_distributed_{self.cycle_id}", 
                {"strategy": self.name, "state": self.get_state()}
            )
            
            # Actualizar fase
            self.cycle_phase = CyclePhase.REBIRTH
            self.state = SeraphimState.RESTING
            
            logger.info(f"Distribución completada: ${profit:.2f} distribuidos a {len(distributions)} participantes, "
                       f"reserva: ${reserve_allocation:.2f}, sistema: ${system_allocation:.2f}")
            
            return {
                "success": True,
                "distributions": distributions,
                "profit": profit,
                "reserve": reserve_allocation,
                "system": system_allocation
            }
            
        except Exception as e:
            logger.error(f"Error al distribuir ganancias: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_cycle_status(self) -> Dict[str, Any]:
        """
        Obtener estado actual del ciclo.
        
        Returns:
            Estado del ciclo
        """
        # Calcular tiempo transcurrido
        elapsed = None
        if self.cycle_start_time is not None:
            elapsed = (datetime.now() - self.cycle_start_time).total_seconds()
        
        return {
            "cycle_id": self.cycle_id,
            "cycle_phase": self.cycle_phase.name if self.cycle_phase else None,
            "strategy_state": self.state.name if self.state else None,
            "behavior": self.current_behavior.name if self.current_behavior else None,
            "start_time": self.cycle_start_time.isoformat() if self.cycle_start_time else None,
            "elapsed_seconds": elapsed,
            "selected_assets": len(self.selected_assets),
            "open_positions": len(self.open_positions),
            "performance": self.cycle_performance
        }
    
    def get_state(self) -> Dict[str, Any]:
        """
        Obtener estado completo de la estrategia para checkpoints.
        
        Returns:
            Estado completo
        """
        return {
            "name": self.name,
            "cycle_id": self.cycle_id,
            "state": self.state.name if self.state else None,
            "cycle_phase": self.cycle_phase.name if self.cycle_phase else None,
            "behavior": self.current_behavior.name if self.current_behavior else None,
            "cycle_start_time": self.cycle_start_time.isoformat() if self.cycle_start_time else None,
            "cycle_capital": self.cycle_capital,
            "cycle_target_return": self.cycle_target_return,
            "cycle_max_loss": self.cycle_max_loss,
            "selected_assets": self.selected_assets,
            "asset_allocations": self.asset_allocations,
            "open_positions": self.open_positions,
            "cycle_performance": self.cycle_performance
        }
        
    async def generate_signal(self, market_data: Dict[str, Any], configuration: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementación requerida por BaseStrategy para generar señales de trading.
        
        Args:
            market_data: Datos de mercado
            configuration: Configuración para generación de señales
            
        Returns:
            Señal de trading generada
        """
        # En la estrategia Seraphim, las señales se generan a través del flujo completo
        # de HumanPoolTrader y no por este método directo, pero se implementa para
        # compatibilidad con BaseStrategy
        
        signal = {
            "timestamp": datetime.now().isoformat(),
            "cycle_id": self.cycle_id,
            "signal_type": "none",
            "message": "Signals are generated through the complete Seraphim cycle flow"
        }
        
        return signal