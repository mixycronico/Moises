"""
Módulo de Decisión para el Pipeline de Genesis.

Este módulo se encarga de tomar decisiones de trading basadas en las señales 
generadas por el análisis, considerando gestión de capital y restricciones.
"""
import logging
import time
import json
import random
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta

from genesis.base import GenesisComponent, GenesisSingleton, validate_mode
from genesis.db.transcendental_database import TranscendentalDatabase
from genesis.pipeline.analysis import SignalType

# Configuración de logging
logger = logging.getLogger("genesis.pipeline.decision")

class Decision:
    """Decisión de trading con información completa."""
    
    def __init__(self, 
                symbol: str, 
                action: str, 
                amount: float, 
                price: Optional[float] = None,
                confidence: float = 0.0):
        """
        Inicializar decisión de trading.
        
        Args:
            symbol: Símbolo de trading
            action: Acción (buy, sell, exit, hold)
            amount: Cantidad a operar
            price: Precio objetivo (opcional)
            confidence: Confianza en la decisión (0-1)
        """
        self.symbol = symbol
        self.action = action
        self.amount = amount
        self.price = price
        self.confidence = confidence
        self.timestamp = time.time()
        self.id = f"decision_{self.timestamp}_{random.randint(1000, 9999)}"
        self.reasons: List[str] = []
        self.source_signals: Dict[str, Any] = {}
        
    def add_reason(self, reason: str) -> None:
        """
        Añadir razón de la decisión.
        
        Args:
            reason: Explicación de la decisión
        """
        self.reasons.append(reason)
    
    def set_source_signals(self, signals: Dict[str, Any]) -> None:
        """
        Establecer señales origen de la decisión.
        
        Args:
            signals: Señales que generaron la decisión
        """
        self.source_signals = signals
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir decisión a diccionario.
        
        Returns:
            Representación como diccionario
        """
        return {
            "id": self.id,
            "symbol": self.symbol,
            "action": self.action,
            "amount": self.amount,
            "price": self.price,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            "reasons": self.reasons,
            "source_signals": self.source_signals
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Decision':
        """
        Crear decisión desde diccionario.
        
        Args:
            data: Diccionario con datos de decisión
            
        Returns:
            Instancia de Decision
        """
        decision = cls(
            symbol=data["symbol"],
            action=data["action"],
            amount=data["amount"],
            price=data.get("price"),
            confidence=data.get("confidence", 0.0)
        )
        decision.id = data.get("id", decision.id)
        decision.timestamp = data.get("timestamp", decision.timestamp)
        decision.reasons = data.get("reasons", [])
        decision.source_signals = data.get("source_signals", {})
        
        return decision

class DecisionMaker(GenesisComponent):
    """Generador de decisiones base con capacidades trascendentales."""
    
    def __init__(self, decision_maker_id: str, decision_maker_name: str, mode: str = "SINGULARITY_V4"):
        """
        Inicializar generador de decisiones.
        
        Args:
            decision_maker_id: Identificador único
            decision_maker_name: Nombre descriptivo
            mode: Modo trascendental
        """
        super().__init__(f"decision_maker_{decision_maker_id}", mode)
        self.decision_maker_id = decision_maker_id
        self.decision_maker_name = decision_maker_name
        self.last_decision = 0
        self.decision_count = 0
        self.db = TranscendentalDatabase()
        
        logger.info(f"Generador de decisiones {decision_maker_name} ({decision_maker_id}) inicializado")
    
    async def make_decision(self, data: Dict[str, Any]) -> List[Decision]:
        """
        Generar decisiones basadas en datos y señales.
        
        Args:
            data: Datos con señales de análisis
            
        Returns:
            Lista de decisiones generadas
        """
        raise NotImplementedError("Las subclases deben implementar make_decision")
    
    async def make_decision_with_resilience(self, data: Dict[str, Any]) -> List[Decision]:
        """
        Generar decisiones con mecanismos de resiliencia.
        
        Args:
            data: Datos con señales de análisis
            
        Returns:
            Lista de decisiones generadas
        """
        self.decision_count += 1
        self.last_decision = time.time()
        
        try:
            start_time = time.time()
            decisions = await self.make_decision(data)
            decision_time = time.time() - start_time
            
            # Actualizar métricas
            self.update_metric("decision_time", decision_time)
            self.update_metric("decision_count", self.decision_count)
            self.update_metric("decisions_per_run", len(decisions))
            
            logger.debug(f"Generación de decisiones {self.decision_maker_id} exitosa en {decision_time:.3f}s: {len(decisions)} decisiones")
            self.register_operation(True)
            return decisions
        
        except Exception as e:
            self.register_operation(False)
            logger.error(f"Error en generador de decisiones {self.decision_maker_id}: {str(e)}")
            
            # En caso de error, devolver lista vacía
            return []

class SignalBasedDecisionMaker(DecisionMaker):
    """Generador de decisiones basado en señales de análisis."""
    
    def __init__(self, mode: str = "SINGULARITY_V4"):
        """
        Inicializar generador de decisiones basado en señales.
        
        Args:
            mode: Modo trascendental
        """
        super().__init__("signal_based", "Decisiones por Señales", mode)
        self.confidence_threshold = 0.3  # Confianza mínima para generar decisión
        self.max_positions = 5  # Número máximo de posiciones simultáneas
        self.capital_per_position = 0.2  # Proporción de capital por posición
    
    async def make_decision(self, data: Dict[str, Any]) -> List[Decision]:
        """
        Generar decisiones basadas en señales de análisis.
        
        Args:
            data: Datos con señales de análisis
            
        Returns:
            Lista de decisiones generadas
        """
        decisions = []
        
        # Extraer señales de análisis
        signals = data.get("signals", {})
        if not signals:
            logger.warning("No hay señales disponibles para tomar decisiones")
            return []
        
        # Extraer posiciones actuales
        positions = data.get("positions", {})
        open_position_symbols = set(positions.keys())
        
        # Extraer información de capital
        capital_info = data.get("capital_info", {})
        available_capital = capital_info.get("available_capital", 10000.0)
        total_capital = capital_info.get("total_capital", 10000.0)
        
        # Calcular capital por posición
        capital_per_position = total_capital * self.capital_per_position
        
        # Procesar cada símbolo con señales
        for symbol, symbol_signals in signals.items():
            # Obtener señal final combinada
            if "final" not in symbol_signals:
                continue
            
            final_signal = symbol_signals["final"]
            signal_type = final_signal.get("signal")
            strength = final_signal.get("strength", 0)
            
            # Extraer precio actual del símbolo
            market_data = data.get("market_data", {}).get(symbol, {})
            latest_candle = market_data.get("latest", {})
            current_price = latest_candle.get("close", 0)
            
            if not current_price:
                logger.warning(f"No hay precio disponible para {symbol}, omitiendo")
                continue
            
            # Comprobar umbral de confianza
            confidence = abs(strength)
            if confidence < self.confidence_threshold:
                logger.debug(f"Confianza insuficiente para {symbol}: {confidence:.2f} < {self.confidence_threshold}")
                continue
            
            # Determinar acción basada en la señal y las posiciones existentes
            if signal_type == SignalType.BUY and symbol not in open_position_symbols:
                # Solo abrir nuevas posiciones si no excedemos el máximo
                if len(open_position_symbols) < self.max_positions:
                    # Calcular cantidad a comprar
                    amount = capital_per_position / current_price
                    
                    # Crear decisión de compra
                    decision = Decision(
                        symbol=symbol,
                        action="buy",
                        amount=amount,
                        price=current_price,
                        confidence=confidence
                    )
                    
                    # Añadir razones
                    decision.add_reason(f"Señal de compra con confianza {confidence:.2f}")
                    decision.add_reason(f"Precio actual: {current_price}")
                    decision.set_source_signals(symbol_signals)
                    
                    decisions.append(decision)
                    logger.info(f"Decisión de COMPRA generada para {symbol}")
                else:
                    logger.debug(f"Máximo de posiciones alcanzado ({self.max_positions}), no se abre {symbol}")
            
            elif signal_type == SignalType.SELL and symbol not in open_position_symbols:
                # Si soportamos posiciones cortas, abrir posición corta
                if "allow_short" in capital_info and capital_info["allow_short"]:
                    # Calcular cantidad a vender (corto)
                    amount = capital_per_position / current_price
                    
                    # Crear decisión de venta corta
                    decision = Decision(
                        symbol=symbol,
                        action="sell_short",
                        amount=amount,
                        price=current_price,
                        confidence=confidence
                    )
                    
                    # Añadir razones
                    decision.add_reason(f"Señal de venta corta con confianza {confidence:.2f}")
                    decision.add_reason(f"Precio actual: {current_price}")
                    decision.set_source_signals(symbol_signals)
                    
                    decisions.append(decision)
                    logger.info(f"Decisión de VENTA CORTA generada para {symbol}")
            
            elif signal_type == SignalType.SELL and symbol in open_position_symbols:
                # Cerrar posición existente
                position = positions[symbol]
                position_size = position.get("size", 0)
                
                if position_size > 0:
                    # Crear decisión de cierre
                    decision = Decision(
                        symbol=symbol,
                        action="exit",
                        amount=position_size,
                        price=current_price,
                        confidence=confidence
                    )
                    
                    # Añadir razones
                    decision.add_reason(f"Señal de salida con confianza {confidence:.2f}")
                    decision.add_reason(f"Precio actual: {current_price}")
                    decision.set_source_signals(symbol_signals)
                    
                    decisions.append(decision)
                    logger.info(f"Decisión de SALIDA generada para {symbol}")
            
            elif signal_type == SignalType.EXIT and symbol in open_position_symbols:
                # Señal explícita de salida
                position = positions[symbol]
                position_size = position.get("size", 0)
                
                if position_size > 0:
                    # Crear decisión de cierre
                    decision = Decision(
                        symbol=symbol,
                        action="exit",
                        amount=position_size,
                        price=current_price,
                        confidence=confidence
                    )
                    
                    # Añadir razones
                    decision.add_reason(f"Señal explícita de salida con confianza {confidence:.2f}")
                    decision.add_reason(f"Precio actual: {current_price}")
                    decision.set_source_signals(symbol_signals)
                    
                    decisions.append(decision)
                    logger.info(f"Decisión de SALIDA generada para {symbol}")
        
        # Actualizar métricas
        self.update_metric("decision_count", len(decisions))
        
        return decisions

class RiskBasedDecisionMaker(DecisionMaker):
    """Generador de decisiones basado en gestión de riesgo y capital."""
    
    def __init__(self, mode: str = "SINGULARITY_V4"):
        """
        Inicializar generador de decisiones basado en riesgo.
        
        Args:
            mode: Modo trascendental
        """
        super().__init__("risk_based", "Decisiones por Riesgo", mode)
        self.risk_per_trade = 0.01  # 1% del capital por operación
        self.max_total_risk = 0.05  # 5% del capital en riesgo total
        self.confidence_threshold = 0.4  # Confianza mínima
        self.stop_loss_atr_factor = 1.5  # Factor ATR para stop loss
    
    async def make_decision(self, data: Dict[str, Any]) -> List[Decision]:
        """
        Generar decisiones basadas en gestión de riesgo.
        
        Args:
            data: Datos con señales de análisis
            
        Returns:
            Lista de decisiones generadas
        """
        decisions = []
        
        # Extraer señales de análisis
        signals = data.get("signals", {})
        if not signals:
            logger.warning("No hay señales disponibles para tomar decisiones")
            return []
        
        # Extraer posiciones actuales
        positions = data.get("positions", {})
        open_position_symbols = set(positions.keys())
        
        # Extraer información de capital
        capital_info = data.get("capital_info", {})
        available_capital = capital_info.get("available_capital", 10000.0)
        total_capital = capital_info.get("total_capital", 10000.0)
        
        # Calcular riesgo actual
        current_risk = sum(
            position.get("risk_percent", 0) 
            for position in positions.values()
        )
        
        # Calcular riesgo disponible
        available_risk = self.max_total_risk - current_risk
        
        # Si no hay riesgo disponible, no abrir nuevas posiciones
        if available_risk <= 0:
            logger.info(f"Máximo riesgo alcanzado: {current_risk:.2%}")
            # Pero aún podemos generar decisiones de salida
            
        # Procesar cada símbolo con señales
        for symbol, symbol_signals in signals.items():
            # Obtener señal final combinada
            if "final" not in symbol_signals:
                continue
            
            final_signal = symbol_signals["final"]
            signal_type = final_signal.get("signal")
            strength = final_signal.get("strength", 0)
            
            # Extraer datos de mercado del símbolo
            market_data = data.get("market_data", {}).get(symbol, {})
            latest_candle = market_data.get("latest", {})
            current_price = latest_candle.get("close", 0)
            atr = latest_candle.get("atr", 0)
            
            if not current_price:
                logger.warning(f"No hay precio disponible para {symbol}, omitiendo")
                continue
            
            # Comprobar umbral de confianza
            confidence = abs(strength)
            if confidence < self.confidence_threshold:
                logger.debug(f"Confianza insuficiente para {symbol}: {confidence:.2f} < {self.confidence_threshold}")
                continue
            
            # Determinar acción basada en la señal y las posiciones existentes
            if signal_type == SignalType.BUY and symbol not in open_position_symbols:
                # Solo abrir si hay riesgo disponible
                if available_risk > 0:
                    # Calcular stop loss basado en ATR
                    stop_loss_price = None
                    if atr:
                        stop_loss_price = current_price - (atr * self.stop_loss_atr_factor)
                    else:
                        # Si no hay ATR, usar un stop loss fijo del 2%
                        stop_loss_price = current_price * 0.98
                    
                    # Calcular riesgo en dólares
                    risk_amount = (current_price - stop_loss_price) / current_price  # porcentaje
                    risk_dollars = total_capital * self.risk_per_trade
                    
                    # Calcular tamaño de posición basado en riesgo
                    position_size_dollars = risk_dollars / risk_amount if risk_amount > 0 else 0
                    amount = position_size_dollars / current_price
                    
                    # Crear decisión de compra con gestión de riesgo
                    decision = Decision(
                        symbol=symbol,
                        action="buy",
                        amount=amount,
                        price=current_price,
                        confidence=confidence
                    )
                    
                    # Añadir información de stop loss
                    decision.stop_loss = stop_loss_price
                    decision.risk_amount = risk_dollars
                    decision.risk_percent = self.risk_per_trade
                    
                    # Añadir razones
                    decision.add_reason(f"Señal de compra con confianza {confidence:.2f}")
                    decision.add_reason(f"Stop loss en {stop_loss_price:.2f} (ATR: {atr:.2f})")
                    decision.add_reason(f"Riesgo: {risk_dollars:.2f} USD ({self.risk_per_trade:.2%})")
                    decision.set_source_signals(symbol_signals)
                    
                    decisions.append(decision)
                    logger.info(f"Decisión de COMPRA generada para {symbol} con gestión de riesgo")
                else:
                    logger.debug(f"Máximo riesgo alcanzado ({current_risk:.2%}), no se abre {symbol}")
            
            elif (signal_type == SignalType.SELL or signal_type == SignalType.EXIT) and symbol in open_position_symbols:
                # Cerrar posición existente
                position = positions[symbol]
                position_size = position.get("size", 0)
                
                if position_size > 0:
                    # Crear decisión de cierre
                    decision = Decision(
                        symbol=symbol,
                        action="exit",
                        amount=position_size,
                        price=current_price,
                        confidence=confidence
                    )
                    
                    # Añadir razones
                    decision.add_reason(f"Señal de salida con confianza {confidence:.2f}")
                    decision.add_reason(f"Precio actual: {current_price}")
                    decision.set_source_signals(symbol_signals)
                    
                    decisions.append(decision)
                    logger.info(f"Decisión de SALIDA generada para {symbol}")
        
        # Actualizar métricas
        self.update_metric("decision_count", len(decisions))
        
        return decisions

class PositionManager(GenesisComponent):
    """
    Gestor de posiciones que mantiene el estado de las posiciones abiertas
    y aplica reglas de gestión de riesgo continuo.
    """
    
    def __init__(self, mode: str = "SINGULARITY_V4"):
        """
        Inicializar gestor de posiciones.
        
        Args:
            mode: Modo trascendental
        """
        super().__init__("position_manager", mode)
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.position_history: List[Dict[str, Any]] = []
        self.db = TranscendentalDatabase()
        
        # Opciones de gestión
        self.trailing_stop_enabled = True
        self.trailing_stop_activation = 0.02  # 2% de beneficio para activar
        self.trailing_stop_distance = 0.015  # 1.5% de distancia
        
        logger.info(f"Gestor de posiciones inicializado en modo {mode}")
    
    async def update_positions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Actualizar estado de posiciones con precios actuales.
        
        Args:
            data: Datos con información de mercado
            
        Returns:
            Datos actualizados con posiciones
        """
        updated_data = data.copy()
        
        # Asegurarnos que existe la estructura de posiciones
        if "positions" not in updated_data:
            updated_data["positions"] = self.positions
        else:
            # Usar las posiciones existentes en los datos o las del gestor
            self.positions = updated_data["positions"]
        
        # Extraer datos de mercado
        market_data = updated_data.get("market_data", {})
        
        # Actualizar cada posición con precios actuales
        for symbol, position in self.positions.items():
            # Obtener precio actual
            symbol_data = market_data.get(symbol, {})
            latest_candle = symbol_data.get("latest", {})
            
            if not latest_candle or "close" not in latest_candle:
                logger.warning(f"No hay precio actual para {symbol}, omitiendo actualización")
                continue
            
            current_price = latest_candle["close"]
            
            # Actualizar valores de la posición
            entry_price = position.get("entry_price", current_price)
            size = position.get("size", 0)
            position_type = position.get("type", "long")
            
            # Calcular P&L
            if position_type == "long":
                pnl_percent = (current_price - entry_price) / entry_price
                pnl_amount = (current_price - entry_price) * size
            else:  # short
                pnl_percent = (entry_price - current_price) / entry_price
                pnl_amount = (entry_price - current_price) * size
            
            # Actualizar posición
            position["current_price"] = current_price
            position["pnl_percent"] = pnl_percent
            position["pnl_amount"] = pnl_amount
            position["last_update"] = time.time()
            
            # Aplicar trailing stop si está habilitado
            if self.trailing_stop_enabled and "stop_loss" in position:
                self._apply_trailing_stop(position, current_price, position_type)
            
            # Comprobar si se ha alcanzado el stop loss
            if "stop_loss" in position and position_type == "long" and current_price <= position["stop_loss"]:
                position["stop_triggered"] = True
                position["stop_price"] = position["stop_loss"]
                logger.info(f"Stop Loss alcanzado para {symbol} en {position['stop_loss']}")
            elif "stop_loss" in position and position_type == "short" and current_price >= position["stop_loss"]:
                position["stop_triggered"] = True
                position["stop_price"] = position["stop_loss"]
                logger.info(f"Stop Loss alcanzado para {symbol} en {position['stop_loss']} (short)")
        
        # Actualizar datos
        updated_data["positions"] = self.positions
        updated_data["position_update_timestamp"] = time.time()
        
        return updated_data
    
    def _apply_trailing_stop(self, position: Dict[str, Any], current_price: float, position_type: str) -> None:
        """
        Aplicar trailing stop a una posición.
        
        Args:
            position: Datos de la posición
            current_price: Precio actual
            position_type: Tipo de posición (long/short)
        """
        entry_price = position.get("entry_price", current_price)
        current_stop = position.get("stop_loss", 0)
        
        if position_type == "long":
            # Calcular beneficio actual
            profit_percent = (current_price - entry_price) / entry_price
            
            # Si el beneficio supera el umbral de activación
            if profit_percent >= self.trailing_stop_activation:
                # Calcular nuevo stop loss
                new_stop = current_price * (1 - self.trailing_stop_distance)
                
                # Solo actualizar si mejora el stop actual
                if not current_stop or new_stop > current_stop:
                    position["stop_loss"] = new_stop
                    position["trailing_active"] = True
                    logger.debug(f"Trailing stop actualizado: {new_stop:.4f} ({profit_percent:.2%} beneficio)")
        
        else:  # short
            # Calcular beneficio actual (inverso para short)
            profit_percent = (entry_price - current_price) / entry_price
            
            # Si el beneficio supera el umbral de activación
            if profit_percent >= self.trailing_stop_activation:
                # Calcular nuevo stop loss (hacia abajo para short)
                new_stop = current_price * (1 + self.trailing_stop_distance)
                
                # Solo actualizar si mejora el stop actual
                if not current_stop or new_stop < current_stop:
                    position["stop_loss"] = new_stop
                    position["trailing_active"] = True
                    logger.debug(f"Trailing stop actualizado (short): {new_stop:.4f} ({profit_percent:.2%} beneficio)")
    
    async def add_position(self, 
                         symbol: str, 
                         entry_price: float, 
                         size: float, 
                         position_type: str = "long",
                         stop_loss: Optional[float] = None,
                         take_profit: Optional[float] = None) -> Dict[str, Any]:
        """
        Añadir nueva posición.
        
        Args:
            symbol: Símbolo de trading
            entry_price: Precio de entrada
            size: Tamaño de la posición
            position_type: Tipo de posición ("long" o "short")
            stop_loss: Nivel de stop loss (opcional)
            take_profit: Nivel de take profit (opcional)
            
        Returns:
            Datos de la posición
        """
        position = {
            "symbol": symbol,
            "entry_price": entry_price,
            "size": size,
            "type": position_type,
            "open_time": time.time(),
            "current_price": entry_price,
            "pnl_percent": 0,
            "pnl_amount": 0,
            "last_update": time.time()
        }
        
        # Añadir niveles si están definidos
        if stop_loss:
            position["stop_loss"] = stop_loss
            position["initial_stop_loss"] = stop_loss
        
        if take_profit:
            position["take_profit"] = take_profit
        
        # Calcular riesgo
        if stop_loss and position_type == "long":
            position["risk_amount"] = (entry_price - stop_loss) * size
            position["risk_percent"] = (entry_price - stop_loss) / entry_price
        elif stop_loss and position_type == "short":
            position["risk_amount"] = (stop_loss - entry_price) * size
            position["risk_percent"] = (stop_loss - entry_price) / entry_price
        
        # Almacenar posición
        self.positions[symbol] = position
        logger.info(f"Nueva posición añadida: {symbol} ({position_type}) a {entry_price:.4f} x {size:.4f}")
        
        # Guardar en base de datos
        await self._save_position(position)
        
        return position
    
    async def close_position(self, 
                           symbol: str, 
                           exit_price: float,
                           exit_reason: str = "manual") -> Optional[Dict[str, Any]]:
        """
        Cerrar posición existente.
        
        Args:
            symbol: Símbolo de trading
            exit_price: Precio de salida
            exit_reason: Razón de cierre
            
        Returns:
            Datos de la posición cerrada o None si no existe
        """
        if symbol not in self.positions:
            logger.warning(f"Intento de cerrar posición inexistente: {symbol}")
            return None
        
        # Obtener posición
        position = self.positions[symbol]
        position["exit_price"] = exit_price
        position["exit_time"] = time.time()
        position["exit_reason"] = exit_reason
        
        # Calcular P&L final
        entry_price = position["entry_price"]
        size = position["size"]
        position_type = position["type"]
        
        if position_type == "long":
            position["final_pnl_percent"] = (exit_price - entry_price) / entry_price
            position["final_pnl_amount"] = (exit_price - entry_price) * size
        else:  # short
            position["final_pnl_percent"] = (entry_price - exit_price) / entry_price
            position["final_pnl_amount"] = (entry_price - exit_price) * size
        
        # Mover a historial y eliminar de activas
        position_copy = position.copy()
        self.position_history.append(position_copy)
        del self.positions[symbol]
        
        # Registrar en base de datos
        await self._save_closed_position(position_copy)
        
        logger.info(f"Posición cerrada: {symbol} a {exit_price:.4f}, P&L: {position['final_pnl_amount']:.2f} ({position['final_pnl_percent']:.2%})")
        
        return position_copy
    
    async def _save_position(self, position: Dict[str, Any]) -> bool:
        """
        Guardar posición en base de datos.
        
        Args:
            position: Datos de la posición
            
        Returns:
            True si se guardó correctamente
        """
        try:
            # Clave única para la posición
            symbol = position["symbol"]
            open_time = position["open_time"]
            position_key = f"{symbol}_{open_time}"
            
            # Guardar en base de datos trascendental
            await self.db.store("positions", position_key, position)
            return True
        except Exception as e:
            logger.error(f"Error al guardar posición en DB: {str(e)}")
            return False
    
    async def _save_closed_position(self, position: Dict[str, Any]) -> bool:
        """
        Guardar posición cerrada en base de datos.
        
        Args:
            position: Datos de la posición cerrada
            
        Returns:
            True si se guardó correctamente
        """
        try:
            # Clave única para la posición cerrada
            symbol = position["symbol"]
            open_time = position["open_time"]
            close_time = position["exit_time"]
            position_key = f"{symbol}_{open_time}_{close_time}"
            
            # Guardar en base de datos trascendental
            await self.db.store("closed_positions", position_key, position)
            return True
        except Exception as e:
            logger.error(f"Error al guardar posición cerrada en DB: {str(e)}")
            return False
    
    async def get_position_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Obtener historial de posiciones cerradas.
        
        Args:
            limit: Número máximo de posiciones a retornar
            
        Returns:
            Lista de posiciones cerradas
        """
        history = self.position_history[-limit:] if limit > 0 else self.position_history
        
        # Ordenar por fecha de cierre (más recientes primero)
        history.sort(key=lambda x: x.get("exit_time", 0), reverse=True)
        
        return history
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del gestor de posiciones.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = super().get_stats()
        
        # Calcular estadísticas adicionales
        open_positions = len(self.positions)
        total_positions = open_positions + len(self.position_history)
        
        # Calcular P&L total
        total_pnl = sum(position.get("final_pnl_amount", 0) for position in self.position_history)
        
        # Calcular tasa de aciertos
        winning_trades = sum(1 for p in self.position_history if p.get("final_pnl_amount", 0) > 0)
        win_rate = winning_trades / max(1, len(self.position_history))
        
        # Ratio ganancia/pérdida
        avg_win = sum(p.get("final_pnl_amount", 0) for p in self.position_history if p.get("final_pnl_amount", 0) > 0) / max(1, winning_trades)
        losing_trades = sum(1 for p in self.position_history if p.get("final_pnl_amount", 0) < 0)
        avg_loss = sum(abs(p.get("final_pnl_amount", 0)) for p in self.position_history if p.get("final_pnl_amount", 0) < 0) / max(1, losing_trades)
        profit_factor = avg_win / max(0.01, avg_loss)
        
        position_stats = {
            "open_positions": open_positions,
            "total_positions": total_positions,
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "active_symbols": list(self.positions.keys())
        }
        
        stats.update(position_stats)
        return stats

class CapitalManager(GenesisComponent, GenesisSingleton):
    """
    Gestor de capital que controla la asignación de fondos y riesgos.
    """
    
    def __init__(self, initial_capital: float = 10000.0, mode: str = "SINGULARITY_V4"):
        """
        Inicializar gestor de capital.
        
        Args:
            initial_capital: Capital inicial
            mode: Modo trascendental
        """
        super().__init__("capital_manager", mode)
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.available_capital = initial_capital
        self.reserved_capital = 0.0
        self.peak_capital = initial_capital
        self.drawdowns: List[Dict[str, Any]] = []
        
        # Opciones de gestión
        self.max_allocation_percent = 0.8  # 80% máximo en mercado
        self.max_allocation_per_position = 0.2  # 20% máximo por posición
        self.allow_compounding = True
        self.risk_limits_enabled = True
        self.max_risk_per_trade = 0.02  # 2% por operación
        self.max_portfolio_risk = 0.1  # 10% total
        
        # Asignación por clases de activo
        self.asset_class_allocation = {
            "crypto": 1.0  # 100% crypto en este caso
        }
        
        logger.info(f"Gestor de capital inicializado con {initial_capital:.2f} en modo {mode}")
    
    async def update_capital(self, positions: Dict[str, Dict[str, Any]], 
                          closed_positions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Actualizar estado del capital basado en posiciones.
        
        Args:
            positions: Posiciones actuales
            closed_positions: Posiciones cerradas recientemente
            
        Returns:
            Información actualizada del capital
        """
        # Valor anterior para comparación
        previous_capital = self.current_capital
        
        # Resetear capital disponible
        self.available_capital = self.current_capital
        self.reserved_capital = 0.0
        
        # Añadir ganancias de posiciones cerradas
        if closed_positions:
            for position in closed_positions:
                pnl = position.get("final_pnl_amount", 0)
                self.current_capital += pnl
                
                # Registrar cierre
                if pnl > 0:
                    logger.info(f"Capital aumentado en {pnl:.2f} por cierre de {position['symbol']}")
                else:
                    logger.info(f"Capital reducido en {abs(pnl):.2f} por cierre de {position['symbol']}")
        
        # Reservar capital para posiciones abiertas
        for symbol, position in positions.items():
            position_value = position.get("size", 0) * position.get("current_price", 0)
            self.reserved_capital += position_value
        
        # Actualizar capital disponible
        self.available_capital = max(0, self.current_capital - self.reserved_capital)
        
        # Actualizar pico de capital
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Calcular drawdown si hay reducción
        if self.current_capital < previous_capital:
            drawdown_percent = (self.peak_capital - self.current_capital) / self.peak_capital
            
            # Registrar si es significativo (>1%)
            if drawdown_percent > 0.01:
                self.drawdowns.append({
                    "timestamp": time.time(),
                    "peak_capital": self.peak_capital,
                    "current_capital": self.current_capital,
                    "drawdown_percent": drawdown_percent,
                    "drawdown_amount": self.peak_capital - self.current_capital
                })
                
                logger.warning(f"Drawdown detectado: {drawdown_percent:.2%} ({self.peak_capital - self.current_capital:.2f})")
        
        # Preparar resumen
        capital_info = {
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "available_capital": self.available_capital,
            "reserved_capital": self.reserved_capital,
            "peak_capital": self.peak_capital,
            "total_return": (self.current_capital - self.initial_capital) / self.initial_capital,
            "drawdown": (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0,
            "allocation_percent": self.reserved_capital / self.current_capital if self.current_capital > 0 else 0,
            "capital_update_timestamp": time.time()
        }
        
        # Actualizar métricas
        self.update_metric("current_capital", self.current_capital)
        self.update_metric("available_capital", self.available_capital)
        self.update_metric("reserved_capital", self.reserved_capital)
        self.update_metric("total_return", capital_info["total_return"])
        
        return capital_info
    
    async def can_open_position(self, amount: float, price: float, 
                             stop_loss: Optional[float] = None) -> Tuple[bool, str]:
        """
        Verificar si se puede abrir una posición según la gestión de capital.
        
        Args:
            amount: Cantidad a operar
            price: Precio de entrada
            stop_loss: Nivel de stop loss (opcional)
            
        Returns:
            Tupla (permitido, razón)
        """
        position_value = amount * price
        
        # Verificar capital disponible
        if position_value > self.available_capital:
            return False, f"Capital insuficiente: {position_value:.2f} > {self.available_capital:.2f}"
        
        # Verificar límite por posición
        max_position_value = self.current_capital * self.max_allocation_per_position
        if position_value > max_position_value:
            return False, f"Excede límite por posición: {position_value:.2f} > {max_position_value:.2f}"
        
        # Verificar asignación total
        new_total_allocation = self.reserved_capital + position_value
        max_allocation = self.current_capital * self.max_allocation_percent
        if new_total_allocation > max_allocation:
            return False, f"Excede asignación máxima: {new_total_allocation:.2f} > {max_allocation:.2f}"
        
        # Verificar riesgo si está habilitado y hay stop loss
        if self.risk_limits_enabled and stop_loss:
            # Calcular riesgo en esta operación
            risk_amount = (price - stop_loss) * amount if price > stop_loss else (stop_loss - price) * amount
            risk_percent = risk_amount / self.current_capital
            
            if risk_percent > self.max_risk_per_trade:
                return False, f"Excede riesgo por operación: {risk_percent:.2%} > {self.max_risk_per_trade:.2%}"
        
        return True, "Posición permitida"
    
    async def distribute_profits(self, profits: float, 
                               distribution: Dict[str, float] = None) -> Dict[str, float]:
        """
        Distribuir ganancias según reglas definidas.
        
        Args:
            profits: Ganancias a distribuir
            distribution: Diccionario con porcentajes de distribución
            
        Returns:
            Distribución realizada
        """
        if profits <= 0:
            logger.warning(f"No hay ganancias para distribuir: {profits:.2f}")
            return {}
        
        # Distribución por defecto
        default_distribution = {
            "reinvest": 0.7,    # 70% reinvertir
            "reserve": 0.2,     # 20% reserva
            "withdraw": 0.1     # 10% retirada
        }
        
        # Usar distribución proporcionada o la predeterminada
        dist = distribution or default_distribution
        
        # Verificar que suma 1.0
        total = sum(dist.values())
        if abs(total - 1.0) > 0.0001:
            logger.warning(f"Distribución no suma 1.0: {total}, normalizando")
            dist = {k: v / total for k, v in dist.items()}
        
        # Calcular montos
        distribution_amounts = {
            key: profits * percentage
            for key, percentage in dist.items()
        }
        
        # Procesar reinversión
        if "reinvest" in distribution_amounts:
            reinvest_amount = distribution_amounts["reinvest"]
            if self.allow_compounding:
                self.current_capital += reinvest_amount
                self.available_capital += reinvest_amount
                logger.info(f"Reinvertido: {reinvest_amount:.2f}")
        
        # Procesamiento adicional según necesidades
        # ...
        
        logger.info(f"Ganancias distribuidas: {profits:.2f} según {dist}")
        return distribution_amounts
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del gestor de capital.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = super().get_stats()
        
        # Calcular estadísticas adicionales
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0
        max_drawdown = max([d.get("drawdown_percent", 0) for d in self.drawdowns], default=0)
        
        capital_stats = {
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "available_capital": self.available_capital,
            "reserved_capital": self.reserved_capital,
            "peak_capital": self.peak_capital,
            "total_return": (self.current_capital - self.initial_capital) / self.initial_capital,
            "current_drawdown": current_drawdown,
            "max_drawdown": max_drawdown,
            "allocation_percent": self.reserved_capital / self.current_capital if self.current_capital > 0 else 0,
            "allow_compounding": self.allow_compounding,
            "risk_limits_enabled": self.risk_limits_enabled
        }
        
        stats.update(capital_stats)
        return stats

class ProfitDistributor(GenesisComponent):
    """
    Distribuidor de ganancias que reparte beneficios según reglas definidas.
    """
    
    def __init__(self, mode: str = "SINGULARITY_V4"):
        """
        Inicializar distribuidor de ganancias.
        
        Args:
            mode: Modo trascendental
        """
        super().__init__("profit_distributor", mode)
        
        # Configuración por defecto
        self.distribution_rules = {
            "reinvest": 0.7,    # 70% reinvertir
            "reserve": 0.2,     # 20% reserva
            "withdraw": 0.1     # 10% retirada
        }
        
        # Distribución por umbral de ganancias
        self.threshold_rules = {
            1000: {"reinvest": 0.6, "reserve": 0.3, "withdraw": 0.1},   # >$1000
            5000: {"reinvest": 0.5, "reserve": 0.3, "withdraw": 0.2},   # >$5000
            10000: {"reinvest": 0.4, "reserve": 0.3, "withdraw": 0.3},  # >$10000
        }
        
        # Historial de distribuciones
        self.distribution_history: List[Dict[str, Any]] = []
        
        logger.info(f"Distribuidor de ganancias inicializado en modo {mode}")
    
    def set_distribution_rules(self, rules: Dict[str, float]) -> None:
        """
        Establecer reglas de distribución.
        
        Args:
            rules: Diccionario con porcentajes por categoría
        """
        # Verificar que suma 1.0
        total = sum(rules.values())
        if abs(total - 1.0) > 0.0001:
            logger.warning(f"Distribución no suma 1.0: {total}, normalizando")
            rules = {k: v / total for k, v in rules.items()}
        
        self.distribution_rules = rules
        logger.info(f"Reglas de distribución actualizadas: {rules}")
    
    def set_threshold_rules(self, thresholds: Dict[float, Dict[str, float]]) -> None:
        """
        Establecer reglas por umbral de ganancias.
        
        Args:
            thresholds: Diccionario con umbrales y reglas asociadas
        """
        # Validar cada conjunto de reglas
        for threshold, rules in thresholds.items():
            total = sum(rules.values())
            if abs(total - 1.0) > 0.0001:
                logger.warning(f"Distribución para umbral {threshold} no suma 1.0: {total}, normalizando")
                thresholds[threshold] = {k: v / total for k, v in rules.items()}
        
        self.threshold_rules = thresholds
        logger.info(f"Reglas por umbral actualizadas: {thresholds}")
    
    def _get_rules_for_amount(self, amount: float) -> Dict[str, float]:
        """
        Obtener reglas de distribución según el monto.
        
        Args:
            amount: Monto a distribuir
            
        Returns:
            Reglas de distribución aplicables
        """
        # Ordenar umbrales de mayor a menor
        thresholds = sorted(self.threshold_rules.keys(), reverse=True)
        
        # Encontrar el primer umbral que se cumple
        for threshold in thresholds:
            if amount >= threshold:
                return self.threshold_rules[threshold]
        
        # Si no se cumple ningún umbral, usar reglas por defecto
        return self.distribution_rules
    
    async def distribute_profits(self, profits: float, 
                               capital_manager: CapitalManager,
                               force_distribution: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Distribuir ganancias según reglas y actualizar capital.
        
        Args:
            profits: Ganancias a distribuir
            capital_manager: Gestor de capital
            force_distribution: Forzar distribución específica (opcional)
            
        Returns:
            Resultados de la distribución
        """
        if profits <= 0:
            logger.warning(f"No hay ganancias para distribuir: {profits:.2f}")
            return {"status": "no_profits", "amount": profits}
        
        # Determinar reglas a aplicar
        distribution_rules = force_distribution or self._get_rules_for_amount(profits)
        
        # Realizar distribución a través del gestor de capital
        distribution_amounts = await capital_manager.distribute_profits(
            profits=profits,
            distribution=distribution_rules
        )
        
        # Registrar distribución
        distribution_record = {
            "timestamp": time.time(),
            "profits": profits,
            "rules_applied": distribution_rules,
            "distribution": distribution_amounts
        }
        
        self.distribution_history.append(distribution_record)
        
        # Actualizar métricas
        self.update_metric("total_distributed", profits)
        self.update_metric("distribution_count", len(self.distribution_history))
        
        logger.info(f"Distribución completada: {profits:.2f} con reglas {distribution_rules}")
        
        return {
            "status": "success",
            "amount": profits,
            "distribution": distribution_amounts,
            "rules_applied": distribution_rules,
            "timestamp": time.time()
        }
    
    async def get_distribution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener historial de distribuciones.
        
        Args:
            limit: Número máximo de registros a retornar
            
        Returns:
            Lista de distribuciones
        """
        history = self.distribution_history[-limit:] if limit > 0 else self.distribution_history
        
        # Ordenar por fecha (más recientes primero)
        history.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        
        return history
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del distribuidor.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = super().get_stats()
        
        # Calcular estadísticas adicionales
        total_distributed = sum(dist.get("profits", 0) for dist in self.distribution_history)
        
        # Calcular totales por categoría
        category_totals = {}
        for dist in self.distribution_history:
            for category, amount in dist.get("distribution", {}).items():
                if category not in category_totals:
                    category_totals[category] = 0
                category_totals[category] += amount
        
        distributor_stats = {
            "distribution_count": len(self.distribution_history),
            "total_distributed": total_distributed,
            "category_totals": category_totals,
            "distribution_rules": self.distribution_rules,
            "threshold_rules": self.threshold_rules
        }
        
        stats.update(distributor_stats)
        return stats

class DecisionEngine(GenesisComponent, GenesisSingleton):
    """
    Motor de decisión con capacidades trascendentales.
    
    Este componente coordina todos los elementos de decisión, gestión de 
    posiciones, capital y distribución de ganancias.
    """
    
    def __init__(self, mode: str = "SINGULARITY_V4"):
        """
        Inicializar motor de decisión.
        
        Args:
            mode: Modo trascendental
        """
        super().__init__("decision_engine", mode)
        self.decision_makers: Dict[str, DecisionMaker] = {}
        self.decision_sequence = []
        self.position_manager = PositionManager(mode)
        self.capital_manager = CapitalManager(10000.0, mode)
        self.profit_distributor = ProfitDistributor(mode)
        self.db = TranscendentalDatabase()
        
        logger.info(f"Motor de decisión inicializado en modo {mode}")
    
    def register_decision_maker(self, maker_id: str, maker: DecisionMaker) -> None:
        """
        Registrar generador de decisiones.
        
        Args:
            maker_id: Identificador único del generador
            maker: Instancia del generador
        """
        self.decision_makers[maker_id] = maker
        logger.info(f"Generador de decisiones {maker_id} registrado")
    
    def set_decision_sequence(self, sequence: List[str]) -> None:
        """
        Establecer secuencia de generación de decisiones.
        
        Args:
            sequence: Lista de IDs de generadores en orden de ejecución
        """
        # Verificar que todos los generadores existan
        for maker_id in sequence:
            if maker_id not in self.decision_makers:
                raise ValueError(f"Generador {maker_id} no encontrado")
        
        self.decision_sequence = sequence
        logger.info(f"Secuencia de decisiones establecida: {sequence}")
    
    async def set_initial_capital(self, amount: float) -> None:
        """
        Establecer capital inicial.
        
        Args:
            amount: Monto de capital inicial
        """
        self.capital_manager.initial_capital = amount
        self.capital_manager.current_capital = amount
        self.capital_manager.available_capital = amount
        self.capital_manager.peak_capital = amount
        
        logger.info(f"Capital inicial establecido en {amount:.2f}")
    
    async def make_decisions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generar decisiones a través de la secuencia completa.
        
        Args:
            data: Datos con señales de análisis
            
        Returns:
            Datos con decisiones generadas
        """
        # Actualizar posiciones con precios actuales
        data = await self.position_manager.update_positions(data)
        
        # Actualizar información de capital
        positions = data.get("positions", {})
        closed_positions = data.get("closed_positions", [])
        
        capital_info = await self.capital_manager.update_capital(positions, closed_positions)
        data["capital_info"] = capital_info
        
        # Inicializar lista de decisiones
        if "decisions" not in data:
            data["decisions"] = []
        
        decisions_made = []
        
        # Si no hay secuencia establecida, usar todos en orden de registro
        if not self.decision_sequence:
            self.decision_sequence = list(self.decision_makers.keys())
        
        logger.info(f"Generando decisiones con {len(self.decision_sequence)} generadores")
        
        for maker_id in self.decision_sequence:
            maker = self.decision_makers[maker_id]
            logger.debug(f"Ejecutando generador: {maker.decision_maker_name} ({maker_id})")
            
            try:
                new_decisions = await maker.make_decision_with_resilience(data)
                decisions_made.extend(new_decisions)
                
                # Convertir decisiones a diccionarios
                decisions_dict = [d.to_dict() for d in new_decisions]
                data["decisions"].extend(decisions_dict)
                
                logger.debug(f"Generador {maker_id} generó {len(new_decisions)} decisiones")
                
            except Exception as e:
                logger.error(f"Error en generador {maker_id}: {str(e)}")
                self.register_operation(False)
                # Continuar con el siguiente generador para mantener resiliencia
        
        # Registrar operación exitosa
        self.register_operation(True)
        data["decision_timestamp"] = time.time()
        data["decisions_count"] = len(decisions_made)
        
        return data
    
    async def execute_decisions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecutar decisiones generadas (simulación).
        
        Args:
            data: Datos con decisiones
            
        Returns:
            Datos con resultados de ejecución
        """
        execution_results = []
        positions_updated = False
        
        # Extraer decisiones
        decisions = data.get("decisions", [])
        
        for decision_dict in decisions:
            # Convertir a objeto Decision si es diccionario
            if isinstance(decision_dict, dict):
                decision = Decision.from_dict(decision_dict)
            else:
                decision = decision_dict
            
            # Procesar según tipo de acción
            if decision.action == "buy":
                # Verificar si se puede abrir la posición
                allowed, reason = await self.capital_manager.can_open_position(
                    amount=decision.amount,
                    price=decision.price,
                    stop_loss=decision.source_signals.get("stop_loss")
                )
                
                if allowed:
                    # Abrir posición
                    position = await self.position_manager.add_position(
                        symbol=decision.symbol,
                        entry_price=decision.price,
                        size=decision.amount,
                        position_type="long",
                        stop_loss=decision.source_signals.get("stop_loss"),
                        take_profit=decision.source_signals.get("take_profit")
                    )
                    
                    result = {
                        "decision_id": decision.id,
                        "action": decision.action,
                        "symbol": decision.symbol,
                        "executed": True,
                        "position_id": position.get("symbol"),
                        "execution_price": decision.price,
                        "amount": decision.amount,
                        "timestamp": time.time()
                    }
                    
                    positions_updated = True
                else:
                    result = {
                        "decision_id": decision.id,
                        "action": decision.action,
                        "symbol": decision.symbol,
                        "executed": False,
                        "reason": reason,
                        "timestamp": time.time()
                    }
                
                execution_results.append(result)
            
            elif decision.action == "exit":
                # Cerrar posición
                closed_position = await self.position_manager.close_position(
                    symbol=decision.symbol,
                    exit_price=decision.price,
                    exit_reason="signal"
                )
                
                if closed_position:
                    # Registrar en decisions.closed_positions si no existe
                    if "closed_positions" not in data:
                        data["closed_positions"] = []
                    
                    data["closed_positions"].append(closed_position)
                    
                    result = {
                        "decision_id": decision.id,
                        "action": decision.action,
                        "symbol": decision.symbol,
                        "executed": True,
                        "execution_price": decision.price,
                        "pnl": closed_position.get("final_pnl_amount"),
                        "pnl_percent": closed_position.get("final_pnl_percent"),
                        "timestamp": time.time()
                    }
                    
                    positions_updated = True
                else:
                    result = {
                        "decision_id": decision.id,
                        "action": decision.action,
                        "symbol": decision.symbol,
                        "executed": False,
                        "reason": "Position not found",
                        "timestamp": time.time()
                    }
                
                execution_results.append(result)
        
        # Actualizar capital si fue necesario
        if positions_updated:
            positions = data.get("positions", {})
            closed_positions = data.get("closed_positions", [])
            
            capital_info = await self.capital_manager.update_capital(positions, closed_positions)
            data["capital_info"] = capital_info
        
        # Guardar resultados
        data["execution_results"] = execution_results
        data["execution_timestamp"] = time.time()
        
        return data
    
    async def distribute_profits(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Distribuir ganancias de operaciones cerradas.
        
        Args:
            data: Datos con posiciones cerradas
            
        Returns:
            Datos con resultados de distribución
        """
        # Extraer posiciones cerradas recientes
        closed_positions = data.get("closed_positions", [])
        
        # Filtrar posiciones cerradas que no han sido distribuidas
        # (basado en algún campo, por ejemplo distributed=True)
        undistributed = [
            p for p in closed_positions 
            if not p.get("profit_distributed", False)
        ]
        
        if not undistributed:
            logger.info("No hay ganancias pendientes para distribuir")
            return data
        
        total_profit = sum(p.get("final_pnl_amount", 0) for p in undistributed)
        
        # Solo distribuir si hay ganancias positivas
        if total_profit > 0:
            distribution_result = await self.profit_distributor.distribute_profits(
                profits=total_profit,
                capital_manager=self.capital_manager
            )
            
            # Marcar posiciones como distribuidas
            for position in undistributed:
                position["profit_distributed"] = True
                position["distribution_result"] = distribution_result
            
            # Guardar resultado
            if "distribution_results" not in data:
                data["distribution_results"] = []
            
            data["distribution_results"].append(distribution_result)
            data["distribution_timestamp"] = time.time()
            
            logger.info(f"Ganancias distribuidas: {total_profit:.2f}")
        else:
            logger.info(f"No hay ganancias positivas para distribuir: {total_profit:.2f}")
        
        return data
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del motor de decisión.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = super().get_stats()
        
        # Agregar estadísticas específicas
        engine_stats = {
            "decision_makers": len(self.decision_makers),
            "decision_sequence": self.decision_sequence,
            "position_stats": self.position_manager.get_stats(),
            "capital_stats": self.capital_manager.get_stats(),
            "distributor_stats": self.profit_distributor.get_stats()
        }
        
        stats.update(engine_stats)
        return stats
    
    async def initialize(self) -> bool:
        """
        Inicializar motor de decisión con generadores estándar.
        
        Returns:
            True si se inicializó correctamente
        """
        try:
            # Registrar generadores estándar
            self.register_decision_maker("signal_based", SignalBasedDecisionMaker(self.mode))
            self.register_decision_maker("risk_based", RiskBasedDecisionMaker(self.mode))
            
            # Establecer secuencia por defecto
            self.set_decision_sequence(["signal_based", "risk_based"])
            
            logger.info(f"Motor de decisión inicializado con {len(self.decision_makers)} generadores")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar motor de decisión: {str(e)}")
            return False

# Función de decisión para el pipeline
async def process_decision_making(data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Función de toma de decisiones para el pipeline.
    
    Args:
        data: Datos analizados con señales
        context: Contexto de ejecución
        
    Returns:
        Datos con decisiones generadas
    """
    engine = DecisionEngine()
    
    # Inicializar si es necesario
    if not engine.decision_makers:
        await engine.initialize()
    
    # Generar decisiones
    decisions_data = await engine.make_decisions(data)
    
    # Simular ejecución de decisiones
    executed_data = await engine.execute_decisions(decisions_data)
    
    # Distribuir ganancias
    final_data = await engine.distribute_profits(executed_data)
    
    # Registrar información en el contexto
    context["decision_makers"] = engine.decision_sequence
    context["decisions_count"] = len(final_data.get("decisions", []))
    context["positions_count"] = len(final_data.get("positions", {}))
    context["current_capital"] = final_data.get("capital_info", {}).get("current_capital", 0)
    
    logger.info(f"Toma de decisiones completada: {context['decisions_count']} decisiones generadas")
    return final_data

# Instancias globales para uso directo
decision_engine = DecisionEngine()
position_manager = PositionManager()
capital_manager = CapitalManager()
profit_distributor = ProfitDistributor()