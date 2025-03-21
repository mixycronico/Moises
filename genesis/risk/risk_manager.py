"""
Gestor de riesgos para el sistema Genesis.

Este módulo proporciona la gestión centralizada de riesgos para las operaciones
de trading, incluyendo validación de señales, gestión de posiciones y
monitoreo de métricas de riesgo.
"""

import logging
from typing import Dict, Any, Optional, List
import asyncio

from genesis.core.event_bus import EventBus
from genesis.risk.position_sizer import PositionSizer
from genesis.risk.stop_loss import StopLossCalculator

class RiskManager:
    """
    Gestor central de riesgos para el sistema de trading.
    
    Este componente se encarga de validar señales de trading, establecer los
    parámetros de gestión de riesgo, y monitorear el rendimiento y el riesgo.
    """
    
    def __init__(self, event_bus: EventBus, name: str = "risk_manager"):
        """
        Inicializar el gestor de riesgos.
        
        Args:
            event_bus: Bus de eventos del sistema
            name: Nombre del componente
        """
        self._name = name
        self._event_bus = event_bus
        self._logger = logging.getLogger(__name__)
        self._running = False
        
        # Componentes de gestión de riesgo
        self._position_sizer = PositionSizer()
        self._stop_loss_calculator = StopLossCalculator()
        
        # Métricas de riesgo por símbolo
        self._risk_metrics = {}
        
        # Límites de riesgo
        self._max_drawdown = 0.25  # 25% drawdown máximo
        self._max_open_positions = 5
        self._max_risk_per_trade = 0.02  # 2% de riesgo máximo por operación
        
        # Estado actual
        self._open_positions = {}
        self._current_drawdown = 0
    
    async def start(self) -> None:
        """Iniciar el gestor de riesgos."""
        if self._running:
            return
            
        self._running = True
        self._logger.info("Iniciando gestor de riesgos")
        
        # Suscribirse a eventos relevantes
        self._event_bus.subscribe("signal.generated", self.handle_event)
        self._event_bus.subscribe("trade.opened", self.handle_event)
        self._event_bus.subscribe("trade.closed", self.handle_event)
        self._event_bus.subscribe("market.update", self.handle_event)
    
    async def stop(self) -> None:
        """Detener el gestor de riesgos."""
        if not self._running:
            return
            
        self._running = False
        self._logger.info("Deteniendo gestor de riesgos")
        
        # No es necesario desuscribirse explícitamente con este bus de eventos
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del sistema.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
        """
        if event_type == "signal.generated":
            await self._handle_signal(data, source)
        elif event_type == "trade.opened":
            await self._handle_trade_opened(data, source)
        elif event_type == "trade.closed":
            await self._handle_trade_closed(data, source)
        elif event_type == "market.update":
            await self._handle_market_update(data, source)
    
    async def _handle_signal(self, data: Dict[str, Any], source: str) -> None:
        """
        Validar una señal de trading.
        
        Args:
            data: Datos de la señal
            source: Fuente de la señal
        """
        symbol = data.get("symbol")
        signal = data.get("signal")
        price = data.get("price")
        
        if not symbol or not signal or not price:
            self._logger.warning(f"Señal incompleta recibida: {data}")
            return
            
        # Inicializar métricas para el símbolo si no existen
        if symbol not in self._risk_metrics:
            self._risk_metrics[symbol] = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_profit": 0,
                "max_drawdown": 0
            }
        
        # Calcular el tamaño de posición
        position_size = self._position_sizer.calculate_position_size(price, symbol)
        
        # Calcular score de riesgo
        risk_score = self._calculate_risk_score(symbol)
        
        # Validar la señal basándose en las métricas de riesgo
        approved = True
        
        # No aprobar si excedemos el máximo de posiciones abiertas
        if len(self._open_positions) >= self._max_open_positions:
            approved = False
            self._logger.warning(f"Señal rechazada: máximo de posiciones abiertas alcanzado")
        
        # No aprobar si el drawdown actual excede el máximo
        if self._current_drawdown > self._max_drawdown:
            approved = False
            self._logger.warning(f"Señal rechazada: drawdown ({self._current_drawdown:.2%}) excede el máximo ({self._max_drawdown:.2%})")
        
        # Emitir evento de validación
        await self._event_bus.emit(
            "signal.validated",
            {
                "signal": signal,
                "symbol": symbol,
                "approved": approved,
                "risk_metrics": {"risk_score": risk_score},
                "position_size": position_size
            },
            self._name
        )
    
    async def _handle_trade_opened(self, data: Dict[str, Any], source: str) -> None:
        """
        Manejar apertura de una operación.
        
        Args:
            data: Datos de la operación
            source: Fuente del evento
        """
        symbol = data.get("symbol")
        side = data.get("side")
        price = data.get("price")
        amount = data.get("amount")
        order_id = data.get("order_id")
        exchange = data.get("exchange")
        
        if not all([symbol, side, price, amount, order_id, exchange]):
            self._logger.warning(f"Datos de operación incompletos: {data}")
            return
            
        # Registrar posición abierta
        self._open_positions[order_id] = {
            "symbol": symbol,
            "side": side,
            "entry_price": price,
            "amount": amount,
            "order_id": order_id,
            "exchange": exchange,
            "timestamp": data.get("timestamp", None)
        }
        
        # Calcular stop-loss
        stop_loss = self._stop_loss_calculator.calculate_stop_loss(
            entry_price=price,
            atr=data.get("atr", None),
            side=side
        )
        
        # Emitir evento de stop-loss
        await self._event_bus.emit(
            "trade.stop_loss_set",
            {
                "symbol": symbol,
                "price": stop_loss,
                "trade_id": order_id,
                "exchange": exchange
            },
            self._name
        )
    
    async def _handle_trade_closed(self, data: Dict[str, Any], source: str) -> None:
        """
        Manejar cierre de una operación.
        
        Args:
            data: Datos de la operación cerrada
            source: Fuente del evento
        """
        symbol = data.get("symbol")
        order_id = data.get("order_id")
        profit = data.get("profit", 0)
        profit_percentage = data.get("profit_percentage", 0)
        
        if not all([symbol, order_id]):
            self._logger.warning(f"Datos de operación cerrada incompletos: {data}")
            return
            
        # Eliminar de las posiciones abiertas
        if order_id in self._open_positions:
            del self._open_positions[order_id]
            
        # Actualizar métricas de riesgo
        if symbol not in self._risk_metrics:
            self._risk_metrics[symbol] = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_profit": 0,
                "max_drawdown": 0
            }
            
        self._risk_metrics[symbol]["total_trades"] += 1
        
        if profit > 0:
            self._risk_metrics[symbol]["winning_trades"] += 1
        else:
            self._risk_metrics[symbol]["losing_trades"] += 1
            
        self._risk_metrics[symbol]["total_profit"] += profit
        
        # Emitir evento de actualización de métricas
        await self._event_bus.emit(
            "risk.metrics_updated",
            {
                "symbol": symbol,
                "risk_score": self._calculate_risk_score(symbol),
                "updated_metrics": {
                    "total_trades": self._risk_metrics[symbol]["total_trades"],
                    "winning_trades": self._risk_metrics[symbol]["winning_trades"],
                    "profit": profit,
                    "profit_percentage": profit_percentage
                }
            },
            self._name
        )
    
    async def _handle_market_update(self, data: Dict[str, Any], source: str) -> None:
        """
        Manejar actualización de mercado.
        
        Args:
            data: Datos de mercado
            source: Fuente del evento
        """
        # Implementar en versiones futuras
        pass
    
    def _calculate_risk_score(self, symbol: str) -> float:
        """
        Calcular un score de riesgo para un símbolo.
        
        Args:
            symbol: Símbolo a evaluar
            
        Returns:
            Score de riesgo (0-100, donde 100 es el riesgo máximo)
        """
        if symbol not in self._risk_metrics:
            return 50.0  # Riesgo neutral para símbolos sin historial
            
        metrics = self._risk_metrics[symbol]
        
        # Calcular win rate
        total_trades = metrics["total_trades"]
        if total_trades == 0:
            win_rate = 0.5  # Neutral
        else:
            win_rate = metrics["winning_trades"] / total_trades
            
        # Ajustar riesgo según win rate (mejor win rate = menor riesgo)
        risk_score = 100 - (win_rate * 100)
        
        # Ajustar según drawdown si está disponible
        if metrics.get("max_drawdown"):
            risk_score += metrics["max_drawdown"] * 100
            
        # Normalizar entre 0-100
        risk_score = max(0, min(100, risk_score))
        
        return risk_score