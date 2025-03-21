import logging
from typing import Dict, Any, Tuple, Optional
import asyncio
from genesis.risk.position_sizer import PositionSizer
from genesis.risk.stop_loss import StopLossCalculator
from genesis.core.event_bus import EventBus

# Configuración de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class RiskManager:
    """Manejo de riesgo avanzado para operaciones de trading."""

    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Inicializar el gestor de riesgos.
        
        Args:
            event_bus: Bus de eventos para comunicación entre componentes
        """
        self._event_bus = event_bus
        self._position_sizer = PositionSizer()
        self._stop_loss_calculator = StopLossCalculator()
        self._running = False
        self._risk_metrics = {}  # Métricas de riesgo por símbolo
        self.logger = logging.getLogger("RiskManager")
        
    async def start(self) -> None:
        """Iniciar el gestor de riesgos."""
        self._running = True
        if self._event_bus:
            await self._event_bus.subscribe(self)
        self.logger.info("Gestor de riesgos iniciado")
        
    async def stop(self) -> None:
        """Detener el gestor de riesgos."""
        self._running = False
        self.logger.info("Gestor de riesgos detenido")
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
        """
        self.logger.debug(f"Evento recibido: {event_type} desde {source}")
        
        if event_type == "signal.generated":
            await self._handle_signal(data)
        elif event_type == "trade.opened":
            await self._handle_trade_opened(data)
        elif event_type == "trade.closed":
            await self._handle_trade_closed(data)
            
    async def _handle_signal(self, data: Dict[str, Any]) -> None:
        """
        Manejar señal de trading.
        
        Args:
            data: Datos de la señal
        """
        symbol = data.get("symbol")
        signal = data.get("signal")
        price = data.get("price")
        
        # Inicializar métricas de riesgo para el símbolo si no existen
        if symbol not in self._risk_metrics:
            self._risk_metrics[symbol] = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "profit": 0,
                "drawdown": 0,
                "max_drawdown": 0
            }
            
        # Calcular puntuación de riesgo
        risk_score = self._calculate_risk_score(symbol)
        
        # Calcular tamaño de posición
        position_size = self._position_sizer.calculate_position_size(price, symbol)
        
        # Emitir evento de validación de señal
        if self._event_bus:
            await self._event_bus.emit(
                "signal.validated",
                {
                    "signal": signal,
                    "symbol": symbol,
                    "approved": True,  # Por defecto aprobamos la señal
                    "risk_metrics": {"risk_score": risk_score},
                    "position_size": position_size
                }
            )
            
    async def _handle_trade_opened(self, data: Dict[str, Any]) -> None:
        """
        Manejar apertura de operación.
        
        Args:
            data: Datos de la operación
        """
        symbol = data.get("symbol")
        side = data.get("side")
        price = data.get("price")
        order_id = data.get("order_id")
        exchange = data.get("exchange")
        
        # Calcular stop loss
        stop_loss = self._stop_loss_calculator.calculate_stop_loss(price, 1000, side)  # Usar ATR=1000 como ejemplo
        
        # Emitir evento de stop loss
        if self._event_bus:
            await self._event_bus.emit(
                "trade.stop_loss_set",
                {
                    "symbol": symbol,
                    "price": stop_loss,
                    "trade_id": order_id,
                    "exchange": exchange
                }
            )
            
    async def _handle_trade_closed(self, data: Dict[str, Any]) -> None:
        """
        Manejar cierre de operación.
        
        Args:
            data: Datos de la operación
        """
        symbol = data.get("symbol")
        profit = data.get("profit", 0)
        profit_percentage = data.get("profit_percentage", 0)
        
        # Inicializar métricas de riesgo para el símbolo si no existen
        # Esto es necesario para el test test_risk_manager_handle_trade_closed
        if symbol not in self._risk_metrics:
            self._risk_metrics[symbol] = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "profit": 0,
                "drawdown": 0,
                "max_drawdown": 0
            }
        
        # Actualizar métricas de riesgo
        self._risk_metrics[symbol]["total_trades"] += 1
        if profit > 0:
            self._risk_metrics[symbol]["winning_trades"] += 1
        else:
            self._risk_metrics[symbol]["losing_trades"] += 1
        self._risk_metrics[symbol]["profit"] += profit
            
        # Emitir evento de actualización de métricas
        if self._event_bus:
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
                }
            )
            
    def _calculate_risk_score(self, symbol: str) -> float:
        """
        Calcular puntuación de riesgo para un símbolo.
        
        Args:
            symbol: Símbolo de trading
            
        Returns:
            Puntuación de riesgo (0-100)
        """
        # Implementación simple de puntuación de riesgo
        # En una implementación real, consideraríamos más factores
        metrics = self._risk_metrics.get(symbol, {})
        total_trades = metrics.get("total_trades", 0)
        
        if total_trades == 0:
            return 50.0  # Riesgo neutral para símbolos sin historial
            
        winning_trades = metrics.get("winning_trades", 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Puntuación simple basada en win rate
        risk_score = win_rate * 100
        
        return min(max(risk_score, 0), 100)  # Limitar entre 0 y 100