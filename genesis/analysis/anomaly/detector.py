"""
Detector de anomalías para el sistema Genesis.

Este módulo proporciona capacidades para detectar anomalías en datos de mercado,
como spikes, spreads extremos, manipulación y errores, utilizando técnicas
de machine learning y análisis estadístico.
"""

import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from collections import deque
from sklearn.ensemble import IsolationForest

from genesis.core.base import Component
from genesis.notifications.email_notifier import EmailNotifier

class AnomalyDetector(Component):
    """
    Detecta spikes, spreads extremos, manipulación y errores en datos de mercado.
    Usa Isolation Forest y análisis estadístico para identificar anomalías en tiempo real.
    """
    
    def __init__(
        self,
        threshold: float = 3.0,
        window_size: int = 100,
        notifier: Optional[EmailNotifier] = None,
        contamination: float = 0.01,
        alert_interval: float = 300.0,
        name: str = "anomaly_detector"
    ):
        """
        Inicializar el detector de anomalías.
        
        Args:
            threshold: Umbral de desviaciones estándar para detección
            window_size: Tamaño de la ventana de datos
            notifier: Objeto notificador para enviar alertas
            contamination: Parámetro de contaminación para Isolation Forest
            alert_interval: Intervalo mínimo entre alertas en segundos
            name: Nombre del componente
        """
        super().__init__(name)
        self.price_history: Dict[str, deque] = {}
        self.threshold = max(0.1, float(threshold))  # Evita umbrales inválidos
        self.window_size = max(10, int(window_size))  # Mínimo razonable
        self.model = IsolationForest(
            n_estimators=100,
            contamination=max(0.001, min(0.5, contamination)),  # Rango válido
            random_state=42,
            n_jobs=-1,  # Usa todos los núcleos
        )
        self.notifier = notifier
        self.last_alert: Dict[str, float] = {}
        self.alert_interval = max(1.0, float(alert_interval))  # Mínimo 1 segundo
        self._min_samples = 20  # Mínimo para análisis estadístico/IA
        self.logger = logging.getLogger(__name__)
        
        # Tarea de monitoreo
        self.monitor_task = None
        
    async def start(self) -> None:
        """Iniciar el detector de anomalías."""
        await super().start()
        self.logger.info("Detector de anomalías iniciado")
        
    async def stop(self) -> None:
        """Detener el detector de anomalías."""
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
                
        await super().stop()
        self.logger.info("Detector de anomalías detenido")
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente de origen
        """
        if event_type == "market.tick":
            symbol = data.get("symbol")
            price = data.get("price")
            
            if symbol and price:
                self.update_price(symbol, price)
                anomaly = await self.detect(symbol)
                
                if anomaly:
                    await self.emit_event("market.anomaly_detected", anomaly)

    def _should_alert(self, symbol: str) -> bool:
        """
        Determina si se debe enviar una alerta según el intervalo.
        
        Args:
            symbol: Símbolo a verificar
            
        Returns:
            True si se debe enviar alerta, False en caso contrario
        """
        now = datetime.utcnow().timestamp()
        last = self.last_alert.get(symbol, 0.0)
        if now - last >= self.alert_interval:
            self.last_alert[symbol] = now
            return True
        return False

    def update_price(self, symbol: str, price: float) -> None:
        """
        Actualiza el historial de precios con validación.
        
        Args:
            symbol: Símbolo del par de trading
            price: Precio actual
        """
        if not isinstance(symbol, str) or not symbol.strip():
            self.logger.debug(f"Symbol inválido: {symbol}")
            return
        if not isinstance(price, (int, float)) or price <= 0:
            self.logger.debug(f"Precio inválido para {symbol}: {price}")
            return

        symbol = symbol.strip()
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.window_size)
        self.price_history[symbol].append(float(price))

    async def detect(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Detecta anomalías usando estadísticas y Isolation Forest.
        
        Args:
            symbol: Símbolo del par de trading
            
        Returns:
            Diccionario con información de la anomalía o None si no hay anomalía
        """
        hist = self.price_history.get(symbol)
        if not hist or len(hist) < self._min_samples:
            return None

        prices = np.array(hist, dtype=np.float32).reshape(-1, 1)
        current_price = prices[-1, 0]

        # Cálculo estadístico eficiente
        mean = np.mean(prices, axis=0)[0]
        std = np.std(prices, axis=0, ddof=1)[0]  # ddof=1 para muestra
        z_score = abs(current_price - mean) / (std + 1e-8)

        # Detección con IA
        try:
            prediction = self.model.fit_predict(prices)[-1]  # -1 es anomalía
        except ValueError as e:
            self.logger.warning(f"Error en IsolationForest para {symbol}: {e}")
            prediction = 1  # Normal si falla

        # Evaluar anomalía
        is_anomaly = z_score > self.threshold or prediction == -1
        if is_anomaly and self._should_alert(symbol):
            mensaje = (
                f"Anomalía detectada en {symbol}\n"
                f"Z-Score: {z_score:.2f}\n"
                f"Precio actual: {current_price:.2f}\n"
                f"Promedio: {mean:.2f}, STD: {std:.2f}\n"
                f"IA: {'Manipulación detectada' if prediction == -1 else 'Normal'}"
            )
            self.logger.warning(mensaje)
            if self.notifier:
                asyncio.create_task(  # No bloquea el flujo principal
                    self.notifier.send_email(
                        subject=f"[ALERTA Genesis] Anomalía en {symbol}",
                        message=mensaje,
                    )
                )
            return {
                "symbol": symbol,
                "z_score": float(z_score),
                "manipulation": prediction == -1,
                "price": float(current_price),
                "mean": float(mean),
                "std": float(std),
                "message": mensaje,
                "timestamp": datetime.utcnow().isoformat()
            }
        return None

    def start_monitoring(self, symbols: List[str], price_fetcher: Callable[[str], float]) -> None:
        """
        Inicia el monitoreo asíncrono de símbolos.
        
        Args:
            symbols: Lista de símbolos a monitorear
            price_fetcher: Función para obtener precios
        """
        if self.monitor_task:
            self.monitor_task.cancel()
            
        self.monitor_task = asyncio.create_task(self.monitor(symbols, price_fetcher))
            
    async def monitor(self, symbols: List[str], price_fetcher: Callable[[str], float]) -> None:
        """
        Monitorea símbolos en tiempo real detectando anomalías.
        
        Args:
            symbols: Lista de símbolos a monitorear
            price_fetcher: Función asíncrona que devuelve el precio de un símbolo
        """
        if not symbols or not callable(price_fetcher):
            self.logger.error("Lista de símbolos vacía o price_fetcher no válido")
            return

        while self.running:
            try:
                tasks = []
                for symbol in symbols:
                    try:
                        price = await price_fetcher(symbol)
                        if price is not None:
                            self.update_price(symbol, price)
                            tasks.append(self.detect(symbol))
                    except Exception as e:
                        self.logger.debug(f"Error al obtener precio de {symbol}: {e}")
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        self.logger.debug(f"Error en detección: {result}")
                    elif result:  # Si se detectó anomalía
                        await self.emit_event("market.anomaly_detected", result)
                        
                await asyncio.sleep(3.0)
            except asyncio.CancelledError:
                self.logger.info("Monitoreo cancelado")
                break
            except Exception as e:
                self.logger.error(f"Error en bucle de monitoreo: {e}")
                await asyncio.sleep(5.0)  # Pausa antes de reintentar