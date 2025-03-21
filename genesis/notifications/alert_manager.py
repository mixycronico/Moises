"""
Gestor de alertas para el sistema Genesis.

Este módulo proporciona funcionalidades para crear, gestionar y enviar
alertas basadas en diferentes condiciones del mercado y del sistema.
"""

import asyncio
import time
import threading
import random
import os
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import logging

from genesis.core.base import Component
from genesis.utils.logger import setup_logging


class AlertCondition:
    """Clase para representar una condición de alerta."""
    
    def __init__(
        self,
        name: str,
        symbol: str,
        condition_type: str,
        params: Dict[str, Any],
        message_template: str
    ):
        """
        Inicializar una condición de alerta.
        
        Args:
            name: Nombre de la alerta
            symbol: Símbolo de trading
            condition_type: Tipo de condición (precio, indicador, etc.)
            params: Parámetros específicos de la condición
            message_template: Plantilla para el mensaje de alerta
        """
        self.name = name
        self.symbol = symbol
        self.condition_type = condition_type
        self.params = params
        self.message_template = message_template
        self.last_triggered = 0  # Timestamp de la última vez que se activó
        self.is_active = True
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir a diccionario.
        
        Returns:
            Diccionario con los datos de la alerta
        """
        return {
            "name": self.name,
            "symbol": self.symbol,
            "condition_type": self.condition_type,
            "params": self.params,
            "message_template": self.message_template,
            "is_active": self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertCondition':
        """
        Crear desde diccionario.
        
        Args:
            data: Diccionario con los datos de la alerta
            
        Returns:
            Instancia de AlertCondition
        """
        condition = cls(
            data["name"],
            data["symbol"],
            data["condition_type"],
            data["params"],
            data["message_template"]
        )
        condition.is_active = data.get("is_active", True)
        return condition


class AlertManager(Component):
    """
    Gestor de alertas para el sistema Genesis.
    
    Este componente monitorea diferentes condiciones del mercado
    y genera alertas cuando se cumplen.
    """
    
    def __init__(
        self,
        name: str = "alert_manager",
        check_interval: float = 10.0
    ):
        """
        Inicializar el gestor de alertas.
        
        Args:
            name: Nombre del componente
            check_interval: Intervalo en segundos para comprobar alertas
        """
        super().__init__(name)
        self.logger = setup_logging(name)
        self.check_interval = check_interval
        
        # Condiciones de alerta registradas
        self.conditions: Dict[str, AlertCondition] = {}
        
        # Callbacks para evaluar condiciones
        self.condition_evaluators: Dict[str, Callable] = {
            "price": self._evaluate_price_condition,
            "indicator": self._evaluate_indicator_condition,
            "volume": self._evaluate_volume_condition,
            "time": self._evaluate_time_condition,
            "custom": self._evaluate_custom_condition
        }
        
        # Cache de datos del mercado
        self.market_data_cache: Dict[str, Dict[str, Any]] = {}
        
        # Ruta para persistencia
        self.storage_path = "data/alerts"
        os.makedirs(self.storage_path, exist_ok=True)
    
    async def start(self) -> None:
        """Iniciar el gestor de alertas."""
        await super().start()
        
        # Cargar alertas guardadas
        self._load_alerts()
        
        # Iniciar bucle de comprobación
        self.check_task = asyncio.create_task(self._check_loop())
        
        self.logger.info("Gestor de alertas iniciado")
    
    async def stop(self) -> None:
        """Detener el gestor de alertas."""
        # Cancelar tarea de comprobación
        if hasattr(self, 'check_task') and self.check_task:
            self.check_task.cancel()
            try:
                await self.check_task
            except asyncio.CancelledError:
                pass
        
        # Guardar alertas
        self._save_alerts()
        
        await super().stop()
        self.logger.info("Gestor de alertas detenido")
    
    def add_condition(self, condition: AlertCondition) -> bool:
        """
        Añadir una condición de alerta.
        
        Args:
            condition: Condición de alerta
            
        Returns:
            True si se añadió correctamente, False en caso contrario
        """
        # Generar ID único si no existe
        condition_id = f"{condition.symbol}_{condition.name}_{int(time.time())}"
        
        # Comprobar si el tipo de condición está soportado
        if condition.condition_type not in self.condition_evaluators:
            self.logger.error(f"Tipo de condición no soportado: {condition.condition_type}")
            return False
        
        # Añadir la condición
        self.conditions[condition_id] = condition
        self.logger.info(f"Condición de alerta añadida: {condition_id}")
        
        # Guardar alertas
        self._save_alerts()
        
        return True
    
    def remove_condition(self, condition_id: str) -> bool:
        """
        Eliminar una condición de alerta.
        
        Args:
            condition_id: ID de la condición
            
        Returns:
            True si se eliminó correctamente, False en caso contrario
        """
        if condition_id in self.conditions:
            del self.conditions[condition_id]
            self.logger.info(f"Condición de alerta eliminada: {condition_id}")
            
            # Guardar alertas
            self._save_alerts()
            
            return True
        
        return False
    
    def enable_condition(self, condition_id: str) -> bool:
        """
        Activar una condición de alerta.
        
        Args:
            condition_id: ID de la condición
            
        Returns:
            True si se activó correctamente, False en caso contrario
        """
        if condition_id in self.conditions:
            self.conditions[condition_id].is_active = True
            self.logger.info(f"Condición de alerta activada: {condition_id}")
            return True
        
        return False
    
    def disable_condition(self, condition_id: str) -> bool:
        """
        Desactivar una condición de alerta.
        
        Args:
            condition_id: ID de la condición
            
        Returns:
            True si se desactivó correctamente, False en caso contrario
        """
        if condition_id in self.conditions:
            self.conditions[condition_id].is_active = False
            self.logger.info(f"Condición de alerta desactivada: {condition_id}")
            return True
        
        return False
    
    def get_conditions(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtener todas las condiciones de alerta.
        
        Returns:
            Diccionario de condiciones
        """
        return {id: condition.to_dict() for id, condition in self.conditions.items()}
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Este método procesa eventos que actualizan los datos del mercado
        y pueden activar alertas.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente de origen
        """
        # Actualizar cache de datos del mercado
        if event_type == "market.ticker_updated":
            self._update_market_data(data)
        elif event_type == "market.indicator_updated":
            self._update_indicator_data(data)
    
    async def _check_loop(self) -> None:
        """Bucle principal para comprobar condiciones de alerta."""
        self.logger.info("Iniciando bucle de comprobación de alertas")
        
        while True:
            # Comprobar todas las condiciones activas
            for condition_id, condition in self.conditions.items():
                if not condition.is_active:
                    continue
                
                # Evaluar la condición
                try:
                    is_triggered, details = await self._evaluate_condition(condition)
                    
                    # Si se activó, emitir evento de alerta
                    if is_triggered:
                        # Actualizar timestamp de última activación
                        condition.last_triggered = time.time()
                        
                        # Crear mensaje personalizado
                        message = self._format_alert_message(condition, details)
                        
                        # Determinar severidad
                        severity = condition.params.get("severity", "info")
                        
                        # Emitir evento
                        await self.emit_event("alert.triggered", {
                            "type": condition.condition_type,
                            "symbol": condition.symbol,
                            "name": condition.name,
                            "message": message,
                            "severity": severity,
                            "details": details,
                            "condition_id": condition_id,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        self.logger.info(f"Alerta activada: {condition_id} - {message}")
                
                except Exception as e:
                    self.logger.error(f"Error al evaluar condición {condition_id}: {e}")
            
            # Esperar el intervalo configurado
            await asyncio.sleep(self.check_interval)
    
    async def _evaluate_condition(self, condition: AlertCondition) -> tuple:
        """
        Evaluar una condición de alerta.
        
        Args:
            condition: Condición a evaluar
            
        Returns:
            Tupla (is_triggered, details)
        """
        # Obtener el evaluador adecuado
        evaluator = self.condition_evaluators.get(condition.condition_type)
        if not evaluator:
            self.logger.error(f"No hay evaluador para el tipo: {condition.condition_type}")
            return False, {}
        
        # Evaluar la condición
        return await evaluator(condition)
    
    async def _evaluate_price_condition(self, condition: AlertCondition) -> tuple:
        """
        Evaluar una condición de precio.
        
        Args:
            condition: Condición a evaluar
            
        Returns:
            Tupla (is_triggered, details)
        """
        # Obtener datos actuales del mercado
        market_data = self.market_data_cache.get(condition.symbol)
        if not market_data:
            return False, {}
        
        # Obtener precio actual
        current_price = market_data.get("price")
        if current_price is None:
            return False, {}
        
        # Obtener parámetros de la condición
        comparator = condition.params.get("comparator", "==")
        target_price = condition.params.get("price", 0)
        
        # Evaluar según el comparador
        is_triggered = False
        if comparator == "==":
            is_triggered = current_price == target_price
        elif comparator == "!=":
            is_triggered = current_price != target_price
        elif comparator == ">":
            is_triggered = current_price > target_price
        elif comparator == ">=":
            is_triggered = current_price >= target_price
        elif comparator == "<":
            is_triggered = current_price < target_price
        elif comparator == "<=":
            is_triggered = current_price <= target_price
        
        details = {
            "current_price": current_price,
            "target_price": target_price,
            "comparator": comparator
        }
        
        return is_triggered, details
    
    async def _evaluate_indicator_condition(self, condition: AlertCondition) -> tuple:
        """
        Evaluar una condición de indicador.
        
        Args:
            condition: Condición a evaluar
            
        Returns:
            Tupla (is_triggered, details)
        """
        # Simulación simple de evaluación de indicadores
        # En una implementación real, se consultaría el valor actual del indicador
        
        # Obtener parámetros
        indicator = condition.params.get("indicator", "")
        comparator = condition.params.get("comparator", "==")
        threshold = condition.params.get("threshold", 0)
        
        # Aquí se consultaría el valor actual del indicador
        # Por ahora, simulamos un valor
        current_value = random.random() * 100  # Simulación
        
        # Evaluar según el comparador
        is_triggered = False
        if comparator == "==":
            is_triggered = current_value == threshold
        elif comparator == "!=":
            is_triggered = current_value != threshold
        elif comparator == ">":
            is_triggered = current_value > threshold
        elif comparator == ">=":
            is_triggered = current_value >= threshold
        elif comparator == "<":
            is_triggered = current_value < threshold
        elif comparator == "<=":
            is_triggered = current_value <= threshold
        
        details = {
            "indicator": indicator,
            "current_value": current_value,
            "threshold": threshold,
            "comparator": comparator
        }
        
        return is_triggered, details
    
    async def _evaluate_volume_condition(self, condition: AlertCondition) -> tuple:
        """
        Evaluar una condición de volumen.
        
        Args:
            condition: Condición a evaluar
            
        Returns:
            Tupla (is_triggered, details)
        """
        # Obtener datos actuales del mercado
        market_data = self.market_data_cache.get(condition.symbol)
        if not market_data:
            return False, {}
        
        # Obtener volumen actual
        current_volume = market_data.get("volume", 0)
        
        # Obtener parámetros de la condición
        comparator = condition.params.get("comparator", "==")
        threshold = condition.params.get("threshold", 0)
        
        # Evaluar según el comparador
        is_triggered = False
        if comparator == "==":
            is_triggered = current_volume == threshold
        elif comparator == "!=":
            is_triggered = current_volume != threshold
        elif comparator == ">":
            is_triggered = current_volume > threshold
        elif comparator == ">=":
            is_triggered = current_volume >= threshold
        elif comparator == "<":
            is_triggered = current_volume < threshold
        elif comparator == "<=":
            is_triggered = current_volume <= threshold
        
        details = {
            "current_volume": current_volume,
            "threshold": threshold,
            "comparator": comparator
        }
        
        return is_triggered, details
    
    async def _evaluate_time_condition(self, condition: AlertCondition) -> tuple:
        """
        Evaluar una condición de tiempo.
        
        Args:
            condition: Condición a evaluar
            
        Returns:
            Tupla (is_triggered, details)
        """
        # Obtener hora actual
        now = datetime.now()
        
        # Obtener parámetros de la condición
        time_str = condition.params.get("time", "")
        days = condition.params.get("days", [])  # 0=lunes, 6=domingo
        
        # Si no hay configuración de tiempo, no se activa
        if not time_str:
            return False, {}
        
        # Parsear hora
        try:
            hour, minute = map(int, time_str.split(":"))
            target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        except Exception:
            return False, {"error": "Formato de hora inválido"}
        
        # Comprobar si la hora actual está dentro del rango de activación
        # (permitimos un margen de un minuto)
        time_diff = abs((now - target_time).total_seconds())
        is_time_match = time_diff <= 60  # Un minuto de margen
        
        # Comprobar si el día actual está en la lista de días
        is_day_match = True
        if days:
            current_day = now.weekday()  # 0=lunes, 6=domingo
            is_day_match = current_day in days
        
        is_triggered = is_time_match and is_day_match
        
        details = {
            "current_time": now.strftime("%H:%M"),
            "target_time": time_str,
            "current_day": now.weekday(),
            "target_days": days
        }
        
        return is_triggered, details
    
    async def _evaluate_custom_condition(self, condition: AlertCondition) -> tuple:
        """
        Evaluar una condición personalizada.
        
        Args:
            condition: Condición a evaluar
            
        Returns:
            Tupla (is_triggered, details)
        """
        # Esta implementación dependerá de la lógica específica
        # Por ahora, devolvemos False
        return False, {}
    
    def _update_market_data(self, data: Dict[str, Any]) -> None:
        """
        Actualizar datos del mercado en el cache.
        
        Args:
            data: Datos del mercado
        """
        symbol = data.get("symbol")
        if not symbol:
            return
        
        # Actualizar o crear entrada en el cache
        if symbol not in self.market_data_cache:
            self.market_data_cache[symbol] = {}
        
        # Actualizar datos
        self.market_data_cache[symbol].update({
            "price": data.get("price"),
            "volume": data.get("volume"),
            "bid": data.get("bid"),
            "ask": data.get("ask"),
            "high": data.get("high"),
            "low": data.get("low"),
            "timestamp": data.get("timestamp") or datetime.now().isoformat()
        })
    
    def _update_indicator_data(self, data: Dict[str, Any]) -> None:
        """
        Actualizar datos de indicadores en el cache.
        
        Args:
            data: Datos del indicador
        """
        symbol = data.get("symbol")
        indicator = data.get("indicator")
        
        if not symbol or not indicator:
            return
        
        # Actualizar o crear entrada en el cache
        if symbol not in self.market_data_cache:
            self.market_data_cache[symbol] = {}
        
        if "indicators" not in self.market_data_cache[symbol]:
            self.market_data_cache[symbol]["indicators"] = {}
        
        # Actualizar indicador
        self.market_data_cache[symbol]["indicators"][indicator] = {
            "value": data.get("value"),
            "timestamp": data.get("timestamp") or datetime.now().isoformat()
        }
    
    def _format_alert_message(self, condition: AlertCondition, details: Dict[str, Any]) -> str:
        """
        Formatear mensaje de alerta.
        
        Args:
            condition: Condición que se activó
            details: Detalles de la activación
            
        Returns:
            Mensaje formateado
        """
        # Usar plantilla o crear mensaje predeterminado
        if condition.message_template:
            try:
                # Formatear usando la plantilla y los detalles
                return condition.message_template.format(**details)
            except Exception as e:
                self.logger.error(f"Error al formatear mensaje: {e}")
        
        # Mensaje predeterminado si no hay plantilla o hay error
        return f"Alerta: {condition.name} para {condition.symbol}"
    
    def _save_alerts(self) -> None:
        """Guardar condiciones de alerta en disco."""
        try:
            alerts_data = {id: condition.to_dict() for id, condition in self.conditions.items()}
            file_path = os.path.join(self.storage_path, "alerts.json")
            
            with open(file_path, 'w') as f:
                json.dump(alerts_data, f, indent=2)
            
            self.logger.debug(f"Alertas guardadas en {file_path}")
        except Exception as e:
            self.logger.error(f"Error al guardar alertas: {e}")
    
    def _load_alerts(self) -> None:
        """Cargar condiciones de alerta desde disco."""
        file_path = os.path.join(self.storage_path, "alerts.json")
        
        if not os.path.exists(file_path):
            self.logger.debug(f"No hay archivo de alertas en {file_path}")
            return
        
        try:
            with open(file_path, 'r') as f:
                alerts_data = json.load(f)
            
            # Crear condiciones desde los datos
            for id, data in alerts_data.items():
                self.conditions[id] = AlertCondition.from_dict(data)
            
            self.logger.info(f"Cargadas {len(self.conditions)} alertas")
        except Exception as e:
            self.logger.error(f"Error al cargar alertas: {e}")


# Exportación para uso fácil
alert_manager = AlertManager()