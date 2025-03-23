"""
Estrategia de escalabilidad adaptativa integrada con el sistema Genesis.

Este módulo implementa una estrategia que utiliza el motor de escalabilidad
adaptativa para distribuir capital entre múltiples instrumentos, manteniendo
eficiencia óptima a medida que el capital crece.
"""

import logging
import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import pandas as pd
import numpy as np

from genesis.accounting.predictive_scaling import PredictiveScalingEngine
from genesis.accounting.balance_manager import CapitalScalingManager
from genesis.strategies.base_strategy import BaseStrategy
from genesis.risk.adaptive_risk_manager import AdaptiveRiskManager
from genesis.db.transcendental_database import TranscendentalDatabase
from genesis.utils.helpers import format_timestamp, safe_divide, generate_id

class AdaptiveScalingStrategy(BaseStrategy):
    """
    Estrategia que implementa escalabilidad adaptativa.
    
    Esta estrategia:
    1. Monitorea y registra la eficiencia observada de los instrumentos
    2. Construye modelos predictivos de eficiencia vs capital
    3. Detecta puntos de saturación para cada instrumento
    4. Optimiza la asignación de capital entre instrumentos
    5. Adapta dinámicamente parámetros en función del nivel de capital
    """
    
    def __init__(
        self,
        strategy_id: str = None,
        name: str = "Estrategia de Escalabilidad Adaptativa",
        symbols: List[str] = None,
        config: Dict[str, Any] = None,
        db: Optional[TranscendentalDatabase] = None
    ):
        """
        Inicializar la estrategia de escalabilidad adaptativa.
        
        Args:
            strategy_id: ID único de la estrategia
            name: Nombre descriptivo
            symbols: Lista de símbolos a operar
            config: Configuración adicional
            db: Conexión a base de datos transcendental
        """
        super().__init__(
            strategy_id=strategy_id or f"adaptive_scaling_{generate_id()}",
            name=name
        )
        
        self.logger = logging.getLogger(f'genesis.strategies.adaptive_scaling')
        self.config = config or {}
        self.symbols = symbols or []
        self.db = db
        
        # Componentes internos
        self.engine: Optional[PredictiveScalingEngine] = None
        self.scaling_manager: Optional[CapitalScalingManager] = None
        self.risk_manager: Optional[AdaptiveRiskManager] = None
        
        # Estado actual
        self.current_allocations: Dict[str, float] = {}
        self.performance_metrics: Dict[str, Any] = {
            "efficiency_by_symbol": {},
            "allocation_history": [],
            "saturation_points": {},
            "capital_utilization": 0.0,
            "entropy": 0.0
        }
        
        # Configuración por defecto
        self.min_efficiency_threshold = self.config.get('min_efficiency_threshold', 0.5)
        self.rebalance_threshold = self.config.get('rebalance_threshold', 0.1)  # 10% de desviación para rebalancear
        self.update_interval = self.config.get('update_interval', 86400)  # 1 día por defecto
        self.last_update_time = 0
        
        self.logger.info(f"Estrategia {self.name} inicializada con {len(self.symbols)} símbolos")
    
    async def initialize(self) -> bool:
        """
        Inicializar la estrategia y sus componentes.
        
        Returns:
            True si la inicialización fue exitosa
        """
        try:
            self.logger.info(f"Inicializando estrategia {self.name}...")
            
            # Inicializar motor predictivo
            self.engine = PredictiveScalingEngine(
                config={
                    "default_model_type": self.config.get('model_type', 'polynomial'),
                    "cache_ttl": self.config.get('cache_ttl', 300),
                    "auto_train": True,
                    "confidence_threshold": self.config.get('confidence_threshold', 0.6)
                }
            )
            
            # Cargar datos históricos si están disponibles
            if self.db:
                await self._load_historical_efficiency_data()
            
            # Inicializar scaling manager si está disponible
            if 'capital_scaling_manager' in self.required_components:
                self.scaling_manager = self.required_components['capital_scaling_manager']
            
            # Inicializar risk manager si está disponible
            if 'risk_manager' in self.required_components:
                self.risk_manager = self.required_components['risk_manager']
            
            # Cargar estado anterior si existe
            await self._load_state()
            
            self.initialized = True
            self.logger.info(f"Estrategia {self.name} inicializada correctamente")
            return True
            
        except Exception as e:
            self.logger.error(f"Error inicializando estrategia {self.name}: {str(e)}")
            return False
    
    async def _load_historical_efficiency_data(self) -> None:
        """
        Cargar datos históricos de eficiencia desde la base de datos.
        """
        try:
            if not self.db:
                return
                
            self.logger.info("Cargando datos históricos de eficiencia...")
            
            # Consultar registros de eficiencia
            query = """
                SELECT symbol, capital_level, efficiency, roi, sharpe, max_drawdown, win_rate
                FROM efficiency_records
                WHERE symbol = ANY($1)
                ORDER BY symbol, capital_level
            """
            records = await self.db.fetch(query, [self.symbols])
            
            # Cargar en el motor predictivo
            count_by_symbol = {}
            for record in records:
                symbol = record['symbol']
                
                # Incrementar contador
                if symbol not in count_by_symbol:
                    count_by_symbol[symbol] = 0
                count_by_symbol[symbol] += 1
                
                # Crear métricas adicionales
                metrics = {
                    "roi": record.get('roi'),
                    "sharpe": record.get('sharpe'),
                    "max_drawdown": record.get('max_drawdown'),
                    "win_rate": record.get('win_rate')
                }
                
                # Añadir al motor
                await self.engine.add_efficiency_record(
                    symbol=symbol,
                    capital=record['capital_level'],
                    efficiency=record['efficiency'],
                    metrics=metrics
                )
            
            # Registrar resultados
            symbols_loaded = len(count_by_symbol)
            total_records = sum(count_by_symbol.values())
            self.logger.info(f"Cargados {total_records} registros de eficiencia para {symbols_loaded} símbolos")
            
        except Exception as e:
            self.logger.error(f"Error cargando datos históricos: {str(e)}")
    
    async def _load_state(self) -> None:
        """
        Cargar el estado anterior de la estrategia si existe.
        """
        file_path = f"checkpoints/{self.strategy_id}_state.json"
        
        if not os.path.exists(file_path):
            self.logger.info("No se encontró estado anterior de la estrategia")
            return
            
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
                
            # Cargar asignaciones actuales
            if 'current_allocations' in state:
                self.current_allocations = state['current_allocations']
                
            # Cargar métricas de rendimiento
            if 'performance_metrics' in state:
                self.performance_metrics = state['performance_metrics']
                
            # Cargar timestamp de última actualización
            if 'last_update_time' in state:
                self.last_update_time = state['last_update_time']
                
            self.logger.info(f"Estado anterior cargado: {len(self.current_allocations)} asignaciones activas")
                
        except Exception as e:
            self.logger.warning(f"Error cargando estado anterior: {str(e)}")
    
    async def _save_state(self) -> None:
        """
        Guardar el estado actual de la estrategia.
        """
        os.makedirs("checkpoints", exist_ok=True)
        file_path = f"checkpoints/{self.strategy_id}_state.json"
        
        try:
            # Preparar estado actual
            state = {
                "strategy_id": self.strategy_id,
                "timestamp": datetime.now().isoformat(),
                "current_allocations": self.current_allocations,
                "performance_metrics": self.performance_metrics,
                "last_update_time": self.last_update_time
            }
            
            # Guardar en archivo
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
                
            self.logger.debug(f"Estado guardado en {file_path}")
                
        except Exception as e:
            self.logger.error(f"Error guardando estado: {str(e)}")
    
    async def update_symbols(self, symbols: List[str]) -> None:
        """
        Actualizar la lista de símbolos a operar.
        
        Args:
            symbols: Nueva lista de símbolos
        """
        self.symbols = symbols
        self.logger.info(f"Lista de símbolos actualizada: {len(self.symbols)} instrumentos")
    
    async def optimize_allocation(self, total_capital: float) -> Dict[str, float]:
        """
        Optimizar la asignación de capital entre instrumentos.
        
        Args:
            total_capital: Capital total disponible
            
        Returns:
            Diccionario con asignaciones por símbolo
        """
        if not self.engine or not self.symbols or total_capital <= 0:
            return {}
        
        self.logger.info(f"Optimizando asignación para capital total: ${total_capital:,.2f}")
        
        # Si estamos usando risk manager, aplicar restricciones
        position_limits = {}
        if self.risk_manager:
            for symbol in self.symbols:
                max_position = await self.risk_manager.get_max_position_size(symbol, total_capital)
                position_limits[symbol] = max_position
        
        # Ejecutar optimización con el motor predictivo
        allocations = await self.engine.optimize_allocation(
            symbols=self.symbols,
            total_capital=total_capital,
            min_efficiency=self.min_efficiency_threshold
        )
        
        # Aplicar límites de posición si existen
        if position_limits:
            for symbol, amount in list(allocations.items()):
                max_allowed = position_limits.get(symbol, float('inf'))
                if amount > max_allowed:
                    allocations[symbol] = max_allowed
        
        # Calcular métricas para esta asignación
        await self._calculate_allocation_metrics(allocations, total_capital)
        
        # Guardar asignación actual
        self.current_allocations = allocations
        self.last_update_time = int(datetime.now().timestamp())
        
        # Persistir estado
        await self._save_state()
        
        # Devolver resultado
        return allocations
    
    async def _calculate_allocation_metrics(self, allocations: Dict[str, float], total_capital: float) -> None:
        """
        Calcular métricas para una asignación específica.
        
        Args:
            allocations: Asignaciones por símbolo
            total_capital: Capital total
        """
        # Calcular eficiencia esperada por símbolo
        efficiency_by_symbol = {}
        total_efficiency = 0.0
        symbol_count = 0
        
        for symbol, amount in allocations.items():
            if amount > 0:
                prediction = await self.engine.predict_efficiency(symbol, amount)
                efficiency_by_symbol[symbol] = prediction.efficiency
                total_efficiency += prediction.efficiency
                symbol_count += 1
        
        # Calcular eficiencia promedio
        avg_efficiency = safe_divide(total_efficiency, symbol_count)
        
        # Calcular utilización de capital
        capital_utilization = safe_divide(sum(allocations.values()), total_capital)
        
        # Calcular entropía (diversificación)
        entropy = 0.0
        if allocations:
            percentages = [amount / total_capital for amount in allocations.values() if amount > 0]
            for p in percentages:
                if p > 0:
                    entropy -= p * np.log(p)
            # Normalizar por el máximo posible (log(n))
            max_entropy = np.log(len(percentages)) if len(percentages) > 0 else 1.0
            entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Actualizar métricas
        self.performance_metrics["efficiency_by_symbol"] = efficiency_by_symbol
        self.performance_metrics["avg_efficiency"] = avg_efficiency
        self.performance_metrics["capital_utilization"] = capital_utilization
        self.performance_metrics["entropy"] = entropy
        
        # Actualizar historial
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "total_capital": total_capital,
            "avg_efficiency": avg_efficiency,
            "capital_utilization": capital_utilization,
            "entropy": entropy,
            "symbols_used": symbol_count
        }
        self.performance_metrics["allocation_history"].append(history_entry)
        
        # Mantener solo los últimos 100 registros
        if len(self.performance_metrics["allocation_history"]) > 100:
            self.performance_metrics["allocation_history"] = self.performance_metrics["allocation_history"][-100:]
    
    async def record_observed_efficiency(
        self, 
        symbol: str, 
        capital: float, 
        efficiency: float,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Registrar eficiencia observada para un instrumento.
        
        Esta información se usa para mejorar los modelos predictivos.
        
        Args:
            symbol: Símbolo del instrumento
            capital: Nivel de capital
            efficiency: Eficiencia observada (0-1)
            metrics: Métricas adicionales (roi, sharpe, etc.)
        """
        if not self.engine:
            return
        
        # Validar datos
        if efficiency < 0 or efficiency > 1:
            self.logger.warning(f"Eficiencia fuera de rango para {symbol}: {efficiency}")
            efficiency = max(0.0, min(1.0, efficiency))
        
        # Alimentar al motor predictivo
        await self.engine.add_efficiency_record(symbol, capital, efficiency, metrics)
        
        # Persistir en base de datos si está disponible
        if self.db:
            try:
                # Preparar datos
                record = {
                    "symbol": symbol,
                    "capital_level": capital,
                    "efficiency": efficiency,
                    "timestamp": datetime.now()
                }
                
                # Añadir métricas si existen
                if metrics:
                    for key, value in metrics.items():
                        record[key] = value
                
                # Guardar en base de datos
                await self.db.execute(
                    """
                    INSERT INTO efficiency_records 
                    (symbol, capital_level, efficiency, timestamp, roi, sharpe, max_drawdown, win_rate)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    record['symbol'],
                    record['capital_level'],
                    record['efficiency'],
                    record['timestamp'],
                    metrics.get('roi') if metrics else None,
                    metrics.get('sharpe') if metrics else None,
                    metrics.get('max_drawdown') if metrics else None,
                    metrics.get('win_rate') if metrics else None
                )
                
                self.logger.debug(f"Eficiencia registrada en BD para {symbol}: {efficiency:.4f} @ ${capital:,.2f}")
                
            except Exception as e:
                self.logger.error(f"Error guardando eficiencia en BD: {str(e)}")
    
    async def update_saturation_points(self) -> Dict[str, Optional[float]]:
        """
        Actualizar los puntos de saturación detectados.
        
        Returns:
            Diccionario con puntos de saturación por símbolo
        """
        if not self.engine:
            return {}
        
        # Obtener saturation points actualizados
        saturation_points = self.engine.get_saturation_points()
        
        # Actualizar métricas
        self.performance_metrics["saturation_points"] = saturation_points
        
        # Persistir en base de datos si está disponible
        if self.db:
            try:
                for symbol, value in saturation_points.items():
                    await self.db.execute(
                        """
                        INSERT INTO saturation_points (symbol, saturation_value, determination_method, confidence)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (symbol) DO UPDATE 
                        SET saturation_value = $2, determination_method = $3, confidence = $4, last_update = NOW()
                        """,
                        symbol, value, "model", 0.85
                    )
                
                self.logger.info(f"Puntos de saturación actualizados en BD: {len(saturation_points)} símbolos")
                
            except Exception as e:
                self.logger.error(f"Error actualizando saturation points en BD: {str(e)}")
        
        return saturation_points
    
    async def needs_rebalance(self, current_allocations: Dict[str, float], total_capital: float) -> bool:
        """
        Determinar si es necesario rebalancear la cartera.
        
        Args:
            current_allocations: Asignaciones actuales
            total_capital: Capital total actual
            
        Returns:
            True si se necesita rebalanceo
        """
        # Si nunca hemos actualizado, necesitamos rebalancear
        if not self.current_allocations or not self.last_update_time:
            return True
        
        # Si ha pasado suficiente tiempo desde la última actualización
        current_time = int(datetime.now().timestamp())
        if current_time - self.last_update_time >= self.update_interval:
            return True
        
        # Si el capital total ha cambiado significativamente
        total_allocated = sum(self.current_allocations.values())
        if abs(total_allocated - total_capital) / total_capital > self.rebalance_threshold:
            return True
        
        # Si la asignación actual difiere significativamente de la óptima
        # (esto requeriría recalcular la asignación óptima, lo cual es costoso)
        # Por ahora nos basamos en los criterios anteriores
        
        return False
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas de rendimiento de la estrategia.
        
        Returns:
            Diccionario con métricas
        """
        # Actualizar saturation points
        await self.update_saturation_points()
        
        # Añadir estadísticas del motor
        if self.engine:
            self.performance_metrics["engine_stats"] = self.engine.get_stats()
        
        return self.performance_metrics
    
    async def execute_cycle(self) -> Dict[str, Any]:
        """
        Ejecutar ciclo principal de la estrategia.
        
        Returns:
            Resultados del ciclo
        """
        if not self.initialized or not self.engine:
            await self.initialize()
        
        # Obtener capital actual del balance manager o valor por defecto
        total_capital = 10000.0  # Valor por defecto
        if self.scaling_manager:
            total_capital = await self.scaling_manager.get_current_capital()
        
        # Verificar si necesitamos rebalancear
        current_allocations = {}  # Obtener de alguna fuente
        needs_rebalance = await self.needs_rebalance(current_allocations, total_capital)
        
        if needs_rebalance:
            # Optimizar asignación
            allocations = await self.optimize_allocation(total_capital)
            
            # Aquí iría el código para ajustar posiciones según las nuevas asignaciones
            # Por ahora, solo registramos el resultado
            
            result = {
                "action": "rebalance",
                "total_capital": total_capital,
                "new_allocations": allocations,
                "metrics": self.performance_metrics
            }
        else:
            result = {
                "action": "hold",
                "total_capital": total_capital,
                "current_allocations": self.current_allocations,
                "metrics": self.performance_metrics
            }
        
        return result
    
    async def on_trade_completed(
        self, 
        symbol: str, 
        trade_data: Dict[str, Any]
    ) -> None:
        """
        Manejar evento de operación completada.
        
        Este método registra la eficiencia observada después de cada operación.
        
        Args:
            symbol: Símbolo operado
            trade_data: Datos de la operación
        """
        try:
            # Extraer datos relevantes de la operación
            capital = trade_data.get('position_size', 0.0)
            roi = trade_data.get('roi', 0.0)
            
            # Si no hay datos suficientes, salir
            if not symbol or capital <= 0:
                return
                
            # Calcular eficiencia observada
            # Aquí usamos una fórmula simplificada basada en ROI
            # En un sistema real, tendríamos una métrica más sofisticada
            efficiency = min(1.0, max(0.0, (roi + 0.1) / 0.2))
            
            # Crear métricas adicionales
            metrics = {
                "roi": roi,
                "trade_count": 1,
                "win": roi > 0,
                "execution_time": trade_data.get('execution_time', 0.0),
                "slippage": trade_data.get('slippage', 0.0)
            }
            
            # Registrar eficiencia observada
            await self.record_observed_efficiency(symbol, capital, efficiency, metrics)
            
            self.logger.debug(f"Eficiencia registrada para {symbol}: {efficiency:.4f} (ROI: {roi:.2%})")
            
        except Exception as e:
            self.logger.error(f"Error en on_trade_completed: {str(e)}")
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado actual de la estrategia.
        
        Returns:
            Diccionario con estado
        """
        # Obtener saturation points actualizados
        saturation_points = await self.update_saturation_points()
        
        status = {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "symbols_count": len(self.symbols),
            "initialized": self.initialized,
            "last_update": format_timestamp(self.last_update_time),
            "current_allocations": self.current_allocations,
            "saturation_points": saturation_points,
            "metrics": self.performance_metrics
        }
        
        return status
    
    async def shutdown(self) -> None:
        """
        Realizar limpieza al apagar la estrategia.
        """
        # Guardar estado actual
        await self._save_state()
        
        self.logger.info(f"Estrategia {self.name} apagada correctamente")