"""
Orquestador de estrategias para el sistema Genesis.

Este módulo proporciona una gestión inteligente de estrategias de trading,
permitiendo seleccionar dinámicamente la mejor estrategia según las
condiciones del mercado y el rendimiento histórico.
"""

import logging
import time
import asyncio
import random
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from genesis.core.base import Component
from genesis.utils.logger import setup_logging


class StrategyOrchestrator(Component):
    """
    Orquestador de estrategias de trading.
    
    Este componente gestiona y coordina múltiples estrategias, seleccionando
    dinámicamente la mejor basada en su rendimiento y las condiciones del
    mercado actuales.
    """
    
    def __init__(self, name: str = "strategy_orchestrator"):
        """
        Inicializar el orquestador de estrategias.
        
        Args:
            name: Nombre del componente
        """
        super().__init__(name)
        self.logger = setup_logging(name)
        
        # Estrategias registradas (nombre -> instancia)
        self.strategies = {}
        
        # Configuración de estrategias
        self.strategy_configs = {}
        
        # Estrategia activa actual
        self.active_strategy_name = None
        
        # Rendimiento de las estrategias
        self.performance_scores = {}
        
        # Historial de señales
        self.history = deque(maxlen=1000)
        
        # Executor para tareas en paralelo
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Configuración
        self.min_performance_threshold = 0.4
        self.eval_cooldown = 60  # Cooldown en segundos para reevaluaciones
        self.last_eval_time = 0
        self.eval_failures = 0
        self.max_eval_failures = 5  # Circuit breaker
    
    async def start(self) -> None:
        """Iniciar el orquestador de estrategias."""
        await super().start()
        self.logger.info("Orquestador de estrategias iniciado")
    
    async def stop(self) -> None:
        """Detener el orquestador de estrategias."""
        await super().stop()
        self.logger.info("Orquestador de estrategias detenido")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente de origen
        """
        # Registrar rendimiento de estrategia
        if event_type == "strategy.performance_update":
            strategy_name = data.get("strategy_name")
            score = data.get("score")
            
            if strategy_name and score is not None:
                self.performance_scores[strategy_name] = score
                self.logger.info(f"Rendimiento actualizado para {strategy_name}: {score:.4f}")
        
        # Solicitud de señal
        elif event_type == "strategy.request_signal":
            symbol = data.get("symbol")
            
            if symbol:
                signal = await self.get_signal(symbol)
                
                await self.emit_event("strategy.signal", {
                    "symbol": symbol,
                    "signal": signal,
                    "strategy": self.active_strategy_name,
                    "timestamp": time.time()
                })
    
    def register_strategy(self, name: str, strategy, config: Dict[str, Any]) -> None:
        """
        Registrar una estrategia en el orquestador.
        
        Args:
            name: Nombre de la estrategia
            strategy: Instancia de la estrategia
            config: Configuración de la estrategia
        """
        self.strategies[name] = strategy
        self.strategy_configs[name] = config
        
        # Si es la primera estrategia, establecerla como activa
        if self.active_strategy_name is None:
            self.active_strategy_name = name
            
        self.logger.info(f"Estrategia registrada: {name}")
    
    def unregister_strategy(self, name: str) -> bool:
        """
        Eliminar una estrategia del orquestador.
        
        Args:
            name: Nombre de la estrategia
            
        Returns:
            True si se eliminó correctamente, False en caso contrario
        """
        if name in self.strategies:
            del self.strategies[name]
            
            if name in self.strategy_configs:
                del self.strategy_configs[name]
                
            if name in self.performance_scores:
                del self.performance_scores[name]
                
            # Si era la estrategia activa, seleccionar otra
            if self.active_strategy_name == name:
                if self.strategies:
                    self.active_strategy_name = next(iter(self.strategies.keys()))
                else:
                    self.active_strategy_name = None
                    
            self.logger.info(f"Estrategia eliminada: {name}")
            return True
            
        return False
    
    async def evaluate_strategies(self, symbol: str) -> str:
        """
        Evaluar todas las estrategias y seleccionar la mejor.
        
        Args:
            symbol: Símbolo de trading
            
        Returns:
            Nombre de la mejor estrategia
        """
        # Verificar cooldown
        if time.time() - self.last_eval_time < self.eval_cooldown:
            self.logger.info("Evaluación en cooldown. Usando estrategia activa.")
            return self.active_strategy_name
        
        # Verificar circuit breaker
        if self.eval_failures >= self.max_eval_failures:
            self.logger.critical("Demasiados fallos en evaluación. Manteniendo estrategia activa.")
            return self.active_strategy_name
        
        # Verificar si hay estrategias
        if not self.strategies:
            self.logger.warning("No hay estrategias registradas.")
            return None
        
        scores = {}
        tasks = []
        
        for name, strategy in self.strategies.items():
            tasks.append(self._simulate_strategy(name, strategy, symbol))
        
        try:
            results = await asyncio.gather(*tasks)
            for name, result in results:
                if result is not None:
                    scores[name] = result.get("performance_score", 0)
                    self.performance_scores[name] = scores[name]
        except Exception as e:
            self.eval_failures += 1
            self.logger.error(f"Fallo en evaluación de estrategias: {e}")
            return self.active_strategy_name
        
        if not scores:
            self.eval_failures += 1
            return self.active_strategy_name
        
        best_strategy = max(scores, key=scores.get)
        self.logger.info(f"Estrategia más efectiva: {best_strategy} (score: {scores[best_strategy]:.4f})")
        
        self.last_eval_time = time.time()
        self.eval_failures = 0  # Resetear fallos tras éxito
        
        return best_strategy
    
    async def _simulate_strategy(self, name: str, strategy, symbol: str):
        """
        Simular una estrategia en un hilo separado.
        
        Args:
            name: Nombre de la estrategia
            strategy: Instancia de la estrategia
            symbol: Símbolo de trading
            
        Returns:
            Tupla (nombre, resultado)
        """
        loop = asyncio.get_event_loop()
        
        try:
            # Obtener datos de mercado a través del bus de eventos
            market_data = None
            
            # Solicitar datos de mercado
            await self.emit_event("market.data_request", {
                "symbol": symbol,
                "requestor": self.name
            })
            
            # Esperar respuesta (simulado por ahora)
            await asyncio.sleep(0.5)
            
            # En un sistema real, aquí procesaríamos los datos recibidos
            
            # Generar señal con la estrategia
            signal_result = await loop.run_in_executor(
                self.executor, 
                lambda: {"performance_score": random.uniform(0, 1)}  # Simulación
            )
            
            return name, signal_result
        except Exception as e:
            self.logger.error(f"Error simulando {name}: {e}")
            return name, None
    
    async def get_signal(self, symbol: str) -> str:
        """
        Obtener señal de la mejor estrategia para un símbolo.
        
        Args:
            symbol: Símbolo de trading
            
        Returns:
            Señal de trading (buy, sell, hold)
        """
        try:
            # Verificar si hay una estrategia activa
            if not self.active_strategy_name or self.active_strategy_name not in self.strategies:
                if self.strategies:
                    self.active_strategy_name = await self.evaluate_strategies(symbol)
                else:
                    self.logger.warning("No hay estrategias disponibles.")
                    return "hold"
            
            # Re-evaluar si el rendimiento es bajo
            current_perf = self.performance_scores.get(self.active_strategy_name)
            if current_perf is not None and current_perf < self.min_performance_threshold:
                self.logger.warning(f"Bajo rendimiento de {self.active_strategy_name} ({current_perf:.2f}). Reevaluando...")
                self.active_strategy_name = await self.evaluate_strategies(symbol)
            
            # Obtener estrategia activa
            strategy = self.strategies[self.active_strategy_name]
            
            # Obtener datos de mercado
            # En un sistema real, aquí obtendríamos datos reales
            
            # Generar señal
            signal = "hold"  # Valor por defecto
            
            # Invocar método generate_signal
            if hasattr(strategy, 'generate_signal'):
                # Convertir datos a formato DataFrame si es necesario
                data = pd.DataFrame()  # Simulado
                signal_result = await strategy.generate_signal(symbol, data)
                
                if signal_result and isinstance(signal_result, dict):
                    signal = signal_result.get("signal", "hold")
            
            self.logger.info(f"Señal generada por {self.active_strategy_name}: {signal}")
            
            # Guardar en historial
            self.history.append((time.time(), self.active_strategy_name, symbol, signal))
            
            return signal
        except Exception as e:
            self.logger.error(f"Error en generación de señal: {e}")
            return "hold"
    
    def force_change_strategy(self, new_strategy_name: str) -> bool:
        """
        Forzar un cambio de estrategia activa.
        
        Args:
            new_strategy_name: Nombre de la nueva estrategia activa
            
        Returns:
            True si se cambió correctamente, False en caso contrario
        """
        if new_strategy_name in self.strategies:
            self.active_strategy_name = new_strategy_name
            self.logger.info(f"Estrategia cambiada manualmente a: {new_strategy_name}")
            return True
        else:
            self.logger.warning(f"Estrategia {new_strategy_name} no está registrada.")
            return False
    
    def get_active_strategy(self) -> Optional[str]:
        """
        Obtener el nombre de la estrategia activa.
        
        Returns:
            Nombre de la estrategia activa o None si no hay
        """
        return self.active_strategy_name
    
    def get_strategy_history(self) -> List[tuple]:
        """
        Obtener el historial de señales generadas.
        
        Returns:
            Lista de tuplas (timestamp, estrategia, símbolo, señal)
        """
        return list(self.history)
    
    async def stress_test(self, symbols: List[str], num_iterations: int) -> Dict[str, int]:
        """
        Ejecutar prueba de estrés del orquestador.
        
        Args:
            symbols: Lista de símbolos
            num_iterations: Número de iteraciones
            
        Returns:
            Diccionario con conteo de señales
        """
        tasks = []
        
        for _ in range(num_iterations):
            for symbol in symbols:
                tasks.append(self.get_signal(symbol))
        
        results = await asyncio.gather(*tasks)
        
        # Contar resultados
        signals = {"buy": 0, "sell": 0, "hold": 0}
        
        for signal in results:
            if signal in signals:
                signals[signal] += 1
            else:
                signals[signal] = 1
                
        return signals