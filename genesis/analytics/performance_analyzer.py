"""
Analizador de rendimiento para el sistema Genesis.

Este módulo proporciona funcionalidades para analizar el rendimiento
de estrategias y generar gráficos de manera eficiente.
"""

import json
import logging
import os
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

from genesis.core.base import Component
from genesis.utils.logger import setup_logging

class PerformanceAnalyzer(Component):
    """
    Analiza el rendimiento de estrategias y genera gráficos de rendimiento de manera eficiente.
    """
    
    def __init__(
        self, 
        historial_path: str = "data/performance.json", 
        max_hist: int = 1000,
        name: str = "performance_analyzer"
    ):
        """
        Inicializar el analizador de rendimiento.
        
        Args:
            historial_path: Ruta al archivo de historial
            max_hist: Tamaño máximo del historial
            name: Nombre del componente
        """
        super().__init__(name)
        self.historial_path = historial_path
        self.history: Dict[str, deque] = {}
        self.max_hist = max(100, max_hist)  # Evita valores negativos o muy pequeños
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.logger = setup_logging(name)
        
        # Crear directorio para imágenes si no existe
        self.plots_dir = os.path.join(os.path.dirname(historial_path), "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    async def start(self) -> None:
        """Iniciar el analizador de rendimiento."""
        await super().start()
        await self._load_history()
        self.logger.info("Analizador de rendimiento iniciado")

    async def stop(self) -> None:
        """Detener el analizador de rendimiento."""
        await self._save_history()
        self.executor.shutdown(wait=True)
        await super().stop()
        self.logger.info("Analizador de rendimiento detenido")

    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente de origen
        """
        if event_type == "strategy.evaluation":
            strategy_name = data.get("strategy")
            score = data.get("score")
            
            if strategy_name and score is not None:
                await self.registrar_resultado(strategy_name, score)
                
        elif event_type == "system.request_best_strategy":
            mejor = self.estrategia_recomendada()
            await self.emit_event("system.best_strategy", {
                "strategy": mejor,
                "score": self.calcular_promedio(mejor) if mejor else None
            })

    async def _load_history(self) -> None:
        """Carga el historial desde un archivo."""
        try:
            if os.path.exists(self.historial_path):
                with open(self.historial_path, "r") as f:
                    data = json.load(f)
                
                self.history = {
                    k: deque(v, maxlen=self.max_hist)
                    for k, v in data.get("history", {}).items()
                }
                self.logger.info("Historial de rendimiento cargado correctamente")
            else:
                self.logger.info("No se encontró historial previo")
        except (json.JSONDecodeError, IOError) as e:
            self.logger.warning(f"No se pudo cargar historial: {type(e).__name__} - {e}")

    async def _save_history(self) -> None:
        """Guarda el historial en un archivo."""
        try:
            os.makedirs(os.path.dirname(self.historial_path), exist_ok=True)
            with open(self.historial_path, "w") as f:
                json.dump(
                    {"history": {k: list(v) for k, v in self.history.items()}},
                    f,
                    indent=2
                )
            self.logger.debug("Historial guardado correctamente")
        except (IOError, TypeError) as e:
            self.logger.error(f"Error al guardar historial: {type(e).__name__} - {e}")

    async def registrar_resultado(self, strategy_name: str, score: float) -> None:
        """
        Registra un resultado en el historial.
        
        Args:
            strategy_name: Nombre de la estrategia
            score: Puntuación obtenida
        """
        if not isinstance(strategy_name, str) or not strategy_name.strip():
            raise ValueError("El nombre de la estrategia debe ser una cadena no vacía")
        if not isinstance(score, (int, float)) or not (-float("inf") < score < float("inf")):
            raise ValueError("El score debe ser un número finito")

        strategy_name = strategy_name.strip()
        if strategy_name not in self.history:
            self.history[strategy_name] = deque(maxlen=self.max_hist)
            
        self.history[strategy_name].append({
            "timestamp": datetime.utcnow().isoformat(),
            "score": float(score)
        })
        
        # Guardar periódicamente
        if len(self.history[strategy_name]) % 10 == 0:
            await self._save_history()
            
        # Emitir evento
        await self.emit_event("strategy.score_recorded", {
            "strategy": strategy_name,
            "score": score,
            "avg_score": self.calcular_promedio(strategy_name)
        })

    def calcular_promedio(self, strategy_name: str) -> Optional[float]:
        """
        Calcula el promedio de puntuaciones de una estrategia.
        
        Args:
            strategy_name: Nombre de la estrategia
            
        Returns:
            Promedio de puntuación o None si no hay datos
        """
        hist = self.history.get(strategy_name)
        if not hist:
            return None
            
        # Usar numpy para mayor eficiencia
        return float(np.mean([entry["score"] for entry in hist]))

    def estrategia_recomendada(self) -> Optional[str]:
        """
        Devuelve la estrategia con el mejor promedio.
        
        Returns:
            Nombre de la mejor estrategia o None si no hay datos
        """
        if not self.history:
            return None
        
        # Calcular promedios en una sola pasada
        promedios = {name: self.calcular_promedio(name) for name in self.history}
        validos = {k: v for k, v in promedios.items() if v is not None}
        
        if not validos:
            return None
        
        mejor = max(validos, key=validos.get)
        self.logger.info(f"Estrategia recomendada: {mejor} ({validos[mejor]:.4f})")
        return mejor

    async def generar_grafico(
        self, 
        strategy_name: str, 
        output_path: Optional[str] = None,
        window: Optional[int] = None
    ) -> Optional[str]:
        """
        Genera un gráfico del rendimiento de una estrategia de forma asíncrona.
        
        Args:
            strategy_name: Nombre de la estrategia
            output_path: Ruta donde guardar el gráfico (opcional)
            window: Ventana de datos a mostrar (opcional)
            
        Returns:
            Ruta del archivo generado o None si hubo error
        """
        hist = self.history.get(strategy_name)
        if not hist:
            self.logger.warning(f"No hay historial para {strategy_name}")
            return None

        # Usar un executor para no bloquear el event loop
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                self.executor, 
                self._generar_grafico_sync, 
                strategy_name, 
                hist, 
                output_path,
                window
            )
        except Exception as e:
            self.logger.error(f"Error generando gráfico: {e}")
            return None

    def _generar_grafico_sync(
        self, 
        strategy_name: str, 
        hist: deque, 
        output_path: Optional[str] = None,
        window: Optional[int] = None
    ) -> Optional[str]:
        """
        Genera un gráfico del rendimiento de forma síncrona.
        
        Args:
            strategy_name: Nombre de la estrategia
            hist: Historial de datos
            output_path: Ruta donde guardar el gráfico
            window: Ventana de datos a mostrar
            
        Returns:
            Ruta del archivo generado o None si hubo error
        """
        try:
            # Limitar ventana si se especifica
            data_to_plot = list(hist)
            if window and window < len(data_to_plot):
                data_to_plot = data_to_plot[-window:]
                
            fechas = [datetime.fromisoformat(d["timestamp"]) for d in data_to_plot]
            scores = [d["score"] for d in data_to_plot]

            plt.figure(figsize=(10, 6))
            plt.plot(fechas, scores, marker="o", linestyle="-", color="b", label=strategy_name)
            
            # Añadir línea de tendencia
            if len(scores) > 1:
                z = np.polyfit(range(len(scores)), scores, 1)
                p = np.poly1d(z)
                plt.plot(fechas, p(range(len(scores))), "r--", label="Tendencia")
            
            plt.title(f"Rendimiento de {strategy_name}", fontsize=14, pad=10)
            plt.xlabel("Fecha", fontsize=12)
            plt.ylabel("Score", fontsize=12)
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()

            # Añadir información estadística
            avg_score = np.mean(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            std_score = np.std(scores)
            
            plt.figtext(
                0.02, 0.02,
                f"Promedio: {avg_score:.4f} | Min: {min_score:.4f} | Max: {max_score:.4f} | Std: {std_score:.4f}",
                ha="left", fontsize=10, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5}
            )

            # Guardar el gráfico
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                output_path = os.path.join(self.plots_dir, f"{strategy_name}_{timestamp}.png")
                
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()  # Liberar memoria

            return output_path
        except Exception as e:
            self.logger.error(f"Error en generación de gráfico: {e}")
            return None

    def obtener_historial(self, strategy_name: str, limit: int = 0) -> List[Dict[str, Any]]:
        """
        Devuelve el historial de una estrategia.
        
        Args:
            strategy_name: Nombre de la estrategia
            limit: Límite de resultados (0 para todos)
            
        Returns:
            Lista de resultados históricos
        """
        hist = list(self.history.get(strategy_name, []))
        if limit > 0 and limit < len(hist):
            return hist[-limit:]
        return hist

    async def generar_reporte_comparativo(self, strategies: List[str]) -> Optional[str]:
        """
        Genera un gráfico comparativo de varias estrategias.
        
        Args:
            strategies: Lista de nombres de estrategias
            
        Returns:
            Ruta del archivo generado o None si hubo error
        """
        if not strategies:
            return None
            
        # Filtrar estrategias sin datos
        valid_strategies = [s for s in strategies if s in self.history and self.history[s]]
        if not valid_strategies:
            self.logger.warning("No hay estrategias válidas para comparar")
            return None
            
        # Usar executor para no bloquear
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                self.executor,
                self._generar_comparativo_sync,
                valid_strategies
            )
        except Exception as e:
            self.logger.error(f"Error generando comparativo: {e}")
            return None
            
    def _generar_comparativo_sync(self, strategies: List[str]) -> Optional[str]:
        """
        Genera un gráfico comparativo de forma síncrona.
        
        Args:
            strategies: Lista de nombres de estrategias
            
        Returns:
            Ruta del archivo generado o None si hubo error
        """
        try:
            plt.figure(figsize=(12, 8))
            
            # Colores para cada estrategia
            colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
            
            # Graficar cada estrategia
            for i, strategy in enumerate(strategies):
                hist = list(self.history[strategy])
                
                # Preparar datos
                fechas = [datetime.fromisoformat(d["timestamp"]) for d in hist]
                scores = [d["score"] for d in hist]
                
                # Dibujar línea
                plt.plot(fechas, scores, linestyle="-", marker=".", color=colors[i], label=strategy)
                
                # Calcular y mostrar promedio
                avg = np.mean(scores)
                plt.axhline(y=avg, color=colors[i], linestyle="--", alpha=0.5)
            
            plt.title("Comparativa de Estrategias", fontsize=14, pad=10)
            plt.xlabel("Fecha", fontsize=12)
            plt.ylabel("Score", fontsize=12)
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            
            # Añadir datos estadísticos
            stats_text = "Promedios:\n"
            for i, strategy in enumerate(strategies):
                scores = [d["score"] for d in self.history[strategy]]
                avg = np.mean(scores)
                stats_text += f"{strategy}: {avg:.4f}\n"
                
            plt.figtext(
                0.02, 0.02,
                stats_text,
                ha="left", fontsize=10, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5}
            )
            
            # Guardar gráfico
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_path = os.path.join(self.plots_dir, f"comparativa_{timestamp}.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            return output_path
        except Exception as e:
            self.logger.error(f"Error en generación de comparativo: {e}")
            return None