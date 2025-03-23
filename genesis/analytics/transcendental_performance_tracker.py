"""
Rastreador de Rendimiento Transcendental para el Sistema Genesis.

Este módulo implementa un sistema avanzado de seguimiento de rendimiento con 
capacidades transcendentales que monitorea y analiza todas las dimensiones
del desempeño del sistema, incluyendo métricas financieras avanzadas, 
calidad de ejecución y eficiencia operativa.

Características principales:
- Análisis multidimensional de rendimiento adaptativo
- Métricas avanzadas de calidad de ejecución y costos implícitos
- Benchmarking contra diversos índices y estrategias de referencia
- Análisis de atribución de rendimiento para identificar fuentes de alpha
- Detección avanzada de desviaciones y anomalías en el rendimiento
"""

import logging
import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
import json
from decimal import Decimal

from genesis.db.transcendental_database import db

# Configuración de logging
logger = logging.getLogger(__name__)

class TranscendentalPerformanceTracker:
    """
    Rastreador de rendimiento avanzado con capacidades transcendentales.
    
    Este sistema monitorea y analiza el rendimiento del sistema Genesis en
    múltiples dimensiones, proporcionando insights avanzados para la
    optimización continua de la estrategia y la adaptación al crecimiento
    del capital.
    """
    
    def __init__(self, capital_inicial: float = 10000.0):
        """
        Inicializar el rastreador de rendimiento.
        
        Args:
            capital_inicial: Capital inicial del sistema en USD
        """
        self.capital_inicial = capital_inicial
        self.fecha_inicio = datetime.now()
        
        # Historial de capital
        self.historial_capital = [{
            "timestamp": self.fecha_inicio.timestamp(),
            "capital": capital_inicial,
            "cambio_diario": 0.0,
            "cambio_porcentual": 0.0
        }]
        
        # Historial de operaciones
        self.operaciones = []
        
        # Métricas de rendimiento
        self.metricas = {
            "capital_actual": capital_inicial,
            "rendimiento_total": 0.0,
            "rendimiento_anualizado": 0.0,
            "volatilidad": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "omega_ratio": 0.0,
            "valor_en_riesgo_95": 0.0,
            "valor_en_riesgo_99": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectativa_matematica": 0.0,
            "factor_recuperacion": 0.0,
            "correlacion_mercado": 0.0
        }
        
        # Métricas de ejecución
        self.metricas_ejecucion = {
            "slippage_promedio": 0.0,
            "latencia_promedio": 0.0,
            "costo_oportunidad": 0.0,
            "market_impact": 0.0,
            "desviacion_vwap": 0.0,
            "eficiencia_ejecucion": 1.0,
            "fill_ratio": 1.0,
            "tiempo_ejecucion_promedio": 0.0
        }
        
        # Análisis por activo
        self.rendimiento_por_activo = {}
        
        # Análisis por estrategia
        self.rendimiento_por_estrategia = {}
        
        # Benchmarks y referencias
        self.benchmarks = {
            "crypto_top10": [],  # [timestamp, valor]
            "bitcoin": [],       # [timestamp, valor]
            "sp500": []          # [timestamp, valor]
        }
        
        # Anomalías detectadas
        self.anomalias = []
        
        logger.info(f"TranscendentalPerformanceTracker inicializado con capital: ${capital_inicial:,.2f}")
    
    async def actualizar_capital(self, 
                              nuevo_capital: float, 
                              fuente: str = "general") -> Dict[str, Any]:
        """
        Actualizar capital actual y recalcular métricas de rendimiento.
        
        Args:
            nuevo_capital: Nuevo monto de capital en USD
            fuente: Fuente de la actualización (general, operacion, ajuste, etc.)
            
        Returns:
            Diccionario con métricas actualizadas
        """
        now = datetime.now()
        timestamp = now.timestamp()
        capital_anterior = self.metricas["capital_actual"]
        
        # Calcular cambios
        cambio_absoluto = nuevo_capital - capital_anterior
        cambio_porcentual = (nuevo_capital / capital_anterior) - 1 if capital_anterior > 0 else 0
        
        # Actualizar historial
        self.historial_capital.append({
            "timestamp": timestamp,
            "capital": nuevo_capital,
            "cambio_diario": cambio_absoluto,
            "cambio_porcentual": cambio_porcentual,
            "fuente": fuente
        })
        
        # Actualizar capital actual
        self.metricas["capital_actual"] = nuevo_capital
        
        # Recalcular todas las métricas
        await self._calcular_metricas_rendimiento()
        
        # Preparar resultado
        resultado = {
            "capital_anterior": capital_anterior,
            "capital_nuevo": nuevo_capital,
            "cambio_absoluto": cambio_absoluto,
            "cambio_porcentual": cambio_porcentual * 100,
            "timestamp": timestamp,
            "metricas_actualizadas": {
                "rendimiento_total": self.metricas["rendimiento_total"] * 100,
                "volatilidad": self.metricas["volatilidad"] * 100,
                "max_drawdown": self.metricas["max_drawdown"] * 100,
                "sharpe_ratio": self.metricas["sharpe_ratio"],
                "sortino_ratio": self.metricas["sortino_ratio"],
                "calmar_ratio": self.metricas["calmar_ratio"]
            }
        }
        
        return resultado
    
    async def registrar_operacion(self, operacion: Dict[str, Any]) -> Dict[str, Any]:
        """
        Registrar una operación completada y actualizar métricas.
        
        Args:
            operacion: Diccionario con detalles de la operación
            
        Returns:
            Diccionario con análisis de la operación
        """
        # Validar operación
        campos_requeridos = ["symbol", "tipo", "entrada_precio", "salida_precio", 
                            "unidades", "resultado_usd"]
        for campo in campos_requeridos:
            if campo not in operacion:
                raise ValueError(f"Falta el campo requerido '{campo}' en la operación")
        
        # Añadir timestamp y campos calculados
        now = datetime.now()
        operacion["timestamp"] = now.timestamp()
        operacion["fecha"] = now.isoformat()
        
        if "resultado_porcentual" not in operacion:
            precio_entrada = operacion["entrada_precio"]
            precio_salida = operacion["salida_precio"]
            
            if operacion["tipo"].upper() == "LONG":
                operacion["resultado_porcentual"] = (precio_salida / precio_entrada) - 1
            else:  # SHORT
                operacion["resultado_porcentual"] = (precio_entrada / precio_salida) - 1
        
        operacion["ganadora"] = operacion["resultado_usd"] > 0
        
        # Añadir duración si se proporcionan timestamps
        if "entrada_timestamp" in operacion and "salida_timestamp" in operacion:
            operacion["duracion_segundos"] = operacion["salida_timestamp"] - operacion["entrada_timestamp"]
        
        # Añadir al registro de operaciones
        self.operaciones.append(operacion)
        
        # Actualizar análisis por activo
        symbol = operacion["symbol"]
        if symbol not in self.rendimiento_por_activo:
            self.rendimiento_por_activo[symbol] = {
                "operaciones_total": 0,
                "operaciones_ganadas": 0,
                "operaciones_perdidas": 0,
                "ganancia_total": 0.0,
                "perdida_total": 0.0,
                "resultado_neto": 0.0,
                "mayor_ganancia": 0.0,
                "mayor_perdida": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectativa": 0.0
            }
        
        activo_stats = self.rendimiento_por_activo[symbol]
        activo_stats["operaciones_total"] += 1
        
        if operacion["ganadora"]:
            activo_stats["operaciones_ganadas"] += 1
            activo_stats["ganancia_total"] += operacion["resultado_usd"]
            activo_stats["mayor_ganancia"] = max(activo_stats["mayor_ganancia"], operacion["resultado_usd"])
        else:
            activo_stats["operaciones_perdidas"] += 1
            activo_stats["perdida_total"] += abs(operacion["resultado_usd"])
            activo_stats["mayor_perdida"] = max(activo_stats["mayor_perdida"], abs(operacion["resultado_usd"]))
        
        activo_stats["resultado_neto"] = activo_stats["ganancia_total"] - activo_stats["perdida_total"]
        activo_stats["win_rate"] = activo_stats["operaciones_ganadas"] / activo_stats["operaciones_total"]
        activo_stats["profit_factor"] = activo_stats["ganancia_total"] / max(1, activo_stats["perdida_total"])
        activo_stats["expectativa"] = (
            (activo_stats["ganancia_total"] / max(1, activo_stats["operaciones_ganadas"])) * activo_stats["win_rate"] -
            (activo_stats["perdida_total"] / max(1, activo_stats["operaciones_perdidas"])) * (1 - activo_stats["win_rate"])
        )
        
        # Actualizar análisis por estrategia
        estrategia = operacion.get("estrategia", "default")
        if estrategia not in self.rendimiento_por_estrategia:
            self.rendimiento_por_estrategia[estrategia] = {
                "operaciones_total": 0,
                "operaciones_ganadas": 0,
                "resultado_neto": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectativa": 0.0
            }
        
        # Similar al análisis por activo...
        estr_stats = self.rendimiento_por_estrategia[estrategia]
        estr_stats["operaciones_total"] += 1
        if operacion["ganadora"]:
            estr_stats["operaciones_ganadas"] += 1
        estr_stats["resultado_neto"] += operacion["resultado_usd"]
        estr_stats["win_rate"] = estr_stats["operaciones_ganadas"] / estr_stats["operaciones_total"]
        
        # Actualizar capital si se proporciona
        if "capital_final" in operacion and operacion["capital_final"] > 0:
            await self.actualizar_capital(operacion["capital_final"], fuente="operacion")
        
        # Analizar calidad de ejecución
        analisis_ejecucion = await self._analizar_calidad_ejecucion(operacion)
        
        # Preparar resultado
        resultado = {
            "operacion_id": len(self.operaciones),
            "symbol": symbol,
            "tipo": operacion["tipo"],
            "resultado_usd": operacion["resultado_usd"],
            "resultado_porcentual": operacion["resultado_porcentual"] * 100,
            "ganadora": operacion["ganadora"],
            "metricas_activo": {
                "win_rate": activo_stats["win_rate"] * 100,
                "profit_factor": activo_stats["profit_factor"],
                "expectativa": activo_stats["expectativa"],
                "operaciones_total": activo_stats["operaciones_total"]
            },
            "analisis_ejecucion": analisis_ejecucion
        }
        
        return resultado
    
    async def analizar_periodo(self, 
                            desde: Optional[datetime] = None, 
                            hasta: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Analizar rendimiento durante un período específico.
        
        Args:
            desde: Fecha de inicio para análisis (None = desde el principio)
            hasta: Fecha de fin para análisis (None = hasta ahora)
            
        Returns:
            Diccionario con análisis detallado del período
        """
        if not desde:
            desde = datetime.fromtimestamp(self.historial_capital[0]["timestamp"])
        
        if not hasta:
            hasta = datetime.now()
        
        desde_ts = desde.timestamp()
        hasta_ts = hasta.timestamp()
        
        # Filtrar historial de capital para el período
        capital_periodo = [
            entry for entry in self.historial_capital 
            if desde_ts <= entry["timestamp"] <= hasta_ts
        ]
        
        if not capital_periodo:
            return {"error": "No hay datos para el período especificado"}
        
        # Filtrar operaciones para el período
        operaciones_periodo = [
            op for op in self.operaciones 
            if desde_ts <= op["timestamp"] <= hasta_ts
        ]
        
        # Calcular métricas para el período
        capital_inicial_periodo = capital_periodo[0]["capital"]
        capital_final_periodo = capital_periodo[-1]["capital"]
        rendimiento_total = (capital_final_periodo / capital_inicial_periodo) - 1
        
        # Calcular rendimiento diario
        rendimientos_diarios = []
        for i in range(1, len(capital_periodo)):
            rendimiento = (capital_periodo[i]["capital"] / capital_periodo[i-1]["capital"]) - 1
            rendimientos_diarios.append(rendimiento)
        
        # Volatilidad
        volatilidad = np.std(rendimientos_diarios) if rendimientos_diarios else 0
        
        # Calcular drawdown
        drawdown_max = 0
        peak = capital_inicial_periodo
        for entry in capital_periodo:
            if entry["capital"] > peak:
                peak = entry["capital"]
            drawdown_actual = 1 - (entry["capital"] / peak)
            drawdown_max = max(drawdown_max, drawdown_actual)
        
        # Análisis de operaciones
        ops_ganadoras = [op for op in operaciones_periodo if op["ganadora"]]
        ops_perdedoras = [op for op in operaciones_periodo if not op["ganadora"]]
        
        win_rate = len(ops_ganadoras) / len(operaciones_periodo) if operaciones_periodo else 0
        
        ganancia_total = sum(op["resultado_usd"] for op in ops_ganadoras)
        perdida_total = sum(abs(op["resultado_usd"]) for op in ops_perdedoras)
        profit_factor = ganancia_total / perdida_total if perdida_total > 0 else float('inf')
        
        # Métricas avanzadas
        duracion_años = (hasta_ts - desde_ts) / (365.25 * 24 * 3600)
        rendimiento_anualizado = (1 + rendimiento_total) ** (1 / max(duracion_años, 0.01)) - 1
        
        tasa_libre_riesgo_anual = 0.02  # 2% anual
        tasa_libre_riesgo_periodo = (1 + tasa_libre_riesgo_anual) ** duracion_años - 1
        
        sharpe_ratio = ((rendimiento_total - tasa_libre_riesgo_periodo) / 
                        volatilidad) if volatilidad > 0 else 0
        
        # Rendimientos negativos para Sortino
        rendimientos_negativos = [r for r in rendimientos_diarios if r < 0]
        volatilidad_downside = np.std(rendimientos_negativos) if rendimientos_negativos else 0.01
        
        sortino_ratio = ((rendimiento_total - tasa_libre_riesgo_periodo) / 
                         volatilidad_downside) if volatilidad_downside > 0 else 0
        
        calmar_ratio = rendimiento_anualizado / drawdown_max if drawdown_max > 0 else 0
        
        # Preparar resultado
        resultado = {
            "periodo": {
                "desde": desde.isoformat(),
                "hasta": hasta.isoformat(),
                "duracion_dias": (hasta_ts - desde_ts) / (24 * 3600)
            },
            "capital": {
                "inicial": capital_inicial_periodo,
                "final": capital_final_periodo,
                "cambio_absoluto": capital_final_periodo - capital_inicial_periodo,
                "cambio_porcentual": rendimiento_total * 100
            },
            "rendimiento": {
                "total": rendimiento_total * 100,
                "anualizado": rendimiento_anualizado * 100,
                "volatilidad": volatilidad * 100,
                "drawdown_maximo": drawdown_max * 100,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": calmar_ratio
            },
            "operaciones": {
                "total": len(operaciones_periodo),
                "ganadoras": len(ops_ganadoras),
                "perdedoras": len(ops_perdedoras),
                "win_rate": win_rate * 100,
                "profit_factor": profit_factor,
                "ganancia_total": ganancia_total,
                "perdida_total": perdida_total,
                "resultado_neto": ganancia_total - perdida_total
            },
            "activos": {}
        }
        
        # Añadir análisis por activo
        activos_periodo = set(op["symbol"] for op in operaciones_periodo)
        for symbol in activos_periodo:
            ops_activo = [op for op in operaciones_periodo if op["symbol"] == symbol]
            win_ops = [op for op in ops_activo if op["ganadora"]]
            
            resultado["activos"][symbol] = {
                "operaciones": len(ops_activo),
                "win_rate": (len(win_ops) / len(ops_activo)) * 100 if ops_activo else 0,
                "resultado_neto": sum(op["resultado_usd"] for op in ops_activo),
                "mejor_operacion": max([op["resultado_usd"] for op in ops_activo], default=0),
                "peor_operacion": min([op["resultado_usd"] for op in ops_activo], default=0)
            }
        
        return resultado
    
    async def comparar_benchmark(self, benchmark: str = "crypto_top10") -> Dict[str, Any]:
        """
        Comparar rendimiento con un benchmark de referencia.
        
        Args:
            benchmark: Benchmark a comparar (crypto_top10, bitcoin, sp500)
            
        Returns:
            Diccionario con análisis comparativo
        """
        # En un sistema real, obtendríamos datos históricos reales
        # Para esta demo, simulamos un benchmark
        
        # Simular historia del benchmark si no existe
        if not self.benchmarks[benchmark]:
            await self._simular_benchmark(benchmark)
        
        # Verificar que tengamos suficientes datos
        if len(self.benchmarks[benchmark]) < 2:
            return {"error": "Datos insuficientes del benchmark"}
        
        # Alinear períodos para comparación justa
        inicio_sistema = self.historial_capital[0]["timestamp"]
        capital_inicial_sistema = self.historial_capital[0]["capital"]
        
        # Encontrar punto más cercano en el benchmark
        benchmark_data = []
        for timestamp, valor in self.benchmarks[benchmark]:
            if timestamp >= inicio_sistema:
                benchmark_data.append((timestamp, valor))
        
        if not benchmark_data:
            return {"error": "No hay datos del benchmark para el período del sistema"}
        
        valor_inicial_benchmark = benchmark_data[0][1]
        
        # Calcular rendimientos para mismos puntos temporales
        comparacion = []
        for i, (timestamp, valor_benchmark) in enumerate(benchmark_data):
            # Encontrar capital del sistema para el timestamp más cercano
            capital_sistema = self._interpolar_capital(timestamp)
            
            # Calcular rendimientos desde el inicio
            rendimiento_sistema = (capital_sistema / capital_inicial_sistema) - 1
            rendimiento_benchmark = (valor_benchmark / valor_inicial_benchmark) - 1
            
            comparacion.append({
                "timestamp": timestamp,
                "fecha": datetime.fromtimestamp(timestamp).isoformat(),
                "rendimiento_sistema": rendimiento_sistema,
                "rendimiento_benchmark": rendimiento_benchmark,
                "diferencia": rendimiento_sistema - rendimiento_benchmark
            })
        
        # Calcular estadísticas de la comparación
        rendimiento_final_sistema = comparacion[-1]["rendimiento_sistema"]
        rendimiento_final_benchmark = comparacion[-1]["rendimiento_benchmark"]
        
        # Calcular correlación
        if len(comparacion) > 1:
            rend_sistema = [c["rendimiento_sistema"] for c in comparacion]
            rend_benchmark = [c["rendimiento_benchmark"] for c in comparacion]
            
            try:
                correlacion = np.corrcoef(rend_sistema, rend_benchmark)[0, 1]
            except:
                correlacion = 0
        else:
            correlacion = 0
        
        # Calcular beta (sensibilidad al mercado)
        beta = 0
        if len(comparacion) > 1:
            try:
                # Beta = cov(sistema, benchmark) / var(benchmark)
                cov_matrix = np.cov(
                    [c["rendimiento_sistema"] for c in comparacion],
                    [c["rendimiento_benchmark"] for c in comparacion]
                )
                beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 0
            except:
                beta = 0
        
        # Calcular alpha (rendimiento ajustado por riesgo)
        tasa_libre_riesgo = 0.02 / 365  # Diaria
        alpha = 0
        if beta != 0:
            # Alpha = rendimiento_sistema - (tasa_libre_riesgo + beta * (rendimiento_benchmark - tasa_libre_riesgo))
            alpha = (rendimiento_final_sistema - 
                    (tasa_libre_riesgo + beta * (rendimiento_final_benchmark - tasa_libre_riesgo)))
        
        # Preparar resultado
        resultado = {
            "benchmark": benchmark,
            "periodo": {
                "desde": datetime.fromtimestamp(benchmark_data[0][0]).isoformat(),
                "hasta": datetime.fromtimestamp(benchmark_data[-1][0]).isoformat(),
                "duracion_dias": (benchmark_data[-1][0] - benchmark_data[0][0]) / (24 * 3600)
            },
            "rendimiento_final": {
                "sistema": rendimiento_final_sistema * 100,
                "benchmark": rendimiento_final_benchmark * 100,
                "diferencia": (rendimiento_final_sistema - rendimiento_final_benchmark) * 100,
                "ratio": (1 + rendimiento_final_sistema) / (1 + rendimiento_final_benchmark) if (1 + rendimiento_final_benchmark) != 0 else float('inf')
            },
            "analisis": {
                "correlacion": correlacion,
                "beta": beta,
                "alpha": alpha * 100,  # En porcentaje
                "tracking_error": np.std([c["diferencia"] for c in comparacion]) * 100 if len(comparacion) > 1 else 0,
                "information_ratio": (rendimiento_final_sistema - rendimiento_final_benchmark) / 
                                    (np.std([c["diferencia"] for c in comparacion]) if len(comparacion) > 1 else 1)
            },
            "datos_comparativos": [
                {
                    "fecha": c["fecha"],
                    "sistema": c["rendimiento_sistema"] * 100,
                    "benchmark": c["rendimiento_benchmark"] * 100,
                    "diferencia": c["diferencia"] * 100
                } for c in comparacion[::max(1, len(comparacion) // 10)]  # Mostrar ~10 puntos
            ]
        }
        
        return resultado
    
    async def analizar_atribucion_rendimiento(self) -> Dict[str, Any]:
        """
        Analizar atribución del rendimiento por activo, estrategia y sector.
        
        Returns:
            Diccionario con análisis detallado de contribución al rendimiento
        """
        if not self.operaciones:
            return {"error": "No hay operaciones para analizar"}
        
        # Atribución por activo
        atribucion_activo = {}
        for symbol, stats in self.rendimiento_por_activo.items():
            atribucion_activo[symbol] = {
                "resultado_neto": stats["resultado_neto"],
                "operaciones": stats["operaciones_total"],
                "porcentaje_contribucion": 0  # Se calculará después
            }
        
        # Atribución por estrategia
        atribucion_estrategia = {}
        for estrategia, stats in self.rendimiento_por_estrategia.items():
            atribucion_estrategia[estrategia] = {
                "resultado_neto": stats["resultado_neto"],
                "operaciones": stats["operaciones_total"],
                "porcentaje_contribucion": 0  # Se calculará después
            }
        
        # Atribución por sector (en un sistema real, tendríamos esta información)
        # Para esta demo, simulamos algunos sectores
        atribucion_sector = {
            "defi": {"resultado_neto": 0, "operaciones": 0},
            "smart_contracts": {"resultado_neto": 0, "operaciones": 0},
            "exchanges": {"resultado_neto": 0, "operaciones": 0},
            "layer1": {"resultado_neto": 0, "operaciones": 0},
            "otros": {"resultado_neto": 0, "operaciones": 0}
        }
        
        # Asignar sectores simulados a activos conocidos
        sectores_conocidos = {
            "BTC": "layer1",
            "ETH": "smart_contracts",
            "BNB": "exchanges",
            "SOL": "smart_contracts",
            "ADA": "smart_contracts",
            "DOT": "layer1",
            "LINK": "defi",
            "UNI": "defi",
            "AAVE": "defi",
            "COMP": "defi",
            "CAKE": "defi",
            "CRV": "defi",
            "SUSHI": "defi",
            "DYDX": "exchanges",
            "FTT": "exchanges",
            "AVAX": "layer1",
            "MATIC": "layer1",
            "ATOM": "layer1",
            "FIL": "otros",
            "XRP": "otros"
        }
        
        # Calcular atribución por sector
        for symbol, stats in self.rendimiento_por_activo.items():
            sector = sectores_conocidos.get(symbol, "otros")
            atribucion_sector[sector]["resultado_neto"] += stats["resultado_neto"]
            atribucion_sector[sector]["operaciones"] += stats["operaciones_total"]
        
        # Calcular contribución porcentual
        resultado_total = sum(stats["resultado_neto"] for stats in atribucion_activo.values())
        
        if resultado_total != 0:
            # Activos
            for symbol in atribucion_activo:
                atribucion_activo[symbol]["porcentaje_contribucion"] = (
                    atribucion_activo[symbol]["resultado_neto"] / abs(resultado_total) * 100
                    if resultado_total != 0 else 0
                )
            
            # Estrategias
            for estrategia in atribucion_estrategia:
                atribucion_estrategia[estrategia]["porcentaje_contribucion"] = (
                    atribucion_estrategia[estrategia]["resultado_neto"] / abs(resultado_total) * 100
                    if resultado_total != 0 else 0
                )
            
            # Sectores
            for sector in atribucion_sector:
                atribucion_sector[sector]["porcentaje_contribucion"] = (
                    atribucion_sector[sector]["resultado_neto"] / abs(resultado_total) * 100
                    if resultado_total != 0 else 0
                )
        
        # Ordenar por contribución (de mayor a menor)
        atribucion_activo_ordenada = sorted(
            [{"symbol": k, **v} for k, v in atribucion_activo.items()],
            key=lambda x: abs(x["resultado_neto"]),
            reverse=True
        )
        
        atribucion_estrategia_ordenada = sorted(
            [{"estrategia": k, **v} for k, v in atribucion_estrategia.items()],
            key=lambda x: abs(x["resultado_neto"]),
            reverse=True
        )
        
        atribucion_sector_ordenada = sorted(
            [{"sector": k, **v} for k, v in atribucion_sector.items()],
            key=lambda x: abs(x["resultado_neto"]),
            reverse=True
        )
        
        # Preparar resultado
        resultado = {
            "resultado_total": resultado_total,
            "por_activo": atribucion_activo_ordenada,
            "por_estrategia": atribucion_estrategia_ordenada,
            "por_sector": atribucion_sector_ordenada
        }
        
        return resultado
    
    async def analizar_calidad_ejecucion(self) -> Dict[str, Any]:
        """
        Analizar la calidad de ejecución de todas las operaciones.
        
        Returns:
            Diccionario con análisis detallado de la calidad de ejecución
        """
        if not self.operaciones:
            return {"error": "No hay operaciones para analizar"}
        
        # En un sistema real, tendríamos datos más detallados para este análisis
        # Para esta demo, usamos datos limitados y simulaciones
        
        # Calcular métricas agregadas
        operaciones_con_slippage = [op for op in self.operaciones if "slippage" in op]
        operaciones_con_latencia = [op for op in self.operaciones if "latencia_ms" in op]
        
        slippage_promedio = sum(op["slippage"] for op in operaciones_con_slippage) / len(operaciones_con_slippage) if operaciones_con_slippage else 0
        
        latencia_promedio = sum(op["latencia_ms"] for op in operaciones_con_latencia) / len(operaciones_con_latencia) if operaciones_con_latencia else 0
        
        # Calcular costo de oportunidad (simplificado)
        costo_oportunidad_total = 0
        for op in self.operaciones:
            if "precio_objetivo" in op and "entrada_precio" in op:
                if op["tipo"].upper() == "LONG":
                    # Para compras, costo = (precio_entrada - precio_objetivo) / precio_objetivo
                    costo = (op["entrada_precio"] - op["precio_objetivo"]) / op["precio_objetivo"]
                else:
                    # Para ventas, costo = (precio_objetivo - precio_entrada) / precio_objetivo
                    costo = (op["precio_objetivo"] - op["entrada_precio"]) / op["precio_objetivo"]
                
                costo_oportunidad_total += max(0, costo) * op["unidades"] * op["precio_objetivo"]
        
        # Calcular impact en el mercado (simplificado)
        market_impact_promedio = 0
        if operaciones_con_slippage:
            # En un sistema real, tendríamos mejor forma de medir el impact
            market_impact_promedio = slippage_promedio * 0.7  # Estimación: 70% del slippage
        
        # Calcular desviación del VWAP (simplificado)
        desviacion_vwap_promedio = 0
        operaciones_con_vwap = [op for op in self.operaciones if "vwap" in op]
        
        if operaciones_con_vwap:
            desviaciones = []
            for op in operaciones_con_vwap:
                if op["tipo"].upper() == "LONG":
                    # Para compras, desviación = (precio_entrada - vwap) / vwap
                    desviacion = (op["entrada_precio"] - op["vwap"]) / op["vwap"]
                else:
                    # Para ventas, desviación = (vwap - precio_entrada) / vwap
                    desviacion = (op["vwap"] - op["entrada_precio"]) / op["vwap"]
                
                desviaciones.append(desviacion)
            
            desviacion_vwap_promedio = sum(desviaciones) / len(desviaciones)
        
        # Actualizar métricas globales
        self.metricas_ejecucion["slippage_promedio"] = slippage_promedio
        self.metricas_ejecucion["latencia_promedio"] = latencia_promedio
        self.metricas_ejecucion["costo_oportunidad"] = costo_oportunidad_total
        self.metricas_ejecucion["market_impact"] = market_impact_promedio
        self.metricas_ejecucion["desviacion_vwap"] = desviacion_vwap_promedio
        
        # Calcular eficiencia general de ejecución (0-1)
        # Una puntuación que combina todos los factores
        factores = [
            max(0, 1 - slippage_promedio / 0.005),  # Normalizado a 0.5% máximo
            max(0, 1 - latencia_promedio / 1000),   # Normalizado a 1000ms máximo
            max(0, 1 - market_impact_promedio / 0.003),  # Normalizado a 0.3% máximo
            max(0, 1 - abs(desviacion_vwap_promedio) / 0.01)  # Normalizado a 1% máximo
        ]
        
        eficiencia_ejecucion = sum(factores) / len(factores)
        self.metricas_ejecucion["eficiencia_ejecucion"] = eficiencia_ejecucion
        
        # Preparar análisis por exchange
        analisis_por_exchange = {}
        for op in self.operaciones:
            exchange = op.get("exchange", "desconocido")
            if exchange not in analisis_por_exchange:
                analisis_por_exchange[exchange] = {
                    "operaciones": 0,
                    "slippage_promedio": 0,
                    "latencia_promedio": 0,
                    "fill_ratio": 0,
                    "operaciones_con_datos": 0
                }
            
            analisis_por_exchange[exchange]["operaciones"] += 1
            
            if "slippage" in op:
                analisis_por_exchange[exchange]["slippage_promedio"] += op["slippage"]
                analisis_por_exchange[exchange]["operaciones_con_datos"] += 1
            
            if "latencia_ms" in op:
                analisis_por_exchange[exchange]["latencia_promedio"] += op["latencia_ms"]
            
            if "fill_ratio" in op:
                analisis_por_exchange[exchange]["fill_ratio"] += op["fill_ratio"]
        
        # Calcular promedios por exchange
        for exchange in analisis_por_exchange:
            stats = analisis_por_exchange[exchange]
            if stats["operaciones_con_datos"] > 0:
                stats["slippage_promedio"] /= stats["operaciones_con_datos"]
            if stats["operaciones"] > 0:
                stats["latencia_promedio"] /= stats["operaciones"]
                stats["fill_ratio"] /= stats["operaciones"]
        
        # Preparar resultado
        resultado = {
            "metricas_globales": {
                "slippage_promedio": slippage_promedio * 100,  # En porcentaje
                "latencia_promedio": latencia_promedio,  # En ms
                "costo_oportunidad_total": costo_oportunidad_total,
                "market_impact_promedio": market_impact_promedio * 100,  # En porcentaje
                "desviacion_vwap_promedio": desviacion_vwap_promedio * 100,  # En porcentaje
                "eficiencia_ejecucion": eficiencia_ejecucion * 100  # En porcentaje
            },
            "analisis_por_exchange": analisis_por_exchange,
            "recomendaciones": []
        }
        
        # Generar recomendaciones
        if slippage_promedio > 0.003:
            resultado["recomendaciones"].append({
                "tipo": "SLIPPAGE",
                "mensaje": "El slippage promedio es elevado. Considerar utilizar órdenes limitadas con mayor frecuencia o reducir el tamaño de las operaciones."
            })
        
        if latencia_promedio > 500:
            resultado["recomendaciones"].append({
                "tipo": "LATENCIA",
                "mensaje": "La latencia promedio es alta. Verificar la calidad de la conexión o considerar servidores más cercanos a los exchanges."
            })
        
        if desviacion_vwap_promedio > 0.005:
            resultado["recomendaciones"].append({
                "tipo": "VWAP",
                "mensaje": "La desviación del VWAP es significativa. Considerar implementar estrategias de ejecución basadas en TWAP/VWAP para operaciones grandes."
            })
        
        # Añadir recomendaciones específicas por exchange
        for exchange, stats in analisis_por_exchange.items():
            if stats["slippage_promedio"] > 0.004:
                resultado["recomendaciones"].append({
                    "tipo": "EXCHANGE",
                    "mensaje": f"El exchange {exchange} muestra un slippage superior al promedio. Considerar redistribuir el volumen a otros exchanges con mejor liquidez."
                })
        
        return resultado
    
    def get_estado_actual(self) -> Dict[str, Any]:
        """
        Obtener estado actual completo del rastreador de rendimiento.
        
        Returns:
            Diccionario con estado actual y métricas principales
        """
        return {
            "capital": {
                "inicial": self.capital_inicial,
                "actual": self.metricas["capital_actual"],
                "cambio_porcentual": ((self.metricas["capital_actual"] / self.capital_inicial) - 1) * 100
            },
            "rendimiento": {
                "total": self.metricas["rendimiento_total"] * 100,
                "anualizado": self.metricas["rendimiento_anualizado"] * 100,
                "volatilidad": self.metricas["volatilidad"] * 100,
                "max_drawdown": self.metricas["max_drawdown"] * 100,
                "sharpe_ratio": self.metricas["sharpe_ratio"],
                "sortino_ratio": self.metricas["sortino_ratio"],
                "calmar_ratio": self.metricas["calmar_ratio"]
            },
            "operaciones": {
                "total": len(self.operaciones),
                "win_rate": self.metricas["win_rate"] * 100,
                "profit_factor": self.metricas["profit_factor"],
                "expectativa_matematica": self.metricas["expectativa_matematica"]
            },
            "ejecucion": {
                "slippage_promedio": self.metricas_ejecucion["slippage_promedio"] * 100,
                "latencia_promedio": self.metricas_ejecucion["latencia_promedio"],
                "eficiencia_ejecucion": self.metricas_ejecucion["eficiencia_ejecucion"] * 100
            },
            "fecha_inicio": self.fecha_inicio.isoformat(),
            "dias_operacion": (datetime.now() - self.fecha_inicio).days,
            "activos_operados": len(self.rendimiento_por_activo),
            "estrategias": list(self.rendimiento_por_estrategia.keys())
        }
    
    # Métodos internos
    
    async def _calcular_metricas_rendimiento(self) -> None:
        """Recalcular todas las métricas de rendimiento."""
        # Verificar que hay suficientes datos
        if len(self.historial_capital) < 2:
            return
        
        # Rendimiento total
        capital_inicial = self.historial_capital[0]["capital"]
        capital_actual = self.historial_capital[-1]["capital"]
        self.metricas["rendimiento_total"] = (capital_actual / capital_inicial) - 1
        
        # Calcular rendimientos diarios
        rendimientos = []
        for i in range(1, len(self.historial_capital)):
            r = (self.historial_capital[i]["capital"] / self.historial_capital[i-1]["capital"]) - 1
            rendimientos.append(r)
        
        # Volatilidad
        self.metricas["volatilidad"] = np.std(rendimientos) if rendimientos else 0
        
        # Calcular drawdown máximo
        drawdown_max = 0
        peak = capital_inicial
        for entry in self.historial_capital:
            if entry["capital"] > peak:
                peak = entry["capital"]
            drawdown_actual = 1 - (entry["capital"] / peak)
            drawdown_max = max(drawdown_max, drawdown_actual)
        
        self.metricas["max_drawdown"] = drawdown_max
        
        # Rendimiento anualizado
        dias_actividad = (datetime.now() - self.fecha_inicio).days
        if dias_actividad > 0:
            self.metricas["rendimiento_anualizado"] = (
                (1 + self.metricas["rendimiento_total"]) ** (365.25 / dias_actividad) - 1
            )
        
        # Calcular ratios
        tasa_libre_riesgo_anual = 0.02  # 2% anual
        tasa_libre_riesgo_diaria = (1 + tasa_libre_riesgo_anual) ** (1/365.25) - 1
        
        # Sharpe Ratio
        if self.metricas["volatilidad"] > 0:
            rendimiento_excedente = self.metricas["rendimiento_anualizado"] - tasa_libre_riesgo_anual
            self.metricas["sharpe_ratio"] = rendimiento_excedente / self.metricas["volatilidad"]
        
        # Sortino Ratio
        rendimientos_negativos = [r for r in rendimientos if r < 0]
        volatilidad_downside = np.std(rendimientos_negativos) if rendimientos_negativos else 0.01
        
        if volatilidad_downside > 0:
            self.metricas["sortino_ratio"] = (
                (self.metricas["rendimiento_anualizado"] - tasa_libre_riesgo_anual) / 
                volatilidad_downside
            )
        
        # Calmar Ratio
        if self.metricas["max_drawdown"] > 0:
            self.metricas["calmar_ratio"] = (
                self.metricas["rendimiento_anualizado"] / self.metricas["max_drawdown"]
            )
        
        # VaR (Valor en Riesgo)
        if rendimientos:
            rendimientos_ordenados = sorted(rendimientos)
            indice_95 = int(len(rendimientos_ordenados) * 0.05)
            indice_99 = int(len(rendimientos_ordenados) * 0.01)
            
            self.metricas["valor_en_riesgo_95"] = abs(rendimientos_ordenados[indice_95]) if indice_95 < len(rendimientos_ordenados) else 0
            self.metricas["valor_en_riesgo_99"] = abs(rendimientos_ordenados[indice_99]) if indice_99 < len(rendimientos_ordenados) else 0
        
        # Métricas de operaciones
        if self.operaciones:
            ops_ganadoras = [op for op in self.operaciones if op["ganadora"]]
            ops_perdedoras = [op for op in self.operaciones if not op["ganadora"]]
            
            self.metricas["win_rate"] = len(ops_ganadoras) / len(self.operaciones) if self.operaciones else 0
            
            ganancia_total = sum(op["resultado_usd"] for op in ops_ganadoras)
            perdida_total = sum(abs(op["resultado_usd"]) for op in ops_perdedoras)
            
            self.metricas["profit_factor"] = ganancia_total / perdida_total if perdida_total > 0 else float('inf')
            
            # Expectativa matemática
            ganancia_promedio = ganancia_total / len(ops_ganadoras) if ops_ganadoras else 0
            perdida_promedio = perdida_total / len(ops_perdedoras) if ops_perdedoras else 0
            
            self.metricas["expectativa_matematica"] = (
                (ganancia_promedio * self.metricas["win_rate"]) - 
                (perdida_promedio * (1 - self.metricas["win_rate"]))
            )
            
            # Factor de recuperación
            if self.metricas["max_drawdown"] > 0:
                self.metricas["factor_recuperacion"] = (
                    self.metricas["rendimiento_total"] / self.metricas["max_drawdown"]
                )
    
    def _interpolar_capital(self, timestamp: float) -> float:
        """
        Interpolar capital para un timestamp específico.
        
        Args:
            timestamp: Timestamp para el que se busca el valor
            
        Returns:
            Valor de capital interpolado
        """
        # Caso 1: Timestamp antes del primer registro
        if timestamp <= self.historial_capital[0]["timestamp"]:
            return self.historial_capital[0]["capital"]
        
        # Caso 2: Timestamp después del último registro
        if timestamp >= self.historial_capital[-1]["timestamp"]:
            return self.historial_capital[-1]["capital"]
        
        # Caso 3: Buscar entre los registros
        for i in range(1, len(self.historial_capital)):
            if self.historial_capital[i]["timestamp"] >= timestamp:
                # Encontramos el intervalo
                t1 = self.historial_capital[i-1]["timestamp"]
                t2 = self.historial_capital[i]["timestamp"]
                v1 = self.historial_capital[i-1]["capital"]
                v2 = self.historial_capital[i]["capital"]
                
                # Interpolar linealmente
                if t2 == t1:  # Evitar división por cero
                    return v1
                
                factor = (timestamp - t1) / (t2 - t1)
                return v1 + factor * (v2 - v1)
        
        # No debería llegar aquí
        return self.historial_capital[-1]["capital"]
    
    async def _simular_benchmark(self, benchmark: str) -> None:
        """
        Simular un benchmark para pruebas y demos.
        
        Args:
            benchmark: Nombre del benchmark a simular
        """
        import random
        
        # Parámetros de simulación según el benchmark
        if benchmark == "bitcoin":
            volatilidad_diaria = 0.03
            tendencia_diaria = 0.0005
            valor_inicial = 40000
        elif benchmark == "crypto_top10":
            volatilidad_diaria = 0.025
            tendencia_diaria = 0.0004
            valor_inicial = 1000
        else:  # sp500
            volatilidad_diaria = 0.01
            tendencia_diaria = 0.0003
            valor_inicial = 4000
        
        # Generar datos históricos
        timestamp_inicio = self.historial_capital[0]["timestamp"]
        timestamp_actual = datetime.now().timestamp()
        
        # Generar un punto por día
        timestamps = []
        current_ts = timestamp_inicio
        while current_ts <= timestamp_actual:
            timestamps.append(current_ts)
            current_ts += 24 * 3600  # Un día en segundos
        
        # Generar valores
        valores = [valor_inicial]
        for i in range(1, len(timestamps)):
            cambio = random.normalvariate(tendencia_diaria, volatilidad_diaria)
            nuevo_valor = valores[-1] * (1 + cambio)
            valores.append(nuevo_valor)
        
        # Guardar benchmark
        self.benchmarks[benchmark] = list(zip(timestamps, valores))
    
    async def _analizar_calidad_ejecucion(self, operacion: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analizar la calidad de ejecución de una operación.
        
        Args:
            operacion: Diccionario con detalles de la operación
            
        Returns:
            Diccionario con análisis de calidad de ejecución
        """
        analisis = {
            "slippage": 0.0,
            "latencia_ms": 0.0,
            "precio_ideal": 0.0,
            "impacto_slippage_usd": 0.0,
            "calidad_ejecucion": 1.0
        }
        
        # En un sistema real, tendríamos datos específicos de la ejecución
        # Para esta demo, simulamos algunos datos o usamos los disponibles
        
        # Calcular slippage si no está ya en la operación
        if "slippage" not in operacion:
            if "precio_objetivo" in operacion and "entrada_precio" in operacion:
                precio_objetivo = operacion["precio_objetivo"]
                precio_entrada = operacion["entrada_precio"]
                
                if operacion["tipo"].upper() == "LONG":
                    # Para compras, slippage = (precio_entrada - precio_objetivo) / precio_objetivo
                    analisis["slippage"] = max(0, (precio_entrada - precio_objetivo) / precio_objetivo)
                else:  # SHORT
                    # Para ventas, slippage = (precio_objetivo - precio_entrada) / precio_objetivo
                    analisis["slippage"] = max(0, (precio_objetivo - precio_entrada) / precio_objetivo)
            else:
                # Simular un slippage típico
                analisis["slippage"] = abs(np.random.normal(0.001, 0.0005))
        else:
            analisis["slippage"] = operacion["slippage"]
        
        # Obtener o simular latencia
        if "latencia_ms" in operacion:
            analisis["latencia_ms"] = operacion["latencia_ms"]
        else:
            # Simular latencia típica (50-300ms)
            analisis["latencia_ms"] = np.random.uniform(50, 300)
        
        # Calcular precio ideal (si no hay slippage)
        if operacion["tipo"].upper() == "LONG":
            analisis["precio_ideal"] = operacion["entrada_precio"] / (1 + analisis["slippage"])
        else:
            analisis["precio_ideal"] = operacion["entrada_precio"] * (1 + analisis["slippage"])
        
        # Calcular impacto del slippage en USD
        analisis["impacto_slippage_usd"] = (
            abs(analisis["precio_ideal"] - operacion["entrada_precio"]) * 
            operacion["unidades"]
        )
        
        # Calcular calidad general de ejecución (0-1)
        calidad_slippage = max(0, 1 - analisis["slippage"] / 0.005)  # Normalizado a 0.5% máximo
        calidad_latencia = max(0, 1 - analisis["latencia_ms"] / 1000)  # Normalizado a 1000ms máximo
        
        analisis["calidad_ejecucion"] = (calidad_slippage * 0.7 + calidad_latencia * 0.3)
        
        return analisis

# Instancia global para acceso desde cualquier módulo
performance_tracker = TranscendentalPerformanceTracker()

async def initialize_performance_tracker():
    """Inicializar el rastreador de rendimiento con configuración predeterminada."""
    logger.info("Inicializando TranscendentalPerformanceTracker...")
    # En un sistema real, aquí cargaríamos configuración desde base de datos
    return performance_tracker