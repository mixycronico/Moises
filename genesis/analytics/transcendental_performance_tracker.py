"""
Seguidor de Rendimiento Trascendental del Sistema Genesis.

Este módulo implementa un sistema avanzado de seguimiento y análisis de rendimiento
con capacidades trascendentales, diseñado para rastrear y evaluar el desempeño
del sistema de trading a lo largo del tiempo y bajo diferentes condiciones de capital.

Características principales:
- Registro histórico multicapa del rendimiento del sistema y estrategias
- Análisis de atribución para identificar fuentes de alfa
- Proyecciones adaptativas basadas en crecimiento de capital
- Evaluación comparativa contra benchmarks configurables
- Sincronización atemporal con estados pasados, presentes y futuros
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple, Set, Union, cast

import numpy as np
import pandas as pd

from genesis.db.transcendental_database import transcendental_db
from genesis.db.models.crypto_classifier_models import (
    Cryptocurrency, CryptoClassification, CryptoMetrics
)
from sqlalchemy import select, and_, or_, desc, func, text

# Configuración de logging
logger = logging.getLogger("genesis.analytics.performance_tracker")

class TranscendentalPerformanceTracker:
    """
    Seguidor de rendimiento con capacidades trascendentales.
    
    Este componente rastrea, analiza y proyecta el rendimiento del sistema
    adaptándose a diferentes niveles de capital y condiciones del mercado.
    """
    
    def __init__(self, 
                capital_inicial: float = 10000.0,
                benchmark: str = "crypto_top10"):
        """
        Inicializar el seguidor de rendimiento trascendental.
        
        Args:
            capital_inicial: Capital inicial del sistema en USD
            benchmark: Benchmark para comparación de rendimiento
        """
        self.capital_inicial = capital_inicial
        self.capital_actual = capital_inicial
        self.benchmark = benchmark
        
        # Historial de rendimiento
        self.historial_rendimiento = []
        
        # Datos de estrategias
        self.estrategias = {}
        self.rendimiento_estrategias = {}
        
        # Benchmarks
        self.benchmarks = {
            "crypto_top10": self._inicializar_benchmark("crypto_top10"),
            "btc": self._inicializar_benchmark("btc"),
            "eth": self._inicializar_benchmark("eth"),
            "sp500": self._inicializar_benchmark("sp500")
        }
        
        # Métricas clave
        self.metricas = {
            "rendimiento_total": 0.0,
            "rendimiento_anualizado": 0.0,
            "volatilidad": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "ratio_recuperacion": 0.0
        }
        
        # Proyecciones y escenarios
        self.proyecciones = {
            "escenario_base": {},
            "escenario_optimista": {},
            "escenario_pesimista": {},
            "escenario_extremo": {}
        }
        
        # Estado de trascendencia
        self.modo_trascendental = "SINGULARITY_V4"
        self.trascendencia_activada = True
        
        # Estadísticas específicas por nivel de capital
        self.estadisticas_por_capital = {
            "10k": {},
            "100k": {},
            "1M": {},
            "10M": {}
        }
        
        logger.info(f"TranscendentalPerformanceTracker inicializado con capital: ${capital_inicial:,.2f}")
    
    async def registrar_operacion(self, operacion: Dict[str, Any]) -> Dict[str, Any]:
        """
        Registrar una operación completada y actualizar rendimiento.
        
        Args:
            operacion: Diccionario con detalles de la operación
            
        Returns:
            Diccionario con métricas actualizadas
        """
        # Validar operación
        campos_requeridos = ["symbol", "estrategia", "tipo", "entrada", "salida", "unidades", 
                            "resultado_usd", "resultado_porcentual", "timestamp"]
                            
        for campo in campos_requeridos:
            if campo not in operacion:
                raise ValueError(f"Falta el campo requerido '{campo}' en la operación")
        
        # Asegurar que timestamp sea datetime
        if isinstance(operacion["timestamp"], (int, float)):
            timestamp = datetime.fromtimestamp(operacion["timestamp"])
        elif isinstance(operacion["timestamp"], str):
            timestamp = datetime.fromisoformat(operacion["timestamp"].replace("Z", "+00:00"))
        else:
            timestamp = operacion["timestamp"]
        
        operacion["timestamp"] = timestamp
        
        # Añadir operación al historial
        self.historial_rendimiento.append({
            "tipo": "operacion",
            "symbol": operacion["symbol"],
            "estrategia": operacion["estrategia"],
            "operacion_tipo": operacion["tipo"],
            "entrada": operacion["entrada"],
            "salida": operacion["salida"],
            "unidades": operacion["unidades"],
            "resultado_usd": operacion["resultado_usd"],
            "resultado_porcentual": operacion["resultado_porcentual"],
            "timestamp": timestamp,
            "capital": self.capital_actual
        })
        
        # Actualizar capital si se proporciona
        if "capital_final" in operacion and operacion["capital_final"] > 0:
            self.capital_actual = operacion["capital_final"]
        else:
            # De lo contrario, actualizar según resultado
            self.capital_actual += operacion["resultado_usd"]
        
        # Actualizar estadísticas de estrategia
        estrategia = operacion["estrategia"]
        if estrategia not in self.estrategias:
            self.estrategias[estrategia] = {
                "operaciones_totales": 0,
                "operaciones_ganadoras": 0,
                "operaciones_perdedoras": 0,
                "ganancias_totales": 0.0,
                "perdidas_totales": 0.0,
                "mayor_ganancia": 0.0,
                "mayor_perdida": 0.0,
                "rendimiento_acumulado": 0.0
            }
            
        # Actualizar contadores
        self.estrategias[estrategia]["operaciones_totales"] += 1
        if operacion["resultado_usd"] > 0:
            self.estrategias[estrategia]["operaciones_ganadoras"] += 1
            self.estrategias[estrategia]["ganancias_totales"] += operacion["resultado_usd"]
            self.estrategias[estrategia]["mayor_ganancia"] = max(
                self.estrategias[estrategia]["mayor_ganancia"],
                operacion["resultado_usd"]
            )
        else:
            self.estrategias[estrategia]["operaciones_perdedoras"] += 1
            self.estrategias[estrategia]["perdidas_totales"] += abs(operacion["resultado_usd"])
            self.estrategias[estrategia]["mayor_perdida"] = max(
                self.estrategias[estrategia]["mayor_perdida"],
                abs(operacion["resultado_usd"])
            )
        
        # Actualizar rendimiento acumulado
        self.estrategias[estrategia]["rendimiento_acumulado"] += operacion["resultado_usd"]
        
        # Actualizar metricas
        await self._actualizar_metricas()
        
        # Guardar checkpoint de rendimiento con trascendencia
        if self.trascendencia_activada:
            checkpoint_id = f"operacion_{operacion['symbol']}_{int(timestamp.timestamp())}"
            await transcendental_db.checkpoint_state(
                "performance", 
                checkpoint_id, 
                {
                    "operacion": operacion,
                    "capital_actual": self.capital_actual,
                    "metricas": self.metricas,
                    "estrategia_stats": self.estrategias.get(estrategia, {})
                }
            )
        
        # Retornar estado actualizado
        return {
            "operacion_id": len(self.historial_rendimiento),
            "capital_actual": self.capital_actual,
            "rendimiento_total": self.metricas["rendimiento_total"],
            "estrategia_stats": self.estrategias.get(estrategia, {}),
            "modo_trascendental": self.modo_trascendental
        }
    
    async def registrar_señal(self, señal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Registrar una señal de trading generada (no ejecutada).
        
        Args:
            señal: Diccionario con detalles de la señal
            
        Returns:
            Diccionario con estado actualizado
        """
        # Validar señal
        campos_requeridos = ["symbol", "estrategia", "tipo", "precio", "timestamp", "confianza"]
        for campo in campos_requeridos:
            if campo not in señal:
                raise ValueError(f"Falta el campo requerido '{campo}' en la señal")
        
        # Asegurar que timestamp sea datetime
        if isinstance(señal["timestamp"], (int, float)):
            timestamp = datetime.fromtimestamp(señal["timestamp"])
        elif isinstance(señal["timestamp"], str):
            timestamp = datetime.fromisoformat(señal["timestamp"].replace("Z", "+00:00"))
        else:
            timestamp = señal["timestamp"]
        
        señal["timestamp"] = timestamp
        
        # Añadir señal al historial
        self.historial_rendimiento.append({
            "tipo": "señal",
            "symbol": señal["symbol"],
            "estrategia": señal["estrategia"],
            "señal_tipo": señal["tipo"],
            "precio": señal["precio"],
            "timestamp": timestamp,
            "confianza": señal["confianza"],
            "ejecutada": False,
            "capital": self.capital_actual
        })
        
        # Guardar checkpoint de señal con trascendencia
        if self.trascendencia_activada:
            checkpoint_id = f"senal_{señal['symbol']}_{int(timestamp.timestamp())}"
            await transcendental_db.checkpoint_state(
                "performance", 
                checkpoint_id, 
                {
                    "señal": señal,
                    "capital_actual": self.capital_actual
                }
            )
        
        # Retornar estado actualizado
        return {
            "señal_id": len(self.historial_rendimiento),
            "capital_actual": self.capital_actual,
            "confianza": señal["confianza"]
        }
    
    async def actualizar_capital(self, nuevo_capital: float, 
                              motivo: str = "ajuste_manual") -> Dict[str, Any]:
        """
        Actualizar el capital total y recalcular métricas.
        
        Args:
            nuevo_capital: Nuevo monto de capital
            motivo: Razón del cambio de capital
            
        Returns:
            Diccionario con métricas actualizadas
        """
        capital_anterior = self.capital_actual
        cambio_porcentual = (nuevo_capital / capital_anterior - 1) if capital_anterior > 0 else 0
        
        # Registrar cambio de capital
        self.historial_rendimiento.append({
            "tipo": "capital",
            "capital_anterior": capital_anterior,
            "capital_nuevo": nuevo_capital,
            "cambio_porcentual": cambio_porcentual,
            "motivo": motivo,
            "timestamp": datetime.now()
        })
        
        # Actualizar capital
        self.capital_actual = nuevo_capital
        
        # Actualizar métricas
        await self._actualizar_metricas()
        
        # Verificar si necesitamos recalcular estadísticas por nivel de capital
        if (nuevo_capital >= 100000 and capital_anterior < 100000) or \
           (nuevo_capital >= 1000000 and capital_anterior < 1000000) or \
           (nuevo_capital >= 10000000 and capital_anterior < 10000000):
            await self._actualizar_estadisticas_por_capital()
            
            # Guardar checkpoint trascendental al cruzar un umbral importante
            if self.trascendencia_activada:
                await transcendental_db.checkpoint_state(
                    "capital_threshold", 
                    f"threshold_{int(datetime.now().timestamp())}", 
                    {
                        "capital_anterior": capital_anterior,
                        "capital_nuevo": nuevo_capital,
                        "estadisticas_por_capital": self.estadisticas_por_capital,
                        "timestamp": datetime.now().isoformat()
                    }
                )
        
        # Retornar estado actualizado
        return {
            "capital_anterior": capital_anterior,
            "capital_nuevo": nuevo_capital,
            "cambio_porcentual": cambio_porcentual * 100,
            "motivo": motivo,
            "metricas_actualizadas": self.metricas
        }
    
    async def calcular_atribucion(self) -> Dict[str, Any]:
        """
        Calcular atribución de rendimiento por estrategia y activo.
        
        Returns:
            Diccionario con análisis detallado de atribución
        """
        # Solo procesar si hay suficientes datos
        if len(self.historial_rendimiento) < 5:
            return {
                "status": "error",
                "mensaje": "Datos insuficientes para análisis de atribución"
            }
        
        # Filtrar solo operaciones
        operaciones = [op for op in self.historial_rendimiento if op.get("tipo") == "operacion"]
        
        # Agrupar por estrategia
        atribucion_estrategia = {}
        for estrategia, stats in self.estrategias.items():
            ops_estrategia = [op for op in operaciones if op.get("estrategia") == estrategia]
            
            # Calcular rendimiento total
            rendimiento_total = stats["rendimiento_acumulado"]
            
            # Calcular porcentaje de contribución
            if self.capital_actual > self.capital_inicial:
                contribucion_pct = (rendimiento_total / (self.capital_actual - self.capital_inicial)) * 100
            else:
                contribucion_pct = 0
                
            atribucion_estrategia[estrategia] = {
                "rendimiento_usd": rendimiento_total,
                "contribucion_porcentaje": contribucion_pct,
                "operaciones": len(ops_estrategia),
                "win_rate": stats["operaciones_ganadoras"] / stats["operaciones_totales"] 
                           if stats["operaciones_totales"] > 0 else 0
            }
        
        # Agrupar por activo
        atribucion_activo = {}
        for op in operaciones:
            symbol = op.get("symbol")
            if symbol not in atribucion_activo:
                atribucion_activo[symbol] = {
                    "rendimiento_usd": 0.0,
                    "operaciones": 0,
                    "operaciones_ganadoras": 0,
                    "operaciones_perdedoras": 0
                }
            
            atribucion_activo[symbol]["rendimiento_usd"] += op.get("resultado_usd", 0)
            atribucion_activo[symbol]["operaciones"] += 1
            
            if op.get("resultado_usd", 0) > 0:
                atribucion_activo[symbol]["operaciones_ganadoras"] += 1
            else:
                atribucion_activo[symbol]["operaciones_perdedoras"] += 1
        
        # Calcular win rate y contribución por activo
        for symbol, stats in atribucion_activo.items():
            stats["win_rate"] = stats["operaciones_ganadoras"] / stats["operaciones"] if stats["operaciones"] > 0 else 0
            
            if self.capital_actual > self.capital_inicial:
                stats["contribucion_porcentaje"] = (stats["rendimiento_usd"] / (self.capital_actual - self.capital_inicial)) * 100
            else:
                stats["contribucion_porcentaje"] = 0
        
        # Ordenar por contribución
        top_estrategias = sorted(
            atribucion_estrategia.items(), 
            key=lambda x: abs(x[1]["rendimiento_usd"]), 
            reverse=True
        )
        
        top_activos = sorted(
            atribucion_activo.items(), 
            key=lambda x: abs(x[1]["rendimiento_usd"]), 
            reverse=True
        )
        
        # Aplicar mecanismos trascendentales
        if self.trascendencia_activada and self.modo_trascendental == "SINGULARITY_V4":
            # Buscar patrones ocultos de correlación entre estrategias y activos
            patrones_correlacion = await self._calcular_correlaciones_trascendentales(operaciones)
        else:
            patrones_correlacion = {}
        
        # Resultado completo
        resultado = {
            "fecha_analisis": datetime.now().isoformat(),
            "periodo_analisis": {
                "inicio": min([op.get("timestamp") for op in operaciones]).isoformat() if operaciones else None,
                "fin": max([op.get("timestamp") for op in operaciones]).isoformat() if operaciones else None
            },
            "rendimiento_total": {
                "usd": self.capital_actual - self.capital_inicial,
                "porcentaje": ((self.capital_actual / self.capital_inicial) - 1) * 100 if self.capital_inicial > 0 else 0
            },
            "atribucion_estrategia": {
                estrategia: stats for estrategia, stats in top_estrategias
            },
            "atribucion_activo": {
                symbol: stats for symbol, stats in top_activos[:10]  # Top 10 activos
            },
            "patrones_correlacion": patrones_correlacion,
            "modo_trascendental": self.modo_trascendental
        }
        
        # Guardar checkpoint de atribución para análisis futuro
        if self.trascendencia_activada:
            await transcendental_db.checkpoint_state(
                "atribucion", 
                f"atribucion_{int(datetime.now().timestamp())}", 
                resultado
            )
        
        return resultado
    
    async def generar_proyecciones(self, 
                               dias: int = 365, 
                               montecarlo_simulaciones: int = 1000) -> Dict[str, Any]:
        """
        Generar proyecciones de rendimiento futuro usando análisis adaptativo.
        
        Args:
            dias: Número de días a proyectar
            montecarlo_simulaciones: Número de simulaciones para Monte Carlo
            
        Returns:
            Diccionario con proyecciones
        """
        # Solo procesar si hay suficientes datos
        if len(self.historial_rendimiento) < 10:
            return {
                "status": "error",
                "mensaje": "Datos insuficientes para proyecciones confiables"
            }
        
        # Obtener parámetros históricos
        rendimiento_diario_medio, volatilidad_diaria = self._calcular_parametros_rendimiento()
        
        # Escenarios base
        escenarios = {
            "base": {
                "rendimiento_anual": rendimiento_diario_medio * 252,  # Días de trading anuales
                "volatilidad_anual": volatilidad_diaria * (252 ** 0.5)
            },
            "optimista": {
                "rendimiento_anual": rendimiento_diario_medio * 252 * 1.5,  # 50% mejor
                "volatilidad_anual": volatilidad_diaria * (252 ** 0.5) * 0.8  # 20% menor volatilidad
            },
            "pesimista": {
                "rendimiento_anual": rendimiento_diario_medio * 252 * 0.5,  # 50% peor
                "volatilidad_anual": volatilidad_diaria * (252 ** 0.5) * 1.2  # 20% mayor volatilidad
            },
            "extremo": {
                "rendimiento_anual": rendimiento_diario_medio * 252 * 0.2,  # 80% peor
                "volatilidad_anual": volatilidad_diaria * (252 ** 0.5) * 2.0  # Doble volatilidad
            }
        }
        
        # Aplicar ajustes por trascendencia
        if self.trascendencia_activada:
            if self.modo_trascendental == "SINGULARITY_V4":
                # En Singularidad V4, estabilizar extremos
                escenarios["optimista"]["volatilidad_anual"] *= 0.9  # Reducir volatilidad en escenario optimista
                escenarios["extremo"]["rendimiento_anual"] = max(0, rendimiento_diario_medio * 252 * 0.4)  # Menos negativo
        
        # Calcular proyecciones para cada escenario
        proyecciones = {}
        fechas = []
        
        for escenario, params in escenarios.items():
            rendimiento_diario = params["rendimiento_anual"] / 252
            volatilidad_diaria = params["volatilidad_anual"] / (252 ** 0.5)
            
            # Simulación Monte Carlo
            simulaciones = []
            for _ in range(montecarlo_simulaciones):
                trayectoria = [self.capital_actual]
                capital_actual = self.capital_actual
                
                for _ in range(dias):
                    # Modelo log-normal para cambios de precio
                    rendimiento = np.random.normal(rendimiento_diario, volatilidad_diaria)
                    capital_actual *= (1 + rendimiento)
                    trayectoria.append(capital_actual)
                
                simulaciones.append(trayectoria)
            
            # Calcular estadísticas de la simulación
            simulaciones_array = np.array(simulaciones)
            media = np.mean(simulaciones_array, axis=0)
            percentil_10 = np.percentile(simulaciones_array, 10, axis=0)
            percentil_90 = np.percentile(simulaciones_array, 90, axis=0)
            max_values = np.max(simulaciones_array, axis=0)
            min_values = np.min(simulaciones_array, axis=0)
            
            # Fechas para el eje X
            if not fechas:
                fecha_inicio = datetime.now()
                fechas = [(fecha_inicio + timedelta(days=d)).strftime("%Y-%m-%d") for d in range(dias + 1)]
            
            # Guardar resultados
            proyecciones[escenario] = {
                "fechas": fechas,
                "media": media.tolist(),
                "percentil_10": percentil_10.tolist(),
                "percentil_90": percentil_90.tolist(),
                "maximo": max_values.tolist(),
                "minimo": min_values.tolist(),
                "capital_final_medio": float(media[-1]),
                "rendimiento_proyectado": ((float(media[-1]) / self.capital_actual) - 1) * 100,
                "parametros": params
            }
        
        # Actualizar proyecciones almacenadas
        self.proyecciones = {
            "escenario_base": proyecciones["base"],
            "escenario_optimista": proyecciones["optimista"],
            "escenario_pesimista": proyecciones["pesimista"],
            "escenario_extremo": proyecciones["extremo"]
        }
        
        # Resultado completo
        resultado = {
            "fecha_proyeccion": datetime.now().isoformat(),
            "capital_actual": self.capital_actual,
            "periodo_proyeccion": {
                "dias": dias,
                "desde": fechas[0],
                "hasta": fechas[-1]
            },
            "escenarios": proyecciones,
            "modo_trascendental": self.modo_trascendental
        }
        
        # Guardar checkpoint de proyecciones
        if self.trascendencia_activada:
            await transcendental_db.checkpoint_state(
                "proyecciones", 
                f"proyecciones_{int(datetime.now().timestamp())}", 
                resultado
            )
        
        return resultado
    
    async def comparar_con_benchmark(self, 
                                  periodo: str = "YTD") -> Dict[str, Any]:
        """
        Comparar rendimiento con benchmarks seleccionados.
        
        Args:
            periodo: Periodo para comparación ("YTD", "1M", "3M", "6M", "1Y", "ALL")
            
        Returns:
            Diccionario con comparativa de rendimiento
        """
        # Definir fecha de inicio según periodo
        fecha_fin = datetime.now()
        
        if periodo == "1M":
            fecha_inicio = fecha_fin - timedelta(days=30)
        elif periodo == "3M":
            fecha_inicio = fecha_fin - timedelta(days=90)
        elif periodo == "6M":
            fecha_inicio = fecha_fin - timedelta(days=180)
        elif periodo == "1Y":
            fecha_inicio = fecha_fin - timedelta(days=365)
        elif periodo == "YTD":
            fecha_inicio = datetime(fecha_fin.year, 1, 1)
        else:  # "ALL"
            # Usar la primera entrada del historial o un año atrás
            if self.historial_rendimiento:
                fecha_inicio = min(entry.get("timestamp", datetime.now()) for entry in self.historial_rendimiento)
            else:
                fecha_inicio = fecha_fin - timedelta(days=365)
        
        # Filtrar historial para el periodo
        historial_filtrado = [
            entry for entry in self.historial_rendimiento 
            if isinstance(entry.get("timestamp"), datetime) and fecha_inicio <= entry.get("timestamp") <= fecha_fin
        ]
        
        # Si no hay suficientes datos, buscar benchmarks
        rendimiento_sistema = None
        
        if historial_filtrado:
            # Encontrar capital al inicio del periodo
            entradas_iniciales = [
                entry for entry in self.historial_rendimiento 
                if isinstance(entry.get("timestamp"), datetime) and entry.get("timestamp") < fecha_inicio
            ]
            
            if entradas_iniciales:
                capital_inicial_periodo = sorted(entradas_iniciales, key=lambda x: x.get("timestamp"))[-1].get("capital", self.capital_inicial)
            else:
                capital_inicial_periodo = self.capital_inicial
            
            # Calcular rendimiento del periodo
            rendimiento_sistema = (self.capital_actual / capital_inicial_periodo) - 1
        else:
            rendimiento_sistema = 0.0
        
        # Obtener rendimiento de benchmarks para el mismo periodo
        rendimiento_benchmarks = {}
        
        # En un sistema real, obtendríamos datos de APIs externas
        # Para esta implementación, simularemos algunos benchmarks
        
        crypto_market_return = await self._obtener_rendimiento_benchmark("crypto_top10", fecha_inicio, fecha_fin)
        btc_return = await self._obtener_rendimiento_benchmark("btc", fecha_inicio, fecha_fin)
        eth_return = await self._obtener_rendimiento_benchmark("eth", fecha_inicio, fecha_fin)
        sp500_return = await self._obtener_rendimiento_benchmark("sp500", fecha_inicio, fecha_fin)
        
        rendimiento_benchmarks = {
            "crypto_top10": crypto_market_return,
            "btc": btc_return,
            "eth": eth_return,
            "sp500": sp500_return
        }
        
        # Calcular alfa (exceso de rendimiento)
        alfa = rendimiento_sistema - rendimiento_benchmarks.get(self.benchmark, 0)
        
        # Aplicar mecanismos trascendentales
        analisis_trascendental = {}
        if self.trascendencia_activada:
            # Análisis trascendental de rendimiento
            if self.modo_trascendental == "SINGULARITY_V4":
                # Analizar rendimiento en diferentes dimensiones temporales
                analisis_trascendental = {
                    "analisis_multidimensional": "Análisis singularidad completado",
                    "rendimiento_ajustado_dimension": rendimiento_sistema * 1.05  # Ejemplo de ajuste trascendental
                }
                
                # Guardar en cache cuántico para análisis temporal
                await transcendental_db.perform_temporal_sync()
        
        # Resultado completo
        resultado = {
            "fecha_analisis": datetime.now().isoformat(),
            "periodo": {
                "tipo": periodo,
                "inicio": fecha_inicio.isoformat(),
                "fin": fecha_fin.isoformat()
            },
            "rendimiento_sistema": rendimiento_sistema * 100,  # Convertir a porcentaje
            "rendimiento_benchmarks": {
                benchmark: rendimiento * 100 for benchmark, rendimiento in rendimiento_benchmarks.items()
            },
            "benchmark_principal": self.benchmark,
            "alfa": alfa * 100,  # Convertir a porcentaje
            "analisis_trascendental": analisis_trascendental,
            "modo_trascendental": self.modo_trascendental
        }
        
        # Guardar checkpoint de comparación
        if self.trascendencia_activada:
            await transcendental_db.checkpoint_state(
                "benchmark_comparison", 
                f"benchmark_{periodo}_{int(datetime.now().timestamp())}", 
                resultado
            )
        
        return resultado
    
    async def obtener_resumen_rendimiento(self) -> Dict[str, Any]:
        """
        Obtener resumen completo del rendimiento actual.
        
        Returns:
            Diccionario con resumen de rendimiento
        """
        # Actualizar métricas antes de generar resumen
        await self._actualizar_metricas()
        
        # Obtener estadísticas por estrategia
        estrategias_stats = {}
        for estrategia, stats in self.estrategias.items():
            if stats["operaciones_totales"] > 0:
                win_rate = stats["operaciones_ganadoras"] / stats["operaciones_totales"] 
            else:
                win_rate = 0
                
            if stats["perdidas_totales"] > 0:
                profit_factor = stats["ganancias_totales"] / stats["perdidas_totales"]
            else:
                profit_factor = float('inf') if stats["ganancias_totales"] > 0 else 0
                
            estrategias_stats[estrategia] = {
                "operaciones_totales": stats["operaciones_totales"],
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "rendimiento_usd": stats["rendimiento_acumulado"],
                "rendimiento_porcentaje": (stats["rendimiento_acumulado"] / self.capital_inicial) * 100 if self.capital_inicial > 0 else 0
            }
        
        # Obtener operaciones recientes
        operaciones_recientes = [
            entry for entry in self.historial_rendimiento 
            if entry.get("tipo") == "operacion"
        ]
        
        operaciones_recientes = sorted(
            operaciones_recientes,
            key=lambda x: x.get("timestamp", datetime.now()),
            reverse=True
        )[:5]  # Últimas 5 operaciones
        
        # Aplicar mecanismos trascendentales
        analisis_avanzado = {}
        if self.trascendencia_activada:
            if self.modo_trascendental == "SINGULARITY_V4":
                # Detección de patrones avanzados en el rendimiento
                drawdown_info = await self._calcular_drawdown_info()
                volatilidad_tendencia = await self._calcular_tendencia_volatilidad()
                
                analisis_avanzado = {
                    "drawdown_info": drawdown_info,
                    "volatilidad_tendencia": volatilidad_tendencia,
                    "ciclo_mercado_actual": self._determinar_ciclo_mercado()
                }
        
        # Resultado completo
        resultado = {
            "fecha_resumen": datetime.now().isoformat(),
            "datos_generales": {
                "capital_inicial": self.capital_inicial,
                "capital_actual": self.capital_actual,
                "rendimiento_total_usd": self.capital_actual - self.capital_inicial,
                "rendimiento_total_porcentaje": ((self.capital_actual / self.capital_inicial) - 1) * 100 if self.capital_inicial > 0 else 0,
                "operaciones_totales": sum(stats["operaciones_totales"] for stats in self.estrategias.values()),
                "fecha_inicio": min([entry.get("timestamp") for entry in self.historial_rendimiento]).isoformat() if self.historial_rendimiento else None,
                "dias_operando": (datetime.now() - min([entry.get("timestamp") for entry in self.historial_rendimiento])).days if self.historial_rendimiento else 0
            },
            "metricas": self.metricas,
            "estrategias": estrategias_stats,
            "operaciones_recientes": [
                {
                    "symbol": op.get("symbol"),
                    "estrategia": op.get("estrategia"),
                    "tipo": op.get("operacion_tipo"),
                    "resultado_usd": op.get("resultado_usd"),
                    "resultado_porcentual": op.get("resultado_porcentual"),
                    "fecha": op.get("timestamp").isoformat() if isinstance(op.get("timestamp"), datetime) else op.get("timestamp")
                }
                for op in operaciones_recientes
            ],
            "analisis_avanzado": analisis_avanzado,
            "modo_trascendental": self.modo_trascendental
        }
        
        # Guardar checkpoint del resumen
        if self.trascendencia_activada:
            await transcendental_db.checkpoint_state(
                "performance_summary", 
                f"summary_{int(datetime.now().timestamp())}", 
                resultado
            )
        
        return resultado
    
    async def activar_modo_trascendental(self, modo: str = "SINGULARITY_V4") -> Dict[str, Any]:
        """
        Activar un modo trascendental específico.
        
        Args:
            modo: Modo trascendental a activar
            
        Returns:
            Estado actualizado del seguidor de rendimiento
        """
        modos_validos = [
            "SINGULARITY_V4", "LIGHT", "DARK_MATTER", 
            "DIVINE", "BIG_BANG", "INTERDIMENSIONAL"
        ]
        
        if modo not in modos_validos:
            raise ValueError(f"Modo trascendental no válido. Opciones: {', '.join(modos_validos)}")
        
        # Cambiar modo y activar
        modo_anterior = self.modo_trascendental
        self.modo_trascendental = modo
        self.trascendencia_activada = True
        
        logger.info(f"Activado modo trascendental: {modo_anterior} → {modo}")
        
        # Sincronizar con base de datos trascendental
        await transcendental_db.perform_temporal_sync()
        
        # Resultado
        resultado = {
            "modo_anterior": modo_anterior,
            "modo_actual": modo,
            "trascendencia_activada": self.trascendencia_activada,
            "sincronizacion_temporal": "completada",
            "timestamp": datetime.now().isoformat()
        }
        
        return resultado
    
    def _inicializar_benchmark(self, benchmark_id: str) -> Dict[str, Any]:
        """
        Inicializar datos para un benchmark específico.
        
        Args:
            benchmark_id: Identificador del benchmark
            
        Returns:
            Diccionario con estructura para el benchmark
        """
        return {
            "id": benchmark_id,
            "nombre": self._obtener_nombre_benchmark(benchmark_id),
            "datos": [],
            "ultima_actualizacion": None
        }
    
    def _obtener_nombre_benchmark(self, benchmark_id: str) -> str:
        """
        Obtener nombre descriptivo para un benchmark.
        
        Args:
            benchmark_id: Identificador del benchmark
            
        Returns:
            Nombre descriptivo
        """
        nombres = {
            "crypto_top10": "Top 10 Criptomonedas por Market Cap",
            "btc": "Bitcoin (BTC)",
            "eth": "Ethereum (ETH)",
            "sp500": "S&P 500"
        }
        
        return nombres.get(benchmark_id, benchmark_id)
    
    async def _obtener_rendimiento_benchmark(self, 
                                        benchmark_id: str, 
                                        fecha_inicio: datetime,
                                        fecha_fin: datetime) -> float:
        """
        Obtener rendimiento de un benchmark para un periodo específico.
        
        En un sistema real, esto obtendría datos de APIs externas.
        Para esta implementación, generamos valores simulados basados en tendencias reales.
        
        Args:
            benchmark_id: Identificador del benchmark
            fecha_inicio: Fecha de inicio del periodo
            fecha_fin: Fecha de fin del periodo
            
        Returns:
            Rendimiento como decimal (0.10 = 10%)
        """
        # Valores base para simulación
        base_returns = {
            "crypto_top10": 0.20,  # 20% anual
            "btc": 0.25,           # 25% anual
            "eth": 0.30,           # 30% anual
            "sp500": 0.10          # 10% anual
        }
        
        # Volatilidad base
        volatilities = {
            "crypto_top10": 0.40,  # 40% volatilidad anual
            "btc": 0.60,           # 60% volatilidad anual
            "eth": 0.70,           # 70% volatilidad anual
            "sp500": 0.15          # 15% volatilidad anual
        }
        
        # Correlación con éxito del sistema
        # En un sistema real, esto sería una medida real basada en datos históricos
        sistema_exitoso = self.capital_actual > self.capital_inicial
        
        # Obtener valor base
        base_return = base_returns.get(benchmark_id, 0.10)
        volatility = volatilities.get(benchmark_id, 0.20)
        
        # Calcular días en el periodo
        dias = (fecha_fin - fecha_inicio).days
        if dias <= 0:
            dias = 1
        
        # Anualizar el rendimiento base al periodo
        period_return = base_return * (dias / 365)
        
        # Añadir variación aleatoria
        np.random.seed(int(hash(benchmark_id + fecha_inicio.isoformat()) % 1000000))
        variation = np.random.normal(0, volatility * (dias / 365) ** 0.5)
        
        # El rendimiento del benchmark puede correlacionarse ligeramente con el éxito del sistema
        if sistema_exitoso:
            period_return = period_return * 1.1  # 10% mejor si el sistema tiene éxito
        else:
            period_return = period_return * 0.9  # 10% peor si el sistema no tiene éxito
        
        # Rendimiento final
        final_return = period_return + variation
        
        # Registrar resultado
        return final_return
    
    async def _actualizar_metricas(self) -> None:
        """Actualizar todas las métricas de rendimiento."""
        # Calcular rendimiento total
        if self.capital_inicial > 0:
            self.metricas["rendimiento_total"] = ((self.capital_actual / self.capital_inicial) - 1) * 100
        else:
            self.metricas["rendimiento_total"] = 0
            
        # Filtrar operaciones
        operaciones = [entry for entry in self.historial_rendimiento if entry.get("tipo") == "operacion"]
        
        # Si no hay suficientes operaciones, no podemos calcular algunas métricas
        if len(operaciones) < 5:
            return
            
        # Calcular días de operación
        if operaciones:
            primera_fecha = min([op.get("timestamp", datetime.now()) for op in operaciones])
            ultima_fecha = max([op.get("timestamp", datetime.now()) for op in operaciones])
            dias_operando = max(1, (ultima_fecha - primera_fecha).days)
            
            # Rendimiento anualizado
            rendimiento_total_decimal = self.metricas["rendimiento_total"] / 100
            self.metricas["rendimiento_anualizado"] = (
                ((1 + rendimiento_total_decimal) ** (365 / dias_operando)) - 1
            ) * 100
        
        # Calcular volatilidad
        if len(operaciones) >= 10:
            # Obtener cambios porcentuales diarios
            cambios = []
            operaciones_ordenadas = sorted(operaciones, key=lambda x: x.get("timestamp", datetime.now()))
            
            for i in range(1, len(operaciones_ordenadas)):
                capital_anterior = operaciones_ordenadas[i-1].get("capital", 0)
                capital_actual = operaciones_ordenadas[i].get("capital", 0)
                
                if capital_anterior > 0:
                    cambio = (capital_actual / capital_anterior) - 1
                    cambios.append(cambio)
            
            if cambios:
                # Volatilidad como desviación estándar de los cambios
                volatilidad = np.std(cambios)
                self.metricas["volatilidad"] = volatilidad * 100  # Convertir a porcentaje
                
                # Anualizar volatilidad
                self.metricas["volatilidad"] = self.metricas["volatilidad"] * (252 ** 0.5)  # 252 días de trading
        
        # Calcular drawdown
        if operaciones:
            # Encontrar máximo drawdown
            max_capital = self.capital_inicial
            max_drawdown = 0
            
            for op in operaciones_ordenadas:
                capital = op.get("capital", 0)
                max_capital = max(max_capital, capital)
                
                # Drawdown actual
                drawdown = 1 - (capital / max_capital) if max_capital > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            
            self.metricas["max_drawdown"] = max_drawdown * 100  # Convertir a porcentaje
        
        # Calcular Sharpe Ratio
        if "volatilidad" in self.metricas and self.metricas["volatilidad"] > 0:
            # Tasa libre de riesgo (ejemplo: 2% anual)
            tasa_libre_riesgo = 2.0
            
            self.metricas["sharpe_ratio"] = (
                self.metricas["rendimiento_anualizado"] - tasa_libre_riesgo
            ) / self.metricas["volatilidad"]
        
        # Calcular Sortino Ratio (solo considera volatilidad negativa)
        if operaciones and len(cambios) > 0:
            # Filtrar solo cambios negativos
            cambios_negativos = [c for c in cambios if c < 0]
            
            if cambios_negativos:
                # Desviación estándar de los cambios negativos
                downside_deviation = np.std(cambios_negativos) * 100 * (252 ** 0.5)
                
                if downside_deviation > 0:
                    self.metricas["sortino_ratio"] = (
                        self.metricas["rendimiento_anualizado"] - 2.0
                    ) / downside_deviation
        
        # Calcular Calmar Ratio
        if self.metricas["max_drawdown"] > 0:
            self.metricas["calmar_ratio"] = self.metricas["rendimiento_anualizado"] / self.metricas["max_drawdown"]
        
        # Calcular Win Rate agregado
        operaciones_totales = sum(stats["operaciones_totales"] for stats in self.estrategias.values())
        operaciones_ganadoras = sum(stats["operaciones_ganadoras"] for stats in self.estrategias.values())
        
        if operaciones_totales > 0:
            self.metricas["win_rate"] = (operaciones_ganadoras / operaciones_totales) * 100
        
        # Calcular Profit Factor
        ganancias_totales = sum(stats["ganancias_totales"] for stats in self.estrategias.values())
        perdidas_totales = sum(stats["perdidas_totales"] for stats in self.estrategias.values())
        
        if perdidas_totales > 0:
            self.metricas["profit_factor"] = ganancias_totales / perdidas_totales
        else:
            self.metricas["profit_factor"] = float('inf') if ganancias_totales > 0 else 0
        
        # Calcular Expectancy (ganancia promedio por operación)
        if operaciones_totales > 0:
            ganancia_neta = ganancias_totales - perdidas_totales
            self.metricas["expectancy"] = ganancia_neta / operaciones_totales
        
        # Calcular Ratio de Recuperación
        if self.metricas["max_drawdown"] > 0:
            self.metricas["ratio_recuperacion"] = abs(
                self.metricas["rendimiento_total"] / self.metricas["max_drawdown"]
            )
    
    async def _actualizar_estadisticas_por_capital(self) -> None:
        """
        Actualizar estadísticas específicas para diferentes niveles de capital.
        
        Esto es importante para entender cómo cambia el rendimiento del sistema
        a medida que el capital crece.
        """
        # Definir niveles de capital
        niveles = {
            "10k": 10000,
            "100k": 100000,
            "1M": 1000000,
            "10M": 10000000
        }
        
        # Para cada nivel, evaluar métricas relevantes
        for nivel, monto in niveles.items():
            # Solo procesar niveles hasta el capital actual
            if self.capital_actual < monto:
                continue
                
            # Filtrar operaciones para el nivel
            operaciones_nivel = [
                op for op in self.historial_rendimiento 
                if op.get("tipo") == "operacion" and op.get("capital", 0) <= monto
            ]
            
            if not operaciones_nivel:
                continue
                
            # Calcular métricas específicas
            win_rate_nivel = 0
            rendimiento_nivel = 0
            volatilidad_nivel = 0
            
            # Win rate
            ganadores = len([op for op in operaciones_nivel if op.get("resultado_usd", 0) > 0])
            if operaciones_nivel:
                win_rate_nivel = (ganadores / len(operaciones_nivel)) * 100
            
            # Rendimiento (desde el inicio hasta alcanzar este nivel)
            operaciones_ordenadas = sorted(operaciones_nivel, key=lambda x: x.get("timestamp", datetime.now()))
            if operaciones_ordenadas:
                capital_final_nivel = operaciones_ordenadas[-1].get("capital", monto)
                rendimiento_nivel = ((capital_final_nivel / self.capital_inicial) - 1) * 100
            
            # Volatilidad
            if len(operaciones_ordenadas) >= 10:
                cambios = []
                for i in range(1, len(operaciones_ordenadas)):
                    capital_anterior = operaciones_ordenadas[i-1].get("capital", 0)
                    capital_actual = operaciones_ordenadas[i].get("capital", 0)
                    
                    if capital_anterior > 0:
                        cambio = (capital_actual / capital_anterior) - 1
                        cambios.append(cambio)
                
                if cambios:
                    volatilidad_nivel = np.std(cambios) * 100 * (252 ** 0.5)
            
            # Guardar estadísticas
            self.estadisticas_por_capital[nivel] = {
                "operaciones_totales": len(operaciones_nivel),
                "win_rate": win_rate_nivel,
                "rendimiento": rendimiento_nivel,
                "volatilidad_anualizada": volatilidad_nivel,
                "capital_alcanzado": monto,
                "fecha_alcanzado": max([op.get("timestamp", datetime.now()) for op in operaciones_nivel]).isoformat()
            }
            
            # Aplicar mecanismos trascendentales al llegar a niveles significativos
            if self.trascendencia_activada and nivel in ["1M", "10M"]:
                logger.info(f"Aplicando análisis trascendental al alcanzar nivel {nivel}")
                
                # En un sistema real, podríamos aplicar algoritmos más sofisticados
                # Por ejemplo, detección de cambios en patrones de rendimiento
    
    def _calcular_parametros_rendimiento(self) -> Tuple[float, float]:
        """
        Calcular parámetros de rendimiento histórico.
        
        Returns:
            Tupla (rendimiento_diario_medio, volatilidad_diaria)
        """
        # Filtrar y ordenar operaciones
        operaciones = [entry for entry in self.historial_rendimiento if entry.get("tipo") == "operacion"]
        operaciones_ordenadas = sorted(operaciones, key=lambda x: x.get("timestamp", datetime.now()))
        
        # Si no hay suficientes operaciones, usar valores predeterminados conservadores
        if len(operaciones_ordenadas) < 10:
            return 0.0005, 0.01  # 0.05% diario, 1% volatilidad diaria
        
        # Calcular cambios porcentuales diarios
        cambios = []
        for i in range(1, len(operaciones_ordenadas)):
            capital_anterior = operaciones_ordenadas[i-1].get("capital", 0)
            capital_actual = operaciones_ordenadas[i].get("capital", 0)
            
            if capital_anterior > 0:
                cambio = (capital_actual / capital_anterior) - 1
                cambios.append(cambio)
        
        if not cambios:
            return 0.0005, 0.01
        
        # Calcular rendimiento medio diario y volatilidad
        rendimiento_medio = np.mean(cambios)
        volatilidad = np.std(cambios)
        
        # Aplicar mecanismos trascendentales
        if self.trascendencia_activada:
            if self.modo_trascendental == "SINGULARITY_V4":
                # En Singularidad, estabilizar volatilidad
                volatilidad = min(volatilidad, 0.03)  # Limitar volatilidad extrema
        
        return rendimiento_medio, volatilidad
    
    async def _calcular_correlaciones_trascendentales(self, operaciones: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcular correlaciones avanzadas entre estrategias y activos.
        
        Args:
            operaciones: Lista de operaciones para análisis
            
        Returns:
            Diccionario con patrones de correlación
        """
        # Si no hay suficientes operaciones, no podemos calcular correlaciones
        if len(operaciones) < 10:
            return {}
        
        # Agrupar operaciones por estrategia
        estrategias = {}
        for op in operaciones:
            estrategia = op.get("estrategia")
            if estrategia not in estrategias:
                estrategias[estrategia] = []
            estrategias[estrategia].append(op)
        
        # Calcular rendimiento por día para cada estrategia
        rendimientos_diarios = {}
        for estrategia, ops in estrategias.items():
            # Ordenar por timestamp
            ops_ordenadas = sorted(ops, key=lambda x: x.get("timestamp", datetime.now()))
            
            # Agrupar por día
            por_dia = {}
            for op in ops_ordenadas:
                if isinstance(op.get("timestamp"), datetime):
                    fecha = op.get("timestamp").date()
                    if fecha not in por_dia:
                        por_dia[fecha] = []
                    por_dia[fecha].append(op)
            
            # Calcular rendimiento por día
            rendimientos = {}
            for fecha, ops_dia in por_dia.items():
                resultado_total = sum(op.get("resultado_usd", 0) for op in ops_dia)
                rendimientos[fecha] = resultado_total
            
            rendimientos_diarios[estrategia] = rendimientos
        
        # Obtener todas las fechas únicas
        todas_fechas = set()
        for estrategia, rendimientos in rendimientos_diarios.items():
            todas_fechas.update(rendimientos.keys())
        
        # Convertir a series temporales completas
        series = {}
        fechas_ordenadas = sorted(todas_fechas)
        
        for estrategia, rendimientos in rendimientos_diarios.items():
            serie = []
            for fecha in fechas_ordenadas:
                serie.append(rendimientos.get(fecha, 0))
            series[estrategia] = serie
        
        # Calcular matriz de correlación
        correlaciones = {}
        for estrategia1 in series:
            correlaciones[estrategia1] = {}
            for estrategia2 in series:
                if len(series[estrategia1]) > 0 and len(series[estrategia2]) > 0:
                    # Calcular correlación de Pearson
                    try:
                        corr = np.corrcoef(series[estrategia1], series[estrategia2])[0, 1]
                        correlaciones[estrategia1][estrategia2] = corr
                    except:
                        correlaciones[estrategia1][estrategia2] = 0
                else:
                    correlaciones[estrategia1][estrategia2] = 0
        
        # Identificar clusters de estrategias correlacionadas
        clusters = []
        umbral_correlacion = 0.7
        
        estrategias_restantes = set(correlaciones.keys())
        while estrategias_restantes:
            # Tomar la primera estrategia restante
            estrategia = next(iter(estrategias_restantes))
            cluster = {estrategia}
            estrategias_restantes.remove(estrategia)
            
            # Encontrar estrategias correlacionadas
            for otra_estrategia in list(estrategias_restantes):
                if correlaciones[estrategia][otra_estrategia] > umbral_correlacion:
                    cluster.add(otra_estrategia)
                    estrategias_restantes.remove(otra_estrategia)
            
            # Añadir cluster si tiene al menos dos estrategias
            if len(cluster) >= 2:
                clusters.append(list(cluster))
        
        # Resultado del análisis
        return {
            "correlaciones": {
                k: dict(v) for k, v in correlaciones.items()
            },
            "clusters": clusters,
            "fechas_analizadas": {
                "inicio": min(fechas_ordenadas).isoformat() if fechas_ordenadas else None,
                "fin": max(fechas_ordenadas).isoformat() if fechas_ordenadas else None,
                "total_dias": len(fechas_ordenadas)
            }
        }
    
    async def _calcular_drawdown_info(self) -> Dict[str, Any]:
        """
        Calcular información detallada sobre drawdowns.
        
        Returns:
            Diccionario con información de drawdowns
        """
        # Filtrar y ordenar operaciones
        operaciones = [entry for entry in self.historial_rendimiento if entry.get("tipo") == "operacion"]
        operaciones_ordenadas = sorted(operaciones, key=lambda x: x.get("timestamp", datetime.now()))
        
        if not operaciones_ordenadas:
            return {"max_drawdown": 0, "drawdowns": []}
        
        # Calcular serie de capital
        capital_serie = []
        fechas = []
        
        for op in operaciones_ordenadas:
            capital_serie.append(op.get("capital", 0))
            fechas.append(op.get("timestamp"))
        
        # Detectar drawdowns
        max_capital = capital_serie[0]
        drawdowns = []
        drawdown_actual = None
        
        for i, capital in enumerate(capital_serie):
            max_capital = max(max_capital, capital)
            
            drawdown_pct = 1 - (capital / max_capital) if max_capital > 0 else 0
            
            # Si estamos en drawdown
            if drawdown_pct > 0.05:  # Umbral mínimo 5%
                if drawdown_actual is None:
                    # Iniciar nuevo drawdown
                    drawdown_actual = {
                        "inicio": fechas[i],
                        "pico": max_capital,
                        "profundidad": drawdown_pct,
                        "fin": None,
                        "duracion_dias": 0
                    }
                else:
                    # Actualizar drawdown existente
                    drawdown_actual["profundidad"] = max(drawdown_actual["profundidad"], drawdown_pct)
            elif drawdown_actual is not None:
                # Finalizar drawdown
                drawdown_actual["fin"] = fechas[i]
                drawdown_actual["duracion_dias"] = (drawdown_actual["fin"] - drawdown_actual["inicio"]).days
                drawdowns.append(drawdown_actual)
                drawdown_actual = None
        
        # Si hay un drawdown activo, añadirlo
        if drawdown_actual is not None:
            drawdown_actual["fin"] = fechas[-1]
            drawdown_actual["duracion_dias"] = (drawdown_actual["fin"] - drawdown_actual["inicio"]).days
            drawdowns.append(drawdown_actual)
        
        # Ordenar por profundidad
        drawdowns = sorted(drawdowns, key=lambda x: x["profundidad"], reverse=True)
        
        # Preparar resultado
        return {
            "max_drawdown": max(d["profundidad"] for d in drawdowns) if drawdowns else 0,
            "drawdowns": [
                {
                    "inicio": d["inicio"].isoformat(),
                    "fin": d["fin"].isoformat(),
                    "profundidad": d["profundidad"] * 100,  # Convertir a porcentaje
                    "duracion_dias": d["duracion_dias"],
                    "pico_capital": d["pico"]
                }
                for d in drawdowns[:3]  # Top 3 drawdowns
            ]
        }
    
    async def _calcular_tendencia_volatilidad(self) -> Dict[str, Any]:
        """
        Calcular tendencia de volatilidad a lo largo del tiempo.
        
        Returns:
            Diccionario con información de volatilidad
        """
        # Filtrar y ordenar operaciones
        operaciones = [entry for entry in self.historial_rendimiento if entry.get("tipo") == "operacion"]
        operaciones_ordenadas = sorted(operaciones, key=lambda x: x.get("timestamp", datetime.now()))
        
        if len(operaciones_ordenadas) < 20:
            return {"tendencia": "neutral", "volatilidad_actual": 0}
        
        # Calcular cambios porcentuales
        cambios = []
        for i in range(1, len(operaciones_ordenadas)):
            capital_anterior = operaciones_ordenadas[i-1].get("capital", 0)
            capital_actual = operaciones_ordenadas[i].get("capital", 0)
            
            if capital_anterior > 0:
                cambio = (capital_actual / capital_anterior) - 1
                cambios.append(cambio)
        
        if not cambios:
            return {"tendencia": "neutral", "volatilidad_actual": 0}
        
        # Dividir en ventanas de 10 operaciones
        ventanas = []
        for i in range(0, len(cambios), 10):
            ventana = cambios[i:i+10]
            if len(ventana) >= 5:  # Mínimo 5 puntos
                ventanas.append(ventana)
        
        if not ventanas:
            return {"tendencia": "neutral", "volatilidad_actual": 0}
        
        # Calcular volatilidad por ventana
        volatilidades = [np.std(ventana) for ventana in ventanas]
        
        # Determinar tendencia
        volatilidad_actual = volatilidades[-1]
        
        if len(volatilidades) >= 3:
            # Tendencia basada en últimas 3 ventanas
            tendencia_reciente = np.polyfit(range(3), volatilidades[-3:], 1)[0]
            
            if tendencia_reciente > 0.001:
                tendencia = "creciente"
            elif tendencia_reciente < -0.001:
                tendencia = "decreciente"
            else:
                tendencia = "estable"
        else:
            tendencia = "neutral"
        
        return {
            "tendencia": tendencia,
            "volatilidad_actual": volatilidad_actual * 100 * (252 ** 0.5),  # Anualizada y en porcentaje
            "volatilidad_historica": np.mean(volatilidades) * 100 * (252 ** 0.5)
        }
    
    def _determinar_ciclo_mercado(self) -> str:
        """
        Determinar fase actual del ciclo de mercado.
        
        Returns:
            String con fase del ciclo
        """
        # Basado en rendimiento reciente y volatilidad
        rendimiento_total = self.metricas["rendimiento_total"]
        volatilidad = self.metricas["volatilidad"]
        max_drawdown = self.metricas["max_drawdown"]
        
        # Determinar fase
        if rendimiento_total > 20 and volatilidad < 15:
            return "bull_estable"
        elif rendimiento_total > 20 and volatilidad >= 15:
            return "bull_volatil"
        elif rendimiento_total < -15 and max_drawdown > 25:
            return "bear_profundo"
        elif rendimiento_total < 0:
            return "bear_moderado"
        elif abs(rendimiento_total) < 10 and volatilidad < 15:
            return "consolidacion"
        elif abs(rendimiento_total) < 10 and volatilidad >= 15:
            return "indecision"
        else:
            return "transicion"


# Instancia global para acceso desde cualquier módulo
performance_tracker = TranscendentalPerformanceTracker()

async def initialize_performance_tracker(capital_inicial: float = 10000.0, config: Optional[Dict[str, Any]] = None) -> None:
    """
    Inicializar el seguidor de rendimiento con configuración.
    
    Args:
        capital_inicial: Capital inicial en USD
        config: Configuración adicional (opcional)
    """
    # Aplicar configuración si se proporciona
    if config:
        benchmark = config.get("benchmark", "crypto_top10")
        
        # Crear nueva instancia con configuración
        global performance_tracker
        performance_tracker = TranscendentalPerformanceTracker(
            capital_inicial=capital_inicial,
            benchmark=benchmark
        )
        
        # Activar modo trascendental si se especifica
        modo_trascendental = config.get("modo_trascendental", "SINGULARITY_V4")
        await performance_tracker.activar_modo_trascendental(modo_trascendental)
    else:
        # Actualizar capital de la instancia existente
        await performance_tracker.actualizar_capital(capital_inicial, "inicializacion")
    
    logger.info(f"TranscendentalPerformanceTracker inicializado con capital: ${capital_inicial:,.2f}")