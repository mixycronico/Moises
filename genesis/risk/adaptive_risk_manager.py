"""
Gestor de Riesgo Adaptativo para el Sistema Genesis.

Este módulo implementa un sistema avanzado de gestión de riesgos con capacidades
transcendentales que se adapta dinámicamente al crecimiento del capital y las
condiciones del mercado para mantener la resiliencia y eficiencia del sistema.

Características principales:
- Adaptación dinámica del tamaño de posición según crecimiento del capital
- Ajuste automático de stop-loss y take-profit basado en volatilidad
- Protección contra drawdown con niveles de protección progresivos
- Correlación adaptativa para evitar sobreexposición a activos correlacionados
- Integración con todos los mecanismos transcendentales de Genesis
"""

import logging
import asyncio
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
import json

from genesis.db.transcendental_database import db
from genesis.db.models.crypto_classifier_models import CryptoMetrics, CryptoScores

# Configuración de logging
logger = logging.getLogger(__name__)

class AdaptiveRiskManager:
    """
    Gestor de riesgo avanzado con capacidades adaptativas transcendentales.
    
    Este sistema ajusta dinámicamente los parámetros de gestión de riesgos
    basándose en el crecimiento del capital, condiciones del mercado y
    rendimiento histórico del sistema.
    """
    
    def __init__(self, 
                capital_inicial: float = 10000.0, 
                max_drawdown_permitido: float = 0.15,
                volatilidad_base: float = 0.02):
        """
        Inicializar el gestor de riesgo adaptativo.
        
        Args:
            capital_inicial: Capital inicial del sistema en USD
            max_drawdown_permitido: Máximo drawdown permitido (0.15 = 15%)
            volatilidad_base: Volatilidad base para cálculos (0.02 = 2%)
        """
        self.capital_actual = capital_inicial
        self.capital_inicial = capital_inicial
        self.max_drawdown_permitido = max_drawdown_permitido
        self.volatilidad_base = volatilidad_base
        
        # Historial de capital para tracking de drawdown
        self.historial_capital = [{
            "timestamp": datetime.now().timestamp(),
            "capital": capital_inicial,
            "cambio_porcentual": 0.0
        }]
        
        # Métricas actuales
        self.max_capital_historico = capital_inicial
        self.drawdown_actual = 0.0
        self.win_rate = 0.5  # Valor inicial conservador
        self.ratio_beneficio_riesgo = 1.5  # Valor inicial conservador
        
        # Registro histórico de operaciones
        self.operaciones = []
        
        # Estado de protección
        self.nivel_proteccion = "NORMAL"  # NORMAL, CAUTIOUS, DEFENSIVE, ULTRADEFENSIVE
        
        # Correlaciones entre activos
        self.matriz_correlacion = {}
        
        # Estadísticas adicionales
        self.stats = {
            "operaciones_totales": 0,
            "operaciones_ganadoras": 0,
            "operaciones_perdedoras": 0,
            "mayor_ganancia": 0.0,
            "mayor_perdida": 0.0,
            "rendimiento_total": 0.0,
            "volatilidad_portfolio": volatilidad_base,
            "correlacion_promedio": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "kelly_criterion": 0.0
        }
        
        logger.info(f"AdaptiveRiskManager inicializado con capital: ${capital_inicial:,.2f}")
    
    async def calcular_tamano_posicion(self, 
                                     symbol: str, 
                                     score: float, 
                                     volatilidad: Optional[float] = None) -> Dict[str, Any]:
        """
        Calcular tamaño óptimo de posición con ajuste adaptativo.
        
        Esta función determina el tamaño óptimo de posición basado en múltiples
        factores adaptativos como la puntuación del activo, el crecimiento del
        capital, la volatilidad actual y el historial de rendimiento.
        
        Args:
            symbol: Símbolo del activo
            score: Puntuación del clasificador (0-1)
            volatilidad: Volatilidad opcional del activo (si no se proporciona se busca)
            
        Returns:
            Diccionario con información sobre el tamaño de posición y parámetros
        """
        # Obtener volatilidad si no se proporcionó
        if volatilidad is None:
            volatilidad = await self._obtener_volatilidad(symbol)
        
        # Obtener métricas y puntuaciones del activo
        metrics, scores = await db.get_crypto_metrics_with_scores(symbol)
        
        # Parámetros base
        riesgo_capital_base = 0.01  # 1% del capital
        
        # Ajuste por tamaño de capital (reducir riesgo al crecer)
        factor_capital = 1.0
        if self.capital_actual > self.capital_inicial * 2:
            # Reducir el riesgo progresivamente al crecer el capital
            factor_capital = 1.0 / np.log10(self.capital_actual / self.capital_inicial + 1)
        
        # Ajuste por drawdown actual (reducir riesgo durante drawdowns)
        factor_drawdown = 1.0 - (self.drawdown_actual / self.max_drawdown_permitido) * 0.8
        factor_drawdown = max(0.2, min(1.0, factor_drawdown))
        
        # Ajuste por win rate y ratio beneficio-riesgo (Kelly Criterion adaptado)
        # Kelly = win_rate - (1 - win_rate) / ratio_beneficio_riesgo
        kelly = self.win_rate - (1 - self.win_rate) / self.ratio_beneficio_riesgo
        factor_kelly = max(0.1, min(0.5, kelly * 0.5))  # Kelly fraccional (50%)
        
        # Ajuste por score del activo
        factor_score = 0.5 + (score * 0.5)  # Entre 0.5 y 1.0
        
        # Ajuste por volatilidad (inversamente proporcional)
        factor_volatilidad = min(1.0, self.volatilidad_base / max(self.volatilidad_base, volatilidad))
        
        # Ajuste por nivel de protección actual
        factor_proteccion = 1.0
        if self.nivel_proteccion == "CAUTIOUS":
            factor_proteccion = 0.7
        elif self.nivel_proteccion == "DEFENSIVE":
            factor_proteccion = 0.5
        elif self.nivel_proteccion == "ULTRADEFENSIVE":
            factor_proteccion = 0.25
        
        # Calcular riesgo por operación ajustado
        riesgo_operacion = (
            riesgo_capital_base * 
            factor_capital * 
            factor_drawdown * 
            factor_kelly * 
            factor_score *
            factor_volatilidad *
            factor_proteccion
        )
        
        # Calcular tamaño de posición en USD
        tamano_posicion_usd = self.capital_actual * riesgo_operacion
        
        # Calcular precio de entrada y salida
        precio_entrada = metrics.price if metrics and metrics.price else 0
        stop_loss_percent = -0.05 - (score * 0.05)  # Entre -5% y -10%
        take_profit_percent = 0.1 + (score * 0.15)  # Entre 10% y 25%
        
        # Ajustar stop-loss y take-profit según volatilidad
        stop_loss_atr = -2.0 - (score * 2.0)  # Entre -2 y -4 ATRs
        take_profit_atr = 4.0 + (score * 4.0)  # Entre 4 y 8 ATRs
        
        if metrics and metrics.atr:
            stop_loss_percent = min(stop_loss_percent, stop_loss_atr * metrics.atr_percent)
            take_profit_percent = max(take_profit_percent, take_profit_atr * metrics.atr_percent)
        
        # Calcular precios de salida
        stop_loss_precio = precio_entrada * (1 + stop_loss_percent) if precio_entrada else 0
        take_profit_precio = precio_entrada * (1 + take_profit_percent) if precio_entrada else 0
        
        # Calcular cantidad de unidades
        unidades = tamano_posicion_usd / precio_entrada if precio_entrada else 0
        
        # Preparar resultado detallado
        resultado = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "capital_actual": self.capital_actual,
            "tamano_posicion_usd": tamano_posicion_usd,
            "tamano_posicion_porcentaje": riesgo_operacion * 100,
            "unidades": unidades,
            "precio_entrada": precio_entrada,
            "stop_loss_precio": stop_loss_precio,
            "take_profit_precio": take_profit_precio,
            "stop_loss_percent": stop_loss_percent * 100,
            "take_profit_percent": take_profit_percent * 100,
            "ratio_beneficio_riesgo": abs(take_profit_percent / stop_loss_percent),
            "factores_ajuste": {
                "capital": factor_capital,
                "drawdown": factor_drawdown,
                "kelly": factor_kelly,
                "score": factor_score,
                "volatilidad": factor_volatilidad,
                "proteccion": factor_proteccion
            },
            "volatilidad": volatilidad,
            "nivel_proteccion": self.nivel_proteccion
        }
        
        return resultado
    
    async def actualizar_capital(self, nuevo_capital: float) -> Dict[str, Any]:
        """
        Actualizar capital actual y recalcular métricas de riesgo.
        
        Args:
            nuevo_capital: Nuevo monto de capital en USD
            
        Returns:
            Diccionario con cambios y métricas actualizadas
        """
        cambio_porcentual = (nuevo_capital / self.capital_actual) - 1 if self.capital_actual > 0 else 0
        
        # Actualizar historial
        self.historial_capital.append({
            "timestamp": datetime.now().timestamp(),
            "capital": nuevo_capital,
            "cambio_porcentual": cambio_porcentual
        })
        
        # Limitar historial a 1000 puntos
        if len(self.historial_capital) > 1000:
            self.historial_capital = self.historial_capital[-1000:]
        
        # Actualizar capital actual
        capital_anterior = self.capital_actual
        self.capital_actual = nuevo_capital
        
        # Actualizar capital máximo histórico
        if nuevo_capital > self.max_capital_historico:
            self.max_capital_historico = nuevo_capital
        
        # Calcular drawdown actual
        self.drawdown_actual = 1 - (nuevo_capital / self.max_capital_historico)
        
        # Actualizar nivel de protección
        self._actualizar_nivel_proteccion()
        
        # Recalcular volatilidad del portfolio
        self._actualizar_volatilidad_portfolio()
        
        # Recalcular métricas de rendimiento
        self._actualizar_metricas_rendimiento()
        
        # Preparar resultado
        resultado = {
            "capital_anterior": capital_anterior,
            "capital_nuevo": nuevo_capital,
            "cambio_porcentual": cambio_porcentual * 100,
            "drawdown_actual": self.drawdown_actual * 100,
            "max_capital_historico": self.max_capital_historico,
            "nivel_proteccion": self.nivel_proteccion,
            "metricas": {
                "volatilidad_portfolio": self.stats["volatilidad_portfolio"] * 100,
                "sharpe_ratio": self.stats["sharpe_ratio"],
                "sortino_ratio": self.stats["sortino_ratio"],
                "calmar_ratio": self.stats["calmar_ratio"],
                "kelly_criterion": self.stats["kelly_criterion"] * 100
            }
        }
        
        return resultado
    
    async def registrar_operacion(self, 
                               operacion: Dict[str, Any]) -> Dict[str, Any]:
        """
        Registrar una operación completada y actualizar métricas.
        
        Args:
            operacion: Diccionario con detalles de la operación
            
        Returns:
            Diccionario con métricas actualizadas
        """
        # Validar operación
        campos_requeridos = ["symbol", "entrada", "salida", "unidades", "resultado_usd", "resultado_porcentual"]
        for campo in campos_requeridos:
            if campo not in operacion:
                raise ValueError(f"Falta el campo requerido '{campo}' en la operación")
        
        # Añadir timestamp y métricas adicionales
        operacion["timestamp"] = datetime.now().timestamp()
        operacion["ganadora"] = operacion["resultado_usd"] > 0
        
        # Añadir al registro de operaciones
        self.operaciones.append(operacion)
        
        # Limitar historial a 1000 operaciones
        if len(self.operaciones) > 1000:
            self.operaciones = self.operaciones[-1000:]
        
        # Actualizar estadísticas
        self.stats["operaciones_totales"] += 1
        if operacion["ganadora"]:
            self.stats["operaciones_ganadoras"] += 1
        else:
            self.stats["operaciones_perdedoras"] += 1
        
        self.stats["mayor_ganancia"] = max(
            self.stats["mayor_ganancia"],
            operacion["resultado_porcentual"] if operacion["ganadora"] else 0
        )
        
        self.stats["mayor_perdida"] = min(
            self.stats["mayor_perdida"],
            operacion["resultado_porcentual"] if not operacion["ganadora"] else 0
        )
        
        # Actualizar win rate y ratio beneficio-riesgo
        self._actualizar_metricas_rendimiento()
        
        # Actualizar capital si se proporciona
        capital_actualizado = None
        if "capital_final" in operacion and operacion["capital_final"] > 0:
            resultado_actualizacion = await self.actualizar_capital(operacion["capital_final"])
            capital_actualizado = resultado_actualizacion
        
        # Preparar resultado
        resultado = {
            "operacion_id": len(self.operaciones),
            "win_rate_actualizado": self.win_rate,
            "ratio_beneficio_riesgo_actualizado": self.ratio_beneficio_riesgo,
            "operaciones_totales": self.stats["operaciones_totales"],
            "operaciones_ganadoras": self.stats["operaciones_ganadoras"],
            "operaciones_perdedoras": self.stats["operaciones_perdedoras"],
            "actualizacion_capital": capital_actualizado
        }
        
        return resultado
    
    async def evaluar_correlaciones(self, activos_actuales: List[str]) -> Dict[str, Any]:
        """
        Evaluar correlaciones entre activos para evitar sobreexposición.
        
        Args:
            activos_actuales: Lista de símbolos de activos actuales
            
        Returns:
            Diccionario con matriz de correlación y recomendaciones
        """
        # Obtener datos históricos para análisis de correlación
        datos_precio = {}
        for symbol in activos_actuales:
            # En un sistema real, obtendríamos datos históricos de precios
            # Para esta demo, generamos datos simulados
            datos_precio[symbol] = self._generar_precios_simulados(symbol)
        
        # Calcular matriz de correlación
        correlaciones = {}
        for symbol1 in activos_actuales:
            correlaciones[symbol1] = {}
            for symbol2 in activos_actuales:
                if symbol1 == symbol2:
                    correlaciones[symbol1][symbol2] = 1.0
                else:
                    # Calcular correlación entre los dos activos
                    correlaciones[symbol1][symbol2] = self._calcular_correlacion(
                        datos_precio[symbol1], 
                        datos_precio[symbol2]
                    )
        
        # Identificar pares altamente correlacionados (>0.7)
        pares_alta_correlacion = []
        for symbol1 in activos_actuales:
            for symbol2 in activos_actuales:
                if symbol1 < symbol2:  # Evitar duplicados
                    if correlaciones[symbol1][symbol2] > 0.7:
                        pares_alta_correlacion.append({
                            "par": [symbol1, symbol2],
                            "correlacion": correlaciones[symbol1][symbol2]
                        })
        
        # Calcular correlación promedio
        correlaciones_valores = []
        for symbol1 in activos_actuales:
            for symbol2 in activos_actuales:
                if symbol1 < symbol2:  # Evitar duplicados
                    correlaciones_valores.append(correlaciones[symbol1][symbol2])
        
        correlacion_promedio = sum(correlaciones_valores) / len(correlaciones_valores) if correlaciones_valores else 0
        self.stats["correlacion_promedio"] = correlacion_promedio
        
        # Preparar resultado
        resultado = {
            "matriz_correlacion": correlaciones,
            "correlacion_promedio": correlacion_promedio,
            "pares_alta_correlacion": pares_alta_correlacion,
            "recomendaciones": []
        }
        
        # Generar recomendaciones
        if correlacion_promedio > 0.6:
            resultado["recomendaciones"].append({
                "tipo": "ADVERTENCIA",
                "mensaje": "Alta correlación general en el portfolio. Se recomienda diversificar en activos menos correlacionados."
            })
        
        for par in pares_alta_correlacion:
            resultado["recomendaciones"].append({
                "tipo": "REDUCIR",
                "mensaje": f"Alta correlación ({par['correlacion']:.2f}) entre {par['par'][0]} y {par['par'][1]}. Considerar reducir exposición a uno de ellos."
            })
        
        return resultado
    
    async def simular_escenarios(self, 
                              activos_actuales: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simular escenarios para evaluar riesgo del portfolio actual.
        
        Args:
            activos_actuales: Lista de diccionarios con posiciones actuales
            
        Returns:
            Diccionario con resultados de simulaciones
        """
        escenarios = {
            "mercado_alcista": {"nombre": "Mercado alcista", "cambio_promedio": 0.15},
            "mercado_lateral": {"nombre": "Mercado lateral", "cambio_promedio": 0.0},
            "correccion_leve": {"nombre": "Corrección leve", "cambio_promedio": -0.1},
            "correccion_moderada": {"nombre": "Corrección moderada", "cambio_promedio": -0.2},
            "crash": {"nombre": "Crash de mercado", "cambio_promedio": -0.4}
        }
        
        resultados = {}
        for escenario_id, escenario in escenarios.items():
            # Simulación de cómo el portfolio respondería al escenario
            impacto_portfolio = 0.0
            impacto_por_activo = {}
            
            for activo in activos_actuales:
                symbol = activo.get("symbol")
                if not symbol:
                    continue
                    
                proporcion_portfolio = activo.get("proporcion_portfolio", 0.1)
                volatilidad = activo.get("volatilidad", self.volatilidad_base)
                
                # Impacto específico en este activo
                factor_volatilidad = volatilidad / self.volatilidad_base
                impacto_base = escenario["cambio_promedio"]
                
                # Activos más volátiles amplificarán los movimientos
                impacto_activo = impacto_base * factor_volatilidad
                
                # Considerar stop-loss y take-profit
                stop_loss = activo.get("stop_loss_percent", -0.1)
                take_profit = activo.get("take_profit_percent", 0.2)
                
                if impacto_activo < stop_loss:
                    impacto_activo = stop_loss  # El stop-loss limita pérdidas
                elif impacto_activo > take_profit:
                    impacto_activo = take_profit  # El take-profit limita ganancias
                
                # Contribución al portfolio
                impacto_portfolio += impacto_activo * proporcion_portfolio
                impacto_por_activo[symbol] = impacto_activo
            
            # Almacenar resultados
            resultados[escenario_id] = {
                "nombre": escenario["nombre"],
                "cambio_promedio_mercado": escenario["cambio_promedio"] * 100,
                "impacto_portfolio": impacto_portfolio * 100,
                "impacto_capital": self.capital_actual * impacto_portfolio,
                "capital_resultante": self.capital_actual * (1 + impacto_portfolio),
                "impacto_por_activo": {s: v * 100 for s, v in impacto_por_activo.items()}
            }
        
        # Calcular valor en riesgo (VaR)
        valor_en_riesgo_95 = -resultados["correccion_moderada"]["impacto_capital"]
        valor_en_riesgo_99 = -resultados["crash"]["impacto_capital"]
        
        # Preparar resultado final
        resultado = {
            "escenarios": resultados,
            "metricas_riesgo": {
                "valor_en_riesgo_95": valor_en_riesgo_95,
                "valor_en_riesgo_95_porcentaje": (valor_en_riesgo_95 / self.capital_actual) * 100,
                "valor_en_riesgo_99": valor_en_riesgo_99,
                "valor_en_riesgo_99_porcentaje": (valor_en_riesgo_99 / self.capital_actual) * 100,
                "drawdown_actual": self.drawdown_actual * 100,
                "drawdown_proyectado_worst_case": min(99.9, (self.drawdown_actual + abs(resultados["crash"]["impacto_portfolio"] / 100)) * 100)
            },
            "recomendaciones": []
        }
        
        # Generar recomendaciones
        var_porcentaje = (valor_en_riesgo_95 / self.capital_actual) * 100
        if var_porcentaje > 15:
            resultado["recomendaciones"].append({
                "tipo": "REDUCIR_RIESGO",
                "mensaje": f"El VaR 95% ({var_porcentaje:.1f}%) es elevado. Considerar reducir el tamaño de posiciones o aumentar diversificación."
            })
        
        if resultados["correccion_leve"]["impacto_portfolio"] < -10:
            resultado["recomendaciones"].append({
                "tipo": "AJUSTAR_STOPS",
                "mensaje": "El portfolio es sensible a correcciones leves. Considerar ajustar stop-loss para limitar pérdidas."
            })
        
        return resultado
    
    def get_estado_actual(self) -> Dict[str, Any]:
        """
        Obtener estado actual completo del gestor de riesgo.
        
        Returns:
            Diccionario con estado actual
        """
        return {
            "capital": {
                "inicial": self.capital_inicial,
                "actual": self.capital_actual,
                "maximo_historico": self.max_capital_historico,
                "drawdown_actual": self.drawdown_actual * 100
            },
            "metricas_rendimiento": {
                "win_rate": self.win_rate * 100,
                "ratio_beneficio_riesgo": self.ratio_beneficio_riesgo,
                "operaciones_totales": self.stats["operaciones_totales"],
                "operaciones_ganadoras": self.stats["operaciones_ganadoras"],
                "operaciones_perdedoras": self.stats["operaciones_perdedoras"],
                "mayor_ganancia": self.stats["mayor_ganancia"] * 100,
                "mayor_perdida": self.stats["mayor_perdida"] * 100
            },
            "metricas_riesgo": {
                "volatilidad_portfolio": self.stats["volatilidad_portfolio"] * 100,
                "sharpe_ratio": self.stats["sharpe_ratio"],
                "sortino_ratio": self.stats["sortino_ratio"],
                "calmar_ratio": self.stats["calmar_ratio"],
                "kelly_criterion": self.stats["kelly_criterion"] * 100,
                "nivel_proteccion": self.nivel_proteccion
            },
            "correlacion": {
                "promedio": self.stats["correlacion_promedio"]
            },
            "configuracion": {
                "max_drawdown_permitido": self.max_drawdown_permitido * 100,
                "volatilidad_base": self.volatilidad_base * 100
            }
        }
    
    # Métodos internos
    
    async def _obtener_volatilidad(self, symbol: str) -> float:
        """
        Obtener volatilidad actual de un activo.
        
        Args:
            symbol: Símbolo del activo
            
        Returns:
            Volatilidad como decimal (ej. 0.02 para 2%)
        """
        # Intentar obtener de base de datos
        metrics, _ = await db.get_crypto_metrics_with_scores(symbol)
        
        if metrics and metrics.atr_percent:
            return metrics.atr_percent
        
        if metrics and metrics.atr and metrics.price and metrics.price > 0:
            return metrics.atr / metrics.price
        
        # Valor por defecto si no hay datos
        return self.volatilidad_base
    
    def _calcular_correlacion(self, serie1: List[float], serie2: List[float]) -> float:
        """
        Calcular correlación entre dos series de precios.
        
        Args:
            serie1: Primera serie de precios
            serie2: Segunda serie de precios
            
        Returns:
            Correlación entre -1 y 1
        """
        # Asegurar longitud igual
        min_len = min(len(serie1), len(serie2))
        s1 = serie1[-min_len:]
        s2 = serie2[-min_len:]
        
        # Convertir a retornos porcentuales
        retornos1 = [s1[i] / s1[i-1] - 1 for i in range(1, len(s1))]
        retornos2 = [s2[i] / s2[i-1] - 1 for i in range(1, len(s2))]
        
        # Calcular correlación
        try:
            return np.corrcoef(retornos1, retornos2)[0, 1]
        except:
            return 0.0
    
    def _generar_precios_simulados(self, symbol: str, n_puntos: int = 100) -> List[float]:
        """
        Generar serie de precios simulados para demostraciones.
        
        Args:
            symbol: Símbolo del activo
            n_puntos: Número de puntos a generar
            
        Returns:
            Lista de precios
        """
        import random
        
        # Determinar características simuladas por símbolo
        volatilidad = 0.02  # Base
        tendencia = 0.0001  # Base
        
        # Algunas cryptos tienen características diferentes
        if symbol in ["BTC", "ETH"]:
            precio_inicial = 45000 if symbol == "BTC" else 3000
            volatilidad = 0.015
            tendencia = 0.0002
        elif symbol in ["SOL", "BNB", "ADA"]:
            precio_inicial = random.uniform(50, 300)
            volatilidad = 0.02
            tendencia = 0.0001
        else:
            precio_inicial = random.uniform(1, 50)
            volatilidad = 0.025
            tendencia = 0.0
        
        # Generar serie de precios
        precios = [precio_inicial]
        for i in range(1, n_puntos):
            cambio = random.normalvariate(tendencia, volatilidad)
            nuevo_precio = precios[-1] * (1 + cambio)
            precios.append(nuevo_precio)
        
        return precios
    
    def _actualizar_nivel_proteccion(self) -> None:
        """Actualizar nivel de protección basado en drawdown actual."""
        # Determinar nivel según drawdown
        if self.drawdown_actual < 0.1:
            self.nivel_proteccion = "NORMAL"
        elif self.drawdown_actual < 0.15:
            self.nivel_proteccion = "CAUTIOUS"
        elif self.drawdown_actual < 0.2:
            self.nivel_proteccion = "DEFENSIVE"
        else:
            self.nivel_proteccion = "ULTRADEFENSIVE"
    
    def _actualizar_volatilidad_portfolio(self) -> None:
        """Actualizar estimación de volatilidad del portfolio."""
        # En un sistema real, calcularíamos esto basado en posiciones reales
        # Para esta demo, lo estimamos desde los cambios recientes en capital
        
        if len(self.historial_capital) > 10:
            # Usar los últimos 10 cambios de capital
            cambios = [entry["cambio_porcentual"] for entry in self.historial_capital[-10:]]
            self.stats["volatilidad_portfolio"] = np.std(cambios) if cambios else self.volatilidad_base
        else:
            self.stats["volatilidad_portfolio"] = self.volatilidad_base
    
    def _actualizar_metricas_rendimiento(self) -> None:
        """Actualizar métricas de rendimiento y ratios financieros."""
        # Win rate
        if self.stats["operaciones_totales"] > 0:
            self.win_rate = self.stats["operaciones_ganadoras"] / self.stats["operaciones_totales"]
        
        # Ratio beneficio-riesgo
        if self.operaciones:
            ganancias = [op["resultado_porcentual"] for op in self.operaciones if op["ganadora"]]
            perdidas = [abs(op["resultado_porcentual"]) for op in self.operaciones if not op["ganadora"]]
            
            ganancia_promedio = sum(ganancias) / len(ganancias) if ganancias else 0
            perdida_promedio = sum(perdidas) / len(perdidas) if perdidas else 1
            
            self.ratio_beneficio_riesgo = ganancia_promedio / perdida_promedio if perdida_promedio > 0 else 1.0
        
        # Rendimiento total
        self.stats["rendimiento_total"] = (self.capital_actual / self.capital_inicial) - 1
        
        # Sharpe Ratio (simplificado)
        rendimiento_anualizado = self.stats["rendimiento_total"] * (365 / max(1, len(self.historial_capital)))
        tasa_libre_riesgo = 0.02  # 2% anual
        
        if self.stats["volatilidad_portfolio"] > 0:
            self.stats["sharpe_ratio"] = (rendimiento_anualizado - tasa_libre_riesgo) / self.stats["volatilidad_portfolio"]
        
        # Sortino Ratio (solo volatilidad negativa)
        if self.operaciones:
            retornos_negativos = [op["resultado_porcentual"] for op in self.operaciones if op["resultado_porcentual"] < 0]
            volatilidad_negativa = np.std(retornos_negativos) if retornos_negativos else 0.01
            
            if volatilidad_negativa > 0:
                self.stats["sortino_ratio"] = (rendimiento_anualizado - tasa_libre_riesgo) / volatilidad_negativa
        
        # Calmar Ratio (rendimiento / max drawdown)
        if self.drawdown_actual > 0:
            self.stats["calmar_ratio"] = rendimiento_anualizado / self.drawdown_actual
        
        # Kelly Criterion
        self.stats["kelly_criterion"] = self.win_rate - ((1 - self.win_rate) / self.ratio_beneficio_riesgo)

# Instancia global para acceso desde cualquier módulo
risk_manager = AdaptiveRiskManager()

async def initialize_risk_manager():
    """Inicializar el gestor de riesgo con configuración predeterminada."""
    logger.info("Inicializando AdaptiveRiskManager...")
    # En un sistema real, aquí cargaríamos configuración desde base de datos
    return risk_manager