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
from typing import Dict, List, Any, Optional, Tuple, Set, Union, cast
from datetime import datetime, timedelta
import json
import random

from genesis.db.transcendental_database import transcendental_db
from genesis.db.models.crypto_classifier_models import (
    Cryptocurrency, CryptoClassification, CryptoMetrics,
    ClassificationHistory, CapitalScaleEffect
)
from sqlalchemy import select, and_, or_, desc, func, text

# Configuración de logging
logger = logging.getLogger("genesis.risk.adaptive_risk_manager")

class AdaptiveRiskManager:
    """
    Gestor de riesgo avanzado con capacidades adaptativas transcendentales.
    
    Este sistema ajusta dinámicamente los parámetros de gestión de riesgos
    basándose en el crecimiento del capital, condiciones del mercado y
    rendimiento histórico del sistema para mantener la eficiencia sin
    importar el tamaño del capital.
    """
    
    def __init__(self, 
                capital_inicial: float = 10000.0, 
                max_drawdown_permitido: float = 0.15,
                volatilidad_base: float = 0.02,
                capital_allocation_method: str = "adaptive"):
        """
        Inicializar el gestor de riesgo adaptativo.
        
        Args:
            capital_inicial: Capital inicial del sistema en USD
            max_drawdown_permitido: Máximo drawdown permitido (0.15 = 15%)
            volatilidad_base: Volatilidad base para cálculos (0.02 = 2%)
            capital_allocation_method: Método de asignación de capital 
                ("adaptive", "fixed", "kelly", "optimal")
        """
        self.capital_actual = capital_inicial
        self.capital_inicial = capital_inicial
        self.max_drawdown_permitido = max_drawdown_permitido
        self.volatilidad_base = volatilidad_base
        self.capital_allocation_method = capital_allocation_method
        
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
        
        # Estado de trascendencia
        self.modo_trascendental = "SINGULARITY_V4"  # Integración con Genesis
        self.trascendencia_activada = True
        
        logger.info(f"AdaptiveRiskManager inicializado con capital: ${capital_inicial:,.2f}, método: {capital_allocation_method}")
    
    async def calcular_tamano_posicion(self, 
                                     symbol: str, 
                                     score: float, 
                                     volatilidad: Optional[float] = None,
                                     saturation_point: Optional[float] = None) -> Dict[str, Any]:
        """
        Calcular tamaño óptimo de posición con ajuste adaptativo.
        
        Esta función determina el tamaño óptimo de posición basado en múltiples
        factores adaptativos como la puntuación del activo, el crecimiento del
        capital, la volatilidad actual y el historial de rendimiento.
        
        Args:
            symbol: Símbolo del activo
            score: Puntuación del clasificador (0-1)
            volatilidad: Volatilidad opcional del activo (si no se proporciona se busca)
            saturation_point: Punto de saturación opcional (USD)
            
        Returns:
            Diccionario con información sobre el tamaño de posición y parámetros
        """
        # Obtener volatilidad si no se proporcionó
        if volatilidad is None:
            volatilidad = await self._obtener_volatilidad(symbol)
        
        # Obtener métricas del activo
        metrics = await self._obtener_metricas_crypto(symbol)
        
        # Obtener punto de saturación si no se proporcionó
        if saturation_point is None and metrics:
            saturation_point = await self._estimar_punto_saturacion(symbol, metrics)
        elif saturation_point is None:
            saturation_point = 1_000_000  # Valor predeterminado
        
        # Parámetros base
        riesgo_capital_base = 0.01  # 1% del capital
        
        # Ajuste por tamaño de capital (reducir riesgo al crecer)
        factor_capital = 1.0
        if self.capital_actual > self.capital_inicial * 2:
            # Reducir el riesgo progresivamente al crecer el capital
            factor_capital = 1.0 / np.log10(self.capital_actual / self.capital_inicial + 1)
        
        # Ajuste por saturación (reducir exposición si se acerca al punto de saturación)
        factor_saturacion = 1.0
        if saturation_point > 0:
            # Posición potencial si usáramos el tamaño completo
            posicion_potencial = self.capital_actual * riesgo_capital_base * 10  # 10x leverage máximo
            
            # Si la posición potencial se acerca al punto de saturación, reducir
            if posicion_potencial > saturation_point * 0.1:
                saturacion_ratio = min(1.0, saturation_point / (posicion_potencial * 10))
                factor_saturacion = saturacion_ratio
        
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
        
        # Aplicar método de asignación de capital seleccionado
        if self.capital_allocation_method == "fixed":
            # Método fijo: riesgo constante
            riesgo_operacion = riesgo_capital_base * factor_proteccion
        elif self.capital_allocation_method == "kelly":
            # Método Kelly: basado principalmente en Kelly Criterion
            riesgo_operacion = factor_kelly * factor_proteccion * factor_score
        elif self.capital_allocation_method == "optimal":
            # Método óptimo: balance de todos los factores
            riesgo_operacion = (
                riesgo_capital_base * 
                (factor_capital + factor_drawdown + factor_kelly + factor_score + 
                 factor_volatilidad + factor_saturacion) / 6.0 * 
                factor_proteccion
            )
        else:  # adaptive (default)
            # Método adaptativo: considerando todos los factores independientemente
            riesgo_operacion = (
                riesgo_capital_base * 
                factor_capital * 
                factor_drawdown * 
                factor_kelly * 
                factor_score *
                factor_volatilidad *
                factor_saturacion *
                factor_proteccion
            )
        
        # Aplicar mecanismos trascendentales de Genesis
        if self.trascendencia_activada:
            # En modo Singularidad V4, minimizar el impacto de factores negativos
            if self.modo_trascendental == "SINGULARITY_V4":
                # Identificar el factor más limitante
                factores = {
                    "capital": factor_capital,
                    "drawdown": factor_drawdown,
                    "kelly": factor_kelly,
                    "score": factor_score,
                    "volatilidad": factor_volatilidad,
                    "saturacion": factor_saturacion,
                    "proteccion": factor_proteccion
                }
                
                min_factor = min(factores.items(), key=lambda x: x[1])
                
                # Si el factor más limitante es muy restrictivo, aplicar transmutación
                if min_factor[1] < 0.3:
                    # Aumentar el factor más limitante
                    boost_factor = 0.3 / min_factor[1]
                    
                    # Aplicar boost con límite para evitar sobrecompensación
                    boost_cap = 1.5
                    boost_factor = min(boost_factor, boost_cap)
                    
                    # Regenerar el riesgo con el factor aumentado
                    factores[min_factor[0]] *= boost_factor
                    
                    # Recalcular riesgo con los factores ajustados
                    if self.capital_allocation_method == "adaptive":
                        riesgo_operacion = (
                            riesgo_capital_base * 
                            factores["capital"] * 
                            factores["drawdown"] * 
                            factores["kelly"] * 
                            factores["score"] *
                            factores["volatilidad"] *
                            factores["saturacion"] *
                            factores["proteccion"]
                        )
                    
                    logger.info(f"Activada transmutación trascendental en factor '{min_factor[0]}': {min_factor[1]:.3f} -> {factores[min_factor[0]]:.3f}")
        
        # Calcular tamaño de posición en USD
        tamano_posicion_usd = self.capital_actual * riesgo_operacion
        
        # Calcular precio de entrada y salida
        precio_entrada = metrics.get("current_price", 0) if metrics else 0
        
        # Para cálculos de stop-loss y take-profit, considerar volatilidad
        vol_factor = volatilidad / 0.02 if volatilidad > 0 else 1.0
        stop_loss_percent = -0.05 - (score * 0.05) * vol_factor  # Entre -5% y -10%
        take_profit_percent = 0.1 + (score * 0.15) * vol_factor  # Entre 10% y 25%
        
        # Ajustar stop-loss y take-profit según métricas adicionales
        if metrics and metrics.get("volatility_30d"):
            vol_ajustada = float(metrics["volatility_30d"])
            atr_percent = vol_ajustada / 30.0  # Aproximación de ATR diario
            
            stop_loss_atr = -2.0 - (score * 2.0)  # Entre -2 y -4 ATRs
            take_profit_atr = 4.0 + (score * 4.0)  # Entre 4 y 8 ATRs
            
            stop_loss_percent = min(stop_loss_percent, stop_loss_atr * atr_percent)
            take_profit_percent = max(take_profit_percent, take_profit_atr * atr_percent)
        
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
            "ratio_beneficio_riesgo": abs(take_profit_percent / stop_loss_percent) if stop_loss_percent != 0 else 0,
            "factores_ajuste": {
                "capital": factor_capital,
                "drawdown": factor_drawdown,
                "kelly": factor_kelly,
                "score": factor_score,
                "volatilidad": factor_volatilidad,
                "saturacion": factor_saturacion,
                "proteccion": factor_proteccion
            },
            "volatilidad": volatilidad,
            "nivel_proteccion": self.nivel_proteccion,
            "saturation_point": saturation_point,
            "modo_trascendental": self.modo_trascendental,
            "trascendencia_activada": self.trascendencia_activada
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
        
        # Actualizar historial con checkpoint trascendental
        timestamp = datetime.now().timestamp()
        entry = {
            "timestamp": timestamp,
            "capital": nuevo_capital,
            "cambio_porcentual": cambio_porcentual,
            "fecha": datetime.now().isoformat()
        }
        
        # Guardar en historial local
        self.historial_capital.append(entry)
        
        # Guardar checkpoint trascendental en la base de datos
        if abs(cambio_porcentual) > 0.01:  # Solo guardar cambios significativos (>1%)
            await transcendental_db.checkpoint_state(
                "capital_history", 
                f"{timestamp}", 
                entry
            )
        
        # Limitar historial a 1000 puntos
        if len(self.historial_capital) > 1000:
            self.historial_capital = self.historial_capital[-1000:]
        
        # Actualizar capital actual
        capital_anterior = self.capital_actual
        self.capital_actual = nuevo_capital
        
        # Actualizar capital máximo histórico
        if nuevo_capital > self.max_capital_historico:
            self.max_capital_historico = nuevo_capital
            
            # Sincronización temporal al alcanzar nuevo máximo
            if self.trascendencia_activada:
                await transcendental_db.perform_temporal_sync()
        
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
        timestamp = datetime.now().timestamp()
        operacion["timestamp"] = timestamp
        operacion["ganadora"] = operacion["resultado_usd"] > 0
        operacion["modo_trascendental"] = self.modo_trascendental
        
        # Añadir al registro de operaciones con checkpoint trascendental
        self.operaciones.append(operacion)
        
        # Guardar checkpoint trascendental para operaciones significativas
        if abs(operacion["resultado_porcentual"]) > 1.0 or operacion["resultado_usd"] > 100:
            await transcendental_db.checkpoint_state(
                "operacion", 
                f"{operacion['symbol']}_{timestamp}", 
                operacion
            )
        
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
        precios_por_activo = {}
        
        # Obtener datos de precios de la base de datos para cada activo
        for symbol in activos_actuales:
            precios = await self._obtener_precios_historicos(symbol)
            if precios:
                precios_por_activo[symbol] = precios
        
        # Si no hay suficientes datos para todos los activos, usar los que tenemos
        activos_con_datos = list(precios_por_activo.keys())
        
        # Calcular matriz de correlación
        correlaciones = {}
        for symbol1 in activos_con_datos:
            correlaciones[symbol1] = {}
            for symbol2 in activos_con_datos:
                if symbol1 == symbol2:
                    correlaciones[symbol1][symbol2] = 1.0
                else:
                    # Calcular correlación entre los dos activos
                    correlaciones[symbol1][symbol2] = self._calcular_correlacion(
                        precios_por_activo[symbol1], 
                        precios_por_activo[symbol2]
                    )
        
        # Identificar pares altamente correlacionados (>0.7)
        pares_alta_correlacion = []
        for symbol1 in activos_con_datos:
            for symbol2 in activos_con_datos:
                if symbol1 < symbol2:  # Evitar duplicados
                    if correlaciones[symbol1][symbol2] > 0.7:
                        pares_alta_correlacion.append({
                            "par": [symbol1, symbol2],
                            "correlacion": correlaciones[symbol1][symbol2]
                        })
        
        # Calcular correlación promedio
        correlaciones_valores = []
        for symbol1 in activos_con_datos:
            for symbol2 in activos_con_datos:
                if symbol1 < symbol2:  # Evitar duplicados
                    correlaciones_valores.append(correlaciones[symbol1][symbol2])
        
        correlacion_promedio = sum(correlaciones_valores) / len(correlaciones_valores) if correlaciones_valores else 0
        
        # Actualizar estadísticas
        self.stats["correlacion_promedio"] = correlacion_promedio
        self.matriz_correlacion = correlaciones
        
        # Calcular exposición efectiva (ajustada por correlación)
        exposicion_nominal = len(activos_actuales)
        exposicion_efectiva = exposicion_nominal * (1 - correlacion_promedio * 0.5)
        
        # Generar recomendaciones
        recomendaciones = []
        
        # Si hay alta correlación, recomendar reducción
        if correlacion_promedio > 0.6:
            recomendaciones.append({
                "tipo": "ALERTA",
                "mensaje": f"Alta correlación promedio ({correlacion_promedio:.2f}), considerar diversificar"
            })
        
        # Recomendar reducción para pares específicos de alta correlación
        for par in pares_alta_correlacion:
            recomendaciones.append({
                "tipo": "REDUCCION",
                "mensaje": f"Alta correlación ({par['correlacion']:.2f}) entre {par['par'][0]} y {par['par'][1]}",
                "accion": "Reducir exposición a uno de estos activos"
            })
        
        # Recomendar diversificación si es necesario
        if exposicion_efectiva < 0.7 * exposicion_nominal:
            recomendaciones.append({
                "tipo": "DIVERSIFICACION",
                "mensaje": f"Baja diversificación efectiva ({exposicion_efectiva:.1f} vs {exposicion_nominal} nominal)",
                "accion": "Añadir activos con baja correlación al portfolio"
            })
        
        # Aplicar mecanismos trascendentales de Genesis
        if self.trascendencia_activada and pares_alta_correlacion:
            # Guardar checkpoint para futuras reconciliaciones
            await transcendental_db.checkpoint_state(
                "correlation_analysis", 
                datetime.now().strftime("%Y%m%d%H%M%S"), 
                {
                    "matriz_correlacion": {k: dict(v) for k, v in correlaciones.items()},
                    "pares_alta_correlacion": pares_alta_correlacion,
                    "correlacion_promedio": correlacion_promedio,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # En modo Singularidad, añadir recomendaciones trascendentales
            if self.modo_trascendental == "SINGULARITY_V4":
                # Sugerir cambios en la asignación para reducir correlación
                recomendaciones.append({
                    "tipo": "SINGULARIDAD",
                    "mensaje": "Aplicado análisis trascendental de correlaciones",
                    "accion": "Redistribuir capital para maximizar diversificación efectiva"
                })
        
        # Preparar resultado
        resultado = {
            "correlaciones": correlaciones,
            "correlacion_promedio": correlacion_promedio,
            "pares_alta_correlacion": pares_alta_correlacion,
            "exposicion_nominal": exposicion_nominal,
            "exposicion_efectiva": exposicion_efectiva,
            "recomendaciones": recomendaciones,
            "timestamp": datetime.now().isoformat()
        }
        
        return resultado
    
    async def optimizar_portfolio(self, 
                               cryptos_clasificadas: List[Dict[str, Any]],
                               max_posiciones: int = 10) -> Dict[str, Any]:
        """
        Optimizar portfolio basado en clasificaciones y restricciones.
        
        Args:
            cryptos_clasificadas: Lista de criptomonedas con clasificaciones
            max_posiciones: Número máximo de posiciones a mantener
            
        Returns:
            Diccionario con asignaciones óptimas y recomendaciones
        """
        # Filtrar por puntuación mínima
        score_minimo = 0.6
        candidatos = [c for c in cryptos_clasificadas if c.get("final_score", 0) >= score_minimo]
        
        # Ordenar por puntuación descendente
        candidatos = sorted(candidatos, key=lambda x: x.get("final_score", 0), reverse=True)
        
        # Limitar al número máximo de posiciones
        candidatos = candidatos[:max_posiciones]
        
        # Obtener datos para análisis de correlación
        symbols = [c.get("symbol") for c in candidatos]
        correlaciones = await self.evaluar_correlaciones(symbols)
        
        # Obtener métricas para estimación de points de saturación
        metricas_por_symbol = {}
        for crypto in candidatos:
            symbol = crypto.get("symbol")
            metricas = await self._obtener_metricas_crypto(symbol)
            if metricas:
                metricas_por_symbol[symbol] = metricas
        
        # Estimar puntos de saturación
        puntos_saturacion = {}
        for symbol, metricas in metricas_por_symbol.items():
            puntos_saturacion[symbol] = await self._estimar_punto_saturacion(symbol, metricas)
        
        # Asignar capital basado en puntuaciones y restricciones
        asignaciones = {}
        capital_restante = self.capital_actual
        
        # Aplicar método de asignación seleccionado
        if self.capital_allocation_method == "optimal" or self.capital_allocation_method == "adaptive":
            # Método que considera saturación y correlación
            
            # Primera pasada: asignación base por puntuación
            puntuacion_total = sum(c.get("final_score", 0) for c in candidatos)
            
            for crypto in candidatos:
                symbol = crypto.get("symbol")
                score = crypto.get("final_score", 0)
                
                # Peso basado en puntuación
                peso = score / puntuacion_total if puntuacion_total > 0 else 1.0 / len(candidatos)
                
                # Asignación inicial
                asignacion_inicial = capital_restante * peso
                
                # Limitar por punto de saturación
                punto_saturacion = puntos_saturacion.get(symbol, 1_000_000)
                asignacion = min(asignacion_inicial, punto_saturacion * 0.5)  # 50% del punto de saturación
                
                asignaciones[symbol] = asignacion
            
            # Segunda pasada: ajustar por correlaciones
            matriz_corr = correlaciones.get("correlaciones", {})
            if matriz_corr:
                # Crear matriz de penalización por correlación
                penalizaciones = {}
                for symbol in asignaciones:
                    if symbol in matriz_corr:
                        # Suma de correlaciones con otros activos
                        penalizacion = sum(matriz_corr[symbol].get(s, 0) for s in asignaciones if s != symbol)
                        # Normalizar
                        penalizacion = penalizacion / (len(asignaciones) - 1) if len(asignaciones) > 1 else 0
                        penalizaciones[symbol] = penalizacion
                
                # Ajustar asignaciones por penalización
                for symbol, asignacion in list(asignaciones.items()):
                    penalizacion = penalizaciones.get(symbol, 0)
                    # Reducir asignación si hay alta correlación
                    factor_ajuste = 1.0 - (penalizacion * 0.5)  # 50% de penalización máxima
                    asignaciones[symbol] = asignacion * factor_ajuste
        
        elif self.capital_allocation_method == "kelly":
            # Método basado en Kelly Criterion
            for crypto in candidatos:
                symbol = crypto.get("symbol")
                score = crypto.get("final_score", 0)
                
                # Estimar win rate y ratio beneficio/riesgo a partir del score
                win_rate_estimado = 0.3 + (score * 0.4)  # Entre 30% y 70%
                ratio_beneficio_riesgo = 1.0 + (score * 2.0)  # Entre 1.0 y 3.0
                
                # Calcular Kelly fraccional
                kelly = win_rate_estimado - (1 - win_rate_estimado) / ratio_beneficio_riesgo
                kelly_fraccional = max(0.1, min(0.5, kelly * 0.5))  # Kelly fraccional (50%)
                
                # Asignar capital
                asignacion = self.capital_actual * kelly_fraccional / len(candidatos)
                
                # Limitar por punto de saturación
                punto_saturacion = puntos_saturacion.get(symbol, 1_000_000)
                asignacion = min(asignacion, punto_saturacion * 0.5)
                
                asignaciones[symbol] = asignacion
        
        else:  # fixed (distribución equitativa)
            # Asignar capital equitativamente
            for crypto in candidatos:
                symbol = crypto.get("symbol")
                asignacion = self.capital_actual / len(candidatos)
                
                # Limitar por punto de saturación
                punto_saturacion = puntos_saturacion.get(symbol, 1_000_000)
                asignacion = min(asignacion, punto_saturacion * 0.5)
                
                asignaciones[symbol] = asignacion
        
        # Aplicar mecanismos trascendentales de Genesis
        if self.trascendencia_activada:
            if self.modo_trascendental == "SINGULARITY_V4":
                # En Singularidad, equilibrar para maximizar eficiencia
                
                # Identificar asignaciones demasiado pequeñas (< 1% del capital)
                asignaciones_pequenas = [s for s, a in asignaciones.items() if a < self.capital_actual * 0.01]
                
                # Si hay demasiadas asignaciones pequeñas, redistribuir
                if len(asignaciones_pequenas) > max_posiciones / 3:
                    # Eliminar asignaciones demasiado pequeñas
                    capital_liberado = sum(asignaciones[s] for s in asignaciones_pequenas)
                    for s in asignaciones_pequenas:
                        del asignaciones[s]
                    
                    # Redistribuir el capital liberado
                    if asignaciones and capital_liberado > 0:
                        for symbol in asignaciones:
                            asignaciones[symbol] += capital_liberado / len(asignaciones)
        
        # Calcular métricas del portfolio optimizado
        total_asignado = sum(asignaciones.values())
        porcentaje_asignado = (total_asignado / self.capital_actual) * 100 if self.capital_actual > 0 else 0
        
        # Generar recomendaciones de portfolio
        recomendaciones = []
        
        # Verificar si se asignó suficiente capital
        if porcentaje_asignado < 80:
            recomendaciones.append({
                "tipo": "EXPOSICION",
                "mensaje": f"Baja exposición total ({porcentaje_asignado:.1f}% del capital)",
                "accion": "Considerar reducir el nivel de protección o aumentar el número de posiciones"
            })
        
        # Verificar concentración
        concentracion_maxima = max(asignaciones.values()) / total_asignado if total_asignado > 0 else 0
        if concentracion_maxima > 0.3:  # >30% en un solo activo
            symbol_max = max(asignaciones.items(), key=lambda x: x[1])[0]
            recomendaciones.append({
                "tipo": "CONCENTRACION",
                "mensaje": f"Alta concentración en {symbol_max} ({concentracion_maxima*100:.1f}%)",
                "accion": "Considerar reducir exposición a este activo para mejor diversificación"
            })
        
        # Preparar resultado con asignaciones en USD y porcentaje
        asignaciones_finales = {
            symbol: {
                "usd": amount,
                "porcentaje": (amount / self.capital_actual) * 100 if self.capital_actual > 0 else 0
            }
            for symbol, amount in asignaciones.items()
        }
        
        # Guardar checkpoint trascendental de la optimización
        if self.trascendencia_activada:
            await transcendental_db.checkpoint_state(
                "portfolio_optimization", 
                datetime.now().strftime("%Y%m%d%H%M%S"), 
                {
                    "asignaciones": asignaciones_finales,
                    "puntos_saturacion": puntos_saturacion,
                    "modo_trascendental": self.modo_trascendental,
                    "capital_total": self.capital_actual,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Resultado completo
        resultado = {
            "candidatos_originales": len(cryptos_clasificadas),
            "posiciones_seleccionadas": len(asignaciones),
            "capital_total": self.capital_actual,
            "capital_asignado": total_asignado,
            "porcentaje_asignado": porcentaje_asignado,
            "asignaciones": asignaciones_finales,
            "puntos_saturacion": puntos_saturacion,
            "recomendaciones": recomendaciones,
            "nivel_proteccion": self.nivel_proteccion,
            "timestamp": datetime.now().isoformat(),
            "modo_trascendental": self.modo_trascendental
        }
        
        return resultado
    
    async def activar_modo_trascendental(self, modo: str = "SINGULARITY_V4") -> Dict[str, Any]:
        """
        Activar un modo trascendental específico.
        
        Esto habilita funcionalidades especiales del sistema Genesis que
        aplican mecanismos trascendentales para mejorar el rendimiento.
        
        Args:
            modo: Modo trascendental a activar 
                  ("SINGULARITY_V4", "LIGHT", "DARK_MATTER", etc.)
            
        Returns:
            Estado actualizado del gestor de riesgos
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
        
        # Efectos específicos según el modo
        if modo == "SINGULARITY_V4":
            # Reducir restrictividad en protección
            if self.nivel_proteccion == "ULTRADEFENSIVE":
                self.nivel_proteccion = "DEFENSIVE"
            elif self.nivel_proteccion == "DEFENSIVE":
                self.nivel_proteccion = "CAUTIOUS"
                
            # Ajustar factor de volatilidad para mayor estabilidad
            self.volatilidad_base = max(0.01, self.volatilidad_base * 0.8)
            
        elif modo == "LIGHT":
            # Modo luz: más proactivo, menos restrictivo
            self.nivel_proteccion = "NORMAL"
            
        elif modo == "DARK_MATTER":
            # Modo materia oscura: protección invisible
            # Mantiene nivel de protección actual pero opera con eficiencia aumentada
            pass
            
        # Sincronizar con base de datos trascendental
        await transcendental_db.perform_temporal_sync()
        
        # Resultado
        resultado = {
            "modo_anterior": modo_anterior,
            "modo_actual": modo,
            "trascendencia_activada": self.trascendencia_activada,
            "nivel_proteccion": self.nivel_proteccion,
            "volatilidad_base": self.volatilidad_base,
            "sincronizacion_temporal": "completada",
            "timestamp": datetime.now().isoformat()
        }
        
        return resultado
    
    async def obtener_estadisticas(self) -> Dict[str, Any]:
        """
        Obtener estadísticas completas del gestor de riesgos.
        
        Returns:
            Diccionario con todas las estadísticas y métricas
        """
        # Asegurar que las métricas estén actualizadas
        self._actualizar_metricas_rendimiento()
        
        # Métricas de capital y rendimiento
        metricas_capital = {
            "capital_inicial": self.capital_inicial,
            "capital_actual": self.capital_actual,
            "max_capital_historico": self.max_capital_historico,
            "drawdown_actual": self.drawdown_actual * 100,
            "rendimiento_total": ((self.capital_actual / self.capital_inicial) - 1) * 100 if self.capital_inicial > 0 else 0
        }
        
        # Métricas de operaciones
        metricas_operaciones = {
            "operaciones_totales": self.stats["operaciones_totales"],
            "operaciones_ganadoras": self.stats["operaciones_ganadoras"],
            "operaciones_perdedoras": self.stats["operaciones_perdedoras"],
            "win_rate": self.win_rate * 100,
            "ratio_beneficio_riesgo": self.ratio_beneficio_riesgo,
            "mayor_ganancia": self.stats["mayor_ganancia"] * 100,
            "mayor_perdida": self.stats["mayor_perdida"] * 100
        }
        
        # Métricas de riesgo
        metricas_riesgo = {
            "nivel_proteccion": self.nivel_proteccion,
            "volatilidad_portfolio": self.stats["volatilidad_portfolio"] * 100,
            "sharpe_ratio": self.stats["sharpe_ratio"],
            "sortino_ratio": self.stats["sortino_ratio"],
            "calmar_ratio": self.stats["calmar_ratio"],
            "kelly_criterion": self.stats["kelly_criterion"] * 100,
            "correlacion_promedio": self.stats["correlacion_promedio"]
        }
        
        # Métricas de estado trascendental
        metricas_trascendentales = {
            "modo_trascendental": self.modo_trascendental,
            "trascendencia_activada": self.trascendencia_activada,
            "capital_allocation_method": self.capital_allocation_method
        }
        
        # Si hay operaciones recientes, añadir algunas métricas adicionales
        ultimas_operaciones = []
        if self.operaciones:
            # Obtener últimas 5 operaciones
            ultimas = sorted(self.operaciones, key=lambda x: x.get("timestamp", 0), reverse=True)[:5]
            for op in ultimas:
                ultimas_operaciones.append({
                    "symbol": op.get("symbol", ""),
                    "resultado_usd": op.get("resultado_usd", 0),
                    "resultado_porcentual": op.get("resultado_porcentual", 0) * 100,
                    "ganadora": op.get("ganadora", False),
                    "fecha": datetime.fromtimestamp(op.get("timestamp", 0)).isoformat() if op.get("timestamp") else ""
                })
        
        # Estadísticas completas
        resultado = {
            "capital": metricas_capital,
            "operaciones": metricas_operaciones,
            "riesgo": metricas_riesgo,
            "trascendental": metricas_trascendentales,
            "ultimas_operaciones": ultimas_operaciones,
            "timestamp": datetime.now().isoformat()
        }
        
        return resultado
    
    async def _obtener_volatilidad(self, symbol: str) -> float:
        """
        Obtener volatilidad histórica para un símbolo.
        
        Args:
            symbol: Símbolo de la criptomoneda
            
        Returns:
            Volatilidad como decimal (0.01 = 1%)
        """
        # Consultar métricas en la base de datos
        async def get_metrics_query():
            return (
                select(CryptoMetrics)
                .join(Cryptocurrency, CryptoMetrics.cryptocurrency_id == Cryptocurrency.id)
                .where(Cryptocurrency.symbol == symbol)
                .order_by(CryptoMetrics.updated_at.desc())
                .limit(1)
            ), []
        
        try:
            result = await transcendental_db.execute_query(get_metrics_query)
            
            if result and len(result) > 0:
                metrics = result[0]
                if hasattr(metrics, "volatility_30d") and metrics.volatility_30d is not None:
                    return float(metrics.volatility_30d)
        except Exception as e:
            logger.warning(f"Error al obtener volatilidad para {symbol}: {str(e)}")
        
        # Valor por defecto si no se encuentra
        return self.volatilidad_base
    
    async def _obtener_metricas_crypto(self, symbol: str) -> Dict[str, Any]:
        """
        Obtener métricas completas para una criptomoneda.
        
        Args:
            symbol: Símbolo de la criptomoneda
            
        Returns:
            Diccionario con métricas o vacío si no se encuentra
        """
        # Consultar métricas en la base de datos
        async def get_crypto_with_metrics_query():
            return (
                select(Cryptocurrency, CryptoMetrics)
                .join(CryptoMetrics, CryptoMetrics.cryptocurrency_id == Cryptocurrency.id, isouter=True)
                .where(Cryptocurrency.symbol == symbol)
                .order_by(CryptoMetrics.updated_at.desc())
                .limit(1)
            ), []
        
        try:
            result = await transcendental_db.execute_query(get_crypto_with_metrics_query)
            
            if result and len(result) > 0:
                crypto, metrics = result[0]
                
                # Combinar datos de crypto y metrics
                combined = {}
                
                # Datos básicos de la criptomoneda
                if crypto:
                    combined["symbol"] = crypto.symbol
                    combined["name"] = crypto.name
                    combined["market_cap"] = crypto.market_cap
                    combined["volume_24h"] = crypto.volume_24h
                    combined["current_price"] = crypto.current_price
                    combined["price_change_24h"] = crypto.price_change_24h
                    combined["price_change_percentage_24h"] = crypto.price_change_percentage_24h
                
                # Métricas avanzadas
                if metrics:
                    combined["orderbook_depth_usd"] = metrics.orderbook_depth_usd
                    combined["slippage_1000usd"] = metrics.slippage_1000usd
                    combined["slippage_10000usd"] = metrics.slippage_10000usd
                    combined["slippage_100000usd"] = metrics.slippage_100000usd
                    combined["volatility_30d"] = metrics.volatility_30d
                    combined["sharpe_ratio"] = metrics.sharpe_ratio
                    combined["sortino_ratio"] = metrics.sortino_ratio
                    combined["drawdown_max"] = metrics.drawdown_max
                
                return combined
                
        except Exception as e:
            logger.warning(f"Error al obtener métricas para {symbol}: {str(e)}")
        
        # Si no hay datos, devolver un diccionario vacío
        return {}
    
    async def _estimar_punto_saturacion(self, symbol: str, metrics: Dict[str, Any]) -> float:
        """
        Estimar punto de saturación de capital para una criptomoneda.
        
        Args:
            symbol: Símbolo de la criptomoneda
            metrics: Métricas disponibles
            
        Returns:
            Punto de saturación estimado en USD
        """
        # Intentar obtener primero de CapitalScaleEffect
        async def get_saturation_point_query():
            return (
                select(CapitalScaleEffect)
                .join(Cryptocurrency, CapitalScaleEffect.cryptocurrency_id == Cryptocurrency.id)
                .where(Cryptocurrency.symbol == symbol)
                .order_by(CapitalScaleEffect.analysis_date.desc())
                .limit(1)
            ), []
        
        try:
            result = await transcendental_db.execute_query(get_saturation_point_query)
            
            if result and len(result) > 0:
                scale_effect = result[0]
                if hasattr(scale_effect, "saturation_point") and scale_effect.saturation_point is not None:
                    return float(scale_effect.saturation_point)
        except Exception as e:
            logger.warning(f"Error al obtener punto de saturación para {symbol}: {str(e)}")
        
        # Si no hay datos en la base de datos, calcular basado en métricas disponibles
        # Valor base de saturación
        base_saturation = 1_000_000  # $1M por defecto
        
        # Ajustar según métricas
        if metrics:
            # Profundidad del libro de órdenes
            if metrics.get("orderbook_depth_usd"):
                depth = float(metrics["orderbook_depth_usd"])
                # Punto de saturación aproximado = 5-10% de la profundidad total
                depth_saturation = depth * 0.08
                base_saturation = max(base_saturation, depth_saturation)
            
            # Volume 24h indica liquidez
            if metrics.get("volume_24h"):
                volume = float(metrics["volume_24h"])
                # Aprox. 1% del volumen diario es sostenible
                volume_saturation = volume * 0.01
                base_saturation = max(base_saturation, volume_saturation)
            
            # Market cap como indicador de tamaño
            if metrics.get("market_cap"):
                mcap = float(metrics["market_cap"])
                # 0.05% - 0.1% del market cap
                mcap_saturation = mcap * 0.0005
                base_saturation = min(base_saturation, mcap_saturation)
        
        # Limitar a rangos razonables
        saturation = max(100_000, min(1_000_000_000, base_saturation))
        
        return saturation
    
    async def _obtener_precios_historicos(self, symbol: str, dias: int = 30) -> List[float]:
        """
        Obtener precios históricos para un símbolo.
        
        En un sistema real, esto obtendría datos de una base de datos.
        Para esta implementación, generamos datos simulados para demostración.
        
        Args:
            symbol: Símbolo de la criptomoneda
            dias: Número de días de historia
            
        Returns:
            Lista de precios históricos
        """
        # En un sistema real, obtendríamos datos históricos de la base de datos
        # Para esta demo, generamos datos basados en el hash del símbolo para consistencia
        random.seed(hash(symbol) % 10000)
        
        # Obtener precio actual si está disponible
        precio_base = 1000.0
        metrics = await self._obtener_metricas_crypto(symbol)
        if metrics and metrics.get("current_price"):
            precio_base = float(metrics["current_price"])
        
        # Volatilidad basada en el símbolo (para consistencia)
        volatilidad = 0.02 + (hash(symbol) % 100) / 1000  # Entre 2% y 12%
        
        # Generar serie de precios con tendencia y volatilidad
        precios = []
        precio = precio_base
        
        for _ in range(dias):
            # Cambio diario: componente aleatorio + tendencia
            cambio_pct = random.normalvariate(0, volatilidad)
            precio = precio * (1 + cambio_pct)
            precios.append(precio)
        
        # Invertir para que el último precio sea el más reciente
        return list(reversed(precios))
    
    def _calcular_correlacion(self, serie1: List[float], serie2: List[float]) -> float:
        """
        Calcular correlación entre dos series de precios.
        
        Args:
            serie1: Primera serie de precios
            serie2: Segunda serie de precios
            
        Returns:
            Coeficiente de correlación (-1 a 1)
        """
        # Asegurar que ambas series tienen la misma longitud
        min_length = min(len(serie1), len(serie2))
        if min_length < 2:
            return 0.0
            
        serie1 = serie1[:min_length]
        serie2 = serie2[:min_length]
        
        # Calcular retornos diarios
        retornos1 = [(serie1[i] / serie1[i-1]) - 1 for i in range(1, len(serie1))]
        retornos2 = [(serie2[i] / serie2[i-1]) - 1 for i in range(1, len(serie2))]
        
        # Calcular medias
        media1 = sum(retornos1) / len(retornos1)
        media2 = sum(retornos2) / len(retornos2)
        
        # Calcular covarianza y varianzas
        cov = sum((r1 - media1) * (r2 - media2) for r1, r2 in zip(retornos1, retornos2))
        var1 = sum((r - media1) ** 2 for r in retornos1)
        var2 = sum((r - media2) ** 2 for r in retornos2)
        
        # Calcular correlación
        if var1 > 0 and var2 > 0:
            return cov / (var1 ** 0.5 * var2 ** 0.5)
        else:
            return 0.0
    
    def _actualizar_nivel_proteccion(self) -> None:
        """
        Actualizar nivel de protección basado en drawdown actual.
        
        El sistema aumenta progresivamente la protección conforme
        aumenta el drawdown para proteger el capital.
        """
        # Umbrales de drawdown para cada nivel
        umbral_cautious = self.max_drawdown_permitido * 0.3    # 30% del máximo
        umbral_defensive = self.max_drawdown_permitido * 0.6   # 60% del máximo
        umbral_ultra = self.max_drawdown_permitido * 0.8       # 80% del máximo
        
        # Determinar nivel basado en drawdown actual
        if self.drawdown_actual >= umbral_ultra:
            nuevo_nivel = "ULTRADEFENSIVE"
        elif self.drawdown_actual >= umbral_defensive:
            nuevo_nivel = "DEFENSIVE"
        elif self.drawdown_actual >= umbral_cautious:
            nuevo_nivel = "CAUTIOUS"
        else:
            nuevo_nivel = "NORMAL"
        
        # Aplicar mecanismos trascendentales de Genesis
        if self.trascendencia_activada:
            if self.modo_trascendental == "SINGULARITY_V4" and nuevo_nivel == "ULTRADEFENSIVE":
                # En modo Singularidad, nunca llegar al nivel ultra-defensivo
                nuevo_nivel = "DEFENSIVE"
            elif self.modo_trascendental == "LIGHT":
                # En modo Luz, reducir un nivel de protección
                if nuevo_nivel == "ULTRADEFENSIVE":
                    nuevo_nivel = "DEFENSIVE"
                elif nuevo_nivel == "DEFENSIVE":
                    nuevo_nivel = "CAUTIOUS"
                elif nuevo_nivel == "CAUTIOUS":
                    nuevo_nivel = "NORMAL"
        
        # Actualizar nivel si cambió
        if nuevo_nivel != self.nivel_proteccion:
            nivel_anterior = self.nivel_proteccion
            self.nivel_proteccion = nuevo_nivel
            logger.info(f"Nivel de protección actualizado: {nivel_anterior} → {nuevo_nivel} (Drawdown: {self.drawdown_actual:.2%})")
    
    def _actualizar_volatilidad_portfolio(self) -> None:
        """
        Actualizar estimación de volatilidad del portfolio.
        
        Utiliza historial reciente de capital para estimar la volatilidad
        del portfolio completo.
        """
        # Necesitamos al menos 5 puntos para calcular volatilidad
        if len(self.historial_capital) < 5:
            return
        
        # Obtener los últimos 30 cambios porcentuales o todos si hay menos
        n_puntos = min(30, len(self.historial_capital) - 1)
        cambios = [entry["cambio_porcentual"] for entry in self.historial_capital[-n_puntos:]]
        
        # Calcular desviación estándar de los cambios
        if cambios:
            media = sum(cambios) / len(cambios)
            suma_cuadrados = sum((x - media) ** 2 for x in cambios)
            desviacion = (suma_cuadrados / len(cambios)) ** 0.5
            
            # Anualizar (aproximado)
            volatilidad_anualizada = desviacion * (252 ** 0.5)  # Asumiendo 252 días de trading
            
            # Actualizar
            self.stats["volatilidad_portfolio"] = volatilidad_anualizada
    
    def _actualizar_metricas_rendimiento(self) -> None:
        """
        Actualizar todas las métricas de rendimiento.
        
        Esto incluye win rate, ratio beneficio/riesgo, y ratios de rendimiento
        como Sharpe, Sortino y Calmar.
        """
        # Actualizar win rate
        if self.stats["operaciones_totales"] > 0:
            self.win_rate = self.stats["operaciones_ganadoras"] / self.stats["operaciones_totales"]
        
        # Actualizar ratio beneficio/riesgo
        if self.stats["operaciones_ganadoras"] > 0 and self.stats["operaciones_perdedoras"] > 0:
            ganancia_media = self.stats["mayor_ganancia"] / 100  # Convertir a decimal
            perdida_media = abs(self.stats["mayor_perdida"] / 100)  # Convertir a decimal y hacer positivo
            
            if perdida_media > 0:
                self.ratio_beneficio_riesgo = ganancia_media / perdida_media
        
        # Actualizar Sharpe Ratio (simplificado)
        rendimiento_total = (self.capital_actual / self.capital_inicial) - 1 if self.capital_inicial > 0 else 0
        volatilidad = self.stats["volatilidad_portfolio"]
        tasa_libre_riesgo = 0.02  # 2% anual
        
        if volatilidad > 0:
            self.stats["sharpe_ratio"] = (rendimiento_total - tasa_libre_riesgo) / volatilidad
        
        # Actualizar Sortino Ratio (simplificado, usando solo drawdown)
        if self.drawdown_actual > 0:
            self.stats["sortino_ratio"] = rendimiento_total / self.drawdown_actual
        
        # Actualizar Calmar Ratio (rendimiento / max drawdown)
        if self.drawdown_actual > 0:
            self.stats["calmar_ratio"] = rendimiento_total / self.drawdown_actual
        
        # Actualizar Kelly Criterion
        self.stats["kelly_criterion"] = self.win_rate - ((1 - self.win_rate) / self.ratio_beneficio_riesgo)


# Instancia global para acceso desde cualquier módulo
risk_manager = AdaptiveRiskManager()

async def initialize_risk_manager(capital_inicial: float = 10000.0, config: Optional[Dict[str, Any]] = None) -> None:
    """
    Inicializar el gestor de riesgo con configuración.
    
    Args:
        capital_inicial: Capital inicial en USD
        config: Configuración adicional (opcional)
    """
    # Aplicar configuración si se proporciona
    if config:
        max_drawdown = config.get("max_drawdown_permitido", 0.15)
        volatilidad_base = config.get("volatilidad_base", 0.02)
        capital_allocation_method = config.get("capital_allocation_method", "adaptive")
        
        # Crear nueva instancia con configuración
        global risk_manager
        risk_manager = AdaptiveRiskManager(
            capital_inicial=capital_inicial,
            max_drawdown_permitido=max_drawdown,
            volatilidad_base=volatilidad_base,
            capital_allocation_method=capital_allocation_method
        )
        
        # Activar modo trascendental si se especifica
        modo_trascendental = config.get("modo_trascendental")
        if modo_trascendental:
            await risk_manager.activar_modo_trascendental(modo_trascendental)
    else:
        # Actualizar capital de la instancia existente
        await risk_manager.actualizar_capital(capital_inicial)
    
    logger.info(f"AdaptiveRiskManager inicializado con capital: ${capital_inicial:,.2f}")