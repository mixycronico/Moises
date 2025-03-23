"""
Gestor de saldos para el sistema Genesis con capacidades de escalabilidad adaptativa.

Este módulo proporciona funcionalidades para gestionar los saldos de los usuarios,
incluyendo seguimiento de operaciones, cálculo de rentabilidad, y generación
de informes financieros.

Además, incorpora un sistema avanzado de escalabilidad que se adapta al crecimiento
del capital, manteniendo la eficiencia del sistema incluso cuando los fondos
aumentan significativamente.
"""

import logging
import asyncio
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from genesis.core.base import Component
from genesis.utils.helpers import generate_id, format_timestamp
from genesis.utils.logger import setup_logging
from genesis.accounting.predictive_scaling import PredictiveScalingEngine, EfficiencyPrediction

# Configurar precisión para operaciones con Decimal
getcontext().prec = 28

# Constantes de configuración para escalabilidad de capital
CAPITAL_CONFIG = {
    'CAPITAL_BASE_POR_DEFECTO': 10000.0,
    'UMBRAL_EFICIENCIA': 0.85,
    'MAX_SIMBOLOS': {'pequeño': 5, 'grande': 15},
    'TIEMPOS': ["15m", "1h", "4h", "1d"],
    'INTERVALO_REASIGNACION_HORAS': 6,
    'SATURACION_POR_DEFECTO': 1000000.0
}

# Clase para monitoreo y métricas
class Metrics:
    """Sistema simple de métricas para monitoreo."""
    def __init__(self):
        self.registry = {}
    
    def gauge(self, name: str, value: float) -> None:
        """Registrar valor absoluto."""
        self.registry[name] = value
    
    def increment(self, name: str) -> None:
        """Incrementar contador."""
        self.registry[name] = self.registry.get(name, 0) + 1
    
    def get_metrics(self) -> Dict[str, float]:
        """Obtener todas las métricas registradas."""
        return self.registry.copy()

# Instancia global de métricas
scaling_metrics = Metrics()

@dataclass
class MetricasInstrumento:
    """Clase de datos para métricas de instrumentos de trading."""
    capital_mercado: float = 0.0
    volumen_24h: float = 0.0
    puntaje_liquidez: float = 0.5
    puntaje_final: float = 0.5
    precio_actual: float = 0.0
    
    
class CapitalScalingManager:
    """
    Gestor de escalabilidad de capital para adaptarse al crecimiento de fondos.
    
    Este componente proporciona mecanismos avanzados para ajustar la distribución
    de capital y parámetros operativos según el nivel de fondos, manteniendo
    la eficiencia incluso cuando el capital crece significativamente.
    
    Características:
    - Integración con Redis para caché de resultados
    - Sistema de métricas para monitoreo en tiempo real
    - Almacenamiento en PostgreSQL para análisis histórico
    - Ajuste adaptativo según el tamaño del capital
    """
    
    def __init__(
        self, 
        config: Dict[str, Any] = None,
        capital_inicial: float = None,
        umbral_eficiencia: float = None
    ):
        """
        Inicializar el gestor de escalabilidad.
        
        Args:
            config: Configuración completa (opcional, prevalece sobre otros parámetros)
            capital_inicial: Capital base de referencia (si no se proporciona config)
            umbral_eficiencia: Umbral mínimo de eficiencia aceptable (si no se proporciona config)
        """
        # Configuración base
        self.config = config or {
            'capital_base': capital_inicial or CAPITAL_CONFIG['CAPITAL_BASE_POR_DEFECTO'],
            'efficiency_threshold': umbral_eficiencia or CAPITAL_CONFIG['UMBRAL_EFICIENCIA'],
            'max_symbols_small': CAPITAL_CONFIG['MAX_SIMBOLOS']['pequeño'],
            'max_symbols_large': CAPITAL_CONFIG['MAX_SIMBOLOS']['grande'],
            'timeframes': CAPITAL_CONFIG['TIEMPOS'],
            'reallocation_interval_hours': CAPITAL_CONFIG['INTERVALO_REASIGNACION_HORAS'],
            'saturation_default': CAPITAL_CONFIG['SATURACION_POR_DEFECTO'],
            'redis_cache_enabled': True,
            'monitoring_enabled': True
        }
        
        # Validar valores
        self.capital_inicial = max(0.0, self.config['capital_base'])
        self.umbral_eficiencia = min(1.0, max(0.0, self.config['efficiency_threshold']))
        self.logger = setup_logging("genesis.accounting.capital_scaling")
        
        # Estructuras de datos optimizadas
        self.registros_eficiencia: Dict[str, Dict[float, float]] = {}
        self.puntos_saturacion: Dict[str, float] = {}
        self.historial_distribucion: List[Dict[str, Any]] = []
        
        # Inicializar motor predictivo de escalabilidad
        self.predictive_engine = PredictiveScalingEngine()
        
        # Pool de hilos para cálculos intensivos
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Recursos externos
        self.redis = None  # Se inicializa en setup()
        self.db = None  # Se inicializa en setup()
        
        self.metricas_rendimiento = {
            "eficiencia_promedio": 1.0,
            "utilizacion_capital": 1.0,
            "factor_escala": 1.0,
            "entropia_asignacion": 0.0,
            "registros_distribucion": 0,
            "ultima_actualizacion": datetime.now()
        }
        
        # Registrar métricas iniciales
        self._actualizar_metricas_globales()
        
        self.logger.info(f"Gestor de escalabilidad inicializado con capital base: ${self.capital_inicial:,.2f}")
    
    async def setup(self):
        """Configurar conexiones a servicios externos como Redis."""
        if self.config.get('redis_cache_enabled', False):
            try:
                import aioredis
                self.redis = await aioredis.create_redis_pool('redis://localhost')
                self.logger.info("Redis conectado para caché de escalabilidad")
            except (ImportError, Exception) as e:
                self.logger.warning(f"No se pudo conectar a Redis: {str(e)}")
                self.config['redis_cache_enabled'] = False
    
    async def close(self):
        """Cerrar conexiones externas."""
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()
            self.logger.info("Conexión a Redis cerrada")
        
        if self.executor:
            self.executor.shutdown(wait=True)
            self.logger.info("ThreadPoolExecutor cerrado")
    
    def _actualizar_metricas_globales(self):
        """Actualizar métricas globales para monitoreo."""
        if not self.config.get('monitoring_enabled', False):
            return
            
        scaling_metrics.gauge("capital_scaling_base_capital", self.capital_inicial)
        scaling_metrics.gauge("capital_scaling_efficiency_threshold", self.umbral_eficiencia)
        scaling_metrics.gauge("capital_scaling_average_efficiency", self.metricas_rendimiento["eficiencia_promedio"])
        scaling_metrics.gauge("capital_scaling_utilization", self.metricas_rendimiento["utilizacion_capital"])
        scaling_metrics.gauge("capital_scaling_symbols_tracked", len(self.puntos_saturacion))
        scaling_metrics.gauge("capital_scaling_efficiency_records", sum(len(records) for records in self.registros_eficiencia.values()))

    async def calculate_optimal_allocation(
        self,
        capital_disponible: float,
        instrumentos: List[str],
        metricas: Dict[str, MetricasInstrumento]
    ) -> Dict[str, float]:
        """
        Calcular distribución óptima de capital entre instrumentos.
        
        Esta implementación avanzada considera puntos de saturación, liquidez
        y otros factores críticos para maximizar la eficiencia a medida que
        el capital crece.
        
        Args:
            capital_disponible: Capital total disponible
            instrumentos: Lista de instrumentos (símbolos)
            metricas: Métricas de cada instrumento
            
        Returns:
            Distribución óptima de capital por instrumento
        """
        try:
            # Registrar tiempo de inicio para medición de rendimiento
            start_time = time.time()
            
            # Validar entradas
            capital_disponible = max(0.0, capital_disponible)
            if not instrumentos or capital_disponible <= 0:
                self.logger.warning("Entrada inválida para asignación de capital")
                return {}
            
            # Calcular factor de escala
            factor_escala = max(1.0, capital_disponible / self.capital_inicial)
            self.logger.info(f"Factor de escala: {factor_escala:.2f}x (Capital: ${capital_disponible:,.2f})")
            
            # Determinar número máximo de instrumentos según capital
            max_instrumentos = self._calcular_max_instrumentos(factor_escala)
            
            # Limitar instrumentos si es necesario
            instrumentos_seleccionados = instrumentos[:max_instrumentos]
            if len(instrumentos) > max_instrumentos:
                self.logger.info(f"Limitando de {len(instrumentos)} a {max_instrumentos} instrumentos por restricciones de capital")
            
            # Verificar caché en Redis primero si está habilitado
            asignaciones = None
            cache_key = None
            cache_hit = False
            
            if self.config.get('redis_cache_enabled', False) and self.redis:
                try:
                    import pickle
                    # Crear clave de caché única
                    key_components = [
                        f"cap_{int(capital_disponible)}",
                        f"scale_{factor_escala:.2f}",
                        f"instr_{'_'.join(sorted(instrumentos_seleccionados[:5]))}",
                        f"count_{len(instrumentos_seleccionados)}"
                    ]
                    cache_key = f"scaling:allocation:{hash('_'.join(key_components))}"
                    
                    # Intentar obtener del caché
                    cached_data = await self.redis.get(cache_key)
                    if cached_data:
                        asignaciones = pickle.loads(cached_data)
                        self.logger.debug(f"Distribución obtenida desde caché: {len(asignaciones)} instrumentos")
                        cache_hit = True
                        
                        # Actualizar métricas para caché hit
                        if self.config.get('monitoring_enabled', False):
                            scaling_metrics.increment("capital_scaling_cache_hits")
                except Exception as e:
                    self.logger.warning(f"Error accediendo a caché Redis: {str(e)}")
            
            # Si no hay caché o falló, calcular normalmente
            if not asignaciones:
                # Calcular puntos de saturación en paralelo
                tareas_saturacion = [
                    self._estimate_saturation_point(simbolo, metricas.get(simbolo, MetricasInstrumento()))
                    for simbolo in instrumentos_seleccionados
                ]
                resultados_saturacion = await asyncio.gather(*tareas_saturacion)
                
                # Actualizar puntos de saturación
                for i, simbolo in enumerate(instrumentos_seleccionados):
                    self.puntos_saturacion[simbolo] = resultados_saturacion[i]
                
                # Calcular asignaciones iniciales
                asignaciones = await self._calcular_asignaciones(capital_disponible, instrumentos_seleccionados)
                
                # Guardar en caché si está configurado
                if self.config.get('redis_cache_enabled', False) and self.redis and cache_key:
                    try:
                        import pickle
                        # Almacenar en Redis con un TTL de 1 hora
                        await self.redis.set(
                            cache_key, 
                            pickle.dumps(asignaciones),
                            expire=3600  # 1 hora
                        )
                        self.logger.debug(f"Distribución guardada en caché: {len(asignaciones)} instrumentos")
                        
                        # Actualizar métricas para caché miss
                        if self.config.get('monitoring_enabled', False):
                            scaling_metrics.increment("capital_scaling_cache_misses")
                    except Exception as e:
                        self.logger.warning(f"Error guardando en caché Redis: {str(e)}")
            
            # Actualizar métricas y registros
            self._actualizar_metricas_rendimiento(capital_disponible, asignaciones)
            self._registrar_distribucion(capital_disponible, factor_escala, asignaciones)
            
            # Registrar en base de datos si está configurado
            if self.db and self.config.get('db_persistence_enabled', False):
                asyncio.create_task(
                    self._persist_allocation_history(capital_disponible, factor_escala, asignaciones)
                )
            
            # Registrar métricas de rendimiento
            if self.config.get('monitoring_enabled', False):
                elapsed_time = time.time() - start_time
                scaling_metrics.gauge("capital_scaling_calculation_time", elapsed_time)
                scaling_metrics.gauge("capital_scaling_instruments_count", len(instrumentos_seleccionados))
                scaling_metrics.gauge("capital_scaling_capital_scale", factor_escala)
                
                # Incrementar contador según si fue caché hit o miss
                scaling_metrics.increment("capital_scaling_allocation_count")
            
            return asignaciones
            
        except Exception as e:
            self.logger.error(f"Error en cálculo de asignación óptima: {str(e)}")
            
            # Registrar error en métricas
            if self.config.get('monitoring_enabled', False):
                scaling_metrics.increment("capital_scaling_errors")
                
            return {}

    async def _calcular_asignaciones(
        self, 
        capital: float, 
        instrumentos: List[str]
    ) -> Dict[str, float]:
        """
        Cálculo optimizado de asignaciones de capital.
        
        Args:
            capital: Capital disponible
            instrumentos: Lista de instrumentos
            
        Returns:
            Diccionario de asignaciones por instrumento
        """
        if not instrumentos:
            return {}
            
        # Calcular saturación total para ponderación
        saturacion_total = sum(
            self.puntos_saturacion.get(simbolo, CAPITAL_CONFIG['SATURACION_POR_DEFECTO'])
            for simbolo in instrumentos
        )
        
        # Primera pasada: asignación proporcional
        asignaciones = {}
        capital_restante = capital
        
        for simbolo in instrumentos:
            punto_saturacion = self.puntos_saturacion.get(simbolo, CAPITAL_CONFIG['SATURACION_POR_DEFECTO'])
            peso = punto_saturacion / saturacion_total if saturacion_total > 0 else 1.0 / len(instrumentos)
            asignacion = min(capital * peso, punto_saturacion)
            
            asignaciones[simbolo] = asignacion
            capital_restante -= asignacion
        
        # Segunda pasada: redistribuir capital no utilizado
        if capital_restante > 0.01:  # Si queda al menos 1 centavo
            no_saturados = [
                s for s in instrumentos 
                if asignaciones[s] < self.puntos_saturacion.get(s, CAPITAL_CONFIG['SATURACION_POR_DEFECTO'])
            ]
            
            if no_saturados:
                # Calcular pesos para redistribución
                pesos_redistribucion = {}
                total_peso = 0.0
                
                for simbolo in no_saturados:
                    sat_point = self.puntos_saturacion.get(simbolo, CAPITAL_CONFIG['SATURACION_POR_DEFECTO'])
                    margen = sat_point - asignaciones[simbolo]
                    pesos_redistribucion[simbolo] = margen
                    total_peso += margen
                
                # Redistribuir según margen disponible
                if total_peso > 0:
                    for simbolo, margen in pesos_redistribucion.items():
                        adicional = capital_restante * (margen / total_peso)
                        asignaciones[simbolo] += adicional
        
        return asignaciones

    def _calcular_max_instrumentos(self, factor_escala: float) -> int:
        """
        Calcular número máximo de instrumentos según escala de capital.
        
        Args:
            factor_escala: Factor de escala del capital
            
        Returns:
            Número máximo de instrumentos a utilizar
        """
        return int(
            CAPITAL_CONFIG['MAX_SIMBOLOS']['pequeño'] + 
            min(10, (CAPITAL_CONFIG['MAX_SIMBOLOS']['grande'] - CAPITAL_CONFIG['MAX_SIMBOLOS']['pequeño']) * 
                np.log10(max(1, factor_escala)))
        )

    async def _estimate_saturation_point(
        self, 
        simbolo: str, 
        metricas: MetricasInstrumento
    ) -> float:
        """
        Estimar punto de saturación para un instrumento.
        
        El punto de saturación es el nivel de capital donde la eficiencia
        comienza a deteriorarse por problemas de liquidez o impacto de mercado.
        
        Args:
            simbolo: Símbolo del instrumento
            metricas: Métricas del instrumento
            
        Returns:
            Punto de saturación estimado
        """
        # Verificar si ya tenemos este punto calculado
        if simbolo in self.puntos_saturacion:
            return self.puntos_saturacion[simbolo]

        try:
            # Ejecutar cálculo en pool de hilos para no bloquear
            saturacion_base = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._calcular_saturacion_base,
                metricas.capital_mercado,
                metricas.volumen_24h
            )
            
            # Ajustar según liquidez
            factor_liquidez = max(0.1, min(1.0, metricas.puntaje_liquidez))
            saturacion = max(10000.0, min(10000000.0, saturacion_base * factor_liquidez))
            
            self.puntos_saturacion[simbolo] = saturacion
            return saturacion
            
        except Exception as e:
            self.logger.warning(f"Error estimando saturación para {simbolo}: {str(e)}")
            return CAPITAL_CONFIG['SATURACION_POR_DEFECTO']

    @staticmethod
    def _calcular_saturacion_base(capital_mercado: float, volumen_24h: float) -> float:
        """
        Calcular saturación base según métricas de mercado.
        
        Args:
            capital_mercado: Capitalización de mercado
            volumen_24h: Volumen en 24 horas
            
        Returns:
            Saturación base estimada
        """
        if capital_mercado > 0 and volumen_24h > 0:
            # Tomar el menor valor entre:
            # - 0.01% del market cap (muy conservador)
            # - 0.1% del volumen diario (estándar en mercados líquidos)
            return min(capital_mercado * 0.0001, volumen_24h * 0.001)
            
        elif volumen_24h > 0:
            # Si sólo tenemos volumen
            return volumen_24h * 0.001
            
        elif capital_mercado > 0:
            # Si sólo tenemos market cap
            return capital_mercado * 0.0001
            
        # Valor por defecto conservador
        return 50000.0

    def _actualizar_metricas_rendimiento(self, capital: float, asignaciones: Dict[str, float]) -> None:
        """
        Actualizar métricas de rendimiento del gestor.
        
        Args:
            capital: Capital total disponible
            asignaciones: Asignaciones realizadas
        """
        # Calcular utilización de capital
        capital_utilizado = sum(asignaciones.values())
        utilizacion = capital_utilizado / capital if capital > 0 else 0.0
        
        # Calcular entropía de distribución (para medir diversificación)
        entropia = 0.0
        if asignaciones and capital > 0:
            proporciones = [monto / capital for monto in asignaciones.values()]
            for p in proporciones:
                if p > 0:
                    entropia -= p * np.log(p)
            # Normalizar entropía
            if len(proporciones) > 1:
                entropia /= np.log(len(proporciones))
        
        # Actualizar métricas
        self.metricas_rendimiento.update({
            "utilizacion_capital": utilizacion,
            "factor_escala": capital / self.capital_inicial if self.capital_inicial > 0 else 1.0,
            "entropia_asignacion": entropia,
            "ultima_actualizacion": datetime.now()
        })

    def _registrar_distribucion(
        self, 
        capital: float, 
        factor_escala: float, 
        asignaciones: Dict[str, float]
    ) -> None:
        """
        Registrar distribución en historial para análisis.
        
        Args:
            capital: Capital total
            factor_escala: Factor de escala
            asignaciones: Asignaciones realizadas
        """
        registro = {
            "timestamp": datetime.now().isoformat(),
            "capital_total": capital,
            "factor_escala": factor_escala,
            "num_instrumentos": len(asignaciones),
            "asignaciones": {k: v for k, v in asignaciones.items()},
            "metricas": self.metricas_rendimiento.copy()
        }
        
        self.historial_distribucion.append(registro)
        self.metricas_rendimiento["registros_distribucion"] += 1
        
        # Limitar tamaño del historial
        if len(self.historial_distribucion) > 100:
            self.historial_distribucion = self.historial_distribucion[-100:]

    async def analizar_eficiencia(
        self, 
        simbolo: str, 
        capital_desplegado: float, 
        datos_rendimiento: Dict[str, Any]
    ) -> float:
        """
        Analizar eficiencia del capital desplegado en un instrumento.
        
        Este método evalúa el rendimiento del capital y actualiza los
        modelos de eficiencia para futuras asignaciones.
        
        Args:
            simbolo: Símbolo del instrumento
            capital_desplegado: Capital asignado
            datos_rendimiento: Datos de rendimiento
            
        Returns:
            Puntuación de eficiencia (0-1)
        """
        try:
            # Validar datos mínimos necesarios
            if not simbolo or capital_desplegado <= 0:
                return 0.0
                
            # Extraer métricas relevantes
            roi = float(datos_rendimiento.get("roi", 0))
            sharpe = float(datos_rendimiento.get("sharpe_ratio", 0))
            operaciones = int(datos_rendimiento.get("operaciones", 0))
            win_rate = float(datos_rendimiento.get("win_rate", 0.5))
            
            # No podemos evaluar sin operaciones
            if operaciones <= 0:
                return 0.5  # Valor neutral
            
            # Calcular eficiencia base
            eficiencia_base = (
                0.4 * win_rate + 
                0.3 * max(0, min(3, sharpe)) / 3 + 
                0.3 * (1.0 if roi > 0 else max(0, 1 + roi))
            )
            
            # Almacenar en registro de eficiencia
            if simbolo not in self.registros_eficiencia:
                self.registros_eficiencia[simbolo] = {}
                
            self.registros_eficiencia[simbolo][capital_desplegado] = eficiencia_base
            
            # Analizar tendencia si tenemos suficientes puntos
            if len(self.registros_eficiencia[simbolo]) >= 3:
                await self._analizar_tendencia_eficiencia(simbolo)
            
            # Actualizar eficiencia promedio global
            todas_eficiencias = [
                eficiencia for registros in self.registros_eficiencia.values()
                for eficiencia in registros.values()
            ]
            
            if todas_eficiencias:
                self.metricas_rendimiento["eficiencia_promedio"] = sum(todas_eficiencias) / len(todas_eficiencias)
            
            return eficiencia_base
            
        except Exception as e:
            self.logger.error(f"Error analizando eficiencia para {simbolo}: {str(e)}")
            return 0.5  # Valor neutral en caso de error

    async def _analizar_tendencia_eficiencia(self, simbolo: str) -> Dict[str, Any]:
        """
        Analizar tendencia de eficiencia con diferentes niveles de capital.
        
        Este método examina cómo la eficiencia cambia a medida que el capital
        desplegado aumenta, identificando puntos de saturación y patrones
        para optimizar la asignación futura.
        
        Args:
            simbolo: Símbolo a analizar
            
        Returns:
            Diccionario con resultados del análisis
        """
        if simbolo not in self.registros_eficiencia:
            return {}
            
        # Obtener puntos ordenados por capital
        puntos = sorted(self.registros_eficiencia[simbolo].items())
        
        if len(puntos) < 2:
            return {}
            
        # Actualizar registros en el motor predictivo
        for capital, eficiencia in self.registros_eficiencia[simbolo].items():
            self.predictive_engine.add_efficiency_record(simbolo, capital, eficiencia)
        
        resultados = {}
        
        try:
            # Análisis básico con regresión lineal simple (para compatibilidad con versiones anteriores)
            x = np.array([p[0] for p in puntos])  # Capital
            y = np.array([p[1] for p in puntos])  # Eficiencia
            
            pendiente, ordenada = np.polyfit(x, y, 1)
            resultados["pendiente"] = pendiente
            resultados["ordenada"] = ordenada
            resultados["r2_lineal"] = self.predictive_engine._calculate_r2(x, y, lambda x: pendiente * x + ordenada)
            
            # Si la pendiente es negativa, hay deterioro de eficiencia
            if pendiente < -0.0001:  # Umbral de significancia
                # Proyectar punto de saturación
                ultimo_punto = puntos[-1]
                capital_actual = ultimo_punto[0]
                eficiencia_actual = ultimo_punto[1]
                
                # Proyectar dónde la eficiencia caería por debajo del umbral
                if eficiencia_actual > self.umbral_eficiencia:
                    capital_margen = (eficiencia_actual - self.umbral_eficiencia) / abs(pendiente)
                    saturacion_proyectada = capital_actual + capital_margen
                    resultados["saturacion_proyectada_lineal"] = saturacion_proyectada
                    
                    # Actualizar punto de saturación si es menor que el actual
                    saturacion_actual = self.puntos_saturacion.get(simbolo, CAPITAL_CONFIG['SATURACION_POR_DEFECTO'])
                    if saturacion_proyectada < saturacion_actual * 0.8:  # Solo actualizar si es significativamente menor
                        self.puntos_saturacion[simbolo] = saturacion_proyectada
                        self.logger.info(f"Punto de saturación para {simbolo} ajustado a ${saturacion_proyectada:,.2f} basado en análisis de eficiencia")
                        resultados["saturacion_actualizada"] = True
            
            # Análisis avanzado con motor predictivo
            # Generar proyecciones a futuro utilizando el motor predictivo
            capital_max = max(x) * 2  # Proyectar hasta el doble del capital actual
            pasos = 5
            proyecciones = []
            
            for i in range(1, pasos + 1):
                capital_proyectado = capital_actual + (capital_max - capital_actual) * (i / pasos)
                prediccion = self.predictive_engine.predict_efficiency(simbolo, capital_proyectado)
                proyecciones.append(prediccion.to_dict())
            
            resultados["proyecciones"] = proyecciones
            
            # Buscar punto de saturación proyectado (donde la eficiencia cae por debajo del umbral)
            for prediccion in proyecciones:
                if prediccion["efficiency"] < self.umbral_eficiencia:
                    saturacion_avanzada = prediccion["capital_level"]
                    resultados["saturacion_proyectada_avanzada"] = saturacion_avanzada
                    
                    # Actualizar punto de saturación si es más preciso que el modelo lineal
                    if prediccion["confidence"] > 0.7 and (
                        "saturacion_proyectada_lineal" not in resultados or 
                        saturacion_avanzada < resultados["saturacion_proyectada_lineal"] * 0.9
                    ):
                        self.puntos_saturacion[simbolo] = saturacion_avanzada
                        self.logger.info(f"Punto de saturación para {simbolo} actualizado a ${saturacion_avanzada:,.2f} con modelo predictivo avanzado")
                        resultados["saturacion_actualizada_avanzada"] = True
                    break
        
        except Exception as e:
            self.logger.warning(f"Error en análisis de tendencia para {simbolo}: {str(e)}")
            resultados["error"] = str(e)
        
        return resultados

    def get_saturation_points(self) -> Dict[str, float]:
        """
        Obtener puntos de saturación estimados.
        
        Returns:
            Diccionario con puntos de saturación por instrumento
        """
        return self.puntos_saturacion.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas de rendimiento del gestor.
        
        Returns:
            Diccionario con métricas
        """
        return {
            **self.metricas_rendimiento,
            "instrumentos_analizados": len(self.registros_eficiencia),
            "puntos_saturacion_estimados": len(self.puntos_saturacion)
        }
    
    def get_distribution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener historial de distribuciones.
        
        Args:
            limit: Número máximo de registros a devolver
            
        Returns:
            Lista de registros históricos
        """
        return self.historial_distribucion[-limit:]
        
    async def predict_capital_efficiency(
        self, 
        symbol: str, 
        capital_levels: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Predecir eficiencia para diferentes niveles de capital.
        
        Este método utiliza el motor predictivo para estimar cómo
        se comportará un instrumento con diferentes niveles de capital,
        permitiendo planificar escalabilidad a futuro.
        
        Args:
            symbol: Símbolo del instrumento
            capital_levels: Lista de niveles de capital a predecir
            
        Returns:
            Lista de predicciones con detalles
        """
        if not capital_levels:
            return []
            
        # Asegurar que el motor tiene todos los datos disponibles
        if symbol in self.registros_eficiencia:
            for capital, eficiencia in self.registros_eficiencia[symbol].items():
                self.predictive_engine.add_efficiency_record(symbol, capital, eficiencia)
        
        # Generar predicciones
        predicciones = []
        for capital_level in capital_levels:
            prediccion = self.predictive_engine.predict_efficiency(symbol, capital_level)
            predicciones.append(prediccion.to_dict())
            
        # Ordenar por nivel de capital
        return sorted(predicciones, key=lambda p: p["capital_level"])
    
    async def optimize_allocations_with_predictor(
        self,
        symbols: List[str],
        total_capital: float,
        min_efficiency: float = 0.7
    ) -> Dict[str, float]:
        """
        Optimizar asignación de capital utilizando el motor predictivo.
        
        Esta versión más avanzada que _calcular_asignaciones utiliza
        el motor predictivo para encontrar la distribución óptima que
        maximiza la eficiencia global del sistema.
        
        Args:
            symbols: Lista de símbolos disponibles
            total_capital: Capital total a distribuir
            min_efficiency: Eficiencia mínima aceptable
            
        Returns:
            Diccionario con asignación por símbolo
        """
        if not symbols or total_capital <= 0:
            return {}
        
        # Actualizar registros de eficiencia en el motor predictivo
        for symbol in symbols:
            if symbol in self.registros_eficiencia:
                for capital, eficiencia in self.registros_eficiencia[symbol].items():
                    self.predictive_engine.add_efficiency_record(symbol, capital, eficiencia)
        
        # Utilizar el optimizador del motor predictivo
        asignaciones = self.predictive_engine.optimize_allocation(
            symbols, total_capital, min_efficiency
        )
        
        return asignaciones
        
    async def _persist_allocation_history(
        self,
        capital: float,
        factor_escala: float,
        asignaciones: Dict[str, float]
    ) -> None:
        """
        Persistir historial de asignación en la base de datos.
        
        Args:
            capital: Capital total
            factor_escala: Factor de escala
            asignaciones: Asignaciones realizadas
        """
        if not self.db or not self.config.get('db_persistence_enabled', False):
            return
            
        try:
            from genesis.db.models.scaling_config_models import AllocationHistory
            
            # Crear nuevo registro de asignación
            allocation_record = AllocationHistory(
                config_id=self.config.get('id', 1),  # ID de configuración por defecto
                total_capital=capital,
                scale_factor=factor_escala,
                instruments_count=len(asignaciones),
                capital_utilization=self.metricas_rendimiento["utilizacion_capital"],
                entropy=self.metricas_rendimiento["entropia_asignacion"],
                efficiency_avg=self.metricas_rendimiento["eficiencia_promedio"],
                allocations=asignaciones,
                metrics=self.metricas_rendimiento
            )
            
            # Guardar en la base de datos
            async with self.db.transaction():
                await self.db.execute(
                    "INSERT INTO allocation_history (config_id, timestamp, total_capital, scale_factor, "
                    "instruments_count, capital_utilization, entropy, efficiency_avg, allocations, metrics) "
                    "VALUES (:config_id, :timestamp, :total_capital, :scale_factor, :instruments_count, "
                    ":capital_utilization, :entropy, :efficiency_avg, :allocations, :metrics)",
                    allocation_record.to_dict()
                )
                
            # Actualizar métricas
            if self.config.get('monitoring_enabled', False):
                scaling_metrics.increment("capital_scaling_db_allocations_saved")
                
            self.logger.debug(f"Asignación guardada en la base de datos: {capital:,.2f} USD, {len(asignaciones)} instrumentos")
            
        except Exception as e:
            self.logger.error(f"Error guardando asignación en la base de datos: {str(e)}")
            
            # Actualizar métricas de error
            if self.config.get('monitoring_enabled', False):
                scaling_metrics.increment("capital_scaling_db_errors")
                
    async def persist_saturation_points(self) -> None:
        """
        Persistir puntos de saturación actuales en la base de datos.
        
        Esto permite reutilizar estos valores en reinicios posteriores.
        """
        if not self.db or not self.config.get('db_persistence_enabled', False):
            return
            
        try:
            from genesis.db.models.scaling_config_models import SaturationPoint
            
            # Crear registros para cada punto de saturación
            async with self.db.transaction():
                # Primero eliminamos los registros anteriores
                await self.db.execute(
                    "DELETE FROM saturation_points WHERE config_id = :config_id",
                    {"config_id": self.config.get('id', 1)}
                )
                
                # Luego insertamos los nuevos
                for symbol, value in self.puntos_saturacion.items():
                    await self.db.execute(
                        "INSERT INTO saturation_points (config_id, symbol, saturation_value) "
                        "VALUES (:config_id, :symbol, :saturation_value)",
                        {
                            "config_id": self.config.get('id', 1),
                            "symbol": symbol,
                            "saturation_value": value
                        }
                    )
            
            # Actualizar métricas
            if self.config.get('monitoring_enabled', False):
                scaling_metrics.gauge("capital_scaling_saturation_points_saved", len(self.puntos_saturacion))
                
            self.logger.info(f"Puntos de saturación guardados en la base de datos: {len(self.puntos_saturacion)} símbolos")
            
        except Exception as e:
            self.logger.error(f"Error guardando puntos de saturación: {str(e)}")
            
            # Actualizar métricas de error
            if self.config.get('monitoring_enabled', False):
                scaling_metrics.increment("capital_scaling_db_errors")
                
    async def load_saturation_points(self) -> None:
        """
        Cargar puntos de saturación desde la base de datos.
        
        Esto permite restaurar los valores aprendidos en ejecuciones anteriores.
        """
        if not self.db or not self.config.get('db_persistence_enabled', False):
            return
            
        try:
            # Consultar los puntos de saturación para esta configuración
            rows = await self.db.fetch(
                "SELECT symbol, saturation_value FROM saturation_points WHERE config_id = :config_id",
                {"config_id": self.config.get('id', 1)}
            )
            
            # Actualizar diccionario en memoria
            if rows:
                for row in rows:
                    self.puntos_saturacion[row['symbol']] = row['saturation_value']
                
                self.logger.info(f"Puntos de saturación cargados desde la base de datos: {len(rows)} símbolos")
                
                # Actualizar métricas
                if self.config.get('monitoring_enabled', False):
                    scaling_metrics.gauge("capital_scaling_saturation_points_loaded", len(rows))
            else:
                self.logger.info("No se encontraron puntos de saturación en la base de datos")
                
        except Exception as e:
            self.logger.error(f"Error cargando puntos de saturación: {str(e)}")
            
            # Actualizar métricas de error
            if self.config.get('monitoring_enabled', False):
                scaling_metrics.increment("capital_scaling_db_errors")

class AccountTransaction:
    """
    Transacción en una cuenta.
    
    Representa un movimiento de fondos en una cuenta, como un depósito,
    retiro, o resultado de una operación de trading.
    """
    
    def __init__(
        self,
        transaction_id: str,
        account_id: str,
        timestamp: datetime,
        transaction_type: str,
        amount: Decimal,
        currency: str,
        description: str = "",
        related_trade_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializar una transacción.
        
        Args:
            transaction_id: Identificador único
            account_id: ID de la cuenta
            timestamp: Fecha y hora de la transacción
            transaction_type: Tipo (deposit, withdrawal, trade_profit, trade_loss, fee, etc.)
            amount: Cantidad (positiva para entradas, negativa para salidas)
            currency: Moneda
            description: Descripción opcional
            related_trade_id: ID de la operación relacionada (si aplica)
            metadata: Metadatos adicionales
        """
        self.transaction_id = transaction_id
        self.account_id = account_id
        self.timestamp = timestamp
        self.transaction_type = transaction_type
        self.amount = Decimal(str(amount))
        self.currency = currency
        self.description = description
        self.related_trade_id = related_trade_id
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir a diccionario.
        
        Returns:
            Diccionario con los datos de la transacción
        """
        return {
            "transaction_id": self.transaction_id,
            "account_id": self.account_id,
            "timestamp": format_timestamp(self.timestamp),
            "transaction_type": self.transaction_type,
            "amount": str(self.amount),
            "currency": self.currency,
            "description": self.description,
            "related_trade_id": self.related_trade_id,
            "metadata": self.metadata
        }
        
class Account:
    """
    Cuenta de un usuario o estrategia.
    
    Representa una cuenta en el sistema, manteniendo el saldo, historial
    de transacciones, y métricas de rendimiento.
    """
    
    def __init__(
        self,
        account_id: str,
        name: str,
        owner_id: str,
        initial_balance: Dict[str, Union[float, Decimal]] = None,
        account_type: str = "user",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializar una cuenta.
        
        Args:
            account_id: Identificador único
            name: Nombre de la cuenta
            owner_id: ID del propietario
            initial_balance: Saldo inicial por moneda
            account_type: Tipo de cuenta (user, strategy, system)
            metadata: Metadatos adicionales
        """
        self.account_id = account_id
        self.name = name
        self.owner_id = owner_id
        self.account_type = account_type
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.last_updated = self.created_at
        
        # Inicializar balances
        self.balances = {}
        if initial_balance:
            for currency, amount in initial_balance.items():
                self.balances[currency] = Decimal(str(amount))
                
        # Historial de transacciones y métricas
        self.transactions = []
        self.performance_metrics = {
            "total_deposits": Decimal("0"),
            "total_withdrawals": Decimal("0"),
            "total_profit": Decimal("0"),
            "total_loss": Decimal("0"),
            "total_fees": Decimal("0"),
            "roi": Decimal("0")
        }
        
    def add_transaction(self, transaction: AccountTransaction) -> bool:
        """
        Añadir una transacción a la cuenta.
        
        Args:
            transaction: Transacción a añadir
            
        Returns:
            True si se añadió correctamente, False en caso contrario
        """
        # Verificar que la transacción pertenece a esta cuenta
        if transaction.account_id != self.account_id:
            return False
            
        # Actualizar balance
        currency = transaction.currency
        if currency not in self.balances:
            self.balances[currency] = Decimal("0")
            
        self.balances[currency] += transaction.amount
        
        # Actualizar métricas según el tipo de transacción
        if transaction.transaction_type == "deposit":
            self.performance_metrics["total_deposits"] += transaction.amount
        elif transaction.transaction_type == "withdrawal":
            self.performance_metrics["total_withdrawals"] += abs(transaction.amount)
        elif transaction.transaction_type == "trade_profit":
            self.performance_metrics["total_profit"] += transaction.amount
        elif transaction.transaction_type == "trade_loss":
            self.performance_metrics["total_loss"] += abs(transaction.amount)
        elif transaction.transaction_type == "fee":
            self.performance_metrics["total_fees"] += abs(transaction.amount)
            
        # Calcular ROI
        total_in = self.performance_metrics["total_deposits"]
        if total_in > 0:
            net_profit = (self.performance_metrics["total_profit"] - 
                          self.performance_metrics["total_loss"] - 
                          self.performance_metrics["total_fees"])
            self.performance_metrics["roi"] = (net_profit / total_in) * 100
            
        # Añadir a historial y actualizar timestamp
        self.transactions.append(transaction)
        self.last_updated = datetime.now()
        
        return True
        
    def get_balance(self, currency: str) -> Decimal:
        """
        Obtener el saldo de una moneda.
        
        Args:
            currency: Código de la moneda
            
        Returns:
            Saldo actual
        """
        return self.balances.get(currency, Decimal("0"))
        
    def get_transactions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        transaction_type: Optional[str] = None,
        currency: Optional[str] = None,
        limit: int = 100
    ) -> List[AccountTransaction]:
        """
        Obtener transacciones filtradas.
        
        Args:
            start_date: Fecha inicial
            end_date: Fecha final
            transaction_type: Tipo de transacción
            currency: Moneda
            limit: Límite de resultados
            
        Returns:
            Lista de transacciones
        """
        filtered = self.transactions
        
        if start_date:
            filtered = [t for t in filtered if t.timestamp >= start_date]
            
        if end_date:
            filtered = [t for t in filtered if t.timestamp <= end_date]
            
        if transaction_type:
            filtered = [t for t in filtered if t.transaction_type == transaction_type]
            
        if currency:
            filtered = [t for t in filtered if t.currency == currency]
            
        # Ordenar por fecha descendente y limitar
        filtered.sort(key=lambda t: t.timestamp, reverse=True)
        return filtered[:limit]
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir a diccionario.
        
        Returns:
            Diccionario con los datos de la cuenta
        """
        return {
            "account_id": self.account_id,
            "name": self.name,
            "owner_id": self.owner_id,
            "account_type": self.account_type,
            "balances": {k: str(v) for k, v in self.balances.items()},
            "created_at": format_timestamp(self.created_at),
            "last_updated": format_timestamp(self.last_updated),
            "performance_metrics": {k: str(v) for k, v in self.performance_metrics.items()},
            "metadata": self.metadata
        }
        
class BalanceManager(Component):
    """
    Gestor de balances y cuentas.
    
    Este componente gestiona las cuentas de usuarios y estrategias,
    procesa transacciones, y calcula métricas de rendimiento.
    """
    
    def __init__(self, name: str = "balance_manager"):
        """
        Inicializar el gestor de balances.
        
        Args:
            name: Nombre del componente
        """
        super().__init__(name)
        self.logger = logging.getLogger(__name__)
        self.accounts = {}  # account_id -> Account
        self.trade_balances = {}  # trade_id -> Dict[str, Any]
        
    async def start(self) -> None:
        """Iniciar el gestor de balances."""
        await super().start()
        self.logger.info("Gestor de balances iniciado")
        
    async def stop(self) -> None:
        """Detener el gestor de balances."""
        await super().stop()
        self.logger.info("Gestor de balances detenido")
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente origen
        """
        if event_type == "trade.opened":
            # Registrar saldo de la operación
            await self._handle_trade_opened(data)
            
        elif event_type == "trade.closed":
            # Actualizar saldo con el resultado de la operación
            await self._handle_trade_closed(data)
            
        elif event_type == "account.deposit":
            # Procesar depósito
            account_id = data.get("account_id")
            amount = data.get("amount")
            currency = data.get("currency")
            
            if account_id and amount is not None and currency:
                await self.deposit(account_id, amount, currency, data.get("description", ""), data.get("metadata"))
                
        elif event_type == "account.withdrawal":
            # Procesar retiro
            account_id = data.get("account_id")
            amount = data.get("amount")
            currency = data.get("currency")
            
            if account_id and amount is not None and currency:
                await self.withdraw(account_id, amount, currency, data.get("description", ""), data.get("metadata"))
                
    async def _handle_trade_opened(self, data: Dict[str, Any]) -> None:
        """
        Manejar evento de operación abierta.
        
        Args:
            data: Datos de la operación
        """
        trade_id = data.get("trade_id")
        account_id = data.get("account_id")
        
        if not trade_id or not account_id:
            self.logger.error("ID de operación o cuenta faltante en evento trade.opened")
            return
            
        # Registrar operación
        self.trade_balances[trade_id] = {
            "trade_id": trade_id,
            "account_id": account_id,
            "entry_price": data.get("entry_price"),
            "amount": data.get("amount"),
            "symbol": data.get("symbol"),
            "side": data.get("side"),
            "timestamp": datetime.now(),
            "entry_fee": data.get("fee", 0)
        }
        
        # Aplicar comisión de entrada si existe
        if data.get("fee"):
            fee = Decimal(str(data.get("fee")))
            currency = data.get("fee_currency", "USDT")
            
            # Crear transacción para la comisión
            transaction = AccountTransaction(
                transaction_id=generate_id(),
                account_id=account_id,
                timestamp=datetime.now(),
                transaction_type="fee",
                amount=-fee,
                currency=currency,
                description=f"Comisión de entrada para operación {trade_id}",
                related_trade_id=trade_id,
                metadata={"fee_type": "entry", "trade_symbol": data.get("symbol")}
            )
            
            # Obtener la cuenta
            account = self.accounts.get(account_id)
            if account:
                account.add_transaction(transaction)
                
                # Emitir evento de actualización
                await self.emit_event("account.updated", {
                    "account_id": account_id,
                    "balance_change": {
                        "currency": currency,
                        "amount": -fee,
                        "reason": "trade_fee"
                    }
                })
        
    async def _handle_trade_closed(self, data: Dict[str, Any]) -> None:
        """
        Manejar evento de operación cerrada.
        
        Args:
            data: Datos de la operación
        """
        trade_id = data.get("trade_id")
        account_id = data.get("account_id")
        
        if not trade_id or not account_id:
            self.logger.error("ID de operación o cuenta faltante en evento trade.closed")
            return
            
        # Obtener detalles de la operación
        trade_entry = self.trade_balances.get(trade_id)
        if not trade_entry:
            self.logger.error(f"No se encontró registro de la operación {trade_id}")
            return
            
        # Calcular rentabilidad
        exit_price = data.get("exit_price")
        amount = Decimal(str(trade_entry.get("amount", 0)))
        entry_price = Decimal(str(trade_entry.get("entry_price", 0)))
        side = trade_entry.get("side", "buy")
        
        if exit_price and amount > 0 and entry_price > 0:
            exit_price = Decimal(str(exit_price))
            
            # Diferentes cálculos según el lado (compra/venta)
            if side == "buy":  # Compra (long)
                pnl = (exit_price - entry_price) * amount
            else:  # Venta (short)
                pnl = (entry_price - exit_price) * amount
                
            # Aplicar comisión de salida si existe
            exit_fee = Decimal(str(data.get("fee", 0)))
            pnl -= exit_fee
            
            # Registro de comisión de salida
            if exit_fee > 0:
                fee_currency = data.get("fee_currency", "USDT")
                
                # Crear transacción para la comisión
                fee_transaction = AccountTransaction(
                    transaction_id=generate_id(),
                    account_id=account_id,
                    timestamp=datetime.now(),
                    transaction_type="fee",
                    amount=-exit_fee,
                    currency=fee_currency,
                    description=f"Comisión de salida para operación {trade_id}",
                    related_trade_id=trade_id,
                    metadata={"fee_type": "exit", "trade_symbol": trade_entry.get("symbol")}
                )
                
                # Añadir transacción a la cuenta
                account = self.accounts.get(account_id)
                if account:
                    account.add_transaction(fee_transaction)
            
            # Crear transacción para el resultado de la operación
            result_type = "trade_profit" if pnl > 0 else "trade_loss"
            result_currency = data.get("profit_currency", "USDT")
            
            result_transaction = AccountTransaction(
                transaction_id=generate_id(),
                account_id=account_id,
                timestamp=datetime.now(),
                transaction_type=result_type,
                amount=pnl,
                currency=result_currency,
                description=f"Resultado de operación {trade_id} - {trade_entry.get('symbol')}",
                related_trade_id=trade_id,
                metadata={
                    "symbol": trade_entry.get("symbol"),
                    "side": side,
                    "entry_price": str(entry_price),
                    "exit_price": str(exit_price),
                    "amount": str(amount),
                    "pnl_percentage": str((pnl / (entry_price * amount)) * 100)
                }
            )
            
            # Añadir transacción a la cuenta
            account = self.accounts.get(account_id)
            if account:
                account.add_transaction(result_transaction)
                
                # Emitir evento de actualización
                await self.emit_event("account.updated", {
                    "account_id": account_id,
                    "balance_change": {
                        "currency": result_currency,
                        "amount": pnl,
                        "reason": "trade_result"
                    },
                    "trade_result": {
                        "trade_id": trade_id,
                        "symbol": trade_entry.get("symbol"),
                        "side": side,
                        "pnl": str(pnl),
                        "pnl_percentage": str((pnl / (entry_price * amount)) * 100)
                    }
                })
                
        # Limpiar entrada de trade_balances
        self.trade_balances.pop(trade_id, None)
        
    async def create_account(
        self,
        name: str,
        owner_id: str,
        initial_balance: Optional[Dict[str, Union[float, Decimal]]] = None,
        account_type: str = "user",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Crear una nueva cuenta.
        
        Args:
            name: Nombre de la cuenta
            owner_id: ID del propietario
            initial_balance: Saldo inicial por moneda
            account_type: Tipo de cuenta
            metadata: Metadatos adicionales
            
        Returns:
            ID de la cuenta creada
        """
        account_id = generate_id()
        
        account = Account(
            account_id=account_id,
            name=name,
            owner_id=owner_id,
            initial_balance=initial_balance,
            account_type=account_type,
            metadata=metadata
        )
        
        self.accounts[account_id] = account
        
        # Registrar depósitos iniciales
        if initial_balance:
            for currency, amount in initial_balance.items():
                decimal_amount = Decimal(str(amount))
                
                # Solo crear transacción si el monto es positivo
                if decimal_amount > 0:
                    transaction = AccountTransaction(
                        transaction_id=generate_id(),
                        account_id=account_id,
                        timestamp=datetime.now(),
                        transaction_type="deposit",
                        amount=decimal_amount,
                        currency=currency,
                        description="Depósito inicial"
                    )
                    
                    account.add_transaction(transaction)
        
        # Emitir evento de cuenta creada
        await self.emit_event("account.created", {
            "account_id": account_id,
            "name": name,
            "owner_id": owner_id,
            "account_type": account_type,
            "initial_balance": {k: str(v) for k, v in (initial_balance or {}).items()}
        })
        
        self.logger.info(f"Cuenta creada: {account_id} para {owner_id}")
        return account_id
        
    async def deposit(
        self,
        account_id: str,
        amount: Union[float, Decimal],
        currency: str,
        description: str = "Depósito",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Realizar un depósito en una cuenta.
        
        Args:
            account_id: ID de la cuenta
            amount: Cantidad a depositar
            currency: Moneda
            description: Descripción
            metadata: Metadatos adicionales
            
        Returns:
            True si se realizó correctamente, False en caso contrario
        """
        if account_id not in self.accounts:
            self.logger.error(f"Cuenta no encontrada: {account_id}")
            return False
            
        decimal_amount = Decimal(str(amount))
        if decimal_amount <= 0:
            self.logger.error(f"Monto de depósito inválido: {amount}")
            return False
            
        # Crear transacción
        transaction = AccountTransaction(
            transaction_id=generate_id(),
            account_id=account_id,
            timestamp=datetime.now(),
            transaction_type="deposit",
            amount=decimal_amount,
            currency=currency,
            description=description,
            metadata=metadata
        )
        
        # Añadir a la cuenta
        result = self.accounts[account_id].add_transaction(transaction)
        
        if result:
            # Emitir evento
            await self.emit_event("account.deposited", {
                "account_id": account_id,
                "amount": str(decimal_amount),
                "currency": currency,
                "transaction_id": transaction.transaction_id,
                "new_balance": str(self.accounts[account_id].get_balance(currency))
            })
            
        return result
        
    async def withdraw(
        self,
        account_id: str,
        amount: Union[float, Decimal],
        currency: str,
        description: str = "Retiro",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Realizar un retiro de una cuenta.
        
        Args:
            account_id: ID de la cuenta
            amount: Cantidad a retirar
            currency: Moneda
            description: Descripción
            metadata: Metadatos adicionales
            
        Returns:
            True si se realizó correctamente, False en caso contrario
        """
        if account_id not in self.accounts:
            self.logger.error(f"Cuenta no encontrada: {account_id}")
            return False
            
        decimal_amount = Decimal(str(amount))
        if decimal_amount <= 0:
            self.logger.error(f"Monto de retiro inválido: {amount}")
            return False
            
        # Verificar saldo suficiente
        account = self.accounts[account_id]
        if account.get_balance(currency) < decimal_amount:
            self.logger.error(f"Saldo insuficiente para retiro: {account.get_balance(currency)} < {decimal_amount}")
            return False
            
        # Crear transacción (monto negativo para retiros)
        transaction = AccountTransaction(
            transaction_id=generate_id(),
            account_id=account_id,
            timestamp=datetime.now(),
            transaction_type="withdrawal",
            amount=-decimal_amount,  # Negativo para retiros
            currency=currency,
            description=description,
            metadata=metadata
        )
        
        # Añadir a la cuenta
        result = account.add_transaction(transaction)
        
        if result:
            # Emitir evento
            await self.emit_event("account.withdrawn", {
                "account_id": account_id,
                "amount": str(decimal_amount),
                "currency": currency,
                "transaction_id": transaction.transaction_id,
                "new_balance": str(account.get_balance(currency))
            })
            
        return result
        
    def get_account(self, account_id: str) -> Optional[Account]:
        """
        Obtener una cuenta por su ID.
        
        Args:
            account_id: ID de la cuenta
            
        Returns:
            Cuenta o None si no existe
        """
        return self.accounts.get(account_id)
        
    def get_accounts_by_owner(self, owner_id: str) -> List[Account]:
        """
        Obtener cuentas por propietario.
        
        Args:
            owner_id: ID del propietario
            
        Returns:
            Lista de cuentas
        """
        return [a for a in self.accounts.values() if a.owner_id == owner_id]
        
    def get_account_balance(self, account_id: str, currency: str) -> Decimal:
        """
        Obtener saldo de una cuenta.
        
        Args:
            account_id: ID de la cuenta
            currency: Moneda
            
        Returns:
            Saldo actual (0 si no existe)
        """
        account = self.get_account(account_id)
        if account:
            return account.get_balance(currency)
        return Decimal("0")
        
    def get_account_transactions(
        self,
        account_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        transaction_type: Optional[str] = None,
        currency: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Obtener transacciones de una cuenta.
        
        Args:
            account_id: ID de la cuenta
            start_date: Fecha inicial
            end_date: Fecha final
            transaction_type: Tipo de transacción
            currency: Moneda
            limit: Límite de resultados
            
        Returns:
            Lista de transacciones como diccionarios
        """
        account = self.get_account(account_id)
        if not account:
            return []
            
        transactions = account.get_transactions(
            start_date=start_date,
            end_date=end_date,
            transaction_type=transaction_type,
            currency=currency,
            limit=limit
        )
        
        return [t.to_dict() for t in transactions]
        
    def get_account_performance(self, account_id: str) -> Dict[str, Any]:
        """
        Obtener métricas de rendimiento de una cuenta.
        
        Args:
            account_id: ID de la cuenta
            
        Returns:
            Métricas de rendimiento
        """
        account = self.get_account(account_id)
        if not account:
            return {}
            
        metrics = {k: str(v) for k, v in account.performance_metrics.items()}
        
        # Añadir métricas adicionales
        trades = [t for t in account.transactions if t.transaction_type in ["trade_profit", "trade_loss"]]
        winning_trades = [t for t in trades if t.transaction_type == "trade_profit"]
        losing_trades = [t for t in trades if t.transaction_type == "trade_loss"]
        
        metrics["total_trades"] = str(len(trades))
        metrics["winning_trades"] = str(len(winning_trades))
        metrics["losing_trades"] = str(len(losing_trades))
        
        if trades:
            win_rate = (len(winning_trades) / len(trades)) * 100
            metrics["win_rate"] = str(win_rate)
            
        # Calcular factor de beneficio (profit factor)
        if account.performance_metrics["total_loss"] > 0:
            profit_factor = account.performance_metrics["total_profit"] / account.performance_metrics["total_loss"]
            metrics["profit_factor"] = str(profit_factor)
            
        return metrics