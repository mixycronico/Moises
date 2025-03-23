"""
Motor predictivo para escalabilidad adaptativa.

Este módulo implementa el sistema de predicción de eficiencia para diferentes niveles
de capital, permitiendo optimizar la asignación de capital entre instrumentos.
"""

import logging
import asyncio
from datetime import datetime
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set

class EfficiencyPrediction:
    """
    Predicción de eficiencia para un nivel de capital específico.
    
    Esta clase encapsula los resultados de una predicción de eficiencia
    para un instrumento con un nivel de capital determinado.
    """
    
    def __init__(
        self,
        symbol: str,
        capital_level: float,
        efficiency: float,
        confidence: float = 0.0,
        prediction_type: str = "model",
        metrics: Dict[str, Any] = None
    ):
        """
        Inicializar predicción.
        
        Args:
            symbol: Símbolo del instrumento
            capital_level: Nivel de capital
            efficiency: Eficiencia predicha (0-1)
            confidence: Nivel de confianza (0-1)
            prediction_type: Tipo de predicción ('model', 'historical', 'extrapolation')
            metrics: Métricas adicionales
        """
        self.symbol = symbol
        self.capital_level = capital_level
        self.efficiency = max(0.0, min(1.0, efficiency))  # Limitamos a 0-1
        self.confidence = confidence
        self.prediction_type = prediction_type
        self.metrics = metrics or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir predicción a diccionario.
        
        Returns:
            Diccionario con los datos de la predicción
        """
        return {
            "symbol": self.symbol,
            "capital_level": self.capital_level,
            "efficiency": self.efficiency,
            "confidence": self.confidence,
            "prediction_type": self.prediction_type,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat()
        }


class PredictiveModel:
    """
    Modelo predictivo para la eficiencia de capital.
    
    Esta clase implementa un modelo que predice la eficiencia
    de un instrumento para diferentes niveles de capital.
    """
    
    def __init__(self, symbol: str, model_type: str = "polynomial"):
        """
        Inicializar modelo.
        
        Args:
            symbol: Símbolo del instrumento
            model_type: Tipo de modelo ('linear', 'polynomial', 'exponential')
        """
        self.symbol = symbol
        self.model_type = model_type
        self.parameters = {}
        self.data_points = []
        self.r_squared = 0.0
        self.mean_error = 0.0
        self.max_error = 0.0
        self.valid_range = (0.0, float('inf'))
        self.training_timestamp = None
        self.saturation_point = None
        self.is_trained = False
    
    def add_data_point(self, capital: float, efficiency: float) -> None:
        """
        Añadir punto de datos al modelo.
        
        Args:
            capital: Nivel de capital
            efficiency: Eficiencia observada
        """
        # Asegurar que no duplicamos puntos
        for i, (c, _) in enumerate(self.data_points):
            if abs(c - capital) < 0.001:
                # Actualizar punto existente
                self.data_points[i] = (capital, efficiency)
                return
        
        # Añadir nuevo punto
        self.data_points.append((capital, efficiency))
        # Marcar como no entrenado
        self.is_trained = False
        
    def train(self) -> bool:
        """
        Entrenar modelo con puntos disponibles.
        
        Returns:
            True si el entrenamiento fue exitoso
        """
        if len(self.data_points) < 3:
            return False
        
        # Ordenar puntos por nivel de capital
        self.data_points.sort(key=lambda x: x[0])
        
        # Separar arrays para X e Y
        x = np.array([p[0] for p in self.data_points])
        y = np.array([p[1] for p in self.data_points])
        
        try:
            if self.model_type == 'linear':
                # Modelo lineal: y = a*x + b
                a, b = np.polyfit(x, y, 1)
                self.parameters = {'a': float(a), 'b': float(b)}
                
                # Calcular predicciones
                y_pred = a * x + b
                
            elif self.model_type == 'exponential':
                # Modelo exponencial: y = a * exp(-b*x) + c
                
                # Estimación inicial
                p0 = [1.0, 0.001, 0.1]
                
                # Función de ajuste
                def exp_func(x, a, b, c):
                    return a * np.exp(-b * x) + c
                
                # Ajustar modelo
                from scipy.optimize import curve_fit
                popt, _ = curve_fit(exp_func, x, y, p0=p0, maxfev=10000)
                a, b, c = popt
                
                self.parameters = {'a': float(a), 'b': float(b), 'c': float(c)}
                
                # Calcular predicciones
                y_pred = exp_func(x, a, b, c)
                
                # Determinar punto de saturación
                # Lo definimos como el punto donde la eficiencia cae por debajo de 0.5
                xs = np.linspace(x[-1], x[-1] * 10, 1000)
                ys = exp_func(xs, a, b, c)
                saturated = np.where(ys < 0.5)[0]
                if len(saturated) > 0:
                    self.saturation_point = float(xs[saturated[0]])
                
            else:  # default: polynomial
                # Modelo polinomial: y = a*x^2 + b*x + c
                coeffs = np.polyfit(x, y, 2)
                a, b, c = coeffs
                self.parameters = {'a': float(a), 'b': float(b), 'c': float(c)}
                
                # Calcular predicciones
                y_pred = a * x**2 + b * x + c
                
                # Determinar punto de saturación si la curva es descendente
                if a < 0:
                    # Punto donde la derivada es cero (máximo)
                    x_max = -b / (2 * a)
                    # Si el máximo está a la derecha de los datos, es válido
                    if x_max > x[-1]:
                        self.saturation_point = float(x_max)
            
            # Calcular métricas de ajuste
            residuals = y - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            
            if ss_tot > 0:
                self.r_squared = 1 - (ss_res / ss_tot)
            else:
                self.r_squared = 0.0
                
            self.mean_error = float(np.mean(np.abs(residuals)))
            self.max_error = float(np.max(np.abs(residuals)))
            
            # Rango válido del modelo
            self.valid_range = (float(min(x)), float(max(x)) * 2)
            
            self.training_timestamp = datetime.now()
            self.is_trained = True
            
            return True
            
        except Exception as e:
            logging.error(f"Error entrenando modelo para {self.symbol}: {str(e)}")
            return False
    
    def predict(self, capital: float) -> float:
        """
        Predecir eficiencia para un nivel de capital.
        
        Args:
            capital: Nivel de capital a predecir
            
        Returns:
            Eficiencia predicha (0-1)
        """
        if not self.is_trained:
            if not self.train():
                # Si no se puede entrenar, usamos la eficiencia del punto más cercano
                if not self.data_points:
                    return 0.5  # Valor por defecto si no hay datos
                    
                # Encontrar el punto más cercano
                closest = min(self.data_points, key=lambda p: abs(p[0] - capital))
                return closest[1]
        
        # Predecir con el modelo entrenado
        if self.model_type == 'linear':
            a, b = self.parameters['a'], self.parameters['b']
            prediction = a * capital + b
            
        elif self.model_type == 'exponential':
            a, b, c = self.parameters['a'], self.parameters['b'], self.parameters['c']
            prediction = a * np.exp(-b * capital) + c
            
        else:  # polynomial
            a, b, c = self.parameters['a'], self.parameters['b'], self.parameters['c']
            prediction = a * capital**2 + b * capital + c
        
        # Verificar límites
        if capital < self.valid_range[0]:
            # Si estamos por debajo del rango, usamos el primer punto
            #return self.data_points[0][1]
            # Mejor usamos la predicción pero con menor confianza
            pass
            
        elif capital > self.valid_range[1]:
            # Si estamos por encima del rango, aplicamos una degradación adicional
            # basada en la distancia al rango válido
            excess_factor = (capital - self.valid_range[1]) / self.valid_range[1]
            degradation = 0.1 * min(1.0, excess_factor)
            prediction = max(0.1, prediction - degradation)
        
        # Asegurar que la predicción está en el rango [0,1]
        return max(0.0, min(1.0, prediction))
    
    def get_confidence(self, capital: float) -> float:
        """
        Obtener nivel de confianza para una predicción.
        
        Args:
            capital: Nivel de capital
            
        Returns:
            Confianza de la predicción (0-1)
        """
        if not self.is_trained or not self.data_points:
            return 0.1
        
        # La confianza depende de:
        # 1. La calidad del modelo (r^2)
        # 2. La cercanía a los puntos de datos
        # 3. Si estamos dentro del rango válido
        
        # Factor de calidad del modelo
        model_quality = max(0.1, min(1.0, self.r_squared))
        
        # Factor de cercanía a los datos
        data_x = np.array([p[0] for p in self.data_points])
        min_distance = min(abs(data_x - capital)) / max(1.0, capital)
        proximity_factor = np.exp(-3 * min_distance)  # Decae exponencialmente con la distancia
        
        # Factor de rango válido
        if self.valid_range[0] <= capital <= self.valid_range[1]:
            range_factor = 1.0
        else:
            # Qué tan lejos estamos del rango válido
            if capital < self.valid_range[0]:
                distance = (self.valid_range[0] - capital) / max(1.0, self.valid_range[0])
            else:
                distance = (capital - self.valid_range[1]) / max(1.0, self.valid_range[1])
            range_factor = np.exp(-2 * distance)  # Decae exponencialmente
            
        # Calcular confianza combinada
        confidence = 0.7 * model_quality + 0.2 * proximity_factor + 0.1 * range_factor
        
        return max(0.1, min(0.95, confidence))  # Limitamos a [0.1, 0.95]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir modelo a diccionario para almacenamiento.
        
        Returns:
            Diccionario con los datos del modelo
        """
        return {
            "symbol": self.symbol,
            "model_type": self.model_type,
            "parameters": self.parameters,
            "r_squared": self.r_squared,
            "mean_error": self.mean_error,
            "max_error": self.max_error,
            "valid_range": list(self.valid_range),
            "training_timestamp": self.training_timestamp.isoformat() if self.training_timestamp else None,
            "saturation_point": self.saturation_point,
            "is_trained": self.is_trained,
            "data_points_count": len(self.data_points)
        }


class PredictiveScalingEngine:
    """
    Motor predictivo para escalabilidad adaptativa.
    
    Esta clase coordina el análisis de eficiencia a diferentes niveles
    de capital y proporciona predicciones para optimizar la asignación.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializar motor predictivo.
        
        Args:
            config: Configuración del motor
        """
        self.logger = logging.getLogger('genesis.accounting.predictive_scaling')
        self.config = config or {}
        
        # Modelos por símbolo
        self.models: Dict[str, PredictiveModel] = {}
        
        # Caché de predicciones recientes
        self.prediction_cache: Dict[str, Dict[float, EfficiencyPrediction]] = {}
        
        # Configuración
        self.default_model_type = self.config.get('default_model_type', 'polynomial')
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutos
        self.auto_train = self.config.get('auto_train', True)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        # Métricas
        self.stats = {
            "prediction_requests": 0,
            "cache_hits": 0,
            "model_trainings": 0,
            "optimization_runs": 0
        }
        
        self.logger.info(f"PredictiveScalingEngine inicializado con modelo {self.default_model_type}")
    
    def add_efficiency_record(
        self, 
        symbol: str, 
        capital: float, 
        efficiency: float, 
        metrics: Dict[str, Any] = None
    ) -> None:
        """
        Añadir registro de eficiencia observada.
        
        Este método alimenta los modelos predictivos con datos
        históricos de eficiencia observada.
        
        Args:
            symbol: Símbolo del instrumento
            capital: Nivel de capital
            efficiency: Eficiencia observada (0-1)
            metrics: Métricas adicionales
        """
        # Obtener o crear modelo
        if symbol not in self.models:
            self.models[symbol] = PredictiveModel(symbol, self.default_model_type)
        
        # Añadir punto de datos
        self.models[symbol].add_data_point(capital, efficiency)
        
        # Borrar caché para este símbolo
        if symbol in self.prediction_cache:
            self.prediction_cache[symbol] = {}
        
        # Entrenar automáticamente si está configurado
        if self.auto_train and len(self.models[symbol].data_points) >= 3:
            self.models[symbol].train()
            self.stats["model_trainings"] += 1
    
    def train_model(self, symbol: str, model_type: Optional[str] = None) -> bool:
        """
        Entrenar modelo para un símbolo específico.
        
        Args:
            symbol: Símbolo del instrumento
            model_type: Tipo de modelo a usar (opcional)
            
        Returns:
            True si el entrenamiento fue exitoso
        """
        if symbol not in self.models:
            if not model_type:
                model_type = self.default_model_type
            self.models[symbol] = PredictiveModel(symbol, model_type)
            return False  # No hay datos para entrenar
        
        if model_type and model_type != self.models[symbol].model_type:
            # Cambiar tipo de modelo
            old_data = self.models[symbol].data_points
            self.models[symbol] = PredictiveModel(symbol, model_type)
            for capital, efficiency in old_data:
                self.models[symbol].add_data_point(capital, efficiency)
        
        # Entrenar modelo
        result = self.models[symbol].train()
        if result:
            self.stats["model_trainings"] += 1
            
            # Borrar caché para este símbolo
            if symbol in self.prediction_cache:
                self.prediction_cache[symbol] = {}
        
        return result
        
    def predict_efficiency(
        self, 
        symbol: str, 
        capital: float
    ) -> EfficiencyPrediction:
        """
        Predecir eficiencia para un nivel de capital.
        
        Args:
            symbol: Símbolo del instrumento
            capital: Nivel de capital a predecir
            
        Returns:
            Objeto EfficiencyPrediction con la predicción
        """
        self.stats["prediction_requests"] += 1
        
        # Verificar caché
        if symbol in self.prediction_cache and capital in self.prediction_cache[symbol]:
            cache_entry = self.prediction_cache[symbol][capital]
            # Verificar si la caché es válida (no expirada)
            age = (datetime.now() - cache_entry.timestamp).total_seconds()
            if age < self.cache_ttl:
                self.stats["cache_hits"] += 1
                return cache_entry
        
        # Si no está en caché o está expirada, calcular predicción
        if symbol not in self.models:
            # No hay modelo para este símbolo, crear uno
            self.models[symbol] = PredictiveModel(symbol, self.default_model_type)
            
            # Sin datos, devolvemos una predicción genérica
            prediction = EfficiencyPrediction(
                symbol=symbol,
                capital_level=capital,
                efficiency=0.8,  # Valor optimista por defecto
                confidence=0.1,  # Baja confianza
                prediction_type="default",
                metrics={"source": "default"}
            )
        else:
            # Entrenar si es necesario
            if not self.models[symbol].is_trained:
                self.train_model(symbol)
            
            # Obtener predicción
            efficiency = self.models[symbol].predict(capital)
            confidence = self.models[symbol].get_confidence(capital)
            
            prediction = EfficiencyPrediction(
                symbol=symbol,
                capital_level=capital,
                efficiency=efficiency,
                confidence=confidence,
                prediction_type="model",
                metrics={
                    "model_type": self.models[symbol].model_type,
                    "r_squared": self.models[symbol].r_squared,
                    "valid_range": self.models[symbol].valid_range
                }
            )
        
        # Guardar en caché
        if symbol not in self.prediction_cache:
            self.prediction_cache[symbol] = {}
        self.prediction_cache[symbol][capital] = prediction
        
        return prediction
    
    def optimize_allocation(
        self,
        symbols: List[str],
        total_capital: float,
        min_efficiency: float = 0.7
    ) -> Dict[str, float]:
        """
        Optimizar asignación de capital entre varios instrumentos.
        
        Este método distribuye el capital total entre los instrumentos
        disponibles, maximizando la eficiencia global.
        
        Args:
            symbols: Lista de símbolos disponibles
            total_capital: Capital total a distribuir
            min_efficiency: Eficiencia mínima aceptable
            
        Returns:
            Diccionario con asignación por símbolo
        """
        if not symbols or total_capital <= 0:
            return {}
        
        self.stats["optimization_runs"] += 1
        
        # Inicializar asignación vacía
        allocation: Dict[str, float] = {}
        
        # Enfoque: asignación incremental ponderada por eficiencia
        # Para cada símbolo, exploraremos cómo varía su eficiencia
        # con distintos niveles de capital y asignaremos proporcionalmente
        
        # Paso 1: Evaluar eficiencia a diferentes niveles para cada símbolo
        symbol_curves: Dict[str, List[Tuple[float, float]]] = {}
        
        # Exploraremos 10 niveles desde total_capital/20 hasta total_capital/len(symbols)
        min_capital = max(100.0, total_capital / 20)
        max_capital = total_capital / max(1, len(symbols) - 1)
        
        levels = np.linspace(min_capital, max_capital, 10).tolist()
        
        for symbol in symbols:
            symbol_curves[symbol] = []
            for level in levels:
                prediction = self.predict_efficiency(symbol, level)
                if prediction.confidence >= self.confidence_threshold:
                    symbol_curves[symbol].append((level, prediction.efficiency))
        
        # Paso 2: Filtrar símbolos sin suficientes datos
        valid_symbols = [s for s in symbols if len(symbol_curves[s]) >= 3]
        
        if not valid_symbols:
            # Si no hay símbolos con datos suficientes, distribuimos equitativamente
            equal_capital = total_capital / len(symbols)
            for symbol in symbols:
                allocation[symbol] = equal_capital
            return allocation
        
        # Paso 3: Calcular la utilidad marginal (derivada de eficiencia / capital)
        # para cada símbolo en diferentes niveles
        marginal_utility: Dict[str, List[Tuple[float, float]]] = {}
        
        for symbol in valid_symbols:
            curve = symbol_curves[symbol]
            marginal_utility[symbol] = []
            
            # Calcular la derivada (tasa de cambio)
            for i in range(1, len(curve)):
                cap1, eff1 = curve[i-1]
                cap2, eff2 = curve[i]
                cap_mid = (cap1 + cap2) / 2
                
                # Derivada: cambio de eficiencia / cambio de capital
                derivative = (eff2 - eff1) / (cap2 - cap1)
                
                # Utilidad marginal: cuánta eficiencia ganamos por unidad de capital
                # Multiplicamos por la eficiencia actual para dar preferencia a símbolos
                # que ya tienen alta eficiencia
                utility = derivative * ((eff1 + eff2) / 2)
                
                marginal_utility[symbol].append((cap_mid, utility))
        
        # Paso 4: Implementar algoritmo de asignación incremental
        remaining_capital = total_capital
        unallocated_symbols = set(valid_symbols)
        
        # Inicializar cada símbolo con capital mínimo
        for symbol in unallocated_symbols:
            allocation[symbol] = min_capital
            remaining_capital -= min_capital
        
        # Asignar el resto incrementalmente en función de la utilidad marginal
        allocation_steps = 20  # Número de pasos de asignación
        step_size = remaining_capital / allocation_steps
        
        for _ in range(allocation_steps):
            if not unallocated_symbols or remaining_capital < step_size:
                break
            
            # Evaluar utilidad marginal actual para cada símbolo
            current_utility: Dict[str, float] = {}
            for symbol in unallocated_symbols:
                current_cap = allocation[symbol]
                
                # Encontrar el punto más cercano en la curva de utilidad
                utility_curve = marginal_utility[symbol]
                closest = min(utility_curve, key=lambda x: abs(x[0] - current_cap))
                
                current_utility[symbol] = closest[1]
            
            # Normalizar utilidades para obtener pesos
            total_utility = sum(max(0, u) for u in current_utility.values())
            if total_utility <= 0:
                # Si todas las utilidades son negativas, distribuir equitativamente
                weights = {s: 1/len(unallocated_symbols) for s in unallocated_symbols}
            else:
                weights = {s: max(0, u)/total_utility if total_utility > 0 else 0 
                          for s, u in current_utility.items()}
            
            # Asignar capital según pesos
            for symbol in list(unallocated_symbols):
                to_add = step_size * weights.get(symbol, 0)
                allocation[symbol] += to_add
                remaining_capital -= to_add
                
                # Verificar si hemos alcanzado la saturación
                pred = self.predict_efficiency(symbol, allocation[symbol])
                if pred.efficiency < min_efficiency:
                    # Si la eficiencia cae por debajo del umbral, detener asignación
                    unallocated_symbols.remove(symbol)
        
        # Si queda capital sin asignar, distribuirlo entre símbolos no saturados
        if remaining_capital > 0 and unallocated_symbols:
            per_symbol = remaining_capital / len(unallocated_symbols)
            for symbol in unallocated_symbols:
                allocation[symbol] += per_symbol
        
        # Verificación final: asegurar que hemos asignado todo el capital
        allocated = sum(allocation.values())
        if abs(allocated - total_capital) > 0.01:
            # Ajustar proporcionalmente
            scale_factor = total_capital / allocated
            for symbol in allocation:
                allocation[symbol] *= scale_factor
        
        return allocation
    
    def get_saturation_points(self) -> Dict[str, float]:
        """
        Obtener puntos de saturación estimados para todos los símbolos.
        
        Returns:
            Diccionario con puntos de saturación por símbolo
        """
        result = {}
        for symbol, model in self.models.items():
            if model.saturation_point:
                result[symbol] = model.saturation_point
        return result
    
    def get_model_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Obtener información del modelo para un símbolo.
        
        Args:
            symbol: Símbolo del instrumento
            
        Returns:
            Diccionario con información del modelo o None si no existe
        """
        if symbol in self.models:
            return self.models[symbol].to_dict()
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del motor predictivo.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            **self.stats,
            "models_count": len(self.models),
            "trained_models": sum(1 for m in self.models.values() if m.is_trained),
            "cache_entries": sum(len(c) for c in self.prediction_cache.values()),
            "high_quality_models": sum(1 for m in self.models.values() if m.r_squared > 0.8)
        }