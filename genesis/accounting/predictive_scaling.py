"""
Motor predictivo de escalabilidad para modelar y optimizar eficiencia vs capital.

Este módulo implementa modelos matemáticos para predecir cómo la eficiencia
de un instrumento financiero cambia en función del nivel de capital, permitiendo
detectar puntos de saturación y optimizar asignaciones de capital.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime
from dataclasses import dataclass
import numpy as np
from scipy.optimize import curve_fit, minimize

# Configurar logging
logger = logging.getLogger("genesis.accounting.predictive_scaling")

@dataclass
class EfficiencyPrediction:
    """
    Representación de una predicción de eficiencia.
    
    Atributos:
        symbol: Símbolo del instrumento
        capital: Nivel de capital
        efficiency: Eficiencia predicha (0-1)
        confidence: Nivel de confianza en la predicción (0-1)
        timestamp: Momento de la predicción
    """
    symbol: str
    capital: float
    efficiency: float
    confidence: float = 0.0
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Inicializar valores por defecto."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para APIs."""
        return {
            "symbol": self.symbol,
            "capital": self.capital,
            "efficiency": self.efficiency,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }

class PredictiveModel:
    """
    Modelo predictivo para la relación entre capital y eficiencia.
    
    Esta clase base define la interfaz común para todos los modelos
    predictivos e implementa funcionalidad compartida.
    """
    
    def __init__(
        self, 
        symbol: str,
        model_type: str = "polynomial",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializar modelo predictivo.
        
        Args:
            symbol: Símbolo del instrumento
            model_type: Tipo de modelo ('linear', 'polynomial', 'exponential')
            config: Configuración adicional
        """
        self.symbol = symbol
        self.model_type = model_type
        self.config = config or {}
        
        # Datos históricos (x: capital, y: eficiencia)
        self.data_points: List[Tuple[float, float]] = []
        
        # Información del modelo
        self.is_trained = False
        self.parameters: Dict[str, Any] = {}
        self.r_squared = 0.0
        self.mean_error = 0.0
        self.max_error = 0.0
        self.samples_count = 0
        self.training_timestamp = None
        
        # Punto de saturación (nivel donde eficiencia comienza a caer significativamente)
        self.saturation_point: Optional[float] = None
        
        logger.debug(f"Modelo predictivo inicializado para {symbol}")
    
    def add_data_point(self, capital: float, efficiency: float) -> None:
        """
        Añadir punto de datos al histórico.
        
        Args:
            capital: Nivel de capital
            efficiency: Eficiencia observada (0-1)
        """
        # Validar datos
        if capital <= 0:
            logger.warning(f"Capital debe ser positivo: {capital}")
            return
            
        if efficiency < 0 or efficiency > 1:
            logger.warning(f"Eficiencia debe estar entre 0 y 1: {efficiency}")
            efficiency = max(0.0, min(1.0, efficiency))
        
        # Añadir punto
        self.data_points.append((capital, efficiency))
        
        # Marcar como no entrenado
        self.is_trained = False
        
        logger.debug(f"Añadido punto de datos: ({capital}, {efficiency}) para {self.symbol}")
    
    def train(self) -> bool:
        """
        Entrenar el modelo con los datos históricos.
        
        Este método debe ser implementado por cada subclase.
        
        Returns:
            True si el entrenamiento fue exitoso
        """
        pass
    
    def predict(self, capital: float) -> Tuple[float, float]:
        """
        Predecir eficiencia para un nivel de capital.
        
        Este método debe ser implementado por cada subclase.
        
        Args:
            capital: Nivel de capital
            
        Returns:
            Tupla (eficiencia, confianza)
        """
        pass
    
    def detect_saturation_point(self) -> Optional[float]:
        """
        Detectar punto de saturación donde eficiencia comienza a caer.
        
        Este método debe ser implementado por cada subclase.
        
        Returns:
            Punto de saturación o None si no se detecta
        """
        pass
    
    def calculate_metrics(self) -> None:
        """
        Calcular métricas de calidad del modelo.
        
        Métricas como R², error medio, error máximo, etc.
        """
        if not self.is_trained or len(self.data_points) < 2:
            return
            
        # Predecir valores para todos los puntos históricos
        actual_values = [y for _, y in self.data_points]
        capital_values = [x for x, _ in self.data_points]
        
        predicted_values = []
        for capital in capital_values:
            efficiency, _ = self.predict(capital)
            predicted_values.append(efficiency)
        
        # Calcular R² (coeficiente de determinación)
        mean_actual = sum(actual_values) / len(actual_values)
        
        ss_total = sum((y - mean_actual) ** 2 for y in actual_values)
        ss_residual = sum((actual - pred) ** 2 for actual, pred in zip(actual_values, predicted_values))
        
        if ss_total > 0:
            self.r_squared = 1 - (ss_residual / ss_total)
        else:
            self.r_squared = 0
        
        # Calcular error medio y máximo
        errors = [abs(actual - pred) for actual, pred in zip(actual_values, predicted_values)]
        self.mean_error = sum(errors) / len(errors) if errors else 0
        self.max_error = max(errors) if errors else 0
        
        # Actualizar valores
        self.samples_count = len(self.data_points)
        self.training_timestamp = datetime.now()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del modelo.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "symbol": self.symbol,
            "model_type": self.model_type,
            "is_trained": self.is_trained,
            "samples_count": self.samples_count,
            "r_squared": self.r_squared,
            "mean_error": self.mean_error,
            "max_error": self.max_error,
            "saturation_point": self.saturation_point,
            "training_timestamp": self.training_timestamp.isoformat() if self.training_timestamp else None,
            "parameters": self.parameters
        }

class LinearModel(PredictiveModel):
    """
    Modelo lineal para eficiencia vs capital.
    
    Fórmula: efficiency = a * capital + b
    """
    
    def __init__(self, symbol: str, config: Optional[Dict[str, Any]] = None):
        """
        Inicializar modelo lineal.
        
        Args:
            symbol: Símbolo del instrumento
            config: Configuración adicional
        """
        super().__init__(symbol, "linear", config)
        
        # Parámetros del modelo
        self.parameters = {
            "a": 0.0,  # Pendiente
            "b": 0.0   # Intercepto
        }
    
    def train(self) -> bool:
        """
        Entrenar modelo con regresión lineal.
        
        Returns:
            True si el entrenamiento fue exitoso
        """
        if len(self.data_points) < 2:
            logger.warning(f"Insuficientes puntos de datos para {self.symbol} (mínimo 2)")
            return False
        
        try:
            # Extraer datos
            x = np.array([point[0] for point in self.data_points])
            y = np.array([point[1] for point in self.data_points])
            
            # Ajustar modelo lineal (y = a*x + b)
            A = np.vstack([x, np.ones(len(x))]).T
            a, b = np.linalg.lstsq(A, y, rcond=None)[0]
            
            # Guardar parámetros
            self.parameters["a"] = float(a)
            self.parameters["b"] = float(b)
            
            # Marcar como entrenado
            self.is_trained = True
            
            # Calcular métricas
            self.calculate_metrics()
            
            # Detectar punto de saturación
            self.saturation_point = self.detect_saturation_point()
            
            return True
            
        except Exception as e:
            logger.error(f"Error entrenando modelo lineal para {self.symbol}: {str(e)}")
            return False
    
    def predict(self, capital: float) -> Tuple[float, float]:
        """
        Predecir eficiencia para un nivel de capital.
        
        Args:
            capital: Nivel de capital
            
        Returns:
            Tupla (eficiencia, confianza)
        """
        if not self.is_trained:
            logger.warning(f"Modelo no entrenado para {self.symbol}")
            return (0.5, 0.0)
        
        # Extraer parámetros
        a = self.parameters["a"]
        b = self.parameters["b"]
        
        # Calcular predicción
        efficiency = a * capital + b
        
        # Limitar a rango [0, 1]
        efficiency = max(0.0, min(1.0, efficiency))
        
        # Calcular confianza
        confidence = self._calculate_confidence(capital)
        
        return (efficiency, confidence)
    
    def _calculate_confidence(self, capital: float) -> float:
        """
        Calcular confianza en la predicción.
        
        La confianza disminuye:
        - Si el capital está fuera del rango de datos históricos
        - Si hay pocos datos
        - Si el modelo tiene bajo R²
        
        Args:
            capital: Nivel de capital
            
        Returns:
            Confianza (0-1)
        """
        if len(self.data_points) < 2:
            return 0.0
        
        # Factores de confianza
        r_squared_factor = self.r_squared
        
        # Factor de rango
        capital_values = [x for x, _ in self.data_points]
        min_capital = min(capital_values)
        max_capital = max(capital_values)
        
        # Si está dentro del rango, confianza por rango es alta
        if min_capital <= capital <= max_capital:
            range_factor = 1.0
        else:
            # Si está fuera del rango, la confianza disminuye con la distancia
            if capital < min_capital:
                range_factor = max(0.0, 1.0 - (min_capital - capital) / min_capital)
            else:  # capital > max_capital
                range_factor = max(0.0, 1.0 - (capital - max_capital) / max_capital)
            
            # Limitar a un mínimo de 0.2
            range_factor = max(0.2, range_factor)
        
        # Factor de muestras
        data_factor = min(1.0, len(self.data_points) / 10)  # Máximo con 10+ muestras
        
        # Combinar factores
        confidence = 0.4 * r_squared_factor + 0.4 * range_factor + 0.2 * data_factor
        
        return confidence
    
    def detect_saturation_point(self) -> Optional[float]:
        """
        Detectar punto de saturación donde eficiencia comienza a caer.
        
        Para un modelo lineal con pendiente negativa, el punto de
        saturación es donde la eficiencia cae por debajo de cierto umbral.
        
        Returns:
            Punto de saturación o None si no se detecta
        """
        if not self.is_trained:
            return None
        
        # Si la pendiente es positiva o cero, no hay saturación
        a = self.parameters["a"]
        if a >= 0:
            return None
        
        # Si la pendiente es negativa, calcular dónde la eficiencia
        # cae por debajo del umbral (p.ej., 0.7)
        b = self.parameters["b"]
        efficiency_threshold = self.config.get("efficiency_threshold", 0.7)
        
        # efficiency = a * capital + b
        # capital = (efficiency - b) / a
        if b <= efficiency_threshold:
            return 0.0  # Ya está por debajo del umbral con capital = 0
        
        saturation_point = (efficiency_threshold - b) / a
        
        return max(0.0, saturation_point)

class PolynomialModel(PredictiveModel):
    """
    Modelo polinomial para eficiencia vs capital.
    
    Fórmula: efficiency = a * capital² + b * capital + c
    """
    
    def __init__(self, symbol: str, config: Optional[Dict[str, Any]] = None):
        """
        Inicializar modelo polinomial.
        
        Args:
            symbol: Símbolo del instrumento
            config: Configuración adicional
        """
        super().__init__(symbol, "polynomial", config)
        
        # Parámetros del modelo
        self.parameters = {
            "a": 0.0,  # Coeficiente cuadrático
            "b": 0.0,  # Coeficiente lineal
            "c": 0.0   # Término constante
        }
        
        # Orden del polinomio
        self.degree = self.config.get("polynomial_degree", 2)
    
    def train(self) -> bool:
        """
        Entrenar modelo con regresión polinomial.
        
        Returns:
            True si el entrenamiento fue exitoso
        """
        if len(self.data_points) < 3:
            logger.warning(f"Insuficientes puntos de datos para {self.symbol} (mínimo 3)")
            return False
        
        try:
            # Extraer datos
            x = np.array([point[0] for point in self.data_points])
            y = np.array([point[1] for point in self.data_points])
            
            # Ajustar modelo polinomial de orden 2
            coeffs = np.polyfit(x, y, self.degree)
            
            # Guardar parámetros
            if self.degree == 2:
                self.parameters["a"] = float(coeffs[0])
                self.parameters["b"] = float(coeffs[1])
                self.parameters["c"] = float(coeffs[2])
            else:
                self.parameters["coeffs"] = [float(c) for c in coeffs]
            
            # Marcar como entrenado
            self.is_trained = True
            
            # Calcular métricas
            self.calculate_metrics()
            
            # Detectar punto de saturación
            self.saturation_point = self.detect_saturation_point()
            
            return True
            
        except Exception as e:
            logger.error(f"Error entrenando modelo polinomial para {self.symbol}: {str(e)}")
            return False
    
    def predict(self, capital: float) -> Tuple[float, float]:
        """
        Predecir eficiencia para un nivel de capital.
        
        Args:
            capital: Nivel de capital
            
        Returns:
            Tupla (eficiencia, confianza)
        """
        if not self.is_trained:
            logger.warning(f"Modelo no entrenado para {self.symbol}")
            return (0.5, 0.0)
        
        try:
            # Calcular predicción
            if self.degree == 2:
                # Modo optimizado para polinomio de orden 2
                a = self.parameters["a"]
                b = self.parameters["b"]
                c = self.parameters["c"]
                
                efficiency = a * capital * capital + b * capital + c
            else:
                # Modo general para cualquier orden
                coeffs = self.parameters["coeffs"]
                efficiency = np.polyval(coeffs, capital)
            
            # Limitar a rango [0, 1]
            efficiency = max(0.0, min(1.0, efficiency))
            
            # Calcular confianza
            confidence = self._calculate_confidence(capital)
            
            return (float(efficiency), confidence)
            
        except Exception as e:
            logger.error(f"Error en predicción polinomial para {self.symbol}: {str(e)}")
            return (0.5, 0.0)
    
    def _calculate_confidence(self, capital: float) -> float:
        """
        Calcular confianza en la predicción.
        
        Args:
            capital: Nivel de capital
            
        Returns:
            Confianza (0-1)
        """
        if len(self.data_points) < 3:
            return 0.0
        
        # Factores de confianza
        r_squared_factor = self.r_squared
        
        # Factor de rango
        capital_values = [x for x, _ in self.data_points]
        min_capital = min(capital_values)
        max_capital = max(capital_values)
        
        # Si está dentro del rango, confianza por rango es alta
        if min_capital <= capital <= max_capital:
            range_factor = 1.0
        else:
            # Si está fuera del rango, la confianza disminuye con la distancia
            if capital < min_capital:
                range_factor = max(0.0, 1.0 - (min_capital - capital) / min_capital)
            else:  # capital > max_capital
                range_factor = max(0.0, 1.0 - (capital - max_capital) / max_capital)
            
            # Limitar a un mínimo de 0.1
            range_factor = max(0.1, range_factor)
        
        # Factor de muestras
        data_factor = min(1.0, len(self.data_points) / 12)  # Máximo con 12+ muestras
        
        # Combinar factores (polinomial necesita más datos para ser confiable)
        confidence = 0.3 * r_squared_factor + 0.5 * range_factor + 0.2 * data_factor
        
        return confidence
    
    def detect_saturation_point(self) -> Optional[float]:
        """
        Detectar punto de saturación donde eficiencia comienza a caer.
        
        Para polinomio orden 2 (parábola), el punto máximo es el punto de saturación.
        
        Returns:
            Punto de saturación o None si no se detecta
        """
        if not self.is_trained:
            return None
        
        try:
            if self.degree == 2:
                # Para una parábola (y = ax² + bx + c), el punto máximo está en x = -b/(2a)
                a = self.parameters["a"]
                b = self.parameters["b"]
                
                # Si a >= 0, no tenemos parábola invertida, no hay saturación
                if a >= 0:
                    return None
                
                # Calcular punto máximo
                saturation_point = -b / (2 * a)
                
                # Verificar que esté en un rango razonable
                if saturation_point <= 0:
                    return None
                
                # Verificar que la eficiencia en ese punto sea suficientemente alta
                efficiency, _ = self.predict(saturation_point)
                if efficiency < 0.5:  # Umbral arbitrario
                    return None
                
                return saturation_point
            else:
                # Para órdenes superiores, habría que buscar la primera derivada = 0
                # Simplificación: buscar el punto más alto dentro del rango observado
                capital_values = [x for x, _ in self.data_points]
                min_capital = min(capital_values)
                max_capital = max(capital_values) * 2  # Extender rango
                
                test_points = np.linspace(min_capital, max_capital, 100)
                efficiencies = [self.predict(x)[0] for x in test_points]
                
                max_index = np.argmax(efficiencies)
                max_efficiency = efficiencies[max_index]
                capital_at_max = test_points[max_index]
                
                # Si el máximo está en el último punto, no considerarlo saturación
                if max_index == len(test_points) - 1:
                    return None
                
                # Si la eficiencia no baja significativamente después del máximo, no hay saturación clara
                threshold = self.config.get("saturation_threshold", 0.05)
                if max_index < len(efficiencies) - 1:
                    later_efficiencies = efficiencies[max_index + 1:]
                    if min(later_efficiencies) > max_efficiency - threshold:
                        return None
                
                return capital_at_max
                
        except Exception as e:
            logger.error(f"Error detectando saturación para {self.symbol}: {str(e)}")
            return None

class ExponentialModel(PredictiveModel):
    """
    Modelo exponencial para eficiencia vs capital.
    
    Fórmula: efficiency = a * exp(-b * capital) + c
    """
    
    def __init__(self, symbol: str, config: Optional[Dict[str, Any]] = None):
        """
        Inicializar modelo exponencial.
        
        Args:
            symbol: Símbolo del instrumento
            config: Configuración adicional
        """
        super().__init__(symbol, "exponential", config)
        
        # Parámetros del modelo
        self.parameters = {
            "a": 0.0,  # Amplitud
            "b": 0.0,  # Tasa de decaimiento
            "c": 0.0   # Asíntota
        }
    
    def _exp_decay_func(self, x, a, b, c):
        """Función de decaimiento exponencial."""
        return a * np.exp(-b * x) + c
    
    def train(self) -> bool:
        """
        Entrenar modelo con ajuste de curva exponencial.
        
        Returns:
            True si el entrenamiento fue exitoso
        """
        if len(self.data_points) < 3:
            logger.warning(f"Insuficientes puntos de datos para {self.symbol} (mínimo 3)")
            return False
        
        try:
            # Extraer datos
            x = np.array([point[0] for point in self.data_points])
            y = np.array([point[1] for point in self.data_points])
            
            # Parámetros iniciales
            p0 = [1.0, 0.001, 0.0]
            
            # Límites para los parámetros
            bounds = ([0, 0, 0], [1, 1, 1])
            
            # Ajustar modelo exponencial (y = a * exp(-b * x) + c)
            params, _ = curve_fit(
                self._exp_decay_func, 
                x, y, 
                p0=p0, 
                bounds=bounds,
                maxfev=10000
            )
            
            # Guardar parámetros
            self.parameters["a"] = float(params[0])
            self.parameters["b"] = float(params[1])
            self.parameters["c"] = float(params[2])
            
            # Marcar como entrenado
            self.is_trained = True
            
            # Calcular métricas
            self.calculate_metrics()
            
            # Detectar punto de saturación
            self.saturation_point = self.detect_saturation_point()
            
            return True
            
        except Exception as e:
            logger.error(f"Error entrenando modelo exponencial para {self.symbol}: {str(e)}")
            return False
    
    def predict(self, capital: float) -> Tuple[float, float]:
        """
        Predecir eficiencia para un nivel de capital.
        
        Args:
            capital: Nivel de capital
            
        Returns:
            Tupla (eficiencia, confianza)
        """
        if not self.is_trained:
            logger.warning(f"Modelo no entrenado para {self.symbol}")
            return (0.5, 0.0)
        
        try:
            # Extraer parámetros
            a = self.parameters["a"]
            b = self.parameters["b"]
            c = self.parameters["c"]
            
            # Calcular predicción
            efficiency = a * np.exp(-b * capital) + c
            
            # Limitar a rango [0, 1]
            efficiency = max(0.0, min(1.0, efficiency))
            
            # Calcular confianza
            confidence = self._calculate_confidence(capital)
            
            return (float(efficiency), confidence)
            
        except Exception as e:
            logger.error(f"Error en predicción exponencial para {self.symbol}: {str(e)}")
            return (0.5, 0.0)
    
    def _calculate_confidence(self, capital: float) -> float:
        """
        Calcular confianza en la predicción.
        
        Args:
            capital: Nivel de capital
            
        Returns:
            Confianza (0-1)
        """
        if len(self.data_points) < 3:
            return 0.0
        
        # Factores de confianza
        r_squared_factor = self.r_squared
        
        # Factor de rango
        capital_values = [x for x, _ in self.data_points]
        min_capital = min(capital_values)
        max_capital = max(capital_values)
        
        # Si está dentro del rango, confianza por rango es alta
        if min_capital <= capital <= max_capital:
            range_factor = 1.0
        else:
            # Si está fuera del rango, la confianza disminuye con la distancia
            if capital < min_capital:
                range_factor = max(0.0, 1.0 - (min_capital - capital) / min_capital)
            else:  # capital > max_capital
                # Exponencial puede extrapolar mejor a valores altos
                range_factor = max(0.3, 1.0 - 0.7 * (capital - max_capital) / max_capital)
            
            # Limitar a un mínimo de 0.2
            range_factor = max(0.2, range_factor)
        
        # Factor de muestras
        data_factor = min(1.0, len(self.data_points) / 15)  # Máximo con 15+ muestras
        
        # Combinar factores (exponencial necesita más datos para ser confiable)
        confidence = 0.3 * r_squared_factor + 0.5 * range_factor + 0.2 * data_factor
        
        return confidence
    
    def detect_saturation_point(self) -> Optional[float]:
        """
        Detectar punto de saturación para un modelo exponencial.
        
        Returns:
            Punto de saturación o None si no se detecta
        """
        if not self.is_trained:
            return None
        
        try:
            # Extraer parámetros
            a = self.parameters["a"]
            b = self.parameters["b"]
            c = self.parameters["c"]
            
            # En un modelo exponencial decreciente, no hay un máximo claro.
            # Usamos un enfoque basado en la tasa de cambio:
            # - Encontrar el punto donde la tasa de cambio cae por debajo del umbral.
            
            # Solo aplica si b > 0 (decaimiento)
            if b <= 0:
                return None
            
            # La derivada es -a*b*exp(-b*x)
            # Buscamos donde la magnitud de la derivada cae por debajo de un umbral
            threshold = self.config.get("derivative_threshold", 0.001)
            
            # Resolver: abs(a*b*exp(-b*x)) < threshold
            # exp(-b*x) < threshold/(a*b)
            # -b*x < ln(threshold/(a*b))
            # x > -ln(threshold/(a*b))/b
            
            if a * b == 0:
                return None
                
            saturation_point = -np.log(threshold / (a * b)) / b
            
            # Verificar que el punto sea positivo y razonable
            if saturation_point <= 0:
                return None
                
            # Verificar que la eficiencia en ese punto sea razonable
            efficiency, _ = self.predict(saturation_point)
            if efficiency < 0.3:  # Umbral arbitrario
                return None
                
            return saturation_point
                
        except Exception as e:
            logger.error(f"Error detectando saturación para {self.symbol}: {str(e)}")
            return None

class ModelFactory:
    """Fábrica para crear modelos predictivos según el tipo."""
    
    @staticmethod
    def create_model(
        symbol: str,
        model_type: str = "polynomial",
        config: Optional[Dict[str, Any]] = None
    ) -> PredictiveModel:
        """
        Crear modelo predictivo del tipo especificado.
        
        Args:
            symbol: Símbolo del instrumento
            model_type: Tipo de modelo ('linear', 'polynomial', 'exponential')
            config: Configuración adicional
            
        Returns:
            Modelo predictivo
        """
        model_type = model_type.lower()
        
        if model_type == "linear":
            return LinearModel(symbol, config)
        elif model_type == "polynomial":
            return PolynomialModel(symbol, config)
        elif model_type == "exponential":
            return ExponentialModel(symbol, config)
        else:
            logger.warning(f"Tipo de modelo desconocido: {model_type}, usando polynomial")
            return PolynomialModel(symbol, config)

class PredictiveScalingEngine:
    """
    Motor principal para gestionar modelos predictivos de escalabilidad.
    
    Esta clase:
    1. Mantiene modelos predictivos para diferentes instrumentos
    2. Entrena modelos con datos históricos
    3. Predice eficiencia a diferentes niveles de capital
    4. Optimiza la asignación de capital entre instrumentos
    5. Detecta puntos de saturación y límites de escalabilidad
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializar motor predictivo.
        
        Args:
            config: Configuración general
        """
        self.config = config or {}
        
        # Modelos por símbolo
        self.models: Dict[str, PredictiveModel] = {}
        
        # Cache de predicciones recientes
        self.prediction_cache: Dict[str, Dict[float, Tuple[float, float, float]]] = {}
        
        # Timestamp de última limpieza de caché
        self.last_cache_clean = time.time()
        
        # Configuraciones
        self.default_model_type = self.config.get("default_model_type", "polynomial")
        self.cache_ttl = self.config.get("cache_ttl", 300)  # 5 minutos
        self.auto_train = self.config.get("auto_train", True)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.6)
        
        logger.info(f"PredictiveScalingEngine inicializado con modelo por defecto: {self.default_model_type}")
    
    def clean_old_cache(self) -> None:
        """Limpiar entradas antiguas del caché."""
        now = time.time()
        
        # Limpiar caché solo si ha pasado suficiente tiempo
        if now - self.last_cache_clean < 60:  # 1 minuto
            return
            
        self.last_cache_clean = now
        
        for symbol in list(self.prediction_cache.keys()):
            capital_dict = self.prediction_cache[symbol]
            
            for capital in list(capital_dict.keys()):
                _, _, timestamp = capital_dict[capital]
                
                if now - timestamp > self.cache_ttl:
                    del capital_dict[capital]
            
            # Si no quedan entradas para este símbolo, eliminar la entrada
            if not capital_dict:
                del self.prediction_cache[symbol]
    
    async def add_efficiency_record(
        self, 
        symbol: str, 
        capital: float, 
        efficiency: float,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Añadir registro de eficiencia observada.
        
        Args:
            symbol: Símbolo del instrumento
            capital: Nivel de capital
            efficiency: Eficiencia observada (0-1)
            metrics: Métricas adicionales (ROI, Sharpe, etc.)
        """
        # Crear modelo si no existe
        if symbol not in self.models:
            self.models[symbol] = ModelFactory.create_model(
                symbol, 
                self.default_model_type,
                self.config
            )
        
        # Añadir punto de datos
        self.models[symbol].add_data_point(capital, efficiency)
        
        # Entrenar modelo automáticamente si está configurado
        if self.auto_train:
            self.models[symbol].train()
    
    async def predict_efficiency(self, symbol: str, capital: float) -> EfficiencyPrediction:
        """
        Predecir eficiencia para un instrumento a un nivel de capital.
        
        Args:
            symbol: Símbolo del instrumento
            capital: Nivel de capital
            
        Returns:
            Predicción de eficiencia
        """
        # Limpiar caché antigua
        self.clean_old_cache()
        
        # Verificar si hay caché reciente
        if (symbol in self.prediction_cache and 
            capital in self.prediction_cache[symbol]):
            
            efficiency, confidence, _ = self.prediction_cache[symbol][capital]
            
            return EfficiencyPrediction(
                symbol=symbol,
                capital=capital,
                efficiency=efficiency,
                confidence=confidence,
                timestamp=datetime.now()
            )
        
        # Si no hay modelo, crear uno con valor por defecto
        if symbol not in self.models:
            logger.warning(f"No hay modelo para {symbol}, usando valor por defecto")
            
            return EfficiencyPrediction(
                symbol=symbol,
                capital=capital,
                efficiency=0.5,
                confidence=0.1,
                timestamp=datetime.now()
            )
        
        # Entrenar modelo si no está entrenado
        if not self.models[symbol].is_trained:
            self.models[symbol].train()
        
        # Obtener predicción
        efficiency, confidence = self.models[symbol].predict(capital)
        
        # Guardar en caché
        if symbol not in self.prediction_cache:
            self.prediction_cache[symbol] = {}
            
        self.prediction_cache[symbol][capital] = (efficiency, confidence, time.time())
        
        # Construir resultado
        return EfficiencyPrediction(
            symbol=symbol,
            capital=capital,
            efficiency=efficiency,
            confidence=confidence,
            timestamp=datetime.now()
        )
    
    async def optimize_allocation(
        self, 
        symbols: List[str], 
        total_capital: float,
        min_efficiency: float = 0.5,
        min_position_size: Optional[float] = None,
        max_position_percentage: Optional[float] = None,
        position_limits: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Optimizar asignación de capital entre instrumentos.
        
        Este método utiliza un algoritmo de utilidad marginal para
        distribuir capital eficientemente.
        
        Args:
            symbols: Lista de símbolos a considerar
            total_capital: Capital total disponible
            min_efficiency: Eficiencia mínima aceptable
            min_position_size: Tamaño mínimo de posición (o None para no establecer)
            max_position_percentage: Porcentaje máximo por posición (0-1)
            position_limits: Límites máximos por símbolo {'symbol': max_amount}
            
        Returns:
            Asignaciones por símbolo {'symbol': amount}
        """
        if not symbols or total_capital <= 0:
            return {}
        
        # Configuraciones adicionales
        min_position_size = min_position_size or self.config.get("min_position_size", 0.0)
        max_position_percentage = max_position_percentage or self.config.get("max_position_percentage", 0.5)
        position_limits = position_limits or {}
        
        # Paso 1: Calcular las asignaciones iniciales por utilidad marginal
        allocations = await self._allocate_by_marginal_utility(
            symbols=symbols,
            total_capital=total_capital,
            min_efficiency=min_efficiency,
            min_position_size=min_position_size
        )
        
        # Paso 2: Aplicar restricciones de posición máxima
        for symbol, amount in list(allocations.items()):
            # Límite por porcentaje del capital total
            percentage_limit = total_capital * max_position_percentage
            
            # Límite específico del símbolo (si existe)
            symbol_limit = position_limits.get(symbol, float('inf'))
            
            # Aplicar el límite más restrictivo
            max_allowed = min(percentage_limit, symbol_limit)
            
            if amount > max_allowed:
                allocations[symbol] = max_allowed
        
        # Paso 3: Redondear asignaciones y ajustar para sumar exactamente el capital total
        allocations = self._round_allocations(allocations, total_capital)
        
        return allocations
    
    async def _allocate_by_marginal_utility(
        self,
        symbols: List[str],
        total_capital: float,
        min_efficiency: float,
        min_position_size: float
    ) -> Dict[str, float]:
        """
        Asignar capital usando el principio de utilidad marginal.
        
        Args:
            symbols: Lista de símbolos
            total_capital: Capital total
            min_efficiency: Eficiencia mínima
            min_position_size: Tamaño mínimo de posición
            
        Returns:
            Asignaciones por símbolo
        """
        # Inicializar asignaciones
        allocations = {symbol: 0.0 for symbol in symbols}
        remaining_capital = total_capital
        
        # Quitar símbolos sin modelo o no entrenados
        active_symbols = [s for s in symbols if s in self.models and self.models[s].is_trained]
        
        # Si no hay símbolos activos, devolver asignación vacía
        if not active_symbols:
            return {}
        
        # Incrementos para asignar capital
        capital_step = min(1000.0, total_capital * 0.01)  # 1% del capital o 1000, lo que sea menor
        
        # Asignar capital incrementalmente
        while remaining_capital >= capital_step and active_symbols:
            best_symbol = None
            best_marginal_utility = -float('inf')
            
            # Encontrar el símbolo con mayor utilidad marginal
            for symbol in active_symbols:
                current_allocation = allocations[symbol]
                new_allocation = current_allocation + capital_step
                
                # Calcular utilidad marginal (eficiencia adicional por capital adicional)
                current_prediction = await self.predict_efficiency(symbol, current_allocation)
                new_prediction = await self.predict_efficiency(symbol, new_allocation)
                
                # Eficiencia diferencial
                efficiency_gain = new_prediction.efficiency - current_prediction.efficiency
                marginal_utility = efficiency_gain / capital_step
                
                # Considerar confianza en la predicción
                adjusted_utility = marginal_utility * new_prediction.confidence
                
                # Si la nueva eficiencia está por debajo del mínimo, descartarlo
                if new_prediction.efficiency < min_efficiency:
                    continue
                
                # Actualizar mejor símbolo
                if adjusted_utility > best_marginal_utility:
                    best_marginal_utility = adjusted_utility
                    best_symbol = symbol
            
            # Si no hay símbolos con utilidad marginal positiva, terminar
            if best_symbol is None or best_marginal_utility <= 0:
                break
            
            # Asignar incremento al mejor símbolo
            allocations[best_symbol] += capital_step
            remaining_capital -= capital_step
        
        # Paso final: eliminar posiciones por debajo del mínimo
        for symbol in list(allocations.keys()):
            if allocations[symbol] < min_position_size:
                del allocations[symbol]
        
        return allocations
    
    def _round_allocations(self, allocations: Dict[str, float], total_capital: float) -> Dict[str, float]:
        """
        Redondear asignaciones y ajustar para sumar exactamente el capital total.
        
        Args:
            allocations: Asignaciones originales
            total_capital: Capital total
            
        Returns:
            Asignaciones redondeadas y ajustadas
        """
        # Si las asignaciones están vacías, devolver
        if not allocations:
            return {}
        
        # Calcular suma actual
        current_sum = sum(allocations.values())
        
        # Si la suma es cero, devolver
        if current_sum == 0:
            return allocations
        
        # Ajustar proporcionalmente para sumar el capital total
        scaling_factor = total_capital / current_sum
        
        # Aplicar factor de escala
        return {symbol: amount * scaling_factor for symbol, amount in allocations.items()}
    
    def get_saturation_points(self) -> Dict[str, Optional[float]]:
        """
        Obtener puntos de saturación para todos los símbolos.
        
        Returns:
            Diccionario con puntos de saturación por símbolo
        """
        result = {}
        
        for symbol, model in self.models.items():
            if model.is_trained:
                result[symbol] = model.saturation_point
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del motor predictivo.
        
        Returns:
            Diccionario con estadísticas
        """
        # Estadísticas generales
        stats = {
            "model_count": len(self.models),
            "cache_entries": sum(len(symbol_cache) for symbol_cache in self.prediction_cache.values()),
            "default_model_type": self.default_model_type,
            "auto_train": self.auto_train,
            "confidence_threshold": self.confidence_threshold,
            "saturation_points": self.get_saturation_points()
        }
        
        # Estadísticas por modelo
        model_stats = {}
        for symbol, model in self.models.items():
            model_stats[symbol] = model.get_stats()
        
        stats["models"] = model_stats
        
        return stats