"""
Módulo de Machine Learning para el Sistema Divino de Base de Datos.

Este módulo implementa modelos predictivos y adaptativos para optimizar
automáticamente el rendimiento del sistema divino, incluyendo:

- Predicción de carga para anticipar picos de operaciones
- Clasificación inteligente de prioridades según características de la tarea
- Ajuste dinámico de recursos basado en predicciones

Estos modelos aprenden continuamente en tiempo real, mejorando progresivamente
la precisión de sus predicciones y optimizaciones.
"""

import time
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

# Importaciones condicionales para permitir ejecución sin dependencias instaladas
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Configuración de logging
logger = logging.getLogger("genesis.db.divine_ml")

class DivineLoadPredictor:
    """
    Predictor de carga para el sistema divino.
    
    Este modelo predice la carga futura del sistema (operaciones por segundo)
    basándose en datos históricos y condiciones actuales, permitiendo
    anticipar picos de actividad y ajustar recursos preventivamente.
    """
    
    def __init__(self):
        """Inicializar predictor de carga."""
        self.initialized = False
        self.data: List[List[float]] = []  # [timestamp, ops, latency]
        self.min_data_points = 10
        self.max_data_points = 1000
        self.last_training = 0
        self.training_interval = 5  # segundos
        
        # Inicializar modelos ML si están disponibles
        if ML_AVAILABLE:
            self.scaler = StandardScaler()
            self.model = LinearRegression()
            self.initialized = True
            logger.info("Predictor de carga inicializado con scikit-learn")
        else:
            logger.warning("scikit-learn no disponible, predictor funcionará en modo básico")
    
    def add_data_point(self, timestamp: float, ops: float, latency: float) -> None:
        """
        Añadir un punto de datos para entrenamiento.
        
        Args:
            timestamp: Marca de tiempo en segundos desde la época
            ops: Operaciones por segundo
            latency: Latencia en segundos
        """
        self.data.append([timestamp, ops, latency])
        
        # Limitar tamaño del conjunto de datos
        if len(self.data) > self.max_data_points:
            self.data = self.data[-self.max_data_points:]
    
    def predict(self, current_ops: float, current_latency: float) -> float:
        """
        Predecir operaciones por segundo futuras.
        
        Args:
            current_ops: Operaciones por segundo actuales
            current_latency: Latencia actual en segundos
            
        Returns:
            Predicción de operaciones por segundo
        """
        # Si ML no está disponible, usar estimación simple
        if not ML_AVAILABLE or len(self.data) < self.min_data_points:
            # Predicción heurística simple: tendencia reciente + latencia
            if len(self.data) > 1:
                # Calcular tasa de cambio
                recent_data = self.data[-5:] if len(self.data) >= 5 else self.data
                changes = [d[1] for d in recent_data]
                avg_change = sum(changes) / len(changes)
                # Ajustar por latencia (más latencia sugiere más carga)
                latency_factor = 1 + (current_latency * 10)
                return current_ops * latency_factor + avg_change
            else:
                return current_ops * 1.1  # Predicción conservadora: 10% más
        
        # Entrenar modelo si es necesario
        current_time = time.time()
        if current_time - self.last_training > self.training_interval:
            self._train_model()
            self.last_training = current_time
        
        # Hacer predicción con el modelo
        features = np.array([[current_time, current_ops, current_latency]])
        scaled_features = self.scaler.transform(features)
        prediction = self.model.predict(scaled_features)[0]
        
        # Garantizar que la predicción sea positiva
        return max(0.1, prediction)
    
    def _train_model(self) -> None:
        """Entrenar modelo con datos actuales."""
        if not ML_AVAILABLE or len(self.data) < self.min_data_points:
            return
        
        try:
            # Preparar datos
            X = np.array(self.data)
            y = np.array([d[1] for d in self.data])  # OPS como target
            
            # Normalizar y entrenar
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            
            logger.debug(f"Modelo de predicción entrenado con {len(self.data)} puntos de datos")
        except Exception as e:
            logger.error(f"Error al entrenar modelo de predicción: {e}")

class DivinePriorityClassifier:
    """
    Clasificador de prioridades para tareas.
    
    Este modelo asigna automáticamente prioridades (1-10) a las tareas
    basándose en sus características, como tipo de operación, volumen,
    criticidad, etc.
    """
    
    def __init__(self):
        """Inicializar clasificador de prioridades."""
        self.initialized = False
        self.data: List[Dict[str, Any]] = []
        self.min_data_points = 10
        self.max_data_points = 500
        self.last_training = 0
        self.training_interval = 10  # segundos
        self.default_priority = 5
        
        # Inicializar modelos ML si están disponibles
        if ML_AVAILABLE:
            self.model = RandomForestClassifier(n_estimators=20, random_state=42)
            
            # Entrenar con datos sintéticos iniciales para tener un modelo base
            # Los datos reales lo reemplazarán rápidamente
            X = np.array([
                [1, 1000, 1],  # Alta prioridad: crítico, alto volumen, transaccional
                [1, 100, 1],   # Prioridad media-alta
                [0, 1000, 0],  # Prioridad media-alta
                [0, 100, 0],   # Prioridad media
                [0, 10, 0],    # Prioridad baja
            ])
            y = np.array([10, 8, 7, 5, 2])  # Prioridades correspondientes
            self.model.fit(X, y)
            
            self.initialized = True
            logger.info("Clasificador de prioridades inicializado con scikit-learn")
        else:
            logger.warning("scikit-learn no disponible, clasificador funcionará en modo básico")
    
    def add_training_data(self, critical: bool, volume: float, transactional: bool, priority: int) -> None:
        """
        Añadir datos de entrenamiento.
        
        Args:
            critical: Si la operación es crítica
            volume: Volumen o importancia de la operación (0-1000)
            transactional: Si la operación es transaccional
            priority: Prioridad asignada (1-10)
        """
        self.data.append({
            "critical": 1 if critical else 0,
            "volume": float(volume),
            "transactional": 1 if transactional else 0,
            "priority": int(priority)
        })
        
        # Limitar tamaño del conjunto de datos
        if len(self.data) > self.max_data_points:
            self.data = self.data[-self.max_data_points:]
    
    def predict_priority(self, critical: bool, volume: float, transactional: bool) -> int:
        """
        Predecir prioridad para una operación.
        
        Args:
            critical: Si la operación es crítica
            volume: Volumen o importancia de la operación (0-1000)
            transactional: Si la operación es transaccional
            
        Returns:
            Prioridad recomendada (1-10)
        """
        # Si ML no está disponible, usar reglas simples
        if not ML_AVAILABLE:
            if critical:
                return 10 if transactional else 9
            elif volume > 500:
                return 8 if transactional else 7
            elif volume > 100:
                return 6 if transactional else 5
            else:
                return 4 if transactional else 3
        
        # Entrenar modelo si es necesario
        current_time = time.time()
        if current_time - self.last_training > self.training_interval and len(self.data) >= self.min_data_points:
            self._train_model()
            self.last_training = current_time
        
        # Hacer predicción con el modelo
        features = np.array([[1 if critical else 0, float(volume), 1 if transactional else 0]])
        try:
            prediction = int(self.model.predict(features)[0])
            # Asegurar rango válido
            return max(1, min(10, prediction))
        except Exception as e:
            logger.error(f"Error al predecir prioridad: {e}")
            # Valor predeterminado seguro
            return self.default_priority
    
    def _train_model(self) -> None:
        """Entrenar modelo con datos actuales."""
        if not ML_AVAILABLE or len(self.data) < self.min_data_points:
            return
        
        try:
            # Preparar datos
            X = np.array([[d["critical"], d["volume"], d["transactional"]] for d in self.data])
            y = np.array([d["priority"] for d in self.data])
            
            # Entrenar
            self.model.fit(X, y)
            
            logger.debug(f"Modelo de prioridad entrenado con {len(self.data)} puntos de datos")
        except Exception as e:
            logger.error(f"Error al entrenar modelo de prioridad: {e}")

class DivineResourceOptimizer:
    """
    Optimizador de recursos para el sistema divino.
    
    Este optimizador determina automáticamente la cantidad óptima de recursos
    (workers, conexiones, etc.) según las predicciones de carga y métricas
    de rendimiento actuales.
    """
    
    def __init__(self, load_predictor: DivineLoadPredictor):
        """
        Inicializar optimizador.
        
        Args:
            load_predictor: Predictor de carga
        """
        self.load_predictor = load_predictor
        self.last_optimization = 0
        self.optimization_interval = 3  # segundos
        
        # Configuración de recursos
        self.min_redis_workers = 2
        self.max_redis_workers = 16
        self.min_rabbitmq_workers = 1
        self.max_rabbitmq_workers = 8
        
        # Eficiencia estimada por worker
        self.ops_per_redis_worker = 1000  # operaciones/segundo
        self.ops_per_rabbitmq_worker = 500  # operaciones/segundo
        
        # Historia de recomendaciones para estabilidad
        self.recent_recommendations: List[Dict[str, int]] = []
        self.max_recommendations = 5
    
    def optimize(self, current_ops: float, current_latency: float, 
                current_redis_workers: int, current_rabbitmq_workers: int) -> Dict[str, int]:
        """
        Recomendar recursos óptimos según carga predicha.
        
        Args:
            current_ops: Operaciones por segundo actuales
            current_latency: Latencia actual en segundos
            current_redis_workers: Número actual de workers Redis
            current_rabbitmq_workers: Número actual de workers RabbitMQ
            
        Returns:
            Recomendación de recursos (workers)
        """
        current_time = time.time()
        
        # Limitar frecuencia de optimización
        if current_time - self.last_optimization < self.optimization_interval:
            # Retornar última recomendación si existe
            if self.recent_recommendations:
                return self.recent_recommendations[-1]
            # En caso contrario, mantener configuración actual
            return {
                "redis_workers": current_redis_workers,
                "rabbitmq_workers": current_rabbitmq_workers
            }
        
        # Predecir carga futura
        predicted_ops = self.load_predictor.predict(current_ops, current_latency)
        
        # Calcular workers necesarios
        target_redis = max(
            self.min_redis_workers,
            min(self.max_redis_workers, int(predicted_ops / self.ops_per_redis_worker) + 1)
        )
        
        # RabbitMQ necesita menos workers porque maneja operaciones a través de canales
        target_rabbitmq = max(
            self.min_rabbitmq_workers,
            min(self.max_rabbitmq_workers, int(predicted_ops / self.ops_per_rabbitmq_workers) + 1)
        )
        
        # Estabilizar las recomendaciones (evitar oscilaciones)
        recommendation = {
            "redis_workers": target_redis,
            "rabbitmq_workers": target_rabbitmq
        }
        
        # Añadir a historial
        self.recent_recommendations.append(recommendation)
        if len(self.recent_recommendations) > self.max_recommendations:
            self.recent_recommendations.pop(0)
        
        # Actualizar tiempo
        self.last_optimization = current_time
        
        # Registrar recomendación
        logger.debug(f"Recomendación de recursos: Redis={target_redis}, RabbitMQ={target_rabbitmq} " +
                   f"(Carga predicha: {predicted_ops:.2f} ops/s)")
        
        return recommendation

class DivineMachineLearning:
    """
    Sistema integrado de Machine Learning para el sistema divino.
    
    Esta clase combina todos los modelos predictivos y adaptativos
    en una interfaz unificada para el sistema divino.
    """
    
    def __init__(self):
        """Inicializar sistema de ML divino."""
        self.load_predictor = DivineLoadPredictor()
        self.priority_classifier = DivinePriorityClassifier()
        self.resource_optimizer = DivineResourceOptimizer(self.load_predictor)
        self.initialized = True
        
        # Métricas de rendimiento
        self.total_predictions = 0
        self.prediction_accuracy = 0.0
        self.start_time = time.time()
        
        logger.info("Sistema de Machine Learning divino inicializado")
    
    def record_operation(self, timestamp: float, ops: float, latency: float) -> None:
        """
        Registrar operación para entrenamiento.
        
        Args:
            timestamp: Marca de tiempo
            ops: Operaciones por segundo
            latency: Latencia en segundos
        """
        self.load_predictor.add_data_point(timestamp, ops, latency)
    
    def record_priority_assignment(self, critical: bool, volume: float, 
                                  transactional: bool, priority: int) -> None:
        """
        Registrar asignación de prioridad para entrenamiento.
        
        Args:
            critical: Si la operación es crítica
            volume: Volumen o importancia (0-1000)
            transactional: Si la operación es transaccional
            priority: Prioridad asignada (1-10)
        """
        self.priority_classifier.add_training_data(critical, volume, transactional, priority)
    
    def predict_priority(self, critical: bool, volume: float, transactional: bool) -> int:
        """
        Predecir prioridad óptima para una operación.
        
        Args:
            critical: Si la operación es crítica
            volume: Volumen o importancia (0-1000)
            transactional: Si la operación es transaccional
            
        Returns:
            Prioridad recomendada (1-10)
        """
        self.total_predictions += 1
        return self.priority_classifier.predict_priority(critical, volume, transactional)
    
    def optimize_resources(self, current_ops: float, current_latency: float,
                          current_redis_workers: int, current_rabbitmq_workers: int) -> Dict[str, int]:
        """
        Optimizar recursos según predicciones.
        
        Args:
            current_ops: Operaciones por segundo actuales
            current_latency: Latencia actual en segundos
            current_redis_workers: Número actual de workers Redis
            current_rabbitmq_workers: Número actual de workers RabbitMQ
            
        Returns:
            Recomendación de recursos
        """
        return self.resource_optimizer.optimize(
            current_ops, current_latency, current_redis_workers, current_rabbitmq_workers
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del sistema ML.
        
        Returns:
            Estadísticas detalladas
        """
        uptime = time.time() - self.start_time
        
        return {
            "ml_available": ML_AVAILABLE,
            "initialized": self.initialized,
            "total_predictions": self.total_predictions,
            "prediction_accuracy": self.prediction_accuracy,
            "uptime_seconds": uptime,
            "load_prediction": {
                "data_points": len(self.load_predictor.data),
                "last_training": datetime.fromtimestamp(self.load_predictor.last_training).isoformat() if self.load_predictor.last_training > 0 else None
            },
            "priority_classification": {
                "data_points": len(self.priority_classifier.data),
                "last_training": datetime.fromtimestamp(self.priority_classifier.last_training).isoformat() if self.priority_classifier.last_training > 0 else None
            }
        }

# Instancia global
divine_ml = DivineMachineLearning()