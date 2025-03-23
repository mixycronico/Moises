"""
Modelos neuronales avanzados (RNN, LSTM, GRU) para series temporales.

Este módulo implementa modelos de redes neuronales profundas
especializados en el análisis y predicción de series temporales financieras.
"""

import numpy as np
import pandas as pd
import logging
import time
import os
import joblib
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit

# Verificar disponibilidad de TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout, BatchNormalization
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Input, Concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

class SequenceGenerator:
    """
    Generador de secuencias para modelos de series temporales.
    
    Esta clase convierte datos de series temporales en secuencias
    adecuadas para el entrenamiento de modelos RNN/LSTM/GRU.
    """
    
    def __init__(self, 
                 sequence_length: int = 10,
                 step: int = 1,
                 target_position: int = 0,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 scale_features: bool = True,
                 scale_target: bool = True):
        """
        Inicializar generador de secuencias.
        
        Args:
            sequence_length: Longitud de la secuencia (ventana de tiempo)
            step: Paso entre secuencias (1 = no hay solapamiento)
            target_position: Posición del target (0 = próximo valor, 1 = valor siguiente, etc.)
            batch_size: Tamaño del batch para entrenamiento
            shuffle: Si es True, aleatoriza los datos
            scale_features: Si es True, escala las características
            scale_target: Si es True, escala el target
        """
        self.logger = logging.getLogger(__name__)
        self.sequence_length = sequence_length
        self.step = step
        self.target_position = target_position
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.scale_features = scale_features
        self.scale_target = scale_target
        
        # Escaladores
        self.feature_scaler = None
        self.target_scaler = None
        
        self.logger.info(f"Generador de secuencias inicializado con longitud {sequence_length}")
    
    def create_sequences(self, 
                         data: pd.DataFrame,
                         feature_columns: List[str],
                         target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crear secuencias a partir de datos.
        
        Args:
            data: DataFrame con datos
            feature_columns: Lista de columnas de características
            target_column: Nombre de la columna objetivo
            
        Returns:
            Tupla de (X, y) - secuencias de características y targets
        """
        # Extraer características y target
        features = data[feature_columns].values
        targets = data[target_column].values
        
        # Escalar características si se solicita
        if self.scale_features:
            if self.feature_scaler is None:
                self.feature_scaler = StandardScaler()
                features = self.feature_scaler.fit_transform(features)
            else:
                features = self.feature_scaler.transform(features)
        
        # Escalar target si se solicita
        if self.scale_target:
            if self.target_scaler is None:
                self.target_scaler = StandardScaler()
                targets = self.target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
            else:
                targets = self.target_scaler.transform(targets.reshape(-1, 1)).flatten()
        
        # Crear secuencias
        X, y = [], []
        
        for i in range(0, len(features) - self.sequence_length - self.target_position, self.step):
            X.append(features[i:i+self.sequence_length])
            y.append(targets[i+self.sequence_length+self.target_position])
        
        return np.array(X), np.array(y)
    
    def create_multivariate_sequences(self, 
                                     data: pd.DataFrame,
                                     feature_columns: List[str],
                                     target_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crear secuencias multivariadas a partir de datos.
        
        Args:
            data: DataFrame con datos
            feature_columns: Lista de columnas de características
            target_columns: Lista de columnas objetivo
            
        Returns:
            Tupla de (X, y) - secuencias de características y targets multivariados
        """
        # Extraer características y targets
        features = data[feature_columns].values
        targets = data[target_columns].values
        
        # Escalar características si se solicita
        if self.scale_features:
            if self.feature_scaler is None:
                self.feature_scaler = StandardScaler()
                features = self.feature_scaler.fit_transform(features)
            else:
                features = self.feature_scaler.transform(features)
        
        # Escalar targets si se solicita
        if self.scale_target:
            if self.target_scaler is None:
                self.target_scaler = StandardScaler()
                targets = self.target_scaler.fit_transform(targets)
            else:
                targets = self.target_scaler.transform(targets)
        
        # Crear secuencias
        X, y = [], []
        
        for i in range(0, len(features) - self.sequence_length - self.target_position, self.step):
            X.append(features[i:i+self.sequence_length])
            y.append(targets[i+self.sequence_length+self.target_position])
        
        return np.array(X), np.array(y)
    
    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Invertir transformación del target.
        
        Args:
            y_scaled: Target escalado
            
        Returns:
            Target en escala original
        """
        if self.target_scaler is None:
            return y_scaled
        
        # Reshape si es necesario
        if len(y_scaled.shape) == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        
        return self.target_scaler.inverse_transform(y_scaled)
    
    def split_train_validation(self, 
                              X: np.ndarray, 
                              y: np.ndarray,
                              validation_split: float = 0.2,
                              time_series_split: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Dividir datos en conjuntos de entrenamiento y validación.
        
        Args:
            X: Secuencias de características
            y: Target
            validation_split: Proporción para validación
            time_series_split: Si es True, usa división temporal en lugar de aleatoria
            
        Returns:
            Tupla de (X_train, X_val, y_train, y_val)
        """
        if time_series_split:
            # División temporal (respeta el orden cronológico)
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            # División aleatoria
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, shuffle=self.shuffle
            )
        
        return X_train, X_val, y_train, y_val
    
    def save(self, filepath: str) -> None:
        """
        Guardar generador en disco.
        
        Args:
            filepath: Ruta donde guardar
        """
        # Guardar configuración y escaladores
        data = {
            'sequence_length': self.sequence_length,
            'step': self.step,
            'target_position': self.target_position,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'scale_features': self.scale_features,
            'scale_target': self.scale_target,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler
        }
        
        joblib.dump(data, filepath)
        self.logger.info(f"Generador guardado en: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'SequenceGenerator':
        """
        Cargar generador desde disco.
        
        Args:
            filepath: Ruta desde donde cargar
            
        Returns:
            Instancia de SequenceGenerator
        """
        # Cargar configuración y escaladores
        data = joblib.load(filepath)
        
        # Crear nueva instancia
        generator = cls(
            sequence_length=data['sequence_length'],
            step=data['step'],
            target_position=data['target_position'],
            batch_size=data['batch_size'],
            shuffle=data['shuffle'],
            scale_features=data['scale_features'],
            scale_target=data['scale_target']
        )
        
        # Restaurar escaladores
        generator.feature_scaler = data['feature_scaler']
        generator.target_scaler = data['target_scaler']
        
        return generator


class NeuralModel(ABC):
    """
    Clase base para modelos neuronales.
    
    Define la interfaz común para todos los modelos neuronales
    como RNN, LSTM, GRU, etc.
    """
    
    def __init__(self, 
                 model_type: str = 'regressor',
                 sequence_length: int = 10,
                 feature_dim: int = 1,
                 target_dim: int = 1,
                 batch_size: int = 32,
                 epochs: int = 100,
                 learning_rate: float = 0.001,
                 early_stopping_patience: int = 10,
                 validation_split: float = 0.2,
                 random_state: int = 42,
                 cache_dir: str = './cache/ml_models'):
        """
        Inicializar modelo neuronal base.
        
        Args:
            model_type: Tipo de modelo ('regressor' o 'classifier')
            sequence_length: Longitud de la secuencia (ventana de tiempo)
            feature_dim: Dimensión de características por paso de tiempo
            target_dim: Dimensión del target
            batch_size: Tamaño del batch para entrenamiento
            epochs: Número máximo de épocas de entrenamiento
            learning_rate: Tasa de aprendizaje para el optimizador
            early_stopping_patience: Paciencia para early stopping
            validation_split: Proporción de datos para validación
            random_state: Semilla para reproducibilidad
            cache_dir: Directorio para guardar modelos entrenados
        """
        # Verificar si TensorFlow está disponible
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow no está disponible. Instala con: pip install tensorflow")
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.target_dim = target_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.random_state = random_state
        self.cache_dir = cache_dir
        
        # Crear directorio de caché si no existe
        os.makedirs(cache_dir, exist_ok=True)
        
        # Establecer semilla para reproducibilidad
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        # Modelo interno
        self.model = None
        self.metrics = {}
        self.history = None
        self.feature_columns = None
        self.target_column = None
        
        self.logger.info(f"Inicializado modelo {self.__class__.__name__} de tipo {model_type}")
    
    @abstractmethod
    def _create_model(self) -> Model:
        """
        Crear arquitectura del modelo específico.
        
        Returns:
            Modelo de Keras
        """
        pass
    
    def compile_model(self, model: Model = None) -> Model:
        """
        Compilar modelo con configuración adecuada.
        
        Args:
            model: Modelo a compilar (si None, usa self.model)
            
        Returns:
            Modelo compilado
        """
        if model is None:
            model = self.model
            
        if model is None:
            raise ValueError("No hay modelo para compilar")
        
        # Configurar optimizador
        optimizer = Adam(learning_rate=self.learning_rate)
        
        # Configurar pérdida y métricas según el tipo de modelo
        if self.model_type == 'classifier':
            if self.target_dim == 1:
                # Clasificación binaria
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                # Clasificación multiclase
                loss = 'categorical_crossentropy'
                metrics = ['accuracy']
        else:
            # Regresión
            loss = 'mse'
            metrics = ['mae']
        
        # Compilar modelo
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return model
    
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            callbacks: Optional[List[tf.keras.callbacks.Callback]] = None) -> Dict[str, Any]:
        """
        Entrenar el modelo con los datos proporcionados.
        
        Args:
            X: Datos de entrenamiento (secuencias)
            y: Target de entrenamiento
            X_val: Datos de validación (opcional)
            y_val: Target de validación (opcional)
            callbacks: Lista de callbacks adicionales
            
        Returns:
            Diccionario con métricas de entrenamiento
        """
        self.logger.info(f"Entrenando modelo con {len(X)} muestras")
        start_time = time.time()
        
        # Verificar dimensiones de entrada
        if len(X.shape) != 3:
            raise ValueError(f"X debe tener forma (n_samples, sequence_length, feature_dim), pero tiene {X.shape}")
        
        # Actualizar dimensiones
        self.sequence_length = X.shape[1]
        self.feature_dim = X.shape[2]
        
        # Verificar dimensiones del target
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            self.target_dim = 1
        else:
            self.target_dim = y.shape[1]
        
        # Crear modelo si no existe
        if self.model is None:
            self.model = self._create_model()
            self.model = self.compile_model(self.model)
        
        # Preparar callbacks
        if callbacks is None:
            callbacks = []
        
        # Early stopping para evitar overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            mode='min'
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint para guardar el mejor modelo
        checkpoint_path = os.path.join(self.cache_dir, f"{self.__class__.__name__}_best.h5")
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=0
        )
        callbacks.append(checkpoint)
        
        # ReduceLROnPlateau para ajustar tasa de aprendizaje
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.early_stopping_patience // 2,
            min_lr=1e-6,
            mode='min'
        )
        callbacks.append(reduce_lr)
        
        # Preparar datos de validación
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Entrenar modelo
        history = self.model.fit(
            X, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data,
            validation_split=self.validation_split if validation_data is None else 0.0,
            callbacks=callbacks,
            verbose=0
        )
        
        # Guardar historia del entrenamiento
        self.history = history.history
        
        # Calcular métricas
        training_time = time.time() - start_time
        
        # Evaluar modelo
        if validation_data is not None:
            val_loss, val_metric = self.model.evaluate(X_val, y_val, verbose=0)
            metric_name = 'accuracy' if self.model_type == 'classifier' else 'mae'
            
            self.metrics = {
                'val_loss': float(val_loss),
                f'val_{metric_name}': float(val_metric),
                'training_time': training_time,
                'epochs_trained': len(history.history['loss']),
                'best_epoch': np.argmin(history.history['val_loss']) + 1
            }
        else:
            self.metrics = {
                'training_time': training_time,
                'epochs_trained': len(history.history['loss']),
                'best_epoch': np.argmin(history.history['val_loss']) + 1
            }
        
        self.logger.info(f"Entrenamiento completado en {training_time:.2f} segundos")
        
        return self.metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realizar predicciones con el modelo entrenado.
        
        Args:
            X: Datos de entrada (secuencias)
            
        Returns:
            Array con predicciones
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado")
        
        # Verificar dimensiones de entrada
        if len(X.shape) != 3:
            raise ValueError(f"X debe tener forma (n_samples, sequence_length, feature_dim), pero tiene {X.shape}")
        
        # Realizar predicción
        predictions = self.model.predict(X, batch_size=self.batch_size)
        
        return predictions
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Guardar modelo entrenado en disco.
        
        Args:
            filepath: Ruta donde guardar el modelo. Si es None, se usa una ruta por defecto.
            
        Returns:
            Ruta donde se guardó el modelo
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado")
        
        # Generar ruta si no se proporciona
        if filepath is None:
            model_name = f"{self.__class__.__name__}_{self.model_type}_{int(time.time())}"
            filepath = os.path.join(self.cache_dir, f"{model_name}.h5")
        
        # Guardar modelo de Keras
        self.model.save(filepath)
        
        # Guardar metadatos
        metadata_path = os.path.splitext(filepath)[0] + "_metadata.pkl"
        metadata = {
            'model_type': self.model_type,
            'sequence_length': self.sequence_length,
            'feature_dim': self.feature_dim,
            'target_dim': self.target_dim,
            'metrics': self.metrics,
            'history': self.history,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'class_name': self.__class__.__name__,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(metadata, metadata_path)
        self.logger.info(f"Modelo guardado en: {filepath}")
        self.logger.info(f"Metadatos guardados en: {metadata_path}")
        
        return filepath
    
    def load_model(self, filepath: str) -> bool:
        """
        Cargar modelo entrenado desde disco.
        
        Args:
            filepath: Ruta al archivo del modelo
            
        Returns:
            True si se cargó correctamente
        """
        try:
            # Cargar modelo de Keras
            self.model = load_model(filepath)
            
            # Cargar metadatos
            metadata_path = os.path.splitext(filepath)[0] + "_metadata.pkl"
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                
                # Restaurar atributos
                self.model_type = metadata.get('model_type', self.model_type)
                self.sequence_length = metadata.get('sequence_length', self.sequence_length)
                self.feature_dim = metadata.get('feature_dim', self.feature_dim)
                self.target_dim = metadata.get('target_dim', self.target_dim)
                self.metrics = metadata.get('metrics', {})
                self.history = metadata.get('history', None)
                self.feature_columns = metadata.get('feature_columns', None)
                self.target_column = metadata.get('target_column', None)
            
            self.logger.info(f"Modelo cargado desde: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al cargar modelo: {str(e)}")
            return False
    
    def plot_training_history(self) -> str:
        """
        Generar gráfico de la historia del entrenamiento.
        
        Returns:
            Imagen en formato base64
        """
        if self.history is None:
            raise ValueError("No hay historia de entrenamiento disponible")
        
        # Crear gráfico
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Gráfico de pérdida
        ax1.plot(self.history['loss'], label='Train Loss')
        if 'val_loss' in self.history:
            ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Loss During Training')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Gráfico de métrica principal
        metric_name = 'accuracy' if self.model_type == 'classifier' else 'mae'
        if metric_name in self.history:
            ax2.plot(self.history[metric_name], label=f'Train {metric_name.capitalize()}')
        if f'val_{metric_name}' in self.history:
            ax2.plot(self.history[f'val_{metric_name}'], label=f'Validation {metric_name.capitalize()}')
        ax2.set_title(f'{metric_name.capitalize()} During Training')
        ax2.set_ylabel(metric_name.capitalize())
        ax2.set_xlabel('Epoch')
        ax2.legend()
        
        plt.tight_layout()
        
        # Convertir a base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return img_str
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluar el modelo con datos de prueba.
        
        Args:
            X: Datos de prueba
            y: Target de prueba
            
        Returns:
            Diccionario con métricas de evaluación
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado")
        
        # Realizar predicciones
        y_pred = self.predict(X)
        
        # Calcular métricas según el tipo de modelo
        metrics = {}
        
        if self.model_type == 'classifier':
            # Clasificación
            if self.target_dim == 1:
                # Clasificación binaria
                y_pred_class = (y_pred > 0.5).astype(int)
                metrics['accuracy'] = accuracy_score(y, y_pred_class)
                metrics['precision'] = precision_score(y, y_pred_class, zero_division=0)
                metrics['recall'] = recall_score(y, y_pred_class, zero_division=0)
                metrics['f1'] = f1_score(y, y_pred_class, zero_division=0)
            else:
                # Clasificación multiclase
                y_pred_class = np.argmax(y_pred, axis=1)
                y_true_class = np.argmax(y, axis=1) if len(y.shape) > 1 and y.shape[1] > 1 else y
                metrics['accuracy'] = accuracy_score(y_true_class, y_pred_class)
                metrics['precision'] = precision_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
                metrics['f1'] = f1_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
        else:
            # Regresión
            metrics['mse'] = mean_squared_error(y, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y, y_pred)
            try:
                metrics['r2'] = r2_score(y, y_pred)
            except:
                metrics['r2'] = 0.0
        
        return metrics


class LSTMModel(NeuralModel):
    """
    Modelo LSTM para predicción de series temporales.
    
    LSTM (Long Short-Term Memory) es una arquitectura de red neuronal recurrente
    diseñada para procesar secuencias y capturar dependencias temporales largas.
    """
    
    def __init__(self, 
                 model_type: str = 'regressor',
                 sequence_length: int = 10,
                 feature_dim: int = 1,
                 target_dim: int = 1,
                 hidden_units: List[int] = [64, 32],
                 dropout_rate: float = 0.2,
                 batch_size: int = 32,
                 epochs: int = 100,
                 learning_rate: float = 0.001,
                 early_stopping_patience: int = 10,
                 validation_split: float = 0.2,
                 random_state: int = 42,
                 cache_dir: str = './cache/ml_models'):
        """
        Inicializar modelo LSTM.
        
        Args:
            model_type: Tipo de modelo ('regressor' o 'classifier')
            sequence_length: Longitud de la secuencia (ventana de tiempo)
            feature_dim: Dimensión de características por paso de tiempo
            target_dim: Dimensión del target
            hidden_units: Lista con unidades ocultas para cada capa LSTM
            dropout_rate: Tasa de dropout para regularización
            batch_size: Tamaño del batch para entrenamiento
            epochs: Número máximo de épocas de entrenamiento
            learning_rate: Tasa de aprendizaje para el optimizador
            early_stopping_patience: Paciencia para early stopping
            validation_split: Proporción de datos para validación
            random_state: Semilla para reproducibilidad
            cache_dir: Directorio para guardar modelos entrenados
        """
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        
        super().__init__(
            model_type=model_type,
            sequence_length=sequence_length,
            feature_dim=feature_dim,
            target_dim=target_dim,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience,
            validation_split=validation_split,
            random_state=random_state,
            cache_dir=cache_dir
        )
    
    def _create_model(self) -> Model:
        """
        Crear arquitectura del modelo LSTM.
        
        Returns:
            Modelo de Keras
        """
        model = Sequential()
        
        # Primera capa LSTM
        return_sequences = len(self.hidden_units) > 1
        model.add(LSTM(
            units=self.hidden_units[0],
            return_sequences=return_sequences,
            input_shape=(self.sequence_length, self.feature_dim)
        ))
        model.add(Dropout(self.dropout_rate))
        
        # Capas LSTM adicionales
        for i in range(1, len(self.hidden_units)):
            return_sequences = i < len(self.hidden_units) - 1
            model.add(LSTM(
                units=self.hidden_units[i],
                return_sequences=return_sequences
            ))
            model.add(Dropout(self.dropout_rate))
        
        # Capa de salida
        if self.model_type == 'classifier':
            if self.target_dim == 1:
                # Clasificación binaria
                model.add(Dense(1, activation='sigmoid'))
            else:
                # Clasificación multiclase
                model.add(Dense(self.target_dim, activation='softmax'))
        else:
            # Regresión
            model.add(Dense(self.target_dim, activation='linear'))
        
        return model


class GRUModel(NeuralModel):
    """
    Modelo GRU para predicción de series temporales.
    
    GRU (Gated Recurrent Unit) es una variante de las redes recurrentes
    que es más simple que LSTM pero mantiene buena capacidad para capturar
    dependencias temporales.
    """
    
    def __init__(self, 
                 model_type: str = 'regressor',
                 sequence_length: int = 10,
                 feature_dim: int = 1,
                 target_dim: int = 1,
                 hidden_units: List[int] = [64, 32],
                 dropout_rate: float = 0.2,
                 batch_size: int = 32,
                 epochs: int = 100,
                 learning_rate: float = 0.001,
                 early_stopping_patience: int = 10,
                 validation_split: float = 0.2,
                 random_state: int = 42,
                 cache_dir: str = './cache/ml_models'):
        """
        Inicializar modelo GRU.
        
        Args:
            model_type: Tipo de modelo ('regressor' o 'classifier')
            sequence_length: Longitud de la secuencia (ventana de tiempo)
            feature_dim: Dimensión de características por paso de tiempo
            target_dim: Dimensión del target
            hidden_units: Lista con unidades ocultas para cada capa GRU
            dropout_rate: Tasa de dropout para regularización
            batch_size: Tamaño del batch para entrenamiento
            epochs: Número máximo de épocas de entrenamiento
            learning_rate: Tasa de aprendizaje para el optimizador
            early_stopping_patience: Paciencia para early stopping
            validation_split: Proporción de datos para validación
            random_state: Semilla para reproducibilidad
            cache_dir: Directorio para guardar modelos entrenados
        """
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        
        super().__init__(
            model_type=model_type,
            sequence_length=sequence_length,
            feature_dim=feature_dim,
            target_dim=target_dim,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience,
            validation_split=validation_split,
            random_state=random_state,
            cache_dir=cache_dir
        )
    
    def _create_model(self) -> Model:
        """
        Crear arquitectura del modelo GRU.
        
        Returns:
            Modelo de Keras
        """
        model = Sequential()
        
        # Primera capa GRU
        return_sequences = len(self.hidden_units) > 1
        model.add(GRU(
            units=self.hidden_units[0],
            return_sequences=return_sequences,
            input_shape=(self.sequence_length, self.feature_dim)
        ))
        model.add(Dropout(self.dropout_rate))
        
        # Capas GRU adicionales
        for i in range(1, len(self.hidden_units)):
            return_sequences = i < len(self.hidden_units) - 1
            model.add(GRU(
                units=self.hidden_units[i],
                return_sequences=return_sequences
            ))
            model.add(Dropout(self.dropout_rate))
        
        # Capa de salida
        if self.model_type == 'classifier':
            if self.target_dim == 1:
                # Clasificación binaria
                model.add(Dense(1, activation='sigmoid'))
            else:
                # Clasificación multiclase
                model.add(Dense(self.target_dim, activation='softmax'))
        else:
            # Regresión
            model.add(Dense(self.target_dim, activation='linear'))
        
        return model


class CNNLSTMModel(NeuralModel):
    """
    Modelo híbrido CNN-LSTM para predicción de series temporales.
    
    Esta arquitectura combina CNN para extraer características locales
    con LSTM para capturar dependencias temporales, lo que la hace
    muy efectiva para series temporales complejas.
    """
    
    def __init__(self, 
                 model_type: str = 'regressor',
                 sequence_length: int = 10,
                 feature_dim: int = 1,
                 target_dim: int = 1,
                 cnn_filters: List[int] = [32, 64],
                 kernel_size: int = 3,
                 lstm_units: List[int] = [64, 32],
                 dropout_rate: float = 0.2,
                 batch_size: int = 32,
                 epochs: int = 100,
                 learning_rate: float = 0.001,
                 early_stopping_patience: int = 10,
                 validation_split: float = 0.2,
                 random_state: int = 42,
                 cache_dir: str = './cache/ml_models'):
        """
        Inicializar modelo CNN-LSTM.
        
        Args:
            model_type: Tipo de modelo ('regressor' o 'classifier')
            sequence_length: Longitud de la secuencia (ventana de tiempo)
            feature_dim: Dimensión de características por paso de tiempo
            target_dim: Dimensión del target
            cnn_filters: Lista con filtros para cada capa CNN
            kernel_size: Tamaño del kernel para CNN
            lstm_units: Lista con unidades ocultas para cada capa LSTM
            dropout_rate: Tasa de dropout para regularización
            batch_size: Tamaño del batch para entrenamiento
            epochs: Número máximo de épocas de entrenamiento
            learning_rate: Tasa de aprendizaje para el optimizador
            early_stopping_patience: Paciencia para early stopping
            validation_split: Proporción de datos para validación
            random_state: Semilla para reproducibilidad
            cache_dir: Directorio para guardar modelos entrenados
        """
        self.cnn_filters = cnn_filters
        self.kernel_size = kernel_size
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        
        super().__init__(
            model_type=model_type,
            sequence_length=sequence_length,
            feature_dim=feature_dim,
            target_dim=target_dim,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience,
            validation_split=validation_split,
            random_state=random_state,
            cache_dir=cache_dir
        )
    
    def _create_model(self) -> Model:
        """
        Crear arquitectura del modelo CNN-LSTM.
        
        Returns:
            Modelo de Keras
        """
        # Definir entrada
        input_layer = Input(shape=(self.sequence_length, self.feature_dim))
        
        # Capas CNN
        x = input_layer
        for filters in self.cnn_filters:
            x = Conv1D(filters=filters, kernel_size=self.kernel_size, padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
        
        # Capas LSTM
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            x = LSTM(units=units, return_sequences=return_sequences)(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Capa de salida
        if self.model_type == 'classifier':
            if self.target_dim == 1:
                # Clasificación binaria
                output_layer = Dense(1, activation='sigmoid')(x)
            else:
                # Clasificación multiclase
                output_layer = Dense(self.target_dim, activation='softmax')(x)
        else:
            # Regresión
            output_layer = Dense(self.target_dim, activation='linear')(x)
        
        # Crear modelo
        model = Model(inputs=input_layer, outputs=output_layer)
        
        return model


class NeuralModelManager:
    """
    Gestor de modelos neuronales para el Sistema Genesis.
    
    Proporciona una interfaz unificada para entrenar, evaluar y usar
    modelos neuronales (RNN, LSTM, GRU) en el contexto del sistema de trading.
    """
    
    def __init__(self, 
                 db: Optional[Any] = None,
                 cache_dir: str = './cache/ml_models',
                 num_cores: int = 4):
        """
        Inicializar gestor de modelos neuronales.
        
        Args:
            db: Conexión a base de datos (opcional)
            cache_dir: Directorio para caché de modelos
            num_cores: Número de núcleos para procesamiento paralelo
        """
        # Verificar si TensorFlow está disponible
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow no está disponible. Instala con: pip install tensorflow")
        
        self.logger = logging.getLogger(__name__)
        self.db = db
        self.cache_dir = cache_dir
        self.num_cores = num_cores
        self.executor = ThreadPoolExecutor(max_workers=num_cores)
        
        # Crear directorio de caché si no existe
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, 'sequence_generators'), exist_ok=True)
        
        # Registro de modelos y generadores
        self.models = {}
        self.sequence_generators = {}
        
        self.logger.info(f"Gestor de modelos neuronales inicializado")
    
    def _get_model_class(self, model_type: str) -> Type[NeuralModel]:
        """
        Obtener clase de modelo según el tipo.
        
        Args:
            model_type: Tipo de modelo ('lstm', 'gru', 'cnn_lstm')
            
        Returns:
            Clase del modelo
        """
        model_type = model_type.lower()
        
        if model_type == 'lstm':
            return LSTMModel
        elif model_type == 'gru':
            return GRUModel
        elif model_type == 'cnn_lstm':
            return CNNLSTMModel
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}. Use 'lstm', 'gru' o 'cnn_lstm'")
    
    async def create_sequence_generator(self,
                                  symbol: str,
                                  sequence_length: int = 10,
                                  step: int = 1,
                                  target_position: int = 0,
                                  batch_size: int = 32,
                                  scale_features: bool = True,
                                  scale_target: bool = True) -> str:
        """
        Crear generador de secuencias para un símbolo.
        
        Args:
            symbol: Símbolo del activo
            sequence_length: Longitud de la secuencia
            step: Paso entre secuencias
            target_position: Posición del target
            batch_size: Tamaño del batch
            scale_features: Si es True, escala las características
            scale_target: Si es True, escala el target
            
        Returns:
            ID del generador creado
        """
        self.logger.info(f"Creando generador de secuencias para {symbol}")
        
        # Crear generador
        generator = SequenceGenerator(
            sequence_length=sequence_length,
            step=step,
            target_position=target_position,
            batch_size=batch_size,
            scale_features=scale_features,
            scale_target=scale_target
        )
        
        # Registrar generador
        generator_id = f"{symbol}_seq_gen_{int(time.time())}"
        self.sequence_generators[generator_id] = generator
        
        # Guardar generador
        generator_path = os.path.join(self.cache_dir, 'sequence_generators', f"{generator_id}.pkl")
        generator.save(generator_path)
        
        self.logger.info(f"Generador creado con ID: {generator_id}")
        
        return generator_id
    
    async def prepare_sequences(self,
                          generator_id: str,
                          data: pd.DataFrame,
                          feature_columns: List[str],
                          target_column: str,
                          validation_split: float = 0.2,
                          time_series_split: bool = True) -> Dict[str, Any]:
        """
        Preparar secuencias para entrenamiento de modelos.
        
        Args:
            generator_id: ID del generador de secuencias
            data: DataFrame con datos
            feature_columns: Lista de columnas de características
            target_column: Nombre de la columna objetivo
            validation_split: Proporción para validación
            time_series_split: Si es True, usa división temporal
            
        Returns:
            Diccionario con secuencias generadas
        """
        self.logger.info(f"Preparando secuencias con generador {generator_id}")
        
        # Verificar si el generador existe
        if generator_id not in self.sequence_generators:
            raise ValueError(f"Generador no encontrado: {generator_id}")
        
        generator = self.sequence_generators[generator_id]
        
        # Ejecutar en un hilo aparte para no bloquear
        loop = asyncio.get_event_loop()
        
        try:
            # Crear secuencias
            X, y = await loop.run_in_executor(
                self.executor,
                lambda: generator.create_sequences(data, feature_columns, target_column)
            )
            
            # Dividir en entrenamiento y validación
            X_train, X_val, y_train, y_val = await loop.run_in_executor(
                self.executor,
                lambda: generator.split_train_validation(X, y, validation_split, time_series_split)
            )
            
            # Preparar resultado
            result = {
                'X_train': X_train,
                'X_val': X_val,
                'y_train': y_train,
                'y_val': y_val,
                'sequence_length': generator.sequence_length,
                'feature_dim': X.shape[2],
                'target_dim': 1 if len(y.shape) == 1 else y.shape[1],
                'num_samples': len(X),
                'num_train_samples': len(X_train),
                'num_val_samples': len(X_val),
                'feature_columns': feature_columns,
                'target_column': target_column
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error preparando secuencias: {str(e)}")
            raise
    
    async def train_model(self,
                    symbol: str,
                    sequences: Dict[str, Any],
                    model_architecture: str = 'lstm',
                    model_type: str = 'regressor',
                    hidden_units: Optional[List[int]] = None,
                    dropout_rate: float = 0.2,
                    batch_size: int = 32,
                    epochs: int = 100,
                    learning_rate: float = 0.001,
                    save_model: bool = True) -> Dict[str, Any]:
        """
        Entrenar un modelo neuronal para un símbolo específico.
        
        Args:
            symbol: Símbolo del activo
            sequences: Diccionario con secuencias (de prepare_sequences)
            model_architecture: Arquitectura del modelo ('lstm', 'gru', 'cnn_lstm')
            model_type: Tipo de modelo ('regressor' o 'classifier')
            hidden_units: Lista con unidades ocultas
            dropout_rate: Tasa de dropout
            batch_size: Tamaño del batch
            epochs: Número máximo de épocas
            learning_rate: Tasa de aprendizaje
            save_model: Si es True, guarda el modelo entrenado
            
        Returns:
            Diccionario con resultados del entrenamiento
        """
        self.logger.info(f"Entrenando modelo {model_architecture} para {symbol}")
        
        # Extraer datos de secuencias
        X_train = sequences['X_train']
        X_val = sequences['X_val']
        y_train = sequences['y_train']
        y_val = sequences['y_val']
        sequence_length = sequences['sequence_length']
        feature_dim = sequences['feature_dim']
        target_dim = sequences['target_dim']
        feature_columns = sequences['feature_columns']
        target_column = sequences['target_column']
        
        # Configurar unidades ocultas por defecto si no se proporcionan
        if hidden_units is None:
            if model_architecture == 'lstm' or model_architecture == 'gru':
                hidden_units = [64, 32]
            elif model_architecture == 'cnn_lstm':
                hidden_units = [64, 32]  # LSTM units
        
        # Seleccionar clase de modelo
        model_class = self._get_model_class(model_architecture)
        
        # Crear modelo
        if model_architecture == 'lstm' or model_architecture == 'gru':
            model = model_class(
                model_type=model_type,
                sequence_length=sequence_length,
                feature_dim=feature_dim,
                target_dim=target_dim,
                hidden_units=hidden_units,
                dropout_rate=dropout_rate,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                cache_dir=self.cache_dir
            )
        elif model_architecture == 'cnn_lstm':
            # Para CNN-LSTM necesitamos parámetros adicionales
            model = model_class(
                model_type=model_type,
                sequence_length=sequence_length,
                feature_dim=feature_dim,
                target_dim=target_dim,
                cnn_filters=[32, 64],
                kernel_size=3,
                lstm_units=hidden_units,
                dropout_rate=dropout_rate,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                cache_dir=self.cache_dir
            )
        else:
            raise ValueError(f"Arquitectura no soportada: {model_architecture}")
        
        # Configurar atributos adicionales
        model.feature_columns = feature_columns
        model.target_column = target_column
        
        # Entrenar modelo en un hilo aparte para no bloquear
        loop = asyncio.get_event_loop()
        
        try:
            # Entrenamiento asíncrono
            training_metrics = await loop.run_in_executor(
                self.executor,
                lambda: model.fit(X_train, y_train, X_val, y_val)
            )
            
            # Registrar modelo
            model_id = f"{symbol}_{model_architecture}_{model_type}_{int(time.time())}"
            self.models[model_id] = model
            
            # Guardar modelo si se solicita
            model_path = None
            if save_model:
                model_path = os.path.join(self.cache_dir, f"{model_id}.h5")
                model.save_model(model_path)
            
            # Obtener gráfico de entrenamiento
            training_plot = None
            try:
                training_plot = await loop.run_in_executor(
                    self.executor,
                    model.plot_training_history
                )
            except Exception as e:
                self.logger.warning(f"Error generando gráfico de entrenamiento: {str(e)}")
            
            # Preparar resultado
            result = {
                'model_id': model_id,
                'symbol': symbol,
                'model_architecture': model_architecture,
                'model_type': model_type,
                'metrics': training_metrics,
                'model_path': model_path,
                'feature_columns': feature_columns,
                'target_column': target_column,
                'training_plot': training_plot,
                'timestamp': int(time.time())
            }
            
            # Guardar en base de datos si está disponible
            if self.db:
                try:
                    # No incluir objetos demasiado grandes
                    db_record = {
                        'model_id': model_id,
                        'symbol': symbol,
                        'model_architecture': model_architecture,
                        'model_type': model_type,
                        'metrics': training_metrics,
                        'model_path': model_path,
                        'feature_columns': feature_columns,
                        'target_column': target_column,
                        'hidden_units': hidden_units,
                        'dropout_rate': dropout_rate,
                        'batch_size': batch_size,
                        'epochs': epochs,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    await self.db.store('ml_neural_models', db_record)
                    self.logger.info(f"Registro del modelo guardado en base de datos: {model_id}")
                except Exception as e:
                    self.logger.error(f"Error guardando registro del modelo en base de datos: {str(e)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error entrenando modelo: {str(e)}")
            raise
    
    async def predict(self,
                model_id: str,
                X: np.ndarray,
                inverse_transform: bool = True,
                generator_id: Optional[str] = None) -> np.ndarray:
        """
        Realizar predicciones con un modelo entrenado.
        
        Args:
            model_id: ID del modelo
            X: Datos de entrada (secuencias)
            inverse_transform: Si es True, invierte la transformación del target
            generator_id: ID del generador para inverse_transform (si es necesario)
            
        Returns:
            Array con predicciones
        """
        # Verificar si el modelo existe
        if model_id not in self.models:
            raise ValueError(f"Modelo no encontrado: {model_id}")
        
        model = self.models[model_id]
        
        # Ejecutar en un hilo aparte para no bloquear
        loop = asyncio.get_event_loop()
        
        try:
            # Predicción asíncrona
            predictions = await loop.run_in_executor(
                self.executor,
                lambda: model.predict(X)
            )
            
            # Invertir transformación si se solicita
            if inverse_transform and generator_id is not None:
                if generator_id not in self.sequence_generators:
                    raise ValueError(f"Generador no encontrado: {generator_id}")
                
                generator = self.sequence_generators[generator_id]
                
                predictions = await loop.run_in_executor(
                    self.executor,
                    lambda: generator.inverse_transform_target(predictions)
                )
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error realizando predicciones: {str(e)}")
            raise
    
    async def evaluate(self,
                 model_id: str,
                 X: np.ndarray,
                 y: np.ndarray) -> Dict[str, float]:
        """
        Evaluar un modelo con datos de prueba.
        
        Args:
            model_id: ID del modelo
            X: Datos de prueba
            y: Target de prueba
            
        Returns:
            Diccionario con métricas de evaluación
        """
        # Verificar si el modelo existe
        if model_id not in self.models:
            raise ValueError(f"Modelo no encontrado: {model_id}")
        
        model = self.models[model_id]
        
        # Ejecutar en un hilo aparte para no bloquear
        loop = asyncio.get_event_loop()
        
        try:
            # Evaluación asíncrona
            metrics = await loop.run_in_executor(
                self.executor,
                lambda: model.evaluate(X, y)
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluando modelo: {str(e)}")
            raise
    
    async def load_model(self, model_path: str) -> str:
        """
        Cargar un modelo desde disco.
        
        Args:
            model_path: Ruta al archivo del modelo
            
        Returns:
            ID del modelo cargado
        """
        try:
            # Inferir tipo de modelo desde el nombre del archivo
            filename = os.path.basename(model_path)
            
            if 'lstm' in filename.lower():
                model_class = LSTMModel
            elif 'gru' in filename.lower():
                model_class = GRUModel
            elif 'cnn_lstm' in filename.lower():
                model_class = CNNLSTMModel
            else:
                # Por defecto, intentar LSTM
                model_class = LSTMModel
            
            # Crear modelo temporal
            model = model_class(cache_dir=self.cache_dir)
            
            # Cargar modelo en un thread aparte
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                self.executor,
                lambda: model.load_model(model_path)
            )
            
            if not success:
                raise ValueError(f"Error al cargar modelo desde: {model_path}")
            
            # Crear ID para el modelo
            model_id = f"loaded_{os.path.splitext(filename)[0]}_{int(time.time())}"
            
            # Registrar modelo
            self.models[model_id] = model
            
            self.logger.info(f"Modelo cargado correctamente con ID: {model_id}")
            
            return model_id
            
        except Exception as e:
            self.logger.error(f"Error cargando modelo: {str(e)}")
            raise
    
    async def load_sequence_generator(self, filepath: str) -> str:
        """
        Cargar un generador de secuencias desde disco.
        
        Args:
            filepath: Ruta al archivo del generador
            
        Returns:
            ID del generador cargado
        """
        try:
            # Cargar generador
            generator = SequenceGenerator.load(filepath)
            
            # Crear ID para el generador
            filename = os.path.basename(filepath)
            generator_id = f"loaded_{os.path.splitext(filename)[0]}_{int(time.time())}"
            
            # Registrar generador
            self.sequence_generators[generator_id] = generator
            
            self.logger.info(f"Generador cargado correctamente con ID: {generator_id}")
            
            return generator_id
            
        except Exception as e:
            self.logger.error(f"Error cargando generador: {str(e)}")
            raise
    
    def get_registered_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtener lista de modelos registrados.
        
        Returns:
            Diccionario con información de los modelos
        """
        result = {}
        
        for model_id, model in self.models.items():
            result[model_id] = {
                'model_type': model.model_type,
                'architecture': model.__class__.__name__,
                'feature_columns': model.feature_columns,
                'target_column': model.target_column,
                'metrics': model.metrics
            }
        
        return result
    
    def get_registered_generators(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtener lista de generadores registrados.
        
        Returns:
            Diccionario con información de los generadores
        """
        result = {}
        
        for generator_id, generator in self.sequence_generators.items():
            result[generator_id] = {
                'sequence_length': generator.sequence_length,
                'step': generator.step,
                'target_position': generator.target_position,
                'batch_size': generator.batch_size,
                'scale_features': generator.scale_features,
                'scale_target': generator.scale_target
            }
        
        return result
    
    async def get_stored_models(self) -> List[Dict[str, Any]]:
        """
        Obtener lista de modelos almacenados en base de datos.
        
        Returns:
            Lista de diccionarios con información de los modelos
        """
        if not self.db:
            return []
        
        try:
            # Consultar base de datos
            records = await self.db.retrieve('ml_neural_models', None)
            
            if not records:
                return []
            
            # Formatear resultados
            result = []
            for record in records:
                result.append({
                    'model_id': record.get('model_id'),
                    'symbol': record.get('symbol'),
                    'model_architecture': record.get('model_architecture'),
                    'model_type': record.get('model_type'),
                    'metrics': record.get('metrics'),
                    'model_path': record.get('model_path'),
                    'timestamp': record.get('timestamp')
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error obteniendo modelos almacenados: {str(e)}")
            return []