"""
Modelos de Gradient Boosting (XGBoost, LightGBM) para predicción y clasificación.

Este módulo implementa modelos de machine learning basados en gradient boosting,
que son muy efectivos para la predicción de series temporales y la clasificación
de señales de trading.
"""

import numpy as np
import pandas as pd
import logging
import time
import os
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

# Verificar disponibilidad de XGBoost y LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class BoostingModelBase(ABC):
    """
    Clase base para modelos de Gradient Boosting.
    
    Define la interfaz común para todos los modelos de gradient boosting
    como XGBoost y LightGBM.
    """
    
    def __init__(self, 
                 model_type: str = 'classifier',
                 feature_columns: Optional[List[str]] = None,
                 target_column: str = 'target',
                 test_size: float = 0.2,
                 random_state: int = 42,
                 model_params: Optional[Dict[str, Any]] = None,
                 scaler_type: str = 'standard',
                 cache_dir: str = './cache/ml_models'):
        """
        Inicializar modelo base.
        
        Args:
            model_type: Tipo de modelo ('classifier' o 'regressor')
            feature_columns: Lista de columnas de características a usar
            target_column: Nombre de la columna objetivo
            test_size: Proporción de datos para test
            random_state: Semilla para reproducibilidad
            model_params: Parámetros específicos del modelo
            scaler_type: Tipo de escalador ('standard' o 'minmax')
            cache_dir: Directorio para guardar modelos entrenados
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.model_type = model_type
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.model_params = model_params or {}
        self.scaler_type = scaler_type
        self.cache_dir = cache_dir
        
        # Crear directorio de caché si no existe
        os.makedirs(cache_dir, exist_ok=True)
        
        # Modelo interno
        self.model = None
        self.scaler = None
        self.feature_importance = {}
        self.training_history = {}
        self.metrics = {}
        
        self.logger.info(f"Inicializado modelo {self.__class__.__name__} de tipo {model_type}")
    
    @abstractmethod
    def _create_model(self) -> Any:
        """Crear instancia del modelo específico."""
        pass
    
    def _create_scaler(self) -> Any:
        """Crear instancia del escalador de características."""
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'minmax':
            return MinMaxScaler()
        else:
            self.logger.warning(f"Tipo de escalador no reconocido: {self.scaler_type}. Usando StandardScaler.")
            return StandardScaler()
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesar datos para entrenamiento o predicción.
        
        Args:
            data: DataFrame con datos
            
        Returns:
            Tupla de (X, y) - características y target procesados
        """
        # Verificar si tenemos columnas de características definidas
        if self.feature_columns is None:
            # Usar todas las columnas excepto el target
            self.feature_columns = [col for col in data.columns if col != self.target_column]
        
        # Verificar si el target está en los datos
        if self.target_column not in data.columns:
            raise ValueError(f"Columna target '{self.target_column}' no encontrada en los datos")
        
        # Preparar características y target
        X = data[self.feature_columns].values
        y = data[self.target_column].values
        
        # Aplicar escalado a las características si el escalador está entrenado
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return X, y
    
    def fit(self, data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Entrenar el modelo con los datos proporcionados.
        
        Args:
            data: DataFrame con datos de entrenamiento
            validation_data: DataFrame con datos de validación (opcional)
            
        Returns:
            Diccionario con métricas de entrenamiento
        """
        self.logger.info(f"Entrenando modelo con {len(data)} muestras")
        start_time = time.time()
        
        # Preprocesar datos
        X, y = self.preprocess_data(data)
        
        # Inicializar escalador y aplicar a los datos
        self.scaler = self._create_scaler()
        X = self.scaler.fit_transform(X)
        
        # Dividir datos en entrenamiento y prueba
        if validation_data is not None:
            # Usar conjunto de validación externo
            X_train, y_train = X, y
            X_val, y_val = self.preprocess_data(validation_data)
            X_val = self.scaler.transform(X_val)
        else:
            # Dividir datos en entrenamiento y validación
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
        
        # Crear modelo
        self.model = self._create_model()
        
        # Entrenar modelo (implementación específica por cada subclase)
        self._fit_model(X_train, y_train, X_val, y_val)
        
        # Calcular métricas
        if self.model_type == 'classifier':
            # Métricas de clasificación
            y_pred_train = self.model.predict(X_train)
            y_pred_val = self.model.predict(X_val)
            
            self.metrics = {
                'accuracy_train': accuracy_score(y_train, y_pred_train),
                'accuracy_val': accuracy_score(y_val, y_pred_val),
                'precision_val': precision_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'recall_val': recall_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'f1_val': f1_score(y_val, y_pred_val, average='weighted', zero_division=0)
            }
        else:
            # Métricas de regresión
            y_pred_train = self.model.predict(X_train)
            y_pred_val = self.model.predict(X_val)
            
            self.metrics = {
                'mse_train': mean_squared_error(y_train, y_pred_train),
                'mse_val': mean_squared_error(y_val, y_pred_val),
                'mae_train': mean_absolute_error(y_train, y_pred_train),
                'mae_val': mean_absolute_error(y_val, y_pred_val),
                'r2_train': r2_score(y_train, y_pred_train),
                'r2_val': r2_score(y_val, y_pred_val)
            }
            
            # Añadir RMSE
            self.metrics['rmse_train'] = np.sqrt(self.metrics['mse_train'])
            self.metrics['rmse_val'] = np.sqrt(self.metrics['mse_val'])
        
        # Registrar tiempo de entrenamiento
        training_time = time.time() - start_time
        self.metrics['training_time'] = training_time
        
        # Guardar información sobre el entrenamiento
        self.training_history = {
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(data),
            'num_features': len(self.feature_columns),
            'feature_columns': self.feature_columns,
            'model_params': self.model_params,
            'metrics': self.metrics
        }
        
        self.logger.info(f"Entrenamiento completado en {training_time:.2f} segundos")
        
        return self.metrics
    
    @abstractmethod
    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                  X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Entrenar modelo específico.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Target de entrenamiento
            X_val: Características de validación
            y_val: Target de validación
        """
        pass
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Realizar predicciones con el modelo entrenado.
        
        Args:
            data: DataFrame con datos para predicción
            
        Returns:
            Array con predicciones
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado")
        
        # Verificar si tenemos las columnas necesarias
        missing_cols = [col for col in self.feature_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Faltan columnas en los datos: {missing_cols}")
        
        # Preparar características
        X = data[self.feature_columns].values
        
        # Escalar características
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Realizar predicción
        return self.model.predict(X)
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Realizar predicciones de probabilidad con el modelo entrenado.
        Solo para clasificadores.
        
        Args:
            data: DataFrame con datos para predicción
            
        Returns:
            Array con probabilidades de predicción
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado")
        
        if self.model_type != 'classifier':
            raise ValueError("predict_proba solo está disponible para clasificadores")
        
        # Verificar si tenemos las columnas necesarias
        missing_cols = [col for col in self.feature_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Faltan columnas en los datos: {missing_cols}")
        
        # Preparar características
        X = data[self.feature_columns].values
        
        # Escalar características
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Realizar predicción
        return self.model.predict_proba(X)
    
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
            filepath = os.path.join(self.cache_dir, f"{model_name}.pkl")
        
        # Guardar modelo y metadatos
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_type': self.model_type,
            'metrics': self.metrics,
            'training_history': self.training_history,
            'model_params': self.model_params
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Modelo guardado en: {filepath}")
        
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
            # Cargar modelo y metadatos
            model_data = joblib.load(filepath)
            
            # Asignar componentes
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data['target_column']
            self.model_type = model_data['model_type']
            self.metrics = model_data['metrics']
            self.training_history = model_data['training_history']
            self.model_params = model_data['model_params']
            
            self.logger.info(f"Modelo cargado desde: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al cargar modelo: {str(e)}")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Obtener importancia de las características del modelo.
        
        Returns:
            Diccionario con importancia de características
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado")
        
        # Implementado por cada subclase específica
        feature_importance = self._get_feature_importance()
        
        # Ordenar por importancia
        sorted_importance = {k: v for k, v in sorted(
            feature_importance.items(), 
            key=lambda item: item[1], 
            reverse=True
        )}
        
        return sorted_importance
    
    @abstractmethod
    def _get_feature_importance(self) -> Dict[str, float]:
        """
        Obtener importancia de características específica del modelo.
        
        Returns:
            Diccionario con importancia de características
        """
        pass
    
    def plot_feature_importance(self, top_n: int = 10) -> str:
        """
        Generar gráfico de importancia de características.
        
        Args:
            top_n: Número de características principales a mostrar
            
        Returns:
            Imagen en formato base64
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado")
        
        # Obtener importancia de características
        feature_importance = self.get_feature_importance()
        
        # Limitar a top_n características
        if len(feature_importance) > top_n:
            top_features = dict(list(feature_importance.items())[:top_n])
        else:
            top_features = feature_importance
        
        # Crear gráfico
        plt.figure(figsize=(10, 6))
        plt.barh(list(top_features.keys()), list(top_features.values()), color='skyblue')
        plt.xlabel('Importancia')
        plt.ylabel('Característica')
        plt.title('Importancia de Características')
        plt.tight_layout()
        
        # Convertir a base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return img_str
    
    def cross_validate(self, data: pd.DataFrame, n_splits: int = 5) -> Dict[str, Any]:
        """
        Realizar validación cruzada con series temporales.
        
        Args:
            data: DataFrame con datos
            n_splits: Número de divisiones para validación cruzada
            
        Returns:
            Diccionario con resultados de validación cruzada
        """
        self.logger.info(f"Realizando validación cruzada con {n_splits} divisiones")
        
        # Preprocesar datos
        X_full, y_full = self.preprocess_data(data)
        
        # Inicializar escalador
        self.scaler = self._create_scaler()
        X_full = self.scaler.fit_transform(X_full)
        
        # Configurar validación cruzada para series temporales
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Resultados por fold
        cv_results = []
        
        # Realizar validación cruzada
        for i, (train_idx, test_idx) in enumerate(tscv.split(X_full)):
            self.logger.info(f"Ejecutando fold {i+1}/{n_splits}")
            
            # Dividir datos
            X_train, X_test = X_full[train_idx], X_full[test_idx]
            y_train, y_test = y_full[train_idx], y_full[test_idx]
            
            # Crear modelo
            model = self._create_model()
            
            # Entrenar modelo
            if self.model_type == 'classifier':
                model.fit(X_train, y_train, 
                         eval_set=[(X_train, y_train), (X_test, y_test)],
                         verbose=False)
                
                # Calcular métricas
                y_pred = model.predict(X_test)
                
                fold_metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                }
            else:
                # Regressor
                model.fit(X_train, y_train, 
                         eval_set=[(X_train, y_train), (X_test, y_test)],
                         verbose=False)
                
                # Calcular métricas
                y_pred = model.predict(X_test)
                
                fold_metrics = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred)
                }
            
            cv_results.append(fold_metrics)
        
        # Calcular métricas agregadas
        aggregated_metrics = {}
        for metric in cv_results[0].keys():
            values = [result[metric] for result in cv_results]
            aggregated_metrics[f'{metric}_mean'] = np.mean(values)
            aggregated_metrics[f'{metric}_std'] = np.std(values)
            aggregated_metrics[f'{metric}_min'] = np.min(values)
            aggregated_metrics[f'{metric}_max'] = np.max(values)
        
        # Preparar resultado
        cv_result = {
            'fold_metrics': cv_results,
            'aggregated_metrics': aggregated_metrics,
            'n_splits': n_splits,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info("Validación cruzada completada")
        
        return cv_result


class XGBoostModel(BoostingModelBase):
    """
    Modelo basado en XGBoost para predicción y clasificación.
    
    XGBoost es un algoritmo de gradient boosting optimizado que destaca
    por su eficiencia y rendimiento en problemas de ML.
    """
    
    def __init__(self, 
                 model_type: str = 'classifier',
                 feature_columns: Optional[List[str]] = None,
                 target_column: str = 'target',
                 test_size: float = 0.2,
                 random_state: int = 42,
                 model_params: Optional[Dict[str, Any]] = None,
                 scaler_type: str = 'standard',
                 cache_dir: str = './cache/ml_models',
                 early_stopping_rounds: int = 50):
        """
        Inicializar modelo XGBoost.
        
        Args:
            model_type: Tipo de modelo ('classifier' o 'regressor')
            feature_columns: Lista de columnas de características a usar
            target_column: Nombre de la columna objetivo
            test_size: Proporción de datos para test
            random_state: Semilla para reproducibilidad
            model_params: Parámetros específicos del modelo
            scaler_type: Tipo de escalador ('standard' o 'minmax')
            cache_dir: Directorio para guardar modelos entrenados
            early_stopping_rounds: Número de rondas para early stopping
        """
        # Verificar si XGBoost está disponible
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost no está disponible. Instala con: pip install xgboost")
        
        # Parámetros por defecto según el tipo de modelo
        if model_params is None:
            if model_type == 'classifier':
                model_params = {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'objective': 'binary:logistic' if model_type == 'classifier' else 'reg:squarederror',
                    'random_state': random_state,
                    'n_jobs': -1
                }
            else:
                model_params = {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'objective': 'reg:squarederror',
                    'random_state': random_state,
                    'n_jobs': -1
                }
        
        self.early_stopping_rounds = early_stopping_rounds
        
        super().__init__(
            model_type=model_type,
            feature_columns=feature_columns,
            target_column=target_column,
            test_size=test_size,
            random_state=random_state,
            model_params=model_params,
            scaler_type=scaler_type,
            cache_dir=cache_dir
        )
    
    def _create_model(self) -> Any:
        """Crear instancia del modelo XGBoost."""
        if self.model_type == 'classifier':
            return xgb.XGBClassifier(**self.model_params)
        else:
            return xgb.XGBRegressor(**self.model_params)
    
    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                  X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Entrenar modelo XGBoost.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Target de entrenamiento
            X_val: Características de validación
            y_val: Target de validación
        """
        # Entrenar modelo con early stopping
        eval_set = [(X_train, y_train), (X_val, y_val)]
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric='logloss' if self.model_type == 'classifier' else 'rmse',
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=False
        )
        
        # Guardar información sobre el entrenamiento
        self.training_history['best_iteration'] = self.model.best_iteration
        self.training_history['best_score'] = self.model.best_score
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """
        Obtener importancia de características del modelo XGBoost.
        
        Returns:
            Diccionario con importancia de características
        """
        # XGBoost feature importance
        importance = self.model.feature_importances_
        
        # Crear diccionario
        importance_dict = {}
        for i, col in enumerate(self.feature_columns):
            importance_dict[col] = float(importance[i])
        
        return importance_dict


class LightGBMModel(BoostingModelBase):
    """
    Modelo basado en LightGBM para predicción y clasificación.
    
    LightGBM es un framework de gradient boosting que utiliza algoritmos basados
    en árboles y está optimizado para velocidad y eficiencia de memoria.
    """
    
    def __init__(self, 
                 model_type: str = 'classifier',
                 feature_columns: Optional[List[str]] = None,
                 target_column: str = 'target',
                 test_size: float = 0.2,
                 random_state: int = 42,
                 model_params: Optional[Dict[str, Any]] = None,
                 scaler_type: str = 'standard',
                 cache_dir: str = './cache/ml_models',
                 early_stopping_rounds: int = 50):
        """
        Inicializar modelo LightGBM.
        
        Args:
            model_type: Tipo de modelo ('classifier' o 'regressor')
            feature_columns: Lista de columnas de características a usar
            target_column: Nombre de la columna objetivo
            test_size: Proporción de datos para test
            random_state: Semilla para reproducibilidad
            model_params: Parámetros específicos del modelo
            scaler_type: Tipo de escalador ('standard' o 'minmax')
            cache_dir: Directorio para guardar modelos entrenados
            early_stopping_rounds: Número de rondas para early stopping
        """
        # Verificar si LightGBM está disponible
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM no está disponible. Instala con: pip install lightgbm")
        
        # Parámetros por defecto según el tipo de modelo
        if model_params is None:
            if model_type == 'classifier':
                model_params = {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'objective': 'binary' if model_type == 'classifier' else 'regression',
                    'random_state': random_state,
                    'n_jobs': -1
                }
            else:
                model_params = {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'objective': 'regression',
                    'random_state': random_state,
                    'n_jobs': -1
                }
        
        self.early_stopping_rounds = early_stopping_rounds
        
        super().__init__(
            model_type=model_type,
            feature_columns=feature_columns,
            target_column=target_column,
            test_size=test_size,
            random_state=random_state,
            model_params=model_params,
            scaler_type=scaler_type,
            cache_dir=cache_dir
        )
    
    def _create_model(self) -> Any:
        """Crear instancia del modelo LightGBM."""
        if self.model_type == 'classifier':
            return lgb.LGBMClassifier(**self.model_params)
        else:
            return lgb.LGBMRegressor(**self.model_params)
    
    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                  X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Entrenar modelo LightGBM.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Target de entrenamiento
            X_val: Características de validación
            y_val: Target de validación
        """
        # Entrenar modelo con early stopping
        eval_set = [(X_val, y_val)]
        callbacks = [lgb.early_stopping(self.early_stopping_rounds)]
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks,
            verbose=False
        )
        
        # Guardar información sobre el entrenamiento
        self.training_history['best_iteration'] = self.model.best_iteration_
        self.training_history['best_score'] = self.model.best_score_
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """
        Obtener importancia de características del modelo LightGBM.
        
        Returns:
            Diccionario con importancia de características
        """
        # LightGBM feature importance
        importance = self.model.feature_importances_
        
        # Crear diccionario
        importance_dict = {}
        for i, col in enumerate(self.feature_columns):
            importance_dict[col] = float(importance[i])
        
        return importance_dict


class BoostingModelManager:
    """
    Gestor de modelos de Gradient Boosting para el Sistema Genesis.
    
    Proporciona una interfaz unificada para entrenar, evaluar y usar
    modelos de XGBoost y LightGBM en el contexto del sistema de trading.
    """
    
    def __init__(self, 
                 db: Optional[Any] = None,
                 cache_dir: str = './cache/ml_models',
                 num_cores: int = 4):
        """
        Inicializar gestor de modelos.
        
        Args:
            db: Conexión a base de datos (opcional)
            cache_dir: Directorio para caché de modelos
            num_cores: Número de núcleos para procesamiento paralelo
        """
        self.logger = logging.getLogger(__name__)
        self.db = db
        self.cache_dir = cache_dir
        self.num_cores = num_cores
        self.executor = ThreadPoolExecutor(max_workers=num_cores)
        
        # Crear directorio de caché si no existe
        os.makedirs(cache_dir, exist_ok=True)
        
        # Registro de modelos
        self.models = {}
        
        self.logger.info(f"Gestor de modelos de Boosting inicializado")
    
    async def train_model(self, 
                    symbol: str,
                    data: pd.DataFrame,
                    model_type: str = 'classifier',
                    library: str = 'xgboost',
                    feature_columns: Optional[List[str]] = None,
                    target_column: str = 'target',
                    model_params: Optional[Dict[str, Any]] = None,
                    validation_data: Optional[pd.DataFrame] = None,
                    save_model: bool = True) -> Dict[str, Any]:
        """
        Entrenar un modelo para un símbolo específico.
        
        Args:
            symbol: Símbolo del activo
            data: DataFrame con datos de entrenamiento
            model_type: Tipo de modelo ('classifier' o 'regressor')
            library: Biblioteca a usar ('xgboost' o 'lightgbm')
            feature_columns: Lista de columnas de características
            target_column: Nombre de la columna objetivo
            model_params: Parámetros específicos del modelo
            validation_data: DataFrame con datos de validación (opcional)
            save_model: Si es True, guarda el modelo entrenado
            
        Returns:
            Diccionario con resultados del entrenamiento
        """
        self.logger.info(f"Entrenando modelo {library} para {symbol}")
        
        # Seleccionar clase de modelo
        if library.lower() == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost no está disponible. Instala con: pip install xgboost")
            model_class = XGBoostModel
        elif library.lower() == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM no está disponible. Instala con: pip install lightgbm")
            model_class = LightGBMModel
        else:
            raise ValueError(f"Biblioteca no soportada: {library}. Use 'xgboost' o 'lightgbm'")
        
        # Crear modelo
        model = model_class(
            model_type=model_type,
            feature_columns=feature_columns,
            target_column=target_column,
            model_params=model_params,
            cache_dir=self.cache_dir
        )
        
        # Entrenar modelo en un hilo aparte para no bloquear
        loop = asyncio.get_event_loop()
        
        try:
            # Entrenamiento asíncrono
            training_metrics = await loop.run_in_executor(
                self.executor,
                lambda: model.fit(data, validation_data)
            )
            
            # Registrar modelo
            model_id = f"{symbol}_{library}_{model_type}_{int(time.time())}"
            self.models[model_id] = model
            
            # Guardar modelo si se solicita
            model_path = None
            if save_model:
                model_path = os.path.join(self.cache_dir, f"{model_id}.pkl")
                model.save_model(model_path)
            
            # Preparar resultado
            result = {
                'model_id': model_id,
                'symbol': symbol,
                'library': library,
                'model_type': model_type,
                'metrics': training_metrics,
                'model_path': model_path,
                'feature_columns': model.feature_columns,
                'timestamp': int(time.time())
            }
            
            # Guardar en base de datos si está disponible
            if self.db:
                try:
                    # No incluir objetos demasiado grandes
                    db_record = {
                        'model_id': model_id,
                        'symbol': symbol,
                        'library': library,
                        'model_type': model_type,
                        'metrics': training_metrics,
                        'model_path': model_path,
                        'feature_columns': model.feature_columns,
                        'target_column': target_column,
                        'model_params': model_params,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    await self.db.store('ml_models', db_record)
                    self.logger.info(f"Registro del modelo guardado en base de datos: {model_id}")
                except Exception as e:
                    self.logger.error(f"Error guardando registro del modelo en base de datos: {str(e)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error entrenando modelo: {str(e)}")
            raise
    
    async def predict(self, 
                model_id: str,
                data: pd.DataFrame,
                predict_proba: bool = False) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Realizar predicciones con un modelo entrenado.
        
        Args:
            model_id: ID del modelo a usar
            data: DataFrame con datos para predicción
            predict_proba: Si es True, devuelve probabilidades (solo clasificación)
            
        Returns:
            Array con predicciones o diccionario con resultados
        """
        # Verificar si el modelo existe
        if model_id not in self.models:
            raise ValueError(f"Modelo no encontrado: {model_id}")
        
        model = self.models[model_id]
        
        try:
            # Predicción asíncrona
            loop = asyncio.get_event_loop()
            
            if predict_proba and model.model_type == 'classifier':
                predictions = await loop.run_in_executor(
                    self.executor,
                    lambda: model.predict_proba(data)
                )
                
                # Formato más amigable para la respuesta
                result = {
                    'model_id': model_id,
                    'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
                    'prediction_type': 'probability',
                    'timestamp': int(time.time())
                }
                
                return result
            else:
                predictions = await loop.run_in_executor(
                    self.executor,
                    lambda: model.predict(data)
                )
                
                # Formato más amigable para la respuesta
                result = {
                    'model_id': model_id,
                    'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
                    'prediction_type': 'class' if model.model_type == 'classifier' else 'value',
                    'timestamp': int(time.time())
                }
                
                return result
                
        except Exception as e:
            self.logger.error(f"Error realizando predicciones: {str(e)}")
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
            if 'xgboost' in filename.lower():
                model_class = XGBoostModel
            elif 'lightgbm' in filename.lower():
                model_class = LightGBMModel
            else:
                # Por defecto, intentar XGBoost
                model_class = XGBoostModel
            
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
    
    async def get_feature_importance(self, model_id: str, top_n: int = 10) -> Dict[str, Any]:
        """
        Obtener importancia de características de un modelo.
        
        Args:
            model_id: ID del modelo
            top_n: Número de características principales a devolver
            
        Returns:
            Diccionario con importancia de características
        """
        # Verificar si el modelo existe
        if model_id not in self.models:
            raise ValueError(f"Modelo no encontrado: {model_id}")
        
        model = self.models[model_id]
        
        try:
            # Obtener importancia en un thread aparte
            loop = asyncio.get_event_loop()
            feature_importance = await loop.run_in_executor(
                self.executor,
                model.get_feature_importance
            )
            
            # Limitar a top_n características
            if len(feature_importance) > top_n:
                top_features = dict(list(feature_importance.items())[:top_n])
            else:
                top_features = feature_importance
            
            # Generar gráfico
            chart_base64 = await loop.run_in_executor(
                self.executor,
                lambda: model.plot_feature_importance(top_n)
            )
            
            # Preparar resultado
            result = {
                'model_id': model_id,
                'feature_importance': top_features,
                'chart_base64': chart_base64,
                'timestamp': int(time.time())
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error obteniendo importancia de características: {str(e)}")
            raise
    
    async def cross_validate(self, 
                       symbol: str,
                       data: pd.DataFrame,
                       library: str = 'xgboost',
                       model_type: str = 'classifier',
                       feature_columns: Optional[List[str]] = None,
                       target_column: str = 'target',
                       model_params: Optional[Dict[str, Any]] = None,
                       n_splits: int = 5) -> Dict[str, Any]:
        """
        Realizar validación cruzada para un modelo.
        
        Args:
            symbol: Símbolo del activo
            data: DataFrame con datos
            library: Biblioteca a usar ('xgboost' o 'lightgbm')
            model_type: Tipo de modelo ('classifier' o 'regressor')
            feature_columns: Lista de columnas de características
            target_column: Nombre de la columna objetivo
            model_params: Parámetros específicos del modelo
            n_splits: Número de divisiones para validación cruzada
            
        Returns:
            Diccionario con resultados de validación cruzada
        """
        self.logger.info(f"Realizando validación cruzada para {symbol} con {library}")
        
        # Seleccionar clase de modelo
        if library.lower() == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost no está disponible. Instala con: pip install xgboost")
            model_class = XGBoostModel
        elif library.lower() == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM no está disponible. Instala con: pip install lightgbm")
            model_class = LightGBMModel
        else:
            raise ValueError(f"Biblioteca no soportada: {library}. Use 'xgboost' o 'lightgbm'")
        
        # Crear modelo
        model = model_class(
            model_type=model_type,
            feature_columns=feature_columns,
            target_column=target_column,
            model_params=model_params,
            cache_dir=self.cache_dir
        )
        
        try:
            # Validación cruzada en un thread aparte
            loop = asyncio.get_event_loop()
            cv_results = await loop.run_in_executor(
                self.executor,
                lambda: model.cross_validate(data, n_splits)
            )
            
            # Añadir información adicional
            cv_results['symbol'] = symbol
            cv_results['library'] = library
            
            # Guardar en base de datos si está disponible
            if self.db:
                try:
                    await self.db.store('ml_cross_validation', {
                        'symbol': symbol,
                        'library': library,
                        'model_type': model_type,
                        'aggregated_metrics': cv_results['aggregated_metrics'],
                        'n_splits': n_splits,
                        'timestamp': datetime.now().isoformat()
                    })
                    self.logger.info(f"Resultados de validación cruzada guardados en base de datos para {symbol}")
                except Exception as e:
                    self.logger.error(f"Error guardando resultados en base de datos: {str(e)}")
            
            return cv_results
            
        except Exception as e:
            self.logger.error(f"Error en validación cruzada: {str(e)}")
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
                'library': model.__class__.__name__,
                'feature_columns': model.feature_columns,
                'metrics': model.metrics,
                'timestamp': model.training_history.get('timestamp', 'Unknown')
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
            records = await self.db.retrieve('ml_models', None)
            
            if not records:
                return []
            
            # Formatear resultados
            result = []
            for record in records:
                result.append({
                    'model_id': record.get('model_id'),
                    'symbol': record.get('symbol'),
                    'library': record.get('library'),
                    'model_type': record.get('model_type'),
                    'metrics': record.get('metrics'),
                    'model_path': record.get('model_path'),
                    'timestamp': record.get('timestamp')
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error obteniendo modelos almacenados: {str(e)}")
            return []