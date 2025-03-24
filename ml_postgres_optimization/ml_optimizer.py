#!/usr/bin/env python3
"""
Sistema de Optimización con ML para PostgreSQL - Sistema Genesis Trascendental

Este script entrena modelos de Machine Learning para predecir latencia y errores,
permitiendo ajustes dinámicos en PostgreSQL para optimizar el rendimiento.
"""
import pandas as pd
import numpy as np
import psycopg2
import time
import os
import logging
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MLPostgresOptimizer')

# Conexión a PostgreSQL usando variables de entorno para seguridad
def get_db_connection():
    """Obtiene una conexión a la base de datos PostgreSQL usando variables de entorno."""
    try:
        conn = psycopg2.connect(
            dbname=os.environ.get("POSTGRES_DB", "postgres"),
            user=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", ""),
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=os.environ.get("POSTGRES_PORT", "5432")
        )
        logger.info("Conexión establecida con PostgreSQL")
        return conn
    except Exception as e:
        logger.error(f"Error conectando a PostgreSQL: {e}")
        raise

class MLPostgresOptimizer:
    """Optimizador de PostgreSQL basado en Machine Learning."""
    
    def __init__(self):
        """Inicializa el optimizador con configuración por defecto."""
        self.lat_model = None  # Modelo para predecir latencia
        self.err_model = None  # Modelo para predecir errores
        self.scaler = StandardScaler()  # Para normalizar features
        self.trained = False
        self.conn = None
        self.cur = None
        self.metrics_file = '/tmp/genesis_metrics.csv'
        self.metrics_history = []  # Guarda historial de métricas recientes
        
        # Configuración de parámetros PostgreSQL con límites
        self.pg_params = {
            'work_mem': {'min': 4, 'max': 64, 'default': 4, 'unit': 'MB'},
            'max_connections': {'min': 50, 'max': 200, 'default': 100, 'unit': ''},
            'shared_buffers': {'min': 128, 'max': 1024, 'default': 128, 'unit': 'MB'},
            'effective_cache_size': {'min': 256, 'max': 8192, 'default': 512, 'unit': 'MB'}
        }
        
        # Métricas de rendimiento
        self.performance = {
            'latency_reductions': 0,
            'errors_prevented': 0,
            'adjustments_made': 0
        }
        
        logger.info("Inicializando MLPostgresOptimizer")
    
    def connect(self):
        """Establece conexión a PostgreSQL."""
        if not self.conn or self.conn.closed:
            self.conn = get_db_connection()
            self.cur = self.conn.cursor()
            logger.info("Conexión establecida con PostgreSQL")
    
    def close(self):
        """Cierra conexión a PostgreSQL."""
        if self.cur:
            self.cur.close()
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("Conexión cerrada con PostgreSQL")
    
    def export_metrics(self):
        """Exporta métricas recientes a CSV para entrenamiento."""
        self.connect()
        try:
            self.cur.execute("""
                COPY (SELECT * FROM genesis_metrics ORDER BY timestamp DESC LIMIT 5000)
                TO STDOUT WITH CSV HEADER
            """)
            with open(self.metrics_file, 'w') as f:
                for line in self.cur.copy_to(f):
                    pass
            logger.info(f"Métricas exportadas a {self.metrics_file}")
            return True
        except Exception as e:
            logger.error(f"Error exportando métricas: {e}")
            return False
    
    def train_models(self, data_file=None):
        """Entrena modelos de ML con datos históricos."""
        try:
            # Usar archivo especificado o el predeterminado
            file_path = data_file or self.metrics_file
            
            # Si no existe el archivo, intentar exportar métricas
            if not os.path.exists(file_path):
                logger.info(f"Archivo {file_path} no encontrado, exportando métricas...")
                if not self.export_metrics():
                    raise FileNotFoundError(f"No se pudo crear {file_path}")
            
            # Cargar datos
            data = pd.read_csv(file_path)
            
            if len(data) < 50:
                logger.warning(f"Datos insuficientes para entrenamiento ({len(data)} filas). Se necesitan al menos 50.")
                return False
            
            # Limpiar datos
            data = data.dropna()
            
            # Features y targets
            X = data[['throughput', 'concurrency', 'work_mem_mb', 'max_connections']]
            y_latency = data['latency_ms']  # Predicción de latencia
            y_errors = (data['errors'] > 0).astype(int)  # Clasificación de errores (0 o 1)
            
            # Normalizar features
            X_scaled = self.scaler.fit_transform(X)
            
            # Dividir datos
            X_train, X_test, y_lat_train, y_lat_test = train_test_split(X_scaled, y_latency, test_size=0.2, random_state=42)
            X_err_train, X_err_test, y_err_train, y_err_test = train_test_split(X_scaled, y_errors, test_size=0.2, random_state=42)
            
            # Modelo de regresión para latencia
            self.lat_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.lat_model.fit(X_train, y_lat_train)
            lat_score = self.lat_model.score(X_test, y_lat_test)
            
            # Modelo de clasificación para errores
            self.err_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.err_model.fit(X_err_train, y_err_train)
            err_score = accuracy_score(y_err_test, self.err_model.predict(X_err_test))
            
            logger.info(f"Modelos entrenados: Latencia R² = {lat_score:.3f}, Errores Accuracy = {err_score:.3f}")
            self.trained = True
            
            # Analizar importancia de características
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'latency_importance': self.lat_model.feature_importances_,
                'error_importance': self.err_model.feature_importances_
            }).sort_values('latency_importance', ascending=False)
            
            logger.info(f"Importancia de características:\n{feature_importance}")
            
            return True
        except Exception as e:
            logger.error(f"Error entrenando modelos: {e}")
            return False
    
    def get_current_metrics(self):
        """Obtiene métricas actuales de la base de datos."""
        self.connect()
        try:
            self.cur.execute("""
                SELECT throughput, concurrency, work_mem_mb, max_connections
                FROM genesis_metrics
                ORDER BY timestamp DESC LIMIT 1
            """)
            metrics = self.cur.fetchone()
            if metrics:
                return {
                    'throughput': metrics[0],
                    'concurrency': metrics[1],
                    'work_mem_mb': metrics[2],
                    'max_connections': metrics[3]
                }
            else:
                logger.warning("No se encontraron métricas recientes")
                return None
        except Exception as e:
            logger.error(f"Error obteniendo métricas actuales: {e}")
            return None
    
    def predict_performance(self, metrics):
        """Predice rendimiento con las métricas actuales."""
        if not self.trained or not self.lat_model or not self.err_model:
            logger.warning("Modelos no entrenados. No se pueden hacer predicciones.")
            return None
        
        try:
            # Preparar datos para predicción
            X_current = [[
                metrics['throughput'],
                metrics['concurrency'],
                metrics['work_mem_mb'],
                metrics['max_connections']
            ]]
            X_current_scaled = self.scaler.transform(X_current)
            
            # Predicciones
            lat_pred = self.lat_model.predict(X_current_scaled)[0]
            err_prob = self.err_model.predict_proba(X_current_scaled)[0][1]
            
            return {
                'latency_prediction': lat_pred,
                'error_probability': err_prob
            }
        except Exception as e:
            logger.error(f"Error prediciendo rendimiento: {e}")
            return None
    
    def adjust_db_settings(self, predictions):
        """Ajusta configuración de PostgreSQL según predicciones."""
        if not predictions:
            return False
        
        self.connect()
        adjustments_made = False
        
        try:
            # Latencia alta - Ajustar work_mem
            if predictions['latency_prediction'] > 5:  # Umbral de latencia
                current = predictions.get('work_mem_mb', self.pg_params['work_mem']['default'])
                new_work_mem = min(
                    self.pg_params['work_mem']['max'],
                    max(self.pg_params['work_mem']['min'], int(current * 1.5))
                )
                
                # Solo ajustar si hay un cambio significativo
                if new_work_mem > current * 1.1:
                    self.cur.execute(f"ALTER SYSTEM SET work_mem = '{new_work_mem}MB';")
                    logger.info(f"Latencia predicha: {predictions['latency_prediction']:.2f} ms. Ajustando work_mem a {new_work_mem}MB.")
                    self.performance['latency_reductions'] += 1
                    adjustments_made = True
            
            # Probabilidad alta de error - Ajustar max_connections
            if predictions['error_probability'] > 0.7:  # Umbral de probabilidad de error
                current = predictions.get('max_connections', self.pg_params['max_connections']['default'])
                new_max_conn = max(
                    self.pg_params['max_connections']['min'],
                    min(self.pg_params['max_connections']['max'], int(current * 0.8))  # Reducir conexiones
                )
                
                # Solo ajustar si hay un cambio significativo
                if new_max_conn < current * 0.9:
                    self.cur.execute(f"ALTER SYSTEM SET max_connections = {new_max_conn};")
                    logger.info(f"Probabilidad de error: {predictions['error_probability']:.2f}. Reduciendo max_connections a {new_max_conn}.")
                    self.performance['errors_prevented'] += 1
                    adjustments_made = True
            
            # Aplicar cambios si se hizo algún ajuste
            if adjustments_made:
                self.conn.commit()
                self.cur.execute("SELECT pg_reload_conf();")  # Aplicar cambios sin reiniciar
                logger.info("Configuración de PostgreSQL actualizada")
                self.performance['adjustments_made'] += 1
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error ajustando configuración: {e}")
            return False
    
    def retry_query(self, query, params=None, max_attempts=3):
        """Ejecuta consulta con reintentos y backoff exponencial."""
        self.connect()
        for attempt in range(max_attempts):
            try:
                if params:
                    self.cur.execute(query, params)
                else:
                    self.cur.execute(query)
                self.conn.commit()
                return True
            except Exception as e:
                wait_time = 2 ** attempt  # Backoff exponencial
                logger.warning(f"Reintento {attempt + 1}: {e}. Esperando {wait_time}s")
                time.sleep(wait_time)
        
        logger.error(f"Error después de {max_attempts} reintentos")
        return False
    
    def get_performance_stats(self):
        """Obtiene estadísticas de rendimiento del optimizador."""
        return self.performance
    
    def monitor_and_optimize(self, interval=10, duration=None):
        """
        Monitorea y optimiza continuamente la base de datos.
        
        Args:
            interval: Intervalo en segundos entre optimizaciones
            duration: Duración total en segundos, None para ejecutar indefinidamente
        """
        logger.info(f"Iniciando monitoreo continuo (intervalo: {interval}s, duración: {'indefinida' if duration is None else f'{duration}s'})")
        
        # Asegurar que los modelos están entrenados
        if not self.trained:
            logger.info("Entrenando modelos iniciales...")
            self.train_models()
        
        start_time = time.time()
        iteration = 0
        
        try:
            while True:
                # Verificar si se alcanzó la duración
                if duration and time.time() - start_time > duration:
                    logger.info(f"Monitoreo finalizado después de {duration}s")
                    break
                
                iteration += 1
                logger.info(f"Iteración {iteration}")
                
                # Obtener métricas actuales
                current_metrics = self.get_current_metrics()
                if not current_metrics:
                    logger.warning("Sin métricas para esta iteración, esperando...")
                    time.sleep(interval)
                    continue
                
                # Guardar métricas para historial
                self.metrics_history.append(current_metrics)
                if len(self.metrics_history) > 100:  # Mantener solo las últimas 100
                    self.metrics_history.pop(0)
                
                # Predecir rendimiento
                predictions = self.predict_performance(current_metrics)
                if predictions:
                    logger.info(f"Predicciones: Latencia={predictions['latency_prediction']:.2f}ms, Prob.Error={predictions['error_probability']:.2f}")
                    
                    # Ajustar configuración si es necesario
                    if self.adjust_db_settings(predictions):
                        logger.info("Configuración ajustada en esta iteración")
                    else:
                        logger.info("No se requieren ajustes en esta iteración")
                
                # Reentrenar modelos periódicamente (cada 50 iteraciones)
                if iteration % 50 == 0:
                    logger.info("Reentrenando modelos con datos recientes...")
                    self.export_metrics()
                    self.train_models()
                
                # Mostrar estadísticas periódicamente
                if iteration % 10 == 0:
                    stats = self.get_performance_stats()
                    logger.info(f"Estadísticas de optimización: {stats}")
                
                # Esperar hasta la próxima iteración
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoreo interrumpido por el usuario")
        finally:
            self.close()
            stats = self.get_performance_stats()
            logger.info(f"Resumen final de optimización: {stats}")

# Función principal para ejecutar el optimizador
def main():
    """Función principal para ejecutar el optimizador de PostgreSQL basado en ML."""
    logger.info("Iniciando MLPostgresOptimizer")
    
    # Crear y configurar el optimizador
    optimizer = MLPostgresOptimizer()
    
    try:
        # Entrenar modelos iniciales
        optimizer.train_models()
        
        # Iniciar monitoreo continuo (indefinido)
        optimizer.monitor_and_optimize(interval=10)
    except KeyboardInterrupt:
        logger.info("Programa interrumpido por el usuario")
    except Exception as e:
        logger.error(f"Error en la aplicación: {e}")
    finally:
        optimizer.close()
        logger.info("MLPostgresOptimizer finalizado")

if __name__ == "__main__":
    main()