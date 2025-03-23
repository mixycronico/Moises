"""
Ejemplos de uso de los adaptadores extendidos de base de datos.

Este script muestra cómo utilizar los adaptadores especializados para
análisis y series temporales derivados del DivineDatabaseAdapter.
"""
import os
import asyncio
import logging
import datetime
from typing import List, Dict, Any

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("examples.extended_adapters")

# Importar adaptadores extendidos
from genesis.db.extended_divine_adapter import (
    get_analytics_db_adapter,
    get_timeseries_db_adapter,
    AnalyticsDBAdapter,
    TimeSeriesDBAdapter
)

# Ejemplo 1: Uso del adaptador analítico
def ejemplo_adaptador_analitico():
    """Demostrar el uso del adaptador analítico."""
    logger.info("Ejemplo 1: Uso del adaptador analítico")
    
    # Obtener instancia
    analytics_db = get_analytics_db_adapter()
    
    # Registrar patrones de consulta comunes
    analytics_db.register_query_pattern(
        "rendimiento_por_componente",
        """
        SELECT component_id, AVG(success_rate) as avg_success_rate, 
               COUNT(*) as total_tests
        FROM gen_components
        WHERE results_id IN (
            SELECT id FROM gen_intensity_results 
            WHERE intensity >= %(min_intensity)s
        )
        GROUP BY component_id
        ORDER BY avg_success_rate DESC
        """
    )
    
    analytics_db.register_query_pattern(
        "estadisticas_intensidad",
        """
        SELECT mode, 
               COUNT(*) as test_count,
               AVG(average_success_rate) as avg_success,
               MIN(average_success_rate) as min_success,
               MAX(average_success_rate) as max_success
        FROM gen_intensity_results
        GROUP BY mode
        ORDER BY avg_success DESC
        """
    )
    
    # Ejecutar consultas
    try:
        # Consulta simple sin parámetros
        stats = analytics_db.fetch_analytics_sync("estadisticas_intensidad")
        logger.info(f"Estadísticas por modo: {len(stats)} registros")
        for stat in stats:
            logger.info(f"  - Modo: {stat['mode']}, Tests: {stat['test_count']}, "
                      f"Éxito: {stat['avg_success']:.2f}")
        
        # Consulta con parámetros
        componentes = analytics_db.fetch_analytics_sync(
            "rendimiento_por_componente", 
            {"min_intensity": 5.0}
        )
        logger.info(f"Rendimiento componentes (intensidad >= 5): {len(componentes)} registros")
        for comp in componentes:
            logger.info(f"  - Componente: {comp['component_id']}, "
                      f"Éxito: {comp['avg_success_rate']:.2f}, "
                      f"Tests: {comp['total_tests']}")
        
        # Precargar para análisis posterior
        analytics_db.prefetch_patterns({"min_intensity": 1.0})
        
    except Exception as e:
        logger.error(f"Error en ejemplo adaptador analítico: {e}")

# Ejemplo 2: Uso del adaptador de series temporales
async def ejemplo_adaptador_series_temporales():
    """Demostrar el uso del adaptador de series temporales."""
    logger.info("Ejemplo 2: Uso del adaptador de series temporales")
    
    # Obtener instancia
    timeseries_db = get_timeseries_db_adapter()
    
    # Registrar tablas de series temporales
    timeseries_db.register_timeseries_table("gen_crypto_prices")
    timeseries_db.register_timeseries_table("gen_system_metrics")
    
    # Insertar datos de ejemplo
    try:
        # Insertar métricas del sistema
        now = datetime.datetime.now().isoformat()
        metric_data = {
            "timestamp": now,
            "metric_name": "cpu_usage",
            "metric_value": 25.5,
            "component_id": "core_processor",
            "metadata": '{"units": "percent", "host": "genesis-server"}'
        }
        
        result_id = await timeseries_db.insert_timeseries_async(
            "gen_system_metrics", 
            metric_data
        )
        logger.info(f"Métrica insertada con ID: {result_id}")
        
        # Consultar datos de series temporales
        # Definir rango temporal (últimas 24 horas)
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(hours=24)
        
        metrics = await timeseries_db.get_timeseries_async(
            "gen_system_metrics",
            start_time.isoformat(),
            end_time.isoformat(),
            fields=["timestamp", "metric_name", "metric_value", "component_id"],
            limit=10
        )
        
        logger.info(f"Métricas obtenidas: {len(metrics)}")
        for metric in metrics:
            logger.info(f"  - {metric['timestamp']}: {metric['metric_name']} = "
                      f"{metric['metric_value']} ({metric['component_id']})")
        
    except Exception as e:
        logger.error(f"Error en ejemplo adaptador series temporales: {e}")

# Función principal para ejecutar todos los ejemplos
async def main():
    """Ejecutar todos los ejemplos."""
    logger.info("Iniciando ejemplos de adaptadores extendidos...")
    
    # Ejemplos síncronos
    ejemplo_adaptador_analitico()
    
    # Ejemplos asíncronos
    await ejemplo_adaptador_series_temporales()
    
    logger.info("Ejemplos completados.")

if __name__ == "__main__":
    asyncio.run(main())