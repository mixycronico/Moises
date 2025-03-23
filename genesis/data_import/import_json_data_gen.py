"""
Script para importar datos JSON en la base de datos del sistema Genesis.

Este script lee los archivos JSON con resultados de pruebas y los importa
en las tablas correspondientes de la base de datos con nombres actualizados.
"""
import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple

from genesis.db.transcendental_database import TranscendentalDatabase

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def importar_resultados_intensidad(db: TranscendentalDatabase, datos: Dict[str, Any]) -> Optional[int]:
    """
    Importar datos de resultados de intensidad.
    
    Args:
        db: Conexión a la base de datos
        datos: Datos a importar
        
    Returns:
        ID del registro creado o None si hubo error
    """
    try:
        # Validar datos mínimos requeridos
        if 'intensity' not in datos or 'average_success_rate' not in datos:
            logger.error(f"Datos incompletos para gen_intensity_results: {datos}")
            return None
            
        # Insertar en la tabla gen_intensity_results (antes resultados_intensidad)
        sql = """
            INSERT INTO gen_intensity_results 
            (intensity, average_success_rate, average_essential_rate) 
            VALUES ($1, $2, $3) 
            RETURNING id
        """
        # Convertir parámetros a un diccionario con índices como claves
        params = {
            "1": datos.get('intensity', 0.0),
            "2": datos.get('average_success_rate', 0.0),
            "3": datos.get('average_essential_rate', 0.0)
        }
        
        result = await db.execute_query(lambda: (sql, params))
        if result and len(result) > 0:
            id_registro = result[0][0]
            logger.info(f"Registro de intensidad creado con ID: {id_registro}")
            return id_registro
        else:
            logger.error("No se pudo obtener el ID del registro creado")
            return None
            
    except Exception as e:
        logger.error(f"Error al importar resultados de intensidad: {e}")
        return None
        
async def importar_ciclos_procesamiento(db: TranscendentalDatabase, ciclos: List[Dict[str, Any]], resultados_id: int) -> int:
    """
    Importar datos de ciclos de procesamiento.
    
    Args:
        db: Conexión a la base de datos
        ciclos: Lista de ciclos a importar
        resultados_id: ID del registro de gen_intensity_results
        
    Returns:
        Número de ciclos importados
    """
    count = 0
    try:
        for ciclo in ciclos:
            # Validar datos mínimos
            if 'cycle_id' not in ciclo:
                logger.warning(f"Ciclo incompleto, ignorando: {ciclo}")
                continue
                
            # Insertar en la tabla gen_processing_cycles (antes ciclos_procesamiento)
            sql = """
                INSERT INTO gen_processing_cycles 
                (cycle_id, intensity, success_rate, essential_success_rate, 
                total_events, successful_events, essential_total, essential_successful, 
                resultados_intensidad_id) 
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """
            params = {
                "1": ciclo.get('cycle_id', ''),
                "2": ciclo.get('intensity', 0.0),
                "3": ciclo.get('success_rate', 0.0),
                "4": ciclo.get('essential_success_rate', 0.0),
                "5": ciclo.get('total_events', 0),
                "6": ciclo.get('successful_events', 0),
                "7": ciclo.get('essential_total', 0),
                "8": ciclo.get('essential_successful', 0),
                "9": resultados_id
            }
            
            await db.execute_query(lambda: (sql, params))
            count += 1
            
        logger.info(f"Se importaron {count} ciclos de procesamiento")
        return count
    except Exception as e:
        logger.error(f"Error al importar ciclos de procesamiento: {e}")
        return count
        
async def importar_componentes(db: TranscendentalDatabase, componentes: List[Dict[str, Any]], resultados_id: int) -> int:
    """
    Importar datos de componentes.
    
    Args:
        db: Conexión a la base de datos
        componentes: Lista de componentes a importar
        resultados_id: ID del registro de gen_intensity_results
        
    Returns:
        Número de componentes importados
    """
    count = 0
    try:
        for componente in componentes:
            # Validar datos mínimos
            if 'component_id' not in componente:
                logger.warning(f"Componente incompleto, ignorando: {componente}")
                continue
                
            # Insertar en la tabla gen_components (antes componentes)
            sql = """
                INSERT INTO gen_components
                (component_id, processed, success, failed, radiation_emissions, 
                transmutations, energy, success_rate, essential, resultados_intensidad_id) 
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """
            params = {
                "1": componente.get('component_id', ''),
                "2": componente.get('processed', 0),
                "3": componente.get('success', 0),
                "4": componente.get('failed', 0),
                "5": componente.get('radiation_emissions', 0),
                "6": componente.get('transmutations', 0),
                "7": componente.get('energy', 0.0),
                "8": componente.get('success_rate', 0.0),
                "9": componente.get('essential', False),
                "10": resultados_id
            }
            
            await db.execute_query(lambda: (sql, params))
            count += 1
            
        logger.info(f"Se importaron {count} componentes")
        return count
    except Exception as e:
        logger.error(f"Error al importar componentes: {e}")
        return count
        
async def importar_estadisticas_temporales(db: TranscendentalDatabase, datos: Dict[str, Any], resultados_id: int) -> bool:
    """
    Importar estadísticas temporales.
    
    Args:
        db: Conexión a la base de datos
        datos: Datos a importar
        resultados_id: ID del registro de gen_intensity_results
        
    Returns:
        True si se importó correctamente, False en caso contrario
    """
    try:
        # Verificar si hay datos de estadísticas temporales
        if 'temporal_stats' not in datos or not datos['temporal_stats']:
            logger.info("No hay estadísticas temporales para importar")
            return False
            
        stats = datos['temporal_stats']
        
        # Insertar en la tabla gen_temporal_stats (antes estadisticas_temporales)
        sql = """
            INSERT INTO gen_temporal_stats
            (total_events, past_events, present_events, future_events,
            verify_continuity_total, verify_continuity_success, verify_continuity_failure,
            induce_anomaly_total, induce_anomaly_success, induce_anomaly_failure,
            record_total, record_success, record_failure,
            protection_level, initialized, last_interaction, time_since_last,
            resultados_intensidad_id) 
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
            RETURNING id
        """
        params = {
            "1": stats.get('total_events', 0),
            "2": stats.get('past_events', 0),
            "3": stats.get('present_events', 0),
            "4": stats.get('future_events', 0),
            "5": stats.get('verify_continuity_total', 0),
            "6": stats.get('verify_continuity_success', 0),
            "7": stats.get('verify_continuity_failure', 0),
            "8": stats.get('induce_anomaly_total', 0),
            "9": stats.get('induce_anomaly_success', 0),
            "10": stats.get('induce_anomaly_failure', 0),
            "11": stats.get('record_total', 0),
            "12": stats.get('record_success', 0),
            "13": stats.get('record_failure', 0),
            "14": stats.get('protection_level', 0),
            "15": stats.get('initialized', False),
            "16": stats.get('last_interaction', 0.0),
            "17": stats.get('time_since_last', 0.0),
            "18": resultados_id
        }
        
        result = await db.execute_query(lambda: (sql, params))
        stats_id = None
        if result and len(result) > 0:
            stats_id = result[0][0]
            logger.info(f"Estadísticas temporales importadas correctamente con ID: {stats_id}")
        else:
            logger.warning("No se pudo obtener el ID de las estadísticas temporales")
            
        # Si hay eventos temporales, importarlos también
        if 'events' in stats and stats['events'] and stats_id is not None:
            await importar_eventos_temporales(db, stats['events'], stats_id)
            
        return True
    except Exception as e:
        logger.error(f"Error al importar estadísticas temporales: {e}")
        return False
        
async def importar_eventos_temporales(db: TranscendentalDatabase, eventos: Dict[str, Dict[str, int]], stats_id: int) -> int:
    """
    Importar eventos temporales.
    
    Args:
        db: Conexión a la base de datos
        eventos: Diccionario de eventos temporales
        stats_id: ID del registro de estadísticas temporales
        
    Returns:
        Número de eventos importados
    """
    count = 0
    try:
        for timeline, events_by_type in eventos.items():
            for event_type, event_count in events_by_type.items():
                # Insertar en la tabla gen_temporal_events (antes eventos_temporales)
                sql = """
                    INSERT INTO gen_temporal_events
                    (timeline, event_type, count, estadisticas_temporales_id) 
                    VALUES ($1, $2, $3, $4)
                """
                params = {
                    "1": timeline,
                    "2": event_type,
                    "3": event_count,
                    "4": stats_id
                }
                
                await db.execute_query(lambda: (sql, params))
                count += 1
                
        logger.info(f"Se importaron {count} eventos temporales")
        return count
    except Exception as e:
        logger.error(f"Error al importar eventos temporales: {e}")
        return count
        
async def importar_puntos_saturacion(db: TranscendentalDatabase, datos: Dict[str, Any], resultados_id: int) -> int:
    """
    Importar puntos de saturación.
    
    Args:
        db: Conexión a la base de datos
        datos: Datos a importar
        resultados_id: ID del registro de gen_intensity_results
        
    Returns:
        Número de puntos importados
    """
    count = 0
    try:
        # Verificar si hay datos de modelos de eficiencia
        if 'saturation_points' not in datos or not datos['saturation_points']:
            logger.info("No hay puntos de saturación para importar")
            return 0
            
        puntos = datos['saturation_points']
        
        # Primero creamos una estrategia
        estrategia_sql = """
            INSERT INTO gen_adaptive_strategies
            (nombre, capital_inicial, umbral_eficiencia, umbral_saturacion, tipo_modelo) 
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
        """
        estrategia_params = {
            "1": f"Estrategia_Autodetectada_{resultados_id}",
            "2": datos.get('capital_inicial', 10000.0),
            "3": datos.get('umbral_eficiencia', 0.75),
            "4": datos.get('umbral_saturacion', 0.9),
            "5": datos.get('tipo_modelo', 'polynomial')
        }
        
        estrategia_result = await db.execute_query(lambda: (estrategia_sql, estrategia_params))
        if not estrategia_result or len(estrategia_result) == 0:
            logger.error("No se pudo crear la estrategia adaptativa")
            return 0
            
        estrategia_id = estrategia_result[0][0]
        logger.info(f"Estrategia adaptativa creada con ID: {estrategia_id}")
        
        for punto in puntos:
            # Validar datos mínimos
            if 'simbolo' not in punto or 'punto_saturacion' not in punto:
                logger.warning(f"Punto de saturación incompleto, ignorando: {punto}")
                continue
                
            # Insertar en la tabla gen_saturation_points
            sql = """
                INSERT INTO gen_saturation_points
                (simbolo, punto_saturacion, confianza, estrategia_id) 
                VALUES ($1, $2, $3, $4)
            """
            params = {
                "1": punto.get('simbolo', ''),
                "2": punto.get('punto_saturacion', 0.0),
                "3": punto.get('confianza', 0.0),
                "4": estrategia_id
            }
            
            await db.execute_query(lambda: (sql, params))
            count += 1
            
        logger.info(f"Se importaron {count} puntos de saturación")
        return count
    except Exception as e:
        logger.error(f"Error al importar puntos de saturación: {e}")
        return count
        
async def importar_singularidad(db: TranscendentalDatabase, datos: Dict[str, Any]) -> Optional[int]:
    """
    Importar datos de singularidad.
    
    Args:
        db: Conexión a la base de datos
        datos: Datos a importar
        
    Returns:
        ID del registro creado o None si hubo error
    """
    try:
        # Validar datos mínimos requeridos
        if 'nivel_singularidad' not in datos:
            logger.error(f"Datos incompletos para gen_singularity_results: {datos}")
            return None
            
        # Insertar en la tabla gen_singularity_results
        sql = """
            INSERT INTO gen_singularity_results 
            (nivel_singularidad, tasa_exito, operaciones_totales, tiempo_total, 
            transmutaciones_realizadas, modo_trascendental) 
            VALUES ($1, $2, $3, $4, $5, $6) 
            RETURNING id
        """
        params = {
            "1": datos.get('nivel_singularidad', 0.0),
            "2": datos.get('tasa_exito', 0.0),
            "3": datos.get('operaciones_totales', 0),
            "4": datos.get('tiempo_total', 0.0),
            "5": datos.get('transmutaciones_realizadas', 0),
            "6": datos.get('modo_trascendental', 'SINGULARITY_V4')
        }
        
        result = await db.execute_query(lambda: (sql, params))
        if result and len(result) > 0:
            id_registro = result[0][0]
            logger.info(f"Registro de singularidad creado con ID: {id_registro}")
            return id_registro
        else:
            logger.error("No se pudo obtener el ID del registro creado")
            return None
            
    except Exception as e:
        logger.error(f"Error al importar resultados de singularidad: {e}")
        return None
        
async def importar_archivo_json(db: TranscendentalDatabase, ruta_archivo: str) -> bool:
    """
    Importar datos de un archivo JSON.
    
    Args:
        db: Conexión a la base de datos
        ruta_archivo: Ruta al archivo JSON
        
    Returns:
        True si se importó correctamente, False en caso contrario
    """
    try:
        logger.info(f"Importando archivo: {ruta_archivo}")
        
        # Leer el archivo JSON
        with open(ruta_archivo, 'r') as f:
            datos = json.load(f)
            
        # Importar resultados de intensidad o singularidad según corresponda
        if 'nivel_singularidad' in datos:
            resultados_id = await importar_singularidad(db, datos)
        else:
            resultados_id = await importar_resultados_intensidad(db, datos)
            
        if resultados_id is None:
            logger.error(f"No se pudo importar datos principales del archivo {ruta_archivo}")
            return False
            
        # Importar ciclos de procesamiento
        if 'cycles' in datos and datos['cycles']:
            await importar_ciclos_procesamiento(db, datos['cycles'], resultados_id)
            
        # Importar componentes
        if 'components' in datos and datos['components']:
            await importar_componentes(db, datos['components'], resultados_id)
            
        # Importar estadísticas temporales
        await importar_estadisticas_temporales(db, datos, resultados_id)
        
        # Importar puntos de saturación
        await importar_puntos_saturacion(db, datos, resultados_id)
        
        logger.info(f"Archivo {ruta_archivo} importado correctamente")
        return True
    except Exception as e:
        logger.error(f"Error al importar archivo {ruta_archivo}: {e}")
        return False
        
async def main():
    """Función principal."""
    try:
        logger.info("Iniciando importación de datos JSON a la base de datos")
        
        # Inicializar la base de datos
        db = TranscendentalDatabase()
        
        # Obtener archivos JSON
        archivos_json = []
        for archivo in os.listdir('.'):
            if archivo.endswith('.json') and not archivo == 'genesis_config.json':
                if os.path.isfile(archivo):
                    archivos_json.append(archivo)
                    
        logger.info(f"Se encontraron {len(archivos_json)} archivos JSON para importar")
        
        # Importar cada archivo
        for archivo in archivos_json:
            await importar_archivo_json(db, archivo)
            
        logger.info("Importación de datos JSON completada")
            
    except Exception as e:
        logger.error(f"Error en la importación: {e}")
        
if __name__ == "__main__":
    # Ejecutar la función principal
    asyncio.run(main())