"""
Script para importar datos JSON en la base de datos del sistema Genesis.

Este script lee los archivos JSON con resultados de pruebas y los importa
en las tablas correspondientes de la base de datos.
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
            logger.error(f"Datos incompletos para resultados_intensidad: {datos}")
            return None
            
        # Insertar en la tabla resultados_intensidad
        sql = """
            INSERT INTO resultados_intensidad 
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
        resultados_id: ID del registro de resultados_intensidad
        
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
                
            # Insertar en la tabla ciclos_procesamiento
            sql = """
                INSERT INTO ciclos_procesamiento 
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
        resultados_id: ID del registro de resultados_intensidad
        
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
                
            # Insertar en la tabla componentes
            sql = """
                INSERT INTO componentes
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
        resultados_id: ID del registro de resultados_intensidad
        
    Returns:
        True si se importó correctamente, False en caso contrario
    """
    try:
        # Verificar si hay datos de estadísticas temporales
        if 'temporal_stats' not in datos or not datos['temporal_stats']:
            logger.info("No hay estadísticas temporales para importar")
            return False
            
        stats = datos['temporal_stats']
        
        # Insertar en la tabla estadisticas_temporales
        sql = """
            INSERT INTO estadisticas_temporales
            (total_events, past_events, present_events, future_events,
            verify_continuity_total, verify_continuity_success, verify_continuity_failure,
            induce_anomaly_total, induce_anomaly_success, induce_anomaly_failure,
            record_total, record_success, record_failure,
            protection_level, initialized, last_interaction, time_since_last,
            resultados_intensidad_id) 
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
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
        
        await db.execute_query(lambda: (sql, params))
        logger.info("Estadísticas temporales importadas correctamente")
        
        # Si hay eventos temporales, importarlos también
        if 'events' in stats and stats['events']:
            await importar_eventos_temporales(db, stats['events'], resultados_id)
            
        return True
    except Exception as e:
        logger.error(f"Error al importar estadísticas temporales: {e}")
        return False
        
async def importar_eventos_temporales(db: TranscendentalDatabase, eventos: Dict[str, Dict[str, int]], resultados_id: int) -> int:
    """
    Importar eventos temporales.
    
    Args:
        db: Conexión a la base de datos
        eventos: Diccionario de eventos temporales
        resultados_id: ID del registro de resultados_intensidad
        
    Returns:
        Número de eventos importados
    """
    count = 0
    try:
        for timeline, events_by_type in eventos.items():
            for event_type, event_count in events_by_type.items():
                # Insertar en la tabla eventos_temporales
                sql = """
                    INSERT INTO eventos_temporales
                    (timeline, event_type, count, resultados_intensidad_id) 
                    VALUES ($1, $2, $3, $4)
                """
                params = {
                    "1": timeline,
                    "2": event_type,
                    "3": event_count,
                    "4": resultados_id
                }
                
                await db.execute_query(lambda: (sql, params))
                count += 1
                
        logger.info(f"Se importaron {count} eventos temporales")
        return count
    except Exception as e:
        logger.error(f"Error al importar eventos temporales: {e}")
        return count
        
async def importar_modelos_eficiencia(db: TranscendentalDatabase, datos: Dict[str, Any], resultados_id: int) -> int:
    """
    Importar modelos de eficiencia.
    
    Args:
        db: Conexión a la base de datos
        datos: Datos a importar
        resultados_id: ID del registro de resultados_intensidad
        
    Returns:
        Número de modelos importados
    """
    count = 0
    try:
        # Verificar si hay datos de modelos de eficiencia
        if 'efficiency_models' not in datos or not datos['efficiency_models']:
            logger.info("No hay modelos de eficiencia para importar")
            return 0
            
        modelos = datos['efficiency_models']
        
        for modelo in modelos:
            # Validar datos mínimos
            if 'capital_amount' not in modelo:
                logger.warning(f"Modelo incompleto, ignorando: {modelo}")
                continue
                
            # Insertar en la tabla modelos_eficiencia
            sql = """
                INSERT INTO modelos_eficiencia
                (capital_amount, eficiencia_predicha, r2_score, mae, mse,
                umbral_eficiencia, umbral_saturacion, tipo_modelo, activa,
                resultados_intensidad_id) 
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """
            params = {
                "1": modelo.get('capital_amount', 0.0),
                "2": modelo.get('predicted_efficiency', 0.0),
                "3": modelo.get('r2_score', 0.0),
                "4": modelo.get('mae', 0.0),
                "5": modelo.get('mse', 0.0),
                "6": modelo.get('efficiency_threshold', 0.0),
                "7": modelo.get('saturation_threshold', 0.0),
                "8": modelo.get('model_type', 'polynomial'),
                "9": modelo.get('active', True),
                "10": resultados_id
            }
            
            await db.execute_query(lambda: (sql, params))
            count += 1
            
        logger.info(f"Se importaron {count} modelos de eficiencia")
        return count
    except Exception as e:
        logger.error(f"Error al importar modelos de eficiencia: {e}")
        return count
        
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
            
        # Importar resultados de intensidad
        resultados_id = await importar_resultados_intensidad(db, datos)
        if resultados_id is None:
            logger.error(f"No se pudo importar resultados de intensidad del archivo {ruta_archivo}")
            return False
            
        # Importar ciclos de procesamiento
        if 'cycles' in datos and datos['cycles']:
            await importar_ciclos_procesamiento(db, datos['cycles'], resultados_id)
            
        # Importar componentes
        if 'components' in datos and datos['components']:
            await importar_componentes(db, datos['components'], resultados_id)
            
        # Importar estadísticas temporales
        await importar_estadisticas_temporales(db, datos, resultados_id)
        
        # Importar modelos de eficiencia
        await importar_modelos_eficiencia(db, datos, resultados_id)
        
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