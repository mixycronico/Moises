"""
Script para importar datos JSON en la base de datos del sistema Genesis.

Este script lee los archivos JSON con resultados de pruebas y los importa
en las tablas correspondientes de la base de datos con nombres actualizados.
"""
import os
import json
import logging
import psycopg2
from psycopg2 import sql
from typing import Dict, Any, List, Optional, Tuple

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Obtener una conexión a la base de datos."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        logger.error("No se encontró la variable de entorno DATABASE_URL")
        return None
    
    try:
        conn = psycopg2.connect(db_url)
        conn.autocommit = False  # Para transacciones de datos es mejor no usar autocommit
        return conn
    except Exception as e:
        logger.error(f"Error al conectar a la base de datos: {e}")
        return None

def importar_resultados_intensidad(conn, datos: Dict[str, Any]) -> Optional[int]:
    """
    Importar datos de resultados de intensidad.
    
    Args:
        conn: Conexión a la base de datos
        datos: Datos a importar
        
    Returns:
        ID del registro creado o None si hubo error
    """
    try:
        # Validar datos mínimos requeridos
        if 'intensity' not in datos or 'average_success_rate' not in datos:
            logger.error(f"Datos incompletos para gen_intensity_results: {datos}")
            return None
        
        cursor = conn.cursor()
        
        # Insertar en la tabla gen_intensity_results (antes resultados_intensidad)
        sql = """
            INSERT INTO gen_intensity_results 
            (intensity, average_success_rate, average_essential_rate) 
            VALUES (%s, %s, %s) 
            RETURNING id
        """
        params = (
            datos.get('intensity', 0.0),
            datos.get('average_success_rate', 0.0),
            datos.get('average_essential_rate', 0.0)
        )
        
        cursor.execute(sql, params)
        result = cursor.fetchone()
        
        if result:
            id_registro = result[0]
            logger.info(f"Registro de intensidad creado con ID: {id_registro}")
            return id_registro
        else:
            logger.error("No se pudo obtener el ID del registro creado")
            return None
            
    except Exception as e:
        logger.error(f"Error al importar resultados de intensidad: {e}")
        conn.rollback()
        return None
        
def importar_ciclos_procesamiento(conn, ciclos: List[Dict[str, Any]], resultados_id: int) -> int:
    """
    Importar datos de ciclos de procesamiento.
    
    Args:
        conn: Conexión a la base de datos
        ciclos: Lista de ciclos a importar
        resultados_id: ID del registro de gen_intensity_results
        
    Returns:
        Número de ciclos importados
    """
    count = 0
    try:
        cursor = conn.cursor()
        
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
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            params = (
                ciclo.get('cycle_id', ''),
                ciclo.get('intensity', 0.0),
                ciclo.get('success_rate', 0.0),
                ciclo.get('essential_success_rate', 0.0),
                ciclo.get('total_events', 0),
                ciclo.get('successful_events', 0),
                ciclo.get('essential_total', 0),
                ciclo.get('essential_successful', 0),
                resultados_id
            )
            
            cursor.execute(sql, params)
            count += 1
            
        logger.info(f"Se importaron {count} ciclos de procesamiento")
        return count
    except Exception as e:
        logger.error(f"Error al importar ciclos de procesamiento: {e}")
        conn.rollback()
        return count
        
def importar_componentes(conn, componentes: List[Dict[str, Any]], resultados_id: int) -> int:
    """
    Importar datos de componentes.
    
    Args:
        conn: Conexión a la base de datos
        componentes: Lista de componentes a importar
        resultados_id: ID del registro de gen_intensity_results
        
    Returns:
        Número de componentes importados
    """
    count = 0
    try:
        cursor = conn.cursor()
        
        for componente in componentes:
            # Comprobar si 'id' o 'component_id' están presentes
            component_id = componente.get('component_id', componente.get('id'))
            if not component_id:
                logger.warning(f"Componente sin ID, ignorando: {componente}")
                continue
                
            # Insertar en la tabla gen_components (antes componentes)
            sql = """
                INSERT INTO gen_components
                (component_id, processed, success, failed, radiation_emissions, 
                transmutations, energy, success_rate, essential, resultados_intensidad_id) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            params = (
                component_id,
                componente.get('processed', 0),
                componente.get('success', 0),
                componente.get('failed', 0),
                componente.get('radiation_emissions', 0),
                componente.get('transmutations', 0),
                componente.get('energy', 0.0),
                componente.get('success_rate', 0.0),
                componente.get('essential', False),
                resultados_id
            )
            
            cursor.execute(sql, params)
            count += 1
            
        logger.info(f"Se importaron {count} componentes")
        return count
    except Exception as e:
        logger.error(f"Error al importar componentes: {e}")
        conn.rollback()
        return count
        
def importar_estadisticas_temporales(conn, datos: Dict[str, Any], resultados_id: int) -> Optional[int]:
    """
    Importar estadísticas temporales.
    
    Args:
        conn: Conexión a la base de datos
        datos: Datos a importar
        resultados_id: ID del registro de gen_intensity_results
        
    Returns:
        ID del registro creado o None si hubo error
    """
    try:
        # Verificar si hay datos de estadísticas temporales
        if 'temporal_stats' not in datos or not datos['temporal_stats']:
            logger.info("No hay estadísticas temporales para importar")
            return None
            
        stats = datos['temporal_stats']
        cursor = conn.cursor()
        
        # Insertar en la tabla gen_temporal_stats (antes estadisticas_temporales)
        sql = """
            INSERT INTO gen_temporal_stats
            (total_events, past_events, present_events, future_events,
            verify_continuity_total, verify_continuity_success, verify_continuity_failure,
            induce_anomaly_total, induce_anomaly_success, induce_anomaly_failure,
            record_total, record_success, record_failure,
            protection_level, initialized, last_interaction, time_since_last,
            resultados_intensidad_id) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        params = (
            stats.get('total_events', 0),
            stats.get('past_events', 0),
            stats.get('present_events', 0),
            stats.get('future_events', 0),
            stats.get('verify_continuity_total', 0),
            stats.get('verify_continuity_success', 0),
            stats.get('verify_continuity_failure', 0),
            stats.get('induce_anomaly_total', 0),
            stats.get('induce_anomaly_success', 0),
            stats.get('induce_anomaly_failure', 0),
            stats.get('record_total', 0),
            stats.get('record_success', 0),
            stats.get('record_failure', 0),
            stats.get('protection_level', 0),
            stats.get('initialized', False),
            stats.get('last_interaction', 0.0),
            stats.get('time_since_last', 0.0),
            resultados_id
        )
        
        cursor.execute(sql, params)
        result = cursor.fetchone()
        stats_id = None
        
        if result:
            stats_id = result[0]
            logger.info(f"Estadísticas temporales importadas correctamente con ID: {stats_id}")
            
            # Si hay eventos temporales, importarlos también
            if 'events' in stats and stats['events']:
                importar_eventos_temporales(conn, stats['events'], stats_id)
                
            return stats_id
        else:
            logger.warning("No se pudo obtener el ID de las estadísticas temporales")
            return None
            
    except Exception as e:
        logger.error(f"Error al importar estadísticas temporales: {e}")
        conn.rollback()
        return None
        
def importar_eventos_temporales(conn, eventos: Dict[str, Dict[str, int]], stats_id: int) -> int:
    """
    Importar eventos temporales.
    
    Args:
        conn: Conexión a la base de datos
        eventos: Diccionario de eventos temporales
        stats_id: ID del registro de estadísticas temporales
        
    Returns:
        Número de eventos importados
    """
    count = 0
    try:
        cursor = conn.cursor()
        
        for timeline, events_by_type in eventos.items():
            for event_type, event_count in events_by_type.items():
                # Insertar en la tabla gen_temporal_events (antes eventos_temporales)
                sql = """
                    INSERT INTO gen_temporal_events
                    (timeline, event_type, count, estadisticas_temporales_id) 
                    VALUES (%s, %s, %s, %s)
                """
                params = (
                    timeline,
                    event_type,
                    event_count,
                    stats_id
                )
                
                cursor.execute(sql, params)
                count += 1
                
        logger.info(f"Se importaron {count} eventos temporales")
        return count
    except Exception as e:
        logger.error(f"Error al importar eventos temporales: {e}")
        conn.rollback()
        return count
        
def importar_puntos_saturacion(conn, datos: Dict[str, Any], resultados_id: int) -> int:
    """
    Importar puntos de saturación.
    
    Args:
        conn: Conexión a la base de datos
        datos: Datos a importar
        resultados_id: ID del registro de gen_intensity_results
        
    Returns:
        Número de puntos importados
    """
    count = 0
    try:
        # Verificar si hay datos de puntos de saturación
        if 'saturation_points' not in datos or not datos['saturation_points']:
            logger.info("No hay puntos de saturación para importar")
            return 0
            
        puntos = datos['saturation_points']
        cursor = conn.cursor()
        
        # Primero creamos una estrategia
        estrategia_sql = """
            INSERT INTO gen_adaptive_strategies
            (nombre, capital_inicial, umbral_eficiencia, umbral_saturacion, tipo_modelo) 
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """
        estrategia_params = (
            f"Estrategia_Autodetectada_{resultados_id}",
            datos.get('capital_inicial', 10000.0),
            datos.get('umbral_eficiencia', 0.75),
            datos.get('umbral_saturacion', 0.9),
            datos.get('tipo_modelo', 'polynomial')
        )
        
        cursor.execute(estrategia_sql, estrategia_params)
        estrategia_result = cursor.fetchone()
        
        if not estrategia_result:
            logger.error("No se pudo crear la estrategia adaptativa")
            return 0
            
        estrategia_id = estrategia_result[0]
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
                VALUES (%s, %s, %s, %s)
            """
            params = (
                punto.get('simbolo', ''),
                punto.get('punto_saturacion', 0.0),
                punto.get('confianza', 0.0),
                estrategia_id
            )
            
            cursor.execute(sql, params)
            count += 1
            
        logger.info(f"Se importaron {count} puntos de saturación")
        return count
    except Exception as e:
        logger.error(f"Error al importar puntos de saturación: {e}")
        conn.rollback()
        return count
        
def importar_singularidad(conn, datos: Dict[str, Any]) -> Optional[int]:
    """
    Importar datos de singularidad.
    
    Args:
        conn: Conexión a la base de datos
        datos: Datos a importar
        
    Returns:
        ID del registro creado o None si hubo error
    """
    try:
        # Validar datos mínimos requeridos
        if 'nivel_singularidad' not in datos:
            logger.error(f"Datos incompletos para gen_singularity_results: {datos}")
            return None
            
        cursor = conn.cursor()
        
        # Insertar en la tabla gen_singularity_results
        sql = """
            INSERT INTO gen_singularity_results 
            (nivel_singularidad, tasa_exito, operaciones_totales, tiempo_total, 
            transmutaciones_realizadas, modo_trascendental) 
            VALUES (%s, %s, %s, %s, %s, %s) 
            RETURNING id
        """
        params = (
            datos.get('nivel_singularidad', 0.0),
            datos.get('tasa_exito', 0.0),
            datos.get('operaciones_totales', 0),
            datos.get('tiempo_total', 0.0),
            datos.get('transmutaciones_realizadas', 0),
            datos.get('modo_trascendental', 'SINGULARITY_V4')
        )
        
        cursor.execute(sql, params)
        result = cursor.fetchone()
        
        if result:
            id_registro = result[0]
            logger.info(f"Registro de singularidad creado con ID: {id_registro}")
            return id_registro
        else:
            logger.error("No se pudo obtener el ID del registro creado")
            return None
            
    except Exception as e:
        logger.error(f"Error al importar resultados de singularidad: {e}")
        conn.rollback()
        return None
        
def importar_archivo_json(conn, ruta_archivo: str) -> bool:
    """
    Importar datos de un archivo JSON.
    
    Args:
        conn: Conexión a la base de datos
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
            resultados_id = importar_singularidad(conn, datos)
        else:
            resultados_id = importar_resultados_intensidad(conn, datos)
            
        if resultados_id is None:
            logger.error(f"No se pudo importar datos principales del archivo {ruta_archivo}")
            conn.rollback()
            return False
            
        # Importar ciclos de procesamiento
        if 'cycles' in datos and datos['cycles']:
            importar_ciclos_procesamiento(conn, datos['cycles'], resultados_id)
            
        # Importar componentes
        if 'components' in datos and datos['components']:
            importar_componentes(conn, datos['components'], resultados_id)
            
        # Importar estadísticas temporales
        importar_estadisticas_temporales(conn, datos, resultados_id)
        
        # Importar puntos de saturación
        importar_puntos_saturacion(conn, datos, resultados_id)
        
        # Confirmar cambios
        conn.commit()
        
        logger.info(f"Archivo {ruta_archivo} importado correctamente")
        return True
    except Exception as e:
        logger.error(f"Error al importar archivo {ruta_archivo}: {e}")
        conn.rollback()
        return False
        
def main():
    """Función principal."""
    try:
        logger.info("Iniciando importación de datos JSON a la base de datos")
        
        # Inicializar la conexión a la base de datos
        conn = get_db_connection()
        if not conn:
            logger.error("No se pudo conectar a la base de datos")
            return
        
        try:
            # Obtener archivos JSON
            archivos_json = []
            for archivo in os.listdir('.'):
                if archivo.endswith('.json') and archivo not in ['genesis_config.json', 'package.json']:
                    if os.path.isfile(archivo):
                        archivos_json.append(archivo)
                        
            logger.info(f"Se encontraron {len(archivos_json)} archivos JSON para importar")
            
            # Importar cada archivo
            for archivo in archivos_json:
                importar_archivo_json(conn, archivo)
                
            logger.info("Importación de datos JSON completada")
        finally:
            # Cerrar conexión
            conn.close()
            
    except Exception as e:
        logger.error(f"Error en la importación: {e}")
        
if __name__ == "__main__":
    main()