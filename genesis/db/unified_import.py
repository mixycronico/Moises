"""
Módulo unificado para importación de datos JSON a la base de datos del sistema Genesis.

Este módulo implementa una interfaz única para importar datos JSON en la base de datos,
aprovechando el DivineDatabaseAdapter para soportar tanto operaciones síncronas como asíncronas.
Elimina la duplicación existente entre los scripts import_json_data.py, import_json_data_gen.py
e import_json_data_sync.py.
"""
import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

from genesis.db.divine_database import DivineDatabaseAdapter, get_divine_db_adapter

# Configuración de logging
logger = logging.getLogger("genesis.db.unified_import")

class UnifiedImporter:
    """
    Importador unificado de datos JSON para el Sistema Genesis.
    
    Esta clase proporciona métodos para importar datos JSON en la base de datos,
    con soporte tanto síncrono como asíncrono utilizando el adaptador divino.
    """
    
    def __init__(self, db_adapter: Optional[DivineDatabaseAdapter] = None):
        """
        Inicializar importador unificado.
        
        Args:
            db_adapter: Adaptador de base de datos divino (usa global si es None)
        """
        self.db = db_adapter or get_divine_db_adapter()
        logger.info("UnifiedImporter inicializado")
    
    def import_file_sync(self, file_path: str) -> bool:
        """
        Importar datos de un archivo JSON de forma síncrona.
        
        Args:
            file_path: Ruta al archivo JSON
            
        Returns:
            True si se importaron correctamente, False en caso contrario
        """
        try:
            # Verificar que el archivo existe
            if not os.path.isfile(file_path):
                logger.error(f"El archivo no existe: {file_path}")
                return False
            
            # Leer el archivo JSON
            with open(file_path, 'r') as f:
                datos = json.load(f)
            
            logger.info(f"Archivo leído correctamente: {file_path}")
            
            # Iniciar transacción
            with self.db.transaction_sync() as tx:
                # Importar datos de resultados de intensidad
                resultados_id = self._importar_resultados_intensidad_sync(tx, datos)
                if resultados_id is None:
                    logger.error("Error al importar resultados de intensidad")
                    return False
                
                # Importar ciclos de procesamiento
                ciclos_importados = 0
                if 'ciclos' in datos:
                    ciclos_importados = self._importar_ciclos_procesamiento_sync(
                        tx, datos['ciclos'], resultados_id)
                    logger.info(f"Importados {ciclos_importados} ciclos de procesamiento")
                
                # Importar componentes
                componentes_importados = 0
                if 'componentes' in datos:
                    componentes_importados = self._importar_componentes_sync(
                        tx, datos['componentes'], resultados_id)
                    logger.info(f"Importados {componentes_importados} componentes")
                
                # Importar eventos
                eventos_importados = 0
                if 'eventos' in datos:
                    eventos_importados = self._importar_eventos_sync(
                        tx, datos['eventos'], resultados_id)
                    logger.info(f"Importados {eventos_importados} eventos")
                
                # Importar métricas
                metricas_importadas = 0
                if 'metricas' in datos:
                    metricas_importadas = self._importar_metricas_sync(
                        tx, datos['metricas'], resultados_id)
                    logger.info(f"Importadas {metricas_importadas} métricas")

            logger.info(f"Importación completa: {file_path} → ID: {resultados_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error al importar archivo {file_path}: {e}")
            return False
    
    async def import_file_async(self, file_path: str) -> bool:
        """
        Importar datos de un archivo JSON de forma asíncrona.
        
        Args:
            file_path: Ruta al archivo JSON
            
        Returns:
            True si se importaron correctamente, False en caso contrario
        """
        try:
            # Verificar que el archivo existe
            if not os.path.isfile(file_path):
                logger.error(f"El archivo no existe: {file_path}")
                return False
            
            # Leer el archivo JSON
            with open(file_path, 'r') as f:
                datos = json.load(f)
            
            logger.info(f"Archivo leído correctamente: {file_path}")
            
            # Iniciar transacción
            async with self.db.transaction_async() as tx:
                # Importar datos de resultados de intensidad
                resultados_id = await self._importar_resultados_intensidad_async(tx, datos)
                if resultados_id is None:
                    logger.error("Error al importar resultados de intensidad")
                    return False
                
                # Importar ciclos de procesamiento
                ciclos_importados = 0
                if 'ciclos' in datos:
                    ciclos_importados = await self._importar_ciclos_procesamiento_async(
                        tx, datos['ciclos'], resultados_id)
                    logger.info(f"Importados {ciclos_importados} ciclos de procesamiento")
                
                # Importar componentes
                componentes_importados = 0
                if 'componentes' in datos:
                    componentes_importados = await self._importar_componentes_async(
                        tx, datos['componentes'], resultados_id)
                    logger.info(f"Importados {componentes_importados} componentes")
                
                # Importar eventos
                eventos_importados = 0
                if 'eventos' in datos:
                    eventos_importados = await self._importar_eventos_async(
                        tx, datos['eventos'], resultados_id)
                    logger.info(f"Importados {eventos_importados} eventos")
                
                # Importar métricas
                metricas_importadas = 0
                if 'metricas' in datos:
                    metricas_importadas = await self._importar_metricas_async(
                        tx, datos['metricas'], resultados_id)
                    logger.info(f"Importadas {metricas_importadas} métricas")

            logger.info(f"Importación completa: {file_path} → ID: {resultados_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error al importar archivo {file_path}: {e}")
            return False
    
    def _importar_resultados_intensidad_sync(self, tx, datos: Dict[str, Any]) -> Optional[int]:
        """
        Importar datos de resultados de intensidad de forma síncrona.
        
        Args:
            tx: Transacción activa
            datos: Datos a importar
            
        Returns:
            ID del registro creado o None si hubo error
        """
        # Validar datos mínimos requeridos
        if 'intensity' not in datos or 'average_success_rate' not in datos:
            logger.error(f"Datos incompletos para gen_intensity_results: {datos}")
            return None
        
        # Construir consulta
        query = """
            INSERT INTO gen_intensity_results (
                intensity, 
                mode, 
                average_success_rate, 
                components_count, 
                total_events, 
                execution_time, 
                timestamp, 
                system_version,
                metadata
            ) VALUES (
                %(intensity)s, 
                %(mode)s, 
                %(average_success_rate)s, 
                %(components_count)s, 
                %(total_events)s, 
                %(execution_time)s, 
                %(timestamp)s, 
                %(system_version)s,
                %(metadata)s
            ) RETURNING id
        """
        
        # Preparar parámetros
        params = {
            'intensity': datos.get('intensity'),
            'mode': datos.get('mode', 'SINGULARITY_V4'),
            'average_success_rate': datos.get('average_success_rate'),
            'components_count': datos.get('components_count', 0),
            'total_events': datos.get('total_events', 0),
            'execution_time': datos.get('execution_time', 0),
            'timestamp': datos.get('timestamp', None),
            'system_version': datos.get('system_version', '1.0.0'),
            'metadata': json.dumps(datos.get('metadata', {}))
        }
        
        # Ejecutar consulta
        result = tx.fetch_one(query, params)
        if result and 'id' in result:
            return result['id']
        return None
    
    async def _importar_resultados_intensidad_async(self, tx, datos: Dict[str, Any]) -> Optional[int]:
        """
        Importar datos de resultados de intensidad de forma asíncrona.
        
        Args:
            tx: Transacción activa
            datos: Datos a importar
            
        Returns:
            ID del registro creado o None si hubo error
        """
        # Validar datos mínimos requeridos
        if 'intensity' not in datos or 'average_success_rate' not in datos:
            logger.error(f"Datos incompletos para gen_intensity_results: {datos}")
            return None
        
        # Construir consulta
        query = """
            INSERT INTO gen_intensity_results (
                intensity, 
                mode, 
                average_success_rate, 
                components_count, 
                total_events, 
                execution_time, 
                timestamp, 
                system_version,
                metadata
            ) VALUES (
                %(intensity)s, 
                %(mode)s, 
                %(average_success_rate)s, 
                %(components_count)s, 
                %(total_events)s, 
                %(execution_time)s, 
                %(timestamp)s, 
                %(system_version)s,
                %(metadata)s
            ) RETURNING id
        """
        
        # Preparar parámetros
        params = {
            'intensity': datos.get('intensity'),
            'mode': datos.get('mode', 'SINGULARITY_V4'),
            'average_success_rate': datos.get('average_success_rate'),
            'components_count': datos.get('components_count', 0),
            'total_events': datos.get('total_events', 0),
            'execution_time': datos.get('execution_time', 0),
            'timestamp': datos.get('timestamp', None),
            'system_version': datos.get('system_version', '1.0.0'),
            'metadata': json.dumps(datos.get('metadata', {}))
        }
        
        # Ejecutar consulta
        result = await tx.fetch_one(query, params)
        if result and 'id' in result:
            return result['id']
        return None
    
    def _importar_ciclos_procesamiento_sync(self, tx, ciclos: List[Dict[str, Any]], resultados_id: int) -> int:
        """
        Importar datos de ciclos de procesamiento de forma síncrona.
        
        Args:
            tx: Transacción activa
            ciclos: Lista de ciclos a importar
            resultados_id: ID del registro de resultados_intensidad
            
        Returns:
            Número de ciclos importados
        """
        contador = 0
        for ciclo in ciclos:
            query = """
                INSERT INTO gen_processing_cycles (
                    results_id,
                    cycle_number,
                    success_rate,
                    processing_time,
                    events_processed,
                    active_components,
                    timestamp,
                    metadata
                ) VALUES (
                    %(results_id)s,
                    %(cycle_number)s,
                    %(success_rate)s,
                    %(processing_time)s,
                    %(events_processed)s,
                    %(active_components)s,
                    %(timestamp)s,
                    %(metadata)s
                )
            """
            
            params = {
                'results_id': resultados_id,
                'cycle_number': ciclo.get('cycle_number', contador + 1),
                'success_rate': ciclo.get('success_rate', 0),
                'processing_time': ciclo.get('processing_time', 0),
                'events_processed': ciclo.get('events_processed', 0),
                'active_components': ciclo.get('active_components', 0),
                'timestamp': ciclo.get('timestamp', None),
                'metadata': json.dumps(ciclo.get('metadata', {}))
            }
            
            tx.execute(query, params)
            contador += 1
        
        return contador
    
    async def _importar_ciclos_procesamiento_async(self, tx, ciclos: List[Dict[str, Any]], resultados_id: int) -> int:
        """
        Importar datos de ciclos de procesamiento de forma asíncrona.
        
        Args:
            tx: Transacción activa
            ciclos: Lista de ciclos a importar
            resultados_id: ID del registro de resultados_intensidad
            
        Returns:
            Número de ciclos importados
        """
        contador = 0
        for ciclo in ciclos:
            query = """
                INSERT INTO gen_processing_cycles (
                    results_id,
                    cycle_number,
                    success_rate,
                    processing_time,
                    events_processed,
                    active_components,
                    timestamp,
                    metadata
                ) VALUES (
                    %(results_id)s,
                    %(cycle_number)s,
                    %(success_rate)s,
                    %(processing_time)s,
                    %(events_processed)s,
                    %(active_components)s,
                    %(timestamp)s,
                    %(metadata)s
                )
            """
            
            params = {
                'results_id': resultados_id,
                'cycle_number': ciclo.get('cycle_number', contador + 1),
                'success_rate': ciclo.get('success_rate', 0),
                'processing_time': ciclo.get('processing_time', 0),
                'events_processed': ciclo.get('events_processed', 0),
                'active_components': ciclo.get('active_components', 0),
                'timestamp': ciclo.get('timestamp', None),
                'metadata': json.dumps(ciclo.get('metadata', {}))
            }
            
            await tx.execute(query, params)
            contador += 1
        
        return contador
    
    def _importar_componentes_sync(self, tx, componentes: List[Dict[str, Any]], resultados_id: int) -> int:
        """
        Importar datos de componentes de forma síncrona.
        
        Args:
            tx: Transacción activa
            componentes: Lista de componentes a importar
            resultados_id: ID del registro de resultados_intensidad
            
        Returns:
            Número de componentes importados
        """
        contador = 0
        for componente in componentes:
            query = """
                INSERT INTO gen_components (
                    results_id,
                    component_id,
                    component_type,
                    success_rate,
                    failure_count,
                    events_processed,
                    avg_response_time,
                    metadata
                ) VALUES (
                    %(results_id)s,
                    %(component_id)s,
                    %(component_type)s,
                    %(success_rate)s,
                    %(failure_count)s,
                    %(events_processed)s,
                    %(avg_response_time)s,
                    %(metadata)s
                )
            """
            
            params = {
                'results_id': resultados_id,
                'component_id': componente.get('component_id', f'comp_{contador}'),
                'component_type': componente.get('component_type', 'UNKNOWN'),
                'success_rate': componente.get('success_rate', 0),
                'failure_count': componente.get('failure_count', 0),
                'events_processed': componente.get('events_processed', 0),
                'avg_response_time': componente.get('avg_response_time', 0),
                'metadata': json.dumps(componente.get('metadata', {}))
            }
            
            tx.execute(query, params)
            contador += 1
        
        return contador
    
    async def _importar_componentes_async(self, tx, componentes: List[Dict[str, Any]], resultados_id: int) -> int:
        """
        Importar datos de componentes de forma asíncrona.
        
        Args:
            tx: Transacción activa
            componentes: Lista de componentes a importar
            resultados_id: ID del registro de resultados_intensidad
            
        Returns:
            Número de componentes importados
        """
        contador = 0
        for componente in componentes:
            query = """
                INSERT INTO gen_components (
                    results_id,
                    component_id,
                    component_type,
                    success_rate,
                    failure_count,
                    events_processed,
                    avg_response_time,
                    metadata
                ) VALUES (
                    %(results_id)s,
                    %(component_id)s,
                    %(component_type)s,
                    %(success_rate)s,
                    %(failure_count)s,
                    %(events_processed)s,
                    %(avg_response_time)s,
                    %(metadata)s
                )
            """
            
            params = {
                'results_id': resultados_id,
                'component_id': componente.get('component_id', f'comp_{contador}'),
                'component_type': componente.get('component_type', 'UNKNOWN'),
                'success_rate': componente.get('success_rate', 0),
                'failure_count': componente.get('failure_count', 0),
                'events_processed': componente.get('events_processed', 0),
                'avg_response_time': componente.get('avg_response_time', 0),
                'metadata': json.dumps(componente.get('metadata', {}))
            }
            
            await tx.execute(query, params)
            contador += 1
        
        return contador
    
    def _importar_eventos_sync(self, tx, eventos: List[Dict[str, Any]], resultados_id: int) -> int:
        """
        Importar datos de eventos de forma síncrona.
        
        Args:
            tx: Transacción activa
            eventos: Lista de eventos a importar
            resultados_id: ID del registro de resultados_intensidad
            
        Returns:
            Número de eventos importados
        """
        contador = 0
        for evento in eventos:
            query = """
                INSERT INTO gen_events (
                    results_id,
                    cycle_number,
                    event_type,
                    source_component,
                    target_component,
                    timestamp,
                    success,
                    duration,
                    metadata
                ) VALUES (
                    %(results_id)s,
                    %(cycle_number)s,
                    %(event_type)s,
                    %(source_component)s,
                    %(target_component)s,
                    %(timestamp)s,
                    %(success)s,
                    %(duration)s,
                    %(metadata)s
                )
            """
            
            params = {
                'results_id': resultados_id,
                'cycle_number': evento.get('cycle_number', 1),
                'event_type': evento.get('event_type', 'UNKNOWN'),
                'source_component': evento.get('source_component'),
                'target_component': evento.get('target_component'),
                'timestamp': evento.get('timestamp'),
                'success': evento.get('success', True),
                'duration': evento.get('duration', 0),
                'metadata': json.dumps(evento.get('metadata', {}))
            }
            
            tx.execute(query, params)
            contador += 1
        
        return contador
    
    async def _importar_eventos_async(self, tx, eventos: List[Dict[str, Any]], resultados_id: int) -> int:
        """
        Importar datos de eventos de forma asíncrona.
        
        Args:
            tx: Transacción activa
            eventos: Lista de eventos a importar
            resultados_id: ID del registro de resultados_intensidad
            
        Returns:
            Número de eventos importados
        """
        contador = 0
        for evento in eventos:
            query = """
                INSERT INTO gen_events (
                    results_id,
                    cycle_number,
                    event_type,
                    source_component,
                    target_component,
                    timestamp,
                    success,
                    duration,
                    metadata
                ) VALUES (
                    %(results_id)s,
                    %(cycle_number)s,
                    %(event_type)s,
                    %(source_component)s,
                    %(target_component)s,
                    %(timestamp)s,
                    %(success)s,
                    %(duration)s,
                    %(metadata)s
                )
            """
            
            params = {
                'results_id': resultados_id,
                'cycle_number': evento.get('cycle_number', 1),
                'event_type': evento.get('event_type', 'UNKNOWN'),
                'source_component': evento.get('source_component'),
                'target_component': evento.get('target_component'),
                'timestamp': evento.get('timestamp'),
                'success': evento.get('success', True),
                'duration': evento.get('duration', 0),
                'metadata': json.dumps(evento.get('metadata', {}))
            }
            
            await tx.execute(query, params)
            contador += 1
        
        return contador
    
    def _importar_metricas_sync(self, tx, metricas: List[Dict[str, Any]], resultados_id: int) -> int:
        """
        Importar datos de métricas de forma síncrona.
        
        Args:
            tx: Transacción activa
            metricas: Lista de métricas a importar
            resultados_id: ID del registro de resultados_intensidad
            
        Returns:
            Número de métricas importadas
        """
        contador = 0
        for metrica in metricas:
            query = """
                INSERT INTO gen_metrics (
                    results_id,
                    cycle_number,
                    metric_name,
                    metric_value,
                    component_id,
                    timestamp,
                    metadata
                ) VALUES (
                    %(results_id)s,
                    %(cycle_number)s,
                    %(metric_name)s,
                    %(metric_value)s,
                    %(component_id)s,
                    %(timestamp)s,
                    %(metadata)s
                )
            """
            
            params = {
                'results_id': resultados_id,
                'cycle_number': metrica.get('cycle_number', 1),
                'metric_name': metrica.get('metric_name', 'unknown'),
                'metric_value': metrica.get('metric_value', 0),
                'component_id': metrica.get('component_id'),
                'timestamp': metrica.get('timestamp'),
                'metadata': json.dumps(metrica.get('metadata', {}))
            }
            
            tx.execute(query, params)
            contador += 1
        
        return contador
    
    async def _importar_metricas_async(self, tx, metricas: List[Dict[str, Any]], resultados_id: int) -> int:
        """
        Importar datos de métricas de forma asíncrona.
        
        Args:
            tx: Transacción activa
            metricas: Lista de métricas a importar
            resultados_id: ID del registro de resultados_intensidad
            
        Returns:
            Número de métricas importadas
        """
        contador = 0
        for metrica in metricas:
            query = """
                INSERT INTO gen_metrics (
                    results_id,
                    cycle_number,
                    metric_name,
                    metric_value,
                    component_id,
                    timestamp,
                    metadata
                ) VALUES (
                    %(results_id)s,
                    %(cycle_number)s,
                    %(metric_name)s,
                    %(metric_value)s,
                    %(component_id)s,
                    %(timestamp)s,
                    %(metadata)s
                )
            """
            
            params = {
                'results_id': resultados_id,
                'cycle_number': metrica.get('cycle_number', 1),
                'metric_name': metrica.get('metric_name', 'unknown'),
                'metric_value': metrica.get('metric_value', 0),
                'component_id': metrica.get('component_id'),
                'timestamp': metrica.get('timestamp'),
                'metadata': json.dumps(metrica.get('metadata', {}))
            }
            
            await tx.execute(query, params)
            contador += 1
        
        return contador

# Singleton global
_unified_importer = None

def get_unified_importer() -> UnifiedImporter:
    """
    Obtener instancia global del importador unificado.
    
    Returns:
        Instancia de UnifiedImporter
    """
    global _unified_importer
    
    if _unified_importer is None:
        _unified_importer = UnifiedImporter()
    
    return _unified_importer

# Funciones de utilidad para llamadas directas
def import_file_sync(file_path: str) -> bool:
    """
    Importar archivo JSON de forma síncrona.
    
    Args:
        file_path: Ruta al archivo JSON
        
    Returns:
        True si se importó correctamente, False en caso contrario
    """
    return get_unified_importer().import_file_sync(file_path)

async def import_file_async(file_path: str) -> bool:
    """
    Importar archivo JSON de forma asíncrona.
    
    Args:
        file_path: Ruta al archivo JSON
        
    Returns:
        True si se importó correctamente, False en caso contrario
    """
    return await get_unified_importer().import_file_async(file_path)

async def batch_import_files_async(file_paths: List[str]) -> Dict[str, bool]:
    """
    Importar múltiples archivos JSON de forma asíncrona.
    
    Args:
        file_paths: Lista de rutas a archivos JSON
        
    Returns:
        Diccionario con rutas y resultados
    """
    results = {}
    tasks = []
    
    for file_path in file_paths:
        task = asyncio.create_task(import_file_async(file_path))
        tasks.append((file_path, task))
    
    for file_path, task in tasks:
        success = await task
        results[file_path] = success
    
    return results

def batch_import_files_sync(file_paths: List[str]) -> Dict[str, bool]:
    """
    Importar múltiples archivos JSON de forma síncrona.
    
    Args:
        file_paths: Lista de rutas a archivos JSON
        
    Returns:
        Diccionario con rutas y resultados
    """
    importer = get_unified_importer()
    return {file_path: importer.import_file_sync(file_path) for file_path in file_paths}
"""