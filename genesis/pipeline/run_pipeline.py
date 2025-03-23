"""
Script principal para ejecutar el Pipeline completo de Genesis.

Este script ejecuta todas las etapas del pipeline de forma secuencial,
desde la adquisición de datos hasta la ejecución de órdenes y la
distribución de ganancias.
"""
import asyncio
import logging
import argparse
import json
import time
from typing import Dict, Any, List, Optional

from genesis.base import setup_logging
from genesis.db.transcendental_database import TranscendentalDatabase
from genesis.pipeline.pipeline_manager import TranscendentalPipeline
from genesis.pipeline.data_acquisition import process_data_acquisition
from genesis.pipeline.processing import process_data_processing
from genesis.pipeline.analysis import process_data_analysis
from genesis.pipeline.decision import process_decision_making
from genesis.pipeline.execution import process_execution

# Configuración de logging
logger = logging.getLogger("genesis.pipeline.run")

async def setup_pipeline() -> TranscendentalPipeline:
    """
    Configurar pipeline con todas las etapas.
    
    Returns:
        Pipeline configurado
    """
    # Crear instancia de pipeline
    pipeline = TranscendentalPipeline()
    
    # Inicializar
    await pipeline.initialize()
    
    # Registrar procesadores para cada etapa
    # 1. Adquisición de datos
    acquisition_stage = pipeline.stages.get("data_acquisition")
    if acquisition_stage:
        acquisition_stage.register_processor(process_data_acquisition)
    
    # 2. Procesamiento
    processing_stage = pipeline.stages.get("data_preprocessing")
    if processing_stage:
        processing_stage.register_processor(process_data_processing)
    
    # 3. Análisis
    analysis_stage = pipeline.stages.get("model_prediction")
    if analysis_stage:
        analysis_stage.register_processor(process_data_analysis)
    
    # 4. Decisión
    decision_stage = pipeline.stages.get("signal_generation")
    if decision_stage:
        decision_stage.register_processor(process_decision_making)
    
    # 5. Ejecución
    execution_stage = pipeline.stages.get("execution")
    if execution_stage:
        execution_stage.register_processor(process_execution)
    
    logger.info("Pipeline configurado con todos los procesadores")
    return pipeline

async def run_full_pipeline(initial_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Ejecutar pipeline completo.
    
    Args:
        initial_data: Datos iniciales opcionales
        
    Returns:
        Resultado del pipeline
    """
    # Configurar pipeline
    pipeline = await setup_pipeline()
    
    # Datos iniciales por defecto
    data = initial_data or {}
    
    # Ejecutar pipeline
    logger.info("Iniciando ejecución del pipeline completo")
    result = await pipeline.execute(data)
    
    logger.info("Pipeline completado")
    return result

async def run_partial_pipeline(stages: List[str], initial_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Ejecutar pipeline parcial con etapas específicas.
    
    Args:
        stages: Lista de etapas a ejecutar
        initial_data: Datos iniciales opcionales
        
    Returns:
        Resultado del pipeline parcial
    """
    # Crear instancia de pipeline
    pipeline = TranscendentalPipeline()
    
    # Inicializar
    await pipeline.initialize()
    
    # Verificar etapas solicitadas
    valid_stages = set(pipeline.stages_order)
    requested_stages = set(stages)
    
    if not requested_stages.issubset(valid_stages):
        invalid_stages = requested_stages - valid_stages
        logger.warning(f"Etapas no válidas: {invalid_stages}")
        logger.info(f"Etapas disponibles: {valid_stages}")
        return {"error": f"Etapas no válidas: {invalid_stages}"}
    
    # Filtrar etapas y reordenar según orden original
    final_stages = [stage for stage in pipeline.stages_order if stage in requested_stages]
    
    # Reordenar stages_order (esto afecta la ejecución)
    pipeline.stages_order = final_stages
    
    # Registrar procesadores para etapas seleccionadas
    processor_map = {
        "data_acquisition": process_data_acquisition,
        "data_validation": process_data_processing,  # Asumir procesamiento en validación
        "data_preprocessing": process_data_processing,
        "feature_engineering": process_data_processing,  # Asumir procesamiento en feature engineering
        "model_prediction": process_data_analysis,
        "signal_generation": process_decision_making,
        "risk_assessment": process_decision_making,  # Asumir decisión en evaluación de riesgo
        "order_creation": process_decision_making,  # Asumir decisión en creación de órdenes
        "execution": process_execution,
        "performance_tracking": None,  # Sin procesador específico
        "profit_distribution": None,  # Sin procesador específico
        "capital_management": None,  # Sin procesador específico
        "reporting": None  # Sin procesador específico
    }
    
    # Registrar procesadores para etapas seleccionadas
    for stage_id in final_stages:
        processor = processor_map.get(stage_id)
        if processor and stage_id in pipeline.stages:
            pipeline.stages[stage_id].register_processor(processor)
    
    # Datos iniciales por defecto
    data = initial_data or {}
    
    # Ejecutar pipeline
    logger.info(f"Iniciando ejecución del pipeline parcial con etapas: {final_stages}")
    result = await pipeline.execute(data)
    
    logger.info("Pipeline parcial completado")
    return result

async def run_continuous_pipeline(interval: int = 3600, iterations: int = -1) -> None:
    """
    Ejecutar pipeline de forma continua con intervalos.
    
    Args:
        interval: Intervalo entre ejecuciones en segundos
        iterations: Número de iteraciones (-1 para infinito)
    """
    # Base de datos para persistencia
    db = TranscendentalDatabase()
    
    # Contador de iteraciones
    iteration = 0
    last_result = {}
    
    while iterations == -1 or iteration < iterations:
        iteration += 1
        logger.info(f"Iniciando iteración #{iteration}")
        
        try:
            # Ejecutar pipeline completo
            start_time = time.time()
            result = await run_full_pipeline(last_result)
            execution_time = time.time() - start_time
            
            # Guardar resultado
            await db.store("pipeline_results", f"iteration_{iteration}", result)
            
            # Actualizar último resultado para la próxima iteración
            last_result = result
            
            logger.info(f"Iteración #{iteration} completada en {execution_time:.2f}s")
            
            # Generar informe de ejecución
            report = {
                "iteration": iteration,
                "timestamp": time.time(),
                "execution_time": execution_time,
                "pipeline_stats": result.get("execution_report", {}).get("metrics", {}),
                "signals_count": len(result.get("signals", {})),
                "decisions_count": len(result.get("decisions", [])),
                "orders_count": len(result.get("execution_results", []))
            }
            
            # Guardar informe
            await db.store("pipeline_reports", f"report_{iteration}", report)
            
            # Esperar hasta la próxima ejecución
            if iterations == -1 or iteration < iterations:
                logger.info(f"Esperando {interval}s hasta la próxima ejecución...")
                await asyncio.sleep(interval)
        
        except Exception as e:
            logger.error(f"Error en iteración #{iteration}: {str(e)}")
            
            # Esperar un intervalo más corto en caso de error
            await asyncio.sleep(min(300, interval / 2))

def save_result_to_file(result: Dict[str, Any], filename: str) -> None:
    """
    Guardar resultado del pipeline en archivo JSON.
    
    Args:
        result: Resultado del pipeline
        filename: Nombre del archivo
    """
    try:
        # Eliminar objetos no serializables
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items() if k != "context" and k != "db"}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            else:
                return obj
        
        clean_result = clean_for_json(result)
        
        with open(filename, 'w') as f:
            json.dump(clean_result, f, indent=2)
        
        logger.info(f"Resultado guardado en {filename}")
    except Exception as e:
        logger.error(f"Error al guardar resultado: {str(e)}")

async def main():
    """Función principal."""
    # Configurar logging
    setup_logging(logging.INFO)
    
    # Analizar argumentos
    parser = argparse.ArgumentParser(description="Ejecutar Pipeline Genesis")
    parser.add_argument("--mode", choices=["full", "partial", "continuous"], default="full",
                        help="Modo de ejecución del pipeline")
    parser.add_argument("--stages", nargs="+", 
                        help="Etapas a ejecutar en modo parcial")
    parser.add_argument("--interval", type=int, default=3600,
                        help="Intervalo en segundos para modo continuo")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Número de iteraciones (-1 para infinito)")
    parser.add_argument("--output", type=str, default="pipeline_result.json",
                        help="Archivo de salida para el resultado")
    
    args = parser.parse_args()
    
    if args.mode == "full":
        # Ejecutar pipeline completo
        result = await run_full_pipeline()
        save_result_to_file(result, args.output)
    
    elif args.mode == "partial":
        if not args.stages:
            logger.error("Debe especificar etapas para el modo parcial")
            return
        
        # Ejecutar pipeline parcial
        result = await run_partial_pipeline(args.stages)
        save_result_to_file(result, args.output)
    
    elif args.mode == "continuous":
        # Ejecutar pipeline continuo
        await run_continuous_pipeline(args.interval, args.iterations)

if __name__ == "__main__":
    asyncio.run(main())