"""
Gestor principal del Pipeline Transcendental de Genesis.

Este módulo coordina todas las etapas del pipeline, desde la adquisición
de datos hasta la ejecución de operaciones y distribución de ganancias,
con mecanismos de resiliencia extrema.
"""
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from datetime import datetime

from genesis.base import GenesisComponent, GenesisSingleton, validate_mode
from genesis.db.transcendental_database import TranscendentalDatabase

# Configuración de logging
logger = logging.getLogger("genesis.pipeline.manager")

class PipelineStage:
    """Etapa del pipeline con capacidades de procesamiento trascendental."""
    
    def __init__(self, stage_id: str, stage_name: str, stage_order: int):
        """
        Inicializar etapa del pipeline.
        
        Args:
            stage_id: Identificador único de la etapa
            stage_name: Nombre descriptivo de la etapa
            stage_order: Orden de ejecución (menor = primero)
        """
        self.stage_id = stage_id
        self.stage_name = stage_name
        self.stage_order = stage_order
        self.creation_time = time.time()
        self.last_execution = 0
        self.execution_count = 0
        self.error_count = 0
        self.total_execution_time = 0.0
        self.processor: Optional[Callable] = None
        
    async def process(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesar datos en esta etapa.
        
        Args:
            data: Datos a procesar
            context: Contexto de ejecución
            
        Returns:
            Datos procesados
        """
        if not self.processor:
            logger.warning(f"Etapa {self.stage_id} no tiene procesador configurado")
            return data
            
        start_time = time.time()
        self.last_execution = start_time
        self.execution_count += 1
        
        try:
            result = await self.processor(data, context)
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            
            # Registrar métricas
            avg_time = self.total_execution_time / self.execution_count
            logger.debug(f"Etapa {self.stage_id} completada en {execution_time:.4f}s (promedio: {avg_time:.4f}s)")
            
            return result
        except Exception as e:
            self.error_count += 1
            error_rate = self.error_count / self.execution_count
            logger.error(f"Error en etapa {self.stage_id}: {str(e)}, tasa de error: {error_rate:.2%}")
            # En caso de error, continuamos con los datos originales para mantener resilencia
            return data
    
    def register_processor(self, processor_func: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]) -> None:
        """
        Registrar función de procesamiento para esta etapa.
        
        Args:
            processor_func: Función que procesará los datos
        """
        self.processor = processor_func
        logger.info(f"Procesador registrado para etapa {self.stage_id}: {processor_func.__name__}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de esta etapa.
        
        Returns:
            Diccionario con estadísticas
        """
        uptime = time.time() - self.creation_time
        
        return {
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "stage_order": self.stage_order,
            "uptime": uptime,
            "last_execution": self.last_execution,
            "execution_count": self.execution_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.execution_count),
            "avg_execution_time": self.total_execution_time / max(1, self.execution_count),
            "total_execution_time": self.total_execution_time,
            "has_processor": self.processor is not None
        }

class TranscendentalPipeline(GenesisComponent, GenesisSingleton):
    """
    Gestor del pipeline trascendental de Genesis con capacidades de 
    resiliencia extrema y procesamiento multi-dimensional.
    """
    
    def __init__(self, mode: str = "SINGULARITY_V4"):
        """
        Inicializar pipeline trascendental.
        
        Args:
            mode: Modo trascendental
        """
        super().__init__("trascendental_pipeline", mode)
        self.stages: Dict[str, PipelineStage] = {}
        self.stages_order: List[str] = []
        self.db = TranscendentalDatabase()
        self.context: Dict[str, Any] = {
            "start_time": time.time(),
            "last_execution": 0,
            "pipeline_runs": 0,
            "current_run_metrics": {},
            "accumulated_metrics": {},
            "is_running": False
        }
        self.checkpoint_interval = 10  # Guardar estado cada 10 ejecuciones
        logger.info(f"Pipeline Trascendental inicializado en modo {mode}")
    
    def register_stage(self, stage_id: str, stage_name: str, stage_order: int) -> PipelineStage:
        """
        Registrar una nueva etapa en el pipeline.
        
        Args:
            stage_id: Identificador único de la etapa
            stage_name: Nombre descriptivo de la etapa
            stage_order: Orden de ejecución (menor = primero)
            
        Returns:
            Instancia de la etapa creada
        """
        if stage_id in self.stages:
            logger.warning(f"La etapa {stage_id} ya está registrada, actualizando")
        
        stage = PipelineStage(stage_id, stage_name, stage_order)
        self.stages[stage_id] = stage
        
        # Reordenar etapas según stage_order
        self._reorder_stages()
        
        logger.info(f"Etapa {stage_id} ({stage_name}) registrada con orden {stage_order}")
        return stage
    
    def _reorder_stages(self) -> None:
        """Reordenar etapas según su stage_order."""
        # Convertir a lista de tuplas (stage_id, stage_order)
        stage_items = [(stage_id, stage.stage_order) for stage_id, stage in self.stages.items()]
        # Ordenar por stage_order
        stage_items.sort(key=lambda x: x[1])
        # Extraer solo los stage_id en el orden correcto
        self.stages_order = [item[0] for item in stage_items]
    
    async def restore_from_checkpoint(self) -> bool:
        """
        Restaurar estado desde el último checkpoint.
        
        Returns:
            True si se restauró correctamente
        """
        try:
            pipeline_state = await self.db.retrieve("pipeline_state", "latest")
            if not pipeline_state:
                logger.warning("No hay checkpoint disponible para restaurar")
                return False
            
            # Restaurar contexto
            for key, value in pipeline_state.get("context", {}).items():
                if key in self.context:
                    self.context[key] = value
            
            # Restaurar métricas acumuladas
            self.context["accumulated_metrics"] = pipeline_state.get("accumulated_metrics", {})
            
            logger.info(f"Pipeline restaurado desde checkpoint (run #{self.context['pipeline_runs']})")
            return True
        except Exception as e:
            logger.error(f"Error al restaurar desde checkpoint: {str(e)}")
            return False
    
    async def save_checkpoint(self) -> bool:
        """
        Guardar estado actual como checkpoint.
        
        Returns:
            True si se guardó correctamente
        """
        try:
            pipeline_state = {
                "timestamp": time.time(),
                "context": self.context,
                "accumulated_metrics": self.context.get("accumulated_metrics", {}),
                "stages": {stage_id: stage.get_stats() for stage_id, stage in self.stages.items()}
            }
            
            # Guardar en base de datos trascendental
            await self.db.store("pipeline_state", "latest", pipeline_state)
            logger.debug(f"Checkpoint guardado (run #{self.context['pipeline_runs']})")
            return True
        except Exception as e:
            logger.error(f"Error al guardar checkpoint: {str(e)}")
            return False
    
    async def execute(self, initial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecutar pipeline completo con el conjunto de datos inicial.
        
        Args:
            initial_data: Datos iniciales para el pipeline
            
        Returns:
            Resultado final del pipeline
        """
        if self.context["is_running"]:
            logger.warning("Pipeline ya está en ejecución, esperando...")
            # Esperar a que termine la ejecución actual (máximo 60 segundos)
            for _ in range(60):
                await asyncio.sleep(1)
                if not self.context["is_running"]:
                    break
            else:
                logger.error("Timeout esperando fin de ejecución anterior")
                return {"error": "Timeout waiting for previous execution", "status": "error"}
        
        self.context["is_running"] = True
        self.context["last_execution"] = time.time()
        self.context["pipeline_runs"] += 1
        self.context["current_run_metrics"] = {}
        run_id = f"run_{int(time.time())}"
        
        logger.info(f"Iniciando pipeline run #{self.context['pipeline_runs']} ({run_id})")
        
        # Datos de trabajo que van pasando por cada etapa
        working_data = initial_data.copy()
        
        # Contexto para esta ejecución específica
        execution_context = {
            "run_id": run_id,
            "start_time": time.time(),
            "run_number": self.context["pipeline_runs"],
            "metrics": {},
            "errors": []
        }
        
        try:
            # Ejecutar cada etapa en orden
            for stage_id in self.stages_order:
                stage = self.stages[stage_id]
                logger.debug(f"Ejecutando etapa: {stage.stage_name} ({stage_id})")
                
                # Procesar datos en esta etapa
                working_data = await stage.process(working_data, execution_context)
                
                # Actualizar métricas
                execution_context["metrics"][stage_id] = {
                    "execution_time": stage.total_execution_time / max(1, stage.execution_count),
                    "error_rate": stage.error_count / max(1, stage.execution_count)
                }
            
            # Guardar métricas de esta ejecución
            execution_context["end_time"] = time.time()
            execution_context["duration"] = execution_context["end_time"] - execution_context["start_time"]
            self.context["current_run_metrics"] = execution_context["metrics"]
            
            # Actualizar métricas acumuladas
            self._update_accumulated_metrics(execution_context["metrics"])
            
            # Guardar checkpoint periódicamente
            if self.context["pipeline_runs"] % self.checkpoint_interval == 0:
                await self.save_checkpoint()
            
            # Generar reporte de esta ejecución
            report = self._generate_execution_report(execution_context, working_data)
            working_data["execution_report"] = report
            
            logger.info(f"Pipeline completado: run #{self.context['pipeline_runs']} en {execution_context['duration']:.2f}s")
            return working_data
        
        except Exception as e:
            logger.error(f"Error en pipeline: {str(e)}")
            execution_context["errors"].append(str(e))
            execution_context["end_time"] = time.time()
            execution_context["duration"] = execution_context["end_time"] - execution_context["start_time"]
            
            # Intentar guardar checkpoint incluso en caso de error
            await self.save_checkpoint()
            
            return {
                "error": str(e),
                "stage": stage_id if 'stage_id' in locals() else "unknown",
                "execution_context": execution_context,
                "status": "error"
            }
        finally:
            self.context["is_running"] = False
    
    def _update_accumulated_metrics(self, current_metrics: Dict[str, Any]) -> None:
        """
        Actualizar métricas acumuladas con los resultados de la ejecución actual.
        
        Args:
            current_metrics: Métricas de la ejecución actual
        """
        if "accumulated_metrics" not in self.context:
            self.context["accumulated_metrics"] = {}
        
        for stage_id, metrics in current_metrics.items():
            if stage_id not in self.context["accumulated_metrics"]:
                self.context["accumulated_metrics"][stage_id] = {
                    "total_time": 0,
                    "count": 0,
                    "error_count": 0
                }
            
            # Actualizar acumulados
            acc = self.context["accumulated_metrics"][stage_id]
            acc["total_time"] += metrics.get("execution_time", 0)
            acc["count"] += 1
            acc["error_count"] += int(metrics.get("error_rate", 0) > 0)
            
            # Calcular promedios
            acc["avg_time"] = acc["total_time"] / acc["count"]
            acc["error_rate"] = acc["error_count"] / acc["count"]
    
    def _generate_execution_report(self, context: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generar reporte de ejecución del pipeline.
        
        Args:
            context: Contexto de ejecución
            results: Resultados del pipeline
            
        Returns:
            Reporte de ejecución
        """
        now = datetime.now()
        
        report = {
            "pipeline_run_id": context["run_id"],
            "run_number": context["run_number"],
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": context["duration"],
            "status": "error" if context.get("errors") else "success",
            "stages_executed": len(self.stages_order),
            "metrics": context["metrics"],
            "errors": context.get("errors", []),
            "summary": {
                "total_stages": len(self.stages),
                "start_time": datetime.fromtimestamp(context["start_time"]).strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": datetime.fromtimestamp(context["end_time"]).strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # Agregar métricas específicas según los resultados
        if "profit" in results:
            report["financial_metrics"] = {
                "profit": results["profit"],
                "profit_percentage": results.get("profit_percentage", 0),
                "balance": results.get("balance", 0)
            }
        
        return report
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas completas del pipeline.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = super().get_stats()
        
        # Agregar estadísticas específicas del pipeline
        pipeline_stats = {
            "total_stages": len(self.stages),
            "stage_order": self.stages_order,
            "pipeline_runs": self.context["pipeline_runs"],
            "last_execution": self.context["last_execution"],
            "uptime": time.time() - self.context["start_time"],
            "is_running": self.context["is_running"],
            "stages": {stage_id: stage.get_stats() for stage_id, stage in self.stages.items()},
            "accumulated_metrics": self.context.get("accumulated_metrics", {})
        }
        
        stats.update(pipeline_stats)
        return stats
    
    async def initialize(self) -> bool:
        """
        Inicializar pipeline completo, incluyendo todas las etapas.
        
        Returns:
            True si se inicializó correctamente
        """
        logger.info(f"Inicializando Pipeline Trascendental en modo {self.mode}")
        
        try:
            # Restaurar estado desde checkpoint si existe
            restored = await self.restore_from_checkpoint()
            if not restored:
                logger.info("No se encontró checkpoint, inicializando desde cero")
            
            # Registrar etapas estándar si no existen
            if len(self.stages) == 0:
                self._register_standard_stages()
            
            logger.info(f"Pipeline inicializado con {len(self.stages)} etapas")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar pipeline: {str(e)}")
            return False
    
    def _register_standard_stages(self) -> None:
        """Registrar etapas estándar del pipeline."""
        standard_stages = [
            # Formato: (id, nombre, orden)
            ("data_acquisition", "Adquisición de Datos", 10),
            ("data_validation", "Validación de Datos", 20),
            ("data_preprocessing", "Preprocesamiento", 30),
            ("feature_engineering", "Ingeniería de Características", 40),
            ("model_prediction", "Predicción del Modelo", 50),
            ("signal_generation", "Generación de Señales", 60),
            ("risk_assessment", "Evaluación de Riesgo", 70),
            ("order_creation", "Creación de Órdenes", 80),
            ("execution", "Ejecución de Órdenes", 90),
            ("performance_tracking", "Seguimiento de Rendimiento", 100),
            ("profit_distribution", "Distribución de Ganancias", 110),
            ("capital_management", "Gestión de Capital", 120),
            ("reporting", "Generación de Informes", 130)
        ]
        
        for stage_id, stage_name, stage_order in standard_stages:
            self.register_stage(stage_id, stage_name, stage_order)
        
        logger.info(f"Registradas {len(standard_stages)} etapas estándar")

# Instancia global (singleton)
pipeline = TranscendentalPipeline()