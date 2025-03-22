"""
Ejemplo de uso del adaptador para migrar componentes existentes al sistema híbrido.

Este script muestra cómo usar el adaptador para migrar componentes
existentes de Genesis al nuevo sistema híbrido sin modificarlos.
"""

import asyncio
import logging
import time
import random
import sys
import os
from typing import Dict, Any, List, Optional

# Asegurar que el directorio raíz está en el path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from genesis.core.component import Component
from genesis.core.component_adapter import HybridEngineAdapter

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("genesis.example")

class ExistingDataSource(Component):
    """
    Componente existente que actúa como fuente de datos.
    
    Este componente usa la interfaz original de Component de Genesis.
    """
    
    def __init__(self, id: str):
        super().__init__(id)
        self.data_counter = 0
        self.generation_task = None
        self.running = False
    
    async def start(self):
        """Iniciar el componente y la generación de datos."""
        logger.info(f"[{self.id}] Iniciando fuente de datos existente")
        self.running = True
        self.generation_task = asyncio.create_task(self._generate_data())
    
    async def stop(self):
        """Detener el componente."""
        logger.info(f"[{self.id}] Deteniendo fuente de datos existente")
        self.running = False
        
        if self.generation_task and not self.generation_task.done():
            self.generation_task.cancel()
            try:
                await self.generation_task
            except asyncio.CancelledError:
                pass
    
    async def _generate_data(self):
        """Generar datos periódicamente."""
        logger.info(f"[{self.id}] Iniciando generación de datos")
        
        while self.running:
            try:
                # Generar un valor aleatorio
                value = random.randint(1, 100)
                self.data_counter += 1
                
                # Emitir el valor como evento
                logger.info(f"[{self.id}] Generando dato #{self.data_counter}: {value}")
                
                # Usando la interfaz original de emit_event
                await self.emit_event("new_data", {
                    "value": value,
                    "counter": self.data_counter,
                    "timestamp": time.time()
                })
                
                # Esperar un intervalo aleatorio
                await asyncio.sleep(random.uniform(1.0, 3.0))
                
            except asyncio.CancelledError:
                logger.info(f"[{self.id}] Generación de datos cancelada")
                break
            except Exception as e:
                logger.error(f"[{self.id}] Error generando datos: {e}")
                await asyncio.sleep(1.0)
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """Manejar eventos según la interfaz original."""
        if event_type == "get_status":
            return {
                "data_counter": self.data_counter,
                "running": self.running
            }
        
        elif event_type == "set_running":
            state = data.get("state")
            if state is not None:
                self.running = state
                logger.info(f"[{self.id}] Estado de generación actualizado a {state}")
                return {"success": True}
            return {"success": False, "error": "Estado no especificado"}
        
        return None  # No hay respuesta inmediata para otros tipos de eventos

class ExistingDataProcessor(Component):
    """
    Componente existente que procesa datos.
    
    Este componente usa la interfaz original de Component de Genesis.
    """
    
    def __init__(self, id: str):
        super().__init__(id)
        self.processed_data = []
        self.total_processed = 0
    
    async def start(self):
        """Iniciar el componente."""
        logger.info(f"[{self.id}] Iniciando procesador de datos existente")
    
    async def stop(self):
        """Detener el componente."""
        logger.info(f"[{self.id}] Deteniendo procesador de datos existente")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """Manejar eventos según la interfaz original."""
        if event_type == "new_data":
            # Procesar datos recibidos
            value = data.get("value", 0)
            counter = data.get("counter", 0)
            
            # Realizar algún procesamiento
            processed_value = value * 2
            
            self.total_processed += 1
            self.processed_data.append({
                "original": value,
                "processed": processed_value,
                "counter": counter,
                "timestamp": time.time()
            })
            
            # Mantener solo los últimos 10 valores
            if len(self.processed_data) > 10:
                self.processed_data.pop(0)
            
            logger.info(f"[{self.id}] Procesado valor {value} -> {processed_value}, total: {self.total_processed}")
            
            # Emitir evento con el resultado del procesamiento
            await self.emit_event("processing_result", {
                "original": value,
                "processed": processed_value,
                "source": source,
                "processor": self.id
            })
            
            return None
        
        elif event_type == "get_results":
            return {
                "total_processed": self.total_processed,
                "recent_data": self.processed_data
            }
        
        return None

class ExistingAnalytics(Component):
    """
    Componente existente para análisis de datos.
    
    Este componente usa la interfaz original de Component de Genesis.
    """
    
    def __init__(self, id: str):
        super().__init__(id)
        self.value_sum = 0
        self.value_count = 0
        self.min_value = float('inf')
        self.max_value = float('-inf')
    
    async def start(self):
        """Iniciar el componente."""
        logger.info(f"[{self.id}] Iniciando componente de análisis existente")
    
    async def stop(self):
        """Detener el componente."""
        logger.info(f"[{self.id}] Deteniendo componente de análisis existente")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """Manejar eventos según la interfaz original."""
        if event_type == "processing_result":
            # Actualizar estadísticas con el resultado procesado
            processed_value = data.get("processed", 0)
            
            self.value_sum += processed_value
            self.value_count += 1
            self.min_value = min(self.min_value, processed_value)
            self.max_value = max(self.max_value, processed_value)
            
            # Calcular estadísticas
            avg = self.value_sum / self.value_count if self.value_count > 0 else 0
            
            logger.info(f"[{self.id}] Estadísticas actualizadas: "
                      f"Promedio={avg:.2f}, Min={self.min_value}, Max={self.max_value}")
            
            # Emitir evento con estadísticas actualizadas
            await self.emit_event("statistics_updated", {
                "average": avg,
                "min": self.min_value,
                "max": self.max_value,
                "count": self.value_count
            })
            
            return None
        
        elif event_type == "get_statistics":
            # Devolver estadísticas actuales
            avg = self.value_sum / self.value_count if self.value_count > 0 else 0
            
            return {
                "average": avg,
                "min": self.min_value,
                "max": self.max_value,
                "count": self.value_count
            }
        
        return None

async def run_example():
    """Ejecutar el ejemplo de adaptación de componentes existentes."""
    logger.info("Iniciando ejemplo de adaptación de componentes existentes")
    
    # Crear el adaptador del motor híbrido
    engine_adapter = HybridEngineAdapter(host="localhost", port=8765)
    
    # Crear componentes existentes
    data_source = ExistingDataSource("data_source")
    processor1 = ExistingDataProcessor("processor1")
    processor2 = ExistingDataProcessor("processor2")
    analytics = ExistingAnalytics("analytics")
    
    # Registrar componentes en el adaptador
    engine_adapter.register_component(data_source)
    engine_adapter.register_component(processor1)
    engine_adapter.register_component(processor2)
    engine_adapter.register_component(analytics)
    
    # Iniciar el sistema adaptado
    await engine_adapter.start()
    logger.info("Sistema adaptado iniciado")
    
    try:
        # Dejar que el sistema corra durante un tiempo
        logger.info("Sistema corriendo durante 30 segundos. Presiona Ctrl+C para detener antes.")
        
        # Simular algunas solicitudes de API
        for i in range(5):
            await asyncio.sleep(5)  # Esperar 5 segundos entre solicitudes
            
            # Solicitar estadísticas
            logger.info("Solicitando estadísticas...")
            stats = await engine_adapter.request(
                "analytics", 
                "get_statistics", 
                {}, 
                "system"
            )
            
            if stats:
                logger.info(f"Estadísticas actuales: {stats}")
            
            # Solicitar resultados recientes de un procesador
            logger.info("Solicitando resultados recientes...")
            results = await engine_adapter.request(
                "processor1", 
                "get_results", 
                {}, 
                "system"
            )
            
            if results:
                logger.info(f"Resultados recientes: {results}")
        
    except KeyboardInterrupt:
        logger.info("Interrupción de teclado recibida")
    except Exception as e:
        logger.error(f"Error durante la ejecución: {e}")
    finally:
        # Detener el sistema
        await engine_adapter.stop()
        logger.info("Sistema adaptado detenido")

if __name__ == "__main__":
    try:
        asyncio.run(run_example())
    except KeyboardInterrupt:
        print("Proceso terminado por usuario")
    except Exception as e:
        print(f"Error en ejemplo: {e}")