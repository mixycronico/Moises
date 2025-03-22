"""
Ejemplo de implementación del sistema híbrido API + WebSocket para Genesis.

Este script muestra cómo utilizar el enfoque híbrido para crear componentes
que se comunican tanto por API directa como por eventos WebSocket.
"""

import asyncio
import logging
import time
import random
import json
import sys
import os
from typing import Dict, Any, Optional, List

# Asegurar que el directorio raíz está en el path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from genesis.core.genesis_hybrid import ComponentAPI, GenesisHybridCoordinator

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("genesis.example")

class DataProcessor(ComponentAPI):
    """
    Componente que procesa datos y emite resultados.
    
    Este componente acepta solicitudes de procesamiento a través de la API
    y emite eventos de notificación a través de WebSocket.
    """
    def __init__(self, id: str):
        super().__init__(id)
        self.processed_count = 0
        self.processing_times = []
    
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """Procesa solicitudes directas."""
        self.metrics["requests_processed"] += 1
        
        if request_type == "process_data":
            # Simular procesamiento
            start_time = time.time()
            value = data.get("value", 0)
            
            # Simular tiempo de procesamiento variable
            processing_time = random.uniform(0.1, 0.5)
            await asyncio.sleep(processing_time)
            
            # Realizar el procesamiento
            processed_value = value * 2  # Transformación simple
            self.processed_count += 1
            self.processing_times.append(processing_time)
            
            logger.info(f"[{self.id}] Procesado valor {value} -> {processed_value} en {processing_time:.2f}s")
            
            return {
                "original_value": value,
                "processed_value": processed_value,
                "processing_time": processing_time,
                "timestamp": time.time()
            }
        
        elif request_type == "get_stats":
            # Retornar estadísticas
            return {
                "processed_count": self.processed_count,
                "avg_processing_time": sum(self.processing_times) / max(len(self.processing_times), 1),
                "metrics": self.metrics
            }
        
        return None
    
    async def on_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Maneja eventos recibidos por WebSocket."""
        await super().on_event(event_type, data, source)
        
        # Manejar tipos específicos de eventos
        if event_type == "data_available":
            logger.info(f"[{self.id}] Datos disponibles notificados por {source}")
            
        elif event_type == "system.heartbeat":
            # Solo log en nivel debug para heartbeats
            logger.debug(f"[{self.id}] Heartbeat recibido")

class AlertMonitor(ComponentAPI):
    """
    Componente que monitorea eventos y genera alertas.
    
    Este componente escucha eventos de otros componentes y genera alertas
    basadas en umbrales definidos.
    """
    def __init__(self, id: str, coordinator: GenesisHybridCoordinator):
        super().__init__(id)
        self.coordinator = coordinator
        self.alert_count = 0
        self.thresholds = {
            "value": 50,  # Umbral para valores
            "time": 0.4   # Umbral para tiempo de procesamiento
        }
    
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """Procesa solicitudes directas."""
        self.metrics["requests_processed"] += 1
        
        if request_type == "check_alert":
            # Verificar si un valor supera el umbral
            value = data.get("value", 0)
            is_alert = value > self.thresholds["value"]
            
            if is_alert:
                self.alert_count += 1
                
            return {
                "is_alert": is_alert,
                "threshold": self.thresholds["value"],
                "alert_count": self.alert_count
            }
        
        elif request_type == "get_thresholds":
            return self.thresholds
        
        elif request_type == "set_threshold":
            # Actualizar un umbral
            threshold_type = data.get("type")
            value = data.get("value")
            
            if threshold_type and value is not None:
                self.thresholds[threshold_type] = value
                logger.info(f"[{self.id}] Umbral {threshold_type} actualizado a {value}")
                return {"success": True}
            
            return {"success": False, "error": "Parámetros incompletos"}
        
        return None
    
    async def on_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Maneja eventos recibidos por WebSocket."""
        await super().on_event(event_type, data, source)
        
        # Verificar eventos de resultado de procesamiento
        if event_type == "processing_result":
            processed_value = data.get("processed_value", 0)
            processing_time = data.get("processing_time", 0)
            
            # Verificar umbrales
            alerts = []
            
            if processed_value > self.thresholds["value"]:
                alerts.append({
                    "type": "high_value",
                    "message": f"Valor procesado {processed_value} excede umbral {self.thresholds['value']}",
                    "severity": "warning"
                })
            
            if processing_time > self.thresholds["time"]:
                alerts.append({
                    "type": "slow_processing",
                    "message": f"Tiempo de procesamiento {processing_time:.2f}s excede umbral {self.thresholds['time']}s",
                    "severity": "info"
                })
            
            # Emitir alertas si es necesario
            if alerts:
                self.alert_count += len(alerts)
                logger.warning(f"[{self.id}] Generadas {len(alerts)} alertas para resultado de {source}")
                
                # Broadcast de alerta vía WebSocket
                await self.coordinator.broadcast_event(
                    "alert_generated",
                    {
                        "alerts": alerts,
                        "source_event": event_type,
                        "source_component": source,
                        "timestamp": time.time()
                    },
                    self.id  # Este componente es la fuente de la alerta
                )

class DataGenerator(ComponentAPI):
    """
    Componente que genera datos para procesamiento.
    
    Este componente genera datos periódicamente y los envía a
    los procesadores a través de la API y notifica por WebSocket.
    """
    def __init__(self, id: str, coordinator: GenesisHybridCoordinator):
        super().__init__(id)
        self.coordinator = coordinator
        self.generation_interval = 1.0  # segundos
        self.generation_task = None
        self.generated_count = 0
        self.target_processors = []
    
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """Procesa solicitudes directas."""
        self.metrics["requests_processed"] += 1
        
        if request_type == "set_interval":
            # Actualizar intervalo de generación
            interval = data.get("interval")
            if interval is not None and interval > 0:
                self.generation_interval = interval
                logger.info(f"[{self.id}] Intervalo actualizado a {interval}s")
                return {"success": True}
            return {"success": False, "error": "Intervalo inválido"}
        
        elif request_type == "set_targets":
            # Configurar procesadores destino
            targets = data.get("targets", [])
            self.target_processors = targets
            logger.info(f"[{self.id}] Procesadores destino actualizados: {targets}")
            return {"success": True, "targets": targets}
        
        elif request_type == "get_stats":
            return {
                "generated_count": self.generated_count,
                "interval": self.generation_interval,
                "targets": self.target_processors
            }
        
        return None
    
    async def _generate_data_loop(self):
        """Ciclo de generación de datos periódica."""
        logger.info(f"[{self.id}] Iniciando generación de datos cada {self.generation_interval}s")
        
        while True:
            try:
                # Generar datos
                value = random.randint(10, 100)
                self.generated_count += 1
                
                logger.info(f"[{self.id}] Generado valor: {value}")
                
                # Notificar disponibilidad de datos por WebSocket
                await self.coordinator.broadcast_event(
                    "data_available",
                    {
                        "value": value, 
                        "generator": self.id,
                        "timestamp": time.time(),
                        "sequence": self.generated_count
                    },
                    self.id
                )
                
                # Enviar a procesadores configurados vía API
                for processor_id in self.target_processors:
                    try:
                        result = await self.coordinator.request(
                            processor_id,
                            "process_data",
                            {"value": value, "generator": self.id, "sequence": self.generated_count},
                            self.id
                        )
                        
                        if result:
                            # Broadcast de resultado vía WebSocket
                            await self.coordinator.broadcast_event(
                                "processing_result",
                                result,
                                processor_id
                            )
                    except Exception as e:
                        logger.error(f"[{self.id}] Error enviando a {processor_id}: {e}")
                
                # Esperar hasta el próximo intervalo
                await asyncio.sleep(self.generation_interval)
                
            except asyncio.CancelledError:
                logger.info(f"[{self.id}] Generación de datos detenida")
                break
            except Exception as e:
                logger.error(f"[{self.id}] Error en generación: {e}")
                await asyncio.sleep(1)  # Esperar antes de reintentar
    
    async def start(self):
        """Iniciar el componente y la generación de datos."""
        await super().start()
        
        # Iniciar tarea de generación
        self.generation_task = asyncio.create_task(self._generate_data_loop())
    
    async def stop(self):
        """Detener el componente y la generación de datos."""
        # Cancelar tarea de generación
        if self.generation_task and not self.generation_task.done():
            self.generation_task.cancel()
            try:
                await self.generation_task
            except asyncio.CancelledError:
                pass
        
        await super().stop()

class DashboardComponent(ComponentAPI):
    """
    Componente que recolecta datos del sistema para un dashboard.
    
    Este componente escucha todos los eventos y almacena métricas
    para proporcionar una visión general del sistema.
    """
    def __init__(self, id: str, coordinator: GenesisHybridCoordinator):
        super().__init__(id)
        self.coordinator = coordinator
        self.system_metrics = {
            "processed_values": [],
            "processing_times": [],
            "alerts": [],
            "component_status": {}
        }
    
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """Procesa solicitudes directas."""
        self.metrics["requests_processed"] += 1
        
        if request_type == "get_dashboard_data":
            # Recolectar métricas adicionales bajo demanda
            component_status = {}
            
            # Solicitar estadísticas de todos los componentes
            for comp_id, comp in self.coordinator.components.items():
                if comp_id != self.id:  # Evitar recursión
                    try:
                        # Intentar obtener estadísticas si el componente las expone
                        stats = await self.coordinator.request(
                            comp_id, "get_stats", {}, self.id, timeout=1.0
                        )
                        if stats:
                            component_status[comp_id] = stats
                    except:
                        # Ignorar errores, algunos componentes pueden no tener get_stats
                        pass
            
            # Actualizar métricas del sistema
            self.system_metrics["component_status"] = component_status
            
            # Retornar todas las métricas
            return {
                "metrics": self.system_metrics,
                "timestamp": time.time(),
                "events_received": len(self.events_received)
            }
        
        return None
    
    async def on_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Maneja eventos recibidos por WebSocket."""
        await super().on_event(event_type, data, source)
        
        # Almacenar métricas según tipo de evento
        if event_type == "processing_result":
            # Guardar valores procesados y tiempos
            processed_value = data.get("processed_value")
            processing_time = data.get("processing_time")
            
            if processed_value is not None:
                self.system_metrics["processed_values"].append({
                    "value": processed_value,
                    "timestamp": time.time(),
                    "processor": source
                })
                
                # Mantener solo los últimos 100 valores
                if len(self.system_metrics["processed_values"]) > 100:
                    self.system_metrics["processed_values"].pop(0)
            
            if processing_time is not None:
                self.system_metrics["processing_times"].append({
                    "time": processing_time,
                    "timestamp": time.time(),
                    "processor": source
                })
                
                # Mantener solo los últimos 100 tiempos
                if len(self.system_metrics["processing_times"]) > 100:
                    self.system_metrics["processing_times"].pop(0)
        
        elif event_type == "alert_generated":
            # Guardar alertas
            alerts = data.get("alerts", [])
            for alert in alerts:
                self.system_metrics["alerts"].append({
                    "alert": alert,
                    "timestamp": time.time(),
                    "source": source
                })
                
                # Mantener solo las últimas 50 alertas
                if len(self.system_metrics["alerts"]) > 50:
                    self.system_metrics["alerts"].pop(0)

async def setup_and_run():
    """Configurar y ejecutar el sistema híbrido de ejemplo."""
    logger.info("Iniciando sistema híbrido de ejemplo")
    
    # Crear coordinador
    coordinator = GenesisHybridCoordinator(
        host="localhost",
        port=8765,
        monitor_interval=5.0
    )
    
    # Crear componentes
    data_generator = DataGenerator("generator", coordinator)
    processor1 = DataProcessor("processor1")
    processor2 = DataProcessor("processor2")
    alert_monitor = AlertMonitor("alert_monitor", coordinator)
    dashboard = DashboardComponent("dashboard", coordinator)
    
    # Registrar componentes en el coordinador
    coordinator.register_component("generator", data_generator)
    coordinator.register_component("processor1", processor1)
    coordinator.register_component("processor2", processor2)
    coordinator.register_component("alert_monitor", alert_monitor)
    coordinator.register_component("dashboard", dashboard)
    
    # Configurar procesadores destino para el generador
    await coordinator.request(
        "generator",
        "set_targets",
        {"targets": ["processor1", "processor2"]},
        "system"
    )
    
    # Iniciar el coordinador
    await coordinator.start()
    
    try:
        # Dejar que el sistema corra
        logger.info("Sistema híbrido ejecutándose. Presiona Ctrl+C para detener.")
        
        # Mantener el sistema corriendo
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Interrupción de teclado recibida, deteniendo sistema...")
    except Exception as e:
        logger.error(f"Error durante la ejecución: {e}")
    finally:
        # Detener componentes y coordinador
        await coordinator.stop()
        logger.info("Sistema híbrido detenido")

if __name__ == "__main__":
    try:
        asyncio.run(setup_and_run())
    except KeyboardInterrupt:
        logger.info("Proceso terminado por usuario")