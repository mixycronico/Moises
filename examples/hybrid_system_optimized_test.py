"""
Prueba del sistema híbrido optimizado Genesis con API, WebSocket local y externo.

Este script prueba la implementación optimizada del sistema híbrido Genesis
que combina API REST con WebSockets locales y externos para evitar deadlocks.
"""

import asyncio
import logging
import time
import random
import json
import sys
import os
from typing import Dict, Any, Optional, List
import websockets

# Asegurar que el directorio raíz está en el path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar la implementación optimizada del sistema híbrido
"""
NOTA: Este script está diseñado para usar la implementación compacta del sistema híbrido 
que se proporcionó en el archivo de texto. Deberás crear ese archivo en:
genesis/core/genesis_hybrid_optimized.py
"""
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from genesis.core.genesis_hybrid_optimized import ComponentAPI, GenesisHybridCoordinator

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("genesis.test_optimized")

class DataGeneratorComponent(ComponentAPI):
    """
    Componente que genera datos y los comparte por eventos locales y eventos externos.
    """
    def __init__(self, id: str, coordinator: GenesisHybridCoordinator):
        super().__init__(id)
        self.coordinator = coordinator
        self.generation_task = None
        self.counter = 0
        self.interval = 1.0
        self.running = False
    
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """Procesar solicitudes API."""
        if request_type == "set_interval":
            self.interval = data.get("interval", 1.0)
            return {"success": True, "new_interval": self.interval}
        
        elif request_type == "get_status":
            return {
                "counter": self.counter,
                "interval": self.interval,
                "running": self.running
            }
        
        elif request_type == "start_generation":
            if not self.running:
                self.running = True
                self.generation_task = asyncio.create_task(self._generate_data())
                return {"success": True, "message": "Generación iniciada"}
            return {"success": False, "message": "Ya está en ejecución"}
        
        elif request_type == "stop_generation":
            if self.running:
                self.running = False
                if self.generation_task:
                    self.generation_task.cancel()
                return {"success": True, "message": "Generación detenida"}
            return {"success": False, "message": "No está en ejecución"}
        
        return None
    
    async def _generate_data(self):
        """Genera datos periódicamente."""
        logger.info(f"[{self.id}] Iniciando generación de datos cada {self.interval}s")
        
        while self.running:
            try:
                # Generar datos
                value = random.randint(1, 100)
                self.counter += 1
                timestamp = time.time()
                
                # Datos a compartir
                data = {
                    "value": value,
                    "counter": self.counter,
                    "timestamp": timestamp,
                    "source": self.id
                }
                
                logger.info(f"[{self.id}] Generado dato #{self.counter}: {value}")
                
                # Compartir localmente (comunicación interna rápida)
                await self.coordinator.emit_local(
                    "data_generated",
                    data,
                    self.id
                )
                
                # Compartir externamente (para clientes conectados por red)
                await self.coordinator.emit_external(
                    "data_generated_external",
                    data,
                    self.id
                )
                
                # Esperar el intervalo configurado
                await asyncio.sleep(self.interval)
                
            except asyncio.CancelledError:
                logger.info(f"[{self.id}] Generación cancelada")
                break
            except Exception as e:
                logger.error(f"[{self.id}] Error en generación: {e}")
                await asyncio.sleep(1.0)  # Pausa antes de reintentar
        
        logger.info(f"[{self.id}] Generación finalizada")

class ProcessorComponent(ComponentAPI):
    """
    Componente que procesa datos recibidos por eventos locales.
    """
    def __init__(self, id: str, coordinator: GenesisHybridCoordinator):
        super().__init__(id)
        self.coordinator = coordinator
        self.processed_data = []
        self.total_processed = 0
        self.threshold = 50  # Umbral para alertas
    
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """Procesar solicitudes API."""
        if request_type == "get_processed":
            return {
                "total_processed": self.total_processed,
                "recent_data": self.processed_data[-10:] if self.processed_data else []
            }
        
        elif request_type == "set_threshold":
            self.threshold = data.get("threshold", 50)
            return {"success": True, "new_threshold": self.threshold}
        
        return None
    
    async def on_local_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Procesar eventos locales."""
        await super().on_local_event(event_type, data, source)
        
        if event_type == "data_generated":
            # Procesar datos
            value = data.get("value", 0)
            counter = data.get("counter", 0)
            timestamp = data.get("timestamp", time.time())
            
            # Procesamiento simple: multiplicar por 2
            processed_value = value * 2
            
            self.total_processed += 1
            result = {
                "original": value,
                "processed": processed_value,
                "counter": counter,
                "timestamp": timestamp,
                "processor": self.id
            }
            
            self.processed_data.append(result)
            
            # Mantener solo los últimos 100 resultados
            if len(self.processed_data) > 100:
                self.processed_data = self.processed_data[-100:]
            
            logger.info(f"[{self.id}] Procesado #{counter}: {value} -> {processed_value}")
            
            # Emitir resultado del procesamiento
            await self.coordinator.emit_local(
                "processing_result",
                result,
                self.id
            )
            
            # Emitir alerta si el valor procesado supera el umbral
            if processed_value > self.threshold:
                alert_data = {
                    "original_value": value,
                    "processed_value": processed_value,
                    "threshold": self.threshold,
                    "timestamp": time.time(),
                    "processor": self.id,
                    "level": "warning"
                }
                
                logger.warning(f"[{self.id}] ALERTA: Valor {processed_value} supera umbral {self.threshold}")
                
                # Emitir alerta local
                await self.coordinator.emit_local(
                    "threshold_alert",
                    alert_data,
                    self.id
                )
                
                # Emitir alerta externa
                await self.coordinator.emit_external(
                    "threshold_alert_external",
                    alert_data,
                    self.id
                )

class AnalyticsComponent(ComponentAPI):
    """
    Componente que analiza resultados de procesamiento y genera estadísticas.
    """
    def __init__(self, id: str, coordinator: GenesisHybridCoordinator):
        super().__init__(id)
        self.coordinator = coordinator
        self.value_sum = 0
        self.value_count = 0
        self.alerts_received = 0
        self.min_value = float('inf')
        self.max_value = float('-inf')
    
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """Procesar solicitudes API."""
        if request_type == "get_stats":
            avg = self.value_sum / max(self.value_count, 1)
            return {
                "count": self.value_count,
                "average": avg,
                "min": self.min_value if self.min_value != float('inf') else None,
                "max": self.max_value if self.max_value != float('-inf') else None,
                "alerts": self.alerts_received
            }
        
        return None
    
    async def on_local_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Procesar eventos locales."""
        await super().on_local_event(event_type, data, source)
        
        if event_type == "processing_result":
            # Actualizar estadísticas
            processed_value = data.get("processed", 0)
            
            self.value_sum += processed_value
            self.value_count += 1
            self.min_value = min(self.min_value, processed_value)
            self.max_value = max(self.max_value, processed_value)
            
            # Calcular promedio
            avg = self.value_sum / self.value_count
            
            # Log cada 5 valores o cuando hay uno extremo
            if self.value_count % 5 == 0 or processed_value in (self.min_value, self.max_value):
                logger.info(f"[{self.id}] Estadísticas: Avg={avg:.2f}, Min={self.min_value}, Max={self.max_value}, Count={self.value_count}")
        
        elif event_type == "threshold_alert":
            self.alerts_received += 1
            logger.warning(f"[{self.id}] Alerta #{self.alerts_received} recibida: Valor {data.get('processed_value')} superó umbral {data.get('threshold')}")

async def run_ws_client(host: str, port: int, component_id: str):
    """
    Ejecutar un cliente WebSocket para pruebas.
    
    Este cliente se conecta al WebSocket externo y registra los eventos recibidos.
    """
    uri = f"ws://{host}:{port}/ws?id={component_id}"
    events_received = []
    
    try:
        async with websockets.connect(uri) as ws:
            logger.info(f"Cliente WebSocket conectado como {component_id}")
            
            # Esperar y recibir eventos durante 15 segundos
            for _ in range(5):  # 5 iteraciones de 3 segundos
                try:
                    # Esperar un mensaje con timeout
                    message = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    data = json.loads(message)
                    
                    events_received.append(data)
                    logger.info(f"Cliente WebSocket recibió: {data.get('type')} de {data.get('source')}")
                    
                except asyncio.TimeoutError:
                    logger.debug("Timeout esperando mensaje en WebSocket")
                except Exception as e:
                    logger.error(f"Error recibiendo mensaje WebSocket: {e}")
    
    except Exception as e:
        logger.error(f"Error conectando WebSocket: {e}")
    
    return events_received

async def test_system():
    """
    Probar el sistema híbrido optimizado.
    
    Esta función configura los componentes, inicia el coordinador y
    ejecuta una secuencia de pruebas para verificar la comunicación
    a través de ambos canales (API y WebSocket).
    """
    # Parámetros de red
    host = "localhost"
    port = 8765
    
    # Crear coordinador
    coordinator = GenesisHybridCoordinator(host=host, port=port)
    
    # Crear componentes
    generator = DataGeneratorComponent("generator", coordinator)
    processor1 = ProcessorComponent("processor1", coordinator)
    processor2 = ProcessorComponent("processor2", coordinator)
    analytics = AnalyticsComponent("analytics", coordinator)
    
    # Registrar componentes
    coordinator.register_component("generator", generator)
    coordinator.register_component("processor1", processor1)
    coordinator.register_component("processor2", processor2)
    coordinator.register_component("analytics", analytics)
    
    # Iniciar coordinador (en una tarea separada)
    coordinator_task = asyncio.create_task(coordinator.start())
    
    try:
        # Esperar a que el servidor web esté listo
        await asyncio.sleep(1.0)
        
        # Iniciar cliente WebSocket en otra tarea
        ws_client_task = asyncio.create_task(
            run_ws_client(host, port, "external_client")
        )
        
        # Configurar procesadores con diferentes umbrales
        await coordinator.request(
            "processor1",
            "set_threshold",
            {"threshold": 80},
            "system"
        )
        
        await coordinator.request(
            "processor2",
            "set_threshold",
            {"threshold": 120},
            "system"
        )
        
        # Iniciar generación de datos
        await coordinator.request(
            "generator",
            "start_generation",
            {"interval": 0.5},  # Generar datos cada 0.5 segundos
            "system"
        )
        
        # Esperar y realizar consultas periódicas
        for i in range(5):
            await asyncio.sleep(2.0)  # Esperar 2 segundos entre consultas
            
            # Obtener estadísticas
            stats = await coordinator.request(
                "analytics",
                "get_stats",
                {},
                "system"
            )
            
            logger.info(f"Consulta {i+1}: Estadísticas: {stats}")
            
            # Consultar datos procesados de procesador1
            processed = await coordinator.request(
                "processor1",
                "get_processed",
                {},
                "system"
            )
            
            logger.info(f"Consulta {i+1}: Procesador1 ha procesado {processed.get('total_processed')} datos")
        
        # Detener generación de datos
        await coordinator.request(
            "generator",
            "stop_generation",
            {},
            "system"
        )
        
        # Esperar a que termine el cliente WebSocket
        events_received = await ws_client_task
        
        # Mostrar resumen de eventos WebSocket externos recibidos
        logger.info(f"Cliente WebSocket recibió {len(events_received)} eventos")
        event_types = {}
        for event in events_received:
            event_type = event.get('type', 'unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        logger.info(f"Tipos de eventos recibidos: {event_types}")
        
        # Obtener estadísticas finales
        final_stats = await coordinator.request(
            "analytics",
            "get_stats",
            {},
            "system"
        )
        
        logger.info(f"Estadísticas finales: {final_stats}")
        
    except Exception as e:
        logger.error(f"Error en prueba: {e}")
    finally:
        # Detener el coordinador
        coordinator.running = False
        
        # Cancelar tarea coordinador
        coordinator_task.cancel()
        try:
            await coordinator_task
        except asyncio.CancelledError:
            pass
        
        logger.info("Prueba completada")

if __name__ == "__main__":
    try:
        asyncio.run(test_system())
    except KeyboardInterrupt:
        print("Prueba interrumpida por usuario")
    except Exception as e:
        print(f"Error ejecutando prueba: {e}")