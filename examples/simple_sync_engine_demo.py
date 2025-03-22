"""
Demostración simple del motor síncrono para el sistema Genesis.

Este script es una prueba de concepto que demuestra cómo el motor síncrono
evita deadlocks y permite una comunicación fluida entre componentes.
"""

import os
import sys
import time
import logging
import threading
from typing import Dict, Any, Optional, List

# Agregar directorio raíz al path para importar módulos de Genesis
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from genesis.core.synchronous_engine import SynchronousEngine

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("genesis_demo")

# Componentes de ejemplo
class BasicComponent:
    """Componente básico que procesa eventos y genera respuestas."""
    
    def __init__(self, component_id: str):
        self.id = component_id
        self.events_received = []
        self.updates = 0
        self.started = False
        self.stopped = False
        logger.info(f"Componente {component_id} creado")
        
    def start(self):
        """Iniciar el componente."""
        self.started = True
        logger.info(f"Componente {self.id} iniciado")
        
    def stop(self):
        """Detener el componente."""
        self.stopped = True
        logger.info(f"Componente {self.id} detenido")
        
    def update(self):
        """Actualización periódica."""
        self.updates += 1
        if self.updates % 10 == 0:
            logger.debug(f"Componente {self.id} actualización #{self.updates}")
        
    def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """Manejar un evento."""
        self.events_received.append((event_type, data, source))
        logger.info(f"Componente {self.id} recibió evento {event_type} de {source}")
        
        # Si se solicita respuesta, proporcionar una
        if "_response_to" in data:
            response = {
                "message": f"Respuesta de {self.id} para {event_type}",
                "timestamp": time.time(),
                "original_data": data.get("value", None)
            }
            logger.info(f"Componente {self.id} envía respuesta para {event_type}")
            return response
            
        return None

class DataProcessor(BasicComponent):
    """Componente que procesa datos y emite resultados."""
    
    def __init__(self, component_id: str, engine: SynchronousEngine):
        super().__init__(component_id)
        self.engine = engine
        self.processed_count = 0
        
    def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """Procesar evento y posiblemente emitir nuevo evento con resultado."""
        result = super().handle_event(event_type, data, source)
        
        # Procesar datos específicos
        if event_type == "process_data":
            value = data.get("value", 0)
            processed_value = value * 2  # Simple transformación
            self.processed_count += 1
            
            # Emitir evento con resultado procesado
            logger.info(f"Componente {self.id} procesa valor {value} -> {processed_value}")
            self.engine.emit(
                "processed_result", 
                {"original": value, "result": processed_value, "processor": self.id},
                self.id
            )
            
        return result
        
    def update(self):
        """Actualización con posible emisión de evento periódico."""
        super().update()
        
        # Cada 20 actualizaciones, emitir evento de estado
        if self.updates % 20 == 0:
            logger.info(f"Componente {self.id} emite evento de estado periódico")
            self.engine.emit(
                "status_update",
                {"component": self.id, "processed_count": self.processed_count},
                self.id
            )

class AlertMonitor(BasicComponent):
    """Componente que monitorea eventos y genera alertas."""
    
    def __init__(self, component_id: str):
        super().__init__(component_id)
        self.alert_count = 0
        self.thresholds = {"value": 50}  # Valores que disparan alertas
        
    def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """Monitorear eventos y generar alertas según umbrales."""
        result = super().handle_event(event_type, data, source)
        
        # Verificar umbrales para datos
        if event_type == "processed_result":
            result_value = data.get("result", 0)
            if result_value > self.thresholds["value"]:
                self.alert_count += 1
                logger.warning(
                    f"ALERTA #{self.alert_count}: Valor {result_value} excede umbral "
                    f"{self.thresholds['value']} (procesado por {data.get('processor', 'desconocido')})"
                )
                
        return result

def run_demo():
    """Ejecutar demostración completa del motor síncrono."""
    logger.info("Iniciando demostración del motor síncrono Genesis")
    
    # Crear motor con tick rate más alto para mayor capacidad de respuesta
    engine = SynchronousEngine(tick_rate=0.02)
    
    # Crear componentes
    component_a = BasicComponent("ComponentA")
    component_b = DataProcessor("ProcessorB", engine)
    component_c = AlertMonitor("MonitorC")
    
    # Registrar componentes (con dependencias explícitas)
    engine.register_component("ComponentA", component_a)
    engine.register_component("ProcessorB", component_b, depends_on=["ComponentA"])
    engine.register_component("MonitorC", component_c, depends_on=["ProcessorB"])
    
    # Obtener orden topológico
    order = engine._get_component_order()
    logger.info(f"Orden topológico de componentes: {' -> '.join(order)}")
    
    # Iniciar motor en un hilo separado
    engine_thread = threading.Thread(target=lambda: engine.start(threaded=False))
    engine_thread.daemon = True
    engine_thread.start()
    logger.info("Motor síncrono iniciado en hilo separado")
    
    # Esperar inicialización
    time.sleep(0.1)
    
    # Demostración 1: Evento simple
    logger.info("\n--- DEMO 1: Evento simple ---")
    engine.emit("test_event", {"message": "Hola mundo"}, "demo")
    time.sleep(0.1)  # Dar tiempo para procesar
    
    # Demostración 2: Evento con respuesta sincrónica
    logger.info("\n--- DEMO 2: Evento con respuesta ---")
    responses = engine.emit_with_response("query_data", {"value": 42}, "demo")
    logger.info(f"Respuestas recibidas: {len(responses)}")
    for resp in responses:
        logger.info(f"Respuesta de {resp['component']}: {resp['response']['message']}")
    
    # Demostración 3: Procesamiento de datos en cadena
    logger.info("\n--- DEMO 3: Procesamiento de datos en cadena ---")
    for i in range(1, 6):
        value = i * 20  # 20, 40, 60, 80, 100
        logger.info(f"Enviando valor {value} para procesar")
        engine.emit("process_data", {"value": value}, "demo")
        time.sleep(0.05)  # Pequeña pausa entre envíos
    
    # Esperar a que se procesen todos los eventos
    time.sleep(0.5)
    
    # Demostración 4: Muchos eventos rápidamente (stress test)
    logger.info("\n--- DEMO 4: Stress test con muchos eventos ---")
    start_time = time.time()
    event_count = 50
    for i in range(event_count):
        engine.emit(f"rapid_event_{i}", {"sequence": i, "timestamp": time.time()}, "demo")
        # Sin espera entre eventos
    
    # Esperar procesamiento
    time.sleep(1.0)
    elapsed = time.time() - start_time
    
    # Verificar estado del sistema
    status = engine.get_status()
    logger.info(f"Estado del sistema después del stress test:")
    logger.info(f"- Eventos procesados: {status['events_processed']}")
    logger.info(f"- Ticks totales: {status['tick_count']}")
    logger.info(f"- Tiempo para {event_count} eventos: {elapsed:.2f}s")
    logger.info(f"- Eventos por segundo: {event_count/elapsed:.2f}")
    
    # Verificar recepción en los componentes
    logger.info(f"\nResumen de eventos recibidos:")
    logger.info(f"ComponentA: {len(component_a.events_received)} eventos")
    logger.info(f"ProcessorB: {len(component_b.events_received)} eventos, {component_b.processed_count} procesados")
    logger.info(f"MonitorC: {len(component_c.events_received)} eventos, {component_c.alert_count} alertas")
    
    # Detener el motor
    logger.info("\n--- Finalizando demostración ---")
    engine.stop()
    engine_thread.join(timeout=1.0)
    
    logger.info("Demostración completada!")

if __name__ == "__main__":
    run_demo()