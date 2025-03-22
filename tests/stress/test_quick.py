"""
Prueba rápida del sistema híbrido para demostración.

Este script ejecuta una prueba abreviada para demostrar
el funcionamiento del sistema híbrido bajo estrés moderado.
"""

import asyncio
import logging
import random
import time
from typing import Dict, Any, List

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("quick_test")

# Clase básica de componente para pruebas
class TestComponent:
    def __init__(self, id: str):
        self.id = id
        self.received_events = []
        self.call_count = 0
        
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """Procesar solicitud API."""
        self.call_count += 1
        
        # Simular latencia de procesamiento
        await asyncio.sleep(0.01)
        
        if request_type == "echo":
            return {"echo": data.get("message", ""), "from": self.id}
        
        if request_type == "compute":
            # Simular trabajo
            result = sum(i for i in range(10000))
            return {"result": result}
        
        return {"status": "ok", "from": self.id}
    
    async def on_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Manejar evento WebSocket."""
        self.received_events.append({
            "type": event_type,
            "data": data,
            "source": source
        })
        
        # Simular procesamiento
        await asyncio.sleep(0.005)
        
    async def start(self) -> None:
        logger.info(f"Componente {self.id} iniciado")
    
    async def stop(self) -> None:
        logger.info(f"Componente {self.id} detenido")

# Coordinador para gestionar componentes y comunicación
class Coordinator:
    def __init__(self):
        self.components = {}
        self.event_subscribers = {}
        
    def register_component(self, id: str, component: TestComponent) -> None:
        """Registrar componente."""
        self.components[id] = component
        logger.info(f"Componente {id} registrado")
    
    def subscribe(self, component_id: str, event_types: List[str]) -> None:
        """Suscribir componente a tipos de eventos."""
        if component_id not in self.components:
            logger.error(f"No se puede suscribir: componente {component_id} no existe")
            return
        
        for event_type in event_types:
            if event_type not in self.event_subscribers:
                self.event_subscribers[event_type] = set()
            self.event_subscribers[event_type].add(component_id)
            logger.info(f"Componente {component_id} suscrito a {event_type}")
    
    async def request(self, target_id: str, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """API: Enviar solicitud directa con timeout."""
        if target_id not in self.components:
            logger.error(f"Componente {target_id} no encontrado")
            return None
        
        try:
            # Clave del sistema híbrido: uso de timeout
            result = await asyncio.wait_for(
                self.components[target_id].process_request(request_type, data, source),
                timeout=0.5  # Timeout corto para la prueba
            )
            return result
        except asyncio.TimeoutError:
            logger.error(f"Timeout en solicitud a {target_id}")
            return None
        except Exception as e:
            logger.error(f"Error en solicitud: {e}")
            return None
    
    async def emit_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """WebSocket: Emitir evento a todos los suscriptores."""
        subscribers = self.event_subscribers.get(event_type, set())
        tasks = []
        
        # Crear tareas de entrega para cada suscriptor
        for comp_id in subscribers:
            if comp_id in self.components and comp_id != source:
                tasks.append(
                    self.components[comp_id].on_event(event_type, data, source)
                )
        
        # Ejecutar entregas en paralelo
        if tasks:
            await asyncio.gather(*tasks)
    
    async def start(self) -> None:
        """Iniciar todos los componentes."""
        start_tasks = [comp.start() for comp in self.components.values()]
        await asyncio.gather(*start_tasks)
        logger.info(f"Coordinador iniciado con {len(self.components)} componentes")
    
    async def stop(self) -> None:
        """Detener todos los componentes."""
        stop_tasks = [comp.stop() for comp in self.components.values()]
        await asyncio.gather(*stop_tasks)
        logger.info("Coordinador detenido")

async def run_quick_test():
    """Ejecutar una prueba rápida del sistema híbrido."""
    logger.info("=== Iniciando Prueba Rápida del Sistema Híbrido ===")
    
    # Crear y configurar sistema
    system = Coordinator()
    
    # Registrar componentes
    for i in range(5):
        system.register_component(f"comp_{i}", TestComponent(f"comp_{i}"))
    
    # Suscribir a eventos
    event_types = ["notification", "data_update", "alert"]
    for i in range(5):
        # Cada componente se suscribe a 1-2 tipos de eventos
        num_events = random.randint(1, 2)
        selected_events = random.sample(event_types, num_events)
        system.subscribe(f"comp_{i}", selected_events)
    
    # Iniciar sistema
    await system.start()
    
    try:
        # Prueba 1: Solicitudes API directas
        logger.info("\n--- Prueba 1: Solicitudes API Directas ---")
        api_results = []
        
        for i in range(3):
            result = await system.request(
                f"comp_{random.randint(0, 4)}",  # Componente aleatorio
                "echo",
                {"message": f"Mensaje de prueba {i}"},
                "test"
            )
            api_results.append(result)
            
        logger.info(f"Resultados API: {api_results}")
        
        # Prueba 2: Emisión de eventos WebSocket
        logger.info("\n--- Prueba 2: Emisión de Eventos WebSocket ---")
        
        for i in range(5):
            event_type = random.choice(event_types)
            await system.emit_event(
                event_type,
                {"message": f"Evento {event_type} #{i}", "priority": random.randint(1, 3)},
                "test"
            )
        
        # Esperar un momento para procesamiento de eventos
        await asyncio.sleep(0.1)
        
        # Mostrar eventos recibidos por componentes
        for comp_id, comp in system.components.items():
            logger.info(f"{comp_id} recibió {len(comp.received_events)} eventos")
        
        # Prueba 3: Llamadas circulares y recursivas
        logger.info("\n--- Prueba 3: Llamadas Recursivas ---")
        
        # Solicitud que genera una cadena de llamadas
        # En el sistema antiguo, esto causaría un deadlock
        for i in range(3):
            target = f"comp_{i}"
            source = f"comp_{(i+1) % 3}"
            
            result = await system.request(
                target,
                "echo",
                {"message": f"Mensaje de {source} a {target}"},
                source
            )
            logger.info(f"Llamada de {source} a {target}: {result}")
        
        logger.info("\n=== Prueba Completada Exitosamente ===")
        return True
        
    except Exception as e:
        logger.error(f"Error en prueba: {e}")
        return False
        
    finally:
        # Asegurar que el sistema se detenga
        await system.stop()

# Ejecutar prueba directamente
if __name__ == "__main__":
    asyncio.run(run_quick_test())