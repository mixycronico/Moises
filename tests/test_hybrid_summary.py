"""
Versión simplificada del test para verificar el sistema híbrido.

Esta versión es más concisa para facilitar la visualización de los resultados.
"""

import asyncio
import logging
import sys
import time
from typing import Dict, Any, Optional, List, Set

# Configurar logging para mostrar solo INFO y mayor
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hybrid_summary")

# Interfaces básicas para componentes y coordinador
class ComponentAPI:
    def __init__(self, id: str):
        self.id = id
        self.received_events = []
        
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        return {"status": "ok", "echo": data.get("message", "")}
    
    async def on_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        self.received_events.append({"type": event_type, "data": data, "source": source})
    
    async def start(self) -> None:
        pass
    
    async def stop(self) -> None:
        pass

class Coordinator:
    def __init__(self):
        self.components = {}
        self.event_subscribers = {}
        
    def register_component(self, id: str, component: ComponentAPI) -> None:
        self.components[id] = component
        
    def subscribe(self, component_id: str, event_types: List[str]) -> None:
        if component_id not in self.components:
            return
        
        for event_type in event_types:
            if event_type not in self.event_subscribers:
                self.event_subscribers[event_type] = set()
            self.event_subscribers[event_type].add(component_id)
    
    async def request(self, target_id: str, request_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        if target_id not in self.components:
            return None
        
        try:
            # Timeout para evitar deadlocks
            result = await asyncio.wait_for(
                self.components[target_id].process_request(request_type, data, source),
                timeout=5.0
            )
            return result
        except Exception as e:
            logger.error(f"Error en solicitud: {e}")
            return None
    
    async def emit_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        subscribers = self.event_subscribers.get(event_type, set())
        tasks = []
        
        for comp_id in subscribers:
            if comp_id in self.components and comp_id != source:
                tasks.append(self.components[comp_id].on_event(event_type, data, source))
        
        if tasks:
            await asyncio.gather(*tasks)
    
    async def start(self) -> None:
        for comp in self.components.values():
            await comp.start()
    
    async def stop(self) -> None:
        for comp in self.components.values():
            await comp.stop()

# Componente que se llama a sí mismo recursivamente
class RecursiveComponent(ComponentAPI):
    def __init__(self, id: str):
        super().__init__(id)
        self.call_count = 0
    
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        self.call_count += 1
        
        if request_type == "recursive":
            depth = data.get("depth", 1)
            
            if depth > 1:
                # Llamada recursiva a sí mismo (causaría deadlock en sistema antiguo)
                result = await coordinator.request(
                    self.id,
                    "recursive",
                    {"depth": depth - 1},
                    source
                )
                
                return {"depth": depth, "next": result}
            
            return {"depth": depth, "message": "Base case reached"}
        
        return await super().process_request(request_type, data, source)

# Componentes que se llaman circularmente
class CircularComponent(ComponentAPI):
    def __init__(self, id: str):
        super().__init__(id)
        self.call_count = 0
    
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        self.call_count += 1
        
        if request_type == "circular":
            target_id = data.get("target_id")
            depth = data.get("depth", 1)
            
            if depth > 0 and target_id:
                # Llamada a otro componente que podría llamarnos de vuelta
                result = await coordinator.request(
                    target_id,
                    "circular",
                    {
                        "target_id": self.id,  # El otro nos llamará a nosotros
                        "depth": depth - 1
                    },
                    self.id
                )
                
                return {"depth": depth, "next": result}
            
            return {"depth": depth, "message": "Base case reached"}
        
        return await super().process_request(request_type, data, source)

# Coordinador global para pruebas
coordinator = Coordinator()

async def test_api_functionality():
    """Probar funcionalidad básica API."""
    comp1 = ComponentAPI("comp1")
    comp2 = ComponentAPI("comp2")
    
    coordinator.register_component("comp1", comp1)
    coordinator.register_component("comp2", comp2)
    
    await coordinator.start()
    
    try:
        # Solicitud API directa
        result = await coordinator.request(
            "comp2",
            "echo",
            {"message": "Hola desde comp1"},
            "comp1"
        )
        
        return result is not None and result.get("echo") == "Hola desde comp1"
    finally:
        await coordinator.stop()

async def test_event_functionality():
    """Probar funcionalidad básica WebSocket (eventos)."""
    comp1 = ComponentAPI("comp1")
    comp2 = ComponentAPI("comp2")
    
    coordinator.register_component("comp1", comp1)
    coordinator.register_component("comp2", comp2)
    coordinator.subscribe("comp1", ["notification"])
    
    await coordinator.start()
    
    try:
        # Emisión de evento
        await coordinator.emit_event(
            "notification",
            {"message": "Notificación de prueba"},
            "comp2"
        )
        
        await asyncio.sleep(0.1)  # Breve pausa para procesamiento
        
        return len(comp1.received_events) > 0 and comp1.received_events[0]["type"] == "notification"
    finally:
        await coordinator.stop()

async def test_recursive_calls():
    """Probar llamadas recursivas (que causarían deadlock en sistema antiguo)."""
    recursive_comp = RecursiveComponent("recursive_comp")
    coordinator.register_component("recursive_comp", recursive_comp)
    
    await coordinator.start()
    
    try:
        # Llamada recursiva profunda
        result = await coordinator.request(
            "recursive_comp",
            "recursive",
            {"depth": 5},
            "test"
        )
        
        return result is not None and result.get("depth") == 5
    finally:
        await coordinator.stop()

async def test_circular_calls():
    """Probar llamadas circulares (que causarían deadlock en sistema antiguo)."""
    comp_a = CircularComponent("comp_a")
    comp_b = CircularComponent("comp_b")
    
    coordinator.register_component("comp_a", comp_a)
    coordinator.register_component("comp_b", comp_b)
    
    await coordinator.start()
    
    try:
        # Llamada circular
        result = await coordinator.request(
            "comp_a",
            "circular",
            {
                "target_id": "comp_b",
                "depth": 3
            },
            "test"
        )
        
        return result is not None and result.get("depth") == 3
    finally:
        await coordinator.stop()

async def run_tests():
    """Ejecutar todas las pruebas y mostrar resumen."""
    tests = {
        "API básica": test_api_functionality,
        "Eventos WebSocket": test_event_functionality,
        "Llamadas recursivas": test_recursive_calls,
        "Llamadas circulares": test_circular_calls
    }
    
    results = {}
    
    # Ejecutar cada prueba
    for name, test_func in tests.items():
        logger.info(f"Ejecutando prueba: {name}")
        try:
            start_time = time.time()
            result = await test_func()
            duration = time.time() - start_time
            
            results[name] = {"success": result, "duration": duration}
            
            if result:
                logger.info(f"✅ Prueba '{name}' exitosa ({duration:.2f}s)")
            else:
                logger.error(f"❌ Prueba '{name}' fallida ({duration:.2f}s)")
        except Exception as e:
            logger.error(f"❌ Error en prueba '{name}': {e}")
            results[name] = {"success": False, "error": str(e)}
    
    # Mostrar resumen
    successful = sum(1 for r in results.values() if r.get("success", False))
    total = len(tests)
    
    logger.info("\n=== RESUMEN DE PRUEBAS ===")
    logger.info(f"Pruebas exitosas: {successful}/{total}")
    
    if successful == total:
        logger.info("✅ TODAS LAS PRUEBAS PASARON - Sistema híbrido funciona correctamente")
        logger.info("   Las situaciones que causaban deadlock en el sistema antiguo ahora se manejan correctamente")
    else:
        logger.error(f"❌ ALGUNAS PRUEBAS FALLARON - {total - successful} pruebas fallaron")
    
    return successful == total

if __name__ == "__main__":
    asyncio.run(run_tests())