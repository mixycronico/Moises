"""
Test simple para verificar el funcionamiento básico del sistema híbrido.

Este script comprueba las capacidades básicas del sistema híbrido
sin utilizar el framework de pruebas completo.
"""

import asyncio
import logging
import sys
import time
from typing import Dict, Any, Optional, List, Set

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hybrid_test")

# Interfaces básicas para componentes y coordinador
class ComponentAPI:
    def __init__(self, id: str):
        self.id = id
        self.received_events: List[Dict[str, Any]] = []
        
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """Procesar solicitudes API."""
        raise NotImplementedError("Los componentes deben implementar process_request")
    
    async def on_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Manejar eventos recibidos por WebSocket."""
        self.received_events.append({
            "type": event_type,
            "data": data,
            "source": source
        })
        logger.debug(f"[{self.id}] Evento recibido: {event_type} de {source}")
    
    async def start(self) -> None:
        """Iniciar el componente."""
        logger.info(f"Componente {self.id} iniciado")
    
    async def stop(self) -> None:
        """Detener el componente."""
        logger.info(f"Componente {self.id} detenido")

class Coordinator:
    def __init__(self):
        self.components: Dict[str, ComponentAPI] = {}
        self.event_subscribers: Dict[str, Set[str]] = {}
        
    def register_component(self, id: str, component: ComponentAPI) -> None:
        """Registrar un componente en el coordinador."""
        self.components[id] = component
        logger.info(f"Componente {id} registrado")
    
    async def request(self, target_id: str, request_type: str, data: Dict[str, Any], source: str) -> Optional[Any]:
        """API: Enviar solicitud directa a un componente."""
        if target_id not in self.components:
            logger.error(f"Componente {target_id} no encontrado")
            return None
        
        try:
            # Timeout para evitar deadlocks
            result = await asyncio.wait_for(
                self.components[target_id].process_request(request_type, data, source),
                timeout=5.0
            )
            return result
        except asyncio.TimeoutError:
            logger.error(f"Timeout en solicitud {request_type} a {target_id}")
            return None
        except Exception as e:
            logger.error(f"Error en solicitud: {e}")
            return None
    
    async def emit_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """WebSocket: Emitir evento a todos los componentes suscritos."""
        subscribers = self.event_subscribers.get(event_type, set())
        tasks = []
        
        # Para cada suscriptor, crear tarea de envío de evento
        for comp_id in subscribers:
            if comp_id in self.components and comp_id != source:  # No enviar a la fuente
                tasks.append(
                    self.components[comp_id].on_event(event_type, data, source)
                )
        
        # Ejecutar todas las tareas concurrentemente
        if tasks:
            await asyncio.gather(*tasks)
    
    def subscribe(self, component_id: str, event_types: List[str]) -> None:
        """Suscribir un componente a tipos de eventos específicos."""
        if component_id not in self.components:
            logger.error(f"No se puede suscribir: componente {component_id} no registrado")
            return
        
        for event_type in event_types:
            if event_type not in self.event_subscribers:
                self.event_subscribers[event_type] = set()
            self.event_subscribers[event_type].add(component_id)
            logger.info(f"Componente {component_id} suscrito a eventos {event_type}")
    
    async def start(self) -> None:
        """Iniciar el coordinador y todos los componentes."""
        for comp_id, comp in self.components.items():
            await comp.start()
        logger.info("Coordinador iniciado")
    
    async def stop(self) -> None:
        """Detener el coordinador y todos los componentes."""
        for comp_id, comp in self.components.items():
            await comp.stop()
        logger.info("Coordinador detenido")

# Coordinador global para pruebas
coordinator = Coordinator()

# Implementación de componente de prueba
class TestComponent(ComponentAPI):
    """Componente de prueba para el sistema híbrido."""
    
    def __init__(self, id: str):
        super().__init__(id)
        self.processed_requests: List[Dict[str, Any]] = []
        
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """Procesar solicitud de prueba."""
        self.processed_requests.append({
            "type": request_type,
            "data": data,
            "source": source
        })
        
        logger.info(f"[{self.id}] Procesando solicitud {request_type} de {source}")
        
        # Echo simple
        if request_type == "echo":
            return {"echo": data.get("message", ""), "source": source}
        
        # Emitir evento
        elif request_type == "emit_event":
            event_type = data.get("event_type")
            event_data = data.get("event_data", {})
            
            if event_type:
                logger.info(f"[{self.id}] Emitiendo evento {event_type}")
                await coordinator.emit_event(event_type, event_data, self.id)
                return {"status": "event_emitted", "event_type": event_type}
            
            return {"status": "error", "message": "No event_type provided"}
        
        return {"status": "unknown_request"}
    
    async def on_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Manejar evento de prueba."""
        await super().on_event(event_type, data, source)
        logger.info(f"[{self.id}] Evento {event_type} recibido de {source}")

async def test_basic_functionality():
    """Probar funcionalidad básica del sistema híbrido."""
    logger.info("\n=== Prueba de Funcionalidad Básica ===")
    
    # Crear componentes
    comp1 = TestComponent("comp1")
    comp2 = TestComponent("comp2")
    
    # Registrar componentes
    coordinator.register_component("comp1", comp1)
    coordinator.register_component("comp2", comp2)
    
    # Suscribir componentes a eventos
    coordinator.subscribe("comp1", ["notification", "alert"])
    coordinator.subscribe("comp2", ["data_update", "notification"])
    
    # Iniciar coordinador y componentes
    await coordinator.start()
    
    try:
        # 1. Probar solicitud API directa
        logger.info("\n1. Probando solicitud API directa (comp1 -> comp2)")
        api_result = await coordinator.request(
            "comp2",
            "echo",
            {"message": "Hola desde comp1"},
            "comp1"
        )
        
        api_success = api_result and api_result.get("echo") == "Hola desde comp1"
        if api_success:
            logger.info(f"✅ Solicitud API exitosa: {api_result}")
        else:
            logger.error(f"❌ Solicitud API fallida: {api_result}")
        
        # 2. Probar emisión de evento por WebSocket
        logger.info("\n2. Probando emisión de evento (comp2 -> comp1)")
        event_result = await coordinator.request(
            "comp2",
            "emit_event",
            {
                "event_type": "notification",
                "event_data": {"message": "Notificación de prueba"}
            },
            "test"
        )
        
        # Dar tiempo para que el evento sea procesado
        await asyncio.sleep(0.1)
        
        event_success = len(comp1.received_events) > 0 and comp1.received_events[-1]["type"] == "notification"
        if event_success:
            logger.info(f"✅ Emisión de evento exitosa, evento recibido: {comp1.received_events[-1]}")
        else:
            logger.error(f"❌ Emisión de evento fallida, eventos recibidos: {comp1.received_events}")
        
        # Resultados
        success = api_success and event_success
        if success:
            logger.info("\n✅ Prueba de funcionalidad básica exitosa")
        else:
            logger.error("\n❌ Prueba de funcionalidad básica fallida")
        
        return success
    
    finally:
        # Detener coordinador y componentes
        await coordinator.stop()

async def test_recursive_calls():
    """Probar llamadas recursivas que causarían deadlock en el sistema anterior."""
    logger.info("\n=== Prueba de Llamadas Recursivas ===")
    
    class RecursiveComponent(ComponentAPI):
        def __init__(self, id: str):
            super().__init__(id)
            self.call_count = 0
        
        async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
            self.call_count += 1
            current_count = self.call_count
            
            if request_type == "recursive":
                depth = data.get("depth", 1)
                logger.info(f"[{self.id}] Procesando llamada recursiva profundidad {depth} (llamada #{current_count})")
                
                if depth > 1:
                    # Llamada recursiva a sí mismo (causaría deadlock en sistema antiguo)
                    result = await coordinator.request(
                        self.id,  # Llamada a sí mismo
                        "recursive",
                        {"depth": depth - 1},
                        source
                    )
                    
                    return {
                        "depth": depth,
                        "next": result,
                        "call_count": current_count
                    }
                
                return {"depth": depth, "message": "Caso base alcanzado", "call_count": current_count}
            
            return {"status": "unknown_request", "call_count": current_count}
    
    # Crear componente recursivo
    recursive_comp = RecursiveComponent("recursive_comp")
    coordinator.register_component("recursive_comp", recursive_comp)
    
    # Iniciar coordinador
    await coordinator.start()
    
    try:
        # Probar llamada recursiva profunda
        logger.info("Iniciando llamada recursiva (profundidad 5)")
        start_time = time.time()
        result = await coordinator.request(
            "recursive_comp",
            "recursive",
            {"depth": 5},  # Profundidad 5, causaría deadlock en sistema antiguo
            "test"
        )
        duration = time.time() - start_time
        
        # Verificar resultado
        success = result and result.get("depth") == 5
        if success:
            logger.info(f"✅ Llamada recursiva exitosa (completada en {duration:.2f}s)")
            logger.info(f"   Llamadas realizadas: {recursive_comp.call_count}")
        else:
            logger.error(f"❌ Llamada recursiva fallida: {result}")
        
        return success
    
    finally:
        await coordinator.stop()

async def test_circular_calls():
    """Probar llamadas circulares que causarían deadlock en el sistema anterior."""
    logger.info("\n=== Prueba de Llamadas Circulares ===")
    
    class CircularComponent(ComponentAPI):
        def __init__(self, id: str):
            super().__init__(id)
            self.call_count = 0
        
        async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
            self.call_count += 1
            current_count = self.call_count
            
            if request_type == "circular":
                target_id = data.get("target_id")
                depth = data.get("depth", 1)
                
                logger.info(f"[{self.id}] Procesando llamada circular, objetivo: {target_id}, profundidad: {depth}")
                
                if depth > 0 and target_id and target_id in coordinator.components:
                    # Llamada a otro componente que podría llamarnos de vuelta
                    logger.info(f"[{self.id}] Llamando a {target_id}")
                    result = await coordinator.request(
                        target_id,
                        "circular",
                        {
                            "target_id": self.id,  # El otro nos llamará a nosotros
                            "depth": depth - 1
                        },
                        self.id
                    )
                    
                    return {
                        "depth": depth,
                        "next": result,
                        "call_count": current_count
                    }
                
                return {"depth": depth, "message": "Caso base alcanzado", "call_count": current_count}
            
            return {"status": "unknown_request", "call_count": current_count}
    
    # Crear componentes circulares
    comp_a = CircularComponent("comp_a")
    comp_b = CircularComponent("comp_b")
    
    coordinator.register_component("comp_a", comp_a)
    coordinator.register_component("comp_b", comp_b)
    
    # Iniciar coordinador
    await coordinator.start()
    
    try:
        # Probar llamada circular
        logger.info("Iniciando llamada circular (A -> B -> A -> B)")
        start_time = time.time()
        result = await coordinator.request(
            "comp_a",
            "circular",
            {
                "target_id": "comp_b",
                "depth": 3  # Profundidad 3, alternará entre A y B
            },
            "test"
        )
        duration = time.time() - start_time
        
        # Verificar resultado
        success = result and result.get("depth") == 3
        total_calls = comp_a.call_count + comp_b.call_count
        
        if success:
            logger.info(f"✅ Llamada circular exitosa (completada en {duration:.2f}s)")
            logger.info(f"   Llamadas realizadas: {total_calls} (A: {comp_a.call_count}, B: {comp_b.call_count})")
        else:
            logger.error(f"❌ Llamada circular fallida: {result}")
        
        return success
    
    finally:
        await coordinator.stop()

async def run_tests():
    """Ejecutar todas las pruebas."""
    results = []
    
    # 1. Probar funcionalidad básica
    results.append(await test_basic_functionality())
    
    # 2. Probar llamadas recursivas (que causarían deadlock en sistema antiguo)
    results.append(await test_recursive_calls())
    
    # 3. Probar llamadas circulares (que causarían deadlock en sistema antiguo)
    results.append(await test_circular_calls())
    
    # Mostrar resumen
    success = all(results)
    total = len(results)
    passed = sum(1 for r in results if r)
    
    logger.info("\n=== Resumen de Pruebas ===")
    logger.info(f"Pruebas pasadas: {passed}/{total}")
    
    if success:
        logger.info("✅ TODAS LAS PRUEBAS PASARON - Sistema híbrido funcionando correctamente")
    else:
        logger.error(f"❌ ALGUNAS PRUEBAS FALLARON - {total - passed} pruebas fallaron")

# Ejecutar pruebas
if __name__ == "__main__":
    asyncio.run(run_tests())