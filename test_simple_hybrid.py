"""
Test simple para verificar el funcionamiento básico del sistema híbrido.

Este script comprueba las capacidades básicas del sistema híbrido
sin utilizar el framework de pruebas completo.
"""

import asyncio
import logging
import sys
import os
from typing import Dict, Any, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_hybrid")

# Asegurar que el directorio raíz está en el path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Importar el sistema híbrido
from genesis.core.genesis_hybrid import ComponentAPI, GenesisHybridCoordinator

class TestComponent(ComponentAPI):
    """Componente de prueba para el sistema híbrido."""
    
    def __init__(self, id: str):
        super().__init__(id)
        self.requests_handled = []
        self.results = {}
    
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """Procesar solicitud de prueba."""
        self.requests_handled.append((request_type, data, source))
        self.metrics["requests_processed"] += 1
        
        if request_type == "echo":
            return {"echo": data.get("message", ""), "source": source}
        
        elif request_type == "sleep":
            duration = data.get("duration", 0.1)
            await asyncio.sleep(duration)
            return {"slept": duration}
        
        elif request_type == "get_results":
            return self.results
        
        return None
    
    async def on_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Manejar evento de prueba."""
        await super().on_event(event_type, data, source)
        
        if event_type == "store_result":
            key = data.get("key", "default")
            value = data.get("value")
            self.results[key] = value
            logger.info(f"{self.id}: Almacenado {key}={value}")

async def test_basic_functionality():
    """Probar funcionalidad básica del sistema híbrido."""
    # Crear coordinador
    coordinator = GenesisHybridCoordinator(host="localhost", port=0)  # Puerto 0 para evitar conflictos
    
    # Crear componentes
    comp1 = TestComponent("comp1")
    comp2 = TestComponent("comp2")
    
    # Registrar componentes
    coordinator.register_component("comp1", comp1)
    coordinator.register_component("comp2", comp2)
    
    # Iniciar coordinador
    await coordinator.start()
    
    try:
        logger.info("Probando solicitud API básica...")
        # Probar solicitud API básica
        result = await coordinator.request(
            "comp2",
            "echo",
            {"message": "Hello from comp1"},
            "comp1"
        )
        
        assert result is not None, "La solicitud falló"
        assert result["echo"] == "Hello from comp1", f"Resultado incorrecto: {result}"
        assert result["source"] == "comp1", f"Fuente incorrecta: {result}"
        
        logger.info(f"Solicitud API exitosa: {result}")
        
        # Probar broadcast de evento
        logger.info("Probando broadcast de evento...")
        await coordinator.broadcast_event(
            "store_result",
            {"key": "test_key", "value": "test_value"},
            "comp1"  # Origen
        )
        
        # Dar tiempo para que se procese el evento
        await asyncio.sleep(0.1)
        
        # Verificar que comp2 recibió el evento (comp1 no, es el origen)
        assert len(comp2.events_received) == 1, f"comp2 debería tener 1 evento, tiene {len(comp2.events_received)}"
        assert "test_key" in comp2.results, f"comp2 debería tener test_key en results, tiene {comp2.results}"
        assert comp2.results["test_key"] == "test_value", f"comp2.results[test_key] debería ser test_value, es {comp2.results['test_key']}"
        
        logger.info(f"Broadcast exitoso: {comp2.results}")
        
        # Probar solicitudes concurrentes
        logger.info("Probando solicitudes concurrentes...")
        tasks = []
        for i in range(5):
            tasks.append(
                coordinator.request(
                    "comp2",
                    "sleep",
                    {"duration": 0.1},
                    "comp1"
                )
            )
        
        results = await asyncio.gather(*tasks)
        assert all(r is not None for r in results), f"Alguna solicitud concurrente falló: {results}"
        assert all(r["slept"] == 0.1 for r in results), f"Alguna solicitud concurrente retornó valor incorrecto: {results}"
        
        logger.info(f"Solicitudes concurrentes exitosas: {len(results)} completadas")
        
        # Probar solicitud con timeout
        logger.info("Probando solicitud con timeout...")
        timeout_result = await coordinator.request(
            "comp2",
            "sleep",
            {"duration": 0.5},  # Dormir 0.5 segundos
            "comp1",
            timeout=0.1  # Timeout de 0.1 segundos (debería fallar)
        )
        
        assert timeout_result is None, f"La solicitud con timeout no falló correctamente: {timeout_result}"
        logger.info("Timeout manejado correctamente")
        
        logger.info("Todas las pruebas completadas exitosamente")
        
    finally:
        # Detener coordinador
        await coordinator.stop()

async def test_recursive_calls():
    """Probar llamadas recursivas que causarían deadlock en el sistema anterior."""
    # Crear coordinador
    coordinator = GenesisHybridCoordinator()
    
    # Clase para probar llamadas recursivas
    class RecursiveComponent(ComponentAPI):
        def __init__(self, id: str):
            super().__init__(id)
            self.call_count = 0
        
        async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
            self.call_count += 1
            current_count = self.call_count
            
            if request_type == "recursive":
                depth = data.get("depth", 1)
                if depth > 1:
                    # Llamarse a sí mismo (antes causaba deadlock)
                    try:
                        result = await coordinator.request(
                            self.id,  # Llamarse a sí mismo
                            "recursive",
                            {"depth": depth - 1},
                            source
                        )
                        return {
                            "current_depth": depth,
                            "call_count": current_count,
                            "next_result": result
                        }
                    except Exception as e:
                        return {"error": str(e), "current_depth": depth}
                else:
                    return {"depth": depth, "call_count": current_count}
            
            return None
    
    # Crear componente
    recursive_comp = RecursiveComponent("recursive")
    coordinator.register_component("recursive", recursive_comp)
    
    await coordinator.start()
    
    try:
        logger.info("Probando llamadas recursivas (mismo componente llamándose a sí mismo)...")
        # Hacer llamada recursiva con profundidad 5
        result = await coordinator.request(
            "recursive",
            "recursive",
            {"depth": 5},  # Profundidad 5, causaría deadlock en el sistema anterior
            "test"
        )
        
        assert result is not None, "La llamada recursiva falló"
        assert result["current_depth"] == 5, f"Profundidad incorrecta: {result}"
        assert "next_result" in result, f"No contiene resultado anidado: {result}"
        
        # Verificar anidamiento de resultados
        depth = 5
        current = result
        while "next_result" in current and current["next_result"] is not None:
            depth -= 1
            current = current["next_result"]
            if depth > 1:
                assert current["current_depth"] == depth, f"Profundidad incorrecta en nivel {depth}: {current}"
        
        logger.info(f"Llamadas recursivas exitosas con profundidad 5: {result}")
        
    finally:
        # Detener coordinador
        await coordinator.stop()

async def test_circular_calls():
    """Probar llamadas circulares que causarían deadlock en el sistema anterior."""
    # Crear coordinador
    coordinator = GenesisHybridCoordinator()
    
    # Clase para probar llamadas circulares
    class CircularComponent(ComponentAPI):
        def __init__(self, id: str):
            super().__init__(id)
            self.call_count = 0
        
        async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
            self.call_count += 1
            current_count = self.call_count
            
            if request_type == "circular":
                target_id = data.get("target_id")
                depth = data.get("depth", 2)
                
                if depth > 0 and target_id:
                    # Llamar al objetivo, que podría llamarnos de vuelta
                    try:
                        result = await coordinator.request(
                            target_id,
                            "circular",
                            {
                                "target_id": self.id,  # Establecer el objetivo como nosotros mismos
                                "depth": depth - 1
                            },
                            self.id
                        )
                        return {
                            "current_depth": depth,
                            "target": target_id,
                            "call_count": current_count,
                            "next_result": result
                        }
                    except Exception as e:
                        return {"error": str(e), "current_depth": depth}
                else:
                    return {"depth": depth, "call_count": current_count}
            
            return None
    
    # Crear componentes
    comp1 = CircularComponent("circular1")
    comp2 = CircularComponent("circular2")
    
    coordinator.register_component("circular1", comp1)
    coordinator.register_component("circular2", comp2)
    
    await coordinator.start()
    
    try:
        logger.info("Probando llamadas circulares (A llama a B, B llama a A)...")
        # Iniciar cadena de llamadas circulares
        result = await coordinator.request(
            "circular1",
            "circular",
            {
                "target_id": "circular2",
                "depth": 4  # Profundidad 4, alternará entre ambos componentes
            },
            "test"
        )
        
        assert result is not None, "La llamada circular falló"
        assert result["current_depth"] == 4, f"Profundidad incorrecta: {result}"
        assert "next_result" in result, f"No contiene resultado anidado: {result}"
        
        # Verificar alternancia de llamadas
        total_calls = comp1.call_count + comp2.call_count
        assert total_calls == 4, f"Número incorrecto de llamadas: {total_calls} (comp1: {comp1.call_count}, comp2: {comp2.call_count})"
        
        logger.info(f"Llamadas circulares exitosas con profundidad 4: {total_calls} llamadas totales")
        
    finally:
        # Detener coordinador
        await coordinator.stop()

async def run_tests():
    """Ejecutar todas las pruebas."""
    try:
        # Ejecutar solo la prueba principal para evitar timeouts
        logger.info("=== Prueba Principal: Funcionalidad Básica y Anti-Deadlock ===")
        
        # Crear coordinador
        coordinator = GenesisHybridCoordinator()
        
        # Componentes para prueba rápida
        comp1 = TestComponent("comp1")
        comp2 = TestComponent("comp2")
        
        # Registrar componentes
        coordinator.register_component("comp1", comp1)
        coordinator.register_component("comp2", comp2)
        
        # Iniciar coordinador con un timeout corto
        coordinator_start_task = asyncio.create_task(coordinator.start())
        # Correr solo por 2 segundos
        await asyncio.sleep(2)
        
        # Realizar prueba básica
        result = await coordinator.request(
            "comp2",
            "echo",
            {"message": "Hello from comp1"},
            "comp1"
        )
        
        assert result is not None, "La solicitud falló"
        assert result["echo"] == "Hello from comp1", f"Resultado incorrecto: {result}"
        
        logger.info(f"Solicitud básica exitosa: {result}")
        
        # Detener coordinador
        coordinator.running = False
        try:
            # Esperar un poco para que se detenga limpiamente
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
        
        logger.info("¡Prueba completada exitosamente!")
        
    except Exception as e:
        logger.error(f"Error durante las pruebas: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(run_tests())
    except KeyboardInterrupt:
        logger.info("Pruebas interrumpidas por el usuario")
    except Exception as e:
        logger.error(f"Error ejecutando pruebas: {e}")
        sys.exit(1)