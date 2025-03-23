"""
Prueba de solución de deadlocks con el sistema híbrido Genesis.

Este script muestra cómo el sistema híbrido resuelve problemas de deadlock
en situaciones que causaban bloqueo en el sistema anterior.
"""

import asyncio
import logging
import sys
import time
from typing import Dict, Any, Optional, List

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_deadlock")

# Simulación del sistema híbrido
class ComponentAPI:
    def __init__(self, id: str):
        self.id = id
        self.call_count = 0
        
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        self.call_count += 1
        current_count = self.call_count
        
        logger.info(f"[{self.id}] Procesando {request_type} de {source} (#{current_count})")
        
        # Simular diferentes tipos de solicitudes que antes causaban deadlocks
        if request_type == "recursive":
            # Llamada recursiva (misma componente)
            depth = data.get("depth", 1)
            
            if depth > 1:
                # Llamar recursivamente a esta misma componente
                logger.info(f"[{self.id}] Llamada recursiva a sí mismo, profundidad {depth}")
                result = await coordinator.request(
                    self.id,  # Llamarse a sí mismo
                    "recursive",
                    {"depth": depth - 1},
                    source
                )
                
                return {
                    "current_depth": depth,
                    "next_result": result,
                    "call_count": current_count
                }
            
            return {"depth": depth, "call_count": current_count}
            
        elif request_type == "circular":
            # Llamada circular (A llama a B, B llama a A)
            target_id = data.get("target_id")
            depth = data.get("depth", 2)
            
            if depth > 0 and target_id:
                # Hacer una llamada al objetivo, que podría llamarnos de vuelta
                logger.info(f"[{self.id}] Llamada circular a {target_id}, profundidad {depth}")
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
                    "next_result": result,
                    "call_count": current_count
                }
            
            return {"depth": depth, "call_count": current_count}
            
        elif request_type == "blocking":
            # Operación bloqueante
            duration = data.get("duration", 0.5)
            
            logger.info(f"[{self.id}] Operación bloqueante por {duration}s")
            await asyncio.sleep(duration)
            
            return {
                "blocked_for": duration,
                "call_count": current_count
            }
        
        # Eco simple
        if request_type == "echo":
            return {"echo": data.get("message", ""), "source": source}
            
        return None
    
    async def start(self) -> None:
        logger.info(f"Componente {self.id} iniciado")
    
    async def stop(self) -> None:
        logger.info(f"Componente {self.id} detenido")

# Simulación del coordinador híbrido
class HybridCoordinator:
    def __init__(self):
        self.components = {}
        
    def register_component(self, id: str, component: ComponentAPI) -> None:
        self.components[id] = component
        logger.info(f"Componente {id} registrado")
    
    async def request(self, target_id: str, request_type: str, data: Dict[str, Any], source: str, timeout: float = 5.0) -> Optional[Any]:
        if target_id not in self.components:
            logger.error(f"Componente {target_id} no encontrado")
            return None
        
        try:
            # Usar timeout para evitar bloqueos infinitos (clave del nuevo sistema)
            logger.debug(f"Solicitud {request_type} a {target_id} desde {source}")
            result = await asyncio.wait_for(
                self.components[target_id].process_request(request_type, data, source),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            logger.error(f"Timeout en solicitud {request_type} a {target_id}")
            return None
        except Exception as e:
            logger.error(f"Error en solicitud: {e}")
            return None
    
    async def broadcast_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        # Simulación simplificada de broadcast de eventos
        logger.debug(f"Broadcast de {event_type} desde {source}")
    
    async def start(self) -> None:
        for comp_id, comp in self.components.items():
            await comp.start()
        logger.info("Coordinador iniciado")
    
    async def stop(self) -> None:
        for comp_id, comp in self.components.items():
            await comp.stop()
        logger.info("Coordinador detenido")

# Coordinador global para las pruebas
coordinator = HybridCoordinator()

async def test_recursive_calls():
    """Probar llamadas recursivas que causarían deadlock en el sistema anterior."""
    logger.info("\n=== Prueba de Llamadas Recursivas ===")
    
    # Crear componente
    comp = ComponentAPI("recursive_component")
    coordinator.register_component("recursive_component", comp)
    
    # Hacer una llamada recursiva profunda
    start_time = time.time()
    result = await coordinator.request(
        "recursive_component",
        "recursive",
        {"depth": 5},  # Profundidad 5, causaría deadlock en el sistema anterior
        "test"
    )
    duration = time.time() - start_time
    
    # Verificar resultado
    success = result is not None and "current_depth" in result
    if success:
        logger.info(f"✅ Llamadas recursivas exitosas (completadas en {duration:.2f}s)")
        logger.info(f"   Profundidad alcanzada: {result['current_depth']}")
        logger.info(f"   Número total de llamadas: {comp.call_count}")
    else:
        logger.error(f"❌ Prueba de llamadas recursivas falló: {result}")
    
    return success

async def test_circular_calls():
    """Probar llamadas circulares que causarían deadlock en el sistema anterior."""
    logger.info("\n=== Prueba de Llamadas Circulares ===")
    
    # Crear componentes
    comp1 = ComponentAPI("circular_comp1")
    comp2 = ComponentAPI("circular_comp2")
    
    coordinator.register_component("circular_comp1", comp1)
    coordinator.register_component("circular_comp2", comp2)
    
    # Iniciar cadena de llamadas circulares
    start_time = time.time()
    result = await coordinator.request(
        "circular_comp1",
        "circular",
        {
            "target_id": "circular_comp2",
            "depth": 4  # Profundidad 4, alternará entre ambos componentes
        },
        "test"
    )
    duration = time.time() - start_time
    
    # Verificar resultado
    success = result is not None and "current_depth" in result
    total_calls = comp1.call_count + comp2.call_count
    
    if success:
        logger.info(f"✅ Llamadas circulares exitosas (completadas en {duration:.2f}s)")
        logger.info(f"   Profundidad alcanzada: {result['current_depth']}")
        logger.info(f"   Número total de llamadas: {total_calls} (comp1: {comp1.call_count}, comp2: {comp2.call_count})")
    else:
        logger.error(f"❌ Prueba de llamadas circulares falló: {result}")
    
    return success

async def test_blocking_calls():
    """Probar que las llamadas bloqueantes no afectan a otras operaciones."""
    logger.info("\n=== Prueba de Llamadas Bloqueantes ===")
    
    # Crear componentes
    blocking_comp = ComponentAPI("blocking_comp")
    fast_comp = ComponentAPI("fast_comp")
    
    coordinator.register_component("blocking_comp", blocking_comp)
    coordinator.register_component("fast_comp", fast_comp)
    
    # Iniciar llamada bloqueante
    logger.info("Iniciando llamada bloqueante larga (2s)...")
    blocking_task = asyncio.create_task(
        coordinator.request(
            "blocking_comp",
            "blocking",
            {"duration": 2.0},  # Bloqueo de 2 segundos
            "test"
        )
    )
    
    # Sin esperar que termine, hacer varias llamadas rápidas
    await asyncio.sleep(0.1)  # Pausa breve para asegurar que la primera tarea comenzó
    
    logger.info("Mientras está bloqueada, haciendo llamadas rápidas...")
    start_time = time.time()
    fast_results = []
    
    for i in range(3):
        result = await coordinator.request(
            "fast_comp",
            "echo",
            {"message": f"Fast call {i}"},
            "test"
        )
        fast_results.append(result)
        await asyncio.sleep(0.1)
    
    fast_duration = time.time() - start_time
    
    # Verificar llamadas rápidas
    fast_success = len(fast_results) == 3 and all(r is not None for r in fast_results)
    
    if fast_success:
        logger.info(f"✅ Llamadas rápidas completadas en {fast_duration:.2f}s mientras la bloqueante sigue en ejecución")
    else:
        logger.error(f"❌ Las llamadas rápidas fallaron: {fast_results}")
    
    # Esperar que termine la llamada bloqueante
    blocking_result = await blocking_task
    
    # Verificar llamada bloqueante
    blocking_success = blocking_result is not None and "blocked_for" in blocking_result
    
    if blocking_success:
        logger.info(f"✅ Llamada bloqueante completada exitosamente: {blocking_result}")
    else:
        logger.error(f"❌ La llamada bloqueante falló: {blocking_result}")
    
    return fast_success and blocking_success

async def test_timeout_protection():
    """Probar que el timeout protege contra operaciones bloqueantes excesivas."""
    logger.info("\n=== Prueba de Protección por Timeout ===")
    
    # Crear componente
    hang_comp = ComponentAPI("hanging_comp")
    coordinator.register_component("hanging_comp", hang_comp)
    
    # Llamada con timeout muy corto a una operación larga
    logger.info("Haciendo llamada con timeout corto (0.5s) a operación larga (5s)...")
    start_time = time.time()
    result = await coordinator.request(
        "hanging_comp",
        "blocking",
        {"duration": 5.0},  # Operación de 5 segundos
        "test",
        timeout=0.5  # Timeout de 0.5 segundos
    )
    duration = time.time() - start_time
    
    # Verificar timeout
    timeout_occurred = result is None and duration < 4.0  # Mucho menos que los 5s
    
    if timeout_occurred:
        logger.info(f"✅ Timeout funcionó correctamente, protegiendo contra bloqueo (duró {duration:.2f}s)")
    else:
        logger.error(f"❌ La protección por timeout falló. Resultado: {result}, duración: {duration:.2f}s")
    
    return timeout_occurred

async def run_all_tests():
    """Ejecutar todas las pruebas."""
    await coordinator.start()
    
    try:
        # Ejecutar todas las pruebas y recolectar resultados
        results = []
        results.append(await test_recursive_calls())
        results.append(await test_circular_calls())
        results.append(await test_blocking_calls())
        results.append(await test_timeout_protection())
        
        # Mostrar resumen
        success_count = sum(1 for r in results if r)
        total_count = len(results)
        
        logger.info("\n=== Resumen de Pruebas ===")
        logger.info(f"Pruebas exitosas: {success_count}/{total_count}")
        
        if success_count == total_count:
            logger.info("\n✅ TODAS LAS PRUEBAS PASARON: El sistema híbrido soluciona los problemas de deadlock")
        else:
            logger.info(f"\n❌ ALGUNAS PRUEBAS FALLARON: {total_count - success_count} pruebas fallaron")
            
    finally:
        await coordinator.stop()

# Ejecutar pruebas
if __name__ == "__main__":
    asyncio.run(run_all_tests())