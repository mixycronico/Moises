"""
Test simple para verificar solución a deadlocks en el sistema híbrido.

Este script demuestra cómo el sistema híbrido soluciona problemas de deadlocks.
"""

import asyncio
import logging
import sys
import time
from typing import Dict, Any, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("deadlock_test")

# Simulación simple del sistema híbrido
class Component:
    """Componente básico para el test."""
    
    def __init__(self, id: str):
        self.id = id
        self.call_count = 0
    
    async def handle_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """Manejar una solicitud."""
        self.call_count += 1
        logger.info(f"[{self.id}] Solicitud: {request_type} de {source}")
        
        if request_type == "echo":
            return {"echo": data.get("message", ""), "source": source}
        
        elif request_type == "recursive":
            # Llamada recursiva que causaría deadlock en el sistema antiguo
            depth = data.get("depth", 1)
            if depth > 1:
                # Hacer llamada recursiva
                result = await engine.request(
                    self.id,  # Llamada a sí mismo
                    "recursive",
                    {"depth": depth - 1},
                    source
                )
                return {"depth": depth, "next": result}
            return {"depth": depth, "message": "Base case reached"}
        
        elif request_type == "sleep":
            # Operación bloqueante
            duration = data.get("duration", 0.5)
            logger.info(f"[{self.id}] Durmiendo por {duration}s")
            await asyncio.sleep(duration)
            return {"slept": duration}
        
        return {"status": "unknown_request"}

class HybridEngine:
    """Motor híbrido simplificado."""
    
    def __init__(self):
        self.components = {}
    
    def register(self, component_id: str, component: Component) -> None:
        """Registrar un componente."""
        self.components[component_id] = component
        logger.info(f"Componente {component_id} registrado")
    
    async def request(self, target_id: str, request_type: str, 
                     data: Dict[str, Any], source: str, 
                     timeout: float = 5.0) -> Optional[Any]:
        """
        Realizar una solicitud directa con timeout.
        
        El timeout es clave para evitar deadlocks.
        """
        if target_id not in self.components:
            logger.error(f"Componente {target_id} no encontrado")
            return None
        
        try:
            # Usar timeout para evitar bloqueos indefinidos
            result = await asyncio.wait_for(
                self.components[target_id].handle_request(request_type, data, source),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            logger.error(f"Timeout en solicitud {request_type} a {target_id}")
            return None
        except Exception as e:
            logger.error(f"Error en solicitud: {e}")
            return None

# Motor global para pruebas
engine = HybridEngine()

async def test_simple_request():
    """Prueba solicitud básica."""
    logger.info("\n== Prueba Solicitud Simple ==")
    
    # Crear componentes
    comp1 = Component("comp1")
    comp2 = Component("comp2")
    
    # Registrar componentes
    engine.register("comp1", comp1)
    engine.register("comp2", comp2)
    
    # Hacer solicitud
    logger.info("Enviando solicitud de eco...")
    result = await engine.request(
        "comp2", 
        "echo", 
        {"message": "Hola desde comp1"}, 
        "comp1"
    )
    
    # Verificar resultado
    if result and result.get("echo") == "Hola desde comp1":
        logger.info(f"✅ Solicitud exitosa: {result}")
        return True
    else:
        logger.error(f"❌ Solicitud fallida: {result}")
        return False

async def test_recursive_call():
    """
    Prueba llamada recursiva que causaría deadlock en sistema antiguo.
    
    En el sistema antiguo, cuando un componente se llamaba a sí mismo,
    se producía un deadlock porque esperaba su propia respuesta.
    """
    logger.info("\n== Prueba Llamada Recursiva ==")
    
    # Crear componente
    comp = Component("recursive_comp")
    
    # Registrar componente
    engine.register("recursive_comp", comp)
    
    # Hacer solicitud recursiva
    logger.info("Enviando solicitud recursiva (profundidad 5)...")
    start_time = time.time()
    result = await engine.request(
        "recursive_comp", 
        "recursive", 
        {"depth": 5},  # Profundidad 5
        "test"
    )
    duration = time.time() - start_time
    
    # Verificar resultado
    if result and "depth" in result and result["depth"] == 5:
        logger.info(f"✅ Llamada recursiva exitosa (completada en {duration:.2f}s)")
        return True
    else:
        logger.error(f"❌ Llamada recursiva fallida: {result}")
        return False

async def test_timeout_protection():
    """
    Prueba que el timeout protege contra bloqueos indefinidos.
    
    En el sistema antiguo, una operación bloqueante podía hacer que 
    todo el sistema se detuviera.
    """
    logger.info("\n== Prueba Protección por Timeout ==")
    
    # Crear componente
    comp = Component("blocking_comp")
    
    # Registrar componente
    engine.register("blocking_comp", comp)
    
    # Hacer solicitud con timeout corto a operación larga
    logger.info("Enviando solicitud con timeout corto (0.2s) a operación larga (1s)...")
    start_time = time.time()
    result = await engine.request(
        "blocking_comp", 
        "sleep", 
        {"duration": 1.0},  # Dormir 1 segundo
        "test",
        timeout=0.2  # Timeout de 0.2 segundos
    )
    duration = time.time() - start_time
    
    # Verificar que ocurrió timeout y fue rápido
    if result is None and duration < 0.5:  # Significativamente menos que 1s
        logger.info(f"✅ Timeout funcionó correctamente, duración: {duration:.2f}s")
        return True
    else:
        logger.error(f"❌ Timeout falló, resultado: {result}, duración: {duration:.2f}s")
        return False

async def run_tests():
    """Ejecutar todas las pruebas."""
    results = []
    
    # Prueba 1: Solicitud simple
    results.append(await test_simple_request())
    
    # Prueba 2: Llamada recursiva (deadlock en sistema antiguo)
    results.append(await test_recursive_call())
    
    # Prueba 3: Protección por timeout
    results.append(await test_timeout_protection())
    
    # Mostrar resultados
    success = all(results)
    total = len(results)
    passed = sum(1 for r in results if r)
    
    logger.info("\n== Resumen de Pruebas ==")
    logger.info(f"Pruebas pasadas: {passed}/{total}")
    
    if success:
        logger.info("✅ TODAS LAS PRUEBAS PASARON - El sistema híbrido resuelve los problemas de deadlock")
    else:
        logger.error(f"❌ ALGUNAS PRUEBAS FALLARON - {total - passed} pruebas fallaron")

# Ejecutar pruebas
if __name__ == "__main__":
    asyncio.run(run_tests())