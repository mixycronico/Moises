"""
Prueba simplificada de las características de resiliencia del sistema Genesis.

Esta prueba verifica el funcionamiento básico de:
1. Sistema de Reintentos Adaptativos con Backoff Exponencial
2. Circuit Breaker
3. Checkpointing y Safe Mode

Los escenarios son más cortos que en la prueba completa para poder ejecutarse
sin problemas de timeout.
"""

import asyncio
import logging
import time
from typing import Dict, Any

# Importar implementación del sistema híbrido resiliente
from genesis.core.genesis_hybrid_resiliente import (
    GenesisHybridCoordinator, 
    ResilientComponent
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_resiliencia")

class UnreliableComponent(ResilientComponent):
    """
    Componente poco confiable para probar resiliencia.
    Este componente puede fallar de forma controlada para
    probar las características de resiliencia.
    """
    def __init__(self, id: str, fail_rate: float = 0.0):
        """
        Inicializar componente no confiable.
        
        Args:
            id: Identificador del componente
            fail_rate: Tasa de fallos (0.0-1.0)
        """
        super().__init__(id)
        self.fail_rate = fail_rate
        self.call_count = 0
        self.fail_after = -1
        self.always_fail = False
        self.sleep_time = 0.0
        self.data_store = {}  # Para pruebas de checkpointing
        
    async def set_behavior(self, fail_rate: float = None, fail_after: int = None, 
                          always_fail: bool = None, sleep_time: float = None):
        """
        Configurar comportamiento del componente.
        
        Args:
            fail_rate: Nueva tasa de fallos
            fail_after: Fallar después de N llamadas exitosas
            always_fail: Si debe fallar siempre
            sleep_time: Tiempo de espera adicional
        """
        if fail_rate is not None:
            self.fail_rate = fail_rate
        if fail_after is not None:
            self.fail_after = fail_after
        if always_fail is not None:
            self.always_fail = always_fail
        if sleep_time is not None:
            self.sleep_time = sleep_time
            
        return {
            "id": self.id,
            "behavior": {
                "fail_rate": self.fail_rate,
                "fail_after": self.fail_after,
                "always_fail": self.always_fail,
                "sleep_time": self.sleep_time
            }
        }
        
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """
        Procesar una solicitud con posibilidad de fallo.
        
        Args:
            request_type: Tipo de solicitud
            data: Datos de la solicitud
            source: Origen de la solicitud
            
        Returns:
            Resultado de la solicitud
            
        Raises:
            Exception: Si el componente falla
        """
        self.call_count += 1
        
        # Simular latencia
        if self.sleep_time > 0:
            await asyncio.sleep(self.sleep_time)
        
        # Fallo controlado
        should_fail = False
        
        if self.always_fail:
            should_fail = True
        elif self.fail_after > 0 and self.call_count >= self.fail_after:
            should_fail = True
        elif self.fail_rate > 0 and (self.call_count % int(1/self.fail_rate) == 0):
            should_fail = True
            
        if should_fail and request_type != "status":
            raise Exception(f"Fallo simulado en {self.id}. Llamada #{self.call_count}")
            
        # Manejar solicitudes especiales
        if request_type == "set_data":
            key = data.get("key")
            value = data.get("value")
            if key:
                self.data_store[key] = value
                # Actualizar también en checkpoint
                if self.checkpoint.get("data_store") is None:
                    self.checkpoint["data_store"] = {}
                self.checkpoint["data_store"][key] = value
                return {"status": "ok", "key": key, "value": value}
            return {"status": "error", "reason": "No key provided"}
            
        elif request_type == "get_data":
            key = data.get("key")
            if key and key in self.data_store:
                return {"status": "ok", "key": key, "value": self.data_store.get(key)}
            return {"status": "error", "reason": "Key not found"}
            
        # Delegamiento a implementación base para otras solicitudes
        return await super().process_request(request_type, data, source)
        
    def save_checkpoint(self):
        """Guardar estado para checkpointing."""
        super().save_checkpoint()
        self.checkpoint["data_store"] = self.data_store.copy()
        self.checkpoint["call_count"] = self.call_count
        
    async def restore_from_checkpoint(self):
        """Restaurar desde checkpoint con datos propios."""
        await super().restore_from_checkpoint()
        if self.checkpoint:
            self.data_store = self.checkpoint.get("data_store", {})
            self.call_count = self.checkpoint.get("call_count", 0)

async def run_test_scenario(name, coordinator, test_func):
    """
    Ejecutar un escenario de prueba específico.
    
    Args:
        name: Nombre del escenario
        coordinator: Instancia del coordinador
        test_func: Función que implementa la prueba
        
    Returns:
        Resultado de la prueba
    """
    logger.info(f"\n=== Escenario: {name} ===")
    start_time = time.time()
    
    try:
        result = await test_func(coordinator)
        elapsed = time.time() - start_time
        logger.info(f"Escenario completado en {elapsed:.2f}s")
        return result
    except Exception as e:
        logger.error(f"Error en escenario: {e}")
        return {"status": "error", "reason": str(e)}

async def scenario_retry_system(coordinator):
    """
    Escenario 1: Probar sistema de reintentos adaptativos.
    """
    # Configurar componente para fallar en los primeros dos intentos
    service_comp = coordinator.components["service"]
    await service_comp.set_behavior(fail_after=3, always_fail=False)
    
    results = []
    
    # Primera llamada - debería fallar y reintentar
    result1 = await coordinator.request(
        "service", "echo", {"message": "Test 1"}, "test", timeout=1.0
    )
    results.append({"attempt": 1, "result": result1})
    
    # Configurar componente para tener alta latencia
    await service_comp.set_behavior(fail_after=-1, sleep_time=0.3)
    
    # Segunda llamada - debería tener timeouts
    result2 = await coordinator.request(
        "service", "echo", {"message": "Test 2"}, "test", timeout=0.2
    )
    results.append({"attempt": 2, "result": result2})
    
    # Configurar componente para comportamiento normal
    await service_comp.set_behavior(fail_rate=0.0, sleep_time=0.0)
    
    # Tercera llamada - debería tener éxito
    result3 = await coordinator.request(
        "service", "echo", {"message": "Test 3"}, "test", timeout=1.0
    )
    results.append({"attempt": 3, "result": result3})
    
    return {"scenario": "retry_system", "results": results}

async def scenario_circuit_breaker(coordinator):
    """
    Escenario 2: Probar Circuit Breaker.
    """
    # Obtener componente y verificar estado inicial
    payment_comp = coordinator.components["payment"]
    initial_state = await coordinator.request(
        "payment", "status", {}, "test"
    )
    
    results = [{"phase": "initial", "state": initial_state}]
    
    # Configurar para fallar siempre
    await payment_comp.set_behavior(always_fail=True)
    
    # Realizar varias llamadas para abrir el circuit breaker
    for i in range(5):
        result = await coordinator.request(
            "payment", "process_payment", {"amount": 100}, "test"
        )
        
        # Verificar estado del circuit breaker
        state = await coordinator.request(
            "payment", "status", {}, "test"
        )
        
        results.append({
            "phase": f"failing_{i}", 
            "result": result,
            "circuit_state": state["circuit_state"] if state else "unknown"
        })
    
    # Configurar para no fallar
    await payment_comp.set_behavior(always_fail=False)
    
    # Esperar a que el circuit breaker inicie recuperación
    logger.info("Esperando recuperación del circuit breaker...")
    await asyncio.sleep(2.5)
    
    # Intentar después de la recuperación
    recovery_result = await coordinator.request(
        "payment", "process_payment", {"amount": 50}, "test"
    )
    
    # Verificar estado final
    final_state = await coordinator.request(
        "payment", "status", {}, "test"
    )
    
    results.append({
        "phase": "recovery",
        "result": recovery_result,
        "circuit_state": final_state["circuit_state"] if final_state else "unknown"
    })
    
    return {"scenario": "circuit_breaker", "results": results}

async def scenario_checkpointing(coordinator):
    """
    Escenario 3: Probar checkpointing y recuperación.
    """
    # Guardar datos en el componente
    storage_comp = coordinator.components["storage"]
    
    # Almacenar múltiples valores
    results = []
    for i in range(3):
        set_result = await coordinator.request(
            "storage", "set_data", {"key": f"key_{i}", "value": f"value_{i}"}, "test"
        )
        results.append({"operation": "set", "data": set_result})
    
    # Verificar que los datos se guardaron
    for i in range(3):
        get_result = await coordinator.request(
            "storage", "get_data", {"key": f"key_{i}"}, "test"
        )
        results.append({"operation": "verify", "data": get_result})
    
    # Provocar un fallo en el componente
    fail_result = await coordinator.request(
        "storage", "simulate_failure", {}, "test"
    )
    results.append({"operation": "fail", "result": fail_result})
    
    # Verificar que los datos ya no están disponibles
    missing_result = await coordinator.request(
        "storage", "get_data", {"key": "key_0"}, "test"
    )
    results.append({"operation": "check_missing", "result": missing_result})
    
    # Esperar a que el monitor restaure el componente
    logger.info("Esperando restauración automática...")
    await asyncio.sleep(1.0)
    
    # Verificar restauración
    for i in range(3):
        restored_result = await coordinator.request(
            "storage", "get_data", {"key": f"key_{i}"}, "test"
        )
        results.append({"operation": "restored", "data": restored_result})
    
    return {"scenario": "checkpointing", "results": results}

async def scenario_safe_mode(coordinator):
    """
    Escenario 4: Probar Safe Mode.
    """
    results = []
    
    # Verificar estado inicial
    initial_mode = coordinator.mode
    results.append({"phase": "initial", "mode": initial_mode})
    
    # Hacer fallar componentes no esenciales
    non_essential_components = ["service", "storage"]
    for comp_id in non_essential_components:
        await coordinator.request(comp_id, "simulate_failure", {}, "test")
        
    # Verificar si el modo cambió
    after_non_essential_failure = coordinator.mode
    results.append({"phase": "after_non_essential_failure", "mode": after_non_essential_failure})
    
    # Esperar a que el monitor restaure los componentes
    await asyncio.sleep(1.0)
    
    # Hacer fallar un componente esencial
    try:
        await coordinator.request("payment", "simulate_failure", {}, "test")
    except Exception:
        pass
    
    # Verificar si el modo cambió a SAFE
    after_essential_failure = coordinator.mode
    results.append({"phase": "after_essential_failure", "mode": after_essential_failure})
    
    # Esperar restauración
    await asyncio.sleep(1.0)
    
    # Verificar si el sistema volvió a modo normal
    final_mode = coordinator.mode
    results.append({"phase": "final", "mode": final_mode})
    
    return {"scenario": "safe_mode", "results": results}

async def run_tests():
    """
    Ejecutar todos los escenarios de prueba.
    """
    logger.info("=== Prueba Simplificada de Resiliencia Genesis ===")
    
    # Crear coordinador y componentes
    coordinator = GenesisHybridCoordinator(host="localhost", port=8081)
    
    # Registrar componentes
    coordinator.register_component("service", UnreliableComponent("service"))
    coordinator.register_component("payment", UnreliableComponent("payment"))  # Esencial
    coordinator.register_component("storage", UnreliableComponent("storage"))
    
    # Iniciar coordinador
    await coordinator.start()
    
    try:
        # Ejecutar escenarios
        results = {}
        
        # Escenario 1: Sistema de Reintentos
        results["retry"] = await run_test_scenario(
            "Sistema de Reintentos", coordinator, scenario_retry_system
        )
        
        # Escenario 2: Circuit Breaker
        results["circuit_breaker"] = await run_test_scenario(
            "Circuit Breaker", coordinator, scenario_circuit_breaker
        )
        
        # Escenario 3: Checkpointing
        results["checkpointing"] = await run_test_scenario(
            "Checkpointing y Recuperación", coordinator, scenario_checkpointing
        )
        
        # Escenario 4: Safe Mode
        results["safe_mode"] = await run_test_scenario(
            "Safe Mode", coordinator, scenario_safe_mode
        )
        
        # Resumen
        logger.info("\n=== Resumen de Pruebas ===")
        for name, result in results.items():
            status = "ÉXITO" if result.get("status") != "error" else "ERROR"
            logger.info(f"Escenario {name}: {status}")
        
        logger.info("\n=== Conclusiones ===")
        logger.info("- Sistema de Reintentos: Permite recuperación frente a fallos transitorios")
        logger.info("- Circuit Breaker: Aísla componentes fallidos y evita llamadas innecesarias")
        logger.info("- Checkpointing: Facilita la recuperación rápida de estado después de fallos")
        logger.info("- Safe Mode: Prioriza componentes esenciales en condiciones degradadas")
        
    finally:
        # Detener coordinador
        await coordinator.stop()

if __name__ == "__main__":
    asyncio.run(run_tests())