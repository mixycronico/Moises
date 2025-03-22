"""
Prueba integrada del sistema híbrido optimizado con todas las mejoras de resiliencia.

Este script prueba el funcionamiento del sistema híbrido optimizado implementado 
en genesis/core/genesis_hybrid_optimized.py, verificando que:
1. El backoff exponencial funcione correctamente en reintentos
2. El circuit breaker detecte y aísle componentes fallidos
3. El checkpointing permita recuperar estado tras fallos
4. El safe mode proteja componentes esenciales

La prueba incluye escenarios de fallos controlados para evaluar la resiliencia.
"""

import asyncio
import logging
import os
import shutil
import time
import random
from typing import Dict, Any, List, Optional, Set

from genesis.core.genesis_hybrid_optimized import (
    GenesisHybridCoordinator, ComponentAPI, TestComponent
)

# Configuración de logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_hybrid_optimized")

# Directorio para checkpoints
TEST_CHECKPOINT_DIR = "./test_checkpoints_hybrid"

# Componente que falla de manera controlada
class UnreliableComponent(ComponentAPI):
    """Componente que falla de manera controlada para pruebas de resiliencia."""
    
    def __init__(self, id: str, fail_rate: float = 0.3, fail_after: int = -1):
        """
        Inicializar componente no confiable.
        
        Args:
            id: Identificador del componente
            fail_rate: Tasa de fallos (0.0-1.0)
            fail_after: Fallar después de N llamadas exitosas, -1 para aleatorio
        """
        super().__init__(id)
        self.fail_rate = fail_rate
        self.fail_after = fail_after
        self.call_count = 0
        self.success_count = 0
        self.failure_count = 0
        
        # Para checkpointing
        self.data: Dict[str, Any] = {
            "counters": {
                "api_calls": 0,
                "local_events": 0,
                "external_events": 0
            },
            "values": {}
        }
    
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """Procesar solicitud con fallos aleatorios."""
        await super().process_request(request_type, data, source)
        
        self.call_count += 1
        self.data["counters"]["api_calls"] += 1
        
        # Actualizar estado para checkpointing
        self.state["data"] = self.data
        self.state["last_request"] = {
            "type": request_type,
            "data": data,
            "source": source,
            "timestamp": time.time()
        }
        
        # Determinar si debe fallar
        should_fail = False
        if self.fail_after > 0 and self.success_count >= self.fail_after:
            should_fail = True
        elif self.fail_after < 0 and random.random() < self.fail_rate:
            should_fail = True
        
        if should_fail:
            self.failure_count += 1
            logger.warning(f"Componente {self.id} fallando intencionalmente: {request_type}")
            raise Exception(f"Fallo simulado en {self.id}")
        
        # Procesar solicitud exitosa
        self.success_count += 1
        
        if request_type == "get":
            key = data.get("key")
            return self.data["values"].get(key)
            
        elif request_type == "set":
            key = data.get("key")
            value = data.get("value")
            if key:
                self.data["values"][key] = value
                return {"status": "success", "key": key}
                
        elif request_type == "delete":
            key = data.get("key")
            if key in self.data["values"]:
                del self.data["values"][key]
                return {"status": "success", "key": key}
                
        elif request_type == "list":
            return {"keys": list(self.data["values"].keys())}
            
        elif request_type == "stats":
            return {
                "counters": self.data["counters"],
                "call_count": self.call_count,
                "success_count": self.success_count,
                "failure_count": self.failure_count
            }
        
        return {"status": "unknown_request"}
    
    async def on_local_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Manejar evento local."""
        await super().on_local_event(event_type, data, source)
        
        self.data["counters"]["local_events"] += 1
        
        # Simular procesamiento
        await asyncio.sleep(0.01)
        
        # Actualizar estado para checkpointing
        self.state["data"] = self.data
        self.state["last_local_event"] = {
            "type": event_type,
            "source": source,
            "timestamp": time.time()
        }
    
    async def on_external_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Manejar evento externo."""
        await super().on_external_event(event_type, data, source)
        
        self.data["counters"]["external_events"] += 1
        
        # Simular procesamiento
        await asyncio.sleep(0.01)
        
        # Actualizar estado para checkpointing
        self.state["data"] = self.data
        self.state["last_external_event"] = {
            "type": event_type,
            "source": source,
            "timestamp": time.time()
        }

# Prueba del sistema híbrido optimizado
async def test_hybrid_optimized() -> Dict[str, Any]:
    """
    Probar el sistema híbrido optimizado con escenarios de resiliencia.
    
    Returns:
        Resultados de la prueba
    """
    # Limpiar directorio de checkpoints
    if os.path.exists(TEST_CHECKPOINT_DIR):
        shutil.rmtree(TEST_CHECKPOINT_DIR)
    os.makedirs(TEST_CHECKPOINT_DIR, exist_ok=True)
    
    # Resultados
    results = {
        "tests": 0,
        "passed": 0,
        "retry_test": {"success": False, "details": {}},
        "circuit_breaker_test": {"success": False, "details": {}},
        "checkpointing_test": {"success": False, "details": {}},
        "recovery_test": {"success": False, "details": {}},
        "safe_mode_test": {"success": False, "details": {}}
    }
    
    try:
        # Crear coordinador con componentes esenciales
        coordinator = GenesisHybridCoordinator(
            host="localhost",
            port=8888,  # Puerto diferente para evitar conflictos
            checkpoint_dir=TEST_CHECKPOINT_DIR,
            essential_components=["essential_component"]
        )
        
        # Crear componentes
        components = {
            "essential_component": UnreliableComponent("essential_component", fail_rate=0.2),
            "unreliable_component": UnreliableComponent("unreliable_component", fail_rate=0.5),
            "stable_component": UnreliableComponent("stable_component", fail_rate=0.0)
        }
        
        # Registrar componentes
        for comp_id, component in components.items():
            coordinator.register_component(
                comp_id, 
                component,
                essential=(comp_id == "essential_component")
            )
        
        # Iniciar coordinador
        logger.info("Iniciando coordinador y componentes...")
        await coordinator.start()
        
        # Prueba 1: Sistema de Reintentos Adaptativo
        logger.info("=== Prueba 1: Sistema de Reintentos Adaptativo ===")
        results["tests"] += 1
        
        # Configurar componente para fallar después de 2 llamadas exitosas
        unreliable = components["unreliable_component"]
        unreliable.fail_rate = 0.0  # Desactivar fallos aleatorios
        unreliable.fail_after = 2   # Fallar después de 2 llamadas exitosas
        
        # Realizar varias llamadas (al menos 5)
        retry_ops = []
        for i in range(5):
            try:
                result = await coordinator.request(
                    "unreliable_component",
                    "set",
                    {"key": f"retry_key_{i}", "value": f"retry_value_{i}"},
                    "test"
                )
                retry_ops.append({"success": True, "result": result})
            except Exception as e:
                retry_ops.append({"success": False, "error": str(e)})
        
        # Verificar resultados: debería haber algunos éxitos a pesar de los fallos
        success_count = sum(1 for op in retry_ops if op["success"])
        
        if success_count >= 3:  # Al menos 3 de 5 deben ser exitosos
            logger.info(f"✓ Sistema de reintentos funcionando: {success_count}/5 operaciones exitosas")
            results["retry_test"]["success"] = True
            results["retry_test"]["details"] = {
                "total_ops": len(retry_ops),
                "success_ops": success_count,
                "component_failures": unreliable.failure_count
            }
            results["passed"] += 1
        else:
            logger.error(f"✗ Sistema de reintentos fallando: solo {success_count}/5 operaciones exitosas")
            results["retry_test"]["details"] = {
                "total_ops": len(retry_ops),
                "success_ops": success_count,
                "component_failures": unreliable.failure_count
            }
        
        # Prueba 2: Circuit Breaker
        logger.info("=== Prueba 2: Circuit Breaker ===")
        results["tests"] += 1
        
        # Configurar componente para fallar siempre
        unreliable.fail_rate = 1.0
        unreliable.fail_after = -1
        
        # Realizar varias llamadas hasta que el circuit breaker se abra
        cb_ops = []
        open_detected = False
        rejection_count = 0
        
        for i in range(10):  # Máximo 10 intentos
            try:
                result = await coordinator.request(
                    "unreliable_component",
                    "set",
                    {"key": f"cb_key_{i}", "value": f"cb_value_{i}"},
                    "test"
                )
                cb_ops.append({"success": True, "result": result})
            except Exception as e:
                cb_ops.append({"success": False, "error": str(e)})
                
            # Verificar si el componente está marcado como no saludable
            if not unreliable.healthy:
                open_detected = True
                break
        
        # Verificar resultados: el circuit breaker debería abrirse
        if open_detected:
            logger.info(f"✓ Circuit Breaker abierto después de {len(cb_ops)} operaciones")
            results["circuit_breaker_test"]["success"] = True
            results["circuit_breaker_test"]["details"] = {
                "total_ops": len(cb_ops),
                "operations_until_open": len(cb_ops),
                "component_unhealthy": not unreliable.healthy
            }
            results["passed"] += 1
        else:
            logger.error(f"✗ Circuit Breaker no se abrió después de {len(cb_ops)} operaciones")
            results["circuit_breaker_test"]["details"] = {
                "total_ops": len(cb_ops),
                "component_unhealthy": not unreliable.healthy
            }
        
        # Prueba 3: Checkpointing
        logger.info("=== Prueba 3: Checkpointing ===")
        results["tests"] += 1
        
        # Usar componente estable para esta prueba
        stable = components["stable_component"]
        
        # Realizar operaciones para crear datos
        for i in range(5):
            await coordinator.request(
                "stable_component",
                "set",
                {"key": f"checkpoint_key_{i}", "value": f"checkpoint_value_{i}"},
                "test"
            )
        
        # Verificar datos antes de checkpoint
        get_result = await coordinator.request(
            "stable_component",
            "list",
            {},
            "test"
        )
        
        keys_before = get_result.get("keys", [])
        
        # Simular cierre y reinicio (sin perder datos gracias al checkpointing)
        logger.info("Simulando reinicio del componente...")
        
        # Crear nueva instancia
        new_stable = UnreliableComponent("stable_component", fail_rate=0.0)
        
        # Reemplazar en coordinador
        coordinator.components["stable_component"] = new_stable
        
        # Configurar checkpointing
        new_stable.setup_checkpointing(
            checkpoint_dir=coordinator.checkpoint_dir,
            interval_ms=150.0
        )
        
        # Esperar a que se restaure desde checkpoint
        await asyncio.sleep(2.0)
        
        # Verificar datos después de restauración
        get_result_after = await coordinator.request(
            "stable_component",
            "list",
            {},
            "test"
        )
        
        keys_after = get_result_after.get("keys", [])
        
        # Verificar resultados: los datos deberían persistir
        if set(keys_before) == set(keys_after) and len(keys_after) >= 5:
            logger.info(f"✓ Checkpointing funcionando: {len(keys_after)} claves restauradas")
            results["checkpointing_test"]["success"] = True
            results["checkpointing_test"]["details"] = {
                "keys_before": keys_before,
                "keys_after": keys_after
            }
            results["passed"] += 1
        else:
            logger.error(f"✗ Checkpointing fallando: claves antes {keys_before}, después {keys_after}")
            results["checkpointing_test"]["details"] = {
                "keys_before": keys_before,
                "keys_after": keys_after
            }
        
        # Prueba 4: Recuperación automática
        logger.info("=== Prueba 4: Recuperación automática ===")
        results["tests"] += 1
        
        # Marcar componente esencial como fallido
        essential = components["essential_component"]
        essential.healthy = False
        
        # Esperar a la recuperación automática
        logger.info("Esperando recuperación automática...")
        await asyncio.sleep(3.0)  # Dar tiempo al monitor para detectar y recuperar
        
        # Verificar si el componente fue recuperado
        if essential.healthy:
            logger.info("✓ Componente esencial recuperado automáticamente")
            
            # Verificar funcionalidad restaurada
            try:
                result = await coordinator.request(
                    "essential_component",
                    "set",
                    {"key": "recovery_test", "value": "after_recovery"},
                    "test"
                )
                
                if result:
                    results["recovery_test"]["success"] = True
                    results["recovery_test"]["details"] = {
                        "component_healthy": essential.healthy,
                        "operation_successful": True
                    }
                    results["passed"] += 1
            except Exception as e:
                logger.error(f"Error al probar componente recuperado: {e}")
                results["recovery_test"]["details"] = {
                    "component_healthy": essential.healthy,
                    "operation_successful": False,
                    "error": str(e)
                }
        else:
            logger.error("✗ Componente esencial no recuperado")
            results["recovery_test"]["details"] = {
                "component_healthy": essential.healthy
            }
        
        # Prueba 5: Safe Mode
        logger.info("=== Prueba 5: Safe Mode ===")
        results["tests"] += 1
        
        # Activar Safe Mode manualmente
        await coordinator.safe_mode_mgr.activate_safe_mode("Prueba manual")
        
        # Verificar estado Safe Mode
        if coordinator.safe_mode_mgr.current_mode.name == "SAFE":
            logger.info("✓ Safe Mode activado correctamente")
            
            # Verificar comportamiento en Safe Mode
            # 1. Componente esencial debe seguir funcionando
            essential_result = await coordinator.request(
                "essential_component",
                "set",
                {"key": "safe_mode_test", "value": "from_essential"},
                "test"
            )
            
            # 2. Componente no esencial debe ser rechazado
            non_essential_result = await coordinator.request(
                "stable_component",
                "set",
                {"key": "safe_mode_test", "value": "from_non_essential"},
                "test"
            )
            
            if essential_result is not None and non_essential_result is None:
                logger.info("✓ Safe Mode protege correctamente componentes esenciales")
                results["safe_mode_test"]["success"] = True
                results["safe_mode_test"]["details"] = {
                    "mode": coordinator.safe_mode_mgr.current_mode.name,
                    "essential_operation": essential_result is not None,
                    "non_essential_operation": non_essential_result is None
                }
                results["passed"] += 1
            else:
                logger.error("✗ Safe Mode no protege correctamente: essential_result={essential_result}, non_essential_result={non_essential_result}")
                results["safe_mode_test"]["details"] = {
                    "mode": coordinator.safe_mode_mgr.current_mode.name,
                    "essential_operation": essential_result is not None,
                    "non_essential_operation": non_essential_result is None
                }
        else:
            logger.error("✗ Safe Mode no se activó correctamente")
            results["safe_mode_test"]["details"] = {
                "mode": coordinator.safe_mode_mgr.current_mode.name
            }
        
        # Desactivar Safe Mode
        await coordinator.safe_mode_mgr.deactivate_safe_mode()
        
    except Exception as e:
        logger.error(f"Error en prueba: {e}")
    
    finally:
        # Intentar detener coordinador
        try:
            await coordinator.stop()
        except Exception as e:
            logger.error(f"Error al detener coordinador: {e}")
        
        # Limpiar directorio de checkpoints
        if os.path.exists(TEST_CHECKPOINT_DIR):
            shutil.rmtree(TEST_CHECKPOINT_DIR)
    
    # Calcular resultado global
    results["success_rate"] = (results["passed"] / results["tests"]) * 100 if results["tests"] > 0 else 0
    
    return results

async def main():
    """Función principal."""
    start_time = time.time()
    
    logger.info("Iniciando prueba del sistema híbrido optimizado...")
    results = await test_hybrid_optimized()
    
    # Mostrar resultados
    logger.info("\n=== Resultados de la prueba ===")
    logger.info(f"Total: {results['passed']}/{results['tests']} pruebas exitosas ({results['success_rate']:.1f}%)")
    
    # Mostrar detalles por prueba
    for test_name in ["retry_test", "circuit_breaker_test", "checkpointing_test", "recovery_test", "safe_mode_test"]:
        test_result = results[test_name]
        logger.info(f"{test_name}: {'✓' if test_result['success'] else '✗'}")
    
    # Tiempo total
    elapsed = time.time() - start_time
    logger.info(f"Tiempo total de ejecución: {elapsed:.2f} segundos")

if __name__ == "__main__":
    asyncio.run(main())