"""
Test de integración para las características de resiliencia del sistema Genesis.

Este script prueba la integración de:
1. Sistema de Reintentos Adaptativo (retry_adaptive.py)
2. Arquitectura de Circuit Breaker (circuit_breaker.py)
3. Sistema de Checkpointing y Safe Mode (checkpoint_recovery.py)

El objetivo es verificar que estas características funcionen correctamente
y mejoren la resiliencia del sistema Genesis ante fallos.
"""

import asyncio
import logging
import os
import random
import tempfile
import time
from typing import Dict, Any, Optional, List, Tuple

# Importar módulos de resiliencia
from genesis.core.retry_adaptive import with_retry, RetryConfig, AdaptiveRetry
from genesis.core.circuit_breaker import with_circuit_breaker, CircuitState, registry
from genesis.core.checkpoint_recovery import (
    CheckpointManager, Checkpoint, StateMetadata, CheckpointType,
    SafeModeManager, RecoveryMode, RecoveryManager
)

# Configurar logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("resilience_test")

# Configuración de prueba de integración
CHECKPOINT_DIR = "./test_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Componente simulado para pruebas
class TestComponent:
    """Componente de prueba que simula fallas aleatorias."""
    
    def __init__(self, component_id: str, fail_rate: float = 0.3):
        """
        Inicializar componente.
        
        Args:
            component_id: ID único del componente
            fail_rate: Tasa de fallo simulada (0.0-1.0)
        """
        self.component_id = component_id
        self.fail_rate = fail_rate
        self.data: Dict[str, Any] = {}
        self.call_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.last_operation = None
        
        # Inicializar checkpoint manager
        self.checkpoint_mgr = CheckpointManager(
            component_id=component_id,
            checkpoint_dir=os.path.join(CHECKPOINT_DIR, component_id),
            checkpoint_interval=150.0,  # 150ms
            checkpoint_type=CheckpointType.DISK
        )
    
    @with_retry(base_delay=0.1, max_retries=3, jitter_factor=0.2)
    @with_circuit_breaker(failure_threshold=3, recovery_timeout=5.0)
    async def process_operation(self, operation: str, value: Any) -> Dict[str, Any]:
        """
        Procesar una operación con retry y circuit breaker.
        
        Args:
            operation: Tipo de operación
            value: Valor para la operación
            
        Returns:
            Resultado de la operación
            
        Raises:
            Exception: Si la operación falla
        """
        self.call_count += 1
        self.last_operation = operation
        
        # Simular fallo aleatorio
        if random.random() < self.fail_rate:
            self.failure_count += 1
            logger.warning(f"Componente {self.component_id}: Simulando fallo en {operation}")
            raise Exception(f"Fallo simulado en {operation}")
        
        # Simular operación exitosa
        await asyncio.sleep(0.1)  # Simular tiempo de procesamiento
        
        if operation == "set":
            key = value.get("key")
            if key:
                self.data[key] = value.get("value")
                
        elif operation == "get":
            key = value.get("key")
            if key and key in self.data:
                return {"key": key, "value": self.data[key]}
                
        elif operation == "list":
            return {"items": list(self.data.items())}
            
        elif operation == "clear":
            self.data.clear()
        
        # Crear checkpoint después de operación exitosa
        await self.checkpoint_mgr.checkpoint(self.data)
        
        self.success_count += 1
        return {"status": "success", "operation": operation}
    
    async def restore_from_checkpoint(self) -> bool:
        """
        Restaurar estado desde el último checkpoint.
        
        Returns:
            True si se restauró correctamente
        """
        result = await self.checkpoint_mgr.restore()
        if result:
            state, metadata = result
            self.data = state
            logger.info(f"Componente {self.component_id}: Restaurado desde checkpoint {metadata.checkpoint_id}")
            return True
        
        logger.warning(f"Componente {self.component_id}: No se pudo restaurar desde checkpoint")
        return False

# Función principal de prueba
async def test_resilience_features() -> Dict[str, Any]:
    """
    Probar características de resiliencia integradas.
    
    Returns:
        Diccionario con resultados de las pruebas
    """
    # Resultado global
    results = {
        "tests_total": 0,
        "tests_passed": 0,
        "components": {},
        "checkpoint_tests": {"total": 0, "passed": 0},
        "retry_tests": {"total": 0, "passed": 0},
        "circuit_breaker_tests": {"total": 0, "passed": 0},
        "safe_mode_tests": {"total": 0, "passed": 0},
        "integration_tests": {"total": 0, "passed": 0}
    }
    
    # 1. Crear componentes de prueba
    components = {
        "data_processor": TestComponent("data_processor", fail_rate=0.3),
        "market_analyzer": TestComponent("market_analyzer", fail_rate=0.5),
        "risk_manager": TestComponent("risk_manager", fail_rate=0.2)
    }
    
    # Registrar resultados iniciales
    for comp_id, component in components.items():
        results["components"][comp_id] = {
            "call_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "checkpoint_restore_success": False
        }
    
    # 2. Configurar Recovery Manager
    recovery_mgr = RecoveryManager(
        checkpoint_dir=CHECKPOINT_DIR,
        essential_components=["data_processor", "risk_manager"]
    )
    
    # 3. Prueba de Retry Adaptativo
    logger.info("=== Prueba de Retry Adaptativo ===")
    for comp_id, component in components.items():
        test_data = [
            ("set", {"key": f"key1_{comp_id}", "value": f"value1_{comp_id}"}),
            ("set", {"key": f"key2_{comp_id}", "value": f"value2_{comp_id}"}),
            ("get", {"key": f"key1_{comp_id}"})
        ]
        
        # Ejecutar operaciones con reintentos
        for op, value in test_data:
            try:
                results["retry_tests"]["total"] += 1
                await component.process_operation(op, value)
                results["retry_tests"]["passed"] += 1
            except Exception as e:
                logger.error(f"Error final en {comp_id} - {op}: {e}")
    
    # 4. Prueba de Circuit Breaker
    logger.info("=== Prueba de Circuit Breaker ===")
    # Forzar apertura del circuit breaker de market_analyzer
    market_analyzer = components["market_analyzer"]
    market_analyzer.fail_rate = 1.0  # 100% de fallos
    
    # Ejecutar operaciones hasta que se abra el circuit breaker
    for i in range(10):
        try:
            results["circuit_breaker_tests"]["total"] += 1
            await market_analyzer.process_operation("set", {"key": f"forced_fail_{i}", "value": "test"})
        except Exception:
            # Esperar error
            pass
        
        # Verificar si el circuit breaker se abrió
        circuit_breaker = market_analyzer.process_operation.circuit_breaker  # type: ignore
        if circuit_breaker.state == CircuitState.OPEN:
            logger.info(f"Circuit breaker abierto después de {i+1} intentos")
            results["circuit_breaker_tests"]["passed"] += 1
            break
    
    # 5. Prueba de Checkpointing y Recuperación
    logger.info("=== Prueba de Checkpointing y Recuperación ===")
    # Guardar datos y crear checkpoints
    for comp_id, component in components.items():
        if comp_id != "market_analyzer":  # Excluir el componente con CB abierto
            # Guardar algunos datos
            for i in range(5):
                try:
                    results["checkpoint_tests"]["total"] += 1
                    await component.process_operation("set", 
                                                     {"key": f"checkpoint_key_{i}", 
                                                      "value": f"checkpoint_value_{comp_id}_{i}"})
                    results["checkpoint_tests"]["passed"] += 1
                except Exception:
                    pass
    
    # Simular reinicio y restauración
    logger.info("Simulando reinicio de componentes...")
    for comp_id, component in components.items():
        if comp_id != "market_analyzer":  # Excluir el componente con CB abierto
            # Crear nuevo componente (simular reinicio)
            new_component = TestComponent(comp_id, fail_rate=0.1)
            
            # Restaurar desde checkpoint
            success = await new_component.restore_from_checkpoint()
            results["components"][comp_id]["checkpoint_restore_success"] = success
            
            if success:
                # Verificar datos restaurados
                try:
                    result = await new_component.process_operation("get", {"key": "checkpoint_key_1"})
                    if result.get("value") == f"checkpoint_value_{comp_id}_1":
                        logger.info(f"Datos restaurados correctamente para {comp_id}")
                        results["integration_tests"]["passed"] += 1
                except Exception as e:
                    logger.error(f"Error al verificar datos restaurados para {comp_id}: {e}")
                
                # Reemplazar componente por el restaurado
                components[comp_id] = new_component
            
            results["integration_tests"]["total"] += 1
    
    # 6. Prueba de Safe Mode
    logger.info("=== Prueba de Safe Mode ===")
    safe_mode_mgr = SafeModeManager(
        essential_components=["data_processor", "risk_manager"]
    )
    
    # Activar Safe Mode
    results["safe_mode_tests"]["total"] += 1
    await safe_mode_mgr.activate_safe_mode("Prueba de activación")
    
    if safe_mode_mgr.current_mode == RecoveryMode.SAFE:
        logger.info("Safe mode activado correctamente")
        results["safe_mode_tests"]["passed"] += 1
        
        # Verificar operaciones permitidas en modo seguro
        allowed_ops = 0
        for comp_id, component in components.items():
            is_essential = safe_mode_mgr.is_component_essential(comp_id)
            if is_essential:
                # Componentes esenciales deben poder operar
                try:
                    results["integration_tests"]["total"] += 1
                    await component.process_operation("set", {"key": "safe_mode_test", "value": "test"})
                    allowed_ops += 1
                    results["integration_tests"]["passed"] += 1
                    logger.info(f"Componente esencial {comp_id} puede operar en modo seguro")
                except Exception as e:
                    logger.error(f"Error inesperado en componente esencial {comp_id}: {e}")
        
        # Desactivar Safe Mode
        await safe_mode_mgr.deactivate_safe_mode()
        logger.info(f"Safe mode desactivado. {allowed_ops} operaciones permitidas en modo seguro")
    
    # 7. Resultados finales
    for comp_id, component in components.items():
        results["components"][comp_id].update({
            "call_count": component.call_count,
            "success_count": component.success_count,
            "failure_count": component.failure_count,
            "last_operation": component.last_operation
        })
    
    # Estadísticas globales
    results["tests_total"] = (results["retry_tests"]["total"] + 
                             results["circuit_breaker_tests"]["total"] +
                             results["checkpoint_tests"]["total"] + 
                             results["safe_mode_tests"]["total"] +
                             results["integration_tests"]["total"])
    
    results["tests_passed"] = (results["retry_tests"]["passed"] + 
                               results["circuit_breaker_tests"]["passed"] +
                               results["checkpoint_tests"]["passed"] + 
                               results["safe_mode_tests"]["passed"] +
                               results["integration_tests"]["passed"])
    
    # Limpiar checkpoints de prueba
    logger.info("Limpiando checkpoints de prueba...")
    
    return results

# Ejecutar prueba
async def main():
    start_time = time.time()
    
    try:
        results = await test_resilience_features()
        
        # Mostrar resultados
        logger.info("=== Resultados de las pruebas ===")
        
        success_rate = results["tests_passed"] / results["tests_total"] * 100 if results["tests_total"] > 0 else 0
        logger.info(f"Total de pruebas: {results['tests_total']}")
        logger.info(f"Pruebas exitosas: {results['tests_passed']}")
        logger.info(f"Tasa de éxito: {success_rate:.1f}%")
        
        # Mostrar detalles por sección
        for section in ["retry_tests", "circuit_breaker_tests", "checkpoint_tests", "safe_mode_tests", "integration_tests"]:
            section_total = results[section]["total"]
            section_passed = results[section]["passed"]
            section_rate = section_passed / section_total * 100 if section_total > 0 else 0
            
            logger.info(f"{section}: {section_passed}/{section_total} ({section_rate:.1f}%)")
        
        logger.info("=== Detalles por componente ===")
        for comp_id, stats in results["components"].items():
            logger.info(f"{comp_id}:")
            logger.info(f"  Llamadas: {stats['call_count']}")
            logger.info(f"  Éxitos: {stats['success_count']}")
            logger.info(f"  Fallos: {stats['failure_count']}")
            logger.info(f"  Restauración: {'Éxito' if stats['checkpoint_restore_success'] else 'Fallido/No intentado'}")
        
    except Exception as e:
        logger.error(f"Error en prueba: {e}")
    
    finally:
        # Tiempo total
        elapsed = time.time() - start_time
        logger.info(f"Tiempo total de ejecución: {elapsed:.2f} segundos")

if __name__ == "__main__":
    asyncio.run(main())