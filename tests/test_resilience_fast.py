"""
Test rápido de resiliencia para el sistema Genesis.

Esta versión tiene tiempos de espera reducidos para completarse rápidamente.
Prueba las tres características de resiliencia:
1. Sistema de Reintentos Adaptativo
2. Circuit Breaker
3. Checkpointing
"""

import asyncio
import logging
import time
import random
import os
import json
from typing import Dict, Any, List, Optional
from enum import Enum, auto

# Configurar logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_resilience_fast")

# ======================================
# SISTEMA DE REINTENTOS CON BACKOFF
# ======================================

async def with_retry(func, max_retries=2, base_delay=0.05):
    """Sistema de reintentos con backoff exponencial simplificado."""
    attempt = 0
    while True:
        try:
            return await func()
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            
            delay = base_delay * (2 ** (attempt - 1))
            jitter = random.uniform(0, 0.05)
            total_delay = delay + jitter
            
            logger.info(f"Reintento {attempt}/{max_retries} tras error: {e}. Esperando {total_delay:.2f}s")
            await asyncio.sleep(total_delay)

# ======================================
# CIRCUIT BREAKER
# ======================================

class CircuitState(Enum):
    """Estados del Circuit Breaker."""
    CLOSED = auto()    # Funcionamiento normal
    OPEN = auto()      # Circuito abierto, rechazan llamadas
    HALF_OPEN = auto() # Semi-abierto, permite algunas llamadas

class CircuitBreaker:
    """Circuit Breaker simplificado."""
    
    def __init__(self, name, failure_threshold=3, recovery_timeout=1.0):
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = 0
        logger.info(f"Circuit Breaker '{name}' creado en estado {self.state.name}")
    
    async def execute(self, func):
        """Ejecutar función con protección del Circuit Breaker."""
        # Si está abierto, verificar si debemos pasar a half-open
        if self.state == CircuitState.OPEN:
            if (time.time() - self.last_failure_time) > self.recovery_timeout:
                logger.info(f"Circuit Breaker '{self.name}' cambiando a HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                logger.warning(f"Circuit Breaker '{self.name}' abierto. Llamada rechazada.")
                return None
        
        try:
            result = await func()
            
            # Registrar éxito
            self.success_count += 1
            self.failure_count = 0
            
            # En HALF_OPEN, cerrar el circuito si se alcanzan 2 éxitos
            if self.state == CircuitState.HALF_OPEN and self.success_count >= 2:
                logger.info(f"Circuit Breaker '{self.name}' cambiando a CLOSED")
                self.state = CircuitState.CLOSED
                
            return result
            
        except Exception as e:
            # Registrar fallo
            self.failure_count += 1
            self.success_count = 0
            self.last_failure_time = time.time()
            
            # Abrir circuito si excedemos umbral
            if self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
                logger.warning(f"Circuit Breaker '{self.name}' cambiando a OPEN tras {self.failure_threshold} fallos")
                self.state = CircuitState.OPEN
            
            # En HALF_OPEN, volver a abrir con un solo fallo
            elif self.state == CircuitState.HALF_OPEN:
                logger.warning(f"Circuit Breaker '{self.name}' volviendo a OPEN")
                self.state = CircuitState.OPEN
                
            raise e

# ======================================
# SISTEMA DE CHECKPOINTING
# ======================================

class CheckpointSystem:
    """Sistema de checkpointing simplificado."""
    
    def __init__(self, component_id, checkpoint_dir="./checkpoints_fast"):
        self.component_id = component_id
        self.checkpoint_dir = checkpoint_dir
        self.component_dir = os.path.join(checkpoint_dir, component_id)
        
        # Crear directorio si no existe
        os.makedirs(self.component_dir, exist_ok=True)
        
        # Estado actual
        self.state = {}
        
        logger.info(f"Sistema de Checkpoint inicializado para '{component_id}'")
    
    async def create_checkpoint(self):
        """Crear checkpoint del estado actual."""
        if not self.state:
            logger.warning(f"No hay estado para checkpoint en '{self.component_id}'")
            return ""
        
        checkpoint_id = f"{int(time.time() * 1000)}"
        checkpoint_path = os.path.join(self.component_dir, f"{checkpoint_id}.json")
        
        checkpoint_data = {
            "id": checkpoint_id,
            "component_id": self.component_id,
            "timestamp": time.time(),
            "state": self.state
        }
        
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f)
            logger.info(f"Checkpoint creado: {checkpoint_id}")
            return checkpoint_id
        except Exception as e:
            logger.error(f"Error al crear checkpoint: {e}")
            return ""
    
    async def restore_latest(self):
        """Restaurar último checkpoint."""
        checkpoints = self._get_checkpoint_list()
        if not checkpoints:
            logger.warning(f"No hay checkpoints disponibles para '{self.component_id}'")
            return False
        
        # Ordenar por timestamp (más reciente primero)
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        latest = checkpoints[0]
        
        try:
            self.state = latest["state"]
            logger.info(f"Checkpoint restaurado: {latest['id']}")
            return True
        except Exception as e:
            logger.error(f"Error al restaurar checkpoint: {e}")
            return False
    
    def _get_checkpoint_list(self):
        """Obtener lista de checkpoints disponibles."""
        checkpoints = []
        
        try:
            for filename in os.listdir(self.component_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(self.component_dir, filename)
                    with open(filepath, 'r') as f:
                        checkpoint_data = json.load(f)
                        checkpoints.append(checkpoint_data)
        except Exception as e:
            logger.error(f"Error al listar checkpoints: {e}")
        
        return checkpoints

# ======================================
# COMPONENTE DE PRUEBA
# ======================================

class TestComponent:
    """Componente de prueba que simula fallas aleatorias."""
    
    def __init__(self, component_id, fail_rate=0.3):
        self.component_id = component_id
        self.fail_rate = fail_rate
        self.data = {}
        
        # Características de resiliencia
        self.circuit_breaker = CircuitBreaker(name=f"cb_{component_id}")
        self.checkpoint_system = CheckpointSystem(component_id)
        
        logger.info(f"Componente '{component_id}' inicializado con fail_rate={fail_rate}")
    
    async def store(self, key, value):
        """Almacenar un valor con protección de resiliencia."""
        async def execute_operation():
            await asyncio.sleep(0.01)  # Simular trabajo
            
            if random.random() < self.fail_rate:
                raise Exception(f"Error simulado en {self.component_id}.store")
            
            self.data[key] = value
            self.checkpoint_system.state = self.data.copy()
            return {"status": "stored", "key": key}
        
        try:
            result = await self.circuit_breaker.execute(
                lambda: with_retry(execute_operation)
            )
            
            if result is None:
                return {"status": "rejected", "reason": "circuit_open"}
            
            # Crear checkpoint después de almacenar
            await self.checkpoint_system.create_checkpoint()
            
            return result
        except Exception as e:
            logger.error(f"Error al almacenar {key}: {e}")
            raise
    
    async def retrieve(self, key):
        """Recuperar un valor con protección de resiliencia."""
        async def execute_operation():
            await asyncio.sleep(0.01)  # Simular trabajo
            
            if random.random() < self.fail_rate:
                raise Exception(f"Error simulado en {self.component_id}.retrieve")
            
            if key in self.data:
                return {"status": "found", "key": key, "value": self.data[key]}
            return {"status": "not_found", "key": key}
        
        try:
            result = await self.circuit_breaker.execute(
                lambda: with_retry(execute_operation)
            )
            
            if result is None:
                return {"status": "rejected", "reason": "circuit_open"}
            
            return result
        except Exception as e:
            logger.error(f"Error al recuperar {key}: {e}")
            raise
    
    async def crash(self):
        """Simular crash que pierde los datos."""
        self.data = {}
        return {"status": "crashed", "component": self.component_id}
    
    async def restore_from_checkpoint(self):
        """Restaurar estado desde el último checkpoint."""
        success = await self.checkpoint_system.restore_latest()
        if success:
            self.data = self.checkpoint_system.state.copy()
            logger.info(f"Componente '{self.component_id}' restaurado con {len(self.data)} elementos")
        return success

# ======================================
# PRUEBAS
# ======================================

async def test_retry_system():
    """Probar sistema de reintentos."""
    logger.info("\n=== Test 1: Sistema de Reintentos ===")
    
    component = TestComponent("retry_test", fail_rate=0.6)
    
    # Almacenar varios valores
    keys_stored = 0
    for i in range(5):
        try:
            result = await component.store(f"key_{i}", f"value_{i}")
            keys_stored += 1
            logger.info(f"Almacenado: {result}")
        except Exception as e:
            logger.error(f"Fallo final: {e}")
    
    return keys_stored > 0

async def test_circuit_breaker():
    """Probar Circuit Breaker."""
    logger.info("\n=== Test 2: Circuit Breaker ===")
    
    component = TestComponent("circuit_test", fail_rate=1.0)  # 100% de fallos
    
    # Ejecutar operaciones que fallarán
    for i in range(5):
        try:
            result = await component.retrieve(f"test_key_{i}")
            logger.info(f"Resultado: {result}")
        except Exception as e:
            logger.error(f"Error esperado: {e}")
        
        logger.info(f"Estado del circuito: {component.circuit_breaker.state.name}")
    
    # Comprobar si el circuito está abierto
    is_open = component.circuit_breaker.state == CircuitState.OPEN
    
    # Probar que se rechazan las llamadas si está abierto
    if is_open:
        result = await component.retrieve("rejected_key")
        logger.info(f"Llamada con circuito abierto: {result}")
        
        # Esperar a que el circuito se recupere
        logger.info(f"Esperando recuperación ({component.circuit_breaker.recovery_timeout}s)...")
        await asyncio.sleep(component.circuit_breaker.recovery_timeout + 0.1)
        
        # Hacer que la siguiente llamada tenga éxito
        component.fail_rate = 0.0
        
        result = await component.retrieve("recovery_test")
        logger.info(f"Después de recuperación: {result}")
        logger.info(f"Estado final del circuito: {component.circuit_breaker.state.name}")
    
    return is_open

async def test_checkpointing():
    """Probar checkpointing y recuperación."""
    logger.info("\n=== Test 3: Checkpointing ===")
    
    component = TestComponent("checkpoint_test", fail_rate=0.0)
    
    # Almacenar datos
    test_data = [
        ("config.url", "https://example.com"),
        ("config.timeout", "30"),
        ("user.name", "Test User")
    ]
    
    for key, value in test_data:
        await component.store(key, value)
    
    # Verificar datos guardados
    for key, _ in test_data:
        result = await component.retrieve(key)
        logger.info(f"Antes de crash: {result}")
    
    # Simular crash
    await component.crash()
    logger.info("Componente ha sufrido crash, datos perdidos")
    
    # Verificar pérdida de datos
    missing = 0
    for key, _ in test_data:
        result = await component.retrieve(key)
        logger.info(f"Después de crash: {result}")
        if result.get("status") == "not_found":
            missing += 1
    
    # Restaurar
    await component.restore_from_checkpoint()
    
    # Verificar recuperación
    recovered = 0
    for key, _ in test_data:
        result = await component.retrieve(key)
        logger.info(f"Después de restauración: {result}")
        if result.get("status") == "found":
            recovered += 1
    
    return missing > 0 and recovered > 0

async def main():
    """Ejecutar todas las pruebas."""
    logger.info("=== PRUEBA RÁPIDA DE RESILIENCIA GENESIS ===")
    
    results = {}
    
    # Test 1: Sistema de Reintentos
    results["retry"] = await test_retry_system()
    
    # Test 2: Circuit Breaker
    results["circuit_breaker"] = await test_circuit_breaker()
    
    # Test 3: Checkpointing
    results["checkpointing"] = await test_checkpointing()
    
    # Resumen
    logger.info("\n=== Resumen de Pruebas ===")
    for name, result in results.items():
        status = "✓ ÉXITO" if result else "✗ ERROR"
        logger.info(f"Test {name}: {status}")
    
    success_rate = sum(1 for r in results.values() if r) / len(results) * 100
    logger.info(f"\nTasa de éxito: {success_rate:.1f}%")
    
    # Conclusiones
    logger.info("\n=== Conclusiones ===")
    logger.info("La prueba rápida demuestra las tres características principales de resiliencia:")
    logger.info("1. Sistema de Reintentos: Permite recuperación ante fallos transitorios")
    logger.info("2. Circuit Breaker: Aísla componentes fallidos y protege el sistema")
    logger.info("3. Checkpointing: Facilita la recuperación rápida tras fallos")

if __name__ == "__main__":
    asyncio.run(main())