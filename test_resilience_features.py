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
import time
import random
from typing import Dict, Any, List, Optional
from enum import Enum, auto
import os
import json

# Configurar logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_resilience")

# ========================= CARACTERÍSTICAS DE RESILIENCIA =========================

async def with_retry(func, max_retries=3, base_delay=0.1, jitter=0.1):
    """
    Ejecutar una función con reintentos adaptativos y backoff exponencial.
    
    Args:
        func: Función asíncrona a ejecutar
        max_retries: Número máximo de reintentos
        base_delay: Tiempo base entre reintentos
        jitter: Variación aleatoria máxima
        
    Returns:
        Resultado de la función
        
    Raises:
        Exception: Si se agotan todos los reintentos
    """
    attempt = 0
    start_time = time.time()
    
    while True:
        try:
            result = await func()
            elapsed = time.time() - start_time
            
            # Solo registrar en log si tomó múltiples intentos
            if attempt > 0:
                logger.info(f"Éxito en {elapsed:.3f}s después de {attempt} reintentos")
                
            return result
            
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                logger.error(f"Agotados {max_retries} reintentos. Error final: {e}")
                raise
            
            # Calcular tiempo de espera con backoff exponencial
            delay = base_delay * (2 ** (attempt - 1))
            random_jitter = random.uniform(0, jitter)
            total_delay = delay + random_jitter
            
            logger.info(f"Intento {attempt}: Fallo - {e}. Reintentando en {total_delay:.3f}s")
            await asyncio.sleep(total_delay)

class CircuitState(Enum):
    """Estados del Circuit Breaker."""
    CLOSED = auto()    # Funcionamiento normal
    OPEN = auto()      # Circuito abierto, rechazan llamadas
    HALF_OPEN = auto() # Semi-abierto, permite algunas llamadas

class CircuitBreaker:
    """
    Implementación del patrón Circuit Breaker.
    
    Protege al sistema contra operaciones que fallan constantemente,
    evitando sobrecarga y permitiendo recuperación gradual.
    """
    
    def __init__(
        self, 
        name: str, 
        failure_threshold: int = 3,
        recovery_timeout: float = 5.0,
        half_open_max_calls: int = 1,
        success_threshold: int = 2
    ):
        """
        Inicializar Circuit Breaker.
        
        Args:
            name: Nombre del circuit breaker
            failure_threshold: Fallos consecutivos para abrir el circuito
            recovery_timeout: Tiempo hasta probar recuperación
            half_open_max_calls: Máximo de llamadas en estado half-open
            success_threshold: Éxitos consecutivos para cerrar el circuito
        """
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
        
        self.total_calls = 0
        self.total_failures = 0
        self.total_rejections = 0
        self.total_successes = 0
        
        logger.info(f"Circuit Breaker '{name}' creado en estado {self.state.name}")
    
    async def execute(self, func, *args, **kwargs):
        """
        Ejecutar función con protección del Circuit Breaker.
        
        Args:
            func: Función a ejecutar
            *args, **kwargs: Argumentos para la función
            
        Returns:
            Resultado de la función o None si el circuito está abierto
            
        Raises:
            Exception: Si ocurre un error y el circuito no está abierto
        """
        self.total_calls += 1
        
        # Verificar estado del circuito
        if self.state == CircuitState.OPEN:
            # Verificar si ha pasado el tiempo de recovery
            if (time.time() - self.last_failure_time) > self.recovery_timeout:
                logger.info(f"Circuit Breaker '{self.name}' cambiando a HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                self.success_count = 0
            else:
                # Rechazar la llamada
                self.total_rejections += 1
                logger.warning(f"Circuit Breaker '{self.name}' abierto. Llamada rechazada.")
                return None
        
        # Verificar límite de llamadas en estado HALF_OPEN
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.half_open_max_calls:
                self.total_rejections += 1
                logger.warning(f"Circuit Breaker '{self.name}' en HALF_OPEN. Máximo de llamadas en prueba alcanzado.")
                return None
            self.half_open_calls += 1
        
        # Ejecutar la función protegida
        try:
            result = await func(*args, **kwargs)
            
            # Registrar éxito
            self.success_count += 1
            self.failure_count = 0
            self.total_successes += 1
            
            # En estado HALF_OPEN, cerrar el circuito si se alcanza el umbral de éxitos
            if self.state == CircuitState.HALF_OPEN and self.success_count >= self.success_threshold:
                logger.info(f"Circuit Breaker '{self.name}' cambiando a CLOSED")
                self.state = CircuitState.CLOSED
            
            return result
            
        except Exception as e:
            # Registrar fallo
            self.failure_count += 1
            self.success_count = 0
            self.last_failure_time = time.time()
            self.total_failures += 1
            
            # En estado CLOSED, abrir el circuito si se excede el umbral de fallos
            if self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
                logger.warning(f"Circuit Breaker '{self.name}' cambiando a OPEN tras {self.failure_threshold} fallos")
                self.state = CircuitState.OPEN
            
            # En estado HALF_OPEN, volver a abrir el circuito ante un solo fallo
            elif self.state == CircuitState.HALF_OPEN:
                logger.warning(f"Circuit Breaker '{self.name}' volviendo a OPEN tras fallo en estado HALF_OPEN")
                self.state = CircuitState.OPEN
            
            raise e
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del circuit breaker.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "name": self.name,
            "state": self.state.name,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls": self.total_calls,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "total_rejections": self.total_rejections,
            "success_rate": self.total_successes / max(1, self.total_calls) * 100,
            "last_failure_time": self.last_failure_time
        }

class CheckpointSystem:
    """
    Sistema de checkpointing para respaldo y recuperación de estado.
    
    Permite crear puntos de recuperación automáticos o manuales
    y restaurar desde ellos en caso de fallos.
    """
    
    def __init__(
        self,
        component_id: str,
        checkpoint_dir: str = "./checkpoints",
        auto_checkpoint: bool = True,
        checkpoint_interval: float = 30.0  # segundos entre checkpoints
    ):
        """
        Inicializar sistema de checkpointing.
        
        Args:
            component_id: ID del componente
            checkpoint_dir: Directorio para checkpoints
            auto_checkpoint: Si se debe hacer checkpointing automático
            checkpoint_interval: Intervalo entre checkpoints automáticos
        """
        self.component_id = component_id
        self.checkpoint_dir = checkpoint_dir
        self.component_dir = os.path.join(checkpoint_dir, component_id)
        
        # Crear directorio si no existe
        os.makedirs(self.component_dir, exist_ok=True)
        
        # Configuración
        self.auto_checkpoint = auto_checkpoint
        self.checkpoint_interval = checkpoint_interval
        
        # Estado actual y checkpoint task
        self.state = {}  # Será actualizado por el componente
        self.checkpoint_task = None
        self.running = False
        
        logger.info(f"Sistema de Checkpoint inicializado para '{component_id}'")
    
    async def start(self) -> None:
        """Iniciar checkpointing automático."""
        if self.auto_checkpoint and not self.running:
            self.running = True
            self.checkpoint_task = asyncio.create_task(self._auto_checkpoint_loop())
            logger.info(f"Checkpointing automático iniciado para '{self.component_id}'")
    
    async def stop(self) -> None:
        """Detener checkpointing automático."""
        if self.running and self.checkpoint_task:
            self.running = False
            self.checkpoint_task.cancel()
            try:
                await self.checkpoint_task
            except asyncio.CancelledError:
                pass
            logger.info(f"Checkpointing automático detenido para '{self.component_id}'")
    
    async def _auto_checkpoint_loop(self) -> None:
        """Bucle de checkpointing automático."""
        while self.running:
            try:
                await self.create_checkpoint()
                await asyncio.sleep(self.checkpoint_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error en checkpointing automático para '{self.component_id}': {e}")
                await asyncio.sleep(1.0)  # Esperar un poco antes de reintentar
    
    async def create_checkpoint(self) -> str:
        """
        Crear checkpoint del estado actual.
        
        Returns:
            ID del checkpoint creado
        """
        if not self.state:
            logger.warning(f"No hay estado para checkpoint en '{self.component_id}'")
            return ""
        
        # Generar ID de checkpoint basado en timestamp
        checkpoint_id = f"{int(time.time() * 1000)}"
        checkpoint_path = os.path.join(self.component_dir, f"{checkpoint_id}.json")
        
        # Crear diccionario de checkpoint
        checkpoint_data = {
            "id": checkpoint_id,
            "component_id": self.component_id,
            "timestamp": time.time(),
            "state": self.state
        }
        
        # Guardar a disco
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            logger.debug(f"Checkpoint creado: {checkpoint_id} para '{self.component_id}'")
            return checkpoint_id
        except Exception as e:
            logger.error(f"Error al crear checkpoint para '{self.component_id}': {e}")
            return ""
    
    async def restore_latest(self) -> bool:
        """
        Restaurar último checkpoint.
        
        Returns:
            True si se restauró correctamente
        """
        checkpoints = self._get_checkpoint_list()
        if not checkpoints:
            logger.warning(f"No hay checkpoints disponibles para '{self.component_id}'")
            return False
        
        # Ordenar por timestamp (más reciente primero)
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        latest = checkpoints[0]
        
        # Restaurar
        try:
            self.state = latest["state"]
            logger.info(f"Checkpoint restaurado: {latest['id']} para '{self.component_id}'")
            return True
        except Exception as e:
            logger.error(f"Error al restaurar checkpoint para '{self.component_id}': {e}")
            return False
    
    def _get_checkpoint_list(self) -> List[Dict[str, Any]]:
        """
        Obtener lista de checkpoints disponibles.
        
        Returns:
            Lista de metadatos de checkpoints
        """
        checkpoints = []
        
        try:
            for filename in os.listdir(self.component_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(self.component_dir, filename)
                    with open(filepath, 'r') as f:
                        checkpoint_data = json.load(f)
                        checkpoints.append(checkpoint_data)
        except Exception as e:
            logger.error(f"Error al listar checkpoints para '{self.component_id}': {e}")
        
        return checkpoints

# ========================= COMPONENTE DE PRUEBA =========================

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
        self.data = {}
        
        # Características de resiliencia
        self.circuit_breaker = CircuitBreaker(name=f"cb_{component_id}")
        self.checkpoint_system = CheckpointSystem(component_id)
        
        logger.info(f"Componente '{component_id}' inicializado con fail_rate={fail_rate}")
    
    async def init(self):
        """Inicializar componente."""
        # Iniciar checkpointing automático
        await self.checkpoint_system.start()
        # Intentar restaurar desde último checkpoint
        await self.restore_from_checkpoint()
    
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
        # Función para ejecutar con Circuit Breaker y Retry
        async def execute_operation():
            await asyncio.sleep(0.05)  # Simular trabajo
            
            # Simular fallo aleatorio
            if random.random() < self.fail_rate:
                raise Exception(f"Error simulado en {self.component_id}.{operation}")
            
            # Procesamiento según tipo de operación
            if operation == "store":
                key, val = value
                self.data[key] = val
                # Actualizar estado para checkpoint
                self.checkpoint_system.state = self.data.copy()
                return {"status": "stored", "key": key}
            
            elif operation == "retrieve":
                key = value
                if key in self.data:
                    return {"status": "found", "key": key, "value": self.data[key]}
                return {"status": "not_found", "key": key}
            
            elif operation == "crash":
                # Simular crash que pierde los datos
                old_data = self.data.copy()
                self.data = {}
                return {"status": "crashed", "lost_items": len(old_data)}
            
            # Operación genérica
            return {"status": "processed", "operation": operation}
        
        # Ejecutar con Circuit Breaker y Retry
        try:
            result = await self.circuit_breaker.execute(
                lambda: with_retry(execute_operation)
            )
            
            if result is None:
                # El circuito está abierto
                return {"status": "rejected", "reason": "circuit_open"}
            
            return result
            
        except Exception as e:
            # Si es un crash, intentar restaurar
            if "crash" in str(e).lower():
                await self.restore_from_checkpoint()
            
            raise e
    
    async def restore_from_checkpoint(self) -> bool:
        """
        Restaurar estado desde el último checkpoint.
        
        Returns:
            True si se restauró correctamente
        """
        success = await self.checkpoint_system.restore_latest()
        if success:
            self.data = self.checkpoint_system.state.copy()
            logger.info(f"Componente '{self.component_id}' restaurado con {len(self.data)} elementos")
        return success

# ========================= PRUEBAS DE RESILIENCIA =========================

async def test_resilience_features() -> Dict[str, Any]:
    """
    Probar características de resiliencia integradas.
    
    Returns:
        Diccionario con resultados de las pruebas
    """
    results = {
        "retry_system": {"success": False, "details": []},
        "circuit_breaker": {"success": False, "details": []},
        "checkpointing": {"success": False, "details": []}
    }
    
    # Crear componente de prueba
    component = TestComponent("resilience_test", fail_rate=0.4)
    await component.init()
    
    # ------------------------- PRUEBA 1: SISTEMA DE REINTENTOS -------------------------
    logger.info("\n=== Prueba 1: Sistema de Reintentos ===")
    
    # Configurar fail_rate alta para probar reintentos
    component.fail_rate = 0.7
    retry_results = []
    
    # Realizar varias operaciones
    for i in range(5):
        try:
            result = await component.process_operation("store", (f"key_{i}", f"value_{i}"))
            retry_results.append({"operation": i, "success": True, "result": result})
        except Exception as e:
            retry_results.append({"operation": i, "success": False, "error": str(e)})
    
    # Verificar resultados
    success_count = sum(1 for r in retry_results if r["success"])
    results["retry_system"]["success"] = success_count > 0
    results["retry_system"]["details"] = retry_results
    results["retry_system"]["success_rate"] = success_count / len(retry_results) * 100
    
    # ------------------------- PRUEBA 2: CIRCUIT BREAKER -------------------------
    logger.info("\n=== Prueba 2: Circuit Breaker ===")
    
    # Configurar fail_rate para garantizar que el circuito se abra
    component.fail_rate = 1.0  # 100% de fallos
    cb_results = []
    
    # Secuencia de operaciones para probar circuit breaker
    for i in range(8):
        try:
            result = await component.process_operation("retrieve", f"test_key_{i}")
            cb_results.append({
                "attempt": i, 
                "success": True, 
                "result": result,
                "circuit_state": component.circuit_breaker.state.name
            })
        except Exception as e:
            cb_results.append({
                "attempt": i, 
                "success": False, 
                "error": str(e),
                "circuit_state": component.circuit_breaker.state.name
            })
    
    # Esperar a que el circuito intente cerrarse
    logger.info("Esperando recuperación del Circuit Breaker...")
    await asyncio.sleep(6.0)
    
    # Configurar fail_rate para permitir operaciones exitosas
    component.fail_rate = 0.0  # 0% de fallos
    
    # Probar recuperación
    for i in range(3):
        try:
            result = await component.process_operation("retrieve", f"recovery_key_{i}")
            cb_results.append({
                "attempt": f"recovery_{i}", 
                "success": True, 
                "result": result,
                "circuit_state": component.circuit_breaker.state.name
            })
        except Exception as e:
            cb_results.append({
                "attempt": f"recovery_{i}", 
                "success": False, 
                "error": str(e),
                "circuit_state": component.circuit_breaker.state.name
            })
    
    # Verificar transición OPEN -> HALF_OPEN -> CLOSED
    states = [r["circuit_state"] for r in cb_results]
    results["circuit_breaker"]["success"] = ("OPEN" in states and "HALF_OPEN" in states and "CLOSED" in states)
    results["circuit_breaker"]["details"] = cb_results
    results["circuit_breaker"]["stats"] = component.circuit_breaker.get_stats()
    
    # ------------------------- PRUEBA 3: CHECKPOINTING -------------------------
    logger.info("\n=== Prueba 3: Checkpointing ===")
    
    # Configurar fail_rate para operaciones fiables
    component.fail_rate = 0.0
    cp_results = []
    
    # Almacenar datos importantes
    test_data = [
        ("config.api_url", "https://api.example.com"),
        ("config.timeout", "30"),
        ("config.retries", "3"),
        ("user.id", "12345"),
        ("user.name", "Test User")
    ]
    
    # Guardar datos y crear checkpoint
    for key, value in test_data:
        result = await component.process_operation("store", (key, value))
        cp_results.append({"operation": f"store_{key}", "result": result})
    
    # Verificar datos guardados
    for key, _ in test_data:
        result = await component.process_operation("retrieve", key)
        cp_results.append({"operation": f"verify_{key}", "result": result})
    
    # Crear checkpoint manual
    checkpoint_id = await component.checkpoint_system.create_checkpoint()
    cp_results.append({"operation": "manual_checkpoint", "checkpoint_id": checkpoint_id})
    
    # Simular crash
    try:
        crash_result = await component.process_operation("crash", None)
        cp_results.append({"operation": "crash", "result": crash_result})
    except Exception as e:
        cp_results.append({"operation": "crash", "error": str(e)})
    
    # Verificar pérdida de datos
    missing_count = 0
    for key, _ in test_data:
        result = await component.process_operation("retrieve", key)
        missing = result.get("status") == "not_found"
        if missing:
            missing_count += 1
        cp_results.append({"operation": f"after_crash_{key}", "result": result, "missing": missing})
    
    # Restaurar desde checkpoint
    restore_success = await component.restore_from_checkpoint()
    cp_results.append({"operation": "restore", "success": restore_success})
    
    # Verificar datos restaurados
    recovered_count = 0
    for key, _ in test_data:
        result = await component.process_operation("retrieve", key)
        recovered = result.get("status") == "found"
        if recovered:
            recovered_count += 1
        cp_results.append({"operation": f"after_restore_{key}", "result": result, "recovered": recovered})
    
    # Evaluar éxito
    results["checkpointing"]["success"] = (missing_count > 0 and recovered_count > 0)
    results["checkpointing"]["details"] = cp_results
    results["checkpointing"]["missing_count"] = missing_count
    results["checkpointing"]["recovered_count"] = recovered_count
    
    # Detener checkpoint automático
    await component.checkpoint_system.stop()
    
    return results

# ========================= FUNCIÓN PRINCIPAL =========================

async def main():
    """Ejecutar pruebas de resiliencia."""
    logger.info("=== PRUEBA INTEGRADA DE RESILIENCIA GENESIS ===")
    
    start_time = time.time()
    results = await test_resilience_features()
    elapsed = time.time() - start_time
    
    # Mostrar resultados
    logger.info("\n=== RESULTADOS DE PRUEBAS ===")
    
    for test_name, test_result in results.items():
        success = "✓ ÉXITO" if test_result["success"] else "✗ ERROR"
        logger.info(f"{test_name}: {success}")
    
    # Calcular tasa de éxito
    success_count = sum(1 for r in results.values() if r["success"])
    success_rate = success_count / len(results) * 100
    
    logger.info(f"\nTasa de éxito: {success_rate:.1f}%")
    logger.info(f"Tiempo total: {elapsed:.2f}s")
    
    # Conclusiones
    logger.info("\n=== CONCLUSIONES ===")
    logger.info("El sistema de resiliencia Genesis integra tres características principales:")
    logger.info("1. Sistema de Reintentos: Permite recuperación ante fallos transitorios")
    logger.info("2. Circuit Breaker: Aísla componentes fallidos y protege el sistema")
    logger.info("3. Checkpointing: Facilita la recuperación rápida tras fallos")
    
    if success_rate == 100:
        logger.info("\n¡TODAS LAS PRUEBAS EXITOSAS! El sistema de resiliencia funciona correctamente.")
    else:
        failed = [name for name, result in results.items() if not result["success"]]
        logger.warning(f"\nPruebas fallidas: {', '.join(failed)}")

if __name__ == "__main__":
    asyncio.run(main())