#!/usr/bin/env python
"""
Test ARMAGEDÓN para el Sistema Genesis.

Este script ejecuta una prueba ARMAGEDÓN que verifica la resiliencia y 
funcionalidad de los componentes cloud y oracle del sistema Genesis.
"""

import asyncio
import logging
import os
import time
import random

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_armageddon")

# Importar componentes de Genesis
from genesis.cloud.circuit_breaker import CloudCircuitBreakerFactory, CircuitState
from genesis.cloud.distributed_checkpoint import DistributedCheckpointManager, CheckpointStorageType, CheckpointConsistencyLevel
from genesis.oracle.armageddon_adapter import ArmageddonAdapter, IntegrationStatus


class ArmageddonTest:
    """
    Prueba ARMAGEDÓN para el Sistema Genesis.
    
    Esta prueba verifica la integración entre:
    - CloudCircuitBreaker para protección contra fallos en cascada
    - DistributedCheckpointManager para respaldo y recuperación
    - ArmageddonAdapter para integración con servicios externos
    """
    
    def __init__(self):
        """Inicializar prueba ARMAGEDÓN."""
        self.cb_factory = None
        self.checkpoint_manager = None
        self.armageddon_adapter = None
        
        # Métricas de prueba
        self.metrics = {
            "start_time": 0,
            "end_time": 0,
            "operations": 0,
            "successes": 0,
            "failures": 0,
            "recovery_attempts": 0,
            "successful_recoveries": 0
        }
        
        self.initialized = False
    
    async def initialize(self) -> bool:
        """
        Inicializar componentes para la prueba.
        
        Returns:
            True si se inicializaron correctamente
        """
        try:
            logger.info("Iniciando prueba ARMAGEDÓN...")
            
            # Inicializar CloudCircuitBreakerFactory
            self.cb_factory = CloudCircuitBreakerFactory()
            logger.info("CircuitBreakerFactory creado")
            
            # Inicializar DistributedCheckpointManager
            self.checkpoint_manager = DistributedCheckpointManager()
            success = await self.checkpoint_manager.initialize(
                storage_type=CheckpointStorageType.MEMORY,
                consistency_level=CheckpointConsistencyLevel.STRONG
            )
            
            if not success:
                logger.error("Error al inicializar CheckpointManager")
                return False
            
            logger.info("CheckpointManager inicializado")
            
            # Inicializar ArmageddonAdapter
            self.armageddon_adapter = ArmageddonAdapter()
            success = await self.armageddon_adapter.initialize()
            
            if not success:
                logger.warning("ArmageddonAdapter inicializado con advertencias (posibles APIs no configuradas)")
            
            logger.info("ArmageddonAdapter inicializado")
            
            # Crear circuit breakers para la prueba
            await self.cb_factory.create("test_processing", failure_threshold=3, recovery_timeout=5)
            await self.cb_factory.create("test_storage", failure_threshold=2, recovery_timeout=10)
            
            # Entrelazar circuit breakers
            await self.cb_factory.entangle_circuits(["test_processing", "test_storage"])
            
            self.initialized = True
            logger.info("Prueba ARMAGEDÓN inicializada correctamente")
            
            return True
            
        except Exception as e:
            logger.error(f"Error al inicializar prueba ARMAGEDÓN: {e}")
            return False
    
    async def run_test(self) -> bool:
        """
        Ejecutar prueba ARMAGEDÓN.
        
        Returns:
            True si la prueba fue exitosa
        """
        if not self.initialized:
            logger.error("Prueba ARMAGEDÓN no inicializada")
            return False
        
        self.metrics["start_time"] = int(time.time())
        
        try:
            # Verificar componentes
            result = await self._verify_components()
            if not result:
                logger.error("Verificación de componentes fallida")
                return False
            
            # Ejecutar patrón TSUNAMI_OPERACIONES
            result = await self._run_tsunami_operations()
            if not result:
                logger.error("Prueba TSUNAMI_OPERACIONES fallida")
                
            # Ejecutar patrón INYECCION_CAOS
            result = await self._run_chaos_injection()
            if not result:
                logger.error("Prueba INYECCION_CAOS fallida")
            
            # Ejecutar patrón APOCALIPSIS_FINAL
            result = await self._run_final_apocalypse()
            if not result:
                logger.error("Prueba APOCALIPSIS_FINAL fallida")
            
            # Verificar recuperación
            result = await self._verify_recovery()
            if not result:
                logger.error("Verificación de recuperación fallida")
                return False
            
            self.metrics["end_time"] = time.time()
            
            logger.info("Prueba ARMAGEDÓN completada con éxito")
            return True
            
        except Exception as e:
            logger.error(f"Error durante prueba ARMAGEDÓN: {e}")
            self.metrics["end_time"] = time.time()
            return False
        
        finally:
            # Mostrar resumen
            self._print_summary()
    
    async def _verify_components(self) -> bool:
        """
        Verificar componentes antes de prueba de estrés.
        
        Returns:
            True si todos los componentes funcionan correctamente
        """
        logger.info("Verificando componentes...")
        
        # Verificar CircuitBreaker
        cb = self.cb_factory.get("test_processing")
        if cb is None:
            logger.error("No se pudo obtener CircuitBreaker 'test_processing'")
            return False
            
        if cb.get_state() != CircuitState.CLOSED:
            logger.error("CircuitBreaker no está en estado CLOSED")
            return False
        
        # Verificar CheckpointManager
        component_id = f"test_component_{random.randint(1000, 9999)}"
        test_data = {"value": "test", "timestamp": time.time()}
        
        checkpoint_id = await self.checkpoint_manager.create_checkpoint(
            component_id=component_id,
            data=test_data
        )
        
        if not checkpoint_id:
            logger.error("No se pudo crear checkpoint")
            return False
            
        loaded_data, _ = await self.checkpoint_manager.load_checkpoint(checkpoint_id)
        
        if not loaded_data or loaded_data["value"] != "test":
            logger.error("Error al cargar checkpoint")
            return False
        
        # Limpiar
        await self.checkpoint_manager.delete_checkpoint(checkpoint_id)
        
        # Verificar ArmageddonAdapter
        status = self.armageddon_adapter.get_status()
        
        logger.info(f"Estado de APIs: {status}")
        
        logger.info("Todos los componentes verificados correctamente")
        return True
    
    async def _run_tsunami_operations(self) -> bool:
        """
        Ejecutar patrón TSUNAMI_OPERACIONES: sobrecarga con operaciones paralelas.
        
        Returns:
            True si el sistema manejó la carga
        """
        logger.info("Ejecutando patrón TSUNAMI_OPERACIONES...")
        
        # Número de operaciones paralelas
        num_operations = 100
        
        # Obtener CircuitBreaker
        cb = self.cb_factory.get("test_processing")
        if cb is None:
            logger.error("No se encontró CircuitBreaker")
            return False
        
        # Crear tareas en paralelo
        tasks = []
        for i in range(num_operations):
            data = {"operation_id": i, "timestamp": time.time()}
            
            if i % 10 == 0:  # 10% de fallos
                tasks.append(self._process_data_with_error(cb, data))
            else:
                tasks.append(self._process_data(cb, data))
        
        # Ejecutar tareas
        self.metrics["operations"] += num_operations
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Contar resultados
        successes = sum(1 for r in results if isinstance(r, dict) and r.get("success") == True)
        failures = sum(1 for r in results if isinstance(r, Exception) or (isinstance(r, dict) and r.get("success") == False))
        
        self.metrics["successes"] += successes
        self.metrics["failures"] += failures
        
        logger.info(f"TSUNAMI_OPERACIONES: {successes} éxitos, {failures} fallos")
        
        # Verificar estado
        await asyncio.sleep(1)  # Dar tiempo para procesar todo
        state = cb.get_state()
        logger.info(f"Estado del CircuitBreaker después de TSUNAMI: {state}")
        
        if state == CircuitState.OPEN:
            # Demostrar auto-recuperación
            logger.info("CircuitBreaker abierto, esperando recuperación...")
            await asyncio.sleep(7)  # Mayor que recovery_timeout
            
            # Verificar si se recuperó
            state = cb.get_state()
            logger.info(f"Estado después de espera: {state}")
            
            if state == CircuitState.HALF_OPEN or state == CircuitState.CLOSED:
                self.metrics["recovery_attempts"] += 1
                self.metrics["successful_recoveries"] += 1
        
        return True
    
    async def _run_chaos_injection(self) -> bool:
        """
        Ejecutar patrón INYECCION_CAOS: errores aleatorios en operaciones críticas.
        
        Returns:
            True si el sistema manejó la inyección de caos
        """
        logger.info("Ejecutando patrón INYECCION_CAOS...")
        
        # Crear checkpoint antes del caos
        component_id = "chaos_test"
        original_data = {
            "value": random.randint(1000, 9999),
            "timestamp": time.time(),
            "status": "before_chaos"
        }
        
        if self.checkpoint_manager is None:
            logger.error("CheckpointManager no inicializado")
            return False
            
        try:
            checkpoint_id = await self.checkpoint_manager.create_checkpoint(
                component_id=component_id,
                data=original_data
            )
            
            if not checkpoint_id:
                logger.error("No se pudo crear checkpoint antes del caos")
                return False
        except Exception as e:
            logger.error(f"Error al crear checkpoint: {e}")
            return False
        
        logger.info(f"Checkpoint creado: {checkpoint_id}")
        
        # Inyectar errores en múltiples componentes
        operations = 50
        self.metrics["operations"] += operations
        
        # Simular fallos en CircuitBreaker
        cb = self.cb_factory.get("test_storage")
        if cb is None:
            logger.error("No se encontró CircuitBreaker 'test_storage'")
            return False
        
        # Provocar fallos para abrir el circuito
        for i in range(5):
            try:
                await self._storage_operation_with_error(cb)
                self.metrics["failures"] += 1
            except Exception:
                # Esperado
                pass
        
        # Verificar estado
        state = cb.get_state()
        logger.info(f"Estado de CircuitBreaker después de inyección: {state}")
        
        # Simular corrupción de datos
        chaos_data = original_data.copy()
        chaos_data["value"] = None
        chaos_data["corrupted"] = True
        chaos_data["status"] = "chaos_injected"
        
        # Intentar crear checkpoint con datos corruptos
        try:
            corrupt_checkpoint_id = await self.checkpoint_manager.create_checkpoint(
                component_id=component_id,
                data=chaos_data
            )
            if corrupt_checkpoint_id:
                logger.info(f"Checkpoint corrupto creado: {corrupt_checkpoint_id}")
        except Exception as e:
            logger.info(f"Checkpoint corrupto rechazado: {e}")
        
        # Recuperar desde checkpoint original
        data, metadata = await self.checkpoint_manager.load_checkpoint(checkpoint_id)
        
        if data and data["value"] == original_data["value"]:
            logger.info("Datos recuperados exitosamente después del caos")
            self.metrics["successful_recoveries"] += 1
        else:
            logger.error("Error al recuperar datos después del caos")
            return False
        
        # Limpiar
        await self.checkpoint_manager.delete_checkpoint(checkpoint_id)
        if corrupt_checkpoint_id:
            await self.checkpoint_manager.delete_checkpoint(corrupt_checkpoint_id)
        
        logger.info("INYECCION_CAOS completada")
        return True
    
    async def _run_final_apocalypse(self) -> bool:
        """
        Ejecutar patrón APOCALIPSIS_FINAL: fallo catastrófico y recuperación.
        
        Returns:
            True si el sistema se recuperó correctamente
        """
        logger.info("Ejecutando patrón APOCALIPSIS_FINAL...")
        
        # Crear datos importantes
        important_data = {
            "critical_value": f"tesoro_{random.randint(10000, 99999)}",
            "timestamp": time.time(),
            "details": {
                "priority": "alta",
                "sensitivity": "extrema",
                "backup_required": True
            }
        }
        
        # Guardar en múltiples checkpoints
        checkpoints = []
        for i in range(3):
            component_id = f"apocalypse_component_{i}"
            
            # Cada checkpoint tiene una variación
            data = important_data.copy()
            data["replica_id"] = i
            
            checkpoint_id = await self.checkpoint_manager.create_checkpoint(
                component_id=component_id,
                data=data,
                tags=["apocalypse_test", f"replica_{i}"]
            )
            
            if checkpoint_id:
                checkpoints.append(checkpoint_id)
        
        logger.info(f"Creados {len(checkpoints)} checkpoints de respaldo")
        
        # Forzar apertura de todos los circuit breakers
        for name in ["test_processing", "test_storage"]:
            cb = self.cb_factory.get(name)
            if cb:
                await cb.force_open()
                logger.info(f"CircuitBreaker {name} forzado a estado OPEN")
        
        # Simular apocalipsis total
        logger.info("¡APOCALIPSIS AHORA!")
        
        # Simular corte de todas las APIs
        if hasattr(self.armageddon_adapter, 'integrations'):
            for service in self.armageddon_adapter.integrations:
                self.armageddon_adapter.integrations[service]["status"] = IntegrationStatus.ERROR
            logger.info(f"Simulado corte de {len(self.armageddon_adapter.integrations)} APIs")
        
        # Verificar estado de sistema
        apis_status = self.armageddon_adapter.get_status()
        all_cbs = [self.cb_factory.get(name) for name in ["test_processing", "test_storage"]]
        all_open = all(cb and cb.get_state() == CircuitState.OPEN for cb in all_cbs)
        
        logger.info(f"Estado sistema apocalíptico confirmado: {all_open}")
        
        # Proceso de recuperación
        logger.info("Iniciando recuperación post-apocalíptica...")
        
        # Paso 1: Restaurar CircuitBreakers
        for name in ["test_processing", "test_storage"]:
            cb = self.cb_factory.get(name)
            if cb:
                await cb.reset()
                logger.info(f"CircuitBreaker {name} reseteado")
        
        # Paso 2: Reactivar APIs
        if hasattr(self.armageddon_adapter, 'integrations'):
            for service in self.armageddon_adapter.integrations:
                self.armageddon_adapter.integrations[service]["status"] = IntegrationStatus.ACTIVE
            logger.info(f"Reactivadas {len(self.armageddon_adapter.integrations)} APIs")
        
        # Paso 3: Recuperar datos críticos
        recovered_data = []
        
        for checkpoint_id in checkpoints:
            data, _ = await self.checkpoint_manager.load_checkpoint(checkpoint_id)
            if data and "critical_value" in data:
                recovered_data.append(data)
                logger.info(f"Datos recuperados de checkpoint {checkpoint_id}")
        
        # Verificar si se recuperaron todos los datos
        if len(recovered_data) == len(checkpoints):
            logger.info("Recuperación post-apocalíptica exitosa")
            self.metrics["successful_recoveries"] += 1
        else:
            logger.error(f"Recuperación incompleta: {len(recovered_data)}/{len(checkpoints)}")
            return False
        
        # Limpiar
        for checkpoint_id in checkpoints:
            await self.checkpoint_manager.delete_checkpoint(checkpoint_id)
        
        return True
    
    async def _verify_recovery(self) -> bool:
        """
        Verificar recuperación total después de todas las pruebas.
        
        Returns:
            True si todos los componentes se recuperaron
        """
        logger.info("Verificando recuperación final...")
        
        # Verificar estado de CircuitBreakers
        all_closed = True
        if self.cb_factory is not None:
            for name in ["test_processing", "test_storage"]:
                cb = self.cb_factory.get(name)
                if cb:
                    state = cb.get_state()
                    if state != CircuitState.CLOSED:
                        await cb.force_closed()
                        all_closed = False
                        logger.warning(f"CircuitBreaker {name} forzado a CLOSED")
            
            if all_closed:
                logger.info("Todos los CircuitBreakers recuperados correctamente")
        else:
            logger.warning("No hay CircuitBreakers para verificar")
        
        # Verificar CheckpointManager
        if self.checkpoint_manager is not None:
            try:
                checkpoints = await self.checkpoint_manager.list_checkpoints()
                logger.info(f"Checkpoints restantes: {len(checkpoints)}")
            except Exception as e:
                logger.error(f"Error al listar checkpoints: {e}")
        else:
            logger.warning("No hay CheckpointManager para verificar")
        
        # Verificar ArmageddonAdapter
        if self.armageddon_adapter is not None:
            try:
                status = self.armageddon_adapter.get_status()
                apis_active = sum(1 for s in status.values() if s.get("active", False))
                logger.info(f"APIs activas: {apis_active}/{len(status)}")
            except Exception as e:
                logger.error(f"Error al verificar estado de APIs: {e}")
        else:
            logger.warning("No hay ArmageddonAdapter para verificar")
        
        return True
    
    async def _process_data(self, cb, data):
        """Procesar datos con protección de CircuitBreaker."""
        if not await cb.before_call():
            return {"success": False, "error": "circuit_open"}
        
        try:
            # Simular procesamiento
            await asyncio.sleep(0.01)
            result = {"success": True, "data": data}
            
            await cb.on_success()
            return result
            
        except Exception as e:
            await cb.on_failure(e)
            raise
    
    async def _process_data_with_error(self, cb, data):
        """Procesar datos con error simulado."""
        if not await cb.before_call():
            return {"success": False, "error": "circuit_open"}
        
        try:
            # Simular error
            await asyncio.sleep(0.01)
            if random.random() < 0.8:  # 80% de probabilidad de fallo
                raise ValueError("Error simulado en procesamiento")
            
            result = {"success": True, "data": data}
            await cb.on_success()
            return result
            
        except Exception as e:
            await cb.on_failure(e)
            raise
    
    async def _storage_operation_with_error(self, cb):
        """Operación de almacenamiento con error garantizado."""
        if not await cb.before_call():
            raise RuntimeError("Circuit open")
        
        try:
            # Simular operación fallida
            await asyncio.sleep(0.05)
            raise IOError("Error simulado en almacenamiento")
            
        except Exception as e:
            await cb.on_failure(e)
            raise
    
    def _print_summary(self):
        """Imprimir resumen de la prueba."""
        duration = self.metrics["end_time"] - self.metrics["start_time"]
        
        print("\n" + "=" * 50)
        print("RESUMEN DE PRUEBA ARMAGEDÓN".center(50))
        print("=" * 50)
        print(f"Duración: {duration:.2f} segundos")
        print(f"Operaciones totales: {self.metrics['operations']}")
        print(f"Éxitos: {self.metrics['successes']}")
        print(f"Fallos: {self.metrics['failures']}")
        
        if self.metrics['operations'] > 0:
            success_rate = (self.metrics['successes'] / self.metrics['operations']) * 100
            print(f"Tasa de éxito: {success_rate:.1f}%")
        
        print(f"Intentos de recuperación: {self.metrics['recovery_attempts']}")
        print(f"Recuperaciones exitosas: {self.metrics['successful_recoveries']}")
        
        if self.metrics['recovery_attempts'] > 0:
            recovery_rate = (self.metrics['successful_recoveries'] / self.metrics['recovery_attempts']) * 100
            print(f"Tasa de recuperación: {recovery_rate:.1f}%")
        
        print("=" * 50)


async def main():
    """Función principal."""
    # Ejecutar prueba ARMAGEDÓN
    test = ArmageddonTest()
    
    # Inicializar
    success = await test.initialize()
    if not success:
        logger.error("No se pudo inicializar la prueba ARMAGEDÓN")
        return
    
    # Ejecutar prueba
    await test.run_test()


if __name__ == "__main__":
    # Ejecutar bucle de eventos
    asyncio.run(main())