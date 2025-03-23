"""
Prueba simple del sistema de Checkpointing y Safe Mode.

Esta es una versión simplificada para demostrar el funcionamiento básico
del sistema de checkpointing y safe mode implementado en Genesis.
"""

import asyncio
import logging
import os
import shutil
import time
import random
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

# Configurar logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("simple_checkpoint_test")

# Directorio para pruebas
TEST_DIR = "./test_checkpoint_simple"

# Definir estado del sistema
class SystemMode(Enum):
    """Modos de operación del sistema."""
    NORMAL = "normal"     # Funcionamiento normal
    SAFE = "safe"         # Modo seguro
    EMERGENCY = "emergency"  # Modo emergencia

@dataclass
class SystemState:
    """Estado del sistema para checkpointing."""
    component_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def update(self, key: str, value: Any) -> None:
        """Actualizar un valor en el estado."""
        self.data[key] = value
        self.updated_at = time.time()

class SimpleCheckpoint:
    """Sistema de checkpointing simplificado."""
    
    def __init__(
        self,
        component_id: str,
        checkpoint_dir: str,
        auto_checkpoint: bool = True,
        checkpoint_interval: float = 1.0  # 1 segundo para pruebas
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
        self.checkpoint_dir = os.path.join(checkpoint_dir, component_id)
        self.auto_checkpoint = auto_checkpoint
        self.checkpoint_interval = checkpoint_interval
        
        # Crear directorio
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Estado inicial
        self.state = SystemState(component_id=component_id)
        self.last_checkpoint_time = 0
        self._checkpoint_task = None
        
        logger.info(f"Sistema de checkpointing creado para {component_id}")
    
    async def start(self) -> None:
        """Iniciar checkpointing automático."""
        if self.auto_checkpoint and not self._checkpoint_task:
            self._checkpoint_task = asyncio.create_task(self._auto_checkpoint_loop())
            logger.info(f"Checkpointing automático iniciado para {self.component_id}")
    
    async def stop(self) -> None:
        """Detener checkpointing automático."""
        if self._checkpoint_task:
            self._checkpoint_task.cancel()
            try:
                await self._checkpoint_task
            except asyncio.CancelledError:
                pass
            self._checkpoint_task = None
            logger.info(f"Checkpointing automático detenido para {self.component_id}")
    
    async def _auto_checkpoint_loop(self) -> None:
        """Bucle de checkpointing automático."""
        try:
            while True:
                await asyncio.sleep(self.checkpoint_interval)
                await self.create_checkpoint()
        except asyncio.CancelledError:
            logger.info(f"Bucle de checkpointing cancelado para {self.component_id}")
            raise
    
    async def create_checkpoint(self) -> str:
        """
        Crear checkpoint del estado actual.
        
        Returns:
            ID del checkpoint creado
        """
        # Crear ID basado en timestamp
        checkpoint_id = f"{int(time.time() * 1000)}"
        
        # Ruta del archivo
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{checkpoint_id}.txt")
        
        # Guardar estado como texto
        with open(checkpoint_path, "w") as f:
            f.write(f"Component: {self.component_id}\n")
            f.write(f"Created: {time.ctime()}\n")
            f.write(f"Updated: {time.ctime(self.state.updated_at)}\n")
            f.write("Data:\n")
            for key, value in self.state.data.items():
                f.write(f"  {key}: {value}\n")
        
        self.last_checkpoint_time = time.time()
        logger.info(f"Checkpoint {checkpoint_id} creado para {self.component_id}")
        
        return checkpoint_id
    
    async def restore_latest(self) -> bool:
        """
        Restaurar último checkpoint.
        
        Returns:
            True si se restauró correctamente
        """
        # Obtener archivos de checkpoint ordenados por timestamp (más reciente primero)
        try:
            files = os.listdir(self.checkpoint_dir)
            checkpoint_files = [f for f in files if f.startswith("checkpoint_")]
            
            if not checkpoint_files:
                logger.warning(f"No hay checkpoints para {self.component_id}")
                return False
            
            # Ordenar por timestamp en nombre (formato: checkpoint_TIMESTAMP.txt)
            checkpoint_files.sort(reverse=True)
            latest = checkpoint_files[0]
            
            # Leer archivo
            checkpoint_path = os.path.join(self.checkpoint_dir, latest)
            data = {}
            
            with open(checkpoint_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("  "):  # Línea de datos
                        parts = line.strip().split(": ", 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            data[key] = value
            
            # Restaurar estado
            self.state.data = data
            self.state.updated_at = time.time()
            
            logger.info(f"Checkpoint {latest} restaurado para {self.component_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error al restaurar checkpoint: {e}")
            return False

class SimpleSafeMode:
    """Implementación simplificada de Safe Mode."""
    
    def __init__(self, essential_components: List[str]):
        """
        Inicializar Safe Mode.
        
        Args:
            essential_components: Lista de componentes esenciales
        """
        self.essential_components = set(essential_components)
        self.mode = SystemMode.NORMAL
        self.mode_change_time = time.time()
        
        logger.info(f"Safe Mode iniciado con {len(essential_components)} componentes esenciales")
    
    def activate_safe_mode(self, reason: str) -> None:
        """
        Activar modo seguro.
        
        Args:
            reason: Razón para activar
        """
        if self.mode != SystemMode.SAFE:
            self.mode = SystemMode.SAFE
            self.mode_change_time = time.time()
            logger.warning(f"SAFE MODE ACTIVADO. Razón: {reason}")
    
    def activate_emergency_mode(self, reason: str) -> None:
        """
        Activar modo emergencia.
        
        Args:
            reason: Razón para activar
        """
        self.mode = SystemMode.EMERGENCY
        self.mode_change_time = time.time()
        logger.critical(f"EMERGENCY MODE ACTIVADO. Razón: {reason}")
    
    def deactivate(self) -> None:
        """Volver a modo normal."""
        if self.mode != SystemMode.NORMAL:
            self.mode = SystemMode.NORMAL
            self.mode_change_time = time.time()
            logger.info("Modo seguro/emergencia desactivado")
    
    def is_essential(self, component_id: str) -> bool:
        """
        Verificar si un componente es esencial.
        
        Args:
            component_id: ID del componente
            
        Returns:
            True si es esencial
        """
        return component_id in self.essential_components
    
    def is_operation_allowed(self, operation: str, component_id: str) -> bool:
        """
        Verificar si una operación está permitida en el modo actual.
        
        Args:
            operation: Operación a realizar
            component_id: ID del componente
            
        Returns:
            True si la operación está permitida
        """
        # En modo normal, todo permitido
        if self.mode == SystemMode.NORMAL:
            return True
        
        # En modo emergencia, solo operaciones en componentes esenciales
        if self.mode == SystemMode.EMERGENCY:
            return self.is_essential(component_id)
        
        # En modo seguro, operaciones en componentes esenciales + algunas en no esenciales
        if self.mode == SystemMode.SAFE:
            if self.is_essential(component_id):
                return True
            else:
                # En componentes no esenciales, solo permitir lectura
                return operation.startswith("get") or operation.startswith("read")
        
        return False

# Componente simulado para pruebas
class TestComponent:
    """Componente de prueba con checkpointing."""
    
    def __init__(
        self, 
        component_id: str, 
        checkpoint_dir: str,
        essential: bool = False
    ):
        """
        Inicializar componente.
        
        Args:
            component_id: ID del componente
            checkpoint_dir: Directorio para checkpoints
            essential: Si es un componente esencial
        """
        self.component_id = component_id
        self.essential = essential
        self.data = {}
        
        # Crear sistema de checkpointing
        self.checkpoint_system = SimpleCheckpoint(
            component_id=component_id,
            checkpoint_dir=checkpoint_dir,
            auto_checkpoint=True
        )
    
    async def start(self) -> None:
        """Iniciar componente."""
        await self.checkpoint_system.start()
    
    async def stop(self) -> None:
        """Detener componente."""
        await self.checkpoint_system.stop()
    
    async def set_data(self, key: str, value: Any) -> Dict[str, Any]:
        """
        Establecer datos.
        
        Args:
            key: Clave
            value: Valor
            
        Returns:
            Resultado de la operación
        """
        self.data[key] = value
        self.checkpoint_system.state.update(key, value)
        return {"status": "success", "operation": "set", "key": key}
    
    async def get_data(self, key: str) -> Dict[str, Any]:
        """
        Obtener datos.
        
        Args:
            key: Clave
            
        Returns:
            Resultado de la operación
        """
        if key in self.data:
            return {"status": "success", "operation": "get", "key": key, "value": self.data[key]}
        else:
            return {"status": "error", "operation": "get", "key": key, "reason": "not_found"}
    
    async def execute_operation(
        self, 
        operation: str, 
        params: Dict[str, Any],
        safe_mode: SimpleSafeMode
    ) -> Dict[str, Any]:
        """
        Ejecutar operación verificando permisos de Safe Mode.
        
        Args:
            operation: Operación a realizar
            params: Parámetros
            safe_mode: Sistema de Safe Mode
            
        Returns:
            Resultado de la operación o error si no está permitida
        """
        # Verificar si la operación está permitida
        if not safe_mode.is_operation_allowed(operation, self.component_id):
            return {
                "status": "error", 
                "operation": operation, 
                "reason": f"not_allowed_in_{safe_mode.mode.value}_mode"
            }
        
        # Ejecutar operación
        if operation == "set":
            return await self.set_data(params.get("key", ""), params.get("value", ""))
        elif operation == "get":
            return await self.get_data(params.get("key", ""))
        else:
            return {"status": "error", "operation": operation, "reason": "unknown_operation"}
    
    async def simulate_crash(self) -> None:
        """Simular un fallo que requiere restauración."""
        logger.warning(f"Componente {self.component_id} fallando...")
        self.data = {}  # Perder todos los datos
    
    async def restore(self) -> bool:
        """
        Restaurar estado desde último checkpoint.
        
        Returns:
            True si se restauró correctamente
        """
        success = await self.checkpoint_system.restore_latest()
        if success:
            # Extraer datos de state
            self.data = {}
            for key, value in self.checkpoint_system.state.data.items():
                self.data[key] = value
            logger.info(f"Componente {self.component_id} restaurado exitosamente")
        return success

async def main():
    """Función principal."""
    logger.info("=== Prueba de Checkpointing y Safe Mode ===")
    
    # Limpiar directorio de pruebas
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    try:
        # Crear sistema de Safe Mode
        safe_mode = SimpleSafeMode(essential_components=["essential_component"])
        
        # Crear componentes
        components = {
            "essential_component": TestComponent("essential_component", TEST_DIR, essential=True),
            "non_essential_component": TestComponent("non_essential_component", TEST_DIR, essential=False)
        }
        
        # Iniciar componentes
        for component in components.values():
            await component.start()
        
        # 1. Prueba de operaciones normales
        logger.info("\n1. Operaciones en modo normal:")
        
        for comp_id, component in components.items():
            result = await component.execute_operation(
                "set", {"key": "test_key", "value": f"test_value_{comp_id}"}, safe_mode
            )
            logger.info(f"Componente {comp_id} SET: {result}")
            
            result = await component.execute_operation(
                "get", {"key": "test_key"}, safe_mode
            )
            logger.info(f"Componente {comp_id} GET: {result}")
        
        # 2. Activar Safe Mode
        logger.info("\n2. Activando Safe Mode:")
        safe_mode.activate_safe_mode("Prueba de Safe Mode")
        
        # Probar operaciones en Safe Mode
        logger.info("Operaciones en Safe Mode:")
        
        for comp_id, component in components.items():
            comp_type = "esencial" if component.essential else "no esencial"
            
            # Probar escritura
            result = await component.execute_operation(
                "set", {"key": "safe_key", "value": f"safe_value_{comp_id}"}, safe_mode
            )
            logger.info(f"Componente {comp_id} ({comp_type}) SET: {result}")
            
            # Probar lectura
            result = await component.execute_operation(
                "get", {"key": "test_key"}, safe_mode
            )
            logger.info(f"Componente {comp_id} ({comp_type}) GET: {result}")
        
        # 3. Prueba de checkpointing
        logger.info("\n3. Prueba de checkpointing:")
        
        # Establecer más datos
        essential = components["essential_component"]
        await essential.execute_operation(
            "set", {"key": "important_data", "value": "critical_value"}, safe_mode
        )
        
        # Forzar checkpoint
        checkpoint_id = await essential.checkpoint_system.create_checkpoint()
        logger.info(f"Checkpoint creado: {checkpoint_id}")
        
        # Simular crash
        logger.info("Simulando fallo del componente...")
        await essential.simulate_crash()
        
        # Verificar datos perdidos
        result = await essential.execute_operation(
            "get", {"key": "important_data"}, safe_mode
        )
        logger.info(f"Después del fallo: {result}")
        
        # Restaurar desde checkpoint
        logger.info("Restaurando desde checkpoint...")
        success = await essential.restore()
        
        # Verificar datos restaurados
        if success:
            result = await essential.execute_operation(
                "get", {"key": "important_data"}, safe_mode
            )
            logger.info(f"Después de restauración: {result}")
        
        # 4. Volver a modo normal
        logger.info("\n4. Volviendo a modo normal:")
        safe_mode.deactivate()
        
        # Probar que todo funciona en modo normal
        for comp_id, component in components.items():
            result = await component.execute_operation(
                "set", {"key": "normal_key", "value": f"normal_value_{comp_id}"}, safe_mode
            )
            logger.info(f"Componente {comp_id} SET: {result}")
        
        # Resumen
        logger.info("\n=== Resumen ===")
        for comp_id, component in components.items():
            logger.info(f"Componente {comp_id}: {len(component.data)} valores almacenados")
        
    finally:
        # Detener componentes
        for component in components.values():
            await component.stop()
        
        # Limpiar directorio de pruebas
        shutil.rmtree(TEST_DIR)

if __name__ == "__main__":
    asyncio.run(main())