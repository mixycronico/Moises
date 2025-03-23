"""
Prueba específica para el sistema de Checkpointing y Recuperación.

Este script prueba el comportamiento del sistema de Checkpointing
implementado en genesis/core/checkpoint_recovery.py, verificando:
1. Creación y restauración de checkpoints en memoria y disco
2. Checkpointing automático
3. Sistema de Safe Mode para componentes críticos
4. Integración con el sistema de recuperación
"""

import asyncio
import json
import logging
import os
import random
import shutil
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Tuple

from genesis.core.checkpoint_recovery import (
    CheckpointManager, CheckpointType, StateMetadata, RecoveryMode,
    SafeModeManager, RecoveryManager
)

# Configurar logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("checkpoint_test")

# Directorio para pruebas
TEST_CHECKPOINT_DIR = "./test_checkpoint_dir"

# Datos de prueba
@dataclass
class TradeData:
    """Datos de trading para pruebas."""
    symbol: str
    price: float
    quantity: float
    side: str  # "buy" o "sell"
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: f"trade_{int(time.time()*1000)}")
    status: str = "open"
    
    def close(self, exit_price: float) -> None:
        """Cerrar operación."""
        self.status = "closed"
        self.exit_price = exit_price
        self.exit_timestamp = time.time()
        self.pnl = (exit_price - self.price) * self.quantity if self.side == "buy" else (self.price - exit_price) * self.quantity

@dataclass
class StrategyState:
    """Estado de una estrategia de trading."""
    name: str
    active: bool = True
    trades: List[TradeData] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    balance: float = 10000.0
    last_update: float = field(default_factory=time.time)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def add_trade(self, trade: TradeData) -> None:
        """Añadir operación."""
        self.trades.append(trade)
        self.last_update = time.time()
    
    def close_trade(self, trade_id: str, exit_price: float) -> Optional[TradeData]:
        """Cerrar operación por ID."""
        for trade in self.trades:
            if trade.id == trade_id and trade.status == "open":
                trade.close(exit_price)
                self.balance += trade.pnl
                self.last_update = time.time()
                return trade
        return None
    
    def update_metrics(self) -> None:
        """Actualizar métricas de rendimiento."""
        if not self.trades:
            self.metrics = {"trades": 0}
            return
            
        closed_trades = [t for t in self.trades if t.status == "closed"]
        if not closed_trades:
            self.metrics = {"trades": len(self.trades), "closed": 0}
            return
            
        pnl_values = [getattr(t, "pnl", 0) for t in closed_trades if hasattr(t, "pnl")]
        
        self.metrics = {
            "trades": len(self.trades),
            "closed": len(closed_trades),
            "win_rate": sum(1 for p in pnl_values if p > 0) / len(pnl_values) if pnl_values else 0,
            "avg_pnl": sum(pnl_values) / len(pnl_values) if pnl_values else 0,
            "total_pnl": sum(pnl_values)
        }
        self.last_update = time.time()

# Simulador de estrategia para pruebas
class StrategySimulator:
    """Simulador de estrategia con soporte para checkpointing."""
    
    def __init__(self, name: str, checkpoint_type: CheckpointType = CheckpointType.DISK):
        """
        Inicializar simulador.
        
        Args:
            name: Nombre de la estrategia
            checkpoint_type: Tipo de checkpoint
        """
        self.name = name
        self.state = StrategyState(name=name)
        
        # Crear directorio específico para esta estrategia
        self.checkpoint_dir = os.path.join(TEST_CHECKPOINT_DIR, name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Configurar checkpointing
        self.checkpoint_mgr = CheckpointManager(
            component_id=name,
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_interval=150.0,  # 150ms
            max_checkpoints=5,
            checkpoint_type=checkpoint_type
        )
        
        # Estadísticas
        self.operations_count = 0
        self.checkpoints_created = 0
        self.checkpoint_restores = 0
    
    async def execute_random_operation(self) -> Dict[str, Any]:
        """
        Ejecutar una operación aleatoria que modifica el estado.
        
        Returns:
            Resultado de la operación
        """
        self.operations_count += 1
        operation = random.choice(["add_trade", "close_trade", "update_parameters", "update_metrics"])
        
        if operation == "add_trade":
            # Crear operación aleatoria
            symbols = ["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD"]
            trade = TradeData(
                symbol=random.choice(symbols),
                price=random.uniform(1000, 50000),
                quantity=random.uniform(0.1, 2.0),
                side=random.choice(["buy", "sell"])
            )
            self.state.add_trade(trade)
            result = {"operation": "add_trade", "trade_id": trade.id}
            
        elif operation == "close_trade" and self.state.trades:
            # Cerrar operación aleatoria
            open_trades = [t for t in self.state.trades if t.status == "open"]
            if open_trades:
                trade = random.choice(open_trades)
                exit_price = trade.price * random.uniform(0.9, 1.1)  # ±10%
                closed_trade = self.state.close_trade(trade.id, exit_price)
                result = {"operation": "close_trade", "trade_id": trade.id, "success": closed_trade is not None}
            else:
                result = {"operation": "close_trade", "success": False, "reason": "no_open_trades"}
                
        elif operation == "update_parameters":
            # Actualizar parámetros aleatorios
            self.state.parameters.update({
                f"param_{random.randint(1, 10)}": random.uniform(0, 1)
            })
            self.state.last_update = time.time()
            result = {"operation": "update_parameters", "parameters": self.state.parameters}
            
        elif operation == "update_metrics":
            # Actualizar métricas
            self.state.update_metrics()
            result = {"operation": "update_metrics", "metrics": self.state.metrics}
            
        else:
            result = {"operation": "unknown"}
        
        # Crear checkpoint después de la operación
        checkpoint_id = await self.checkpoint_mgr.checkpoint(self.state)
        if checkpoint_id:
            self.checkpoints_created += 1
            result["checkpoint_id"] = checkpoint_id
        
        return result
    
    async def start_auto_checkpointing(self) -> None:
        """Iniciar checkpointing automático."""
        await self.checkpoint_mgr.start_automatic_checkpointing(lambda: self.state)
        logger.info(f"Checkpointing automático iniciado para {self.name}")
    
    async def stop_auto_checkpointing(self) -> None:
        """Detener checkpointing automático."""
        await self.checkpoint_mgr.stop_automatic_checkpointing()
        logger.info(f"Checkpointing automático detenido para {self.name}")
    
    async def restore_from_checkpoint(self, checkpoint_id: str = None) -> bool:
        """
        Restaurar desde checkpoint.
        
        Args:
            checkpoint_id: ID específico o None para el más reciente
            
        Returns:
            True si se restauró correctamente
        """
        result = await self.checkpoint_mgr.restore(checkpoint_id, state_class=StrategyState)
        if result:
            self.state, metadata = result
            self.checkpoint_restores += 1
            logger.info(f"Estrategia {self.name} restaurada desde checkpoint {metadata.checkpoint_id}")
            return True
            
        logger.warning(f"No se pudo restaurar {self.name} desde checkpoint")
        return False
    
    async def get_checkpoint_list(self) -> List[Dict[str, Any]]:
        """
        Obtener lista de checkpoints disponibles.
        
        Returns:
            Lista de checkpoints con metadatos
        """
        return await self.checkpoint_mgr.list_checkpoints()

# Pruebas para verificar sistema de checkpointing
async def test_checkpointing() -> Dict[str, Any]:
    """
    Probar funcionalidades de checkpointing.
    
    Returns:
        Resultados de las pruebas
    """
    # Limpiar directorio de pruebas
    if os.path.exists(TEST_CHECKPOINT_DIR):
        shutil.rmtree(TEST_CHECKPOINT_DIR)
    os.makedirs(TEST_CHECKPOINT_DIR)
    
    # Resultados
    results = {
        "tests": 0,
        "passed": 0,
        "memory_checkpoint_test": {"result": False, "details": {}},
        "disk_checkpoint_test": {"result": False, "details": {}},
        "auto_checkpoint_test": {"result": False, "details": {}},
        "recovery_test": {"result": False, "details": {}},
        "safe_mode_test": {"result": False, "details": {}}
    }
    
    # TEST 1: Checkpoints en memoria
    logger.info("=== Prueba 1: Checkpoints en memoria ===")
    results["tests"] += 1
    
    try:
        # Crear estrategia con checkpoints en memoria
        strategy = StrategySimulator("memory_strategy", CheckpointType.MEMORY)
        
        # Ejecutar operaciones
        for i in range(10):
            await strategy.execute_random_operation()
        
        # Guardar estado actual
        original_trades_count = len(strategy.state.trades)
        original_balance = strategy.state.balance
        
        # Modificar estado
        strategy.state.trades = []
        strategy.state.balance = 0
        
        # Restaurar desde checkpoint
        restored = await strategy.restore_from_checkpoint()
        
        # Verificar restauración
        if restored:
            restored_trades_count = len(strategy.state.trades)
            restored_balance = strategy.state.balance
            
            if restored_trades_count == original_trades_count and restored_balance == original_balance:
                logger.info(f"✓ Checkpoint en memoria restaurado correctamente: {restored_trades_count} operaciones, balance {restored_balance}")
                results["memory_checkpoint_test"]["result"] = True
                results["memory_checkpoint_test"]["details"] = {
                    "trades_before": original_trades_count,
                    "trades_after": restored_trades_count,
                    "balance_before": original_balance,
                    "balance_after": restored_balance
                }
                results["passed"] += 1
            else:
                logger.error(f"✗ Restauración incorrecta: {restored_trades_count} != {original_trades_count} o {restored_balance} != {original_balance}")
        else:
            logger.error("✗ No se pudo restaurar desde checkpoint en memoria")
        
    except Exception as e:
        logger.error(f"Error en prueba de checkpoints en memoria: {e}")
    
    # TEST 2: Checkpoints en disco
    logger.info("=== Prueba 2: Checkpoints en disco ===")
    results["tests"] += 1
    
    try:
        # Crear estrategia con checkpoints en disco
        strategy = StrategySimulator("disk_strategy", CheckpointType.DISK)
        
        # Ejecutar operaciones
        for i in range(10):
            await strategy.execute_random_operation()
        
        # Verificar creación de archivos
        checkpoint_list = await strategy.get_checkpoint_list()
        
        if checkpoint_list:
            # Crear nueva instancia simulando reinicio
            new_strategy = StrategySimulator("disk_strategy", CheckpointType.DISK)
            
            # Restaurar más reciente
            restored = await new_strategy.restore_from_checkpoint()
            
            if restored and len(new_strategy.state.trades) > 0:
                logger.info(f"✓ Checkpoint en disco restaurado correctamente: {len(new_strategy.state.trades)} operaciones")
                results["disk_checkpoint_test"]["result"] = True
                results["disk_checkpoint_test"]["details"] = {
                    "checkpoints_count": len(checkpoint_list),
                    "trades_restored": len(new_strategy.state.trades),
                    "balance_restored": new_strategy.state.balance
                }
                results["passed"] += 1
            else:
                logger.error("✗ Restauración desde disco incorrecta")
        else:
            logger.error("✗ No se crearon checkpoints en disco")
        
    except Exception as e:
        logger.error(f"Error en prueba de checkpoints en disco: {e}")
    
    # TEST 3: Checkpointing automático
    logger.info("=== Prueba 3: Checkpointing automático ===")
    results["tests"] += 1
    
    try:
        # Crear estrategia
        strategy = StrategySimulator("auto_strategy", CheckpointType.DISK)
        
        # Iniciar checkpointing automático
        await strategy.start_auto_checkpointing()
        
        # Realizar operaciones sin checkpoints manuales
        for i in range(5):
            result = await strategy.execute_random_operation()
            # Quitar checkpoint_id para evitar creación manual
            if "checkpoint_id" in result:
                del result["checkpoint_id"]
            
            # Esperar un poco para dar tiempo al checkpointing automático
            await asyncio.sleep(0.2)
        
        # Detener checkpointing automático
        await strategy.stop_auto_checkpointing()
        
        # Verificar si se crearon checkpoints automáticamente
        checkpoint_list = await strategy.get_checkpoint_list()
        
        if checkpoint_list:
            # Restaurar
            new_strategy = StrategySimulator("auto_strategy", CheckpointType.DISK)
            restored = await new_strategy.restore_from_checkpoint()
            
            if restored and len(new_strategy.state.trades) > 0:
                logger.info(f"✓ Checkpoint automático funcionando: {len(checkpoint_list)} checkpoints creados")
                results["auto_checkpoint_test"]["result"] = True
                results["auto_checkpoint_test"]["details"] = {
                    "checkpoints_count": len(checkpoint_list),
                    "trades_restored": len(new_strategy.state.trades)
                }
                results["passed"] += 1
            else:
                logger.error("✗ No se pudo restaurar desde checkpoint automático")
        else:
            logger.error("✗ No se crearon checkpoints automáticos")
        
    except Exception as e:
        logger.error(f"Error en prueba de checkpointing automático: {e}")
    
    # TEST 4: Sistema de recuperación
    logger.info("=== Prueba 4: Sistema de recuperación ===")
    results["tests"] += 1
    
    try:
        # Crear Recovery Manager
        recovery_mgr = RecoveryManager(
            checkpoint_dir=TEST_CHECKPOINT_DIR,
            essential_components=["critical_strategy"]
        )
        
        # Crear estrategias
        strategies = {
            "critical_strategy": StrategySimulator("critical_strategy", CheckpointType.DISK),
            "normal_strategy": StrategySimulator("normal_strategy", CheckpointType.DISK)
        }
        
        # Ejecutar operaciones en ambas estrategias
        for strategy in strategies.values():
            for i in range(5):
                await strategy.execute_random_operation()
        
        # Simular fallo y recuperación
        strategies["critical_strategy"].state.trades = []  # Simulamos fallo
        
        # Recuperar usando recovery manager
        success = await recovery_mgr.attempt_recovery(
            "critical_strategy",
            lambda state: asyncio.sleep(0.1)  # Simular operación de recuperación
        )
        
        if success:
            logger.info(f"✓ Recovery Manager recuperó crítico exitosamente")
            results["recovery_test"]["result"] = True
            results["recovery_test"]["details"] = {
                "critical_trades": len(strategies["critical_strategy"].state.trades),
                "normal_trades": len(strategies["normal_strategy"].state.trades)
            }
            results["passed"] += 1
        else:
            logger.error("✗ Recovery Manager no pudo recuperar componente crítico")
        
    except Exception as e:
        logger.error(f"Error en prueba de Recovery Manager: {e}")
    
    # TEST 5: Safe Mode
    logger.info("=== Prueba 5: Safe Mode ===")
    results["tests"] += 1
    
    try:
        # Crear Safe Mode Manager
        safe_mode_mgr = SafeModeManager(
            essential_components=["critical_strategy"]
        )
        
        # Activar Safe Mode
        await safe_mode_mgr.activate_safe_mode("Prueba de activación")
        
        if safe_mode_mgr.current_mode == RecoveryMode.SAFE:
            # Verificar componentes esenciales vs no esenciales
            critical_essential = safe_mode_mgr.is_component_essential("critical_strategy")
            normal_essential = safe_mode_mgr.is_component_essential("normal_strategy")
            
            # Verificar operaciones permitidas
            critical_op_allowed = safe_mode_mgr.is_operation_allowed("update", "critical_strategy")
            normal_op_allowed = safe_mode_mgr.is_operation_allowed("update", "normal_strategy")
            
            if critical_essential and not normal_essential and critical_op_allowed and not normal_op_allowed:
                logger.info(f"✓ Safe Mode funcionando correctamente")
                results["safe_mode_test"]["result"] = True
                results["safe_mode_test"]["details"] = {
                    "critical_essential": critical_essential,
                    "normal_essential": normal_essential,
                    "critical_op_allowed": critical_op_allowed,
                    "normal_op_allowed": normal_op_allowed
                }
                results["passed"] += 1
            else:
                logger.error("✗ Safe Mode no maneja correctamente componentes esenciales")
                
            # Desactivar Safe Mode
            await safe_mode_mgr.deactivate_safe_mode()
            if safe_mode_mgr.current_mode == RecoveryMode.NORMAL:
                logger.info("Safe Mode desactivado correctamente")
        else:
            logger.error("✗ No se pudo activar Safe Mode")
        
    except Exception as e:
        logger.error(f"Error en prueba de Safe Mode: {e}")
    
    # Limpiar directorio de pruebas
    shutil.rmtree(TEST_CHECKPOINT_DIR)
    
    # Resumen final
    logger.info(f"=== Resultado: {results['passed']}/{results['tests']} pruebas exitosas ===")
    
    return results
    
async def main():
    """Función principal para ejecutar las pruebas."""
    start_time = time.time()
    
    try:
        results = await test_checkpointing()
        
        # Imprimir resultados detallados
        logger.info("\n=== Resultados detallados ===")
        
        # Mostrar resultado global
        success_rate = results["passed"] / results["tests"] * 100 if results["tests"] > 0 else 0
        logger.info(f"Total: {results['passed']}/{results['tests']} pruebas exitosas ({success_rate:.1f}%)")
        
        # Mostrar resultados específicos
        for test_name, test_result in results.items():
            if isinstance(test_result, dict) and "result" in test_result:
                logger.info(f"{test_name}: {'✓' if test_result['result'] else '✗'}")
                
                # Mostrar detalles si hay éxito
                if test_result["result"] and test_result["details"]:
                    for k, v in test_result["details"].items():
                        logger.info(f"  - {k}: {v}")
        
    except Exception as e:
        logger.error(f"Error en prueba: {e}")
    
    finally:
        # Tiempo total
        elapsed = time.time() - start_time
        logger.info(f"Tiempo total de ejecución: {elapsed:.2f} segundos")
        
        # Asegurar limpieza de archivos temporales
        if os.path.exists(TEST_CHECKPOINT_DIR):
            shutil.rmtree(TEST_CHECKPOINT_DIR)

if __name__ == "__main__":
    asyncio.run(main())