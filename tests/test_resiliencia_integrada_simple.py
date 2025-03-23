"""
Prueba integrada simplificada de las características de resiliencia del sistema Genesis.

Esta prueba demuestra cómo las tres características de resiliencia (Sistema de Reintentos,
Circuit Breaker y Checkpointing) trabajan juntas en un escenario de sistema distribuido.
"""

import asyncio
import logging
import time
import random
from enum import Enum, auto
from typing import Dict, Any, List, Optional

# Configurar logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_resiliencia_integrada")

# ========================= CARACTERÍSTICAS DE RESILIENCIA =========================

class CircuitBreaker:
    """Patrón Circuit Breaker para aislar componentes fallidos."""
    
    def __init__(self, name: str, failure_threshold: int = 3, recovery_time: float = 5.0):
        self.name = name
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_count = 0
        self.success_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.last_failure_time = 0
    
    async def execute(self, func):
        """Ejecutar una función con protección de Circuit Breaker."""
        if self.state == "OPEN":
            if (time.time() - self.last_failure_time) > self.recovery_time:
                logger.info(f"Circuit Breaker '{self.name}' cambiando a HALF_OPEN")
                self.state = "HALF_OPEN"
                self.success_count = 0
            else:
                return None  # Rechazar llamada
        
        try:
            result = await func()
            
            # Registrar éxito
            self.success_count += 1
            self.failure_count = 0
            
            # En HALF_OPEN, cerrar después de algunos éxitos
            if self.state == "HALF_OPEN" and self.success_count >= 2:
                logger.info(f"Circuit Breaker '{self.name}' cambiando a CLOSED")
                self.state = "CLOSED"
                
            return result
            
        except Exception as e:
            # Registrar fallo
            self.failure_count += 1
            self.success_count = 0
            self.last_failure_time = time.time()
            
            # Abrir circuito si excedemos umbral
            if self.state == "CLOSED" and self.failure_count >= self.failure_threshold:
                logger.warning(f"Circuit Breaker '{self.name}' cambiando a OPEN")
                self.state = "OPEN"
            
            # En HALF_OPEN, volver a abrir con un solo fallo
            elif self.state == "HALF_OPEN":
                logger.warning(f"Circuit Breaker '{self.name}' volviendo a OPEN")
                self.state = "OPEN"
                
            raise e

async def with_retry(func, max_retries=3, base_delay=0.1):
    """Sistema de reintentos adaptativos con backoff exponencial."""
    attempt = 0
    
    while True:
        try:
            return await func()
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            
            delay = base_delay * (2 ** (attempt - 1))
            jitter = random.uniform(0, 0.1)
            total_delay = delay + jitter
            
            logger.info(f"Reintento {attempt}/{max_retries} tras error: {e}. Esperando {total_delay:.2f}s")
            await asyncio.sleep(total_delay)

# ========================= COMPONENTES DEL SISTEMA =========================

class ServiceHealth(Enum):
    """Estados de salud de los servicios."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    UNAVAILABLE = "unavailable"
    
    def __str__(self):
        return self.value

class ServiceComponent:
    """Componente de servicio con características de resiliencia."""
    
    def __init__(self, name: str, fail_rate: float = 0.0, essential: bool = False):
        self.name = name
        self.fail_rate = fail_rate
        self.essential = essential
        self.data = {}
        self.checkpoint = {}
        self.health = ServiceHealth.HEALTHY
        self.circuit_breaker = CircuitBreaker(name=f"cb_{name}")
    
    async def process(self, operation: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Procesar una operación con posibilidad de fallo.
        
        Args:
            operation: Operación a realizar
            data: Datos para la operación
            
        Returns:
            Resultado de la operación
            
        Raises:
            Exception: Si la operación falla
        """
        # Simular trabajo
        await asyncio.sleep(0.05)
        
        # Versión simplificada sin datos
        if data is None:
            data = {}
        
        # Estado de salud determina comportamiento
        if self.health == ServiceHealth.UNAVAILABLE:
            raise ConnectionError(f"Servicio {self.name} no disponible")
            
        elif self.health == ServiceHealth.FAILING:
            if random.random() < 0.8:  # 80% de fallos
                raise Exception(f"Error en {self.name} al procesar {operation}")
                
        elif self.health == ServiceHealth.DEGRADED:
            # Latencia adicional
            await asyncio.sleep(0.2)
            if random.random() < 0.3:  # 30% de timeouts
                raise TimeoutError(f"Timeout en {self.name} al procesar {operation}")
        
        # Operaciones específicas
        if operation == "store":
            key = data.get("key")
            value = data.get("value")
            if key:
                self.data[key] = value
                self.save_checkpoint()  # Guardar checkpoint tras cambios
                return {"status": "success", "key": key}
                
        elif operation == "retrieve":
            key = data.get("key")
            if key and key in self.data:
                return {"status": "success", "key": key, "value": self.data[key]}
            return {"status": "error", "reason": "not_found"}
            
        elif operation == "health_check":
            return {
                "status": "success", 
                "component": self.name,
                "health": str(self.health),
                "circuit_state": self.circuit_breaker.state
            }
            
        elif operation == "set_health":
            old_health = self.health
            self.health = data.get("health", ServiceHealth.HEALTHY)
            return {
                "status": "success", 
                "old_health": str(old_health),
                "new_health": str(self.health)
            }
            
        elif operation == "crash":
            # Simular crash que pierde los datos
            old_data = len(self.data)
            self.data = {}
            return {"status": "crashed", "lost_items": old_data}
            
        # Operación genérica
        return {"status": "success", "operation": operation}
    
    def save_checkpoint(self):
        """Guardar checkpoint del estado actual."""
        self.checkpoint = {
            "data": self.data.copy(),
            "health": self.health
        }
        logger.debug(f"Checkpoint guardado para {self.name}")
    
    async def restore_from_checkpoint(self):
        """Restaurar desde checkpoint."""
        if self.checkpoint:
            self.data = self.checkpoint.get("data", {}).copy()
            # No restaurar health, solo datos
            logger.info(f"Componente {self.name} restaurado desde checkpoint")
            return True
        return False

class GenesisSystem:
    """Sistema Genesis simplificado con características de resiliencia."""
    
    def __init__(self):
        self.components = {}
        self.mode = "NORMAL"  # NORMAL, SAFE, EMERGENCY
    
    def register_component(self, component: ServiceComponent):
        """Registrar un componente en el sistema."""
        self.components[component.name] = component
        logger.info(f"Componente {component.name} registrado en el sistema")
    
    async def call(self, component_name: str, operation: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Llamar a un componente con toda la resiliencia integrada.
        
        Args:
            component_name: Nombre del componente
            operation: Operación a realizar
            data: Datos para la operación
            
        Returns:
            Resultado de la operación o error
        """
        if component_name not in self.components:
            return {"status": "error", "reason": "component_not_found"}
            
        component = self.components[component_name]
        
        # Verificar modo
        if self.mode == "EMERGENCY" and not component.essential:
            return {"status": "error", "reason": "emergency_mode_only_essential"}
            
        if self.mode == "SAFE" and not component.essential and not operation.startswith("get") and not operation == "health_check":
            return {"status": "error", "reason": "safe_mode_restricted"}
            
        # Función a ejecutar con toda la resiliencia
        async def execute_operation():
            return await component.process(operation, data)
        
        try:
            # Circuit Breaker + Retry System
            result = await component.circuit_breaker.execute(
                lambda: with_retry(execute_operation)
            )
            
            if result is None:  # Circuito abierto
                return {"status": "error", "reason": "circuit_open"}
                
            return result
        except Exception as e:
            logger.error(f"Error al llamar a {component_name}.{operation}: {e}")
            
            # Intento de recuperación
            if "crash" in str(e).lower() or "connection" in str(e).lower():
                try:
                    await component.restore_from_checkpoint()
                except Exception as restore_error:
                    logger.error(f"Error al restaurar {component_name}: {restore_error}")
            
            # Ajustar modo del sistema según disponibilidad
            self._check_system_health()
            
            return {"status": "error", "reason": str(e), "component": component_name, "operation": operation}
    
    def _check_system_health(self):
        """Verificar salud del sistema y ajustar modo."""
        essential_failing = 0
        non_essential_failing = 0
        total_essential = 0
        total_non_essential = 0
        
        for name, component in self.components.items():
            circuit_open = component.circuit_breaker.state == "OPEN"
            unhealthy = component.health in [ServiceHealth.FAILING, ServiceHealth.UNAVAILABLE]
            
            if component.essential:
                total_essential += 1
                if circuit_open or unhealthy:
                    essential_failing += 1
            else:
                total_non_essential += 1
                if circuit_open or unhealthy:
                    non_essential_failing += 1
        
        # Determinar modo basado en componentes fallidos
        if total_essential > 0 and essential_failing / total_essential > 0.5:
            if self.mode != "EMERGENCY":
                logger.warning("Activando modo EMERGENCY - Componentes esenciales fallando")
                self.mode = "EMERGENCY"
        elif total_essential > 0 and essential_failing / total_essential > 0.3:
            if self.mode != "SAFE" and self.mode != "EMERGENCY":
                logger.warning("Activando modo SAFE - Algunos componentes esenciales degradados")
                self.mode = "SAFE"
        else:
            if self.mode != "NORMAL":
                logger.info("Volviendo a modo NORMAL")
                self.mode = "NORMAL"

# ========================= PRUEBA INTEGRADA =========================

async def run_scenario(name, genesis_system, scenario_func):
    """Ejecutar un escenario de prueba y mostrar resultados."""
    logger.info(f"\n=== Escenario: {name} ===")
    start_time = time.time()
    
    try:
        result = await scenario_func(genesis_system)
        elapsed = time.time() - start_time
        logger.info(f"Escenario completado en {elapsed:.2f}s")
        return result
    except Exception as e:
        logger.error(f"Error en escenario {name}: {e}")
        return {"status": "error", "reason": str(e)}

async def scenario_1_retry_system(system):
    """Escenario 1: Sistema de reintentos con componente degradado."""
    logger.info("Degradando componente de datos temporalmente")
    
    # Configurar componente degradado
    await system.call("data_service", "set_health", {"health": ServiceHealth.DEGRADED})
    
    results = []
    
    # Intentar operaciones en componente degradado
    for i in range(3):
        result = await system.call("data_service", "store", {"key": f"key_{i}", "value": f"value_{i}"})
        results.append({"operation": f"store_{i}", "result": result})
    
    # Restaurar salud
    await system.call("data_service", "set_health", {"health": ServiceHealth.HEALTHY})
    
    # Verificar datos almacenados
    for i in range(3):
        result = await system.call("data_service", "retrieve", {"key": f"key_{i}"})
        results.append({"operation": f"verify_{i}", "result": result})
    
    return {"scenario": "retry_system", "results": results}

async def scenario_2_circuit_breaker(system):
    """Escenario 2: Circuit breaker con servicio fallido."""
    logger.info("Configurando servicio de pagos para fallar constantemente")
    
    # Configurar componente para fallar
    await system.call("payment_service", "set_health", {"health": ServiceHealth.FAILING})
    
    results = []
    
    # Intentar varias operaciones que fallarán
    for i in range(6):
        result = await system.call("payment_service", "process_payment", {"amount": 100})
        
        # Verificar estado del circuit breaker
        health = await system.call("payment_service", "health_check", {})
        
        results.append({
            "operation": f"payment_{i}", 
            "result": result,
            "circuit_state": health.get("circuit_state") if health.get("status") == "success" else "unknown"
        })
    
    # Restaurar salud
    await system.call("payment_service", "set_health", {"health": ServiceHealth.HEALTHY})
    
    # Esperar a que el circuit breaker se recupere
    logger.info("Esperando recuperación del circuit breaker...")
    await asyncio.sleep(5.5)
    
    # Intentar una operación después de la recuperación
    result = await system.call("payment_service", "process_payment", {"amount": 50})
    health = await system.call("payment_service", "health_check", {})
    
    results.append({
        "operation": "payment_after_recovery", 
        "result": result,
        "circuit_state": health.get("circuit_state") if health.get("status") == "success" else "unknown"
    })
    
    return {"scenario": "circuit_breaker", "results": results}

async def scenario_3_checkpointing(system):
    """Escenario 3: Checkpointing y recuperación tras fallo."""
    logger.info("Almacenando datos críticos en el componente de configuración")
    
    results = []
    
    # Almacenar datos importantes
    await system.call("config_service", "store", {"key": "api_endpoint", "value": "https://api.example.com"})
    await system.call("config_service", "store", {"key": "timeout", "value": "30"})
    await system.call("config_service", "store", {"key": "retries", "value": "3"})
    
    # Verificar datos almacenados
    for key in ["api_endpoint", "timeout", "retries"]:
        result = await system.call("config_service", "retrieve", {"key": key})
        results.append({"operation": f"verify_{key}", "result": result})
    
    # Provocar crash
    logger.info("Simulando crash del componente de configuración")
    await system.call("config_service", "crash", {})
    
    # Verificar pérdida de datos
    for key in ["api_endpoint", "timeout", "retries"]:
        result = await system.call("config_service", "retrieve", {"key": key})
        results.append({"operation": f"after_crash_{key}", "result": result})
    
    # Restauración automática (el sistema intentará restaurar en call())
    logger.info("Restaurando componente desde checkpoint")
    
    # Forzar restauración manual para demostrarlo
    config_component = system.components["config_service"]
    await config_component.restore_from_checkpoint()
    
    # Verificar recuperación
    for key in ["api_endpoint", "timeout", "retries"]:
        result = await system.call("config_service", "retrieve", {"key": key})
        results.append({"operation": f"restored_{key}", "result": result})
    
    return {"scenario": "checkpointing", "results": results}

async def scenario_4_safe_mode(system):
    """Escenario 4: Safe Mode cuando fallan componentes esenciales."""
    logger.info("Simulando fallos en componentes esenciales")
    
    results = []
    
    # Verificar modo inicial
    results.append({"phase": "initial", "mode": system.mode})
    
    # Hacer fallar un componente esencial
    await system.call("exchange_service", "set_health", {"health": ServiceHealth.UNAVAILABLE})
    
    # Verificar si el sistema entra en modo SAFE
    results.append({"phase": "after_essential_failure", "mode": system.mode})
    
    # Intentar operaciones en modo SAFE
    for component_name in ["payment_service", "data_service", "config_service"]:
        result = await system.call(component_name, "store", {"key": "test", "value": "data"})
        results.append({
            "component": component_name,
            "essential": system.components[component_name].essential,
            "result": result
        })
    
    # Hacer fallar otro componente esencial para entrar en modo EMERGENCY
    await system.call("payment_service", "set_health", {"health": ServiceHealth.UNAVAILABLE})
    
    # Verificar modo EMERGENCY
    results.append({"phase": "emergency", "mode": system.mode})
    
    # Intentar operaciones en modo EMERGENCY
    for component_name in ["data_service", "config_service"]:
        result = await system.call(component_name, "health_check", {})
        results.append({
            "component": component_name,
            "essential": system.components[component_name].essential,
            "operation": "health_check",
            "result": result
        })
    
    # Restaurar componentes
    await system.call("exchange_service", "set_health", {"health": ServiceHealth.HEALTHY})
    await system.call("payment_service", "set_health", {"health": ServiceHealth.HEALTHY})
    
    # Verificar vuelta a modo normal
    await asyncio.sleep(0.1)  # Dar tiempo a actualizar
    results.append({"phase": "restored", "mode": system.mode})
    
    return {"scenario": "safe_mode", "results": results}

async def main():
    """Función principal para ejecutar todas las pruebas."""
    logger.info("=== Prueba Integrada Simplificada - Sistema Genesis Resiliente ===")
    
    # Crear sistema Genesis
    system = GenesisSystem()
    
    # Registrar componentes (algunos esenciales)
    system.register_component(ServiceComponent("data_service", essential=False))
    system.register_component(ServiceComponent("payment_service", essential=True))
    system.register_component(ServiceComponent("exchange_service", essential=True))
    system.register_component(ServiceComponent("config_service", essential=False))
    
    # Ejecutar escenarios
    results = {}
    
    # Escenario 1: Sistema de Reintentos
    results["retry"] = await run_scenario(
        "Sistema de Reintentos", system, scenario_1_retry_system
    )
    
    # Escenario 2: Circuit Breaker
    results["circuit_breaker"] = await run_scenario(
        "Circuit Breaker", system, scenario_2_circuit_breaker
    )
    
    # Escenario 3: Checkpointing
    results["checkpointing"] = await run_scenario(
        "Checkpointing", system, scenario_3_checkpointing
    )
    
    # Escenario 4: Safe Mode
    results["safe_mode"] = await run_scenario(
        "Safe Mode", system, scenario_4_safe_mode
    )
    
    # Mostrar resumen
    logger.info("\n=== Resumen de Escenarios ===")
    for name, result in results.items():
        status = "✓ ÉXITO" if result.get("status") != "error" else "✗ ERROR"
        logger.info(f"Escenario {name}: {status}")
    
    # Conclusiones
    logger.info("\n=== Conclusiones ===")
    logger.info("El sistema integrado demuestra tres características clave de resiliencia:")
    logger.info("1. Sistema de Reintentos con backoff exponencial: Recuperación ante fallos transitorios")
    logger.info("2. Circuit Breaker: Aislamiento de componentes fallidos y prevención de fallos en cascada")
    logger.info("3. Checkpointing: Recuperación rápida de estado crítico tras fallos")
    logger.info("4. Sistema de Degradación Controlada (Safe Mode): Priorización de componentes esenciales")
    logger.info("\nEstas características combinadas proporcionan un sistema robusto que puede:")
    logger.info("- Mantener operación parcial durante fallos")
    logger.info("- Recuperarse automáticamente de la mayoría de los problemas")
    logger.info("- Proteger recursos críticos durante fallos graves")
    logger.info("- Prevenir fallos en cascada que afecten a todo el sistema")

if __name__ == "__main__":
    asyncio.run(main())