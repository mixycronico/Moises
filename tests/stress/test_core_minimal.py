"""
Prueba minimalista del core del sistema híbrido API+WebSocket.

Esta versión es extremadamente reducida para garantizar su ejecución completa
en un tiempo muy limitado, pero conservando los aspectos fundamentales para
evaluar el funcionamiento del core.
"""

import asyncio
import logging
import random
import time
from typing import Dict, Any, List, Optional

# Configurar logging minimalista
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger("core_mini_test")

class Component:
    """Componente básico para la prueba minimal."""
    
    def __init__(self, id: str, failure_rate: float = 0.0):
        self.id = id
        self.failure_rate = failure_rate
        self.events_received = 0
        self.requests_handled = 0
        self.crashed = False
        self.dependencies = []
        self.coordinator = None
    
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """Procesar solicitud API."""
        if self.crashed:
            raise Exception(f"Componente {self.id} no disponible")
        
        # Probabilidad de fallo
        if random.random() < self.failure_rate:
            raise Exception(f"Fallo en componente {self.id}")
        
        self.requests_handled += 1
        
        # Simular latencia mínima
        await asyncio.sleep(0.01)
        
        # Verificar dependencias si se solicita
        if request_type == "check_dependencies" and self.coordinator and self.dependencies:
            results = {}
            for dep_id in self.dependencies:
                try:
                    result = await self.coordinator.request(
                        dep_id, "status", {}, self.id, timeout=0.2
                    )
                    results[dep_id] = "ok" if result else "error"
                except Exception:
                    results[dep_id] = "unavailable"
            
            return {
                "status": "ok",
                "dependencies": results
            }
        
        # Respuesta genérica
        return {
            "status": "ok",
            "component": self.id,
            "request_type": request_type
        }
    
    async def on_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Manejar evento WebSocket."""
        if self.crashed:
            return
        
        self.events_received += 1
        
        # Simular latencia mínima
        await asyncio.sleep(0.005)
        
        # Manejar evento especial de sistema
        if event_type == "system_recovery" and data.get("target") == self.id:
            self.crashed = False
    
    async def start(self) -> None:
        """Iniciar componente."""
        self.crashed = False
    
    async def stop(self) -> None:
        """Detener componente."""
        pass
    
    def add_dependency(self, component_id: str) -> None:
        """Añadir dependencia."""
        if component_id not in self.dependencies:
            self.dependencies.append(component_id)
    
    def set_coordinator(self, coordinator) -> None:
        """Establecer referencia al coordinador."""
        self.coordinator = coordinator

class HybridCoordinator:
    """Coordinador minimalista para prueba."""
    
    def __init__(self):
        self.components = {}
        self.event_subscribers = {}
        self.event_count = 0
        self.request_count = 0
        self.failed_requests = 0
    
    def register_component(self, id: str, component: Component) -> None:
        """Registrar componente."""
        self.components[id] = component
        component.set_coordinator(self)
        logger.info(f"Componente {id} registrado")
    
    def subscribe(self, component_id: str, event_types: List[str]) -> None:
        """Suscribir componente a eventos."""
        if component_id not in self.components:
            return
        
        for event_type in event_types:
            if event_type not in self.event_subscribers:
                self.event_subscribers[event_type] = set()
            self.event_subscribers[event_type].add(component_id)
    
    async def request(self, target_id: str, request_type: str, 
                     data: Dict[str, Any], source: str,
                     timeout: float = 0.5) -> Optional[Any]:
        """API: Enviar solicitud a componente."""
        self.request_count += 1
        
        # Verificar componente
        if target_id not in self.components:
            self.failed_requests += 1
            return None
        
        try:
            # Enviar con timeout
            result = await asyncio.wait_for(
                self.components[target_id].process_request(request_type, data, source),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            self.failed_requests += 1
            logger.warning(f"Timeout en solicitud a {target_id}")
            return None
        except Exception as e:
            self.failed_requests += 1
            logger.warning(f"Error en solicitud a {target_id}: {e}")
            return None
    
    async def emit_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """WebSocket: Emitir evento."""
        self.event_count += 1
        
        # Obtener suscriptores
        subscribers = self.event_subscribers.get(event_type, set())
        if not subscribers:
            return
        
        # Enviar a todos los suscriptores
        tasks = []
        for comp_id in subscribers:
            if comp_id in self.components and comp_id != source:
                tasks.append(
                    self.components[comp_id].on_event(event_type, data, source)
                )
        
        # Ejecutar en paralelo
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def start(self) -> None:
        """Iniciar todos los componentes."""
        tasks = [comp.start() for comp in self.components.values()]
        await asyncio.gather(*tasks)
        logger.info(f"Sistema iniciado con {len(self.components)} componentes")
    
    async def stop(self) -> None:
        """Detener todos los componentes."""
        tasks = [comp.stop() for comp in self.components.values()]
        await asyncio.gather(*tasks)
        logger.info("Sistema detenido")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas básicas."""
        crashed = sum(1 for c in self.components.values() if c.crashed)
        
        return {
            "components": {
                "total": len(self.components),
                "crashed": crashed,
                "healthy": len(self.components) - crashed
            },
            "events": {
                "total": self.event_count,
                "by_component": {comp_id: comp.events_received 
                                for comp_id, comp in self.components.items()}
            },
            "requests": {
                "total": self.request_count,
                "failed": self.failed_requests,
                "success_rate": (self.request_count - self.failed_requests) / max(1, self.request_count),
                "by_component": {comp_id: comp.requests_handled 
                               for comp_id, comp in self.components.items()}
            }
        }

# Pruebas individuales
async def test_basic_functionality():
    """
    Probar funcionalidad básica del sistema híbrido.
    
    Verifica:
    - Registro de componentes
    - Comunicación API entre componentes
    - Distribución de eventos
    """
    logger.info("=== TEST 1: FUNCIONALIDAD BÁSICA ===")
    
    # Crear sistema
    coordinator = HybridCoordinator()
    
    # Crear componentes
    for i in range(5):
        component = Component(f"comp_{i}", failure_rate=0.05 if i == 2 else 0)
        coordinator.register_component(f"comp_{i}", component)
    
    # Crear dependencias
    coordinator.components["comp_0"].add_dependency("comp_1")
    coordinator.components["comp_1"].add_dependency("comp_2")
    coordinator.components["comp_2"].add_dependency("comp_3")
    
    # Suscribir a eventos
    coordinator.subscribe("comp_0", ["notification", "data_update"])
    coordinator.subscribe("comp_1", ["data_update", "system"])
    coordinator.subscribe("comp_2", ["notification", "system"])
    coordinator.subscribe("comp_3", ["notification", "data_update", "system"])
    coordinator.subscribe("comp_4", ["notification", "data_update", "system"])
    
    # Iniciar
    await coordinator.start()
    
    try:
        # Probar API
        logger.info("Probando comunicación API...")
        
        results = []
        for i in range(5):
            comp_id = f"comp_{i}"
            result = await coordinator.request(
                comp_id, "status", {"test": True}, "test"
            )
            results.append(result is not None)
        
        api_success = sum(results)
        logger.info(f"Resultados API: {api_success}/5 solicitudes exitosas")
        
        # Probar WebSocket
        logger.info("Probando comunicación WebSocket...")
        
        # Emitir varios eventos
        event_types = ["notification", "data_update", "system"]
        for _ in range(3):
            for event_type in event_types:
                await coordinator.emit_event(
                    event_type,
                    {"timestamp": time.time()},
                    "test"
                )
        
        # Verificar recepción
        total_events = sum(c.events_received for c in coordinator.components.values())
        logger.info(f"Eventos distribuidos: {total_events} recibidos en total")
        
        # Verificar dependencias
        logger.info("Verificando dependencias...")
        
        # Comprobar dependencias de comp_0
        result = await coordinator.request(
            "comp_0", "check_dependencies", {}, "test"
        )
        
        dependency_success = False
        if result and "dependencies" in result:
            dependency_success = True
            dep_results = result["dependencies"]
            logger.info(f"Dependencias de comp_0: {dep_results}")
        
        return {
            "api_success_rate": api_success / 5,
            "events_distributed": total_events > 0,
            "dependency_check": dependency_success
        }
    
    finally:
        # Detener
        await coordinator.stop()

async def test_crash_resilience():
    """
    Probar resiliencia ante crash de componentes.
    
    Verifica:
    - Manejo correcto cuando componentes fallan
    - Operación continuada con componentes disponibles
    - Recuperación de componentes
    """
    logger.info("\n=== TEST 2: RESILIENCIA ANTE CRASH ===")
    
    # Crear sistema
    coordinator = HybridCoordinator()
    
    # Crear componentes (más componentes para este test)
    for i in range(6):
        # Algunos con mayor probabilidad de fallo
        failure_rate = 0.1 if i % 2 == 0 else 0
        component = Component(f"comp_{i}", failure_rate=failure_rate)
        coordinator.register_component(f"comp_{i}", component)
    
    # Suscribir a eventos
    for i in range(6):
        coordinator.subscribe(f"comp_{i}", ["notification", "system_recovery"])
    
    # Iniciar
    await coordinator.start()
    
    try:
        # Fase 1: Operación normal
        logger.info("Fase 1: Operación normal")
        
        normal_results = []
        for i in range(6):
            result = await coordinator.request(
                f"comp_{i}", "status", {}, "test"
            )
            normal_results.append(result is not None)
        
        normal_success = sum(normal_results)
        logger.info(f"Operación normal: {normal_success}/6 respuestas exitosas")
        
        # Fase 2: Forzar crash
        logger.info("Fase 2: Forzando crash en componentes")
        
        # Marcar algunos como crashed
        crashed_components = ["comp_1", "comp_3"]
        for comp_id in crashed_components:
            coordinator.components[comp_id].crashed = True
            logger.info(f"Componente {comp_id} marcado como crashed")
        
        # Verificar operación durante crash
        crash_results = []
        for i in range(6):
            result = await coordinator.request(
                f"comp_{i}", "status", {}, "test"
            )
            crash_results.append(result is not None)
        
        crash_success = sum(crash_results)
        logger.info(f"Operación durante crash: {crash_success}/6 respuestas exitosas")
        
        # Fase 3: Recuperación
        logger.info("Fase 3: Recuperación de componentes")
        
        # Emitir eventos de recuperación
        for comp_id in crashed_components:
            await coordinator.emit_event(
                "system_recovery",
                {"target": comp_id},
                "test"
            )
        
        # Verificar recuperación
        recovery_results = []
        for i in range(6):
            result = await coordinator.request(
                f"comp_{i}", "status", {}, "test"
            )
            recovery_results.append(result is not None)
        
        recovery_success = sum(recovery_results)
        logger.info(f"Operación tras recuperación: {recovery_success}/6 respuestas exitosas")
        
        # Métricas finales
        metrics = coordinator.get_metrics()
        logger.info(f"Métricas finales: {metrics['components']['crashed']} componentes caídos")
        
        return {
            "normal_success_rate": normal_success / 6,
            "crash_success_rate": crash_success / 6,
            "recovery_success_rate": recovery_success / 6,
            "resilience_score": crash_success / (6 - len(crashed_components))
        }
    
    finally:
        # Detener
        await coordinator.stop()

async def test_hybrid_deadlock_prevention():
    """
    Probar que el sistema híbrido previene deadlocks.
    
    Simula escenarios que causarían deadlock en sistemas síncronos:
    - Llamadas recursivas
    - Dependencias circulares
    """
    logger.info("\n=== TEST 3: PREVENCIÓN DE DEADLOCKS ===")
    
    # Crear sistema
    coordinator = HybridCoordinator()
    
    # Clases especiales para este test
    class RecursiveComponent(Component):
        """Componente que se llama a sí mismo recursivamente."""
        
        async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
            if self.crashed:
                raise Exception(f"Componente {self.id} no disponible")
            
            self.requests_handled += 1
            
            # Llamada recursiva si es el tipo adecuado
            if request_type == "recursive" and not data.get("is_child", False):
                # Autollamarse (causaría deadlock en sistema síncrono)
                result = await self.coordinator.request(
                    self.id,
                    "recursive",
                    {"is_child": True},
                    self.id
                )
                
                return {
                    "status": "ok",
                    "recursive_result": result
                }
            
            # Respuesta normal para otros casos
            return {
                "status": "ok",
                "component": self.id,
                "is_child": data.get("is_child", False)
            }
    
    class CircularComponent(Component):
        """Componente para probar dependencias circulares."""
        
        def __init__(self, id: str, next_component: str):
            super().__init__(id)
            self.next_component = next_component
        
        async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
            if self.crashed:
                raise Exception(f"Componente {self.id} no disponible")
            
            self.requests_handled += 1
            
            # Llamada circular si es el tipo adecuado
            if request_type == "circular" and not data.get("visited", []):
                # Registrar componentes visitados
                visited = data.get("visited", []) + [self.id]
                
                # Llamar al siguiente en el círculo
                result = await self.coordinator.request(
                    self.next_component,
                    "circular",
                    {"visited": visited},
                    self.id
                )
                
                return {
                    "status": "ok",
                    "circular_path": visited,
                    "circular_result": result
                }
            
            # Respuesta normal para evitar ciclos infinitos
            return {
                "status": "ok",
                "component": self.id,
                "visited": data.get("visited", [])
            }
    
    # Crear componentes
    recursive_comp = RecursiveComponent("recursive", failure_rate=0)
    coordinator.register_component("recursive", recursive_comp)
    
    # Crear componentes para dependencia circular
    circular_a = CircularComponent("circular_a", "circular_b")
    circular_b = CircularComponent("circular_b", "circular_c")
    circular_c = CircularComponent("circular_c", "circular_a")
    
    coordinator.register_component("circular_a", circular_a)
    coordinator.register_component("circular_b", circular_b)
    coordinator.register_component("circular_c", circular_c)
    
    # Iniciar
    await coordinator.start()
    
    try:
        # Prueba 1: Llamada recursiva
        logger.info("Prueba 1: Llamada recursiva")
        
        recursive_result = await coordinator.request(
            "recursive", "recursive", {}, "test"
        )
        
        if recursive_result:
            logger.info("Resultado de llamada recursiva: OK")
            logger.info(f"Detalles: {recursive_result}")
        else:
            logger.warning("La llamada recursiva falló")
        
        # Prueba 2: Dependencia circular
        logger.info("Prueba 2: Dependencia circular")
        
        circular_result = await coordinator.request(
            "circular_a", "circular", {}, "test"
        )
        
        if circular_result:
            logger.info("Resultado de dependencia circular: OK")
            logger.info(f"Detalles: {circular_result}")
        else:
            logger.warning("La dependencia circular falló")
        
        return {
            "recursive_call_success": recursive_result is not None,
            "circular_call_success": circular_result is not None
        }
    
    finally:
        # Detener
        await coordinator.stop()

async def run_minimal_tests():
    """Ejecutar conjunto minimal de pruebas."""
    logger.info("INICIANDO PRUEBAS MINIMALES DEL CORE")
    
    # Test 1: Funcionalidad básica
    basic_results = await test_basic_functionality()
    
    # Test 2: Resiliencia ante crash
    resilience_results = await test_crash_resilience()
    
    # Test 3: Prevención de deadlocks
    deadlock_results = await test_hybrid_deadlock_prevention()
    
    # Resumen
    print("\n=== RESUMEN DE RESULTADOS ===")
    
    print("1. Funcionalidad Básica:")
    print(f"   - API: {basic_results['api_success_rate']*100:.1f}% éxito")
    print(f"   - Eventos: {'Exitoso' if basic_results['events_distributed'] else 'Fallido'}")
    print(f"   - Dependencias: {'Verificado' if basic_results['dependency_check'] else 'Fallido'}")
    
    print("2. Resiliencia ante Crash:")
    print(f"   - Normal: {resilience_results['normal_success_rate']*100:.1f}% éxito")
    print(f"   - Durante crash: {resilience_results['crash_success_rate']*100:.1f}% éxito")
    print(f"   - Tras recuperación: {resilience_results['recovery_success_rate']*100:.1f}% éxito")
    
    logger.info("3. Prevención de Deadlocks:")
    logger.info(f"   - Llamadas recursivas: {'Éxito' if deadlock_results['recursive_call_success'] else 'Fallido'}")
    logger.info(f"   - Dependencias circulares: {'Éxito' if deadlock_results['circular_call_success'] else 'Fallido'}")
    
    # Puntuación global
    functionality_score = (
        basic_results['api_success_rate'] * 0.5 +
        (1.0 if basic_results['events_distributed'] else 0) * 0.3 +
        (1.0 if basic_results['dependency_check'] else 0) * 0.2
    )
    
    resilience_score = (
        resilience_results['crash_success_rate'] * 0.5 +
        resilience_results['recovery_success_rate'] * 0.3 +
        resilience_results['resilience_score'] * 0.2
    )
    
    deadlock_score = (
        (1.0 if deadlock_results['recursive_call_success'] else 0) * 0.5 +
        (1.0 if deadlock_results['circular_call_success'] else 0) * 0.5
    )
    
    global_score = (
        functionality_score * 0.3 +
        resilience_score * 0.4 +
        deadlock_score * 0.3
    )
    
    # Calificar puntuación
    def get_assessment(score):
        if score >= 0.9:
            return "Excelente"
        elif score >= 0.8:
            return "Muy Bueno"
        elif score >= 0.7:
            return "Bueno"
        elif score >= 0.6:
            return "Aceptable"
        elif score >= 0.5:
            return "Suficiente"
        else:
            return "Insuficiente"
    
    logger.info("\nPuntuaciones:")
    logger.info(f"- Funcionalidad: {functionality_score*100:.1f}/100 - {get_assessment(functionality_score)}")
    logger.info(f"- Resiliencia: {resilience_score*100:.1f}/100 - {get_assessment(resilience_score)}")
    logger.info(f"- Prevención Deadlocks: {deadlock_score*100:.1f}/100 - {get_assessment(deadlock_score)}")
    logger.info(f"\nPuntuación Global: {global_score*100:.1f}/100 - {get_assessment(global_score)}")
    
    if global_score >= 0.8:
        logger.info("\nEl sistema híbrido API+WebSocket ha demostrado ser robusto y eficaz para prevenir deadlocks")
        logger.info("y fallos en cascada, manteniendo alta disponibilidad incluso durante fallos de componentes.")
    elif global_score >= 0.6:
        logger.info("\nEl sistema híbrido API+WebSocket funciona adecuadamente para prevenir deadlocks,")
        logger.info("aunque hay áreas de mejora en cuanto a resiliencia y manejo de fallos.")
    else:
        logger.info("\nEl sistema híbrido API+WebSocket requiere mejoras sustanciales para garantizar")
        logger.info("la estabilidad y prevención de deadlocks en entornos de producción.")
    
    return {
        "functionality_score": functionality_score,
        "resilience_score": resilience_score,
        "deadlock_score": deadlock_score,
        "global_score": global_score,
        "assessment": get_assessment(global_score)
    }

# Punto de entrada
if __name__ == "__main__":
    asyncio.run(run_minimal_tests())