"""
Prueba específica para evaluar la capacidad del sistema híbrido para manejar fallos simultáneos.
Esta prueba es mucho más ligera y se enfoca exclusivamente en la resiliencia ante fallos.
"""

import asyncio
import logging
import random
import time
from typing import Dict, Any, List, Optional, Set

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("multi_failure_test")

# Componente básico resiliente
class ResilienceTestComponent:
    """Componente simplificado para pruebas de resiliencia."""
    
    def __init__(self, id: str, failure_rate: float = 0.0):
        self.id = id
        self.failure_rate = failure_rate
        self.crashed = False
        self.event_count = 0
        self.request_count = 0
        self.failure_count = 0
        self.start_time = time.time()
    
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """Procesar solicitud API con posible fallo."""
        if self.crashed:
            self.failure_count += 1
            logger.warning(f"[{self.id}] Solicitud rechazada: componente no disponible")
            raise Exception(f"Componente {self.id} no disponible")
        
        self.request_count += 1
        
        # Simular fallo aleatorio
        if random.random() < self.failure_rate:
            self.failure_count += 1
            logger.warning(f"[{self.id}] Fallo aleatorio en solicitud {request_type}")
            raise Exception(f"Error simulado en {self.id}")
        
        # Simular procesamiento
        await asyncio.sleep(0.01)
        
        return {
            "status": "success",
            "processor": self.id,
            "request_type": request_type
        }
    
    async def on_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """Procesar evento WebSocket."""
        if self.crashed:
            # Los eventos se descartan silenciosamente cuando el componente está caído
            return
        
        self.event_count += 1
        
        # Simular procesamiento
        await asyncio.sleep(0.005)
    
    async def start(self) -> None:
        self.start_time = time.time()
        self.crashed = False
        logger.debug(f"Componente {self.id} iniciado")
    
    async def stop(self) -> None:
        uptime = time.time() - self.start_time
        logger.debug(f"Componente {self.id} detenido. Uptime: {uptime:.2f}s")

# Coordinador simplificado
class ResilienceTestCoordinator:
    """Coordinador simplificado para pruebas de resiliencia."""
    
    def __init__(self):
        self.components = {}
        self.event_subscribers = {}
        self.metrics = {
            "request_success": 0,
            "request_failures": 0,
            "events_emitted": 0
        }
    
    def register_component(self, id: str, component: ResilienceTestComponent) -> None:
        """Registrar componente."""
        self.components[id] = component
        logger.info(f"Componente {id} registrado")
    
    def subscribe(self, component_id: str, event_types: List[str]) -> None:
        """Suscribir componente a tipos de eventos."""
        if component_id not in self.components:
            logger.warning(f"No se puede suscribir: componente {component_id} no registrado")
            return
        
        for event_type in event_types:
            if event_type not in self.event_subscribers:
                self.event_subscribers[event_type] = set()
            self.event_subscribers[event_type].add(component_id)
        
        logger.info(f"Componente {component_id} suscrito a {len(event_types)} tipos de eventos")
    
    async def request(self, target_id: str, request_type: str, 
                     data: Dict[str, Any], source: str,
                     timeout: float = 1.0) -> Optional[Any]:
        """API: Solicitud directa con timeout."""
        if target_id not in self.components:
            self.metrics["request_failures"] += 1
            logger.warning(f"Solicitud a componente inexistente: {target_id}")
            return None
        
        try:
            # Solicitud con timeout para prevenir bloqueos
            result = await asyncio.wait_for(
                self.components[target_id].process_request(request_type, data, source),
                timeout=timeout
            )
            
            self.metrics["request_success"] += 1
            return result
            
        except asyncio.TimeoutError:
            self.metrics["request_failures"] += 1
            logger.warning(f"Timeout en solicitud a {target_id}")
            return None
            
        except Exception as e:
            self.metrics["request_failures"] += 1
            logger.warning(f"Error en solicitud a {target_id}: {e}")
            return None
    
    async def emit_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """WebSocket: Emitir evento a suscriptores."""
        self.metrics["events_emitted"] += 1
        
        # Obtener suscriptores
        subscribers = self.event_subscribers.get(event_type, set())
        if not subscribers:
            return
        
        # Crear tareas para cada suscriptor
        tasks = []
        for comp_id in subscribers:
            if comp_id in self.components and comp_id != source:
                tasks.append(
                    self.components[comp_id].on_event(event_type, data, source)
                )
        
        # Ejecutar entregas en paralelo
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def start(self) -> None:
        """Iniciar todos los componentes."""
        start_tasks = [comp.start() for comp in self.components.values()]
        await asyncio.gather(*start_tasks)
        logger.info(f"Coordinador iniciado con {len(self.components)} componentes")
    
    async def stop(self) -> None:
        """Detener todos los componentes."""
        stop_tasks = [comp.stop() for comp in self.components.values()]
        await asyncio.gather(*stop_tasks)
        logger.info("Coordinador detenido")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas del sistema."""
        metrics = self.metrics.copy()
        
        # Añadir métricas por componente
        metrics["components"] = {}
        for comp_id, comp in self.components.items():
            metrics["components"][comp_id] = {
                "status": "crashed" if comp.crashed else "active",
                "event_count": comp.event_count,
                "request_count": comp.request_count,
                "failure_count": comp.failure_count,
                "failure_rate": comp.failure_count / max(1, comp.request_count) if comp.request_count > 0 else 0
            }
        
        # Calcular tasa de éxito global
        total_requests = metrics["request_success"] + metrics["request_failures"]
        metrics["request_success_rate"] = metrics["request_success"] / max(1, total_requests)
        
        return metrics

async def setup_resilience_test(num_components: int = 6) -> ResilienceTestCoordinator:
    """Configurar sistema para prueba de resiliencia."""
    # Crear coordinador
    coordinator = ResilienceTestCoordinator()
    
    # Crear componentes
    for i in range(num_components):
        # Tasa de fallo base (varía entre componentes)
        failure_rate = 0.01 * (i + 1)  # 1% a 6%
        
        # Crear componente
        component = ResilienceTestComponent(
            id=f"comp_{i}",
            failure_rate=failure_rate
        )
        
        # Registrar componente
        coordinator.register_component(f"comp_{i}", component)
        
        # Suscribir a eventos (cada componente a 1-3 tipos)
        event_types = ["data", "status", "alert", "metric", "command"]
        num_subscriptions = random.randint(1, 3)
        subscribed_events = random.sample(event_types, num_subscriptions)
        
        coordinator.subscribe(f"comp_{i}", subscribed_events)
    
    # Iniciar sistema
    await coordinator.start()
    
    return coordinator

async def test_multiple_failures(coordinator: ResilienceTestCoordinator,
                               num_iterations: int = 20,
                               components_to_crash: int = 3) -> Dict[str, Any]:
    """
    Probar resiliencia ante fallos múltiples simultáneos.
    
    Args:
        coordinator: Coordinador del sistema
        num_iterations: Número de ciclos de prueba
        components_to_crash: Número de componentes que fallarán
    """
    logger.info(f"Iniciando prueba de fallos múltiples")
    logger.info(f"- Iteraciones: {num_iterations}")
    logger.info(f"- Componentes a fallar: {components_to_crash}")
    
    # Estadísticas
    total_requests = 0
    failed_requests = 0
    successful_requests = 0
    events_emitted = 0
    
    # Hacer crash en componentes seleccionados
    all_components = list(coordinator.components.keys())
    if len(all_components) < components_to_crash:
        components_to_crash = len(all_components)
    
    crashed_components = random.sample(all_components, components_to_crash)
    
    logger.info(f"Forzando crash en componentes: {crashed_components}")
    for comp_id in crashed_components:
        coordinator.components[comp_id].crashed = True
    
    # Ciclos de prueba
    for i in range(num_iterations):
        logger.info(f"Ejecutando iteración {i+1}/{num_iterations}")
        
        # 1. Emitir varios eventos
        event_tasks = []
        for _ in range(5):  # 5 eventos por iteración
            event_type = random.choice(["data", "status", "alert", "metric", "command"])
            data = {
                "iteration": i,
                "timestamp": time.time(),
                "message": f"Evento de prueba #{events_emitted}"
            }
            
            event_tasks.append(
                coordinator.emit_event(event_type, data, "test_harness")
            )
            events_emitted += 1
        
        # Esperar que se completen las emisiones
        await asyncio.gather(*event_tasks)
        
        # 2. Realizar solicitudes a todos los componentes (incluyendo los crasheados)
        request_results = []
        request_tasks = []
        
        for comp_id in coordinator.components:
            request_type = random.choice(["query", "status", "command"])
            data = {"iteration": i, "timestamp": time.time()}
            
            task = coordinator.request(comp_id, request_type, data, "test_harness")
            request_tasks.append(task)
            total_requests += 1
        
        # Esperar que se completen todas las solicitudes
        request_results = await asyncio.gather(*request_tasks)
        
        # Contar resultados
        for result in request_results:
            if result is None:
                failed_requests += 1
            else:
                successful_requests += 1
        
        # Breve pausa entre iteraciones
        await asyncio.sleep(0.1)
    
    # Recolectar métricas finales
    metrics = coordinator.get_metrics()
    
    # Añadir métricas específicas de la prueba
    metrics["test_specific"] = {
        "iterations": num_iterations,
        "components_crashed": components_to_crash,
        "crashed_component_ids": crashed_components,
        "total_requests_sent": total_requests,
        "failed_requests": failed_requests,
        "successful_requests": successful_requests,
        "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
        "events_emitted": events_emitted
    }
    
    # Mostrar resumen
    logger.info("=== RESULTADOS DE PRUEBA DE RESILIENCIA ===")
    logger.info(f"Componentes crasheados: {components_to_crash} de {len(coordinator.components)}")
    logger.info(f"Solicitudes totales: {total_requests}")
    logger.info(f"Solicitudes exitosas: {successful_requests} ({(successful_requests/total_requests)*100:.1f}%)")
    logger.info(f"Solicitudes fallidas: {failed_requests} ({(failed_requests/total_requests)*100:.1f}%)")
    logger.info(f"Eventos emitidos: {events_emitted}")
    
    return metrics

async def run_resilience_test():
    """Ejecutar prueba completa de resiliencia a fallos."""
    try:
        logger.info("=== INICIANDO PRUEBA DE RESILIENCIA A FALLOS MÚLTIPLES ===")
        
        # 1. Configurar sistema
        system = await setup_resilience_test(num_components=6)
        
        try:
            # 2. Ejecutar prueba
            results = await test_multiple_failures(
                system,
                num_iterations=20,
                components_to_crash=3  # 50% de los componentes
            )
            
            # 3. Evaluar resultados
            success_rate = results["test_specific"]["success_rate"]
            crashed_percent = results["test_specific"]["components_crashed"] / len(system.components)
            
            logger.info("\n=== EVALUACIÓN DE RESILIENCIA ===")
            if success_rate >= 0.6:
                logger.info(f"EXCELENTE: El sistema mantuvo un {success_rate*100:.1f}% de éxito con {crashed_percent*100:.1f}% de componentes caídos")
            elif success_rate >= 0.4:
                logger.info(f"BUENO: El sistema mantuvo un {success_rate*100:.1f}% de éxito con {crashed_percent*100:.1f}% de componentes caídos")
            elif success_rate >= 0.2:
                logger.info(f"ACEPTABLE: El sistema mantuvo un {success_rate*100:.1f}% de éxito con {crashed_percent*100:.1f}% de componentes caídos")
            else:
                logger.info(f"INSUFICIENTE: El sistema solo tuvo un {success_rate*100:.1f}% de éxito con {crashed_percent*100:.1f}% de componentes caídos")
            
            return results
            
        finally:
            # Asegurar que el sistema se detenga
            await system.stop()
    
    except Exception as e:
        logger.error(f"Error durante la prueba de resiliencia: {e}")
        import traceback
        traceback.print_exc()

# Punto de entrada
if __name__ == "__main__":
    asyncio.run(run_resilience_test())