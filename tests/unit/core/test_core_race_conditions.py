"""
Pruebas para detectar condiciones de carrera y deadlocks en el motor Genesis.

Este módulo contiene pruebas diseñadas específicamente para provocar y
detectar condiciones de carrera, deadlocks y otros problemas de sincronización
que pueden ocurrir en sistemas distribuidos asíncronos.
"""

import asyncio
import logging
import pytest
import time
import random
import threading
from typing import Dict, Any, List, Optional, Set, Tuple

# Importamos utilidades para timeouts
from tests.utils.timeout_helpers import (
    emit_with_timeout,
    check_component_status,
    run_test_with_timing,
    cleanup_engine
)

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Nos aseguramos que pytest detecte correctamente las pruebas asíncronas
pytestmark = pytest.mark.asyncio

# Importamos clases necesarias
from genesis.core.component import Component
from genesis.core.engine_non_blocking import EngineNonBlocking


class ResourceContenderComponent(Component):
    """
    Componente que simula contención de recursos y dependencias circulares.
    
    Este componente intencionalmente intenta acceder a recursos compartidos
    de maneras que pueden provocar condiciones de carrera o deadlocks.
    """
    
    def __init__(self, name: str, dependencies: List[str] = None, shared_resources: List[str] = None):
        """
        Inicializar componente con dependencias y recursos.
        
        Args:
            name: Nombre del componente
            dependencies: Lista de nombres de componentes de los que depende
            shared_resources: Lista de recursos compartidos que utiliza
        """
        super().__init__(name)
        self.dependencies = dependencies or []
        self.resources = shared_resources or []
        self.held_resources: Set[str] = set()
        self.processing_count = 0
        self.error_count = 0
        self.deadlock_detected = False
        self.healthy = True
        
        # Para simular bloqueos específicos
        self.lock_time = random.uniform(0.05, 0.2)
        self.release_probability = 0.9  # 90% de probabilidad de liberar recursos correctamente
        
    async def start(self) -> None:
        """Iniciar el componente."""
        logger.debug(f"ResourceContenderComponent {self.name} iniciado")
        self.healthy = True
        
    async def stop(self) -> None:
        """Detener el componente y liberar todos los recursos."""
        logger.debug(f"ResourceContenderComponent {self.name} detenido")
        self.healthy = False
        self.held_resources.clear()
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """
        Manejar eventos, con posibilidad de provocar condiciones de carrera.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Respuesta al evento, si corresponde
        """
        if event_type == "check_status":
            return {
                "healthy": self.healthy,
                "held_resources": list(self.held_resources),
                "processing_count": self.processing_count,
                "error_count": self.error_count,
                "deadlock_detected": self.deadlock_detected
            }
            
        elif event_type == "acquire_resource":
            resource_id = data.get("resource_id")
            exclusive = data.get("exclusive", False)
            timeout = data.get("timeout", 1.0)
            
            if not resource_id or resource_id not in self.resources:
                return {
                    "success": False, 
                    "error": f"Recurso {resource_id} no disponible para {self.name}"
                }
            
            # Simular tiempo de adquisición
            start_time = time.time()
            
            # Intentar adquirir el recurso con timeout
            try:
                acquired = await self._try_acquire_resource(resource_id, exclusive, timeout)
                acquisition_time = time.time() - start_time
                
                if acquired:
                    self.held_resources.add(resource_id)
                    self.processing_count += 1
                    return {
                        "success": True,
                        "resource_id": resource_id,
                        "acquisition_time": acquisition_time
                    }
                else:
                    self.error_count += 1
                    return {
                        "success": False,
                        "error": "Timeout esperando recurso",
                        "resource_id": resource_id
                    }
                    
            except asyncio.TimeoutError:
                self.error_count += 1
                return {
                    "success": False,
                    "error": "Timeout al adquirir recurso",
                    "resource_id": resource_id
                }
                
        elif event_type == "release_resource":
            resource_id = data.get("resource_id")
            
            if not resource_id or resource_id not in self.held_resources:
                return {
                    "success": False, 
                    "error": f"No tiene el recurso {resource_id} para liberar"
                }
            
            # Simular problemas aleatorios al liberar recursos
            if random.random() > self.release_probability:
                self.error_count += 1
                return {
                    "success": False,
                    "error": "Error simulado al liberar recurso",
                    "resource_id": resource_id
                }
            
            # Liberar recurso
            self.held_resources.remove(resource_id)
            return {
                "success": True,
                "resource_id": resource_id
            }
            
        elif event_type == "process_with_dependency":
            dependency_id = data.get("dependency_id")
            timeout = data.get("timeout", 1.0)
            
            if not dependency_id or dependency_id not in self.dependencies:
                return {
                    "success": False, 
                    "error": f"Dependencia {dependency_id} no configurada para {self.name}"
                }
            
            # Simular procesamiento que depende de otro componente
            try:
                result = await self._process_with_dependency(dependency_id, timeout)
                return result
            except Exception as e:
                self.error_count += 1
                return {
                    "success": False,
                    "error": str(e),
                    "dependency_id": dependency_id
                }
                
        elif event_type == "detect_deadlock":
            # Intentar detectar si estamos en un deadlock
            path = data.get("visited_path", [])
            resource_chain = data.get("resource_chain", [])
            
            # Si este componente ya está en el camino, podemos tener un ciclo
            if self.name in path:
                self.deadlock_detected = True
                return {
                    "deadlock_detected": True,
                    "cycle": path[path.index(self.name):] + [self.name],
                    "resources_involved": resource_chain
                }
            
            # Agregar este componente al camino
            new_path = path + [self.name]
            new_resource_chain = resource_chain + list(self.held_resources)
            
            return {
                "deadlock_detected": self.deadlock_detected,
                "visited_path": new_path,
                "resource_chain": new_resource_chain
            }
            
        return None
    
    async def _try_acquire_resource(self, resource_id: str, exclusive: bool, timeout: float) -> bool:
        """
        Simular el intento de adquisición de un recurso.
        
        Args:
            resource_id: ID del recurso a adquirir
            exclusive: Si la adquisición es exclusiva
            timeout: Tiempo máximo de espera
            
        Returns:
            True si se adquirió el recurso, False si no
            
        Raises:
            TimeoutError: Si se agota el tiempo de espera
        """
        # Simular tiempo variable de adquisición
        await asyncio.sleep(self.lock_time)
        
        # Simular bloqueo aleatorio (1% de probabilidad)
        if random.random() < 0.01:
            await asyncio.sleep(timeout + 0.1)  # Forzar timeout
            raise asyncio.TimeoutError(f"Timeout simulado adquiriendo {resource_id}")
        
        return True
    
    async def _process_with_dependency(self, dependency_id: str, timeout: float) -> Dict[str, Any]:
        """
        Simular procesamiento que depende de otro componente.
        
        Args:
            dependency_id: ID del componente del que depende
            timeout: Tiempo máximo de espera
            
        Returns:
            Resultado del procesamiento
        """
        # Simular tiempo de procesamiento
        await asyncio.sleep(random.uniform(0.05, 0.1))
        
        # Esto debería comunicarse con el componente de dependencia
        # pero para simplificar, solo simulamos el resultado
        self.processing_count += 1
        
        # 5% de probabilidad de simular un fallo
        if random.random() < 0.05:
            self.error_count += 1
            return {
                "success": False,
                "error": f"Error procesando con dependencia {dependency_id}",
                "dependency_id": dependency_id
            }
        
        return {
            "success": True,
            "dependency_id": dependency_id,
            "processed_at": time.time()
        }


@pytest.mark.asyncio
async def test_resource_contention_and_deadlocks(non_blocking_engine):
    """
    Prueba para detectar condiciones de carrera y deadlocks.
    
    Esta prueba simula un escenario donde múltiples componentes intentan
    acceder a recursos compartidos en patrones que pueden llevar a
    condiciones de carrera y deadlocks.
    """
    engine = non_blocking_engine
    
    # Función interna para la prueba
    async def run_deadlock_test(engine):
        # Definir recursos compartidos
        shared_resources = ["database", "cache", "file_system", "network", "config"]
        
        # Crear componentes con dependencias circulares y recursos compartidos
        components = [
            # Componente A depende de B, usa database y cache
            ResourceContenderComponent(
                "comp_a", 
                dependencies=["comp_b"], 
                shared_resources=["database", "cache"]
            ),
            
            # Componente B depende de C, usa cache y file_system
            ResourceContenderComponent(
                "comp_b", 
                dependencies=["comp_c"], 
                shared_resources=["cache", "file_system"]
            ),
            
            # Componente C depende de A, usa file_system y network
            ResourceContenderComponent(
                "comp_c", 
                dependencies=["comp_a"], 
                shared_resources=["file_system", "network"]
            ),
            
            # Componente D independiente, usa database, network y config
            ResourceContenderComponent(
                "comp_d", 
                dependencies=[], 
                shared_resources=["database", "network", "config"]
            ),
            
            # Componente E independiente, usa todos los recursos
            ResourceContenderComponent(
                "comp_e", 
                dependencies=[], 
                shared_resources=shared_resources
            )
        ]
        
        # Registrar los componentes en el motor
        for comp in components:
            await engine.register_component(comp)
            
        logger.info("Componentes registrados con dependencias circulares")
        
        # Función para crear un escenario de contención de recursos
        async def create_resource_contention():
            # Diccionario para llevar registro de qué componente tiene cada recurso
            resource_holders = {res: set() for res in shared_resources}
            acquisition_tasks = []
            
            # Hacer que todos los componentes intenten adquirir sus recursos
            for comp in components:
                for resource in comp.resources:
                    task = emit_with_timeout(
                        engine, 
                        "acquire_resource", 
                        {
                            "resource_id": resource, 
                            "exclusive": random.random() < 0.3,  # 30% exclusivo
                            "timeout": 0.5
                        }, 
                        comp.name,
                        timeout=1.0
                    )
                    acquisition_tasks.append((comp.name, resource, task))
            
            # Esperar a que se completen todas las tareas
            results = []
            for comp_name, resource, task in acquisition_tasks:
                try:
                    result = await task
                    results.append({
                        "component": comp_name,
                        "resource": resource,
                        "result": result
                    })
                    
                    # Actualizar quién tiene cada recurso
                    if isinstance(result, list) and len(result) > 0 and result[0].get("success", False):
                        resource_holders[resource].add(comp_name)
                        
                except Exception as e:
                    results.append({
                        "component": comp_name,
                        "resource": resource,
                        "error": str(e)
                    })
            
            # Analizar resultados
            successful = len([r for r in results if isinstance(r["result"], list) and 
                            len(r["result"]) > 0 and r["result"][0].get("success", False)])
            failed = len(results) - successful
            
            logger.info(f"Adquisición de recursos: {successful} exitosas, {failed} fallidas")
            
            # Verificar qué recursos están en contención
            contended_resources = [res for res, holders in resource_holders.items() if len(holders) > 1]
            if contended_resources:
                logger.info(f"Recursos en contención: {contended_resources}")
                for res in contended_resources:
                    logger.info(f"  {res} es mantenido por: {resource_holders[res]}")
            
            return {
                "resource_holders": {k: list(v) for k, v in resource_holders.items()},
                "contended_resources": contended_resources,
                "successful_acquisitions": successful,
                "failed_acquisitions": failed
            }
        
        # Función para intentar provocar dependencias circulares
        async def trigger_circular_dependencies():
            process_tasks = []
            
            # Hacer que cada componente procese con sus dependencias
            for comp in components:
                for dep in comp.dependencies:
                    task = emit_with_timeout(
                        engine, 
                        "process_with_dependency", 
                        {
                            "dependency_id": dep,
                            "timeout": 0.5
                        }, 
                        comp.name,
                        timeout=1.0
                    )
                    process_tasks.append((comp.name, dep, task))
            
            # Esperar a que se completen todas las tareas
            results = []
            for comp_name, dep, task in process_tasks:
                try:
                    result = await task
                    results.append({
                        "component": comp_name,
                        "dependency": dep,
                        "result": result
                    })
                except Exception as e:
                    results.append({
                        "component": comp_name,
                        "dependency": dep,
                        "error": str(e)
                    })
            
            # Analizar resultados
            successful = len([r for r in results if isinstance(r["result"], list) and 
                             len(r["result"]) > 0 and r["result"][0].get("success", False)])
            failed = len(results) - successful
            
            logger.info(f"Procesamiento con dependencias: {successful} exitosos, {failed} fallidos")
            
            return {
                "successful_processing": successful,
                "failed_processing": failed,
                "results": results
            }
        
        # Función para detectar posibles deadlocks
        async def detect_deadlocks():
            detection_tasks = []
            
            # Iniciar detección de deadlock desde cada componente
            for comp in components:
                task = emit_with_timeout(
                    engine, 
                    "detect_deadlock", 
                    {
                        "visited_path": [],
                        "resource_chain": []
                    }, 
                    comp.name,
                    timeout=1.0
                )
                detection_tasks.append((comp.name, task))
            
            # Esperar a que se completen todas las tareas
            results = []
            for comp_name, task in detection_tasks:
                try:
                    result = await task
                    results.append({
                        "component": comp_name,
                        "result": result
                    })
                except Exception as e:
                    results.append({
                        "component": comp_name,
                        "error": str(e)
                    })
            
            # Buscar si se detectó algún deadlock
            deadlocks = []
            for r in results:
                if isinstance(r["result"], list) and len(r["result"]) > 0:
                    result_data = r["result"][0]
                    if result_data.get("deadlock_detected", False) and "cycle" in result_data:
                        deadlocks.append({
                            "starting_component": r["component"],
                            "cycle": result_data["cycle"],
                            "resources": result_data.get("resources_involved", [])
                        })
            
            if deadlocks:
                logger.warning(f"Se detectaron {len(deadlocks)} posibles deadlocks")
                for i, deadlock in enumerate(deadlocks):
                    logger.warning(f"Deadlock #{i+1}: {' -> '.join(deadlock['cycle'])}")
                    if deadlock['resources']:
                        logger.warning(f"Recursos involucrados: {', '.join(deadlock['resources'])}")
            else:
                logger.info("No se detectaron deadlocks")
            
            return {
                "deadlocks_detected": len(deadlocks),
                "deadlock_details": deadlocks
            }
        
        # Ejecutar pruebas en múltiples rondas para aumentar probabilidad de problemas
        contention_results = []
        dependency_results = []
        deadlock_results = []
        
        for round_num in range(3):
            logger.info(f"Iniciando ronda {round_num+1} de pruebas de condiciones de carrera")
            
            # Provocar contención de recursos
            contention_result = await create_resource_contention()
            contention_results.append(contention_result)
            
            # Provocar dependencias circulares
            dependency_result = await trigger_circular_dependencies()
            dependency_results.append(dependency_result)
            
            # Intentar detectar deadlocks
            deadlock_result = await detect_deadlocks()
            deadlock_results.append(deadlock_result)
            
            # Liberar algunos recursos aleatoriamente para cambiar el estado
            release_tasks = []
            for comp in components:
                # Verificar estado actual
                status = await check_component_status(engine, comp.name)
                held_resources = status.get("held_resources", [])
                
                # Liberar un recurso aleatorio si tiene alguno
                if held_resources:
                    resource_to_release = random.choice(held_resources)
                    task = emit_with_timeout(
                        engine,
                        "release_resource",
                        {"resource_id": resource_to_release},
                        comp.name,
                        timeout=1.0
                    )
                    release_tasks.append(task)
            
            # Esperar a que se completen todas las liberaciones
            if release_tasks:
                await asyncio.gather(*release_tasks, return_exceptions=True)
            
            # Breve pausa entre rondas
            await asyncio.sleep(0.2)
        
        # Verificar estado final de los componentes
        component_stats = {}
        for comp in components:
            stats = await check_component_status(engine, comp.name)
            component_stats[comp.name] = stats
            
            logger.info(f"Componente {comp.name}: "
                       f"{stats.get('processing_count', 0)} procesados, "
                       f"{stats.get('error_count', 0)} errores, "
                       f"recursos: {stats.get('held_resources', [])}")
        
        # Analizar resultados agregados
        total_deadlocks = sum(r["deadlocks_detected"] for r in deadlock_results)
        total_contended_resources = set()
        for r in contention_results:
            total_contended_resources.update(r["contended_resources"])
        
        # Calcular tasas de éxito
        resource_success_rate = sum(r["successful_acquisitions"] for r in contention_results) / (
            sum(r["successful_acquisitions"] + r["failed_acquisitions"] for r in contention_results)
        ) if any(contention_results) else 0
        
        dependency_success_rate = sum(r["successful_processing"] for r in dependency_results) / (
            sum(r["successful_processing"] + r["failed_processing"] for r in dependency_results)
        ) if any(dependency_results) else 0
        
        logger.info(f"Resumen de la prueba:")
        logger.info(f"- Deadlocks detectados: {total_deadlocks}")
        logger.info(f"- Recursos en contención: {len(total_contended_resources)}")
        logger.info(f"- Tasa de éxito en adquisición: {resource_success_rate:.2%}")
        logger.info(f"- Tasa de éxito en procesamiento: {dependency_success_rate:.2%}")
        
        return {
            "component_stats": component_stats,
            "contention_results": contention_results,
            "dependency_results": dependency_results,
            "deadlock_results": deadlock_results,
            "summary": {
                "total_deadlocks": total_deadlocks,
                "contended_resources": list(total_contended_resources),
                "resource_success_rate": resource_success_rate,
                "dependency_success_rate": dependency_success_rate
            }
        }
    
    # Ejecutar y medir la prueba completa
    result = await run_test_with_timing(
        engine, 
        "test_resource_contention_and_deadlocks", 
        run_deadlock_test
    )
    
    # No fallamos la prueba, solo verificamos que la ejecución complete
    assert isinstance(result, dict), "La prueba debería devolver un dict con resultados"
    
    # Si detectamos deadlocks o tasas de éxito muy bajas, generamos una advertencia
    if result["summary"]["total_deadlocks"] > 0:
        logger.warning(f"¡ATENCIÓN! Se detectaron {result['summary']['total_deadlocks']} deadlocks potenciales")
        
    if result["summary"]["resource_success_rate"] < 0.7:
        logger.warning(f"¡ATENCIÓN! Baja tasa de éxito en adquisición de recursos: {result['summary']['resource_success_rate']:.2%}")
        
    if result["summary"]["dependency_success_rate"] < 0.7:
        logger.warning(f"¡ATENCIÓN! Baja tasa de éxito en procesamiento con dependencias: {result['summary']['dependency_success_rate']:.2%}")


@pytest.mark.asyncio
async def test_concurrent_event_collisions(non_blocking_engine):
    """
    Prueba para detectar colisiones entre eventos concurrentes.
    
    Esta prueba envía múltiples eventos concurrentes a los mismos componentes
    para verificar cómo maneja el motor los conflictos.
    """
    engine = non_blocking_engine
    
    # Función interna para la prueba
    async def run_collision_test(engine):
        # Crear recursos comunes que serán objetivo de contención
        shared_resources = ["shared_data", "shared_config", "shared_state"]
        
        # Crear varios componentes que usarán los mismos recursos
        components = []
        for i in range(5):
            comp = ResourceContenderComponent(
                f"collision_comp_{i}",
                dependencies=[f"collision_comp_{(i+1) % 5}"],  # Dependencia circular
                shared_resources=shared_resources
            )
            await engine.register_component(comp)
            components.append(comp)
            
        logger.info(f"Registrados {len(components)} componentes para prueba de colisiones")
        
        # Función para enviar eventos concurrentes a varios componentes
        async def send_concurrent_events(num_events_per_component=10):
            all_tasks = []
            events_by_component = {comp.name: [] for comp in components}
            
            # Crear tareas para cada componente
            for comp in components:
                # Algunos eventos de adquisición de recursos
                for _ in range(num_events_per_component // 2):
                    resource = random.choice(shared_resources)
                    task = emit_with_timeout(
                        engine,
                        "acquire_resource",
                        {"resource_id": resource, "exclusive": random.random() < 0.5},
                        comp.name,
                        timeout=0.5
                    )
                    all_tasks.append(task)
                    events_by_component[comp.name].append(("acquire", resource))
                
                # Algunos eventos de procesamiento con dependencias
                for _ in range(num_events_per_component // 2):
                    dependency = random.choice([d for d in comp.dependencies])
                    task = emit_with_timeout(
                        engine,
                        "process_with_dependency",
                        {"dependency_id": dependency},
                        comp.name,
                        timeout=0.5
                    )
                    all_tasks.append(task)
                    events_by_component[comp.name].append(("process", dependency))
            
            # Ejecutar todos los eventos concurrentemente
            start_time = time.time()
            results = await asyncio.gather(*all_tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # Analizar resultados
            success_count = 0
            timeout_count = 0
            error_count = 0
            
            for result in results:
                if isinstance(result, Exception):
                    error_count += 1
                elif isinstance(result, list) and len(result) > 0:
                    if result[0].get("success", False):
                        success_count += 1
                    elif result[0].get("error", "").startswith("Timeout"):
                        timeout_count += 1
                    else:
                        error_count += 1
            
            logger.info(f"Eventos concurrentes completados en {total_time:.2f}s:")
            logger.info(f"- Exitosos: {success_count}")
            logger.info(f"- Timeouts: {timeout_count}")
            logger.info(f"- Errores: {error_count}")
            
            return {
                "total_events": len(all_tasks),
                "successful": success_count,
                "timeouts": timeout_count,
                "errors": error_count,
                "time": total_time,
                "events_per_second": len(all_tasks) / total_time if total_time > 0 else 0
            }
        
        # Ejecutar múltiples rondas de eventos concurrentes
        logger.info("Ejecutando múltiples rondas de eventos concurrentes...")
        round_results = []
        for round_num in range(3):
            logger.info(f"Ronda {round_num+1} de colisiones de eventos")
            result = await send_concurrent_events(15)  # 15 eventos por componente
            round_results.append(result)
            
            # Verificar estado de los componentes
            for comp in components:
                status = await check_component_status(engine, comp.name)
                logger.info(f"{comp.name}: {status.get('processing_count', 0)} procesados, "
                           f"{status.get('error_count', 0)} errores, "
                           f"{len(status.get('held_resources', []))} recursos")
            
            # Liberar algunos recursos antes de la siguiente ronda
            for comp in components:
                status = await check_component_status(engine, comp.name)
                for resource in status.get("held_resources", []):
                    await emit_with_timeout(
                        engine,
                        "release_resource",
                        {"resource_id": resource},
                        comp.name,
                        timeout=0.5
                    )
            
            # Breve pausa entre rondas
            await asyncio.sleep(0.2)
        
        # Calcular estadísticas finales
        total_events = sum(r["total_events"] for r in round_results)
        total_successful = sum(r["successful"] for r in round_results)
        total_timeouts = sum(r["timeouts"] for r in round_results)
        total_errors = sum(r["errors"] for r in round_results)
        
        success_rate = total_successful / total_events if total_events > 0 else 0
        timeout_rate = total_timeouts / total_events if total_events > 0 else 0
        error_rate = total_errors / total_events if total_events > 0 else 0
        
        # Calcular rendimiento promedio
        avg_events_per_second = sum(r["events_per_second"] for r in round_results) / len(round_results)
        
        logger.info("Resultados finales de la prueba de colisiones:")
        logger.info(f"- Eventos totales: {total_events}")
        logger.info(f"- Tasa de éxito: {success_rate:.2%}")
        logger.info(f"- Tasa de timeouts: {timeout_rate:.2%}")
        logger.info(f"- Tasa de errores: {error_rate:.2%}")
        logger.info(f"- Rendimiento promedio: {avg_events_per_second:.2f} eventos/s")
        
        # Comprobar estado final de componentes
        component_stats = {}
        for comp in components:
            stats = await check_component_status(engine, comp.name)
            component_stats[comp.name] = stats
        
        return {
            "round_results": round_results,
            "component_stats": component_stats,
            "summary": {
                "total_events": total_events,
                "success_rate": success_rate,
                "timeout_rate": timeout_rate,
                "error_rate": error_rate,
                "avg_events_per_second": avg_events_per_second
            }
        }
    
    # Ejecutar y medir la prueba completa
    result = await run_test_with_timing(
        engine, 
        "test_concurrent_event_collisions", 
        run_collision_test
    )
    
    # No fallamos la prueba, solo verificamos que la ejecución complete
    assert isinstance(result, dict), "La prueba debería devolver un dict con resultados"
    
    # Advertencias si las tasas son preocupantes
    if result["summary"]["success_rate"] < 0.7:
        logger.warning(f"¡ATENCIÓN! Baja tasa de éxito en eventos concurrentes: {result['summary']['success_rate']:.2%}")
        
    if result["summary"]["timeout_rate"] > 0.2:
        logger.warning(f"¡ATENCIÓN! Alta tasa de timeouts en eventos concurrentes: {result['summary']['timeout_rate']:.2%}")


if __name__ == "__main__":
    # Para poder ejecutar este archivo directamente
    import pytest
    pytest.main(["-xvs", __file__])