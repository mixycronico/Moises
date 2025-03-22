"""
Utilidades para ayudar a gestionar timeouts en pruebas asíncronas.

Este módulo proporciona funciones auxiliares para emitir eventos con timeouts
y realizar mediciones de tiempo en pruebas asíncronas del motor Genesis.
"""

import time
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, TypeVar, Coroutine

# Configuración del logger
logger = logging.getLogger(__name__)

T = TypeVar('T')

async def emit_with_timeout(
    engine, 
    event_type: str, 
    data: Dict[str, Any], 
    source: str, 
    timeout: float = 5.0
) -> List[Dict[str, Any]]:
    """
    Emitir evento con timeout y manejo robusto de errores.
    
    Args:
        engine: Instancia del motor de eventos
        event_type: Tipo de evento a emitir
        data: Datos del evento
        source: Fuente del evento
        timeout: Tiempo máximo de espera en segundos
        
    Returns:
        Lista de respuestas o un objeto de error si falla
    """
    try:
        response = await asyncio.wait_for(
            engine.emit_event_with_response(event_type, data, source),
            timeout=timeout
        )
        
        # Manejar el caso de respuesta None
        if response is None:
            logger.warning(f"No response for {event_type} from {source}")
            return [{"healthy": False, 
                    "error": f"No response for {event_type} from {source}",
                    "event": event_type, 
                    "source": source}]
        
        return response
        
    except asyncio.TimeoutError:
        logger.warning(f"Timeout de {timeout}s al emitir {event_type} desde {source}")
        return [{"healthy": False, 
                "error": "timeout", 
                "event": event_type, 
                "source": source}]
                
    except Exception as e:
        logger.error(f"Error inesperado en {event_type} desde {source}: {str(e)}")
        return [{"healthy": False, 
                "error": str(e), 
                "event": event_type, 
                "source": source}]

async def check_component_status(engine, component_id: str, timeout: float = 2.0) -> Dict[str, Any]:
    """
    Verificar el estado de un componente con timeout.
    
    Args:
        engine: Instancia del motor de eventos
        component_id: ID del componente a verificar
        timeout: Tiempo máximo de espera en segundos
        
    Returns:
        Diccionario con el estado del componente
    """
    resp = await emit_with_timeout(engine, "check_status", {}, component_id, timeout=timeout)
    # Extraer el primer elemento si es una lista, o usar un valor por defecto
    if isinstance(resp, list) and len(resp) > 0:
        return resp[0]
    return {"healthy": False, "error": f"Respuesta inválida de {component_id}"}

async def run_test_with_timing(engine, test_name: str, test_func: Callable[[Any], Coroutine[Any, Any, T]]) -> T:
    """
    Ejecutar una función de prueba midiendo su tiempo de ejecución.
    
    Args:
        engine: Instancia del motor de eventos
        test_name: Nombre de la prueba (para logging)
        test_func: Función asíncrona que contiene la prueba
        
    Returns:
        El resultado de la función de prueba
    """
    start_time = time.time()
    result = await test_func(engine)
    elapsed = time.time() - start_time
    logger.info(f"{test_name} completado en {elapsed:.3f} segundos")
    return result

async def cleanup_engine(engine):
    """
    Limpieza completa del motor y tareas pendientes.
    
    Args:
        engine: Instancia del motor de eventos a limpiar
    """
    # Desregistrar todos los componentes
    if hasattr(engine, 'components') and engine.components:
        for component_name in list(engine.components.keys()):
            try:
                await engine.unregister_component(component_name)
            except Exception as e:
                logger.warning(f"Error al desregistrar componente {component_name}: {str(e)}")
    
    # Detener el motor
    if hasattr(engine, 'stop'):
        await engine.stop()
    
    # Cancelar tareas pendientes
    pending = [t for t in asyncio.all_tasks() 
              if not t.done() and t != asyncio.current_task()]
    
    if pending:
        logger.warning(f"Cancelando {len(pending)} tareas pendientes")
        for task in pending:
            task.cancel()
        
        try:
            await asyncio.gather(*pending, return_exceptions=True)
        except Exception as e:
            logger.warning(f"Error durante la cancelación de tareas: {str(e)}")