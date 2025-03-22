"""
Adaptador para componentes existentes de Genesis al sistema híbrido.

Este módulo proporciona un adaptador que permite usar componentes
existentes del sistema Genesis (implementados con la interfaz 
Component original) con el nuevo sistema híbrido API + WebSocket.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Set, Type

from genesis.core.component import Component
from genesis.core.genesis_hybrid import ComponentAPI, GenesisHybridCoordinator

logger = logging.getLogger(__name__)

class ComponentAdapter(ComponentAPI):
    """
    Adaptador para componentes existentes de Genesis al sistema híbrido.
    
    Este adaptador permite usar componentes existentes con la interfaz Component
    en el nuevo sistema híbrido, traduciendo las llamadas entre ambos sistemas.
    """
    
    def __init__(self, component: Component, coordinator: GenesisHybridCoordinator):
        """
        Inicializar adaptador para un componente existente.
        
        Args:
            component: Componente existente de Genesis a adaptar
            coordinator: Coordinador híbrido para registro de eventos
        """
        super().__init__(component.id)
        self.component = component
        self.coordinator = coordinator
        self.response_futures: Dict[str, asyncio.Future] = {}
        self.last_request_id = 0
        self.responses_pending: Set[str] = set()
        
    async def process_request(self, request_type: str, data: Dict[str, Any], source: str) -> Any:
        """
        Procesa una solicitud API traduciéndola al formato antiguo.
        
        Args:
            request_type: Tipo de solicitud
            data: Datos de la solicitud
            source: ID del componente o sistema que origina la solicitud
            
        Returns:
            Resultado de la solicitud, en el formato que devuelve handle_event
        """
        # Crear un future para la respuesta
        request_id = f"{request_type}_{self.last_request_id}"
        self.last_request_id += 1
        
        response_future = asyncio.Future()
        self.response_futures[request_id] = response_future
        self.responses_pending.add(request_id)
        
        # Añadir el ID de solicitud a los datos para rastrear la respuesta
        request_data = {**data, "__request_id": request_id}
        
        try:
            # Llamar al método handle_event del componente original
            logger.debug(f"Adaptador {self.id}: solicitud {request_type} de {source}")
            
            response = await self.component.handle_event(request_type, request_data, source)
            
            # Si handle_event devuelve algo inmediatamente, usarlo directamente
            if response is not None:
                self.responses_pending.remove(request_id)
                self.response_futures.pop(request_id, None)
                return response
            
            # Esperar una respuesta asíncrona (si el componente usa emit_event)
            try:
                return await asyncio.wait_for(response_future, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"Adaptador {self.id}: timeout esperando respuesta para {request_id}")
                return None
            finally:
                self.responses_pending.discard(request_id)
                self.response_futures.pop(request_id, None)
                
        except Exception as e:
            logger.error(f"Adaptador {self.id}: error en process_request: {e}")
            self.responses_pending.discard(request_id)
            self.response_futures.pop(request_id, None)
            return None
    
    async def on_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Recibe un evento WebSocket y lo reenvía al componente adaptado.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: ID del componente que originó el evento
        """
        await super().on_event(event_type, data, source)
        
        try:
            # Verificar si es una respuesta a una solicitud anterior
            request_id = data.get("__request_id")
            if request_id and request_id in self.responses_pending:
                future = self.response_futures.get(request_id)
                if future and not future.done():
                    future.set_result(data.get("result"))
                    return
            
            # Si no es una respuesta, procesarlo como evento normal
            await self.component.handle_event(event_type, data, source)
            
        except Exception as e:
            logger.error(f"Adaptador {self.id}: error en on_event: {e}")
    
    async def start(self) -> None:
        """Iniciar el componente adaptado."""
        await super().start()
        await self.component.start()
        logger.info(f"Adaptador {self.id}: componente adaptado iniciado")
    
    async def stop(self) -> None:
        """Detener el componente adaptado."""
        await self.component.stop()
        await super().stop()
        logger.info(f"Adaptador {self.id}: componente adaptado detenido")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado del componente adaptado.
        
        Returns:
            Diccionario con estado completo
        """
        basic_status = super().get_status()
        
        # Añadir información específica del adaptador
        adapted_status = {
            "component_type": self.component.__class__.__name__,
            "pending_responses": len(self.responses_pending),
            "adapter_metrics": {
                "last_request_id": self.last_request_id
            }
        }
        
        # Combinar estados
        return {**basic_status, **adapted_status}
    
    # Implementación de métodos del motor original para que el componente
    # existente pueda seguir usándolos
    
    async def emit_event(self, event_type: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Emular el método emit_event del motor original.
        
        Esta función permite que los componentes adaptados sigan usando
        emit_event como lo hacían antes, pero internamente usa el sistema híbrido.
        
        Args:
            event_type: Tipo de evento a emitir
            data: Datos del evento
            
        Returns:
            Lista de respuestas (para mantener compatibilidad)
        """
        # Verificar si es una respuesta a una solicitud
        request_id = data.get("__request_id")
        if request_id:
            # Es una respuesta, emitirla solo al origen de la solicitud
            source_id = data.get("__source", "unknown")
            response_data = {**data, "result": data.get("result")}
            
            # Enviar como solicitud API directa
            await self.coordinator.request(
                source_id, 
                f"response_{event_type}", 
                response_data, 
                self.id
            )
            return [{"status": "sent", "target": source_id}]
        
        # Es un evento normal, emitirlo por WebSocket
        await self.coordinator.broadcast_event(
            event_type, 
            data, 
            self.id
        )
        
        # Para mantener compatibilidad, devolver una lista vacía de respuestas
        return []
    
    async def request(self, target_id: str, request_type: str, 
                    data: Dict[str, Any], timeout: float = 5.0) -> Optional[Any]:
        """
        Emular el método request para mantener compatibilidad.
        
        Args:
            target_id: ID del componente destino
            request_type: Tipo de solicitud
            data: Datos de la solicitud
            timeout: Tiempo máximo de espera
            
        Returns:
            Resultado de la solicitud o None si hubo error
        """
        return await self.coordinator.request(
            target_id, 
            request_type, 
            data, 
            self.id, 
            timeout
        )

class HybridEngineAdapter:
    """
    Adaptador del motor híbrido completo para componentes existentes.
    
    Esta clase permite migrar fácilmente componentes existentes al nuevo sistema
    híbrido sin tener que modificarlos. Actúa como una capa de compatibilidad.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        """
        Inicializar el adaptador de motor híbrido.
        
        Args:
            host: Host para el servidor web
            port: Puerto para el servidor web
        """
        self.coordinator = GenesisHybridCoordinator(host, port)
        self.adapters: Dict[str, ComponentAdapter] = {}
        self.components: Dict[str, Component] = {}
    
    def register_component(self, component: Component, depends_on: Optional[List[str]] = None) -> None:
        """
        Registrar un componente existente adaptándolo al sistema híbrido.
        
        Args:
            component: Componente existente a registrar
            depends_on: Lista de IDs de componentes de los que depende (opcional)
        """
        # Crear un adaptador para el componente
        adapter = ComponentAdapter(component, self.coordinator)
        
        # Registrar el adaptador en el coordinador
        self.coordinator.register_component(
            component.id,
            adapter,
            depends_on
        )
        
        # Guardar referencias
        self.adapters[component.id] = adapter
        self.components[component.id] = component
        
        logger.info(f"Componente {component.id} adaptado y registrado en sistema híbrido")
    
    async def start(self) -> None:
        """Iniciar el motor híbrido adaptado."""
        await self.coordinator.start()
    
    async def stop(self) -> None:
        """Detener el motor híbrido adaptado."""
        await self.coordinator.stop()
    
    async def emit_event(self, event_type: str, data: Dict[str, Any], 
                        source: str) -> List[Dict[str, Any]]:
        """
        Emitir un evento desde el motor para compatibilidad con código existente.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: ID del componente origen
            
        Returns:
            Lista de respuestas (para mantener compatibilidad)
        """
        # Emitir por WebSocket
        await self.coordinator.broadcast_event(event_type, data, source)
        
        # Para mantener compatibilidad, devolver una lista vacía de respuestas
        return []
    
    async def request(self, target_id: str, request_type: str, 
                     data: Dict[str, Any], source: str, 
                     timeout: float = 5.0) -> Optional[Any]:
        """
        Realizar una solicitud directa desde el motor.
        
        Args:
            target_id: ID del componente destino
            request_type: Tipo de solicitud
            data: Datos de la solicitud
            source: ID del origen
            timeout: Tiempo máximo de espera
            
        Returns:
            Resultado de la solicitud o None si hubo error
        """
        return await self.coordinator.request(
            target_id, 
            request_type, 
            data, 
            source, 
            timeout
        )
    
    def get_component(self, component_id: str) -> Optional[Component]:
        """
        Obtener un componente por su ID.
        
        Args:
            component_id: ID del componente
            
        Returns:
            Instancia del componente o None si no existe
        """
        return self.components.get(component_id)
    
    def get_all_components(self) -> Dict[str, Component]:
        """
        Obtener todos los componentes registrados.
        
        Returns:
            Diccionario de componentes por ID
        """
        return self.components.copy()
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Obtener estado completo del sistema.
        
        Returns:
            Diccionario con estado del sistema
        """
        # Recolectar estado de componentes
        component_status = {}
        for comp_id, adapter in self.adapters.items():
            component_status[comp_id] = adapter.get_status()
        
        return {
            "components": component_status,
            "component_count": len(self.components),
            "adapter_count": len(self.adapters)
        }