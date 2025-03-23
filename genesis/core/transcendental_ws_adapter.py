"""
Adaptador para integrar el WebSocket Externo Trascendental con el sistema core Genesis.

Este adaptador permite que el WebSocket Externo Trascendental se integre
de manera fluida con la arquitectura actual del sistema Genesis, reemplazando
el manejador de WebSocket externo original con capacidades trascendentales.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable, Coroutine
from aiohttp import web

# Importar el WebSocket Externo Trascendental
from genesis.core.transcendental_external_websocket import TranscendentalExternalWebSocket

# Configuración de logging
logger = logging.getLogger("Genesis.WSAdapter")

class TranscendentalWebSocketAdapter:
    """
    Adaptador que integra el WebSocket Externo Trascendental con el sistema Genesis.
    
    Este adaptador reemplaza el manejador _external_websocket_handler original
    en GenesisHybridCoordinator, proporcionando todas las capacidades trascendentales
    mientras mantiene compatibilidad con la interfaz existente.
    """
    
    def __init__(self, coordinator: Any):
        """
        Inicializar adaptador con referencia al coordinador.
        
        Args:
            coordinator: Instancia de GenesisHybridCoordinator o similar
        """
        self.coordinator = coordinator
        self.ws_trascendental = TranscendentalExternalWebSocket()
        self.original_handler = getattr(coordinator, "_external_websocket_handler", None)
        
        logger.info("Adaptador WebSocket Trascendental inicializado")

    async def connect_to_core(self) -> None:
        """
        Conecta el WebSocket Trascendental al sistema core.
        
        Este método reemplaza el manejador original con la versión trascendental,
        manteniendo una referencia al original para compatibilidad.
        """
        if not hasattr(self.coordinator, "_external_websocket_handler"):
            logger.error("Coordinador no tiene manejador WebSocket externo")
            return
            
        # Guardar método original (por si necesitamos restaurar)
        if not self.original_handler:
            self.original_handler = self.coordinator._external_websocket_handler
            
        # Reemplazar con nuestro manejador trascendental
        self.coordinator._external_websocket_handler = self.trascendental_handler
        
        logger.info("WebSocket Externo Trascendental conectado al sistema core")
            
    async def disconnect_from_core(self) -> None:
        """
        Desconecta el WebSocket Trascendental del sistema core.
        
        Este método restaura el manejador original si está disponible.
        """
        if self.original_handler and hasattr(self.coordinator, "_external_websocket_handler"):
            self.coordinator._external_websocket_handler = self.original_handler
            logger.info("Restaurado manejador WebSocket original")
    
    async def trascendental_handler(self, request: web.Request) -> web.WebSocketResponse:
        """
        Manejador trascendental que procesa solicitudes WebSocket.
        
        Este método adapta la interfaz del WebSocket Trascendental al sistema core,
        capturando conexiones WebSocket y procesándolas con capacidades trascendentales.
        
        Args:
            request: Solicitud HTTP con conexión WebSocket
            
        Returns:
            Respuesta WebSocket procesada trascendentalmente
        """
        # Usar directamente el manejador trascendental
        return await self.ws_trascendental.handle_connection(request)
    
    async def send_message(self, component_id: str, message: Dict[str, Any]) -> bool:
        """
        Envía un mensaje a través del WebSocket Trascendental.
        
        Args:
            component_id: ID del componente destinatario
            message: Mensaje a enviar
            
        Returns:
            True si se envió correctamente, False en caso contrario
        """
        return await self.ws_trascendental.send_message_transcendentally(component_id, message)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del WebSocket Trascendental.
        
        Returns:
            Diccionario con estadísticas
        """
        return self.ws_trascendental.get_stats()