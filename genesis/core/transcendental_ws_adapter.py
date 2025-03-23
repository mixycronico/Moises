"""
Adaptador para integrar el WebSocket Externo Trascendental con el sistema core Genesis.

Este adaptador permite que el WebSocket Externo Trascendental se integre
de manera fluida con la arquitectura actual del sistema Genesis, reemplazando
el manejador de WebSocket externo original con capacidades trascendentales.
"""

import logging
import asyncio
import json
from typing import Dict, Any, Optional, List, Callable, Coroutine, Union
import websockets
from websockets.server import WebSocketServerProtocol

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
    
    def __init__(self, coordinator: Any, host: str = "0.0.0.0", port: int = 8080):
        """
        Inicializar adaptador con referencia al coordinador.
        
        Args:
            coordinator: Instancia de GenesisHybridCoordinator o similar
            host: Host para WebSocket trascendental
            port: Puerto para WebSocket trascendental
        """
        self.coordinator = coordinator
        self.host = host
        self.port = port
        self.ws_trascendental = TranscendentalExternalWebSocket(host, port)
        
        # Guardar referencia al manejador original
        self.original_handler = None
        if hasattr(coordinator, "_external_websocket_handler"):
            self.original_handler = coordinator._external_websocket_handler
        
        self._server = None
        self._running = False
        
        logger.info("Adaptador WebSocket Trascendental inicializado")

    async def connect_to_core(self) -> None:
        """
        Conecta el WebSocket Trascendental al sistema core.
        
        Este método inicia el servidor WebSocket Trascendental y lo integra
        con el sistema core, reemplazando el manejador original.
        """
        if self._running:
            logger.warning("WebSocket Trascendental ya está conectado")
            return
            
        # Reemplazar manejador en el coordinador si aplica
        if hasattr(self.coordinator, "_external_websocket_handler"):
            if not self.original_handler:
                self.original_handler = self.coordinator._external_websocket_handler
                
            # Iniciar el servidor WebSocket
            await self._start_server()
            
            # Reemplazar con nuestro manejador adaptado
            self.coordinator._external_websocket_handler = self.trascendental_handler
            
            logger.info("WebSocket Externo Trascendental conectado al sistema core")
        else:
            # Si no hay coordinador con manejador, simplemente iniciar el servidor
            await self._start_server()
            logger.info("WebSocket Externo Trascendental iniciado en modo independiente")
            
    async def disconnect_from_core(self) -> None:
        """
        Desconecta el WebSocket Trascendental del sistema core.
        
        Este método detiene el servidor WebSocket y restaura el manejador original.
        """
        # Detener servidor
        await self._stop_server()
        
        # Restaurar manejador original
        if self.original_handler and hasattr(self.coordinator, "_external_websocket_handler"):
            self.coordinator._external_websocket_handler = self.original_handler
            logger.info("Restaurado manejador WebSocket original")
    
    async def _start_server(self) -> None:
        """Iniciar servidor WebSocket Trascendental."""
        if self._running:
            return
            
        try:
            # Iniciar el WebSocket Trascendental
            asyncio.create_task(self.ws_trascendental.start())
            self._running = True
            logger.info(f"Servidor WebSocket Trascendental iniciado en {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Error al iniciar servidor WebSocket Trascendental: {e}")
            # Intentar transmutación de error si está disponible
            try:
                if hasattr(self.ws_trascendental, "mechanisms") and "horizon" in self.ws_trascendental.mechanisms:
                    await self.ws_trascendental.mechanisms["horizon"].transmute_error(
                        e, {"operation": "start_server"}
                    )
            except Exception:
                pass  # Ignorar errores secundarios
    
    async def _stop_server(self) -> None:
        """Detener servidor WebSocket Trascendental."""
        if not self._running:
            return
            
        try:
            # Detener el WebSocket Trascendental
            await self.ws_trascendental.stop()
            self._running = False
            logger.info("Servidor WebSocket Trascendental detenido")
        except Exception as e:
            logger.error(f"Error al detener servidor WebSocket Trascendental: {e}")
    
    async def trascendental_handler(self, websocket: WebSocketServerProtocol, path: str) -> None:
        """
        Manejador trascendental que procesa conexiones WebSocket.
        
        Este método adapta la interfaz del sistema core al WebSocket Trascendental,
        delegando el manejo de la conexión al WebSocket Trascendental.
        
        Args:
            websocket: Conexión WebSocket
            path: Ruta de la conexión
        """
        # Delegar directamente al manejador del WebSocket Trascendental
        await self.ws_trascendental.handle_connection(websocket, path)
    
    async def send_message(self, component_id: str, message: Dict[str, Any]) -> bool:
        """
        Envía un mensaje a través del WebSocket Trascendental.
        
        Args:
            component_id: ID del componente destinatario
            message: Mensaje a enviar
            
        Returns:
            True si se envió correctamente, False en caso contrario
        """
        if isinstance(message, dict):
            return await self.ws_trascendental.send_message_transcendentally(component_id, message)
        elif isinstance(message, str):
            try:
                # Intentar parsear como JSON
                data = json.loads(message)
                return await self.ws_trascendental.send_message_transcendentally(component_id, data)
            except json.JSONDecodeError:
                # Enviar como texto plano envuelto en dict
                return await self.ws_trascendental.send_message_transcendentally(
                    component_id, {"text": message, "timestamp": asyncio.get_event_loop().time()}
                )
        else:
            # Convertir a string y envolver
            return await self.ws_trascendental.send_message_transcendentally(
                component_id, {"data": str(message), "timestamp": asyncio.get_event_loop().time()}
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del WebSocket Trascendental.
        
        Returns:
            Diccionario con estadísticas
        """
        return self.ws_trascendental.get_stats()
    
    def is_running(self) -> bool:
        """
        Verifica si el WebSocket Trascendental está en ejecución.
        
        Returns:
            True si está en ejecución, False en caso contrario
        """
        return self._running