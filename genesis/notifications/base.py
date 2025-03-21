"""
Clases base para las notificaciones.

Este módulo define interfaces para todos los tipos de notificaciones
soportados por el sistema Genesis.
"""

import abc
from typing import Dict, Any, Optional

class NotificationChannel(abc.ABC):
    """Clase base abstracta para canales de notificación."""
    
    def __init__(self, name: str):
        """
        Inicializar el canal de notificación.
        
        Args:
            name: Nombre del canal
        """
        self.name = name
        
    @abc.abstractmethod
    async def send(self, 
                  recipient: str, 
                  subject: str, 
                  message: str, 
                  **kwargs) -> bool:
        """
        Enviar una notificación.
        
        Args:
            recipient: Destinatario
            subject: Asunto
            message: Mensaje
            **kwargs: Argumentos adicionales
            
        Returns:
            True si se envió correctamente, False en caso contrario
        """
        pass