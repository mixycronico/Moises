"""
Extensión para la base de datos transcendental con métodos adicionales.

Este módulo añade funcionalidad a la clase TranscendentalDatabase para
proporcionar mayor compatibilidad y flexibilidad.
"""
import logging
import sys
import inspect
from typing import Dict, Any, Optional

# Configuración de logging
logger = logging.getLogger("genesis.db.trascendental_extension")

def initialize_extensions():
    """
    Inicializar extensiones para la clase TranscendentalDatabase.
    
    Añade métodos dinámicamente a la clase para proporcionar
    funcionalidad adicional que no existe en la implementación original.
    """
    try:
        # Importar la clase original
        from genesis.db.transcendental_database import TranscendentalDatabase
        
        # Comprobar si ya tiene los métodos que queremos añadir
        if hasattr(TranscendentalDatabase, 'configure') and hasattr(TranscendentalDatabase, 'set_connection_url'):
            logger.debug("La clase TranscendentalDatabase ya tiene las extensiones necesarias")
            return True
        
        # Definir los métodos que queremos añadir
        def configure(self, url: str) -> None:
            """
            Configurar la URL de conexión.
            
            Args:
                url: URL de conexión a la base de datos
            """
            # Establecer URL de conexión
            if hasattr(self, 'url'):
                self.url = url
            elif hasattr(self, 'connection_url'):
                self.connection_url = url
            else:
                # Crear el atributo si no existe
                self.__dict__['url'] = url
            
            logger.info(f"TranscendentalDatabase configurada con nueva URL de conexión")
        
        def set_connection_url(self, url: str) -> None:
            """
            Establecer URL de conexión.
            
            Args:
                url: URL de conexión a la base de datos
            """
            # Este es un método alternativo por si configure no está disponible
            if hasattr(self, 'configure'):
                self.configure(url)
            else:
                if hasattr(self, 'url'):
                    self.url = url
                elif hasattr(self, 'connection_url'):
                    self.connection_url = url
                else:
                    # Crear el atributo si no existe
                    self.__dict__['url'] = url
            
            logger.info(f"URL de conexión actualizada para TranscendentalDatabase")
        
        def list_checkpoints(self) -> Dict[str, Any]:
            """
            Listar checkpoints disponibles.
            
            Returns:
                Diccionario con información de checkpoints
            """
            if hasattr(self, 'checkpoint') and hasattr(self.checkpoint, 'list_all'):
                checkpoints = self.checkpoint.list_all()
                result = {}
                for cp_id in checkpoints:
                    metadata = self.checkpoint.get_metadata(cp_id) or {}
                    result[cp_id] = metadata
                return result
            return {}
        
        # Añadir los métodos a la clase
        setattr(TranscendentalDatabase, 'configure', configure)
        setattr(TranscendentalDatabase, 'set_connection_url', set_connection_url)
        setattr(TranscendentalDatabase, 'list_checkpoints', list_checkpoints)
        
        logger.info("Extensiones añadidas correctamente a TranscendentalDatabase")
        return True
        
    except Exception as e:
        logger.error(f"Error al inicializar extensiones: {str(e)}")
        return False

# Inicializar extensiones automáticamente al importar el módulo
success = initialize_extensions()