"""
Servidor de API para el sistema Genesis.

Este módulo proporciona un servidor de API que integra los diferentes endpoints
y gestiona las solicitudes externas al sistema.
"""

import os
import json
import asyncio
import threading
import logging
from typing import Dict, List, Any, Optional, Callable, Awaitable
import time
import signal

from flask import Flask
from werkzeug.serving import make_server

from genesis.core.base import Component
from genesis.utils.logger import setup_logging
from genesis.api.init_api import init_api


class APIServer(Component):
    """
    Servidor de API para el sistema Genesis.
    
    Este componente proporciona una API para acceso externo al sistema.
    """
    
    def __init__(
        self,
        name: str = "api_server",
        host: str = "0.0.0.0",
        port: int = 5000,
        debug: bool = False
    ):
        """
        Inicializar el servidor de API.
        
        Args:
            name: Nombre del componente
            host: Host para escuchar solicitudes
            port: Puerto para escuchar solicitudes
            debug: Ejecutar en modo debug
        """
        super().__init__(name)
        self.logger = setup_logging(name)
        
        # Configuración del servidor
        self.host = host
        self.port = port
        self.debug = debug
        
        # Estado del servidor
        self.running = False
        self.server = None
        self.server_thread = None
        
        # Bus de eventos
        self.event_bus = None
        
        # Diccionario para almacenar contexto y datos compartidos
        self.shared_context = {
            "engine": None,
            "components": {}
        }
    
    async def start(self) -> None:
        """Iniciar el servidor de API."""
        await super().start()
        
        # Iniciar el servidor Flask en un thread separado
        self._start_server()
        
        self.logger.info(f"Servidor de API iniciado en {self.host}:{self.port}")
    
    async def stop(self) -> None:
        """Detener el servidor de API."""
        self._stop_server()
        
        await super().stop()
        self.logger.info("Servidor de API detenido")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente de origen
        """
        # Almacenar referencia al bus de eventos la primera vez
        if event_type == "system.init" and "event_bus" in data:
            self.event_bus = data["event_bus"]
            self.shared_context["event_bus"] = self.event_bus
            self.logger.debug("Referencia al bus de eventos almacenada")
        
        # Almacenar referencia al motor principal
        elif event_type == "system.engine_ready" and "engine" in data:
            self.shared_context["engine"] = data["engine"]
            self.logger.debug("Referencia al motor almacenada")
        
        # Almacenar referencias a componentes importantes
        elif event_type == "system.component_ready":
            component_type = data.get("type")
            component = data.get("component")
            if component_type and component:
                self.shared_context["components"][component_type] = component
                self.logger.debug(f"Componente registrado: {component_type}")
    
    def _create_app(self) -> Flask:
        """
        Crear la aplicación Flask.
        
        Returns:
            Aplicación Flask configurada
        """
        from app import app
        
        # Exponer el contexto compartido a la aplicación Flask
        app.config["GENESIS_CONTEXT"] = self.shared_context
        
        # Inicializar la API REST
        init_api(app)
        
        return app
    
    def _start_server(self) -> None:
        """Iniciar el servidor en un thread separado."""
        if self.running:
            return
        
        self.running = True
        
        # Crear y configurar la aplicación Flask
        app = self._create_app()
        
        # Función para ejecutar el servidor
        def run_server():
            self.logger.info(f"Iniciando servidor HTTP en {self.host}:{self.port}")
            
            # Crear servidor
            self.server = make_server(self.host, self.port, app)
            
            # Configurar manejo de señales para SIGINT y SIGTERM
            original_sigint_handler = signal.getsignal(signal.SIGINT)
            original_sigterm_handler = signal.getsignal(signal.SIGTERM)
            
            def handle_signal(sig, frame):
                self.logger.info(f"Señal recibida ({sig}), deteniendo servidor...")
                # Restaurar manejadores originales
                signal.signal(signal.SIGINT, original_sigint_handler)
                signal.signal(signal.SIGTERM, original_sigterm_handler)
                
                # Detener el servidor
                if self.server:
                    self.server.shutdown()
            
            # Establecer manejadores
            signal.signal(signal.SIGINT, handle_signal)
            signal.signal(signal.SIGTERM, handle_signal)
            
            # Iniciar servidor
            try:
                self.server.serve_forever()
            finally:
                self.logger.info("Servidor HTTP detenido")
                self.running = False
        
        # Iniciar el thread del servidor
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
    
    def _stop_server(self) -> None:
        """Detener el servidor HTTP."""
        if not self.running or not self.server:
            return
        
        self.logger.info("Deteniendo servidor HTTP...")
        
        # Detener el servidor
        if self.server:
            self.server.shutdown()
        
        # Esperar al thread
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5)
        
        self.running = False
        self.logger.info("Servidor HTTP detenido")


# Exportación para uso fácil
api_server = APIServer()