"""
Implementación de entidades especializadas en comunicación WebSocket para Sistema Genesis.

Este módulo implementa dos entidades especializadas:
1. Hermes: Entidad para WebSocket local
2. Apollo: Entidad para WebSocket externo/remoto
"""

import os
import logging
import random
import time
import threading
import json
import asyncio
import websockets
from typing import Dict, Any, List, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor

from enhanced_simple_cosmic_trader import EnhancedCosmicTrader

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebSocketEntity(EnhancedCosmicTrader):
    """
    Clase base para entidades de comunicación WebSocket.
    Extiende las capacidades de la entidad de trading para enfocarse en
    la gestión de comunicaciones en tiempo real.
    """
    
    def __init__(self, name: str, role: str = "Communication", father: str = "otoniel", 
                 frequency_seconds: int = 30, ws_scope: str = "local"):
        """
        Inicializar entidad WebSocket.
        
        Args:
            name: Nombre de la entidad
            role: Rol (normalmente "Communication")
            father: Nombre del creador/dueño
            frequency_seconds: Período de ciclo de vida en segundos
            ws_scope: Ámbito del WebSocket ("local" o "external")
        """
        super().__init__(name, role, father, frequency_seconds)
        
        self.ws_scope = ws_scope
        self.server = None
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.running = False
        self.messages_sent = 0
        self.messages_received = 0
        self.connection_errors = 0
        self.last_heartbeat = time.time()
        self.heartbeat_interval = 30  # segundos
        
        # Estadísticas específicas
        self.stats = {
            "total_connections": 0,
            "peak_concurrent": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0,
            "uptime_seconds": 0,
            "start_time": time.time()
        }
        
        # Personalidad y rasgos específicos
        self.personality_traits = ["Comunicativo", "Veloz", "Adaptable"]
        self.emotional_volatility = 0.5  # Volatilidad emocional media
        
        # Especializaciones
        self.specializations = {
            "Real-time Communication": 0.9,
            "Data Streaming": 0.8,
            "Connection Management": 0.7,
            "Protocol Translation": 0.8,
            "Signal Processing": 0.6
        }
        
        # Estado del servidor
        self.server_status = "Inactivo"
        
        logger.info(f"[{self.name}] Entidad WebSocket ({ws_scope}) inicializada")
    
    async def handle_client(self, websocket, path):
        """
        Manejar conexión de cliente WebSocket.
        
        Args:
            websocket: Conexión WebSocket
            path: Ruta de la conexión
        """
        # Registrar cliente
        self.connected_clients.add(websocket)
        self.stats["total_connections"] += 1
        
        # Actualizar pico concurrente
        concurrent = len(self.connected_clients)
        if concurrent > self.stats["peak_concurrent"]:
            self.stats["peak_concurrent"] = concurrent
        
        client_id = id(websocket)
        logger.info(f"[{self.name}] Nueva conexión [{client_id}] en {path}, total: {concurrent}")
        
        try:
            # Enviar mensaje de bienvenida
            welcome_msg = {
                "type": "welcome",
                "sender": self.name,
                "message": f"Bienvenido a la red cósmica. Comunicación establecida con {self.name}.",
                "timestamp": time.time()
            }
            await websocket.send(json.dumps(welcome_msg))
            self.messages_sent += 1
            self.stats["messages_sent"] += 1
            
            # Ciclo de recepción de mensajes
            async for message in websocket:
                self.messages_received += 1
                self.stats["messages_received"] += 1
                
                try:
                    # Procesar mensaje
                    data = json.loads(message)
                    logger.info(f"[{self.name}] Mensaje recibido de [{client_id}]: {data.get('type', 'unknown')}")
                    
                    # Manejar diferentes tipos de mensajes
                    await self.process_client_message(websocket, data)
                    
                except json.JSONDecodeError:
                    logger.warning(f"[{self.name}] Mensaje no-JSON recibido: {message[:50]}...")
                    error_msg = {
                        "type": "error",
                        "sender": self.name,
                        "message": "Formato JSON inválido",
                        "timestamp": time.time()
                    }
                    await websocket.send(json.dumps(error_msg))
                    self.messages_sent += 1
                    self.stats["messages_sent"] += 1
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"[{self.name}] Conexión cerrada [{client_id}]")
        
        except Exception as e:
            logger.error(f"[{self.name}] Error manejando cliente [{client_id}]: {str(e)}")
            self.stats["errors"] += 1
        
        finally:
            # Desregistrar cliente
            self.connected_clients.remove(websocket)
            logger.info(f"[{self.name}] Cliente desconectado [{client_id}], quedan: {len(self.connected_clients)}")
    
    async def process_client_message(self, websocket, data):
        """
        Procesar mensaje recibido de cliente.
        
        Args:
            websocket: Conexión WebSocket del cliente
            data: Datos del mensaje recibido
        """
        # Obtener tipo de mensaje
        msg_type = data.get("type", "unknown")
        
        # Manejar diferentes tipos de mensajes
        if msg_type == "ping":
            # Responder con pong
            response = {
                "type": "pong",
                "sender": self.name,
                "timestamp": time.time()
            }
            await websocket.send(json.dumps(response))
            
        elif msg_type == "broadcast":
            # Reenviar mensaje a todos los clientes
            broadcast_msg = {
                "type": "broadcast",
                "sender": data.get("sender", "anonymous"),
                "message": data.get("message", ""),
                "timestamp": time.time(),
                "relayed_by": self.name
            }
            await self.broadcast_to_clients(broadcast_msg)
            
        elif msg_type == "status":
            # Enviar estado actual del sistema
            status_msg = {
                "type": "status",
                "sender": self.name,
                "status": self.get_status(),
                "timestamp": time.time()
            }
            await websocket.send(json.dumps(status_msg))
            
        elif msg_type == "command":
            # Procesar comando (requiere autenticación)
            if data.get("auth_token") == "cosmic_secret":
                response = await self.process_command(data.get("command"), data.get("params", {}))
                await websocket.send(json.dumps(response))
            else:
                error_msg = {
                    "type": "error",
                    "sender": self.name,
                    "message": "Autenticación requerida para comandos",
                    "timestamp": time.time()
                }
                await websocket.send(json.dumps(error_msg))
        
        else:
            # Mensaje desconocido
            unknown_msg = {
                "type": "ack",
                "sender": self.name,
                "message": f"Mensaje de tipo '{msg_type}' recibido",
                "timestamp": time.time()
            }
            await websocket.send(json.dumps(unknown_msg))
        
        self.messages_sent += 1
        self.stats["messages_sent"] += 1
    
    async def process_command(self, command, params):
        """
        Procesar comando administrativo.
        
        Args:
            command: Comando a ejecutar
            params: Parámetros del comando
            
        Returns:
            Respuesta al comando
        """
        if command == "restart":
            # Simular reinicio (realmente solo actualiza estado)
            self.server_status = "Reiniciando"
            self.emotion = "Expectante"
            
            return {
                "type": "command_response",
                "sender": self.name,
                "command": command,
                "status": "success",
                "message": "Servicio reiniciado correctamente",
                "timestamp": time.time()
            }
            
        elif command == "stats":
            # Devolver estadísticas detalladas
            return {
                "type": "command_response",
                "sender": self.name,
                "command": command,
                "status": "success",
                "data": self.stats,
                "timestamp": time.time()
            }
            
        # Comando no reconocido
        return {
            "type": "command_response",
            "sender": self.name,
            "command": command,
            "status": "error",
            "message": "Comando no reconocido",
            "timestamp": time.time()
        }
    
    async def broadcast_to_clients(self, message):
        """
        Enviar mensaje a todos los clientes conectados.
        
        Args:
            message: Mensaje a enviar
        """
        if not isinstance(message, str):
            message = json.dumps(message)
            
        websockets_coroutines = [client.send(message) for client in self.connected_clients]
        
        if websockets_coroutines:
            await asyncio.gather(*websockets_coroutines, return_exceptions=True)
    
    async def send_heartbeat(self):
        """Enviar latido periódico a todos los clientes."""
        while self.running:
            try:
                # Solo enviar si hay clientes conectados
                if self.connected_clients:
                    heartbeat_msg = {
                        "type": "heartbeat",
                        "sender": self.name,
                        "timestamp": time.time(),
                        "clients_connected": len(self.connected_clients)
                    }
                    await self.broadcast_to_clients(heartbeat_msg)
                    self.messages_sent += len(self.connected_clients)
                    self.stats["messages_sent"] += len(self.connected_clients)
                    
                self.last_heartbeat = time.time()
                
            except Exception as e:
                logger.error(f"[{self.name}] Error enviando heartbeat: {str(e)}")
                self.stats["errors"] += 1
            
            await asyncio.sleep(self.heartbeat_interval)
    
    async def start_server(self, host: str, port: int):
        """
        Iniciar servidor WebSocket.
        
        Args:
            host: Dirección donde escuchar
            port: Puerto donde escuchar
        """
        if self.running:
            logger.warning(f"[{self.name}] El servidor ya está en ejecución")
            return
        
        try:
            self.server = await websockets.serve(self.handle_client, host, port)
            self.running = True
            self.server_status = "Activo"
            logger.info(f"[{self.name}] Servidor WebSocket iniciado en {host}:{port}")
            
            # Iniciar heartbeat en tarea separada
            heartbeat_task = asyncio.create_task(self.send_heartbeat())
            
            # Mantener servidor activo
            await asyncio.Future()  # Corre indefinidamente
            
        except Exception as e:
            logger.error(f"[{self.name}] Error iniciando servidor: {str(e)}")
            self.server_status = f"Error: {str(e)}"
            self.running = False
            self.stats["errors"] += 1
            
    def start_server_thread(self, host: str, port: int):
        """
        Iniciar servidor WebSocket en un thread separado.
        
        Args:
            host: Dirección donde escuchar
            port: Puerto donde escuchar
        """
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                loop.run_until_complete(self.start_server(host, port))
            except Exception as e:
                logger.error(f"[{self.name}] Error en thread del servidor: {str(e)}")
            finally:
                loop.close()
        
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()
        
        # Esperar a que inicie
        time.sleep(2)
        
        if self.running:
            logger.info(f"[{self.name}] Servidor iniciado correctamente")
            return True
        else:
            logger.error(f"[{self.name}] No se pudo iniciar el servidor")
            return False
    
    def process_cycle(self):
        """
        Procesar ciclo de vida de la entidad WebSocket.
        Sobreescribe el método de la clase base.
        """
        if not self.is_alive:
            return
        
        # Actualizar ciclo base
        super().process_base_cycle()
        
        # Ciclo específico de entidad WebSocket
        try:
            # Actualizar estadísticas
            self.stats["uptime_seconds"] = int(time.time() - self.stats["start_time"])
            
            # Actualizar estado
            self.update_state()
            
            # Generar mensaje informativo sobre conexiones (20% de probabilidad)
            if random.random() < 0.2:
                insight = self.generate_communication_insight()
                self.broadcast_message(insight)
                
        except Exception as e:
            logger.error(f"[{self.name}] Error en ciclo de proceso: {str(e)}")
            self.handle_error(str(e))
    
    def generate_communication_insight(self):
        """
        Generar insight sobre el estado de las comunicaciones.
        
        Returns:
            Mensaje con insight
        """
        insights = [
            f"Mantengo {len(self.connected_clients)} conexiones activas en este momento.",
            f"He procesado {self.stats['messages_sent']} mensajes salientes y {self.stats['messages_received']} entrantes.",
            f"El pico de conexiones concurrentes ha sido de {self.stats['peak_concurrent']}.",
            f"Llevo {self.stats['uptime_seconds'] // 3600} horas y {(self.stats['uptime_seconds'] % 3600) // 60} minutos en servicio continuo.",
            f"Mi esencia {self.dominant_trait} me permite optimizar el flujo de comunicaciones."
        ]
        
        # Elegir un insight aleatorio
        insight = random.choice(insights)
        
        # Formatear como mensaje
        return self.generate_message("insight", insight)
    
    def handle_error(self, error_message: str):
        """
        Manejar error de comunicación.
        
        Args:
            error_message: Mensaje de error
        """
        # Registrar error
        logger.error(f"[{self.name}] Error detectado: {error_message}")
        self.stats["errors"] += 1
        
        # Informar del error
        error_notification = self.generate_message(
            "error", 
            f"He detectado un error en las comunicaciones: {error_message[:50]}..."
        )
        self.broadcast_message(error_notification)
    
    def update_state(self):
        """Actualizar estado interno basado en métricas de comunicación."""
        # Simulación de variación de estado basado en actividad
        energy_variation = 0
        
        # Perder energía por mensajes enviados/recibidos
        energy_loss = (self.messages_sent + self.messages_received) * 0.001
        energy_variation -= energy_loss
        
        # Ganar energía por nuevas conexiones
        if self.stats["total_connections"] > 0:
            energy_variation += 0.1
        
        # Ajustar nivel basado en estadísticas
        level_adjustment = (
            self.stats["total_connections"] * 0.01 +
            (self.messages_sent + self.messages_received) * 0.0001 -
            self.stats["errors"] * 0.1
        )
        
        # Aplicar cambios
        self.adjust_energy(energy_variation)
        self.adjust_level(level_adjustment)
        
        # Actualizar emoción basada en estado de comunicaciones
        if not self.running:
            self.emotion = "Preocupación"
        elif self.stats["errors"] > 10:
            self.emotion = "Alerta"
        elif time.time() - self.last_heartbeat > self.heartbeat_interval * 2:
            self.emotion = "Inquietud"
        else:
            emotions = ["Fluido", "Comunicativo", "Vibrante", "Conectado"]
            self.emotion = random.choice(emotions)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado actual de la entidad para mostrar en UI.
        Extiende el método base con información específica de comunicaciones.
        
        Returns:
            Diccionario con estado
        """
        base_status = super().get_status()
        
        # Añadir métricas específicas de WebSocket
        ws_status = {
            "ws_scope": self.ws_scope,
            "server_status": self.server_status,
            "connected_clients": len(self.connected_clients),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "last_heartbeat": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.last_heartbeat)),
            "stats": self.stats,
            "specializations": self.specializations
        }
        
        # Combinar estados
        combined_status = {**base_status, **ws_status}
        return combined_status


class LocalWebSocketEntity(WebSocketEntity):
    """Entidad especializada en WebSocket local."""
    
    def __init__(self, name: str, father: str = "otoniel", frequency_seconds: int = 30):
        """
        Inicializar entidad WebSocket local.
        
        Args:
            name: Nombre de la entidad
            father: Nombre del creador/dueño
            frequency_seconds: Período de ciclo de vida en segundos
        """
        super().__init__(name, "WebSocket-Local", father, frequency_seconds, "local")
        
        # Especialización adicional para conexiones locales
        self.specializations["Local Optimization"] = 0.9
        self.specializations["Zero-latency Communication"] = 0.8
        
        logger.info(f"[{self.name}] Entidad WebSocket Local inicializada")


class ExternalWebSocketEntity(WebSocketEntity):
    """Entidad especializada en WebSocket externo/remoto."""
    
    def __init__(self, name: str, father: str = "otoniel", frequency_seconds: int = 35):
        """
        Inicializar entidad WebSocket externo.
        
        Args:
            name: Nombre de la entidad
            father: Nombre del creador/dueño
            frequency_seconds: Período de ciclo de vida en segundos
        """
        super().__init__(name, "WebSocket-External", father, frequency_seconds, "external")
        
        # Especialización adicional para conexiones externas
        self.specializations["Security Protocol"] = 0.9
        self.specializations["External Gateway"] = 0.8
        self.specializations["Load Balancing"] = 0.7
        
        # Configuración adicional para conexiones externas
        self.heartbeat_interval = 45  # segundos (más largo para externos)
        
        logger.info(f"[{self.name}] Entidad WebSocket Externo inicializada")
        
        # Capacidades adicionales para conexiones externas
        self.relay_servers = {}
        self.connection_regions = ["Americas", "Europe", "Asia", "Oceania"]
        
    async def setup_regional_relay(self, region):
        """
        Configurar relay para una región específica.
        
        Args:
            region: Nombre de la región
        """
        # Simulación de configuración de relay regional
        self.relay_servers[region] = {
            "status": "Activo",
            "latency": random.uniform(50, 200),  # ms
            "connections": random.randint(0, 20),
            "last_check": time.time()
        }
        
        logger.info(f"[{self.name}] Relay para región {region} configurado")
        
    def check_global_status(self):
        """
        Verificar estado global de relays.
        
        Returns:
            Estado de los relays
        """
        # Asegurar que tenemos relays configurados
        for region in self.connection_regions:
            if region not in self.relay_servers:
                # Configurar relay usando asyncio
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.setup_regional_relay(region))
                loop.close()
        
        # Calcular métricas globales
        total_connections = sum(r.get("connections", 0) for r in self.relay_servers.values())
        avg_latency = sum(r.get("latency", 0) for r in self.relay_servers.values()) / len(self.relay_servers)
        
        return {
            "relay_count": len(self.relay_servers),
            "regions": list(self.relay_servers.keys()),
            "total_connections": total_connections,
            "average_latency": avg_latency,
            "status": "Operative" if all(r.get("status") == "Activo" for r in self.relay_servers.values()) else "Degraded"
        }
    
    def get_status(self):
        """
        Obtener estado extendido incluyendo relays.
        
        Returns:
            Estado completo
        """
        status = super().get_status()
        status["global_relay"] = self.check_global_status()
        return status


def create_local_websocket_entity(name="Hermes", father="otoniel", frequency_seconds=30, start_server=True):
    """
    Crear y configurar una entidad WebSocket local.
    
    Args:
        name: Nombre de la entidad
        father: Nombre del creador/dueño
        frequency_seconds: Período de ciclo de vida en segundos
        start_server: Si es True, inicia el servidor WebSocket
        
    Returns:
        Instancia de LocalWebSocketEntity
    """
    entity = LocalWebSocketEntity(name, father, frequency_seconds)
    
    if start_server:
        entity.start_server_thread("localhost", 8765)
    
    return entity


def create_external_websocket_entity(name="Apollo", father="otoniel", frequency_seconds=35, start_server=True):
    """
    Crear y configurar una entidad WebSocket externa.
    
    Args:
        name: Nombre de la entidad
        father: Nombre del creador/dueño
        frequency_seconds: Período de ciclo de vida en segundos
        start_server: Si es True, inicia el servidor WebSocket
        
    Returns:
        Instancia de ExternalWebSocketEntity
    """
    entity = ExternalWebSocketEntity(name, father, frequency_seconds)
    
    if start_server:
        entity.start_server_thread("0.0.0.0", 8766)
    
    return entity


if __name__ == "__main__":
    # Prueba básica de las entidades
    hermes = create_local_websocket_entity(start_server=True)
    apollo = create_external_websocket_entity(start_server=True)
    
    print(f"Entidad local {hermes.name} creada con rol {hermes.role}")
    print(f"Entidad externa {apollo.name} creada con rol {apollo.role}")
    
    # Iniciar ciclos de vida en hilos separados
    threads = []
    for entity in [hermes, apollo]:
        thread = threading.Thread(target=entity.start_lifecycle)
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # Mantener vivos por un tiempo
    try:
        for i in range(30):
            time.sleep(1)
            if i % 5 == 0:
                for entity in [hermes, apollo]:
                    status = entity.get_status()
                    print(f"Estado de {entity.name}: Energía={status['energy']:.1f}, "
                          f"Nivel={status['level']:.1f}, Emoción={status['emotion']}, "
                          f"Clientes={status['connected_clients']}")
    
    except KeyboardInterrupt:
        print("Deteniendo prueba...")
    finally:
        # Detener ciclos de vida
        for entity in [hermes, apollo]:
            entity.stop_lifecycle()
            print(f"Entidad {entity.name} detenida")