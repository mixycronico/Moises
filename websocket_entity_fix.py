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
from enhanced_cosmic_entity_mixin import EnhancedCosmicEntityMixin

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
    
    Características de entidades avanzadas:
    - Auto-reparación: Pueden detectar y solucionar problemas automáticamente
    - Propuesta de mejoras: Pueden sugerir cambios al sistema
    - Evolución de funciones: Pueden desarrollar nuevas capacidades con el tiempo
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
        
        # Atributos específicos de WebSocket
        self.ws_scope = ws_scope
        self.server = None
        self.server_thread = None
        self.connected_clients = set()
        self.messages_received = 0
        self.messages_sent = 0
        self.server_status = "Initialized"
        self.last_heartbeat = time.time()
        self.is_server_running = False
        self.server_host = None
        self.server_port = None
        
        # Estadísticas y monitoreo
        self.stats = {
            "connections_total": 0,
            "messages_total": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0,
            "uptime_seconds": 0,
            "start_time": time.time()
        }
        
        # Especializaciones de comunicación
        self.specializations["Real-time Messaging"] = 0.85
        self.specializations["Network Resilience"] = 0.75
        self.specializations["Binary Protocols"] = 0.6
        
        # Estado emocional inicial
        self.emotion = "Expectante"
        
        logger.info(f"[{self.name}] Entidad WebSocket inicializada con ámbito {ws_scope}")
    
    async def server_handler(self, websocket, path):
        """
        Manejador principal del servidor WebSocket.
        
        Args:
            websocket: Conexión WebSocket del cliente
            path: Ruta de conexión
        """
        # Generar ID único para este cliente
        client_id = f"{id(websocket)}-{int(time.time())}"
        
        # Registrar nueva conexión
        self.connected_clients.add(websocket)
        self.stats["connections_total"] += 1
        logger.info(f"[{self.name}] Nuevo cliente conectado [{client_id}], total: {len(self.connected_clients)}")
        
        try:
            # Enviar mensaje de bienvenida
            welcome_msg = {
                "type": "welcome",
                "sender": self.name,
                "message": f"Bienvenido a {self.name} WebSocket Service",
                "timestamp": time.time()
            }
            await websocket.send(json.dumps(welcome_msg))
            
            # Loop principal para recibir mensajes
            async for message in websocket:
                self.messages_received += 1
                self.stats["messages_received"] += 1
                
                try:
                    # Parsear mensaje JSON
                    data = json.loads(message)
                    logger.debug(f"[{self.name}] Mensaje recibido de [{client_id}]: {data.get('type', 'unknown')}")
                    
                    # Procesar mensaje
                    await self.process_client_message(websocket, data)
                    
                except json.JSONDecodeError:
                    # Mensaje no es JSON válido
                    error_msg = {
                        "type": "error",
                        "sender": self.name,
                        "message": "Formato JSON inválido",
                        "timestamp": time.time()
                    }
                    await websocket.send(json.dumps(error_msg))
                    self.stats["errors"] += 1
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"[{self.name}] Cliente desconectado normalmente [{client_id}]")
        
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
            # Enviar estadísticas detalladas
            uptime = time.time() - self.stats["start_time"]
            self.stats["uptime_seconds"] = int(uptime)
            
            return {
                "type": "command_response",
                "sender": self.name,
                "command": command,
                "status": "success",
                "stats": self.stats,
                "timestamp": time.time()
            }
            
        elif command == "broadcast":
            # Broadcast administrativo
            message = params.get("message", "")
            if message:
                broadcast_msg = {
                    "type": "admin_broadcast",
                    "sender": "Admin",
                    "message": message,
                    "timestamp": time.time(),
                    "relayed_by": self.name
                }
                await self.broadcast_to_clients(broadcast_msg)
                
                return {
                    "type": "command_response",
                    "sender": self.name,
                    "command": command,
                    "status": "success",
                    "message": f"Mensaje enviado a {len(self.connected_clients)} clientes",
                    "timestamp": time.time()
                }
            else:
                return {
                    "type": "command_response",
                    "sender": self.name,
                    "command": command,
                    "status": "error",
                    "message": "Mensaje vacío",
                    "timestamp": time.time()
                }
                
        else:
            # Comando desconocido
            return {
                "type": "command_response",
                "sender": self.name,
                "command": command,
                "status": "error",
                "message": f"Comando '{command}' desconocido",
                "timestamp": time.time()
            }
    
    async def broadcast_to_clients(self, message):
        """
        Enviar mensaje a todos los clientes conectados.
        
        Args:
            message: Mensaje a enviar (debe ser dict)
        """
        if not isinstance(message, dict):
            logger.error(f"[{self.name}] Intento de broadcast con mensaje no-dict")
            return
        
        # Convertir a JSON
        json_message = json.dumps(message)
        
        # Crear tareas para enviar a cada cliente
        tasks = []
        for websocket in self.connected_clients:
            tasks.append(asyncio.create_task(self.safe_send(websocket, json_message)))
        
        # Esperar a que se completen todas las tareas
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def safe_send(self, websocket, message):
        """
        Enviar mensaje a un cliente de forma segura, manejando excepciones.
        
        Args:
            websocket: Conexión WebSocket del cliente
            message: Mensaje a enviar (string JSON)
        """
        try:
            await websocket.send(message)
        except Exception as e:
            logger.error(f"[{self.name}] Error enviando mensaje: {str(e)}")
            # No eliminamos el cliente aquí, se manejará en el loop principal
    
    def start_server_thread(self, host, port):
        """
        Iniciar servidor WebSocket en un hilo separado.
        
        Args:
            host: Host a escuchar
            port: Puerto a escuchar
        """
        self.server_host = host
        self.server_port = port
        self.server_thread = threading.Thread(target=self.run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        logger.info(f"[{self.name}] Servidor WebSocket iniciado en {host}:{port}")
    
    def run_server(self):
        """Ejecutar servidor WebSocket en loop de eventos asyncio."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        start_server = websockets.serve(
            self.server_handler, 
            self.server_host, 
            self.server_port
        )
        
        self.server = loop.run_until_complete(start_server)
        self.is_server_running = True
        self.server_status = "Running"
        
        # Ejecutar heartbeat en el mismo loop
        heartbeat_task = loop.create_task(self.heartbeat_loop())
        
        try:
            loop.run_forever()
        except Exception as e:
            logger.error(f"[{self.name}] Error en servidor WebSocket: {str(e)}")
        finally:
            self.is_server_running = False
            self.server_status = "Stopped"
            heartbeat_task.cancel()
            self.server.close()
            loop.run_until_complete(self.server.wait_closed())
            loop.close()
    
    async def heartbeat_loop(self):
        """Loop de heartbeat para mantener el servidor activo y realizar mantenimiento."""
        while self.is_server_running:
            try:
                self.last_heartbeat = time.time()
                
                # Enviar heartbeat a todos los clientes conectados
                heartbeat_msg = {
                    "type": "heartbeat",
                    "sender": self.name,
                    "timestamp": self.last_heartbeat
                }
                
                # Solo enviar si hay clientes
                if self.connected_clients:
                    await self.broadcast_to_clients(heartbeat_msg)
                
                # Auto-reparación
                self.check_health()
            except Exception as e:
                logger.error(f"[{self.name}] Error en heartbeat: {str(e)}")
            
            # Esperar para el próximo heartbeat
            await asyncio.sleep(30)  # 30 segundos
    
    def check_health(self):
        """
        Verificar salud del sistema y realizar auto-reparación.
        """
        # Si la energía es muy baja, descansa
        if self.energy < 20:
            self.log_state("Energía baja, descansando...")
            time.sleep(5)  # Pausa 5 segundos (reducido para no bloquear demasiado)
            self.energy += 10  # Recarga un poco
            self.log_state("Energía restaurada: " + str(self.energy))

        # Verificar estado del servidor
        if not self.is_server_running and self.is_alive:
            self.log_state("Servidor caído, reiniciando...")
            
            # Reintentar iniciar el servidor
            if self.server_host and self.server_port:
                self.start_server_thread(self.server_host, self.server_port)
    
    def auto_repair(self):
        """
        Realizar auto-reparación completa del sistema.
        """
        # Revisa y repara cada ciclo
        self.check_health()
        
        # Asegurar que las conexiones están bien
        if hasattr(self, "process_base_cycle"):
            self.process_base_cycle()
        
        self.log_state("Auto-reparación completa, estoy bien!")
    
    def start_lifecycle(self):
        """Iniciar ciclo de vida de la entidad."""
        self.is_alive = True
        self.lifecycle_thread = threading.Thread(target=self._lifecycle_loop)
        self.lifecycle_thread.daemon = True
        self.lifecycle_thread.start()
        
        logger.info(f"[{self.name}] Ciclo de vida iniciado")
        return self.lifecycle_thread
    
    # Añadir métodos necesarios para el correcto funcionamiento
    def adjust_energy(self):
        """
        Ajustar nivel de energía basado en actividad y estado del sistema.
        Implementa balance de energía adaptativo según el nivel de carga.
        """
        # Recuperación básica de energía
        if self.energy < 50:
            # Recuperación más rápida cuando está baja
            self.energy += 5
            logger.debug(f"[{self.name}] Recuperación rápida de energía: +5")
        elif self.energy < 80:
            # Recuperación normal
            self.energy += 2
            logger.debug(f"[{self.name}] Recuperación normal de energía: +2")
        
        # Límite superior
        if self.energy > 100:
            self.energy = 100
            
        # Factor de carga - más gasto si hay más clientes
        client_count = len(getattr(self, "connected_clients", []))
        energy_cost = 0.5 + (client_count * 0.2)
        
        # Aplicar costo pero evitar que baje de 10
        self.energy = max(10, self.energy - energy_cost)
        
        logger.debug(f"[{self.name}] Energía ajustada: {self.energy:.1f} (costo: {energy_cost:.1f})")
    
    def adjust_level(self):
        """
        Ajustar nivel de experiencia basado en actividad.
        Refleja el crecimiento y aprendizaje de la entidad.
        """
        # Factores que influyen en el nivel
        messages = self.stats["messages_total"]
        connections = self.stats["connections_total"]
        uptime = time.time() - self.stats["start_time"]
        
        # Calcular nuevo nivel basado en actividad
        base_level = 1.0
        message_factor = min(messages / 100, 5.0)  # Máximo +5 por mensajes
        connection_factor = min(connections / 10, 3.0)  # Máximo +3 por conexiones
        uptime_factor = min(uptime / 3600, 2.0)  # Máximo +2 por hora de actividad
        
        new_level = base_level + message_factor + connection_factor + uptime_factor
        
        # Aplicar cambio de nivel gradualmente
        if new_level > self.level:
            level_increase = min(0.1, new_level - self.level)  # Máximo 0.1 por ciclo
            self.level += level_increase
            if level_increase > 0.05:
                # Notificar solo cambios significativos
                logger.info(f"[{self.name}] Nivel incrementado: {self.level:.2f}")
                
                # Al subir de nivel, proponer una mejora
                if int(self.level) > int(self.level - level_increase):
                    self.propose_improvement()
    
    def propose_improvement(self):
        """
        Proponer una mejora al sistema basada en el nivel y especialización.
        Las entidades pueden sugerir cambios que mejorarían su funcionamiento.
        """
        # Determinar tipo de mejora según especialización
        level_int = int(self.level)
        improvement_type = "general"
        
        # Tipo de mejora según características
        if self.ws_scope == "local":
            improvement_type = "optimización interna"
        elif self.ws_scope == "external":
            improvement_type = "conectividad externa"
        
        # Ideas de mejora según nivel
        improvements = {
            1: "Implementar un sistema de caché simple para mensajes frecuentes",
            2: "Añadir compresión básica en mensajes grandes para reducir tráfico",
            3: "Implementar reconexión automática para clientes interrumpidos",
            4: "Crear un sistema de prioridad en la cola de mensajes",
            5: "Añadir encriptación de extremo a extremo para mensajes sensibles",
            6: "Desarrollar un mecanismo de balanceo de carga entre instancias",
            7: "Implementar un sistema de fragmentación para mensajes muy grandes",
            8: "Crear un modo de emergencia con funcionalidad reducida para alta demanda",
            9: "Añadir capacidad de migración en caliente entre servidores",
            10: "Desarrollar un sistema predictivo de comportamiento de usuarios"
        }
        
        # Seleccionar mejora según nivel (o la última disponible)
        improvement = improvements.get(min(level_int, max(improvements.keys())), 
                                      "Optimización avanzada del sistema")
        
        # Registrar la propuesta
        logger.info(f"[{self.name}] PROPUESTA DE MEJORA (Nivel {level_int}): {improvement} para {improvement_type}")
        
        # Si tiene capacidad de aprendizaje avanzado, generar nueva función
        if level_int >= 3:
            self.generate_new_function()
    
    def generate_new_function(self):
        """
        Generar nueva funcionalidad basada en el nivel y especialización.
        Las entidades más avanzadas pueden desarrollar capacidades emergentes.
        """
        level_int = int(self.level)
        
        # Definir posibles nuevas funciones según especialización
        if self.ws_scope == "local":
            functions = {
                3: "monitor_interno",
                4: "cache_inteligente", 
                5: "compresion_adaptativa",
                6: "enrutamiento_dinamico",
                7: "respuesta_predictiva"
            }
        else:  # external
            functions = {
                3: "detector_anomalias",
                4: "balanceo_regional", 
                5: "conexion_resiliente",
                6: "proxy_inteligente",
                7: "filtrado_avanzado"
            }
        
        # Seleccionar función a desarrollar
        function_name = functions.get(min(level_int, max(functions.keys())), "funcion_avanzada")
        
        # Determinar costo y tiempo de desarrollo
        energy_cost = 15
        dev_cycles = 3
        
        # Verificar si tiene suficiente energía
        if self.energy >= energy_cost:
            self.energy -= energy_cost
            logger.info(f"[{self.name}] INICIANDO DESARROLLO: Nueva función '{function_name}' (costo: {energy_cost} energía)")
            logger.info(f"[{self.name}] Estimado {dev_cycles} ciclos para completar desarrollo")
            
            # Simular desarrollo (en la vida real sería un proceso más complejo)
            self.development_queue = self.development_queue if hasattr(self, "development_queue") else []
            self.development_queue.append({
                "function": function_name,
                "cycles_left": dev_cycles,
                "progress": 0
            })
        else:
            logger.info(f"[{self.name}] FUNCIÓN PROPUESTA: '{function_name}' (energía insuficiente: {self.energy}/{energy_cost})")
    
    def process_developments(self):
        """Procesar desarrollos en curso de nuevas funcionalidades."""
        if not hasattr(self, "development_queue") or not self.development_queue:
            return
        
        # Avanzar el desarrollo de cada función en cola
        updated_queue = []
        for dev in self.development_queue:
            # Actualizar progreso
            dev["cycles_left"] -= 1
            dev["progress"] += random.uniform(20, 40)  # Progreso aleatorio entre 20-40%
            
            if dev["cycles_left"] <= 0 or dev["progress"] >= 100:
                # Desarrollo completado
                logger.info(f"[{self.name}] DESARROLLO COMPLETADO: Función '{dev['function']}' lista para usar!")
                # Aquí se implementaría realmente la nueva funcionalidad
                self.add_capability(dev["function"])
            else:
                # Continuar desarrollo
                updated_queue.append(dev)
                logger.debug(f"[{self.name}] Desarrollo en progreso: '{dev['function']}' - {dev['progress']:.1f}% completado")
        
        self.development_queue = updated_queue
    
    def add_capability(self, capability_name):
        """
        Añadir nueva capacidad a la entidad.
        
        Args:
            capability_name: Nombre de la capacidad
        """
        # Registrar nueva capacidad
        if not hasattr(self, "capabilities"):
            self.capabilities = []
        
        self.capabilities.append(capability_name)
        
        # Añadir especialización relacionada
        self.specializations[capability_name] = 0.5  # Comienza con nivel medio
        
        logger.info(f"[{self.name}] Nueva capacidad añadida: {capability_name}")
        
        # Anunciar nueva capacidad a la red si es posible
        if hasattr(self, "broadcast_message"):
            try:
                self.broadcast_message("new_capability", {
                    "entity": self.name,
                    "capability": capability_name,
                    "level": self.level,
                    "timestamp": time.time()
                })
            except Exception as e:
                logger.error(f"[{self.name}] Error anunciando capacidad: {str(e)}")
    
    def _lifecycle_loop(self):
        """Loop principal de ciclo de vida."""
        while self.is_alive:
            try:
                # Metabolismo básico
                self.metabolize()
                
                # Evolución
                self.evolve()
                
                # Procesar desarrollos en curso
                self.process_developments()
                
                # Procesar mensaje/comando según tipo
                if hasattr(self, "dominant_trait"):
                    if self.dominant_trait == "Adaptive":
                        # Mayor probabilidad de transmitir estado
                        if random.random() < 0.6:
                            if hasattr(self, "broadcast_message"):
                                self.broadcast_message("status_update", 
                                                   {"entity": self.name, 
                                                    "status": self.get_status()})
                    
                    elif self.dominant_trait == "Analytical":
                        # Mayor probabilidad de enviar estadísticas
                        if random.random() < 0.4:
                            if hasattr(self, "broadcast_message"):
                                self.broadcast_message("stats", 
                                                   {"entity": self.name, 
                                                    "uptime": self.stats["uptime_seconds"],
                                                    "messages": self.stats["messages_total"]})
                
                # Auto-reparación
                self.auto_repair()
                
                # Ajustes de rendimiento y balance energético
                self.adjust_energy()
                self.adjust_level()
                
                time.sleep(self.frequency_seconds)  # Periodo de ciclo
                
            except Exception as e:
                logger.error(f"[{self.name}] Error en ciclo de vida: {str(e)}")
                time.sleep(5)  # Esperar un poco antes de reintentar
    
    def stop_lifecycle(self):
        """Detener ciclo de vida."""
        self.is_alive = False
        
        # Esperar a que termine
        if hasattr(self, "lifecycle_thread") and self.lifecycle_thread:
            self.lifecycle_thread.join(timeout=5)
        
        logger.info(f"[{self.name}] Ciclo de vida detenido")
    
    def log_state(self, message):
        """
        Registrar estado en log.
        
        Args:
            message: Mensaje de estado
        """
        logger.info(f"[{self.name}] {message}")
        
        # Registrar en DB si está disponible
        if hasattr(self, "network") and self.network:
            if hasattr(self.network, "log_message"):
                self.network.log_message(self.name, message)
    
    def get_status(self):
        """
        Obtener estado actual de la entidad.
        
        Returns:
            Dict con estado
        """
        # Estado base (heredado)
        base_status = super().get_status()
        
        # Estado de WebSocket
        ws_status = {
            "server_status": self.server_status,
            "connected_clients": len(self.connected_clients),
            "messages_received": self.messages_received,
            "messages_sent": self.messages_sent,
            "last_heartbeat": self.last_heartbeat,
            "uptime": int(time.time() - self.stats["start_time"]),
            "endpoint": f"{self.server_host}:{self.server_port}" if self.server_host and self.server_port else None
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
        
        # Inicializar diccionario de especializaciones si no existe
        if not hasattr(self, "specializations"):
            self.specializations = {}
            
        # Especialización adicional para conexiones locales
        self.specializations["Local Optimization"] = 0.9
        self.specializations["Zero-latency Communication"] = 0.8
        
        logger.info(f"[{self.name}] Entidad WebSocket Local inicializada")
    
    def trade(self):
        """
        Implementación del método abstracto trade para entidad WebSocket local.
        En lugar de ejecutar operaciones de trading, esta entidad gestiona las
        comunicaciones WebSocket internas del sistema.
        
        Returns:
            Dict con información sobre el estado de comunicación
        """
        # Crear respuesta estándar
        response = {
            "entity": self.name,
            "role": self.role,
            "action": "local_communication_relay",
            "timestamp": time.time(),
            "clients_connected": len(self.connected_clients),
            "messages_processed": self.messages_received,
            "messages_sent": self.messages_sent,
            "status": self.server_status
        }
        
        # Añadir mensaje al log si está disponible
        if hasattr(self, "network") and self.network:
            if hasattr(self.network, "log_message"):
                self.network.log_message(
                    self.name, 
                    f"Relay local activo: {response['clients_connected']} clientes"
                )
        
        # Registrar actividad
        logger.debug(f"[{self.name}] Actividad de comunicación local: {response['clients_connected']} clientes conectados")
        
        return response


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
        
        # Inicializar diccionario de especializaciones si no existe
        if not hasattr(self, "specializations"):
            self.specializations = {}
            
        # Especialización adicional para conexiones externas
        self.specializations["Security Protocol"] = 0.9
        self.specializations["External Gateway"] = 0.8
        self.specializations["Load Balancing"] = 0.7
        
        # Configuración adicional para conexiones externas
        self.heartbeat_interval = 45  # segundos (más largo para externos)
        
        # Propiedades específicas para conexiones externas
        self.relay_servers = {}
        self.connection_regions = ["Americas", "Europe", "Asia", "Oceania"]
        
        logger.info(f"[{self.name}] Entidad WebSocket Externo inicializada")
    
    def trade(self):
        """
        Implementación del método abstracto trade para entidad WebSocket externa.
        En lugar de ejecutar operaciones de trading, esta entidad gestiona las
        comunicaciones WebSocket externas del sistema y la conexión con servicios remotos.
        
        Returns:
            Dict con información sobre el estado de comunicación externa
        """
        # Crear respuesta estándar
        response = {
            "entity": self.name,
            "role": self.role,
            "action": "external_communication_relay",
            "timestamp": time.time(),
            "clients_connected": len(self.connected_clients),
            "messages_processed": self.messages_received,
            "messages_sent": self.messages_sent,
            "status": self.server_status,
            "regions_active": len(self.relay_servers)
        }
        
        # Añadir información de relays regionales si está disponible
        if hasattr(self, "check_global_status"):
            response["global_status"] = self.check_global_status()
        
        # Añadir mensaje al log si está disponible
        if hasattr(self, "network") and self.network:
            if hasattr(self.network, "log_message"):
                self.network.log_message(
                    self.name, 
                    f"Relay externo activo: {response['clients_connected']} clientes, {response['regions_active']} regiones"
                )
        
        # Registrar actividad
        logger.debug(f"[{self.name}] Actividad de comunicación externa: {response['clients_connected']} clientes conectados")
        
        return response
    
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