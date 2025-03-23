"""
Prueba del WebSocket Externo Trascendental y su integración con el sistema core Genesis.

Este script crea un entorno de prueba completo para demostrar las capacidades 
del WebSocket Externo Trascendental, incluyendo:
- Integración con el sistema core
- Resiliencia ante fallos de conexión
- Transmutación de errores en energía
- Operación bajo carga extrema
- Recuperación desde memoria omniversal
"""

import asyncio
import json
import logging
import time
import random
import websockets
import aiohttp
from typing import Dict, Any, List, Optional
import sys
import os

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("test_external_ws_trascendental.log")
    ]
)

logger = logging.getLogger("Test.ExternalWS")

# Importar componentes necesarios
sys.path.insert(0, os.path.abspath('.'))
from genesis.core.transcendental_external_websocket import TranscendentalExternalWebSocket
from genesis.core.transcendental_ws_adapter import TranscendentalWebSocketAdapter
from genesis.core.genesis_hybrid_optimized import GenesisHybridCoordinator, ComponentAPI, TestComponent

# Importar mecanismos trascendentales para simulación
from genesis_singularity_transcendental_v4 import (
    DimensionalCollapseV4, 
    EventHorizonV4,
    QuantumTimeV4,
    InfiniteDensityV4,
    OmniversalSharedMemory
)

class SimulatedExternalClient:
    """Cliente simulado para pruebas del WebSocket externo."""
    
    def __init__(self, id: str, uri: str = "ws://localhost:8080/ws"):
        """
        Inicializar cliente simulado.
        
        Args:
            id: Identificador del cliente
            uri: URI del WebSocket
        """
        self.id = id
        self.base_uri = uri
        self.uri = f"{uri}?id={id}"
        self.ws = None
        self.connected = False
        self.logger = logging.getLogger(f"Client.{id}")
        
    async def connect(self) -> bool:
        """
        Conectar al servidor WebSocket.
        
        Returns:
            True si la conexión fue exitosa, False en caso contrario
        """
        try:
            self.ws = await websockets.connect(self.uri)
            self.connected = True
            self.logger.info(f"Cliente {self.id} conectado")
            return True
        except Exception as e:
            self.logger.error(f"Error conectando cliente {self.id}: {str(e)}")
            self.connected = False
            return False
    
    async def send_message(self, message_type: str, data: Dict[str, Any]) -> bool:
        """
        Enviar mensaje al servidor.
        
        Args:
            message_type: Tipo de mensaje
            data: Datos del mensaje
            
        Returns:
            True si se envió correctamente, False en caso contrario
        """
        if not self.connected or not self.ws:
            self.logger.error(f"Cliente {self.id} no está conectado")
            return False
            
        try:
            # Construir mensaje
            message = {
                "type": message_type,
                "data": data,
                "source": self.id,
                "timestamp": time.time()
            }
            
            # Enviar al servidor
            await self.ws.send(json.dumps(message))
            self.logger.info(f"Cliente {self.id} envió mensaje: {message_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error enviando mensaje desde {self.id}: {str(e)}")
            self.connected = False
            return False
    
    async def receive_message(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """
        Recibir mensaje del servidor.
        
        Args:
            timeout: Tiempo máximo de espera en segundos
            
        Returns:
            Mensaje recibido o None si hubo error
        """
        if not self.connected or not self.ws:
            self.logger.error(f"Cliente {self.id} no está conectado")
            return None
            
        try:
            # Esperar mensaje con timeout
            message = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
            data = json.loads(message)
            self.logger.info(f"Cliente {self.id} recibió: {data.get('type')}")
            return data
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout esperando mensaje para {self.id}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error recibiendo mensaje para {self.id}: {str(e)}")
            self.connected = False
            return None
    
    async def close(self) -> None:
        """Cerrar conexión WebSocket."""
        if self.ws:
            await self.ws.close()
            self.connected = False
            self.logger.info(f"Cliente {self.id} desconectado")

class TestServer:
    """Servidor de prueba para el WebSocket externo trascendental."""
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        """
        Inicializar servidor de prueba.
        
        Args:
            host: Dirección del servidor
            port: Puerto del servidor
        """
        self.host = host
        self.port = port
        self.app = None
        self.runner = None
        self.site = None
        self.coordinator = None
        self.ws_adapter = None
        self.logger = logging.getLogger("TestServer")
        
    async def start(self) -> None:
        """Iniciar servidor de prueba."""
        # Crear componentes de prueba
        component1 = TestComponent("component1")
        component2 = TestComponent("component2")
        
        # Crear coordinador híbrido
        self.coordinator = GenesisHybridCoordinator()
        
        # Registrar componentes
        self.coordinator.register_component("component1", component1)
        self.coordinator.register_component("component2", component2)
        
        # Iniciar coordinador y servidor
        await self.coordinator.start()
        
        # Crear adaptador para WebSocket trascendental
        self.ws_adapter = TranscendentalWebSocketAdapter(self.coordinator)
        
        # Conectar WebSocket trascendental al sistema
        await self.ws_adapter.connect_to_core()
        
        self.logger.info(f"Servidor iniciado en {self.host}:{self.port}")
        
    async def stop(self) -> None:
        """Detener servidor de prueba."""
        # Desconectar adaptador
        if self.ws_adapter:
            await self.ws_adapter.disconnect_from_core()
        
        # Detener coordinador
        if self.coordinator:
            await self.coordinator.stop()
            
        self.logger.info("Servidor detenido")

async def test_connection():
    """Prueba de conexión básica."""
    logger.info("=== PRUEBA DE CONEXIÓN BÁSICA ===")
    
    # Iniciar servidor
    server = TestServer()
    await server.start()
    
    try:
        # Crear cliente y conectar
        client = SimulatedExternalClient("test_client")
        connected = await client.connect()
        
        assert connected, "La conexión debería ser exitosa"
        logger.info("Conexión básica exitosa")
        
    finally:
        # Limpiar
        await asyncio.sleep(0.5)  # Dar tiempo para procesar mensajes
        await client.close()
        await server.stop()

async def test_message_exchange():
    """Prueba de intercambio de mensajes."""
    logger.info("=== PRUEBA DE INTERCAMBIO DE MENSAJES ===")
    
    # Iniciar servidor
    server = TestServer()
    await server.start()
    
    try:
        # Crear cliente y conectar
        client = SimulatedExternalClient("component1")  # Usar ID de componente registrado
        connected = await client.connect()
        assert connected, "La conexión debería ser exitosa"
        
        # Enviar mensaje
        sent = await client.send_message("test_event", {"content": "Test content", "value": 42})
        assert sent, "El mensaje debería enviarse correctamente"
        
        # Esperar procesamiento
        await asyncio.sleep(0.5)
        
        # Verificar estadísticas
        stats = server.ws_adapter.get_stats()
        logger.info(f"Estadísticas: {stats}")
        
        assert stats["messages_received"] > 0, "Debería haber mensajes recibidos"
        
        logger.info("Intercambio de mensajes exitoso")
        
    finally:
        # Limpiar
        await asyncio.sleep(0.5)
        await client.close()
        await server.stop()

async def test_error_transmutation():
    """Prueba de transmutación de errores."""
    logger.info("=== PRUEBA DE TRANSMUTACIÓN DE ERRORES ===")
    
    # Iniciar servidor
    server = TestServer()
    await server.start()
    
    try:
        # Crear cliente con ID inválido para provocar error
        client = SimulatedExternalClient("invalid_component")
        
        # Intentar conectar (debería fallar pero ser transmutado)
        connected = await client.connect()
        
        # El cliente local cree que está conectado porque el servidor acepta la conexión
        # antes de validar el ID y luego la cierra con código 1008
        
        # Esperar a que se procese la desconexión
        await asyncio.sleep(1)
        
        # Verificar estadísticas
        stats = server.ws_adapter.get_stats()
        logger.info(f"Estadísticas después de error: {stats}")
        
        # Debería haber al menos un error transmutado
        assert stats["errors_transmuted"] > 0, "Debería haber errores transmutados"
        
        logger.info("Transmutación de errores exitosa")
        
    finally:
        # Limpiar
        await asyncio.sleep(0.5)
        await client.close()
        await server.stop()

async def test_extreme_load():
    """Prueba de carga extrema."""
    logger.info("=== PRUEBA DE CARGA EXTREMA ===")
    
    # Iniciar servidor
    server = TestServer()
    await server.start()
    
    # Crear múltiples clientes
    num_clients = 10
    num_messages = 50
    clients = []
    
    try:
        # Crear y conectar clientes
        for i in range(num_clients):
            client = SimulatedExternalClient(f"component{i % 2 + 1}")  # Alternar entre component1 y component2
            connected = await client.connect()
            if connected:
                clients.append(client)
        
        logger.info(f"Conectados {len(clients)} de {num_clients} clientes")
        assert len(clients) > 0, "Al menos un cliente debería conectarse"
        
        # Enviar múltiples mensajes en paralelo
        start_time = time.time()
        
        # Usar QuantumTimeV4 para acelerar el proceso fuera del tiempo lineal
        quantum_time = QuantumTimeV4()
        
        async with quantum_time.nullify_time():
            # Crear tareas para enviar mensajes
            tasks = []
            for client in clients:
                for i in range(num_messages):
                    tasks.append(client.send_message(
                        "stress_test", 
                        {"iteration": i, "random_data": random.random(), "intensity": 1000.0}
                    ))
            
            # Ejecutar en paralelo
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calcular estadísticas
        elapsed_time = time.time() - start_time
        total_messages = len(results)
        successful_messages = sum(1 for r in results if r is True)
        
        # Verificar servidor
        server_stats = server.ws_adapter.get_stats()
        
        logger.info(f"Tiempo total: {elapsed_time:.4f}s")
        logger.info(f"Mensajes enviados: {total_messages}")
        logger.info(f"Mensajes exitosos: {successful_messages}")
        logger.info(f"Tasa de éxito: {successful_messages/total_messages*100:.2f}%")
        logger.info(f"Mensajes por segundo: {successful_messages/elapsed_time:.2f}")
        logger.info(f"Estadísticas del servidor: {server_stats}")
        
        assert successful_messages > 0, "Deberían enviarse mensajes exitosamente"
        assert server_stats["messages_received"] > 0, "El servidor debería recibir mensajes"
        
        logger.info("Prueba de carga extrema completada con éxito")
        
    finally:
        # Limpiar
        await asyncio.sleep(1)
        for client in clients:
            await client.close()
        await server.stop()

async def test_omniversal_memory_recovery():
    """Prueba de recuperación desde memoria omniversal."""
    logger.info("=== PRUEBA DE RECUPERACIÓN DESDE MEMORIA OMNIVERSAL ===")
    
    # Iniciar servidor
    server = TestServer()
    await server.start()
    
    # Obtener instancia de memoria omniversal
    memory = OmniversalSharedMemory()
    
    try:
        # Crear cliente y conectar
        client = SimulatedExternalClient("component1")
        connected = await client.connect()
        assert connected, "La conexión debería ser exitosa"
        
        # Enviar mensaje para almacenar en memoria omniversal
        test_data = {
            "unique_id": f"test_{int(time.time())}",
            "value": random.random(),
            "critical_info": "Datos importantes para recuperación"
        }
        
        sent = await client.send_message("store_critical", test_data)
        assert sent, "El mensaje debería enviarse correctamente"
        
        # Esperar a que se procese
        await asyncio.sleep(0.5)
        
        # Simular desconexión
        await client.close()
        
        # Recuperar datos desde memoria omniversal
        recovery_key = {"component_id": "component1"}
        stored_state = await memory.retrieve_state(recovery_key)
        
        logger.info(f"Datos recuperados desde memoria omniversal: {stored_state}")
        
        # Debería haber algún estado almacenado
        assert stored_state is not None, "Debería haber estado en memoria omniversal"
        
        logger.info("Recuperación desde memoria omniversal exitosa")
        
    finally:
        # Limpiar
        await asyncio.sleep(0.5)
        await server.stop()

async def main():
    """Función principal de prueba."""
    logger.info("INICIANDO PRUEBAS DEL WEBSOCKET EXTERNO TRASCENDENTAL")
    
    try:
        # Ejecutar pruebas
        await test_connection()
        await test_message_exchange()
        await test_error_transmutation()
        await test_extreme_load()
        await test_omniversal_memory_recovery()
        
        logger.info("TODAS LAS PRUEBAS COMPLETADAS CON ÉXITO")
        
    except Exception as e:
        logger.error(f"ERROR EN PRUEBAS: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())