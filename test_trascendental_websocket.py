"""
Script de prueba para el WebSocket Externo Trascendental.

Este script crea una instancia del WebSocket Externo Trascendental,
lo inicia, y luego ejecuta un cliente WebSocket para conectarse y 
verificar su funcionamiento.
"""

import asyncio
import json
import logging
import websockets
import random
from typing import Dict, Any, List

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestTrascendentalWS")

# Importar WebSocket Trascendental
from genesis.core.transcendental_external_websocket import TranscendentalExternalWebSocket

async def start_server(host: str = "localhost", port: int = 8765) -> TranscendentalExternalWebSocket:
    """
    Iniciar servidor WebSocket Trascendental.
    
    Args:
        host: Host para escuchar
        port: Puerto para escuchar
        
    Returns:
        Instancia del WebSocket Trascendental
    """
    ws_server = TranscendentalExternalWebSocket(host, port)
    
    # Iniciar servidor en tarea separada
    asyncio.create_task(ws_server.start())
    
    # Esperar a que el servidor esté listo
    logger.info(f"Iniciando servidor WebSocket Trascendental en {host}:{port}")
    await asyncio.sleep(1)
    
    return ws_server

async def test_client(uri: str, messages_count: int = 5, delay_ms: int = 100) -> List[Dict[str, Any]]:
    """
    Cliente de prueba para WebSocket Trascendental.
    
    Args:
        uri: URI del servidor WebSocket
        messages_count: Número de mensajes a enviar
        delay_ms: Retardo entre mensajes en milisegundos
        
    Returns:
        Lista de respuestas recibidas
    """
    logger.info(f"Conectando a {uri}...")
    responses = []
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("Conexión establecida")
            
            for i in range(messages_count):
                # Generar mensaje de prueba
                message = {
                    "operation": "test",
                    "sequence": i,
                    "data": {
                        "value": random.random(),
                        "timestamp": asyncio.get_event_loop().time()
                    }
                }
                
                # Enviar mensaje
                await websocket.send(json.dumps(message))
                logger.info(f"Mensaje enviado: {message}")
                
                # Recibir respuesta
                response = await websocket.recv()
                try:
                    response_data = json.loads(response)
                    logger.info(f"Respuesta recibida: {response_data}")
                    responses.append(response_data)
                except json.JSONDecodeError:
                    logger.error(f"Respuesta no es JSON válido: {response}")
                
                # Esperar entre mensajes
                await asyncio.sleep(delay_ms / 1000)
                
    except Exception as e:
        logger.error(f"Error en cliente WebSocket: {e}")
        
    return responses

async def main():
    """Función principal de prueba."""
    host = "localhost"
    port = 8765
    server = None
    
    try:
        logger.info("Iniciando prueba con tiempo limitado (10 segundos máximo)...")
        
        # Iniciar servidor con tiempo límite
        server_task = asyncio.create_task(
            asyncio.wait_for(
                start_server(host, port),
                timeout=5.0
            )
        )
        
        # Esperar a que el servidor esté listo o timeout
        try:
            server = await server_task
            logger.info("Servidor iniciado correctamente")
        except asyncio.TimeoutError:
            logger.error("Tiempo de espera agotado al iniciar servidor")
            return
            
        # Ejecutar cliente de prueba con tiempo límite
        client_task = asyncio.create_task(
            asyncio.wait_for(
                test_client(f"ws://{host}:{port}/test-component", messages_count=3, delay_ms=300),
                timeout=5.0
            )
        )
        
        # Esperar a que el cliente complete o timeout
        try:
            responses = await client_task
            logger.info(f"Cliente completó con {len(responses)} respuestas")
        except asyncio.TimeoutError:
            logger.error("Tiempo de espera agotado en cliente")
            
        # Si tenemos servidor, mostrar estadísticas y detenerlo
        if server:
            # Mostrar estadísticas básicas
            connections = server.stats.get("connections_accepted", 0)
            messages_recv = server.stats.get("messages_received", 0)
            messages_sent = server.stats.get("messages_sent", 0)
            
            logger.info(f"Conexiones: {connections}, Mensajes recibidos: {messages_recv}, Mensajes enviados: {messages_sent}")
            
            # Detener servidor con tiempo límite
            try:
                await asyncio.wait_for(server.stop(), timeout=2.0)
                logger.info("Servidor detenido correctamente")
            except asyncio.TimeoutError:
                logger.error("Tiempo de espera agotado al detener servidor")
                
    except Exception as e:
        logger.error(f"Error en prueba: {e}")
    finally:
        # Asegurar que todo termina
        logger.info("Prueba finalizada")

if __name__ == "__main__":
    asyncio.run(main())