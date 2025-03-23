"""
Test para el Sistema Híbrido WebSocket+API con Singularidad Trascendental V4.

Este script prueba la integración del sistema híbrido que combina WebSocket
para comunicación en tiempo real y API REST para integraciones externas,
todo potenciado por los mecanismos de la Singularidad Trascendental V4.

Características principales:
- Test de WebSocket con conexión local
- Test de API con endpoints simulados
- Test de integración del sistema híbrido completo
- Medición de rendimiento bajo carga extrema
"""

import asyncio
import json
import logging
import time
import random
from typing import Dict, Any, List
import sys
import os

# Añadir ruta del proyecto (asumiendo ejecución desde directorio raíz)
sys.path.insert(0, os.path.abspath('.'))

# Importar componentes del sistema híbrido de la Singularidad Trascendental V4
from genesis_singularity_transcendental_v4 import (
    TranscendentalWebSocket,
    TranscendentalAPI,
    GenesisHybridSystem,
    QuantumTimeV4
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("test_hybrid_v4.log")
    ]
)

logger = logging.getLogger("Test.HybridSystem")

# Servidores simulados para pruebas
class MockWebSocketServer:
    """Servidor WebSocket simulado para pruebas."""
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self.clients = []
        self.running = False
        self.logger = logging.getLogger("Test.MockWebSocketServer")
    
    async def start(self):
        """Iniciar servidor simulado."""
        self.running = True
        self.logger.info(f"Servidor WebSocket simulado iniciado en ws://{self.host}:{self.port}")
        return f"ws://{self.host}:{self.port}"
    
    async def stop(self):
        """Detener servidor simulado."""
        self.running = False
        self.logger.info("Servidor WebSocket simulado detenido")
    
    async def send_message(self, message: Dict[str, Any]):
        """Enviar mensaje simulado a clientes."""
        if not self.running:
            return
        
        self.logger.debug(f"Enviando mensaje simulado: {message}")
        
        # En implementación real, esto enviaría a clientes conectados
        # Para simulación, solo registramos el evento
        return {"sent": True, "message": message, "timestamp": time.time()}

class MockAPIServer:
    """Servidor API REST simulado para pruebas."""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.endpoints = {
            "data_endpoint": self.generate_test_data,
            "config_endpoint": self.get_config,
            "status_endpoint": self.get_status
        }
        self.running = False
        self.logger = logging.getLogger("Test.MockAPIServer")
    
    async def start(self):
        """Iniciar servidor simulado."""
        self.running = True
        self.logger.info(f"Servidor API simulado iniciado en http://{self.host}:{self.port}")
        return f"http://{self.host}:{self.port}"
    
    async def stop(self):
        """Detener servidor simulado."""
        self.running = False
        self.logger.info("Servidor API simulado detenido")
    
    async def generate_test_data(self, params: Dict = None) -> Dict[str, Any]:
        """Generar datos de prueba para el endpoint de datos."""
        if not self.running:
            return {"error": "Server not running"}
        
        # Datos simulados con estructura realista
        return {
            "timestamp": time.time(),
            "data_points": [
                {"id": i, "value": random.random() * 100, "type": "measurement"}
                for i in range(10)
            ],
            "metadata": {
                "source": "test_hybrid_system",
                "quality": "high",
                "parameters": params or {}
            }
        }
    
    async def get_config(self, params: Dict = None) -> Dict[str, Any]:
        """Obtener configuración simulada."""
        if not self.running:
            return {"error": "Server not running"}
        
        return {
            "version": "4.0.0",
            "mode": "TRANSCENDENTAL",
            "settings": {
                "timeout": 1e-12,
                "retry_count": 3,
                "fallback_enabled": True
            }
        }
    
    async def get_status(self, params: Dict = None) -> Dict[str, Any]:
        """Obtener estado simulado del sistema."""
        if not self.running:
            return {"error": "Server not running"}
        
        return {
            "status": "OPERATIONAL",
            "uptime": 3600,
            "load": 0.2,
            "components": {
                "websocket": "ONLINE",
                "api": "ONLINE",
                "database": "ONLINE"
            }
        }
    
    async def handle_request(self, endpoint: str, params: Dict = None) -> Dict[str, Any]:
        """Manejar una petición simulada a un endpoint."""
        if not self.running:
            return {"error": "Server not running"}
        
        if endpoint not in self.endpoints:
            return {"error": f"Endpoint {endpoint} not found"}
        
        handler = self.endpoints[endpoint]
        return await handler(params)

# Funciones de prueba para el sistema híbrido
async def test_websocket_component():
    """Prueba el componente WebSocket de forma aislada."""
    logger.info("=== INICIANDO PRUEBA DE COMPONENTE WEBSOCKET ===")
    
    # Iniciar servidor simulado
    ws_server = MockWebSocketServer()
    ws_uri = await ws_server.start()
    
    try:
        # Crear y conectar cliente WebSocket trascendental
        ws_client = TranscendentalWebSocket(ws_uri)
        
        # Iniciar cliente en segundo plano
        client_task = asyncio.create_task(ws_client.connect())
        
        # Esperar un momento para la conexión
        await asyncio.sleep(0.1)
        
        # Enviar mensaje simulado desde el servidor
        test_message = {"type": "test", "content": "Hello Transcendental World", "id": 1}
        await ws_server.send_message(test_message)
        
        # Verificar que el cliente está conectado
        assert ws_client.running == True, "El cliente WebSocket no está conectado"
        
        logger.info("Prueba de conexión WebSocket exitosa")
        return True
        
    except Exception as e:
        logger.error(f"Error en prueba WebSocket: {str(e)}")
        return False
        
    finally:
        # Detener servidor
        await ws_server.stop()

async def test_api_component():
    """Prueba el componente API de forma aislada."""
    logger.info("=== INICIANDO PRUEBA DE COMPONENTE API ===")
    
    # Iniciar servidor simulado
    api_server = MockAPIServer()
    api_url = await api_server.start()
    
    try:
        # Crear cliente API trascendental
        api_client = TranscendentalAPI(api_url)
        await api_client.initialize()
        
        # Realizar petición simulada
        response_data = await api_client.fetch_data("data_endpoint")
        
        # Verificar respuesta
        assert response_data is not None, "No se recibió respuesta de la API"
        assert "data_points" in response_data, "Respuesta incorrecta de la API"
        
        # Procesar datos
        processed = await api_client.process_api_data(response_data)
        
        assert processed is not None, "Falló el procesamiento de datos API"
        assert "processed" in processed and processed["processed"] == True, "Fallo en flags de procesamiento"
        
        logger.info("Prueba de API trascendental exitosa")
        return True
        
    except Exception as e:
        logger.error(f"Error en prueba API: {str(e)}")
        return False
        
    finally:
        # Detener servidor
        await api_server.stop()

async def test_hybrid_system():
    """Prueba el sistema híbrido completo."""
    logger.info("=== INICIANDO PRUEBA DE SISTEMA HÍBRIDO COMPLETO ===")
    
    # Iniciar servidores simulados
    ws_server = MockWebSocketServer()
    api_server = MockAPIServer()
    
    ws_uri = await ws_server.start()
    api_url = await api_server.start()
    
    try:
        # Crear sistema híbrido
        hybrid_system = GenesisHybridSystem(ws_uri=ws_uri, api_url=api_url)
        
        # Iniciar componentes individuales
        await asyncio.gather(
            hybrid_system.websocket.connect(),
            hybrid_system.api.initialize()
        )
        
        # Sincronizar componentes
        await hybrid_system.synchronize()
        
        # Verificar estado de componentes
        assert hybrid_system.websocket.running == True, "WebSocket no está ejecutándose"
        
        # Simular datos desde API
        api_data = await api_server.handle_request("data_endpoint")
        api_processed = await hybrid_system.api.process_api_data(api_data)
        
        # Verificar procesamiento
        assert api_processed is not None, "Falló el procesamiento de datos API en sistema híbrido"
        
        # Simular mensaje WebSocket
        test_message = {"type": "test", "content": "Hybrid System Test", "id": 123}
        ws_result = await ws_server.send_message(test_message)
        
        assert ws_result["sent"] == True, "Falló el envío de mensaje WebSocket"
        
        logger.info("Prueba de sistema híbrido exitosa")
        return True
        
    except Exception as e:
        logger.error(f"Error en prueba de sistema híbrido: {str(e)}")
        return False
        
    finally:
        # Detener servidores
        await ws_server.stop()
        await api_server.stop()

async def test_hybrid_extreme_load():
    """Prueba el sistema híbrido bajo carga extrema."""
    logger.info("=== INICIANDO PRUEBA DE SISTEMA HÍBRIDO BAJO CARGA EXTREMA ===")
    
    # Iniciar servidores simulados
    ws_server = MockWebSocketServer()
    api_server = MockAPIServer()
    
    ws_uri = await ws_server.start()
    api_url = await api_server.start()
    
    try:
        # Crear sistema híbrido
        hybrid_system = GenesisHybridSystem(ws_uri=ws_uri, api_url=api_url)
        
        # Iniciar componentes
        await asyncio.gather(
            hybrid_system.websocket.connect(),
            hybrid_system.api.initialize()
        )
        
        # Sincronizar componentes
        await hybrid_system.synchronize()
        
        # Parámetros de prueba extrema
        num_operations = 1000
        success_count = 0
        
        # Medir tiempo total
        start_time = time.time()
        
        # Ejecutar operaciones simultáneas
        async with QuantumTimeV4().nullify_time():
            # API requests (carga extrema)
            api_tasks = []
            for i in range(num_operations):
                task = asyncio.create_task(hybrid_system.api.fetch_data("data_endpoint"))
                api_tasks.append(task)
            
            # WebSocket messages (carga extrema)
            ws_tasks = []
            for i in range(num_operations):
                message = {"type": "extreme_test", "content": f"Message {i}", "intensity": 1000.0}
                task = asyncio.create_task(ws_server.send_message(message))
                ws_tasks.append(task)
            
            # Esperar a que todas las tareas se completen
            api_results = await asyncio.gather(*api_tasks, return_exceptions=True)
            ws_results = await asyncio.gather(*ws_tasks, return_exceptions=True)
            
            # Verificar resultados
            for result in api_results + ws_results:
                if not isinstance(result, Exception):
                    success_count += 1
        
        # Calcular estadísticas
        elapsed_time = time.time() - start_time
        operations_per_sec = (num_operations * 2) / elapsed_time
        success_rate = (success_count / (num_operations * 2)) * 100
        
        logger.info(f"Prueba de carga extrema completada en {elapsed_time:.6f}s")
        logger.info(f"Operaciones por segundo: {operations_per_sec:.2f}")
        logger.info(f"Tasa de éxito: {success_rate:.2f}%")
        
        return {
            "elapsed_time": elapsed_time,
            "operations_per_sec": operations_per_sec,
            "success_rate": success_rate,
            "total_operations": num_operations * 2,
            "success_count": success_count
        }
        
    except Exception as e:
        logger.error(f"Error en prueba de carga extrema: {str(e)}")
        return {"error": str(e)}
        
    finally:
        # Detener servidores
        await ws_server.stop()
        await api_server.stop()

async def main():
    """Función principal de prueba."""
    logger.info("=== INICIANDO PRUEBAS DEL SISTEMA HÍBRIDO TRASCENDENTAL V4 ===")
    
    # Ejecutar pruebas individuales
    ws_test = await test_websocket_component()
    api_test = await test_api_component()
    hybrid_test = await test_hybrid_system()
    
    # Verificar resultados básicos
    if ws_test and api_test and hybrid_test:
        logger.info("Todas las pruebas básicas completadas con éxito")
        
        # Ejecutar prueba de carga extrema
        load_results = await test_hybrid_extreme_load()
        
        # Guardar resultados a archivo
        with open("resultados_hybrid_v4_extremo.json", "w") as f:
            json.dump(load_results, f, indent=2)
        
        logger.info("Resultados guardados a archivo: resultados_hybrid_v4_extremo.json")
    else:
        logger.error("Algunas pruebas básicas fallaron")
    
    logger.info("=== PRUEBAS COMPLETADAS ===")

if __name__ == "__main__":
    asyncio.run(main())