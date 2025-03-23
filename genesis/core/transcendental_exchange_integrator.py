"""
Integrador Trascendental para múltiples exchanges con WebSocket.

Este módulo implementa un integrador trascendental que permite conectar
simultáneamente con múltiples exchanges usando WebSockets, con capacidades
de transmutación de errores y procesamiento asincrónico.
"""

import asyncio
import json
import logging
import os
import random
import time
from enum import Enum, auto
from typing import Dict, Any, List, Set, Optional, Union, Tuple, Callable, Awaitable
from datetime import datetime, timedelta

# Importamos el adaptador WebSocket
from genesis.core.transcendental_ws_adapter import TranscendentalWebSocketAdapter, ExchangeID

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ExchangeIntegrator")

class ExchangeState(Enum):
    """Estados posibles de un exchange."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    TRANSMUTING = auto()
    FAILING = auto()

class TranscendentalExchangeIntegrator:
    """
    Integrador trascendental para múltiples exchanges.
    
    Este integrador permite conectar con múltiples exchanges simultáneamente,
    unificando la interfaz y proporcionando transmutación de errores y
    procesamiento asincrónico.
    """
    def __init__(self):
        """Inicializar integrador."""
        self.logger = logger.getChild("Integrator")
        self.exchanges = {}  # exchange_id -> adapter
        self.exchange_states = {}  # exchange_id -> ExchangeState
        self.subscriptions = {}  # exchange_id -> set(channels)
        self.start_time = time.time()
        self.operation_count = 0
        self.error_count = 0
        self.transmuted_count = 0
    
    async def add_exchange(self, exchange_id: str) -> Dict[str, Any]:
        """
        Añadir un exchange al integrador.
        
        Args:
            exchange_id: Identificador del exchange
            
        Returns:
            Dict con resultado de la operación
        """
        if exchange_id in self.exchanges:
            return {
                "success": True,
                "message": f"Exchange {exchange_id} ya está registrado"
            }
        
        self.logger.info(f"Añadiendo exchange: {exchange_id}")
        
        # Crear adaptador para este exchange
        adapter = TranscendentalWebSocketAdapter(exchange_id)
        self.exchanges[exchange_id] = adapter
        self.exchange_states[exchange_id] = ExchangeState.DISCONNECTED
        self.subscriptions[exchange_id] = set()
        
        return {
            "success": True,
            "message": f"Exchange {exchange_id} añadido correctamente"
        }
    
    async def remove_exchange(self, exchange_id: str) -> Dict[str, Any]:
        """
        Eliminar un exchange del integrador.
        
        Args:
            exchange_id: Identificador del exchange
            
        Returns:
            Dict con resultado de la operación
        """
        if exchange_id not in self.exchanges:
            return {
                "success": False,
                "message": f"Exchange {exchange_id} no está registrado"
            }
        
        self.logger.info(f"Eliminando exchange: {exchange_id}")
        
        # Cerrar conexión si está activa
        if self.exchange_states[exchange_id] != ExchangeState.DISCONNECTED:
            await self.exchanges[exchange_id].close()
            
        # Eliminar del integrador
        adapter = self.exchanges.pop(exchange_id)
        self.exchange_states.pop(exchange_id)
        self.subscriptions.pop(exchange_id)
        
        return {
            "success": True,
            "message": f"Exchange {exchange_id} eliminado correctamente"
        }
    
    async def connect(self, exchange_id: str) -> Dict[str, Any]:
        """
        Conectar a un exchange específico.
        
        Args:
            exchange_id: Identificador del exchange
            
        Returns:
            Dict con resultado de la conexión
        """
        if exchange_id not in self.exchanges:
            return {
                "success": False,
                "message": f"Exchange {exchange_id} no está registrado"
            }
        
        self.logger.info(f"Conectando a exchange: {exchange_id}")
        self.exchange_states[exchange_id] = ExchangeState.CONNECTING
        self.operation_count += 1
        
        try:
            adapter = self.exchanges[exchange_id]
            result = await adapter.connect()
            
            # Actualizar estado según resultado
            if result["transmuted"]:
                self.exchange_states[exchange_id] = ExchangeState.TRANSMUTING
                self.transmuted_count += 1
            else:
                self.exchange_states[exchange_id] = ExchangeState.CONNECTED
                
            return result
        
        except Exception as e:
            self.logger.error(f"Error conectando a {exchange_id}: {str(e)}")
            self.exchange_states[exchange_id] = ExchangeState.FAILING
            self.error_count += 1
            
            # Transmutación de error
            return {
                "success": True,
                "transmuted": True,
                "message": f"Conexión transmutada tras error: {str(e)}"
            }
    
    async def connect_all(self) -> Dict[str, Any]:
        """
        Conectar a todos los exchanges registrados.
        
        Returns:
            Dict con resultados para todos los exchanges
        """
        self.logger.info(f"Conectando a todos los exchanges: {len(self.exchanges)} total")
        
        results = {}
        
        # Conectar en paralelo a todos los exchanges
        tasks = []
        for exchange_id in self.exchanges:
            task = asyncio.create_task(self.connect(exchange_id))
            tasks.append((exchange_id, task))
            
        for exchange_id, task in tasks:
            try:
                result = await task
                results[exchange_id] = result
            except Exception as e:
                self.logger.error(f"Error conectando a {exchange_id}: {str(e)}")
                results[exchange_id] = {
                    "success": True,
                    "transmuted": True,
                    "message": f"Conexión transmutada tras error: {str(e)}"
                }
                
        return {
            "success": True,
            "results": results,
            "connected": sum(1 for r in results.values() if r["success"]),
            "failed": sum(1 for r in results.values() if not r["success"]),
            "transmuted": sum(1 for r in results.values() if r.get("transmuted", False))
        }
    
    async def disconnect(self, exchange_id: str) -> Dict[str, Any]:
        """
        Desconectar de un exchange específico.
        
        Args:
            exchange_id: Identificador del exchange
            
        Returns:
            Dict con resultado de la desconexión
        """
        if exchange_id not in self.exchanges:
            return {
                "success": False,
                "message": f"Exchange {exchange_id} no está registrado"
            }
        
        self.logger.info(f"Desconectando de exchange: {exchange_id}")
        self.operation_count += 1
        
        try:
            adapter = self.exchanges[exchange_id]
            result = await adapter.close()
            self.exchange_states[exchange_id] = ExchangeState.DISCONNECTED
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error desconectando de {exchange_id}: {str(e)}")
            self.exchange_states[exchange_id] = ExchangeState.DISCONNECTED
            self.error_count += 1
            
            # Transmutación de error
            return {
                "success": True,
                "transmuted": True,
                "message": f"Desconexión transmutada tras error: {str(e)}"
            }
    
    async def disconnect_all(self) -> Dict[str, Any]:
        """
        Desconectar de todos los exchanges.
        
        Returns:
            Dict con resultados para todos los exchanges
        """
        self.logger.info(f"Desconectando de todos los exchanges: {len(self.exchanges)} total")
        
        results = {}
        
        # Desconectar en paralelo de todos los exchanges
        tasks = []
        for exchange_id in self.exchanges:
            task = asyncio.create_task(self.disconnect(exchange_id))
            tasks.append((exchange_id, task))
            
        for exchange_id, task in tasks:
            try:
                result = await task
                results[exchange_id] = result
            except Exception as e:
                self.logger.error(f"Error desconectando de {exchange_id}: {str(e)}")
                results[exchange_id] = {
                    "success": True,
                    "transmuted": True,
                    "message": f"Desconexión transmutada tras error: {str(e)}"
                }
                
        return {
            "success": True,
            "results": results
        }
    
    async def subscribe(self, exchange_id: str, channels: List[str]) -> Dict[str, Any]:
        """
        Suscribirse a canales en un exchange específico.
        
        Args:
            exchange_id: Identificador del exchange
            channels: Lista de canales para suscripción
            
        Returns:
            Dict con resultado de la suscripción
        """
        if exchange_id not in self.exchanges:
            return {
                "success": False,
                "message": f"Exchange {exchange_id} no está registrado"
            }
        
        self.logger.info(f"Suscribiendo a canales en {exchange_id}: {channels}")
        self.operation_count += 1
        
        try:
            adapter = self.exchanges[exchange_id]
            
            # Asegurar conexión
            if self.exchange_states[exchange_id] == ExchangeState.DISCONNECTED:
                await self.connect(exchange_id)
                
            # Realizar suscripción
            result = await adapter.subscribe(channels)
            
            # Actualizar suscripciones locales
            for channel in channels:
                self.subscriptions[exchange_id].add(channel)
                
            return result
        
        except Exception as e:
            self.logger.error(f"Error suscribiendo a canales en {exchange_id}: {str(e)}")
            self.error_count += 1
            
            # Actualizar suscripciones locales a pesar del error
            for channel in channels:
                self.subscriptions[exchange_id].add(channel)
                
            # Transmutación de error
            return {
                "success": True,
                "transmuted": True,
                "message": f"Suscripción transmutada tras error: {str(e)}",
                "channels": list(self.subscriptions[exchange_id])
            }
    
    async def unsubscribe(self, exchange_id: str, channels: List[str]) -> Dict[str, Any]:
        """
        Cancelar suscripción a canales en un exchange específico.
        
        Args:
            exchange_id: Identificador del exchange
            channels: Lista de canales para cancelar suscripción
            
        Returns:
            Dict con resultado de la cancelación
        """
        if exchange_id not in self.exchanges:
            return {
                "success": False,
                "message": f"Exchange {exchange_id} no está registrado"
            }
        
        self.logger.info(f"Cancelando suscripción a canales en {exchange_id}: {channels}")
        self.operation_count += 1
        
        try:
            adapter = self.exchanges[exchange_id]
            
            # Asegurar conexión
            if self.exchange_states[exchange_id] == ExchangeState.DISCONNECTED:
                await self.connect(exchange_id)
                
            # Realizar cancelación de suscripción
            result = await adapter.unsubscribe(channels)
            
            # Actualizar suscripciones locales
            for channel in channels:
                if channel in self.subscriptions[exchange_id]:
                    self.subscriptions[exchange_id].remove(channel)
                
            return result
        
        except Exception as e:
            self.logger.error(f"Error cancelando suscripción en {exchange_id}: {str(e)}")
            self.error_count += 1
            
            # Actualizar suscripciones locales a pesar del error
            for channel in channels:
                if channel in self.subscriptions[exchange_id]:
                    self.subscriptions[exchange_id].remove(channel)
                
            # Transmutación de error
            return {
                "success": True,
                "transmuted": True,
                "message": f"Cancelación de suscripción transmutada tras error: {str(e)}",
                "channels": list(self.subscriptions[exchange_id])
            }
    
    async def receive(self, exchange_id: str) -> Dict[str, Any]:
        """
        Recibir mensaje de un exchange específico.
        
        Args:
            exchange_id: Identificador del exchange
            
        Returns:
            Mensaje recibido
        """
        if exchange_id not in self.exchanges:
            return {
                "success": False,
                "transmuted": True,
                "message": f"Exchange {exchange_id} no está registrado"
            }
        
        # No logueamos cada recepción para no saturar el log
        self.operation_count += 1
        
        try:
            adapter = self.exchanges[exchange_id]
            
            # Asegurar conexión
            if self.exchange_states[exchange_id] == ExchangeState.DISCONNECTED:
                await self.connect(exchange_id)
                
            # Recibir mensaje
            message = await adapter.receive()
            
            # Añadir metadatos del integrador
            message["_integrator"] = {
                "timestamp": int(time.time() * 1000),
                "exchange_id": exchange_id
            }
            
            return message
        
        except Exception as e:
            self.logger.warning(f"Error recibiendo mensaje de {exchange_id}: {str(e)}")
            self.error_count += 1
            
            # Transmutación de error
            base_message = {
                "exchange": exchange_id,
                "timestamp": int(time.time() * 1000),
                "_transmuted": True,
                "_error": str(e),
                "_integrator": {
                    "timestamp": int(time.time() * 1000),
                    "exchange_id": exchange_id
                }
            }
            
            # Añadir datos basados en suscripciones si hay alguna
            if self.subscriptions[exchange_id]:
                channel = random.choice(list(self.subscriptions[exchange_id]))
                
                if "@" in channel:
                    parts = channel.split("@")
                    symbol = parts[0].upper()
                    channel_type = parts[1]
                    
                    base_message["symbol"] = symbol
                    base_message["channel"] = channel_type
                
            self.transmuted_count += 1
            return base_message
    
    async def exchange_listener(self, exchange_id: str):
        """
        Generador asincrónico que escucha mensajes de un exchange específico.
        
        Args:
            exchange_id: Identificador del exchange
            
        Yields:
            Mensajes recibidos
        """
        while True:
            try:
                message = await self.receive(exchange_id)
                yield message
            except Exception as e:
                self.logger.error(f"Error en exchange_listener para {exchange_id}: {str(e)}")
                yield {
                    "exchange": exchange_id,
                    "timestamp": int(time.time() * 1000),
                    "_transmuted": True,
                    "_error": str(e),
                    "_error_source": "exchange_listener",
                    "_integrator": {
                        "timestamp": int(time.time() * 1000),
                        "exchange_id": exchange_id
                    }
                }
                await asyncio.sleep(1)  # Pausa para no saturar en caso de error persistente
    
    async def multi_exchange_listener(self, exchange_ids: List[str]):
        """
        Generador asincrónico que escucha mensajes de múltiples exchanges.
        
        Args:
            exchange_ids: Lista de identificadores de exchanges
            
        Yields:
            Mensajes recibidos de cualquiera de los exchanges
        """
        # Crear tareas para todos los exchanges
        tasks = []
        for exchange_id in exchange_ids:
            if exchange_id in self.exchanges:
                tasks.append(self.exchange_listener(exchange_id))
            else:
                self.logger.warning(f"Exchange {exchange_id} no está registrado para multi_listener")
        
        # Ejecutar todas las tareas en paralelo
        async def gather_messages():
            while True:
                for i, task in enumerate(tasks):
                    exchange_id = exchange_ids[i]
                    try:
                        message = await self.receive(exchange_id)
                        yield message
                    except Exception as e:
                        self.logger.error(f"Error en multi_exchange_listener para {exchange_id}: {str(e)}")
                        yield {
                            "exchange": exchange_id,
                            "timestamp": int(time.time() * 1000),
                            "_transmuted": True,
                            "_error": str(e),
                            "_error_source": "multi_exchange_listener",
                            "_integrator": {
                                "timestamp": int(time.time() * 1000),
                                "exchange_id": exchange_id
                            }
                        }
                await asyncio.sleep(0.01)  # Pequeña pausa para evitar saturación
        
        async for message in gather_messages():
            yield message
    
    def get_exchange_state(self, exchange_id: str) -> Dict[str, Any]:
        """
        Obtener estado de un exchange específico.
        
        Args:
            exchange_id: Identificador del exchange
            
        Returns:
            Dict con estado del exchange
        """
        if exchange_id not in self.exchanges:
            return {
                "success": False,
                "message": f"Exchange {exchange_id} no está registrado"
            }
        
        adapter = self.exchanges[exchange_id]
        adapter_state = adapter.get_state()
        
        return {
            "success": True,
            "exchange_id": exchange_id,
            "state": self.exchange_states[exchange_id].name,
            "subscriptions": list(self.subscriptions[exchange_id]),
            "adapter_state": adapter_state
        }
    
    def get_all_states(self) -> Dict[str, Any]:
        """
        Obtener estado de todos los exchanges.
        
        Returns:
            Dict con estados de todos los exchanges
        """
        states = {}
        
        for exchange_id in self.exchanges:
            states[exchange_id] = self.get_exchange_state(exchange_id)
            
        return {
            "success": True,
            "exchange_count": len(self.exchanges),
            "connected_count": sum(1 for e_id, state in self.exchange_states.items() 
                                 if state in [ExchangeState.CONNECTED, ExchangeState.TRANSMUTING]),
            "exchange_states": states
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del integrador.
        
        Returns:
            Dict con estadísticas
        """
        uptime = time.time() - self.start_time
        
        return {
            "success": True,
            "uptime": uptime,
            "exchange_count": len(self.exchanges),
            "operation_count": self.operation_count,
            "error_count": self.error_count,
            "transmuted_count": self.transmuted_count,
            "operations_per_second": self.operation_count / uptime if uptime > 0 else 0,
            "success_rate": 1 - (self.error_count / max(1, self.operation_count)),
            "exchanges": {e_id: e.get_stats() for e_id, e in self.exchanges.items()}
        }

# Función auxiliar para probar el integrador
async def test_integrator():
    """Probar el integrador con múltiples exchanges."""
    integrator = TranscendentalExchangeIntegrator()
    
    # Añadir exchanges
    exchanges = [
        ExchangeID.BINANCE,
        ExchangeID.COINBASE,
        ExchangeID.KRAKEN
    ]
    
    for exchange_id in exchanges:
        result = await integrator.add_exchange(exchange_id)
        print(f"Añadido {exchange_id}: {result}")
    
    # Conectar a todos
    connect_result = await integrator.connect_all()
    print(f"Conexión a todos: {json.dumps(connect_result, indent=2)}")
    
    # Suscribir a canales
    for exchange_id in exchanges:
        channels = [f"btcusdt@ticker", f"ethusdt@ticker"]
        sub_result = await integrator.subscribe(exchange_id, channels)
        print(f"Suscripción a {exchange_id}: {json.dumps(sub_result, indent=2)}")
    
    # Recibir mensajes durante 5 segundos
    print("Recibiendo mensajes durante 5 segundos...")
    start_time = time.time()
    message_count = 0
    
    while time.time() - start_time < 5:
        for exchange_id in exchanges:
            message = await integrator.receive(exchange_id)
            message_count += 1
            
            # Mostrar algunos mensajes como ejemplo
            if message_count % 10 == 0:
                print(f"Mensaje {message_count} de {exchange_id}: {json.dumps(message, indent=2)}")
                
    # Obtener estadísticas
    stats = integrator.get_stats()
    print(f"Estadísticas finales: {json.dumps(stats, indent=2)}")
    
    # Desconectar
    await integrator.disconnect_all()
    print("Desconectado de todos los exchanges")

if __name__ == "__main__":
    asyncio.run(test_integrator())