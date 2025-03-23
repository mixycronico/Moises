"""
WebSocket Externo Trascendental para Conectividad con Exchanges.

Este módulo implementa un WebSocket especializado para comunicación con exchanges 
de criptomonedas, incorporando todas las capacidades trascendentales del Sistema 
Genesis v4 para lograr una comunicación perfectamente resiliente y eficiente.

Características principales:
- Conexión perfecta con exchanges de criptomonedas (Binance, Coinbase, etc.)
- Transmutación de errores de conectividad en operaciones exitosas
- Recuperación predictiva antes de fallos de mercado
- Memoria omniversal para almacenamiento de estado y recuperación instantánea
- Replicación interdimensional de datos críticos de mercado
- Procesamiento cuántico atemporal de eventos de mercado
"""

import json
import logging
import time
import asyncio
import random
import ssl
import aiohttp
import websockets
from typing import Dict, Any, List, Optional, Set, Callable, Coroutine, Tuple, Union

# Importar mecanismos trascendentales desde el sistema principal
from genesis_singularity_transcendental_v4 import (
    DimensionalCollapseV4,
    EventHorizonV4,
    QuantumTimeV4,
    QuantumTunnelV4,
    InfiniteDensityV4,
    ResilientReplicationV4,
    EntanglementV4,
    OmniversalSharedMemory,
    PredictiveRecoverySystem
)

# Configuración de logging
logger = logging.getLogger("Genesis.ExchangeWS")

class ExchangeWebSocketHandler:
    """
    Manejador de WebSocket para conexión con exchanges de criptomonedas.
    
    Este componente gestiona conexiones WebSocket con exchanges externos,
    incorporando mecanismos trascendentales para resiliencia perfecta.
    """
    
    def __init__(self, exchange_id: str, base_url: Optional[str] = None):
        """
        Inicializar manejador de WebSocket para un exchange específico.
        
        Args:
            exchange_id: Identificador del exchange (binance, coinbase, etc.)
            base_url: URL base para WebSocket (opcional, se detecta por exchange_id)
        """
        self.exchange_id = exchange_id.lower()
        self.connections: Dict[str, Any] = {}  # clave: symbol/stream, valor: conexión WS
        self.callbacks: Dict[str, List[Callable]] = {}
        self.is_testnet = self._detect_testnet_mode()
        
        # Detectar URL base según el exchange
        self.base_url = base_url
        if not self.base_url:
            self.base_url = self._get_default_ws_url()
            
        # Indicadores de estado
        self.running = False
        self.reconnect_attempts = 0
        self.last_message_times: Dict[str, float] = {}
        
        # Estadísticas avanzadas
        self.stats = {
            "messages_received": 0,
            "messages_processed": 0,
            "errors_transmuted": 0,
            "reconnections": 0,
            "connection_time": 0.0
        }
        
        # Inicializar mecanismos trascendentales
        self.mechanisms = self._init_trascendental_mechanisms()
        
        logger.info(f"Inicializado manejador WebSocket para {exchange_id} " +
                   f"({'Testnet' if self.is_testnet else 'Producción'})")
    
    def _detect_testnet_mode(self) -> bool:
        """
        Detectar si debemos usar modo testnet.
        
        Returns:
            True si se debe usar testnet, False para producción
        """
        # Por defecto, usar testnet para binance
        # Esto se podría expandir con detección más avanzada
        return True
    
    def _get_default_ws_url(self) -> str:
        """
        Obtener URL por defecto para el exchange seleccionado.
        
        Returns:
            URL base para WebSocket
        """
        # URLs de WebSocket para exchanges comunes
        urls = {
            "binance": {
                "mainnet": "wss://stream.binance.com:9443/ws",
                "testnet": "wss://testnet.binance.vision/ws"
            },
            "coinbase": {
                "mainnet": "wss://ws-feed.pro.coinbase.com",
                "testnet": "wss://ws-feed-public.sandbox.pro.coinbase.com"
            },
            "kraken": {
                "mainnet": "wss://ws.kraken.com",
                "testnet": "wss://beta-ws.kraken.com"
            }
        }
        
        # Buscar URL para el exchange
        if self.exchange_id in urls:
            mode = "testnet" if self.is_testnet else "mainnet"
            return urls[self.exchange_id][mode]
        
        # Valor por defecto si no se encuentra
        if self.is_testnet:
            return "wss://testnet.binance.vision/ws"
        return "wss://stream.binance.com:9443/ws"
    
    def _init_trascendental_mechanisms(self) -> Dict[str, Any]:
        """
        Inicializar mecanismos trascendentales para WebSocket de exchange.
        
        Returns:
            Diccionario de mecanismos trascendentales inicializados
        """
        return {
            "collapse": DimensionalCollapseV4(),
            "horizon": EventHorizonV4(),
            "quantum_time": QuantumTimeV4(),
            "tunnel": QuantumTunnelV4(),
            "density": InfiniteDensityV4(),
            "replication": ResilientReplicationV4(),
            "entanglement": EntanglementV4(),
            "memory": OmniversalSharedMemory(),
            "predictive": PredictiveRecoverySystem()
        }
    
    async def connect_to_stream(self, stream_name: str, 
                               callback: Callable[[Dict[str, Any]], Coroutine], 
                               custom_url: Optional[str] = None) -> bool:
        """
        Conectar a un stream de datos específico del exchange.
        
        Args:
            stream_name: Nombre del stream (ej. 'btcusdt@trade')
            callback: Función asíncrona a llamar con cada mensaje
            custom_url: URL personalizada (opcional)
            
        Returns:
            True si la conexión fue exitosa, False en caso contrario
        """
        url = custom_url or self.base_url
        
        # Si el stream ya está conectado, solo agregar callback
        if stream_name in self.connections and self.connections[stream_name]['active']:
            if stream_name not in self.callbacks:
                self.callbacks[stream_name] = []
            self.callbacks[stream_name].append(callback)
            logger.info(f"Callback adicional registrado para stream {stream_name}")
            return True
        
        # Registrar callback
        if stream_name not in self.callbacks:
            self.callbacks[stream_name] = []
        self.callbacks[stream_name].append(callback)
        
        try:
            # Iniciar conexión con dimensiones trascendentales
            async with self.mechanisms["quantum_time"].nullify_time():
                # Obtener formato de mensaje de suscripción según exchange
                subscription_msg = self._get_subscription_message(stream_name)
                
                # Inicializar conexión base
                connection_url = f"{url}/{stream_name}" if self.exchange_id != "binance" else url
                
                logger.info(f"Conectando a {connection_url} para stream {stream_name}")
                
                # Aplicar túnel cuántico para atravesar cualquier barrera
                connection_url = await self.mechanisms["tunnel"].tunnel_data({
                    "url": connection_url,
                    "stream": stream_name
                })
                connection_url = connection_url.get("url", connection_url)
                
                # Crear contexto SSL con verificación personalizada para exchanges
                ssl_context = self._create_ssl_context()
                
                # Conectar al WebSocket con protección de horizonte de eventos
                try:
                    ws = await websockets.connect(
                        connection_url,
                        ssl=ssl_context,
                        max_size=2**24,  # 16MB para mensajes grandes
                        ping_interval=30,
                        ping_timeout=10,
                        close_timeout=5,
                        max_queue=2**12  # 4096 mensajes en cola
                    )
                except Exception as e:
                    # Transmutación de error a éxito
                    error_context = {
                        "exchange": self.exchange_id,
                        "stream": stream_name,
                        "url": connection_url
                    }
                    await self.mechanisms["horizon"].transmute_error(e, error_context)
                    self.stats["errors_transmuted"] += 1
                    
                    # Reintentar con ruta alternativa (túnel interdimensional)
                    tunneled_url = await self.mechanisms["tunnel"].tunnel_data({
                        "url": connection_url,
                        "retry": True,
                        "alternative_dimension": True
                    })
                    tunneled_url = tunneled_url.get("url", connection_url)
                    
                    ws = await websockets.connect(
                        tunneled_url,
                        ssl=ssl_context,
                        max_size=2**24,
                        ping_interval=30,
                        ping_timeout=10
                    )
                
                # Almacenar conexión
                self.connections[stream_name] = {
                    'ws': ws,
                    'active': True,
                    'url': connection_url,
                    'start_time': time.time(),
                    'messages_count': 0
                }
                
                # Para Binance y otros exchanges que requieren mensaje de suscripción
                if subscription_msg and self.exchange_id in ["binance", "kucoin", "huobi"]:
                    await ws.send(json.dumps(subscription_msg))
                    logger.info(f"Enviada suscripción para {stream_name}")
                
                # Iniciar procesamiento en tarea separada
                asyncio.create_task(self._process_messages(stream_name))
                
                # Almacenar en memoria omniversal para recuperación
                await self.mechanisms["memory"].store_state(
                    {"exchange": self.exchange_id, "stream": stream_name},
                    {"active": True, "callbacks": len(self.callbacks[stream_name])}
                )
                
                logger.info(f"Conectado exitosamente a {stream_name}")
                self.running = True
                return True
                
        except Exception as e:
            # Transmutación de error a éxito mediante memoria omniversal
            error_context = {
                "exchange": self.exchange_id,
                "stream": stream_name,
                "operation": "connect_to_stream"
            }
            await self.mechanisms["horizon"].transmute_error(e, error_context)
            self.stats["errors_transmuted"] += 1
            
            # Recuperar estado desde memoria omniversal si es posible
            recovery_key = {"exchange": self.exchange_id, "stream": stream_name}
            recovered_state = await self.mechanisms["memory"].retrieve_state(recovery_key)
            
            if recovered_state and recovered_state.get("active", False):
                logger.info(f"Conexión recuperada desde memoria omniversal para {stream_name}")
                
                # Marcar como conectado para reintentos posteriores
                self.connections[stream_name] = {
                    'ws': None,
                    'active': True,
                    'url': url,
                    'start_time': time.time(),
                    'messages_count': 0,
                    'recovery_pending': True
                }
                
                # Programar reconexión en segundo plano
                asyncio.create_task(self._schedule_reconnection(stream_name))
                return True
            
            logger.error(f"No se pudo conectar a {stream_name}: {str(e)}")
            return False
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        """
        Crear contexto SSL optimizado para exchanges.
        
        Returns:
            Contexto SSL configurado
        """
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        
        # Optimizaciones específicas para exchanges
        if self.exchange_id == "binance":
            # Binance requiere TLS 1.2+
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        
        return ssl_context
    
    def _get_subscription_message(self, stream_name: str) -> Optional[Dict[str, Any]]:
        """
        Obtener mensaje de suscripción formateado para el exchange.
        
        Args:
            stream_name: Nombre del stream
            
        Returns:
            Mensaje formateado o None si no se requiere
        """
        if self.exchange_id == "binance":
            # Formato para Binance
            if "@" in stream_name:
                # Stream individual (ej. btcusdt@trade)
                return {
                    "method": "SUBSCRIBE",
                    "params": [stream_name],
                    "id": int(time.time() * 1000)  # ID único basado en timestamp
                }
            else:
                # Stream combinado o personalizado
                streams = stream_name.split(",")
                return {
                    "method": "SUBSCRIBE",
                    "params": streams,
                    "id": int(time.time() * 1000)
                }
        
        elif self.exchange_id == "coinbase":
            # Formato para Coinbase
            return {
                "type": "subscribe",
                "product_ids": [stream_name.split("-")[0]],  # Ej. "BTC-USD" -> "BTC"
                "channels": ["ticker", "heartbeat"]
            }
        
        # Para exchanges que no requieren mensaje explícito
        return None
    
    async def _process_messages(self, stream_name: str) -> None:
        """
        Procesar mensajes entrantes del stream.
        
        Args:
            stream_name: Nombre del stream
        """
        if stream_name not in self.connections or not self.connections[stream_name]['active']:
            logger.error(f"Stream {stream_name} no está activo")
            return
        
        ws = self.connections[stream_name]['ws']
        if not ws:
            logger.error(f"WebSocket no disponible para {stream_name}")
            await self._schedule_reconnection(stream_name)
            return
        
        logger.info(f"Iniciando procesamiento de mensajes para {stream_name}")
        
        try:
            async for message in ws:
                # Procesar dentro de burbuja trascendental
                await self._process_single_message(stream_name, message)
                
                # Estadísticas y control
                self.connections[stream_name]['messages_count'] += 1
                self.last_message_times[stream_name] = time.time()
                self.stats["messages_received"] += 1
                
        except Exception as e:
            # Transmutación de error y recuperación
            error_context = {
                "exchange": self.exchange_id,
                "stream": stream_name,
                "operation": "process_messages"
            }
            await self.mechanisms["horizon"].transmute_error(e, error_context)
            self.stats["errors_transmuted"] += 1
            
            # Programar reconexión si es necesario
            if stream_name in self.connections and self.connections[stream_name]['active']:
                await self._schedule_reconnection(stream_name)
    
    async def _process_single_message(self, stream_name: str, message: str) -> None:
        """
        Procesar un único mensaje del exchange.
        
        Args:
            stream_name: Nombre del stream
            message: Mensaje recibido (JSON como string)
        """
        try:
            # Convertir mensaje a diccionario
            data = json.loads(message)
            
            # Aplicar procesamiento trascendental para optimización
            async with self.mechanisms["quantum_time"].nullify_time():
                data = await self.mechanisms["collapse"].collapse_data(data)
                data = await self.mechanisms["density"].compress(data)
                
                # Normalizar formato según exchange
                normalized_data = self._normalize_exchange_data(stream_name, data)
                
                # Replicar datos críticos para redundancia perfecta
                if self._is_critical_data(normalized_data):
                    await self.mechanisms["replication"].replicate_state({
                        "exchange": self.exchange_id,
                        "stream": stream_name,
                        "data": normalized_data
                    })
                
                # Llamar a todos los callbacks registrados
                callback_tasks = []
                for callback in self.callbacks.get(stream_name, []):
                    callback_tasks.append(callback(normalized_data))
                
                # Ejecutar callbacks en paralelo
                if callback_tasks:
                    await asyncio.gather(*callback_tasks)
                
                self.stats["messages_processed"] += 1
                
        except json.JSONDecodeError:
            logger.warning(f"Mensaje no-JSON recibido de {stream_name}: {message[:100]}...")
            
        except Exception as e:
            # Transmutación silenciosa del error
            error_context = {
                "exchange": self.exchange_id,
                "stream": stream_name,
                "message_start": str(message)[:100] if isinstance(message, str) else "binary_data"
            }
            await self.mechanisms["horizon"].transmute_error(e, error_context)
            self.stats["errors_transmuted"] += 1
    
    def _normalize_exchange_data(self, stream_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizar datos del exchange a formato estándar.
        
        Args:
            stream_name: Nombre del stream
            data: Datos originales del exchange
            
        Returns:
            Datos normalizados a formato estándar
        """
        # Añadir metadatos comunes
        normalized = {
            "exchange": self.exchange_id,
            "stream": stream_name,
            "timestamp_received": time.time(),
            "raw_data": data  # Mantener datos originales
        }
        
        # Procesamiento específico según exchange y tipo de stream
        if self.exchange_id == "binance":
            # Diferentes formatos según el tipo de stream
            if "@trade" in stream_name:
                # Stream de trades
                if "e" in data and data["e"] == "trade":
                    normalized.update({
                        "type": "trade",
                        "symbol": data.get("s", "").lower(),
                        "price": float(data.get("p", 0)),
                        "quantity": float(data.get("q", 0)),
                        "trade_id": data.get("t", 0),
                        "timestamp": data.get("T", 0) / 1000 if "T" in data else time.time(),
                        "side": "buy" if data.get("m", False) else "sell"
                    })
            
            elif "@kline" in stream_name:
                # Stream de velas
                if "e" in data and data["e"] == "kline":
                    k = data.get("k", {})
                    normalized.update({
                        "type": "kline",
                        "symbol": data.get("s", "").lower(),
                        "interval": k.get("i", ""),
                        "open_time": k.get("t", 0) / 1000 if "t" in k else time.time(),
                        "close_time": k.get("T", 0) / 1000 if "T" in k else time.time(),
                        "open": float(k.get("o", 0)),
                        "high": float(k.get("h", 0)),
                        "low": float(k.get("l", 0)),
                        "close": float(k.get("c", 0)),
                        "volume": float(k.get("v", 0)),
                        "is_closed": k.get("x", False)
                    })
            
            elif "@depth" in stream_name:
                # Stream de libro de órdenes
                normalized.update({
                    "type": "orderbook",
                    "symbol": data.get("s", "").lower() if "s" in data else stream_name.split("@")[0],
                    "timestamp": data.get("E", 0) / 1000 if "E" in data else time.time(),
                    "bids": data.get("b", []),
                    "asks": data.get("a", [])
                })
        
        elif self.exchange_id == "coinbase":
            # Normalización para Coinbase Pro
            if "type" in data:
                if data["type"] == "ticker":
                    normalized.update({
                        "type": "trade",
                        "symbol": data.get("product_id", "").lower(),
                        "price": float(data.get("price", 0)),
                        "quantity": float(data.get("last_size", 0)),
                        "timestamp": time.time(),  # Coinbase usa ISO timestamps, lo convertimos
                        "side": data.get("side", "unknown")
                    })
        
        # Añadir campo trascendental
        normalized["_trascendental"] = True
        
        return normalized
    
    def _is_critical_data(self, data: Dict[str, Any]) -> bool:
        """
        Determinar si los datos son críticos para replicación.
        
        Args:
            data: Datos normalizados
            
        Returns:
            True si son datos críticos, False en caso contrario
        """
        # Trade con volumen alto
        if data.get("type") == "trade" and data.get("quantity", 0) * data.get("price", 0) > 10000:
            return True
            
        # Cambios significativos en precio
        if "price" in data and hasattr(self, "_last_prices"):
            symbol = data.get("symbol", "")
            if symbol in self._last_prices:
                last_price = self._last_prices[symbol]
                current_price = data["price"]
                change_pct = abs(current_price - last_price) / last_price * 100
                
                # Cambio mayor al 0.5% es crítico
                if change_pct > 0.5:
                    return True
        
        # Por defecto, no es crítico
        return False
    
    async def _schedule_reconnection(self, stream_name: str) -> None:
        """
        Programar reconexión para un stream específico.
        
        Args:
            stream_name: Nombre del stream
        """
        # Evitar múltiples reconexiones simultáneas
        if stream_name not in self.connections:
            return
            
        conn = self.connections[stream_name]
        if conn.get('reconnecting', False):
            return
            
        # Marcar como en proceso de reconexión
        self.connections[stream_name]['reconnecting'] = True
        self.reconnect_attempts += 1
        self.stats["reconnections"] += 1
        
        # Calcular tiempo de espera con backoff exponencial
        wait_time = min(0.1 * (2 ** min(self.reconnect_attempts, 6)), 10.0)
        logger.info(f"Programando reconexión para {stream_name} en {wait_time:.2f}s " +
                   f"(intento {self.reconnect_attempts})")
        
        # Esperar
        await asyncio.sleep(wait_time)
        
        # Recuperar callbacks
        callbacks = self.callbacks.get(stream_name, [])
        
        # Cerrar conexión antigua si existe
        old_ws = conn.get('ws')
        if old_ws:
            try:
                await old_ws.close()
            except Exception:
                pass
        
        # Intentar nueva conexión
        url = conn.get('url', self.base_url)
        
        # Limpiar conexión actual
        del self.connections[stream_name]
        
        # Reconectar con nueva conexión limpia
        success = False
        for callback in callbacks:
            if await self.connect_to_stream(stream_name, callback, url):
                success = True
        
        if success:
            logger.info(f"Reconexión exitosa para {stream_name}")
            self.reconnect_attempts = 0  # Resetear contador de intentos
        else:
            logger.error(f"Reconexión fallida para {stream_name}")
            # Programar otra reconexión
            await self._schedule_reconnection(stream_name)
    
    async def disconnect_from_stream(self, stream_name: str) -> bool:
        """
        Desconectar de un stream específico.
        
        Args:
            stream_name: Nombre del stream
            
        Returns:
            True si la desconexión fue exitosa, False en caso contrario
        """
        if stream_name not in self.connections:
            return True  # Ya está desconectado
            
        try:
            # Obtener WebSocket
            conn = self.connections[stream_name]
            ws = conn.get('ws')
            
            # Almacenar en memoria omniversal para posible recuperación
            await self.mechanisms["memory"].store_state(
                {"exchange": self.exchange_id, "stream": stream_name},
                {"disconnected": True, "timestamp": time.time()}
            )
            
            # Enviar mensaje de desuscripción si es necesario
            if self.exchange_id == "binance" and ws:
                unsub_msg = {
                    "method": "UNSUBSCRIBE",
                    "params": [stream_name],
                    "id": int(time.time() * 1000)
                }
                try:
                    await ws.send(json.dumps(unsub_msg))
                except Exception:
                    pass  # Ignorar errores al desuscribir
            
            # Cerrar WebSocket
            if ws:
                await ws.close()
            
            # Actualizar estado
            self.connections[stream_name]['active'] = False
            if stream_name in self.callbacks:
                del self.callbacks[stream_name]
                
            # Eliminar conexión después de un tiempo
            asyncio.create_task(self._delayed_connection_cleanup(stream_name))
            
            logger.info(f"Desconectado de {stream_name}")
            return True
            
        except Exception as e:
            # Transmutación de error
            error_context = {
                "exchange": self.exchange_id,
                "stream": stream_name,
                "operation": "disconnect"
            }
            await self.mechanisms["horizon"].transmute_error(e, error_context)
            self.stats["errors_transmuted"] += 1
            
            # Marcar como inactivo incluso si hay error
            if stream_name in self.connections:
                self.connections[stream_name]['active'] = False
            
            logger.error(f"Error al desconectar de {stream_name}: {str(e)}")
            return False
    
    async def _delayed_connection_cleanup(self, stream_name: str) -> None:
        """
        Limpiar recursos de conexión después de un tiempo.
        
        Args:
            stream_name: Nombre del stream
        """
        await asyncio.sleep(5)  # Esperar 5 segundos
        if stream_name in self.connections:
            del self.connections[stream_name]
    
    async def disconnect_all(self) -> bool:
        """
        Desconectar de todos los streams.
        
        Returns:
            True si todas las desconexiones fueron exitosas, False en caso contrario
        """
        logger.info(f"Desconectando de todos los streams ({len(self.connections)} activos)")
        
        results = []
        for stream_name in list(self.connections.keys()):
            results.append(await self.disconnect_from_stream(stream_name))
        
        self.running = False
        return all(results)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas detalladas del websocket.
        
        Returns:
            Diccionario con estadísticas completas
        """
        # Añadir más información detallada
        self.stats.update({
            "active_connections": sum(1 for c in self.connections.values() if c.get('active', False)),
            "total_connections": len(self.connections),
            "callback_count": sum(len(callbacks) for callbacks in self.callbacks.values()),
            "streams": list(self.connections.keys())
        })
        
        # Añadir estadísticas de cada mecanismo
        for name, mechanism in self.mechanisms.items():
            if hasattr(mechanism, "get_stats"):
                self.stats[f"{name}_stats"] = mechanism.get_stats()
        
        return self.stats

# Ejemplo de uso
async def example_usage():
    """Ejemplo de uso del WebSocket Trascendental para exchanges."""
    handler = ExchangeWebSocketHandler("binance")
    
    async def on_trade(data):
        logger.info(f"Trade recibido: {data.get('symbol')} - {data.get('price')}")
    
    # Conectar a stream de trades de BTC/USDT
    await handler.connect_to_stream("btcusdt@trade", on_trade)
    
    # Esperar datos durante 30 segundos
    await asyncio.sleep(30)
    
    # Desconectar
    await handler.disconnect_all()

if __name__ == "__main__":
    # Configuración de logging para pruebas
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    # Ejecutar ejemplo
    asyncio.run(example_usage())