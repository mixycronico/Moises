"""
Módulo de Adquisición de Datos para el Pipeline de Genesis.

Este módulo se encarga de la obtención de datos de múltiples fuentes
con capacidades de resiliencia extrema y verificación de integridad.
"""
import logging
import asyncio
import time
import json
import random
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from datetime import datetime, timedelta

import aiohttp
import pandas as pd
import numpy as np

from genesis.base import GenesisComponent
from genesis.db.transcendental_database import TranscendentalDatabase

# Configuración de logging
logger = logging.getLogger("genesis.pipeline.data_acquisition")

class DataSource(GenesisComponent):
    """Fuente de datos abstracta con capacidades de resiliencia extrema."""
    
    def __init__(self, source_id: str, source_name: str, mode: str = "SINGULARITY_V4"):
        """
        Inicializar fuente de datos.
        
        Args:
            source_id: Identificador único de la fuente
            source_name: Nombre descriptivo de la fuente
            mode: Modo trascendental
        """
        super().__init__(f"data_source_{source_id}", mode)
        self.source_id = source_id
        self.source_name = source_name
        self.last_retrieval = 0
        self.data_points_retrieved = 0
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        self.backoff_factor = 1.5
        self.initial_retry_delay = 1.0  # segundos
        self.db = TranscendentalDatabase()
        
        # Caché dimensional para almacenar datos redundantes
        self.cache: Dict[str, Any] = {}
        self.cache_expiry: Dict[str, float] = {}
        self.default_cache_ttl = 300  # 5 minutos
        
        logger.info(f"Fuente de datos {source_name} ({source_id}) inicializada")
    
    async def retrieve_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Obtener datos de la fuente.
        
        Args:
            params: Parámetros para la obtención de datos
            
        Returns:
            Datos obtenidos
        """
        raise NotImplementedError("Las subclases deben implementar retrieve_data")
    
    async def retrieve_with_resilience(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Obtener datos con mecanismos de resiliencia.
        
        Args:
            params: Parámetros para la obtención de datos
            
        Returns:
            Datos obtenidos o datos de respaldo
        """
        cache_key = self._get_cache_key(params)
        
        # Verificar caché primero
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            logger.debug(f"Datos obtenidos de caché para {self.source_id}")
            return cached_data
        
        # Intentar obtener datos frescos
        try:
            start_time = time.time()
            data = await self.retrieve_data(params)
            retrieval_time = time.time() - start_time
            
            self.last_retrieval = time.time()
            self.data_points_retrieved += len(data.get("data", []))
            self.consecutive_errors = 0
            
            # Actualizar métricas
            self.update_metric("retrieval_time", retrieval_time)
            self.update_metric("success_rate", 1.0 - (self.error_count / max(1, self.operation_count)))
            
            # Guardar en caché
            ttl = params.get("cache_ttl", self.default_cache_ttl)
            self._store_in_cache(cache_key, data, ttl)
            
            # Guardar copia en base de datos para redundancia extrema
            await self._store_in_db(cache_key, data)
            
            logger.debug(f"Datos obtenidos exitosamente de {self.source_id} en {retrieval_time:.3f}s")
            self.register_operation(True)
            return data
            
        except Exception as e:
            self.consecutive_errors += 1
            retry_delay = self.initial_retry_delay * (self.backoff_factor ** (self.consecutive_errors - 1))
            self.register_operation(False)
            
            logger.warning(f"Error al obtener datos de {self.source_id}: {str(e)}, intentos fallidos: {self.consecutive_errors}")
            
            # Si superamos el máximo de errores consecutivos, usar datos de respaldo
            if self.consecutive_errors >= self.max_consecutive_errors:
                logger.error(f"Máximo de errores consecutivos alcanzado para {self.source_id}, usando datos de respaldo")
                backup_data = await self._get_backup_data(cache_key, params)
                return backup_data
            
            # Esperar antes de reintentar según backoff exponencial
            await asyncio.sleep(min(retry_delay, 60.0))  # Máximo 60 segundos
            return await self.retrieve_with_resilience(params)
    
    def _get_cache_key(self, params: Dict[str, Any]) -> str:
        """
        Generar clave de caché a partir de los parámetros.
        
        Args:
            params: Parámetros de la consulta
            
        Returns:
            Clave de caché única
        """
        # Ordenar parámetros para asegurar consistencia
        sorted_items = sorted(params.items(), key=lambda x: x[0])
        param_str = "_".join(f"{k}={v}" for k, v in sorted_items if k != "cache_ttl")
        return f"{self.source_id}_{param_str}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Obtener datos de caché si están disponibles y vigentes.
        
        Args:
            cache_key: Clave de caché
            
        Returns:
            Datos en caché o None si no están disponibles
        """
        now = time.time()
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > now:
            return self.cache[cache_key]
        return None
    
    def _store_in_cache(self, cache_key: str, data: Dict[str, Any], ttl: float) -> None:
        """
        Almacenar datos en caché con tiempo de expiración.
        
        Args:
            cache_key: Clave de caché
            data: Datos a almacenar
            ttl: Tiempo de vida en segundos
        """
        self.cache[cache_key] = data
        self.cache_expiry[cache_key] = time.time() + ttl
        
        # Limpiar entradas expiradas (cada 10 operaciones)
        if self.operation_count % 10 == 0:
            self._clean_expired_cache()
    
    def _clean_expired_cache(self) -> None:
        """Eliminar entradas de caché expiradas."""
        now = time.time()
        expired_keys = [k for k, exp in self.cache_expiry.items() if exp <= now]
        
        for key in expired_keys:
            if key in self.cache:
                del self.cache[key]
            if key in self.cache_expiry:
                del self.cache_expiry[key]
    
    async def _store_in_db(self, cache_key: str, data: Dict[str, Any]) -> bool:
        """
        Almacenar datos en base de datos para redundancia.
        
        Args:
            cache_key: Clave de caché
            data: Datos a almacenar
            
        Returns:
            True si se guardó correctamente
        """
        try:
            storage_data = {
                "timestamp": time.time(),
                "source_id": self.source_id,
                "cache_key": cache_key,
                "data": data
            }
            
            # Guardar en base de datos trascendental con clave compuesta
            await self.db.store("data_backup", f"{self.source_id}_{cache_key}", storage_data)
            return True
        except Exception as e:
            logger.warning(f"Error al guardar datos en DB: {str(e)}")
            return False
    
    async def _get_backup_data(self, cache_key: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Obtener datos de respaldo en caso de fallo.
        
        Args:
            cache_key: Clave de caché
            params: Parámetros originales
            
        Returns:
            Datos de respaldo o datos vacíos
        """
        try:
            # Intentar obtener de base de datos
            backup = await self.db.retrieve("data_backup", f"{self.source_id}_{cache_key}")
            if backup and "data" in backup:
                logger.info(f"Datos recuperados de respaldo para {self.source_id}")
                return backup["data"]
        except Exception as e:
            logger.warning(f"Error al recuperar datos de respaldo: {str(e)}")
        
        # Si todo falla, devolver estructura vacía pero válida
        logger.warning(f"No hay datos de respaldo disponibles para {self.source_id}, devolviendo vacíos")
        return {
            "data": [],
            "source": self.source_id,
            "timestamp": time.time(),
            "status": "backup_empty",
            "is_backup": True
        }

class MarketDataSource(DataSource):
    """Fuente de datos de mercado con capacidades trascendentales."""
    
    def __init__(self, 
                exchange_id: str, 
                mode: str = "SINGULARITY_V4",
                api_key: Optional[str] = None,
                api_secret: Optional[str] = None):
        """
        Inicializar fuente de datos de mercado.
        
        Args:
            exchange_id: Identificador del exchange
            mode: Modo trascendental
            api_key: API key opcional
            api_secret: API secret opcional
        """
        super().__init__(f"market_{exchange_id}", f"Datos de Mercado - {exchange_id}", mode)
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_urls = {
            "binance": "https://api.binance.com",
            "binance_testnet": "https://testnet.binance.vision",
            "coinbase": "https://api.coinbase.com",
            "kucoin": "https://api.kucoin.com",
            "bybit": "https://api.bybit.com"
        }
        self.symbols_info: Dict[str, Dict[str, Any]] = {}
        
        # Parámetros específicos para cada exchange
        self.exchange_params = {
            "binance": {
                "kline_endpoint": "/api/v3/klines",
                "ticker_endpoint": "/api/v3/ticker/24hr"
            },
            "binance_testnet": {
                "kline_endpoint": "/api/v3/klines",
                "ticker_endpoint": "/api/v3/ticker/24hr"
            }
            # Otros exchanges...
        }
        
        logger.info(f"Fuente de datos de mercado para {exchange_id} inicializada")
    
    async def retrieve_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Obtener datos del exchange.
        
        Args:
            params: Parámetros para la consulta
                - data_type: Tipo de datos (klines, ticker, orderbook)
                - symbol: Símbolo (ej. BTCUSDT)
                - interval: Intervalo para klines (1m, 5m, 15m, 1h, etc.)
                - limit: Límite de registros
            
        Returns:
            Datos obtenidos del exchange
        """
        data_type = params.get("data_type", "klines")
        symbol = params.get("symbol", "BTCUSDT")
        
        if self.exchange_id not in self.base_urls:
            raise ValueError(f"Exchange no soportado: {self.exchange_id}")
        
        base_url = self.base_urls[self.exchange_id]
        exchange_config = self.exchange_params.get(self.exchange_id, {})
        
        if data_type == "klines":
            interval = params.get("interval", "15m")
            limit = params.get("limit", 100)
            
            endpoint = exchange_config.get("kline_endpoint", "/api/v3/klines")
            url = f"{base_url}{endpoint}"
            
            query_params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            
            # Agregar parámetros adicionales específicos
            if "startTime" in params:
                query_params["startTime"] = params["startTime"]
            if "endTime" in params:
                query_params["endTime"] = params["endTime"]
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=query_params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"Error al obtener klines: {response.status}, {error_text}")
                    
                    raw_data = await response.json()
                    
                    # Convertir a formato estándar
                    processed_data = []
                    for candle in raw_data:
                        # Binance: [time, open, high, low, close, volume, ...]
                        processed_data.append({
                            "timestamp": candle[0],
                            "open": float(candle[1]),
                            "high": float(candle[2]),
                            "low": float(candle[3]),
                            "close": float(candle[4]),
                            "volume": float(candle[5]),
                            "datetime": datetime.fromtimestamp(candle[0]/1000).strftime('%Y-%m-%d %H:%M:%S')
                        })
                    
                    return {
                        "data": processed_data,
                        "source": self.exchange_id,
                        "symbol": symbol,
                        "interval": interval,
                        "timestamp": time.time(),
                        "count": len(processed_data)
                    }
        
        elif data_type == "ticker":
            endpoint = exchange_config.get("ticker_endpoint", "/api/v3/ticker/24hr")
            url = f"{base_url}{endpoint}"
            
            query_params = {}
            if symbol != "ALL":
                query_params["symbol"] = symbol
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=query_params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"Error al obtener ticker: {response.status}, {error_text}")
                    
                    raw_data = await response.json()
                    
                    # Procesar según sea un solo ticker o múltiples
                    if isinstance(raw_data, list):
                        processed_data = [{
                            "symbol": item["symbol"],
                            "price": float(item.get("lastPrice", 0)),
                            "volume": float(item.get("volume", 0)),
                            "change_percent": float(item.get("priceChangePercent", 0)),
                            "high": float(item.get("highPrice", 0)),
                            "low": float(item.get("lowPrice", 0))
                        } for item in raw_data]
                    else:
                        processed_data = [{
                            "symbol": raw_data["symbol"],
                            "price": float(raw_data.get("lastPrice", 0)),
                            "volume": float(raw_data.get("volume", 0)),
                            "change_percent": float(raw_data.get("priceChangePercent", 0)),
                            "high": float(raw_data.get("highPrice", 0)),
                            "low": float(raw_data.get("lowPrice", 0))
                        }]
                    
                    return {
                        "data": processed_data,
                        "source": self.exchange_id,
                        "data_type": "ticker",
                        "timestamp": time.time(),
                        "count": len(processed_data)
                    }
        
        else:
            raise ValueError(f"Tipo de datos no soportado: {data_type}")
    
    async def get_hot_symbols(self, top_n: int = 10, min_volume: float = 1000.0) -> List[Dict[str, Any]]:
        """
        Obtener símbolos "calientes" basados en volumen y cambio de precio.
        
        Args:
            top_n: Número de símbolos a retornar
            min_volume: Volumen mínimo en USDT
            
        Returns:
            Lista de símbolos ordenados por "calor"
        """
        params = {
            "data_type": "ticker",
            "symbol": "ALL",
            "cache_ttl": 300  # 5 minutos
        }
        
        result = await self.retrieve_with_resilience(params)
        tickers = result.get("data", [])
        
        # Filtrar por USD o USDT y volumen mínimo
        usdt_symbols = [t for t in tickers if (t["symbol"].endswith("USDT") or t["symbol"].endswith("USD")) 
                       and t["volume"] >= min_volume]
        
        # Calcular "calor" basado en volumen y cambio absoluto
        for symbol in usdt_symbols:
            symbol["heat_score"] = symbol["volume"] * abs(symbol["change_percent"]) / 100.0
        
        # Ordenar por calor descendente
        usdt_symbols.sort(key=lambda x: x["heat_score"], reverse=True)
        
        return usdt_symbols[:top_n]

class NewsDataSource(DataSource):
    """Fuente de datos de noticias y sentimiento de mercado."""
    
    def __init__(self, source_id: str, mode: str = "SINGULARITY_V4", api_key: Optional[str] = None):
        """
        Inicializar fuente de datos de noticias.
        
        Args:
            source_id: Identificador de la fuente
            mode: Modo trascendental
            api_key: API key opcional
        """
        super().__init__(f"news_{source_id}", f"Noticias - {source_id}", mode)
        self.api_key = api_key
        self.news_sources = {
            "cryptopanic": "https://cryptopanic.com/api/v1/posts/",
            "newsapi": "https://newsapi.org/v2/everything"
        }
        
        logger.info(f"Fuente de noticias {source_id} inicializada")
    
    async def retrieve_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Obtener noticias y sentimiento.
        
        Args:
            params: Parámetros para la consulta
                - currencies: Lista de monedas (ej. ["BTC", "ETH"])
                - limit: Límite de noticias
                - filter: Filtro de sentimiento (all, positive, negative)
                
        Returns:
            Datos de noticias
        """
        if self.source_id not in self.news_sources:
            raise ValueError(f"Fuente de noticias no soportada: {self.source_id}")
        
        base_url = self.news_sources[self.source_id]
        currencies = params.get("currencies", ["BTC"])
        limit = params.get("limit", 50)
        filter_type = params.get("filter", "all")
        
        if self.source_id == "cryptopanic":
            query_params = {
                "auth_token": self.api_key,
                "limit": limit,
                "public": "true"
            }
            
            if currencies and currencies != ["ALL"]:
                query_params["currencies"] = ",".join(currencies)
            
            if filter_type != "all":
                query_params["filter"] = filter_type
            
            async with aiohttp.ClientSession() as session:
                async with session.get(base_url, params=query_params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"Error al obtener noticias: {response.status}, {error_text}")
                    
                    raw_data = await response.json()
                    
                    # Procesar resultados
                    processed_data = []
                    for news in raw_data.get("results", []):
                        processed_data.append({
                            "title": news["title"],
                            "url": news["url"],
                            "source": news["source"]["title"],
                            "created_at": news["created_at"],
                            "currencies": [c["code"] for c in news.get("currencies", [])],
                            "sentiment": news.get("votes", {}).get("positive", 0) - news.get("votes", {}).get("negative", 0)
                        })
                    
                    return {
                        "data": processed_data,
                        "source": self.source_id,
                        "currencies": currencies,
                        "timestamp": time.time(),
                        "count": len(processed_data)
                    }
        
        elif self.source_id == "newsapi":
            # Implementar para NewsAPI u otras fuentes...
            pass
        
        raise ValueError(f"Implementación para {self.source_id} no disponible")

class DataAcquisitionEngine(GenesisComponent):
    """
    Motor de adquisición de datos con capacidades trascendentales.
    
    Este componente coordina la obtención de datos de múltiples fuentes
    y los integra para su uso en el pipeline.
    """
    
    def __init__(self, mode: str = "SINGULARITY_V4"):
        """
        Inicializar motor de adquisición de datos.
        
        Args:
            mode: Modo trascendental
        """
        super().__init__("data_acquisition_engine", mode)
        self.data_sources: Dict[str, DataSource] = {}
        self.market_sources: Dict[str, MarketDataSource] = {}
        self.news_sources: Dict[str, NewsDataSource] = {}
        self.db = TranscendentalDatabase()
        self.active_symbols: Set[str] = set()
        
        logger.info(f"Motor de adquisición de datos inicializado en modo {mode}")
    
    def register_market_source(self, exchange_id: str, api_key: Optional[str] = None, 
                              api_secret: Optional[str] = None) -> MarketDataSource:
        """
        Registrar fuente de datos de mercado.
        
        Args:
            exchange_id: Identificador del exchange
            api_key: API key opcional
            api_secret: API secret opcional
            
        Returns:
            Instancia de MarketDataSource
        """
        source = MarketDataSource(exchange_id, self.mode, api_key, api_secret)
        self.data_sources[source.source_id] = source
        self.market_sources[exchange_id] = source
        logger.info(f"Fuente de mercado registrada: {exchange_id}")
        return source
    
    def register_news_source(self, source_id: str, api_key: Optional[str] = None) -> NewsDataSource:
        """
        Registrar fuente de datos de noticias.
        
        Args:
            source_id: Identificador de la fuente
            api_key: API key opcional
            
        Returns:
            Instancia de NewsDataSource
        """
        source = NewsDataSource(source_id, self.mode, api_key)
        self.data_sources[source.source_id] = source
        self.news_sources[source_id] = source
        logger.info(f"Fuente de noticias registrada: {source_id}")
        return source
    
    def set_active_symbols(self, symbols: List[str]) -> None:
        """
        Establecer símbolos activos para seguimiento.
        
        Args:
            symbols: Lista de símbolos
        """
        self.active_symbols = set(symbols)
        logger.info(f"Símbolos activos establecidos: {len(symbols)}")
    
    async def get_latest_market_data(self, exchange_id: str, symbol: str, 
                                   interval: str = "15m", limit: int = 100) -> Dict[str, Any]:
        """
        Obtener datos de mercado más recientes.
        
        Args:
            exchange_id: Identificador del exchange
            symbol: Símbolo a consultar
            interval: Intervalo para klines
            limit: Límite de registros
            
        Returns:
            Datos de mercado
        """
        if exchange_id not in self.market_sources:
            logger.warning(f"Exchange {exchange_id} no registrado, usando fuente por defecto")
            exchange_id = next(iter(self.market_sources), None)
            if not exchange_id:
                raise ValueError("No hay fuentes de mercado registradas")
        
        params = {
            "data_type": "klines",
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "cache_ttl": 60  # 1 minuto para datos recientes
        }
        
        return await self.market_sources[exchange_id].retrieve_with_resilience(params)
    
    async def get_market_sentiment(self, currencies: List[str] = None) -> Dict[str, Any]:
        """
        Obtener sentimiento del mercado desde fuentes de noticias.
        
        Args:
            currencies: Lista de monedas (opcional)
            
        Returns:
            Datos de sentimiento
        """
        if not self.news_sources:
            logger.warning("No hay fuentes de noticias registradas")
            return {"data": [], "source": "none", "timestamp": time.time(), "count": 0}
        
        # Usar la primera fuente disponible
        news_source_id = next(iter(self.news_sources))
        
        if not currencies:
            currencies = ["BTC", "ETH", "XRP"]  # Default
        
        params = {
            "currencies": currencies,
            "limit": 50,
            "filter": "all",
            "cache_ttl": 900  # 15 minutos
        }
        
        news_data = await self.news_sources[news_source_id].retrieve_with_resilience(params)
        
        # Calcular sentimiento agregado por moneda
        sentiment_by_currency = {}
        for news in news_data.get("data", []):
            sentiment = news.get("sentiment", 0)
            for currency in news.get("currencies", []):
                if currency not in sentiment_by_currency:
                    sentiment_by_currency[currency] = []
                sentiment_by_currency[currency].append(sentiment)
        
        # Promediar sentimiento
        aggregated_sentiment = {}
        for currency, values in sentiment_by_currency.items():
            if not values:
                aggregated_sentiment[currency] = 0
            else:
                aggregated_sentiment[currency] = sum(values) / len(values)
        
        # Añadir sentimiento al resultado
        news_data["sentiment"] = aggregated_sentiment
        
        return news_data
    
    async def get_hot_markets(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Obtener mercados más activos combinando datos y sentimiento.
        
        Args:
            top_n: Número de mercados a retornar
            
        Returns:
            Lista de mercados "calientes"
        """
        if not self.market_sources:
            logger.warning("No hay fuentes de mercado registradas")
            return []
        
        # Usar la primera fuente disponible
        exchange_id = next(iter(self.market_sources))
        market_source = self.market_sources[exchange_id]
        
        # Obtener símbolos hot por volumen/precio
        hot_symbols = await market_source.get_hot_symbols(top_n=top_n*2)  # Obtener el doble para filtrar
        
        if not hot_symbols:
            return []
        
        # Obtener sentimiento para estos símbolos
        currencies = []
        for symbol in hot_symbols:
            # Extraer moneda base (ej. "BTC" de "BTCUSDT")
            if symbol["symbol"].endswith("USDT"):
                base = symbol["symbol"][:-4]
            elif symbol["symbol"].endswith("USD"):
                base = symbol["symbol"][:-3]
            else:
                continue
            
            if base not in currencies:
                currencies.append(base)
        
        # Limitar a top 10 monedas para consulta de sentimiento
        sentiment_data = await self.get_market_sentiment(currencies[:10])
        sentiment_by_currency = sentiment_data.get("sentiment", {})
        
        # Combinar datos de mercado con sentimiento
        for symbol in hot_symbols:
            # Extraer moneda base
            if symbol["symbol"].endswith("USDT"):
                base = symbol["symbol"][:-4]
            elif symbol["symbol"].endswith("USD"):
                base = symbol["symbol"][:-3]
            else:
                base = ""
            
            # Asignar sentimiento
            symbol["sentiment"] = sentiment_by_currency.get(base, 0)
            
            # Calcular score combinado
            volume_score = min(1.0, symbol["volume"] / 1000000)  # Normalizar volumen
            price_change = abs(symbol["change_percent"]) / 10.0  # Normalizar cambio de precio
            sentiment_score = (sentiment_by_currency.get(base, 0) + 10) / 20.0  # Normalizar sentimiento (-10 a +10 -> 0 a 1)
            
            # Fórmula combinada (ajustar pesos según necesidad)
            symbol["combined_score"] = (volume_score * 0.4) + (price_change * 0.4) + (sentiment_score * 0.2)
        
        # Ordenar por score combinado
        hot_symbols.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        
        return hot_symbols[:top_n]
    
    async def acquire_initial_data(self) -> Dict[str, Any]:
        """
        Adquirir conjunto inicial de datos para el pipeline.
        
        Returns:
            Conjunto de datos inicial
        """
        logger.info("Iniciando adquisición de datos inicial")
        
        # Registrar fuentes por defecto si no hay ninguna
        if not self.market_sources:
            self.register_market_source("binance")
        
        if not self.news_sources:
            self.register_news_source("cryptopanic")
        
        results = {}
        
        # 1. Obtener mercados hot
        hot_markets = await self.get_hot_markets(top_n=5)
        results["hot_markets"] = hot_markets
        
        # Actualizar símbolos activos si no hay ninguno
        if not self.active_symbols and hot_markets:
            self.set_active_symbols([m["symbol"] for m in hot_markets[:3]])
        
        # 2. Obtener datos de mercado para símbolos activos
        market_data = {}
        for symbol in self.active_symbols:
            exchange_id = next(iter(self.market_sources))
            symbol_data = await self.get_latest_market_data(exchange_id, symbol)
            market_data[symbol] = symbol_data
        
        results["market_data"] = market_data
        
        # 3. Obtener sentimiento general
        sentiment_data = await self.get_market_sentiment()
        results["sentiment"] = sentiment_data
        
        # Registrar la operación
        self.register_operation(True)
        logger.info(f"Adquisición inicial completada con {len(market_data)} símbolos")
        
        return {
            "data": results,
            "timestamp": time.time(),
            "source": "data_acquisition_engine"
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del motor de adquisición.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = super().get_stats()
        
        # Agregar estadísticas específicas
        engine_stats = {
            "active_symbols": list(self.active_symbols),
            "market_sources": len(self.market_sources),
            "news_sources": len(self.news_sources),
            "total_sources": len(self.data_sources),
            "sources": {
                source_id: source.get_stats() 
                for source_id, source in self.data_sources.items()
            }
        }
        
        stats.update(engine_stats)
        return stats
    
    async def initialize(self) -> bool:
        """
        Inicializar motor de adquisición.
        
        Returns:
            True si se inicializó correctamente
        """
        try:
            # Registrar fuentes por defecto
            if not self.market_sources:
                self.register_market_source("binance")
            
            if not self.news_sources:
                self.register_news_source("cryptopanic")
            
            logger.info(f"Motor de adquisición inicializado con {len(self.data_sources)} fuentes")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar motor de adquisición: {str(e)}")
            return False

# Función de procesamiento para el pipeline
async def process_data_acquisition(data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Función de adquisición de datos para el pipeline.
    
    Args:
        data: Datos de entrada (posiblemente vacíos al inicio)
        context: Contexto de ejecución
        
    Returns:
        Datos adquiridos
    """
    engine = DataAcquisitionEngine()
    
    # Inicializar si es necesario
    if not engine.market_sources:
        await engine.initialize()
    
    # Adquirir datos
    acquisition_result = await engine.acquire_initial_data()
    
    # Actualizar datos de entrada con lo adquirido
    if not data:
        data = {}
    
    data["market_data"] = acquisition_result["data"].get("market_data", {})
    data["hot_markets"] = acquisition_result["data"].get("hot_markets", [])
    data["sentiment"] = acquisition_result["data"].get("sentiment", {})
    data["acquisition_timestamp"] = acquisition_result["timestamp"]
    
    # Registrar fuentes en el contexto
    context["data_sources"] = list(engine.data_sources.keys())
    
    logger.info(f"Adquisición de datos completada para {len(data['market_data'])} símbolos")
    return data

# Instancia global para uso directo
acquisition_engine = DataAcquisitionEngine()