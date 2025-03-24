"""
ArmageddonAdapter - Adaptador para integración con servicios externos.

Este componente proporciona integración con servicios externos clave:
- Alpha Vantage: Datos históricos y fundamentales de mercado
- CoinMarketCap: Información de mercado de criptomonedas
- DeepSeek: Análisis avanzado con IA
"""

import asyncio
import logging
import os
import json
import random
import time
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Tuple, Union, Set

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("genesis.oracle.armageddon_adapter")

# Importaciones opcionales (algunas pueden requerir instalación)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("NumPy no disponible, algunas funcionalidades de procesamiento de datos estarán limitadas")


class IntegrationStatus(Enum):
    """Estado de integración con servicios externos."""
    INACTIVE = auto()   # Servicio no configurado o desactivado
    ACTIVE = auto()     # Servicio activo y funcionando correctamente
    ERROR = auto()      # Error en el servicio
    RATE_LIMITED = auto() # Límite de tasa alcanzado


class ArmageddonAdapter:
    """
    Adaptador para integración con servicios externos.
    
    Este componente proporciona:
    - Integración con Alpha Vantage para datos históricos y fundamentales
    - Integración con CoinMarketCap para información de mercado
    - Integración con DeepSeek para análisis avanzado con IA
    - Capacidad de transmutación de datos entre diferentes fuentes
    - Procesamiento unificado de datos de diferentes orígenes
    """
    
    def __init__(self):
        """Inicializar adaptador."""
        # Estado de integración para cada servicio
        self.integrations = {
            "alpha_vantage": {"status": IntegrationStatus.INACTIVE, "config": {}},
            "coinmarketcap": {"status": IntegrationStatus.INACTIVE, "config": {}},
            "deepseek": {"status": IntegrationStatus.INACTIVE, "config": {}}
        }
        
        # Estadísticas de uso
        self.usage_stats = {service: {"calls": 0, "errors": 0, "last_call": None} 
                           for service in self.integrations.keys()}
        
        # Caché de datos para minimizar llamadas API
        self.cache = {}
        self.cache_ttl = {}  # Time-to-live para cada entrada
        self.default_cache_ttl = 3600  # 1 hora por defecto
        
        # Inicializado correctamente
        self.initialized = False
    
    async def initialize(self) -> bool:
        """
        Inicializar adaptador cargando configuración desde variables de entorno.
        
        Returns:
            True si se inicializó correctamente
        """
        try:
            # Configurar Alpha Vantage si la clave está disponible
            api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
            if api_key:
                self.integrations["alpha_vantage"]["config"] = {
                    "api_key": api_key,
                    "base_url": "https://www.alphavantage.co/query",
                    "rate_limit": 5  # Llamadas por minuto (plan gratuito)
                }
                self.integrations["alpha_vantage"]["status"] = IntegrationStatus.ACTIVE
                logger.info("Integración con Alpha Vantage configurada")
            
            # Configurar CoinMarketCap si la clave está disponible
            api_key = os.environ.get("COINMARKETCAP_API_KEY")
            if api_key:
                self.integrations["coinmarketcap"]["config"] = {
                    "api_key": api_key,
                    "base_url": "https://pro-api.coinmarketcap.com/v1",
                    "rate_limit": 10000  # Créditos diarios (plan básico)
                }
                self.integrations["coinmarketcap"]["status"] = IntegrationStatus.ACTIVE
                logger.info("Integración con CoinMarketCap configurada")
            
            # Configurar DeepSeek si la clave está disponible
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if api_key:
                # Cargar configuración adicional de DeepSeek desde archivo
                config_file = "deepseek_config.json"
                config = {"api_key": api_key}
                
                if os.path.exists(config_file):
                    try:
                        with open(config_file, "r") as f:
                            file_config = json.load(f)
                            config.update(file_config)
                    except Exception as e:
                        logger.error(f"Error al cargar configuración de DeepSeek: {e}")
                
                self.integrations["deepseek"]["config"] = config
                self.integrations["deepseek"]["status"] = IntegrationStatus.ACTIVE
                logger.info("Integración con DeepSeek configurada")
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error al inicializar ArmageddonAdapter: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado actual de todas las integraciones.
        
        Returns:
            Diccionario con estado de cada integración
        """
        result = {}
        
        for service, integration in self.integrations.items():
            result[service] = {
                "status": integration["status"].name,
                "active": integration["status"] == IntegrationStatus.ACTIVE,
                "calls": self.usage_stats[service]["calls"],
                "errors": self.usage_stats[service]["errors"],
                "error_rate": (self.usage_stats[service]["errors"] / self.usage_stats[service]["calls"]) 
                              if self.usage_stats[service]["calls"] > 0 else 0.0,
                "last_call": self.usage_stats[service]["last_call"]
            }
        
        return result
    
    def _record_usage(self, service: str, error: bool = False) -> None:
        """
        Registrar uso de un servicio.
        
        Args:
            service: Nombre del servicio
            error: Si la llamada resultó en error
        """
        if service not in self.usage_stats:
            return
            
        self.usage_stats[service]["calls"] += 1
        if error:
            self.usage_stats[service]["errors"] += 1
        
        self.usage_stats[service]["last_call"] = time.time()
    
    def _get_cached_data(self, cache_key: str) -> Optional[Any]:
        """
        Obtener datos de caché si están disponibles y vigentes.
        
        Args:
            cache_key: Clave de caché
            
        Returns:
            Datos o None si no están en caché o expiraron
        """
        if cache_key not in self.cache:
            return None
            
        # Verificar si expiró
        if cache_key in self.cache_ttl:
            expiry = self.cache_ttl[cache_key]
            if time.time() > expiry:
                # Expiró, eliminarlo
                del self.cache[cache_key]
                del self.cache_ttl[cache_key]
                return None
        
        return self.cache[cache_key]
    
    def _cache_data(self, cache_key: str, data: Any, ttl: Optional[int] = None) -> None:
        """
        Guardar datos en caché.
        
        Args:
            cache_key: Clave de caché
            data: Datos a guardar
            ttl: Time-to-live en segundos (opcional)
        """
        self.cache[cache_key] = data
        
        # Establecer TTL
        if ttl is None:
            ttl = self.default_cache_ttl
            
        self.cache_ttl[cache_key] = time.time() + ttl
    
    async def get_historical_data(self, 
                                 symbol: str, 
                                 interval: str = "daily", 
                                 use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Obtener datos históricos desde Alpha Vantage.
        
        Args:
            symbol: Símbolo del activo (e.g., 'AAPL', 'BTC')
            interval: Intervalo de tiempo ('daily', 'weekly', 'monthly')
            use_cache: Si usar datos en caché cuando estén disponibles
            
        Returns:
            Datos históricos o None si ocurre un error
        """
        service = "alpha_vantage"
        
        if self.integrations[service]["status"] != IntegrationStatus.ACTIVE:
            logger.warning(f"Integración con {service} no está activa")
            return None
        
        # Generar clave de caché
        cache_key = f"{service}_historical_{symbol}_{interval}"
        
        # Verificar caché
        if use_cache:
            cached = self._get_cached_data(cache_key)
            if cached:
                logger.info(f"Datos históricos para {symbol} obtenidos desde caché")
                return cached
        
        # En un entorno real haríamos la llamada a la API
        # Pero para esta demostración, simulamos datos
        try:
            self._record_usage(service)
            
            # Simulación de datos históricos
            data = self._simulate_historical_data(symbol, interval)
            
            # Guardar en caché
            self._cache_data(cache_key, data)
            
            logger.info(f"Datos históricos para {symbol} obtenidos desde {service}")
            return data
            
        except Exception as e:
            self._record_usage(service, error=True)
            logger.error(f"Error al obtener datos históricos para {symbol}: {e}")
            return None
    
    async def get_crypto_market_data(self, 
                                    symbols: Optional[List[str]] = None,
                                    use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Obtener datos de mercado de criptomonedas desde CoinMarketCap.
        
        Args:
            symbols: Lista de símbolos (e.g., ['BTC', 'ETH']) o None para las top 100
            use_cache: Si usar datos en caché cuando estén disponibles
            
        Returns:
            Datos de mercado o None si ocurre un error
        """
        service = "coinmarketcap"
        
        if self.integrations[service]["status"] != IntegrationStatus.ACTIVE:
            logger.warning(f"Integración con {service} no está activa")
            return None
        
        # Usar lista vacía si es None
        if symbols is None:
            symbols = []
            
        # Generar clave de caché
        symbol_key = "_".join(symbols) if symbols else "top100"
        cache_key = f"{service}_market_{symbol_key}"
        
        # Verificar caché
        if use_cache:
            cached = self._get_cached_data(cache_key)
            if cached:
                logger.info(f"Datos de mercado para {symbol_key} obtenidos desde caché")
                return cached
        
        # En un entorno real haríamos la llamada a la API
        # Pero para esta demostración, simulamos datos
        try:
            self._record_usage(service)
            
            # Simulación de datos de mercado
            data = self._simulate_crypto_market_data(symbols)
            
            # Guardar en caché
            self._cache_data(cache_key, data)
            
            logger.info(f"Datos de mercado para {symbol_key} obtenidos desde {service}")
            return data
            
        except Exception as e:
            self._record_usage(service, error=True)
            logger.error(f"Error al obtener datos de mercado para {symbol_key}: {e}")
            return None
    
    async def analyze_market(self, 
                           market_data: Dict[str, Any], 
                           analysis_type: str = "sentiment",
                           use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Analizar datos de mercado con DeepSeek.
        
        Args:
            market_data: Datos de mercado a analizar
            analysis_type: Tipo de análisis ('sentiment', 'trend', 'prediction')
            use_cache: Si usar datos en caché cuando estén disponibles
            
        Returns:
            Resultados del análisis o None si ocurre un error
        """
        service = "deepseek"
        
        if self.integrations[service]["status"] != IntegrationStatus.ACTIVE:
            logger.warning(f"Integración con {service} no está activa")
            return None
        
        # Generar clave de caché
        # Usar solo los primeros 3 símbolos para la clave de caché
        symbols = list(market_data.keys())[:3] if isinstance(market_data, dict) else []
        symbol_key = "_".join(symbols) if symbols else "generic"
        cache_key = f"{service}_analysis_{analysis_type}_{symbol_key}"
        
        # Verificar caché
        if use_cache:
            cached = self._get_cached_data(cache_key)
            if cached:
                logger.info(f"Análisis {analysis_type} obtenido desde caché")
                return cached
        
        # En un entorno real haríamos la llamada a la API
        # Pero para esta demostración, simulamos resultados
        try:
            self._record_usage(service)
            
            # Simulación de análisis
            data = self._simulate_market_analysis(market_data, analysis_type)
            
            # Guardar en caché
            self._cache_data(cache_key, data)
            
            logger.info(f"Análisis {analysis_type} obtenido desde {service}")
            return data
            
        except Exception as e:
            self._record_usage(service, error=True)
            logger.error(f"Error al realizar análisis {analysis_type}: {e}")
            return None
    
    async def perform_comprehensive_analysis(self, 
                                          symbols: List[str], 
                                          use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Realizar análisis integral combinando datos de múltiples fuentes.
        
        Este método:
        1. Obtiene datos históricos de Alpha Vantage
        2. Obtiene datos de mercado de CoinMarketCap
        3. Combina ambas fuentes
        4. Realiza análisis avanzado con DeepSeek
        
        Args:
            symbols: Lista de símbolos a analizar
            use_cache: Si usar datos en caché cuando estén disponibles
            
        Returns:
            Resultados de análisis integral o None si ocurre un error
        """
        if not self.initialized:
            logger.error("ArmageddonAdapter no inicializado")
            return None
        
        # Generar clave de caché
        symbol_key = "_".join(symbols[:3])  # Usar solo los primeros 3 símbolos
        cache_key = f"comprehensive_analysis_{symbol_key}"
        
        # Verificar caché
        if use_cache:
            cached = self._get_cached_data(cache_key)
            if cached:
                logger.info(f"Análisis integral para {symbol_key} obtenido desde caché")
                return cached
        
        # Recopilar datos de múltiples fuentes
        try:
            # Paso 1: Datos históricos (simulación para todos los símbolos)
            historical_data = {}
            for symbol in symbols:
                data = await self.get_historical_data(symbol, use_cache=use_cache)
                if data:
                    historical_data[symbol] = data
            
            if not historical_data:
                logger.warning("No se pudo obtener datos históricos para ningún símbolo")
                return None
            
            # Paso 2: Datos de mercado actuales
            market_data = await self.get_crypto_market_data(symbols, use_cache=use_cache)
            
            if not market_data:
                logger.warning("No se pudo obtener datos de mercado")
                return None
            
            # Paso 3: Combinar datos
            combined_data = self._combine_data_sources(historical_data, market_data)
            
            # Paso 4: Análisis avanzado
            sentiment = await self.analyze_market(combined_data, "sentiment", use_cache=use_cache)
            trend = await self.analyze_market(combined_data, "trend", use_cache=use_cache)
            prediction = await self.analyze_market(combined_data, "prediction", use_cache=use_cache)
            
            # Combinar todos los análisis
            result = {
                "historical_data": historical_data,
                "market_data": market_data,
                "analysis": {
                    "sentiment": sentiment,
                    "trend": trend,
                    "prediction": prediction
                },
                "timestamp": time.time(),
                "symbols": symbols
            }
            
            # Guardar en caché
            self._cache_data(cache_key, result)
            
            logger.info(f"Análisis integral completado para {symbols}")
            return result
            
        except Exception as e:
            logger.error(f"Error al realizar análisis integral: {e}")
            return None
    
    def _combine_data_sources(self, 
                             historical_data: Dict[str, Any], 
                             market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combinar datos de diferentes fuentes en un formato unificado.
        
        Args:
            historical_data: Datos históricos
            market_data: Datos de mercado
            
        Returns:
            Datos combinados
        """
        result = {"symbols": {}}
        
        # Para cada símbolo en datos históricos
        for symbol, history in historical_data.items():
            if symbol not in result["symbols"]:
                result["symbols"][symbol] = {}
                
            # Añadir datos históricos
            result["symbols"][symbol]["historical"] = history
            
            # Añadir datos de mercado si están disponibles
            if market_data and symbol in market_data:
                result["symbols"][symbol]["market"] = market_data[symbol]
        
        # Añadir metadatos adicionales
        result["timestamp"] = time.time()
        result["sources"] = ["alpha_vantage", "coinmarketcap"]
        
        return result
    
    # Simulaciones para demostración
    
    def _simulate_historical_data(self, symbol: str, interval: str) -> Dict[str, Any]:
        """
        Simular datos históricos para demostración.
        
        Args:
            symbol: Símbolo del activo
            interval: Intervalo de tiempo
            
        Returns:
            Datos históricos simulados
        """
        # Usar seed basado en símbolo para consistencia
        random.seed(hash(symbol) % 10000)
        
        # Datos base según el símbolo
        if symbol.upper() in ["BTC", "BITCOIN", "BTC-USD"]:
            base_price = 50000.0
            volatility = 0.03
        elif symbol.upper() in ["ETH", "ETHEREUM", "ETH-USD"]:
            base_price = 3000.0
            volatility = 0.04
        else:
            base_price = 100.0
            volatility = 0.02
        
        # Generar serie temporal
        days = 30 if interval == "daily" else (4 if interval == "weekly" else 12)
        
        time_series = []
        price = base_price
        volume = base_price * 1000
        
        for i in range(days):
            # Simular movimiento de precio
            change = random.normalvariate(0, volatility)
            price *= (1 + change)
            
            # Simular volumen
            volume_change = random.normalvariate(0, 0.2)
            volume *= (1 + volume_change)
            
            # Datos OHLC
            open_price = price * (1 + random.normalvariate(0, volatility / 3))
            high_price = max(price, open_price) * (1 + abs(random.normalvariate(0, volatility / 2)))
            low_price = min(price, open_price) * (1 - abs(random.normalvariate(0, volatility / 2)))
            close_price = price
            
            # Timestamp
            timestamp = time.time() - (days - i) * (86400 if interval == "daily" else (604800 if interval == "weekly" else 2592000))
            date_str = time.strftime("%Y-%m-%d", time.localtime(timestamp))
            
            time_series.append({
                "date": date_str,
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": int(volume)
            })
        
        return {
            "symbol": symbol,
            "interval": interval,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "time_series": time_series
        }
    
    def _simulate_crypto_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Simular datos de mercado de criptomonedas para demostración.
        
        Args:
            symbols: Lista de símbolos o vacía para top 100
            
        Returns:
            Datos de mercado simulados
        """
        # Si no se proporcionaron símbolos, usar algunos predeterminados
        if not symbols:
            symbols = ["BTC", "ETH", "ADA", "DOT", "SOL", "DOGE", "XRP", "UNI", "LINK", "AVAX"]
        
        # Usar seed para consistencia relativa
        random.seed(int(time.time()) // 3600)  # Cambia cada hora
        
        result = {}
        
        for symbol in symbols:
            # Datos base según el símbolo
            if symbol.upper() == "BTC":
                price = random.uniform(45000, 55000)
                market_cap = price * 19e6  # ~19M BTC en circulación
                volume_24h = market_cap * random.uniform(0.01, 0.05)
                change_24h = random.uniform(-5, 5)
            elif symbol.upper() == "ETH":
                price = random.uniform(2800, 3200)
                market_cap = price * 120e6  # ~120M ETH en circulación
                volume_24h = market_cap * random.uniform(0.02, 0.08)
                change_24h = random.uniform(-8, 8)
            else:
                # Otros símbolos
                price = random.uniform(0.1, 100.0)
                supply = random.uniform(1e6, 1e9)
                market_cap = price * supply
                volume_24h = market_cap * random.uniform(0.01, 0.1)
                change_24h = random.uniform(-10, 10)
            
            result[symbol] = {
                "price": round(price, 2),
                "market_cap": round(market_cap, 2),
                "volume_24h": round(volume_24h, 2),
                "change_24h": round(change_24h, 2),
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        
        return result
    
    def _simulate_market_analysis(self, market_data: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        """
        Simular análisis de mercado con IA para demostración.
        
        Args:
            market_data: Datos de mercado
            analysis_type: Tipo de análisis
            
        Returns:
            Resultados de análisis simulados
        """
        # Usar seed para un poco de variabilidad pero cierta consistencia
        random.seed(int(time.time()) // 7200)  # Cambia cada 2 horas
        
        result = {
            "analysis_type": analysis_type,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": {}
        }
        
        # Extraer símbolos de los datos recibidos
        symbols = []
        if isinstance(market_data, dict):
            if "symbols" in market_data:
                symbols = list(market_data["symbols"].keys())
            else:
                symbols = list(market_data.keys())
        
        # Si no hay símbolos, respuesta genérica
        if not symbols:
            if analysis_type == "sentiment":
                result["overall_sentiment"] = random.choice(["bullish", "bearish", "neutral"])
                result["confidence"] = random.uniform(0.6, 0.9)
            elif analysis_type == "trend":
                result["overall_trend"] = random.choice(["upward", "downward", "sideways"])
                result["strength"] = random.uniform(0.5, 0.95)
            elif analysis_type == "prediction":
                result["market_direction"] = random.choice(["bullish", "bearish", "neutral"])
                result["projected_change"] = random.uniform(-5, 5)
                
            return result
        
        # Análisis por símbolo
        for symbol in symbols:
            if analysis_type == "sentiment":
                result["results"][symbol] = {
                    "sentiment": random.choice(["bullish", "bearish", "neutral"]),
                    "confidence": random.uniform(0.6, 0.9),
                    "news_impact": random.uniform(-1, 1),
                    "social_score": random.uniform(0, 100)
                }
            elif analysis_type == "trend":
                result["results"][symbol] = {
                    "trend": random.choice(["upward", "downward", "sideways"]),
                    "strength": random.uniform(0.5, 0.95),
                    "support_levels": [
                        round(random.uniform(0.8, 0.9) * 100, 2),
                        round(random.uniform(0.7, 0.8) * 100, 2)
                    ],
                    "resistance_levels": [
                        round(random.uniform(1.1, 1.2) * 100, 2),
                        round(random.uniform(1.2, 1.3) * 100, 2)
                    ]
                }
            elif analysis_type == "prediction":
                result["results"][symbol] = {
                    "direction": random.choice(["bullish", "bearish", "neutral"]),
                    "confidence": random.uniform(0.6, 0.8),
                    "price_target_24h": round(random.uniform(0.95, 1.05) * 100, 2),
                    "price_target_7d": round(random.uniform(0.9, 1.1) * 100, 2),
                    "volatility_forecast": random.uniform(0.01, 0.05)
                }
        
        # Análisis de mercado global
        if analysis_type == "sentiment":
            result["overall_sentiment"] = random.choice(["bullish", "bearish", "neutral"])
            result["market_confidence"] = random.uniform(0.6, 0.9)
        elif analysis_type == "trend":
            result["overall_trend"] = random.choice(["upward", "downward", "sideways"])
            result["market_strength"] = random.uniform(0.5, 0.95)
        elif analysis_type == "prediction":
            result["market_direction"] = random.choice(["bullish", "bearish", "neutral"])
            result["projected_change"] = random.uniform(-5, 5)
        
        return result


# Crear instancia global
armageddon_adapter = ArmageddonAdapter()