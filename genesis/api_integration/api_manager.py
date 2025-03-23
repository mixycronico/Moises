"""
Gestor de APIs externas para el Sistema Genesis.

Este módulo proporciona una interfaz unificada para interactuar con APIs externas
como Alpha Vantage, NewsAPI, CoinMarketCap, Reddit, y DeepSeek.
"""

import os
import logging
import asyncio
import json
import time
import aiohttp
from typing import Dict, Any, List, Optional, Tuple, Callable, Coroutine

# Configurar logging
logger = logging.getLogger(__name__)

class APIRateLimiter:
    """
    Controlador de límites de tasa para llamadas a APIs externas.
    
    Gestiona los límites de tasa por API, implementando esperas automáticas
    cuando sea necesario para evitar bloqueos por exceso de solicitudes.
    """
    
    def __init__(self):
        """Inicializar controlador de límites de tasa."""
        self.api_limits = {
            "alpha_vantage": {"calls_per_minute": 5, "calls_per_day": 500},
            "news_api": {"calls_per_day": 100},
            "coinmarketcap": {"calls_per_minute": 30, "calls_per_day": 10000},
            "reddit": {"calls_per_minute": 60},
            "deepseek": {"calls_per_minute": 10, "tokens_per_minute": 500000}
        }
        
        self.api_usage = {}
        self._reset_usage_counters()
    
    def _reset_usage_counters(self):
        """Reiniciar contadores de uso."""
        current_time = time.time()
        
        for api_name in self.api_limits:
            self.api_usage[api_name] = {
                "last_call_time": 0,
                "calls_this_minute": 0,
                "calls_today": 0,
                "minute_start": current_time,
                "day_start": current_time,
                "tokens_this_minute": 0,
                "backoff_until": 0,
                "consecutive_errors": 0
            }
    
    async def check_rate_limit(self, api_name: str, token_count: int = 0) -> Tuple[bool, float]:
        """
        Verificar si se puede hacer una llamada a la API.
        
        Args:
            api_name: Nombre de la API
            token_count: Número de tokens a consumir (para APIs basadas en tokens)
            
        Returns:
            Tupla (permitido, tiempo_de_espera)
        """
        if api_name not in self.api_usage:
            logger.warning(f"API desconocida: {api_name}, permitiendo llamada")
            return True, 0
        
        current_time = time.time()
        usage = self.api_usage[api_name]
        limits = self.api_limits.get(api_name, {})
        
        # Verificar si estamos en backoff por errores
        if usage["backoff_until"] > current_time:
            wait_time = usage["backoff_until"] - current_time
            logger.info(f"En backoff para {api_name} por {wait_time:.2f}s debido a errores previos")
            return False, wait_time
        
        # Resetear contadores si ha pasado el tiempo
        if current_time - usage["minute_start"] >= 60:
            usage["calls_this_minute"] = 0
            usage["tokens_this_minute"] = 0
            usage["minute_start"] = current_time
        
        if current_time - usage["day_start"] >= 86400:  # 24 horas
            usage["calls_today"] = 0
            usage["day_start"] = current_time
        
        # Verificar límites
        if "calls_per_minute" in limits and usage["calls_this_minute"] >= limits["calls_per_minute"]:
            wait_time = 60 - (current_time - usage["minute_start"])
            logger.info(f"Límite de llamadas por minuto alcanzado para {api_name}, esperando {wait_time:.2f}s")
            return False, wait_time
        
        if "calls_per_day" in limits and usage["calls_today"] >= limits["calls_per_day"]:
            wait_time = 86400 - (current_time - usage["day_start"])
            logger.warning(f"Límite diario alcanzado para {api_name}, esperando {wait_time/3600:.2f}h")
            return False, wait_time
        
        if "tokens_per_minute" in limits and usage["tokens_this_minute"] + token_count > limits["tokens_per_minute"]:
            wait_time = 60 - (current_time - usage["minute_start"])
            logger.info(f"Límite de tokens por minuto alcanzado para {api_name}, esperando {wait_time:.2f}s")
            return False, wait_time
        
        return True, 0
    
    async def wait_if_needed(self, api_name: str, token_count: int = 0) -> bool:
        """
        Esperar si es necesario debido a límites de tasa.
        
        Args:
            api_name: Nombre de la API
            token_count: Número de tokens a consumir
            
        Returns:
            True si se puede proceder, False si debe abortar
        """
        allowed, wait_time = await self.check_rate_limit(api_name, token_count)
        
        if not allowed:
            if wait_time > 300:  # Más de 5 minutos de espera
                logger.warning(f"Tiempo de espera demasiado largo para {api_name}: {wait_time:.2f}s")
                return False
                
            logger.info(f"Esperando {wait_time:.2f}s antes de llamar a {api_name}")
            await asyncio.sleep(wait_time)
            return await self.wait_if_needed(api_name, token_count)
        
        return True
    
    def record_api_call(self, api_name: str, success: bool, token_count: int = 0):
        """
        Registrar una llamada a la API.
        
        Args:
            api_name: Nombre de la API
            success: Si la llamada fue exitosa
            token_count: Tokens consumidos (para APIs basadas en tokens)
        """
        if api_name not in self.api_usage:
            logger.warning(f"API desconocida en record_api_call: {api_name}")
            return
        
        usage = self.api_usage[api_name]
        usage["last_call_time"] = time.time()
        usage["calls_this_minute"] += 1
        usage["calls_today"] += 1
        
        if token_count > 0:
            usage["tokens_this_minute"] += token_count
        
        # Manejar errores consecutivos
        if not success:
            usage["consecutive_errors"] += 1
            
            # Implementar backoff exponencial
            if usage["consecutive_errors"] > 0:
                backoff_time = min(60 * 2 ** (usage["consecutive_errors"] - 1), 3600)  # Máximo 1 hora
                usage["backoff_until"] = time.time() + backoff_time
                logger.warning(f"Backoff para {api_name} durante {backoff_time:.2f}s debido a {usage['consecutive_errors']} errores consecutivos")
        else:
            usage["consecutive_errors"] = 0
    
    def get_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtener estadísticas de uso de APIs.
        
        Returns:
            Estadísticas de uso por API
        """
        stats = {}
        current_time = time.time()
        
        for api_name, usage in self.api_usage.items():
            minute_usage = usage["calls_this_minute"] / max(1, self.api_limits.get(api_name, {}).get("calls_per_minute", 100))
            day_usage = usage["calls_today"] / max(1, self.api_limits.get(api_name, {}).get("calls_per_day", 10000))
            
            stats[api_name] = {
                "minute_usage_pct": min(100, round(minute_usage * 100, 2)),
                "day_usage_pct": min(100, round(day_usage * 100, 2)),
                "backoff_status": "active" if usage["backoff_until"] > current_time else "none",
                "consecutive_errors": usage["consecutive_errors"],
                "calls_today": usage["calls_today"],
                "last_call_time": usage["last_call_time"]
            }
        
        return stats


class APIManager:
    """
    Gestor unificado de APIs externas para el Sistema Genesis.
    
    Proporciona una interfaz única para todas las APIs externas,
    gestionando claves, límites de tasa, retries y transformaciones de datos.
    """
    
    def __init__(self):
        """Inicializar gestor de APIs."""
        self.rate_limiter = APIRateLimiter()
        self.api_keys = self._load_api_keys()
        self.session = None
        self.initialized = False
    
    def _load_api_keys(self) -> Dict[str, str]:
        """
        Cargar claves de API desde variables de entorno.
        
        Returns:
            Diccionario con las claves disponibles
        """
        return {
            "alpha_vantage": os.environ.get("ALPHA_VANTAGE_API_KEY", ""),
            "news_api": os.environ.get("NEWS_API_KEY", ""),
            "coinmarketcap": os.environ.get("COINMARKETCAP_API_KEY", ""),
            "reddit_client_id": os.environ.get("REDDIT_CLIENT_ID", ""),
            "reddit_client_secret": os.environ.get("REDDIT_CLIENT_SECRET", ""),
            "deepseek": os.environ.get("DEEPSEEK_API_KEY", "")
        }
    
    async def initialize(self):
        """Inicializar gestor de APIs."""
        if self.initialized:
            return
        
        self.session = aiohttp.ClientSession()
        self.initialized = True
        
        # Verificar disponibilidad de APIs
        available_apis = []
        for api_name, key in self.api_keys.items():
            if key and key.strip() and not api_name.startswith("reddit_"):
                available_apis.append(api_name)
        
        if self.api_keys["reddit_client_id"] and self.api_keys["reddit_client_secret"]:
            available_apis.append("reddit")
        
        logger.info(f"APIs disponibles: {', '.join(available_apis)}")
    
    async def close(self):
        """Cerrar gestor de APIs."""
        if self.session:
            await self.session.close()
            self.session = None
        self.initialized = False
    
    def is_api_available(self, api_name: str) -> bool:
        """
        Verificar si una API está disponible.
        
        Args:
            api_name: Nombre de la API
            
        Returns:
            True si la API está disponible
        """
        if api_name == "reddit":
            return bool(self.api_keys["reddit_client_id"] and self.api_keys["reddit_client_secret"])
        
        return bool(self.api_keys.get(api_name, ""))
    
    async def call_api(self, 
                      api_name: str, 
                      endpoint: str, 
                      method: str = "GET", 
                      params: Optional[Dict[str, Any]] = None, 
                      json_data: Optional[Dict[str, Any]] = None,
                      headers: Optional[Dict[str, str]] = None,
                      retry_count: int = 3) -> Dict[str, Any]:
        """
        Realizar una llamada a una API externa.
        
        Args:
            api_name: Nombre de la API
            endpoint: Endpoint a llamar
            method: Método HTTP
            params: Parámetros para la solicitud
            json_data: Datos para solicitudes POST/PUT
            headers: Encabezados adicionales
            retry_count: Número máximo de reintentos
            
        Returns:
            Respuesta de la API
            
        Raises:
            ValueError: Si la API no está disponible
            Exception: Si la llamada falla después de los reintentos
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.is_api_available(api_name):
            return {"error": f"API {api_name} no disponible (API key no configurada)"}
        
        # Esperar si es necesario debido a límites de tasa
        can_proceed = await self.rate_limiter.wait_if_needed(api_name)
        if not can_proceed:
            return {"error": f"Límite de tasa excedido para {api_name}, abortando llamada"}
        
        # Preparar headers según la API
        all_headers = self._prepare_headers(api_name, headers or {})
        
        response_json = {}
        success = False
        try:
            for attempt in range(retry_count):
                try:
                    async with getattr(self.session, method.lower())(
                        endpoint,
                        params=params,
                        json=json_data,
                        headers=all_headers,
                        timeout=30
                    ) as response:
                        response_text = await response.text()
                        
                        if response.status == 429:  # Too Many Requests
                            retry_after = response.headers.get("Retry-After")
                            wait_time = int(retry_after) if retry_after and retry_after.isdigit() else 60
                            logger.warning(f"Límite de tasa alcanzado en {api_name}, esperando {wait_time}s (intento {attempt+1}/{retry_count})")
                            await asyncio.sleep(wait_time)
                            continue
                        
                        try:
                            response_json = json.loads(response_text)
                        except json.JSONDecodeError:
                            response_json = {"error": "Respuesta no es JSON válido", "text": response_text[:500]}
                        
                        if not 200 <= response.status < 300:
                            error_msg = f"Error en llamada a {api_name}: {response.status}"
                            logger.warning(f"{error_msg} - {response_json.get('error', response_text[:200])}")
                            
                            if response.status >= 500:  # Server error, retry
                                if attempt < retry_count - 1:
                                    wait_time = 2 ** attempt  # Exponential backoff
                                    await asyncio.sleep(wait_time)
                                    continue
                            
                            response_json = {"error": error_msg, "status": response.status, "details": response_json}
                        else:
                            success = True
                            break
                            
                except asyncio.TimeoutError:
                    if attempt < retry_count - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"Timeout en llamada a {api_name}, reintentando en {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        response_json = {"error": f"Timeout en llamada a {api_name} después de {retry_count} intentos"}
                
                except Exception as e:
                    if attempt < retry_count - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"Error en llamada a {api_name}: {str(e)}, reintentando en {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        response_json = {"error": f"Error en llamada a {api_name}: {str(e)}"}
        finally:
            # Registrar uso de la API
            self.rate_limiter.record_api_call(api_name, success)
        
        return response_json
    
    def _prepare_headers(self, api_name: str, additional_headers: Dict[str, str]) -> Dict[str, str]:
        """
        Preparar headers para una API específica.
        
        Args:
            api_name: Nombre de la API
            additional_headers: Headers adicionales
            
        Returns:
            Headers completos para la solicitud
        """
        headers = {**additional_headers}
        
        if api_name == "alpha_vantage":
            # Alpha Vantage usa el api_key como parámetro de consulta, no como header
            pass
        elif api_name == "news_api":
            headers["X-Api-Key"] = self.api_keys["news_api"]
        elif api_name == "coinmarketcap":
            headers["X-CMC_PRO_API_KEY"] = self.api_keys["coinmarketcap"]
        elif api_name == "reddit":
            # Reddit usa autenticación OAuth
            pass
        elif api_name == "deepseek":
            headers["Authorization"] = f"Bearer {self.api_keys['deepseek']}"
            headers["Content-Type"] = "application/json"
        
        headers["User-Agent"] = "GenesisSystem/1.0"
        return headers
    
    # --- Métodos específicos por API ---
    
    async def get_alpha_vantage_data(self, function: str, symbol: str, **kwargs) -> Dict[str, Any]:
        """
        Obtener datos de Alpha Vantage.
        
        Args:
            function: Función de Alpha Vantage (TIME_SERIES_DAILY, etc.)
            symbol: Símbolo del instrumento
            **kwargs: Parámetros adicionales
            
        Returns:
            Datos de Alpha Vantage
        """
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_keys["alpha_vantage"],
            **kwargs
        }
        
        endpoint = "https://www.alphavantage.co/query"
        return await self.call_api("alpha_vantage", endpoint, params=params)
    
    async def get_news_api_data(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Obtener noticias de NewsAPI.
        
        Args:
            query: Consulta de búsqueda
            **kwargs: Parámetros adicionales
            
        Returns:
            Noticias de NewsAPI
        """
        params = {
            "q": query,
            **kwargs
        }
        
        endpoint = "https://newsapi.org/v2/everything"
        return await self.call_api("news_api", endpoint, params=params)
    
    async def get_coinmarketcap_data(self, endpoint_path: str, **kwargs) -> Dict[str, Any]:
        """
        Obtener datos de CoinMarketCap.
        
        Args:
            endpoint_path: Ruta del endpoint
            **kwargs: Parámetros adicionales
            
        Returns:
            Datos de CoinMarketCap
        """
        endpoint = f"https://pro-api.coinmarketcap.com/v1/{endpoint_path}"
        return await self.call_api("coinmarketcap", endpoint, params=kwargs)
    
    async def get_reddit_data(self, subreddit: str, category: str = "hot", limit: int = 10) -> Dict[str, Any]:
        """
        Obtener datos de Reddit.
        
        Args:
            subreddit: Subreddit a consultar
            category: Categoría (hot, new, top, etc.)
            limit: Límite de posts a recuperar
            
        Returns:
            Datos de Reddit
        """
        # Implementar autenticación OAuth para Reddit
        auth_data = await self._get_reddit_auth_token()
        if "error" in auth_data:
            return auth_data
        
        token = auth_data.get("access_token")
        if not token:
            return {"error": "No se pudo obtener token de autenticación para Reddit"}
        
        headers = {"Authorization": f"Bearer {token}"}
        endpoint = f"https://oauth.reddit.com/r/{subreddit}/{category}"
        params = {"limit": limit}
        
        return await self.call_api("reddit", endpoint, headers=headers, params=params)
    
    async def _get_reddit_auth_token(self) -> Dict[str, Any]:
        """
        Obtener token de autenticación para Reddit.
        
        Returns:
            Token de autenticación
        """
        if not self.api_keys["reddit_client_id"] or not self.api_keys["reddit_client_secret"]:
            return {"error": "Credenciales de Reddit no configuradas"}
        
        auth = aiohttp.BasicAuth(
            login=self.api_keys["reddit_client_id"],
            password=self.api_keys["reddit_client_secret"]
        )
        
        headers = {
            "User-Agent": "GenesisSystem/1.0"
        }
        
        data = {
            "grant_type": "client_credentials"
        }
        
        try:
            async with self.session.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=auth,
                headers=headers,
                data=data
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Error al obtener token de Reddit: {response.status}"}
        except Exception as e:
            return {"error": f"Error en autenticación Reddit: {str(e)}"}
    
    async def call_deepseek_api(self, messages: List[Dict[str, str]], 
                              model: str = "deepseek-chat", 
                              temperature: float = 0.7,
                              max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Realizar llamada a la API de DeepSeek.
        
        Args:
            messages: Listado de mensajes para el modelo
            model: Modelo a utilizar
            temperature: Temperatura (creatividad) para la generación
            max_tokens: Número máximo de tokens a generar
            
        Returns:
            Respuesta de DeepSeek
        """
        json_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        endpoint = "https://api.deepseek.com/v1/chat/completions"
        return await self.call_api("deepseek", endpoint, method="POST", json_data=json_data)
    
    def get_api_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtener estado de las APIs configuradas.
        
        Returns:
            Estado de cada API
        """
        status = {}
        
        for api_name in ["alpha_vantage", "news_api", "coinmarketcap", "deepseek"]:
            api_key = self.api_keys.get(api_name, "")
            status[api_name] = {
                "available": bool(api_key),
                "key_configured": bool(api_key),
                "key_masked": f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "N/A"
            }
        
        # Caso especial para Reddit que usa dos claves
        reddit_client_id = self.api_keys.get("reddit_client_id", "")
        reddit_client_secret = self.api_keys.get("reddit_client_secret", "")
        status["reddit"] = {
            "available": bool(reddit_client_id and reddit_client_secret),
            "client_id_configured": bool(reddit_client_id),
            "client_secret_configured": bool(reddit_client_secret)
        }
        
        # Añadir estadísticas de uso
        usage_stats = self.rate_limiter.get_usage_stats()
        for api_name, api_status in status.items():
            if api_name in usage_stats:
                api_status["usage"] = usage_stats[api_name]
                
        return status


# Instancia global para uso en todo el sistema
api_manager = APIManager()


async def initialize():
    """Inicializar gestor de APIs."""
    await api_manager.initialize()


async def close():
    """Cerrar gestor de APIs."""
    await api_manager.close()


async def test_apis():
    """Probar APIs disponibles."""
    await initialize()
    
    results = {}
    
    # Probar Alpha Vantage
    if api_manager.is_api_available("alpha_vantage"):
        results["alpha_vantage"] = await api_manager.get_alpha_vantage_data(
            function="GLOBAL_QUOTE",
            symbol="BTC"
        )
    
    # Probar NewsAPI
    if api_manager.is_api_available("news_api"):
        results["news_api"] = await api_manager.get_news_api_data(
            query="bitcoin",
            language="es",
            sortBy="publishedAt",
            pageSize=1
        )
    
    # Probar CoinMarketCap
    if api_manager.is_api_available("coinmarketcap"):
        results["coinmarketcap"] = await api_manager.get_coinmarketcap_data(
            endpoint_path="cryptocurrency/listings/latest",
            start=1,
            limit=1,
            convert="USD"
        )
    
    # Probar Reddit
    if api_manager.is_api_available("reddit"):
        results["reddit"] = await api_manager.get_reddit_data(
            subreddit="CryptoCurrency",
            limit=1
        )
    
    # Probar DeepSeek
    if api_manager.is_api_available("deepseek"):
        results["deepseek"] = await api_manager.call_deepseek_api(
            messages=[
                {"role": "system", "content": "Eres un asistente experto en análisis de criptomonedas."},
                {"role": "user", "content": "¿Cuál es tu opinión sobre el Bitcoin en el contexto actual?"}
            ],
            max_tokens=100
        )
    
    # Mostrar estado general
    results["api_status"] = api_manager.get_api_status()
    
    await close()
    return results


if __name__ == "__main__":
    import asyncio
    
    async def run_test():
        """Ejecutar prueba de APIs."""
        results = await test_apis()
        print(json.dumps(results, indent=2))
    
    asyncio.run(run_test())