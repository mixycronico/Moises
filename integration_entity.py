"""
Implementación de entidad de integración y balanceo para Sistema Genesis.

Este módulo implementa la entidad Harmonia, especializada en la integración de API externas,
balanceo de carga y coordinación general del sistema.
"""

import os
import logging
import random
import time
import threading
import json
import requests
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from enhanced_simple_cosmic_trader import EnhancedCosmicTrader

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntegrationEntity(EnhancedCosmicTrader):
    """
    Entidad especializada en integración, coordinación y balance del sistema.
    Proporciona capacidades de integración con APIs externas y balanceo de carga.
    """
    
    def __init__(self, name: str, role: str = "Integration", father: str = "otoniel", 
                 frequency_seconds: int = 40):
        """
        Inicializar entidad de integración.
        
        Args:
            name: Nombre de la entidad
            role: Rol (siempre será "Integration")
            father: Nombre del creador/dueño
            frequency_seconds: Período de ciclo de vida en segundos
        """
        super().__init__(name, role, father, frequency_seconds)
        
        # Configuración de integración
        self.integrations = {}
        self.api_keys = {}
        self.api_endpoints = {}
        self.api_status = {}
        self.retry_backoff = {}
        
        # Estadísticas específicas
        self.stats = {
            "api_requests_sent": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "active_integrations": 0,
            "data_processed_bytes": 0,
            "last_sync_time": 0,
            "average_response_time": 0
        }
        
        # Personalidad y rasgos específicos
        self.personality_traits = ["Armónico", "Equilibrado", "Coordinador"]
        self.emotional_volatility = 0.4  # Volatilidad emocional media-baja
        
        # Especializaciones
        self.specializations = {
            "API Integration": 0.9,
            "Load Balancing": 0.8,
            "System Coordination": 0.9,
            "Resource Allocation": 0.7,
            "Data Synchronization": 0.8
        }
        
        # Métricas de rendimiento
        self.response_times = []
        self.max_response_times = 100  # Mantener solo las últimas 100 mediciones
        
        # Estado de balance del sistema
        self.system_balance_score = 0.75  # Escala 0-1
        
        # Inicializar APIs comunes para trading
        self.initialize_default_integrations()
        
        logger.info(f"[{self.name}] Entidad de integración y balance inicializada")
    
    def initialize_default_integrations(self):
        """Inicializar integraciones predeterminadas para trading."""
        # Configurar integraciones comunes (sin credenciales reales)
        default_integrations = [
            {
                "name": "alpha_vantage",
                "display_name": "Alpha Vantage",
                "base_url": "https://www.alphavantage.co/query",
                "status": "configured",
                "category": "market_data",
                "retry_policy": {"max_retries": 3, "backoff_factor": 2}
            },
            {
                "name": "binance",
                "display_name": "Binance",
                "base_url": "https://api.binance.com",
                "status": "configured",
                "category": "exchange",
                "retry_policy": {"max_retries": 5, "backoff_factor": 1.5}
            },
            {
                "name": "coinmarketcap",
                "display_name": "CoinMarketCap",
                "base_url": "https://pro-api.coinmarketcap.com/v1",
                "status": "configured",
                "category": "market_data",
                "retry_policy": {"max_retries": 3, "backoff_factor": 2}
            },
            {
                "name": "newsapi",
                "display_name": "News API",
                "base_url": "https://newsapi.org/v2",
                "status": "configured",
                "category": "news",
                "retry_policy": {"max_retries": 2, "backoff_factor": 1}
            },
            {
                "name": "finnhub",
                "display_name": "Finnhub",
                "base_url": "https://finnhub.io/api/v1",
                "status": "configured",
                "category": "market_data",
                "retry_policy": {"max_retries": 3, "backoff_factor": 2}
            }
        ]
        
        # Registrar cada integración
        for integration in default_integrations:
            self.register_integration(integration)
    
    def register_integration(self, config):
        """
        Registrar una nueva integración de API.
        
        Args:
            config: Configuración de la integración
            
        Returns:
            True si se registró correctamente, False en caso contrario
        """
        integration_name = config.get("name")
        if not integration_name:
            logger.error(f"[{self.name}] Error al registrar integración: nombre faltante")
            return False
        
        # Guardar configuración
        self.integrations[integration_name] = config
        
        # Inicializar estado de API
        self.api_status[integration_name] = {
            "status": config.get("status", "configured"),
            "last_check": time.time(),
            "health": "unknown",
            "error_count": 0,
            "last_error": None,
            "average_response_time": 0
        }
        
        # Configurar política de reintentos
        retry_policy = config.get("retry_policy", {"max_retries": 3, "backoff_factor": 2})
        self.retry_backoff[integration_name] = retry_policy
        
        # Actualizar estadísticas
        self.stats["active_integrations"] = len(self.integrations)
        
        logger.info(f"[{self.name}] Integración '{config.get('display_name')}' registrada")
        
        # Mensaje de integración
        integration_message = self.generate_message(
            "integración", 
            f"He registrado una nueva integración: {config.get('display_name')}"
        )
        self.broadcast_message(integration_message)
        
        return True
    
    def set_api_key(self, integration_name, api_key):
        """
        Establecer clave API para una integración.
        
        Args:
            integration_name: Nombre de la integración
            api_key: Clave API
            
        Returns:
            True si se configuró correctamente, False en caso contrario
        """
        if integration_name not in self.integrations:
            logger.error(f"[{self.name}] Integración '{integration_name}' no encontrada")
            return False
        
        # Almacenar API key (en producción debería cifrarse)
        self.api_keys[integration_name] = api_key
        
        # Actualizar estado
        self.api_status[integration_name]["status"] = "ready"
        
        logger.info(f"[{self.name}] API key configurada para '{integration_name}'")
        return True
    
    def api_request(self, integration_name, endpoint, method="GET", params=None, data=None, headers=None):
        """
        Realizar petición a API externa con manejo de errores y reintentos.
        
        Args:
            integration_name: Nombre de la integración a usar
            endpoint: Endpoint específico (se añade a la URL base)
            method: Método HTTP (GET, POST, etc.)
            params: Parámetros de query string
            data: Datos para enviar en body
            headers: Cabeceras adicionales
            
        Returns:
            Respuesta de la API o None si hay error
        """
        if integration_name not in self.integrations:
            logger.error(f"[{self.name}] Integración '{integration_name}' no encontrada")
            return None
        
        # Obtener configuración
        integration = self.integrations[integration_name]
        base_url = integration.get("base_url")
        retry_policy = self.retry_backoff[integration_name]
        
        # Construir URL completa
        url = f"{base_url}/{endpoint.lstrip('/')}"
        
        # Preparar headers
        request_headers = {"User-Agent": f"Genesis-Trading-System/{self.name}"}
        if headers:
            request_headers.update(headers)
        
        # Añadir API key si existe
        api_key = self.api_keys.get(integration_name)
        if api_key:
            # Diferentes APIs usan diferentes métodos para API keys
            if integration_name == "alpha_vantage":
                if params is None:
                    params = {}
                params["apikey"] = api_key
            elif integration_name == "coinmarketcap":
                request_headers["X-CMC_PRO_API_KEY"] = api_key
            elif integration_name == "newsapi":
                request_headers["X-Api-Key"] = api_key
            elif integration_name == "finnhub":
                request_headers["X-Finnhub-Token"] = api_key
            elif integration_name == "binance":
                if params is None:
                    params = {}
                params["timestamp"] = int(time.time() * 1000)
                # En un caso real, aquí se calcularía la firma HMAC
        
        # Actualizar estadísticas
        self.stats["api_requests_sent"] += 1
        
        # Implementar política de reintentos
        max_retries = retry_policy.get("max_retries", 3)
        backoff_factor = retry_policy.get("backoff_factor", 2)
        
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                
                # Realizar petición
                response = requests.request(
                    method, 
                    url, 
                    params=params, 
                    json=data if method in ["POST", "PUT", "PATCH"] else None,
                    headers=request_headers,
                    timeout=10
                )
                
                # Calcular tiempo de respuesta
                response_time = time.time() - start_time
                self.track_response_time(integration_name, response_time)
                
                # Si la respuesta no es exitosa, lanzar excepción
                response.raise_for_status()
                
                # Procesar tamaño de respuesta para estadísticas
                content_length = len(response.content)
                self.stats["data_processed_bytes"] += content_length
                
                # Actualizar estadísticas
                self.stats["successful_requests"] += 1
                
                # Actualizar último sincronización
                self.stats["last_sync_time"] = time.time()
                
                # Actualizar estado de la API
                self.api_status[integration_name]["health"] = "healthy"
                self.api_status[integration_name]["last_check"] = time.time()
                
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"[{self.name}] Error en petición a {integration_name} (intento {attempt+1}/{max_retries+1}): {str(e)}")
                
                # Actualizar estadísticas de error
                self.api_status[integration_name]["error_count"] += 1
                self.api_status[integration_name]["last_error"] = str(e)
                self.api_status[integration_name]["health"] = "degraded"
                self.api_status[integration_name]["last_check"] = time.time()
                
                # Si es el último intento, registrar como fallido
                if attempt == max_retries:
                    self.stats["failed_requests"] += 1
                    return None
                
                # Esperar antes de reintentar (backoff exponencial)
                wait_time = backoff_factor ** attempt
                time.sleep(wait_time)
    
    def track_response_time(self, integration_name, response_time):
        """
        Registrar tiempo de respuesta para métricas.
        
        Args:
            integration_name: Nombre de la integración
            response_time: Tiempo de respuesta en segundos
        """
        # Añadir a lista de tiempos de respuesta
        self.response_times.append(response_time)
        
        # Limitar tamaño de lista
        if len(self.response_times) > self.max_response_times:
            self.response_times.pop(0)
        
        # Actualizar tiempo promedio
        if self.response_times:
            self.stats["average_response_time"] = sum(self.response_times) / len(self.response_times)
        
        # Actualizar tiempo promedio específico de la API
        if integration_name in self.api_status:
            self.api_status[integration_name]["average_response_time"] = response_time
    
    def balance_resources(self):
        """
        Realizar balanceo de recursos del sistema.
        
        Returns:
            Resultados del balanceo
        """
        # Simular balanceo de recursos
        results = {
            "balanced_entities": 0,
            "resource_adjustments": 0,
            "communication_optimizations": 0,
            "load_distribution": []
        }
        
        try:
            # Verificar estado de integraciones
            active_integrations = [name for name, status in self.api_status.items() 
                                  if status.get("health") != "degraded"]
            results["active_integrations"] = len(active_integrations)
            
            # Simular ajustes de carga
            for i in range(len(active_integrations)):
                results["load_distribution"].append({
                    "integration": active_integrations[i],
                    "load_percentage": round(100 / len(active_integrations), 1)
                })
            
            # Actualizar estado de balance
            current_health = sum(1 for s in self.api_status.values() if s.get("health") == "healthy")
            total_apis = len(self.api_status)
            
            if total_apis > 0:
                self.system_balance_score = 0.5 + (0.5 * current_health / total_apis)
            
            results["system_balance_score"] = self.system_balance_score
            
            # Mensaje de balanceo
            balance_message = self.generate_message(
                "balanceo", 
                f"He realizado un balanceo de recursos. Puntaje de equilibrio del sistema: {self.system_balance_score:.2f}"
            )
            self.broadcast_message(balance_message)
            
            return results
            
        except Exception as e:
            logger.error(f"[{self.name}] Error en balanceo de recursos: {str(e)}")
            return results
    
    def monitor_integrations(self):
        """
        Monitorear estado de todas las integraciones.
        
        Returns:
            Estado de las integraciones
        """
        results = {
            "total_integrations": len(self.integrations),
            "healthy_integrations": 0,
            "degraded_integrations": 0,
            "inactive_integrations": 0,
            "integration_status": {}
        }
        
        for name, integration in self.integrations.items():
            status = self.api_status.get(name, {})
            
            # Calcular tiempo desde última verificación
            last_check = status.get("last_check", 0)
            time_since_check = time.time() - last_check
            
            # Estado actual
            current_status = status.get("status", "unknown")
            health = status.get("health", "unknown")
            
            # Registrar en resultados
            results["integration_status"][name] = {
                "name": integration.get("display_name", name),
                "status": current_status,
                "health": health,
                "time_since_check": f"{time_since_check:.1f}s",
                "error_count": status.get("error_count", 0)
            }
            
            # Contabilizar por estado
            if health == "healthy":
                results["healthy_integrations"] += 1
            elif health == "degraded":
                results["degraded_integrations"] += 1
            else:
                results["inactive_integrations"] += 1
        
        return results
    
    def get_market_data(self, symbol="BTC/USD", source="alpha_vantage"):
        """
        Obtener datos de mercado para un símbolo.
        
        Args:
            symbol: Símbolo a consultar
            source: Fuente de datos (integración)
            
        Returns:
            Datos de mercado
        """
        try:
            # Adaptar llamada según la fuente
            if source == "alpha_vantage":
                endpoint = "query"
                params = {
                    "function": "CRYPTO_INTRADAY",
                    "symbol": symbol.split('/')[0],
                    "market": symbol.split('/')[1],
                    "interval": "5min"
                }
                return self.api_request(source, endpoint, params=params)
                
            elif source == "binance":
                symbol_formatted = symbol.replace('/', '')
                endpoint = "/api/v3/klines"
                params = {
                    "symbol": symbol_formatted,
                    "interval": "5m",
                    "limit": 10
                }
                return self.api_request(source, endpoint, params=params)
                
            elif source == "coinmarketcap":
                endpoint = "cryptocurrency/quotes/latest"
                params = {
                    "symbol": symbol.split('/')[0]
                }
                return self.api_request(source, endpoint, params=params)
            
            else:
                logger.warning(f"[{self.name}] Fuente {source} no soportada para datos de mercado")
                return None
                
        except Exception as e:
            logger.error(f"[{self.name}] Error obteniendo datos de mercado: {str(e)}")
            return None
    
    def get_news(self, keywords=None, source="newsapi"):
        """
        Obtener noticias relacionadas con trading/finanzas.
        
        Args:
            keywords: Palabras clave para filtrar noticias
            source: Fuente de noticias
            
        Returns:
            Lista de noticias
        """
        try:
            if source == "newsapi":
                endpoint = "everything"
                
                query = "crypto OR bitcoin OR cryptocurrency OR finance"
                if keywords:
                    if isinstance(keywords, list):
                        keywords = " OR ".join(keywords)
                    query = f"{query} AND ({keywords})"
                
                params = {
                    "q": query,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 10
                }
                
                return self.api_request(source, endpoint, params=params)
            
            else:
                logger.warning(f"[{self.name}] Fuente {source} no soportada para noticias")
                return None
                
        except Exception as e:
            logger.error(f"[{self.name}] Error obteniendo noticias: {str(e)}")
            return None
    
    def process_cycle(self):
        """
        Procesar ciclo de vida de la entidad de integración.
        Sobreescribe el método de la clase base.
        """
        if not self.is_alive:
            return
        
        # Actualizar ciclo base
        super().process_base_cycle()
        
        # Ciclo específico de entidad de integración
        try:
            # Monitorear integraciones (30% de probabilidad)
            if random.random() < 0.3:
                self.monitor_integrations()
            
            # Balancear recursos (20% de probabilidad)
            if random.random() < 0.2:
                self.balance_resources()
            
            # Actualizar estado
            self.update_state()
            
            # Generar mensaje informativo (10% de probabilidad)
            if random.random() < 0.1:
                insight = self.generate_integration_insight()
                self.broadcast_message(insight)
                
        except Exception as e:
            logger.error(f"[{self.name}] Error en ciclo de proceso: {str(e)}")
            self.handle_error(str(e))
    
    def generate_integration_insight(self):
        """
        Generar insight sobre el estado de las integraciones.
        
        Returns:
            Mensaje con insight
        """
        monitor_data = self.monitor_integrations()
        
        insights = [
            f"Mantengo {monitor_data['healthy_integrations']} integraciones saludables y {monitor_data['degraded_integrations']} degradadas.",
            f"He procesado {self.stats['data_processed_bytes'] / 1024:.1f} KB de datos desde APIs externas.",
            f"El tiempo de respuesta promedio de las APIs es {self.stats['average_response_time'] * 1000:.1f} ms.",
            f"Estado actual de equilibrio del sistema: {self.system_balance_score:.2f} (0-1).",
            f"Mi esencia {self.dominant_trait} me permite coordinar las integraciones con precisión."
        ]
        
        # Elegir un insight aleatorio
        insight = random.choice(insights)
        
        # Formatear como mensaje
        return self.generate_message("insight", insight)
    
    def handle_error(self, error_message: str):
        """
        Manejar error de integración.
        
        Args:
            error_message: Mensaje de error
        """
        # Registrar error
        logger.error(f"[{self.name}] Error detectado: {error_message}")
        
        # Informar del error
        error_notification = self.generate_message(
            "error", 
            f"He detectado un error en las integraciones: {error_message[:50]}..."
        )
        self.broadcast_message(error_notification)
    
    def update_state(self):
        """Actualizar estado interno basado en métricas de integración."""
        # Simulación de variación de estado basado en actividad
        energy_variation = 0
        
        # Perder energía por peticiones
        energy_loss = (self.stats["api_requests_sent"]) * 0.001
        energy_variation -= energy_loss
        
        # Ganar energía por peticiones exitosas
        energy_gain = self.stats["successful_requests"] * 0.002
        energy_variation += energy_gain
        
        # Ajustar nivel basado en estadísticas
        level_adjustment = (
            self.stats["successful_requests"] * 0.0005 -
            self.stats["failed_requests"] * 0.001 +
            self.system_balance_score * 0.01
        )
        
        # Aplicar cambios
        self.adjust_energy(energy_variation)
        self.adjust_level(level_adjustment)
        
        # Actualizar emoción basada en estado de integraciones
        monitor_data = self.monitor_integrations()
        if monitor_data["degraded_integrations"] > monitor_data["healthy_integrations"]:
            self.emotion = "Preocupación"
        elif self.stats["failed_requests"] > self.stats["successful_requests"] * 0.2:
            self.emotion = "Alerta"
        elif self.system_balance_score < 0.5:
            self.emotion = "Inquietud"
        else:
            emotions = ["Armonía", "Equilibrio", "Sincronía", "Fluidez"]
            self.emotion = random.choice(emotions)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado actual de la entidad para mostrar en UI.
        Extiende el método base con información específica de integración.
        
        Returns:
            Diccionario con estado
        """
        base_status = super().get_status()
        
        # Añadir métricas específicas de integración
        integration_status = {
            "system_balance_score": self.system_balance_score,
            "active_integrations": self.stats["active_integrations"],
            "api_requests": {
                "total": self.stats["api_requests_sent"],
                "successful": self.stats["successful_requests"],
                "failed": self.stats["failed_requests"]
            },
            "average_response_time_ms": self.stats["average_response_time"] * 1000,
            "data_processed_mb": self.stats["data_processed_bytes"] / (1024 * 1024),
            "last_sync": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.stats["last_sync_time"])) if self.stats["last_sync_time"] > 0 else "Nunca",
            "specializations": self.specializations
        }
        
        # Combinar estados
        combined_status = {**base_status, **integration_status}
        return combined_status


def create_integration_entity(name="Harmonia", father="otoniel", frequency_seconds=40):
    """
    Crear y configurar una entidad de integración.
    
    Args:
        name: Nombre de la entidad
        father: Nombre del creador/dueño
        frequency_seconds: Período de ciclo de vida en segundos
        
    Returns:
        Instancia de IntegrationEntity
    """
    return IntegrationEntity(name, "Integration", father, frequency_seconds)

if __name__ == "__main__":
    # Prueba básica de la entidad
    harmonia = create_integration_entity()
    print(f"Entidad {harmonia.name} creada con rol {harmonia.role}")
    
    # Iniciar ciclo de vida en un hilo separado
    thread = threading.Thread(target=harmonia.start_lifecycle)
    thread.daemon = True
    thread.start()
    
    # Mantener vivo por un tiempo
    try:
        # Simular configuración de API key
        harmonia.set_api_key("alpha_vantage", "demo_key")
        
        for i in range(10):
            time.sleep(1)
            if i == 5:
                # Simular petición a API
                harmonia.api_request("alpha_vantage", "query", params={"function": "TIME_SERIES_DAILY", "symbol": "MSFT"})
            
            print(f"Estado de {harmonia.name}: Energía={harmonia.energy:.1f}, Nivel={harmonia.level:.1f}, Emoción={harmonia.emotion}")
    
    except KeyboardInterrupt:
        print("Deteniendo prueba...")
    finally:
        # Detener ciclo de vida
        harmonia.stop_lifecycle()
        print(f"Entidad {harmonia.name} detenida")