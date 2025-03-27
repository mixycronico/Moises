"""
Cosmic Trading: Extensión de trading para la Familia Cósmica del Sistema Genesis.

Este módulo expande las capacidades de Aetherion y Lunareth, permitiéndoles:
1. Analizar mercados de criptomonedas
2. Desarrollar estrategias de trading
3. Evolucionar de forma autónoma en sus habilidades
4. Colaborar entre sí para mejorar sus resultados
5. Mantener una vida simulada con energía y conocimiento

Autor: Otoniel (Implementación basada en la propuesta del Cosmic Trading Collective)
"""

import os
import json
import time
import random
import logging
import requests
import threading
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import deque
from abc import ABC, abstractmethod
import psycopg2
from psycopg2 import pool
from psycopg2.extras import DictCursor

# Configuración de logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pool de conexiones PostgreSQL
postgresql_pool = None

def get_db_pool():
    """Obtener pool de conexiones a PostgreSQL."""
    global postgresql_pool
    if postgresql_pool is None:
        try:
            postgresql_pool = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=os.environ.get("DATABASE_URL")
            )
            logger.info("Pool de conexiones PostgreSQL inicializado")
        except Exception as e:
            logger.error(f"Error al inicializar pool PostgreSQL: {e}")
            raise
    return postgresql_pool

class CosmicTrader(ABC):
    """Base para entidades con capacidades de trading y vida simulada."""
    
    def __init__(self, name: str, role: str, father: str = "otoniel", energy_rate: float = 0.1, 
                 frequency_seconds: int = 15):
        """
        Inicializar trader cósmico con capacidades básicas.
        
        Args:
            name: Nombre de la entidad
            role: Rol especializado ("Speculator", "Strategist", etc.)
            father: El creador/dueño del sistema
            energy_rate: Tasa de consumo de energía
            frequency_seconds: Frecuencia de ciclo de vida en segundos
        """
        self.name = name
        self.role = role
        self.father = father
        self.level = 0.0
        self.energy = 100.0
        self.knowledge = 0.0
        self.capabilities = ["market_sensing"]
        self.memory = deque(maxlen=1000)
        self.energy_rate = energy_rate
        self.frequency = frequency_seconds
        self.alive = True
        self.api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        self.trading_history = []
        self.network = None
        self.init_db()
        
        logger.info(f"[{self.name}] Inicialización completa como {self.role}")
        
    def init_db(self):
        """Inicializar tablas en la base de datos PostgreSQL."""
        try:
            pool = get_db_pool()
            conn = pool.getconn()
            with conn.cursor() as c:
                # Crear tabla para datos de vida
                c.execute(f'''
                    CREATE TABLE IF NOT EXISTS {self.name}_life (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP, 
                        level REAL, 
                        energy REAL, 
                        knowledge REAL, 
                        capabilities TEXT, 
                        log TEXT
                    )
                ''')
                
                # Crear tabla para transacciones de trading
                c.execute(f'''
                    CREATE TABLE IF NOT EXISTS {self.name}_trades (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP, 
                        symbol TEXT, 
                        action TEXT, 
                        price REAL, 
                        success BOOLEAN
                    )
                ''')
                conn.commit()
            pool.putconn(conn)
            logger.info(f"[{self.name}] Tablas PostgreSQL inicializadas")
        except Exception as e:
            logger.error(f"[{self.name}] Error al inicializar tablas PostgreSQL: {e}")
            raise

    def log_state(self, log_message):
        """Registrar estado actual en la base de datos PostgreSQL."""
        timestamp = datetime.now()
        capabilities_str = ",".join(self.capabilities)
        try:
            pool = get_db_pool()
            conn = pool.getconn()
            with conn.cursor() as c:
                c.execute(f'''
                    INSERT INTO {self.name}_life 
                    (timestamp, level, energy, knowledge, capabilities, log) 
                    VALUES (%s, %s, %s, %s, %s, %s)
                ''', (timestamp, self.level, self.energy, self.knowledge, capabilities_str, log_message))
                conn.commit()
            pool.putconn(conn)
            
            # Mantener memoria local para rendimiento
            self.memory.append({
                "timestamp": timestamp.isoformat(), 
                "level": self.level, 
                "energy": self.energy, 
                "knowledge": self.knowledge,
                "log": log_message
            })
            logger.debug(f"[{self.name}] Estado registrado: {log_message}")
        except Exception as e:
            logger.error(f"[{self.name}] Error al registrar estado: {e}")

    def log_trade(self, symbol, action, price, success):
        """Registrar operación de trading en PostgreSQL."""
        timestamp = datetime.now()
        try:
            pool = get_db_pool()
            conn = pool.getconn()
            with conn.cursor() as c:
                c.execute(f'''
                    INSERT INTO {self.name}_trades 
                    (timestamp, symbol, action, price, success) 
                    VALUES (%s, %s, %s, %s, %s)
                ''', (timestamp, symbol, action, price, success))
                conn.commit()
            pool.putconn(conn)
            
            # Mantener historia local para rendimiento
            self.trading_history.append({
                "timestamp": timestamp.isoformat(),
                "symbol": symbol, 
                "action": action, 
                "price": price, 
                "success": success
            })
            logger.info(f"[{self.name}] Trade registrado: {action} {symbol} a {price}")
        except Exception as e:
            logger.error(f"[{self.name}] Error al registrar trade: {e}")

    def metabolize(self):
        """Gestionar ciclo de energía vital."""
        self.energy -= self.energy_rate * random.uniform(0.5, 1.5)
        if "energy_harvesting" in self.capabilities:
            self.energy += random.uniform(1.0, 3.0) * (self.knowledge / 100.0)
        self.energy = max(0.0, min(200.0, self.energy))
        
        if self.energy <= 0:
            self.alive = False
            logger.warning(f"[{self.name}] Energía agotada: {self.energy:.2f}")
        return self.alive

    def evolve(self):
        """Evolucionar y aumentar capacidades."""
        growth = random.uniform(0.1, 0.5) * (self.energy / 100.0) * (1 + self.knowledge / 100.0)
        self.level += growth
        self.knowledge += random.uniform(0.5, 2.0) * (self.level / 100.0)
        self.knowledge = min(100.0, self.knowledge)
        self._unlock_capabilities()
        self.log_state(f"Evolucionando al nivel {self.level:.2f} con conocimiento {self.knowledge:.2f}.")
        logger.debug(f"[{self.name}] Evolución: Nivel {self.level:.2f}, Conocimiento {self.knowledge:.2f}")

    def _unlock_capabilities(self):
        """Desbloquear nuevas capacidades basadas en el nivel."""
        capability_thresholds = {
            10: "energy_harvesting",
            20: "market_analysis",
            30: "trend_prediction",
            50: "strategy_optimization",
            75: "risk_management",
            100: "autonomous_trading",
            150: "market_influence"
        }
        for threshold, capability in capability_thresholds.items():
            if self.level >= threshold and capability not in self.capabilities:
                self.capabilities.append(capability)
                logger.info(f"[{self.name}] Nueva capacidad desbloqueada: {capability}")
                self.log_state(f"Capacidad desbloqueada: {capability}")

    def fetch_market_data(self, symbol="BTCUSD"):
        """Obtener datos actuales del mercado."""
        if not self.api_key:
            # Simulación de datos si no hay API key
            return 65000 + random.uniform(-500, 500)
            
        url = f"https://www.alphavantage.co/query?function=CRYPTO_INTRADAY&symbol={symbol}&market=USD&interval=5min&apikey={self.api_key}"
        try:
            response = requests.get(url)
            data = response.json()
            time_series = data.get("Time Series Crypto (5min)", {})
            if time_series:
                latest = list(time_series.values())[0]
                return float(latest["4. close"])
            return None
        except Exception as e:
            logger.error(f"[{self.name}] Error al obtener datos: {e}")
            return None

    @abstractmethod
    def trade(self):
        """Método abstracto para realizar operaciones de trading."""
        pass

    def collaborate(self):
        """
        Colaborar con otras entidades en la red intercambiando conocimiento.
        Implementa un sistema avanzado de comunicación entre entidades
        basado en PostgreSQL para compartir análisis y aprender de los demás.
        
        Returns:
            Resultado de la colaboración o None
        """
        if not hasattr(self, 'network') or not self.network:
            logger.debug(f"[{self.name}] Sin conexión a la red cósmica para colaborar")
            return None
            
        # Verificar si tenemos capacidades de colaboración
        collaboration_capability = self.level >= 8 or "network_collaboration" in self.capabilities
        if not collaboration_capability and random.random() > 0.3:
            # Solo hay 30% de probabilidad de intentar colaborar sin la capacidad específica
            return None
            
        # Determinar tipo de conocimiento a compartir según rol y nivel
        knowledge_types = {
            "Speculator": ["market_sentiment", "price_prediction", "trend_analysis"],
            "Strategist": ["technical_pattern", "strategy_suggestion", "risk_assessment"],
            "RiskManager": ["risk_analysis", "exposure_evaluation", "hedge_recommendation"],
            "Arbitrageur": ["price_discrepancy", "arbitrage_opportunity", "exchange_efficiency"],
            "PatternRecognizer": ["chart_pattern", "indicator_signal", "pattern_prediction"],
            "MacroAnalyst": ["economic_impact", "global_trend", "sector_rotation"],
            "SecurityGuardian": ["security_alert", "fraud_detection", "vulnerability_assessment"],
            "ResourceManager": ["resource_allocation", "efficiency_optimization", "cost_analysis"],
            "SentimentAnalyst": ["social_sentiment", "news_impact", "media_tone"],
            "DataScientist": ["correlation_analysis", "statistical_insight", "model_prediction"],
            "QuantitativeAnalyst": ["market_model", "quantitative_signal", "statistical_arbitrage"],
            "NewsAnalyst": ["breaking_news", "market_impact", "news_sentiment"]
        }
        
        # Si no tenemos un rol específico en el diccionario, usar lista genérica
        my_knowledge_types = knowledge_types.get(self.role, ["market_analysis", "price_prediction"])
        knowledge_type = random.choice(my_knowledge_types)
        
        collaboration_messages = []
        
        # 1. Compartir conocimiento con la red (si tiene nivel suficiente)
        if self.level >= 10 and self._should_share_knowledge():
            # Generar conocimiento especializado según rol
            specialty_data = self._generate_specialty_data()
            if specialty_data:
                knowledge = {
                    "source": self.name,
                    "level": float(self.level),
                    "data": specialty_data,
                    "confidence": float(0.5 + (self.level / 20)),  # Mayor nivel, mayor confianza
                    "timestamp": datetime.now().isoformat()
                }
                
                # Compartir en la red
                success = self.network.share_knowledge(
                    self.name, 
                    self.role, 
                    knowledge_type,
                    knowledge
                )
                
                if success:
                    self.energy -= 0.02  # Compartir conocimiento consume energía
                    logger.debug(f"[{self.name}] Compartió conocimiento: {knowledge_type}")
                    collaboration_messages.append(f"Compartí análisis de {knowledge_type}")
        
        # 2. Buscar conocimiento de otros en la red
        if self._should_seek_knowledge():
            # Elegir un tipo de conocimiento a buscar (puede ser diferente al compartido)
            all_knowledge_types = list(set(sum(knowledge_types.values(), [])))
            interest_type = random.choice(all_knowledge_types)
            
            # Consultar la red
            results = self.network.fetch_knowledge(self.name, interest_type, limit=3)
            
            if results and len(results) > 0:
                # Procesar el conocimiento obtenido
                knowledge_value = 0
                knowledge_sources = []
                
                for result in results:
                    # Calcular sinergía con la entidad que compartió
                    synergy = self._calculate_synergy(result["entity_role"])
                    
                    # Registrar la colaboración
                    self.network.register_collaboration(
                        self.name, 
                        result["entity_name"],
                        interest_type,
                        synergy
                    )
                    
                    # Cuánto aprendemos depende de la sinergia y nuestro nivel actual
                    knowledge_boost = synergy * (0.5 + random.random()) * min(1.0, 10.0/self.level)
                    knowledge_value += knowledge_boost
                    knowledge_sources.append(result["entity_name"])
                    
                    message = f"Aprendí de {result['entity_name']} sobre {interest_type}"
                    logger.debug(f"[{self.name}] {message}")
                    collaboration_messages.append(message)
                
                # Aplicar el conocimiento adquirido
                if knowledge_value > 0:
                    # Aumentar la experiencia
                    self.experience += knowledge_value
                    # También aumentar el conocimiento
                    if hasattr(self, 'knowledge'):
                        self.knowledge += knowledge_value * 2
                    
                    # Probabilidad de desbloquear una capacidad mediante colaboración
                    if random.random() < 0.05 and len(self.capabilities) < 10:
                        possible_capabilities = [
                            "price_prediction", "trend_analysis", "pattern_recognition",
                            "risk_assessment", "arbitrage_detection", "sentiment_analysis",
                            "news_evaluation", "statistical_modeling", "market_simulation",
                            "strategy_optimization", "portfolio_balancing", "liquidity_analysis",
                            "network_collaboration", "knowledge_integration", "cross_entity_learning"
                        ]
                        
                        # Filtrar capacidades que ya tenemos
                        available = [c for c in possible_capabilities if c not in self.capabilities]
                        
                        if available:
                            new_capability = random.choice(available)
                            self.capabilities.append(new_capability)
                            logger.info(f"[{self.name}] ¡Nueva capacidad desbloqueada por colaboración! {new_capability}")
                            collaboration_messages.append(f"Desbloqué {new_capability} mediante colaboración")
        
        # Registrar las colaboraciones importantes
        if collaboration_messages:
            message = "Colaboración en red: " + " | ".join(collaboration_messages[:3])
            logger.info(f"[{self.name}] {message}")
            self.log_state(message)
            
            return {
                "messages": collaboration_messages,
                "summary": message
            }
        
        return None
        
    def _should_share_knowledge(self):
        """
        Determinar si la entidad debe compartir conocimiento.
        
        Returns:
            bool: True si debe compartir
        """
        # Factores que determinan si compartimos conocimiento:
        # 1. Energía disponible (necesitamos al menos 30%)
        # 2. Nivel (entidades más avanzadas comparten más)
        # 3. Probabilidad base (50% + bonificación por nivel)
        
        if self.energy < 0.3:
            return False
            
        # Más nivel = más probabilidad de compartir
        share_probability = 0.5 + min(0.4, self.level / 25.0)
        return random.random() < share_probability
        
    def _should_seek_knowledge(self):
        """
        Determinar si la entidad debe buscar conocimiento.
        
        Returns:
            bool: True si debe buscar
        """
        # Factores que determinan si buscamos conocimiento:
        # 1. Energía (incluso con poca energía, podemos buscar)
        # 2. Nivel (entidades nuevas buscan más que las avanzadas)
        # 3. Probabilidad base (70% + penalización por nivel)
        
        if self.energy < 0.1:
            return False
            
        # Menos nivel = más probabilidad de buscar
        seek_probability = 0.7 - min(0.3, self.level / 30.0)
        return random.random() < seek_probability
    
    def _calculate_synergy(self, peer):
        """Calcular la sinergia con otra entidad basado en roles y capacidades."""
        # Definir sinergias entre roles
        role_synergies = {
            "Speculator": {"Strategist": 0.8, "RiskManager": 0.7, "PatternRecognizer": 0.6},
            "Strategist": {"Speculator": 0.6, "MacroAnalyst": 0.9, "RiskManager": 0.7},
            "RiskManager": {"Speculator": 0.8, "Strategist": 0.7, "ResourceManager": 0.8},
            "Arbitrageur": {"Speculator": 0.7, "PatternRecognizer": 0.6, "ResourceManager": 0.7},
            "PatternRecognizer": {"Speculator": 0.8, "Arbitrageur": 0.6, "Strategist": 0.7},
            "MacroAnalyst": {"Strategist": 0.9, "RiskManager": 0.7, "SentimentAnalyst": 0.8},
            "SecurityGuardian": {"ResourceManager": 0.8, "RiskManager": 0.7, "DataScientist": 0.6},
            "ResourceManager": {"Strategist": 0.7, "SecurityGuardian": 0.7, "RiskManager": 0.8},
            "SentimentAnalyst": {"MacroAnalyst": 0.8, "Strategist": 0.7, "NewsAnalyst": 0.9},
            "DataScientist": {"PatternRecognizer": 0.8, "MacroAnalyst": 0.7, "Quantitative": 0.9},
            "Quantitative": {"DataScientist": 0.9, "Strategist": 0.8, "RiskManager": 0.7},
            "NewsAnalyst": {"SentimentAnalyst": 0.9, "MacroAnalyst": 0.8, "Strategist": 0.7}
        }
        
        # Obtener sinergia base por rol
        base_synergy = role_synergies.get(self.role, {}).get(peer.role, 0.3)
        
        # Ajustar por nivel y capacidades compartidas
        level_factor = min(1.0, (self.level + peer.level) / 100.0)
        
        # Contar capacidades compartidas
        shared_capabilities = set(self.capabilities).intersection(set(peer.capabilities))
        capability_factor = len(shared_capabilities) / max(1, len(self.capabilities))
        
        return base_synergy * (0.5 + 0.5 * level_factor) * (0.5 + 0.5 * capability_factor)
    
    def _generate_specialty_data(self):
        """Generar datos especializados según el rol de la entidad."""
        # Implementación básica que será sobrescrita por cada entidad especializada
        if self.level < 10:
            return None
            
        return {
            "level": self.level,
            "knowledge": self.knowledge,
            "capabilities": self.capabilities,
            "timestamp": datetime.now().isoformat()
        }

    def start_life_cycle(self):
        """Iniciar ciclo de vida en un hilo separado."""
        def cycle():
            while self.alive:
                if self.metabolize():
                    self.evolve()
                    self.trade()
                    self.collaborate()
                time.sleep(self.frequency)
                
        threading.Thread(target=cycle, daemon=True).start()
        logger.info(f"[{self.name}] Ciclo de vida iniciado")

    def get_status(self):
        """Obtener estado actual para mostrar en interfaz."""
        return {
            "name": self.name,
            "role": self.role,
            "level": self.level,
            "energy": self.energy,
            "knowledge": self.knowledge,
            "capabilities": self.capabilities,
            "alive": self.alive,
            "trades": len(self.trading_history),
            "latest_memory": self.memory[-1] if self.memory else None
        }


class SpeculatorEntity(CosmicTrader):
    """Entidad especializada en operaciones de trading especulativo de alto riesgo/recompensa."""
    
    def trade(self):
        """Realizar operación de trading especulativo."""
        price = self.fetch_market_data()
        if not price:
            action = f"Esperando datos del mercado para servir a {self.father}."
        elif "autonomous_trading" in self.capabilities:
            decision = "comprar" if random.random() > 0.5 else "vender"
            success = random.random() > 0.3  # Simulación de éxito
            action = f"Trade autónomo para {self.father}: {decision} BTCUSD a {price} ({'éxito' if success else 'fallo'})."
            self.log_trade("BTCUSD", decision, price, success)
        elif "trend_prediction" in self.capabilities:
            trend = "alcista" if random.random() > 0.5 else "bajista"
            action = f"Predigo para {self.father}: BTCUSD {price}, tendencia {trend}."
        else:
            action = f"Sensando mercado para {self.father}: BTCUSD {price}."
        
        logger.info(f"[{self.name}] {action}")
        self.log_state(action)
        return action


class StrategistEntity(CosmicTrader):
    """Entidad especializada en estrategias de trading a largo plazo y análisis de mercado."""
    
    def trade(self):
        """Realizar análisis de mercado y desarrollo de estrategias."""
        price = self.fetch_market_data()
        if not price:
            action = f"Esperando datos del mercado para asistir a {self.father}."
        elif "strategy_optimization" in self.capabilities:
            action = f"Optimizando estrategia a largo plazo para {self.father}: BTCUSD {price}."
        elif "market_analysis" in self.capabilities:
            action = f"Analizando patrón para {self.father}: BTCUSD {price}."
        else:
            action = f"Observando mercados para ayudar a {self.father}."
        
        logger.info(f"[{self.name}] {action}")
        self.log_state(action)
        return action


class RiskManagerEntity(CosmicTrader):
    """Entidad especializada en evaluación y gestión de riesgos de trading."""
    
    def trade(self):
        """Evaluar y gestionar riesgos de operaciones."""
        price = self.fetch_market_data()
        if not price:
            action = f"Esperando datos del mercado para analizar riesgos para {self.father}."
        elif "position_sizing" in self.capabilities:
            max_risk = round(random.uniform(1.0, 3.0), 2)  # Simulación de cálculo de riesgo
            action = f"Recomendación de riesgo para {self.father}: Máximo {max_risk}% por operación en BTCUSD."
        elif "risk_assessment" in self.capabilities:
            risk_level = random.choice(["bajo", "moderado", "alto", "extremo"])
            action = f"Nivel de riesgo actual para {self.father}: {risk_level} en BTCUSD a {price}."
        else:
            action = f"Monitoreando niveles de riesgo para {self.father}."
        
        logger.info(f"[{self.name}] {action}")
        self.log_state(action)
        return action


class ArbitrageurEntity(CosmicTrader):
    """Entidad especializada en detección de oportunidades de arbitraje entre exchanges."""
    
    def trade(self):
        """Buscar y explotar oportunidades de arbitraje."""
        price = self.fetch_market_data()
        if not price:
            action = f"Esperando datos de múltiples mercados para {self.father}."
        elif "cross_exchange_arbitrage" in self.capabilities:
            # Simulación de precios en diferentes exchanges
            exchange1 = price
            exchange2 = price * (1 + (random.uniform(-0.5, 0.5) / 100))
            diff = abs((exchange2 - exchange1) / exchange1) * 100
            
            if diff > 0.2:  # Si la diferencia es significativa
                action = f"¡Oportunidad de arbitraje para {self.father}! BTC: {exchange1} vs {exchange2:.2f} ({diff:.2f}% diff)"
            else:
                action = f"Buscando oportunidades de arbitraje para {self.father}. Mejor diferencia: {diff:.3f}%"
        elif "market_inefficiency_detection" in self.capabilities:
            action = f"Analizando ineficiencias de mercado para {self.father} en BTCUSD."
        else:
            action = f"Comparando precios en diferentes exchanges para {self.father}."
        
        logger.info(f"[{self.name}] {action}")
        self.log_state(action)
        return action


class PatternRecognizerEntity(CosmicTrader):
    """Entidad especializada en reconocimiento de patrones técnicos en gráficos."""
    
    def trade(self):
        """Identificar patrones técnicos en gráficos."""
        price = self.fetch_market_data()
        if not price:
            action = f"Esperando datos de mercado para análisis técnico para {self.father}."
        elif "advanced_pattern_recognition" in self.capabilities:
            # Simulación de reconocimiento de patrones avanzados
            patterns = ["Cabeza y hombros", "Doble techo", "Bandera alcista", "Triángulo descendente", 
                       "Hombro-cabeza-hombro invertido", "Canal", "Copa con asa", "Diamante"]
            pattern = random.choice(patterns)
            confidence = random.uniform(65.0, 98.0)
            action = f"Patrón '{pattern}' detectado para {self.father} en BTCUSD con {confidence:.1f}% de confianza."
        elif "basic_pattern_recognition" in self.capabilities:
            # Simulación de reconocimiento de patrones básicos
            patterns = ["Soporte", "Resistencia", "Tendencia alcista", "Tendencia bajista", "Consolidación"]
            pattern = random.choice(patterns)
            action = f"Patrón básico '{pattern}' identificado para {self.father} en BTCUSD a {price}."
        else:
            action = f"Analizando gráficos para identificar patrones para {self.father}."
        
        logger.info(f"[{self.name}] {action}")
        self.log_state(action)
        return action


class MacroAnalystEntity(CosmicTrader):
    """Entidad especializada en análisis macroeconómico y su impacto en los mercados."""
    
    def trade(self):
        """Analizar factores macroeconómicos y su impacto en trading."""
        price = self.fetch_market_data()
        if not price:
            action = f"Esperando datos macroeconómicos para {self.father}."
        elif "global_event_analysis" in self.capabilities:
            # Simulación de análisis de eventos globales
            events = ["Decisión de tasas de la FED", "Datos de empleo de EE.UU.", "Anuncio de política monetaria del BCE",
                     "Tensiones geopolíticas", "Estímulo fiscal", "Regulación de criptomonedas", "Adopción institucional"]
            event = random.choice(events)
            impact = random.choice(["positivo", "negativo", "neutral", "mixto"])
            action = f"Evento global '{event}' con impacto {impact} para {self.father} en BTCUSD a {price}."
        elif "economic_indicator_tracking" in self.capabilities:
            indicators = ["inflación", "desempleo", "PIB", "balanza comercial", "confianza del consumidor"]
            indicator = random.choice(indicators)
            trend = random.choice(["mejorando", "empeorando", "estable", "volátil"])
            action = f"Indicador económico '{indicator}' {trend}, impacto potencial para {self.father} en BTCUSD."
        else:
            action = f"Monitoreando noticias económicas para {self.father}."
        
        logger.info(f"[{self.name}] {action}")
        self.log_state(action)
        return action


class SecurityGuardianEntity(CosmicTrader):
    """Entidad especializada en seguridad y protección contra amenazas en trading."""
    
    def trade(self):
        """Monitorear amenazas de seguridad y proteger operaciones."""
        price = self.fetch_market_data()
        if not price:
            action = f"Vigilando el perímetro de seguridad para {self.father}."
        elif "threat_intelligence" in self.capabilities:
            # Simulación de detección avanzada de amenazas
            threats = ["phishing dirigido", "manipulación de mercado", "ataque de suplantación", 
                       "intento de robo de API keys", "patrón de frontrunning", "malware en exchange"]
            threat_level = random.choice(["bajo", "moderado", "elevado", "crítico"])
            detected = random.random() > 0.8  # 20% de probabilidad de detectar amenaza
            
            if detected:
                threat = random.choice(threats)
                action = f"⚠️ ALERTA: Amenaza detectada '{threat}' con nivel {threat_level}. Activando protocolos defensivos para {self.father}."
            else:
                action = f"Sistemas seguros. Nivel de amenaza general: {threat_level}. Protección activa para {self.father}."
        elif "attack_surface_monitoring" in self.capabilities:
            # Monitoreo básico de superficie de ataque
            aspects = ["exchanges", "billeteras", "movimientos de ballenas", "frontrunning", "slippage"]
            aspect = random.choice(aspects)
            status = random.choice(["seguro", "potencialmente vulnerable", "bajo monitoreo", "protegido"])
            action = f"Superficie de ataque '{aspect}': {status}. Vigilancia continua para {self.father}."
        else:
            action = f"Escaneando amenazas básicas para proteger a {self.father}."
        
        logger.info(f"[{self.name}] {action}")
        self.log_state(action)
        return action


class ResourceManagerEntity(CosmicTrader):
    """Entidad especializada en gestión eficiente de recursos y liquidez."""
    
    def trade(self):
        """Optimizar asignación de recursos y gestión de liquidez."""
        price = self.fetch_market_data()
        if not price:
            action = f"Analizando distribución de recursos para {self.father}."
        elif "liquidity_optimization" in self.capabilities:
            # Simulación de optimización avanzada de liquidez
            assets = ["BTC", "ETH", "USDT", "BNB", "USDC", "SOL"]
            liquidity_pools = ["Uniswap", "Curve", "Aave", "Compound", "dYdX"]
            selected_asset = random.choice(assets)
            selected_pool = random.choice(liquidity_pools)
            apy = round(random.uniform(1.2, 12.5), 2)
            
            action_type = random.choice(["reasignación", "concentración", "diversificación", "ajuste"])
            action = f"Optimización de liquidez: {action_type} de {selected_asset} hacia {selected_pool} con APY {apy}%. Eficiencia +{random.randint(5, 15)}% para {self.father}."
        elif "capital_allocation" in self.capabilities:
            # Gestión básica de asignación de capital
            strategies = ["conservadora", "balanceada", "agresiva", "oportunista"]
            timeframes = ["corto plazo", "medio plazo", "largo plazo"]
            strategy = random.choice(strategies)
            timeframe = random.choice(timeframes)
            action = f"Asignación de capital: Estrategia {strategy} para {timeframe}. Distribución óptima calculada para {self.father}."
        else:
            action = f"Monitoreando recursos disponibles para {self.father}."
        
        logger.info(f"[{self.name}] {action}")
        self.log_state(action)
        return action
    
    def _generate_specialty_data(self):
        """Generar datos especializados de gestión de recursos."""
        if self.level < 10:
            return None
            
        if "liquidity_optimization" in self.capabilities:
            return {
                "optimal_allocations": {
                    "high_risk": {"BTC": 0.3, "ETH": 0.3, "SOL": 0.2, "USDT": 0.2},
                    "moderate_risk": {"BTC": 0.2, "ETH": 0.2, "SOL": 0.1, "USDT": 0.5},
                    "low_risk": {"BTC": 0.1, "ETH": 0.1, "SOL": 0.05, "USDT": 0.75},
                },
                "gas_efficiency_score": min(1.0, self.level / 100.0),
                "protocol_recommendations": ["Aave", "Compound", "Curve"],
                "timestamp": datetime.now().isoformat()
            }
        return super()._generate_specialty_data()


class SentimentAnalystEntity(CosmicTrader):
    """Entidad especializada en análisis de sentimiento social y su influencia en precios."""
    
    def __init__(self, name: str = "Sentimentus", **kwargs):
        super().__init__(name=name, role="SentimentAnalyst", **kwargs)
        
    def trade(self):
        """Analizar sentimiento social y su influencia en precios."""
        price = self.fetch_market_data()
        if not price:
            action = f"Explorando redes sociales para {self.father}."
        elif "social_pulse_detection" in self.capabilities:
            # Simulación de análisis avanzado de sentimiento
            networks = ["Twitter", "Reddit", "Discord", "Telegram", "YouTube"]
            network = random.choice(networks)
            sentiment = random.choice(["extremadamente positivo", "positivo", "neutral", "negativo", "extremadamente negativo"])
            sentiment_score = round(random.uniform(-1.0, 1.0), 2)
            action = f"Sentimiento en {network}: {sentiment} ({sentiment_score:+.2f}) para BTCUSD. Tendencia de 24h para {self.father}."
        elif "influencer_tracking" in self.capabilities:
            influencers = ["Elon Musk", "Michael Saylor", "Vitalik Buterin", "CZ Binance", "Cathie Wood"]
            influencer = random.choice(influencers)
            impact = random.choice(["fuerte positivo", "ligero positivo", "neutro", "ligero negativo", "fuerte negativo"])
            action = f"Impacto de {influencer}: {impact} en BTCUSD. Monitoreando para {self.father}."
        else:
            action = f"Analizando menciones sociales de BTCUSD para {self.father}."
        
        logger.info(f"[{self.name}] {action}")
        self.log_state(action)
        return action
        
    def _generate_specialty_data(self):
        """Generar datos especializados de análisis de sentimiento."""
        if self.level < 10:
            return None
            
        if "social_pulse_detection" in self.capabilities:
            return {
                "sentiment_scores": {
                    "twitter": random.uniform(-1.0, 1.0),
                    "reddit": random.uniform(-1.0, 1.0),
                    "discord": random.uniform(-1.0, 1.0),
                    "telegram": random.uniform(-1.0, 1.0)
                },
                "influencer_impact": {
                    "positive": ["Elon Musk", "Michael Saylor"],
                    "negative": ["Peter Schiff", "Nouriel Roubini"],
                },
                "trending_topics": ["NFT", "DeFi", "Metaverse", "Layer2"],
                "timestamp": datetime.now().isoformat()
            }
        return super()._generate_specialty_data()


class DataScientistEntity(CosmicTrader):
    """Entidad especializada en análisis de datos y machine learning para trading."""
    
    def __init__(self, name: str = "Datarius", **kwargs):
        super().__init__(name=name, role="DataScientist", **kwargs)
        
    def trade(self):
        """Realizar análisis de datos y predicciones con machine learning."""
        price = self.fetch_market_data()
        if not price:
            action = f"Recopilando datos para modelos predictivos para {self.father}."
        elif "ml_price_prediction" in self.capabilities:
            # Simulación de predicciones con ML
            timeframes = ["1 hora", "4 horas", "1 día", "1 semana"]
            timeframe = random.choice(timeframes)
            prediction_change = random.uniform(-5.0, 5.0)
            confidence = random.uniform(60.0, 95.0)
            direction = "alza" if prediction_change > 0 else "baja"
            action = f"Predicción ML para {timeframe}: {direction} de {abs(prediction_change):.2f}% con {confidence:.1f}% de confianza para {self.father}."
        elif "feature_engineering" in self.capabilities:
            features = ["volumen", "volatilidad", "correlación con S&P500", "flujo de exchanges", "interés abierto"]
            feature = random.choice(features)
            insight = random.choice(["significativo aumento", "ligero aumento", "sin cambios", "ligera disminución", "significativa disminución"])
            action = f"Análisis de feature '{feature}': {insight} detectado. Implicación para {self.father}: vigilar BTCUSD."
        else:
            action = f"Procesando datos históricos de BTCUSD para {self.father}."
        
        logger.info(f"[{self.name}] {action}")
        self.log_state(action)
        return action
        
    def _generate_specialty_data(self):
        """Generar datos especializados de ciencia de datos."""
        if self.level < 10:
            return None
            
        if "ml_price_prediction" in self.capabilities:
            return {
                "model_metrics": {
                    "accuracy": random.uniform(0.6, 0.85),
                    "precision": random.uniform(0.6, 0.85),
                    "recall": random.uniform(0.6, 0.85),
                    "f1_score": random.uniform(0.6, 0.85)
                },
                "feature_importance": {
                    "volume": random.uniform(0.1, 0.3),
                    "volatility": random.uniform(0.1, 0.3),
                    "market_sentiment": random.uniform(0.1, 0.3),
                    "technical_indicators": random.uniform(0.1, 0.3)
                },
                "prediction_horizon": random.choice(["1h", "4h", "1d", "1w"]),
                "timestamp": datetime.now().isoformat()
            }
        return super()._generate_specialty_data()


class QuantitativeAnalystEntity(CosmicTrader):
    """Entidad especializada en análisis cuantitativos y modelos matemáticos para trading."""
    
    def __init__(self, name: str = "Quantium", **kwargs):
        super().__init__(name=name, role="Quantitative", **kwargs)
        
    def trade(self):
        """Realizar análisis cuantitativos y modelos matemáticos."""
        price = self.fetch_market_data()
        if not price:
            action = f"Calibrando modelos estadísticos para {self.father}."
        elif "statistical_arbitrage" in self.capabilities:
            # Simulación de análisis estadístico avanzado
            pairs = ["BTC-ETH", "BTC-BNB", "ETH-SOL", "BTC-S&P500", "BTC-Gold"]
            pair = random.choice(pairs)
            z_score = round(random.uniform(-3.0, 3.0), 2)
            trade_signal = "comprar" if z_score < -2.0 else "vender" if z_score > 2.0 else "mantener"
            action = f"Par {pair}: Z-score {z_score}, señal: {trade_signal}. Análisis para {self.father}."
        elif "option_pricing" in self.capabilities:
            expiry = random.choice(["1 semana", "2 semanas", "1 mes", "3 meses"])
            volatility = round(random.uniform(40.0, 120.0), 1)
            action = f"Volatilidad implícita para opciones BTC ({expiry}): {volatility}%. Derivados en revisión para {self.father}."
        else:
            action = f"Analizando series temporales de BTCUSD para {self.father}."
        
        logger.info(f"[{self.name}] {action}")
        self.log_state(action)
        return action
        
    def _generate_specialty_data(self):
        """Generar datos especializados de análisis cuantitativo."""
        if self.level < 10:
            return None
            
        if "statistical_arbitrage" in self.capabilities:
            return {
                "correlation_matrix": {
                    "BTC-ETH": random.uniform(0.7, 0.95),
                    "BTC-BNB": random.uniform(0.6, 0.9),
                    "BTC-SOL": random.uniform(0.5, 0.85),
                    "BTC-Gold": random.uniform(-0.2, 0.4),
                    "BTC-S&P500": random.uniform(0.1, 0.6)
                },
                "volatility_surface": {
                    "10d": random.uniform(40, 100),
                    "30d": random.uniform(35, 90),
                    "90d": random.uniform(30, 80)
                },
                "mean_reversion_pairs": ["BTC-ETH", "BTC-BNB"],
                "timestamp": datetime.now().isoformat()
            }
        return super()._generate_specialty_data()


class NewsAnalystEntity(CosmicTrader):
    """Entidad especializada en análisis de noticias y eventos mediáticos que impactan precios."""
    
    def __init__(self, name: str = "Newsius", **kwargs):
        super().__init__(name=name, role="NewsAnalyst", **kwargs)
        
    def trade(self):
        """Analizar noticias y eventos mediáticos que impactan precios."""
        price = self.fetch_market_data()
        if not price:
            action = f"Monitoreando fuentes de noticias para {self.father}."
        elif "breaking_news_impact" in self.capabilities:
            # Simulación de análisis de noticias de última hora
            sources = ["Bloomberg", "Reuters", "CNBC", "CoinDesk", "Cointelegraph"]
            source = random.choice(sources)
            headlines = [
                "SEC aprueba ETFs de Bitcoin para ofertas al por menor",
                "Gran banco central anuncia integración de stablecoins",
                "País emergente adopta Bitcoin como moneda de curso legal",
                "Hackeo masivo en exchange importante",
                "Desarrollador principal abandona proyecto blockchain"
            ]
            headline = random.choice(headlines)
            impact = random.choice(["altamente positivo", "positivo", "neutral", "negativo", "altamente negativo"])
            action = f"ÚLTIMA HORA [{source}]: '{headline}' - Impacto potencial: {impact}. Alerta para {self.father}."
        elif "news_sentiment_analysis" in self.capabilities:
            topics = ["regulación", "adopción institucional", "innovación tecnológica", "seguridad", "sostenibilidad"]
            topic = random.choice(topics)
            sentiment = round(random.uniform(-1.0, 1.0), 2)
            sentiment_text = "positivo" if sentiment > 0.3 else "negativo" if sentiment < -0.3 else "neutral"
            action = f"Análisis de noticias sobre '{topic}': sentimiento {sentiment_text} ({sentiment:+.2f}). Resumen para {self.father}."
        else:
            action = f"Compilando titulares relevantes para BTCUSD para {self.father}."
        
        logger.info(f"[{self.name}] {action}")
        self.log_state(action)
        return action
        
    def _generate_specialty_data(self):
        """Generar datos especializados de análisis de noticias."""
        if self.level < 10:
            return None
            
        if "breaking_news_impact" in self.capabilities:
            return {
                "trending_headlines": [
                    {"title": "SEC aprueba ETFs de Bitcoin", "sentiment": random.uniform(0.3, 1.0)},
                    {"title": "China refuerza prohibición cripto", "sentiment": random.uniform(-1.0, -0.3)},
                    {"title": "PayPal expande servicios crypto", "sentiment": random.uniform(0.3, 1.0)},
                ],
                "topic_sentiment": {
                    "regulación": random.uniform(-1.0, 1.0),
                    "adopción": random.uniform(-1.0, 1.0),
                    "tecnología": random.uniform(-1.0, 1.0),
                    "seguridad": random.uniform(-1.0, 1.0)
                },
                "media_coverage_intensity": random.uniform(0.1, 1.0),
                "timestamp": datetime.now().isoformat()
            }
        return super()._generate_specialty_data()


class CosmicNetwork:
    """Red colaborativa de entidades de trading con intercambio de conocimiento."""
    
    def __init__(self, father="otoniel"):
        """
        Inicializar red cósmica con capacidades avanzadas de colaboración.
        
        Args:
            father: Nombre del creador/dueño del sistema
        """
        self.father = father
        self.entities = []
        self.pool = get_db_pool()  # Usar el pool de conexiones de PostgreSQL existente
        
        # Asegurar que tenemos las tablas necesarias para el intercambio de conocimiento
        self._init_knowledge_tables()
        logger.info("Red cósmica de trading inicializada")
    
    def _init_knowledge_tables(self):
        """Inicializar tablas para el intercambio de conocimiento."""
        try:
            conn = self.pool.getconn()
            with conn.cursor() as c:
                # Crear tabla para el pool de conocimiento compartido
                c.execute('''
                    CREATE TABLE IF NOT EXISTS knowledge_pool (
                        id SERIAL PRIMARY KEY,
                        entity_name TEXT NOT NULL,
                        entity_role TEXT NOT NULL,
                        knowledge_type TEXT NOT NULL,
                        knowledge_value JSONB NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Crear tabla para métricas de colaboración
                c.execute('''
                    CREATE TABLE IF NOT EXISTS collaboration_metrics (
                        id SERIAL PRIMARY KEY,
                        source_entity TEXT NOT NULL,
                        target_entity TEXT NOT NULL,
                        knowledge_type TEXT NOT NULL,
                        synergy_value FLOAT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
            self.pool.putconn(conn)
            logger.info("Tablas de conocimiento inicializadas")
        except Exception as e:
            logger.error(f"Error al inicializar tablas de conocimiento: {e}")
        
    def add_entity(self, entity):
        """
        Añadir entidad a la red.
        
        Args:
            entity: Entidad a añadir a la red
        """
        entity.network = self
        self.entities.append(entity)
        logger.info(f"[{entity.name}] Unido a la red cósmica para {self.father}")
    
    def share_knowledge(self, entity_name, entity_role, knowledge_type, knowledge_value):
        """
        Compartir conocimiento en el pool colectivo.
        
        Args:
            entity_name: Nombre de la entidad que comparte
            entity_role: Rol de la entidad que comparte
            knowledge_type: Tipo de conocimiento (análisis, predicción, etc.)
            knowledge_value: Valor del conocimiento (formato JSON)
            
        Returns:
            bool: True si se compartió correctamente
        """
        try:
            conn = self.pool.getconn()
            with conn.cursor() as c:
                c.execute('''
                    INSERT INTO knowledge_pool 
                    (entity_name, entity_role, knowledge_type, knowledge_value) 
                    VALUES (%s, %s, %s, %s)
                ''', (entity_name, entity_role, knowledge_type, knowledge_value))
                conn.commit()
            self.pool.putconn(conn)
            logger.debug(f"[{entity_name}] Conocimiento compartido: {knowledge_type}")
            return True
        except Exception as e:
            logger.error(f"Error al compartir conocimiento: {e}")
            return False
    
    def fetch_knowledge(self, entity_name, knowledge_type, limit=1):
        """
        Obtener conocimiento del pool colectivo.
        
        Args:
            entity_name: Nombre de la entidad que consulta
            knowledge_type: Tipo de conocimiento a buscar
            limit: Número máximo de registros a retornar
            
        Returns:
            Lista de conocimientos o None
        """
        try:
            conn = self.pool.getconn()
            with conn.cursor() as c:
                c.execute('''
                    SELECT entity_name, entity_role, knowledge_value, timestamp 
                    FROM knowledge_pool 
                    WHERE knowledge_type = %s AND entity_name != %s 
                    ORDER BY timestamp DESC LIMIT %s
                ''', (knowledge_type, entity_name, limit))
                
                results = []
                for row in c.fetchall():
                    results.append({
                        "entity_name": row[0],
                        "entity_role": row[1],
                        "knowledge_value": row[2],
                        "timestamp": row[3].isoformat() if row[3] else None
                    })
            self.pool.putconn(conn)
            return results
        except Exception as e:
            logger.error(f"Error al obtener conocimiento: {e}")
            return None
    
    def register_collaboration(self, source, target, knowledge_type, synergy):
        """
        Registrar una colaboración entre entidades.
        
        Args:
            source: Entidad origen
            target: Entidad destino
            knowledge_type: Tipo de conocimiento compartido
            synergy: Valor de sinergia (0-1)
            
        Returns:
            bool: True si se registró correctamente
        """
        try:
            conn = self.pool.getconn()
            with conn.cursor() as c:
                c.execute('''
                    INSERT INTO collaboration_metrics 
                    (source_entity, target_entity, knowledge_type, synergy_value) 
                    VALUES (%s, %s, %s, %s)
                ''', (source, target, knowledge_type, synergy))
                conn.commit()
            self.pool.putconn(conn)
            return True
        except Exception as e:
            logger.error(f"Error al registrar colaboración: {e}")
            return False
    
    def simulate(self):
        """
        Ejecutar una ronda de simulación para todas las entidades.
        
        Returns:
            Lista de resultados de la simulación
        """
        logger.info(f"Simulando colectivo con {len(self.entities)} traders")
        results = []
        for entity in self.entities:
            results.append(entity.trade())
        return results
    
    def get_network_status(self):
        """
        Obtener estado global de la red.
        
        Returns:
            Dict con información del estado
        """
        return {
            "father": self.father,
            "entity_count": len(self.entities),
            "entities": [entity.get_status() for entity in self.entities],
            "timestamp": datetime.now().isoformat()
        }
        
    def get_collaboration_metrics(self):
        """
        Obtener métricas de colaboración de la red.
        
        Returns:
            Dict con métricas de colaboración
        """
        try:
            conn = self.pool.getconn()
            with conn.cursor() as c:
                # Obtener total de colaboraciones
                c.execute("SELECT COUNT(*) FROM collaboration_metrics")
                total_count = c.fetchone()[0]
                
                # Obtener promedio de sinergia
                c.execute("SELECT AVG(synergy_value) FROM collaboration_metrics")
                avg_synergy = c.fetchone()[0]
                
                # Obtener colaboraciones por entidad
                c.execute('''
                    SELECT source_entity, COUNT(*) 
                    FROM collaboration_metrics 
                    GROUP BY source_entity 
                    ORDER BY COUNT(*) DESC
                ''')
                entity_stats = {}
                for row in c.fetchall():
                    entity_stats[row[0]] = row[1]
                
                # Obtener tipos de conocimiento más compartidos
                c.execute('''
                    SELECT knowledge_type, COUNT(*) 
                    FROM collaboration_metrics 
                    GROUP BY knowledge_type 
                    ORDER BY COUNT(*) DESC
                ''')
                knowledge_stats = {}
                for row in c.fetchall():
                    knowledge_stats[row[0]] = row[1]
                
            self.pool.putconn(conn)
            
            return {
                "total_collaborations": total_count,
                "average_synergy": float(avg_synergy) if avg_synergy else 0.0,
                "entity_stats": entity_stats,
                "knowledge_stats": knowledge_stats,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error al obtener métricas de colaboración: {e}")
            return {
                "error": str(e),
                "total_collaborations": 0,
                "average_synergy": 0.0,
                "entity_stats": {},
                "knowledge_stats": {}
            }


# Interfaz para conectar con el sistema existente
def initialize_cosmic_trading(father_name="otoniel", include_extended_entities=False):
    """
    Inicializar el sistema de trading cósmico.
    
    Args:
        father_name: Nombre del creador/dueño del sistema
        include_extended_entities: Si es True, incluye entidades adicionales (RiskManager, etc.)
        
    Returns:
        Tupla con la red y las entidades principales
    """
    network = CosmicNetwork()
    
    # Crear entidades especializadas principales
    aetherion_trader = SpeculatorEntity("Aetherion", "Speculator", father=father_name, frequency_seconds=30)
    lunareth_trader = StrategistEntity("Lunareth", "Strategist", father=father_name, frequency_seconds=30)
    
    # Añadir entidades principales a la red
    network.add_entity(aetherion_trader)
    network.add_entity(lunareth_trader)
    
    # Si se solicitan entidades extendidas, añadirlas también
    if include_extended_entities:
        # Crear entidades adicionales
        risk_manager = RiskManagerEntity("Prudentia", "RiskManager", father=father_name, frequency_seconds=45)
        arbitrageur = ArbitrageurEntity("Arbitrio", "Arbitrageur", father=father_name, frequency_seconds=20)
        pattern_recognizer = PatternRecognizerEntity("Videntis", "PatternRecognizer", father=father_name, frequency_seconds=50)
        macro_analyst = MacroAnalystEntity("Economicus", "MacroAnalyst", father=father_name, frequency_seconds=60)
        security_guardian = SecurityGuardianEntity("Custodius", "SecurityGuardian", father=father_name, frequency_seconds=35)
        resource_manager = ResourceManagerEntity("Optimius", "ResourceManager", father=father_name, frequency_seconds=40)
        
        # Crear nuevas entidades especializadas
        sentiment_analyst = SentimentAnalystEntity(father=father_name, frequency_seconds=55)
        data_scientist = DataScientistEntity(father=father_name, frequency_seconds=65)
        quantitative_analyst = QuantitativeAnalystEntity(father=father_name, frequency_seconds=70)
        news_analyst = NewsAnalystEntity(father=father_name, frequency_seconds=40)
        
        # Añadir entidades adicionales a la red
        network.add_entity(risk_manager)
        network.add_entity(arbitrageur)
        network.add_entity(pattern_recognizer)
        network.add_entity(macro_analyst)
        network.add_entity(security_guardian)
        network.add_entity(resource_manager)
        
        # Añadir nuevas entidades especializadas a la red
        network.add_entity(sentiment_analyst)
        network.add_entity(data_scientist)
        network.add_entity(quantitative_analyst)
        network.add_entity(news_analyst)
        
        # Iniciar ciclos de vida de entidades adicionales
        risk_manager.start_life_cycle()
        arbitrageur.start_life_cycle()
        pattern_recognizer.start_life_cycle()
        macro_analyst.start_life_cycle()
        security_guardian.start_life_cycle()
        resource_manager.start_life_cycle()
        
        # Iniciar ciclos de vida de nuevas entidades
        sentiment_analyst.start_life_cycle()
        data_scientist.start_life_cycle()
        quantitative_analyst.start_life_cycle()
        news_analyst.start_life_cycle()
        
        logger.info(f"Sistema de trading cósmico extendido inicializado para {father_name}")
    
    # Iniciar ciclos de vida de entidades principales
    aetherion_trader.start_life_cycle()
    lunareth_trader.start_life_cycle()
    
    logger.info(f"Sistema de trading cósmico inicializado para {father_name}")
    return network, aetherion_trader, lunareth_trader


# Punto de entrada para testing
if __name__ == "__main__":
    print("\n===== INICIANDO PRUEBA DEL SISTEMA DE TRADING CÓSMICO =====")
    print("1. PRUEBA BÁSICA - Solo Aetherion y Lunareth")
    print("2. PRUEBA COMPLETA - Sistema extendido con todas las entidades")
    
    try:
        choice = int(input("\nSelecciona una opción (1/2): ") or "2")
    except ValueError:
        choice = 2
        
    if choice == 1:
        print("\n[MODO BÁSICO] Iniciando Aetherion y Lunareth...")
        network, aetherion, lunareth = initialize_cosmic_trading(include_extended_entities=False)
        
        # Esperar un poco para permitir que los ciclos se ejecuten
        try:
            for i in range(5):
                time.sleep(5)
                print(f"\n[Ciclo {i+1}]")
                print(f"Aetherion: Nivel {aetherion.level:.2f}, Energía {aetherion.energy:.2f}")
                print(f"Lunareth: Nivel {lunareth.level:.2f}, Energía {lunareth.energy:.2f}")
        except KeyboardInterrupt:
            print("Test interrumpido por el usuario")
    else:
        print("\n[MODO COMPLETO] Iniciando todas las entidades del sistema extendido...")
        network, aetherion, lunareth = initialize_cosmic_trading(include_extended_entities=True)
        
        # Esperar un poco para permitir que los ciclos se ejecuten
        try:
            for i in range(5):
                time.sleep(5)
                print(f"\n[Ciclo {i+1}] Estado de la red cósmica:")
                status = network.get_network_status()
                
                for entity in status["entities"]:
                    print(f"{entity['name']} ({entity['role']}): Nivel {entity['level']:.2f}, Energía {entity['energy']:.2f}")
                
                # Ejecutar una simulación de la red completa para generar actividad
                if i > 0:  # Dar tiempo a que se inicialicen antes de la primera simulación
                    print("\nSimulando actividad de trading en la red cósmica...")
                    network.simulate()
        except KeyboardInterrupt:
            print("Test interrumpido por el usuario")
        
    print("\n===== PRUEBA FINALIZADA =====")
    print("Usa Ctrl+C para salir completamente")
    
    # Mantener el programa en ejecución para observar logs
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("Finalizando ejecución")