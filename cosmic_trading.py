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
import sqlite3
import logging
import requests
import threading
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import deque
from abc import ABC, abstractmethod

# Configuración de logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        """Inicializar tablas en la base de datos."""
        conn = sqlite3.connect("cosmic_trading.db")
        c = conn.cursor()
        c.execute(f'''CREATE TABLE IF NOT EXISTS {self.name}_life (
            timestamp TEXT, level REAL, energy REAL, knowledge REAL, capabilities TEXT, log TEXT
        )''')
        c.execute(f'''CREATE TABLE IF NOT EXISTS {self.name}_trades (
            timestamp TEXT, symbol TEXT, action TEXT, price REAL, success INTEGER
        )''')
        conn.commit()
        conn.close()
        logger.info(f"[{self.name}] Base de datos inicializada")

    def log_state(self, log_message):
        """Registrar estado actual en la base de datos."""
        timestamp = datetime.now().isoformat()
        capabilities_str = ",".join(self.capabilities)
        conn = sqlite3.connect("cosmic_trading.db")
        c = conn.cursor()
        c.execute(f"INSERT INTO {self.name}_life VALUES (?, ?, ?, ?, ?, ?)",
                  (timestamp, self.level, self.energy, self.knowledge, capabilities_str, log_message))
        conn.commit()
        conn.close()
        self.memory.append({
            "timestamp": timestamp, 
            "level": self.level, 
            "energy": self.energy, 
            "knowledge": self.knowledge,
            "log": log_message
        })
        logger.debug(f"[{self.name}] Estado registrado: {log_message}")

    def log_trade(self, symbol, action, price, success):
        """Registrar operación de trading."""
        timestamp = datetime.now().isoformat()
        conn = sqlite3.connect("cosmic_trading.db")
        c = conn.cursor()
        c.execute(f"INSERT INTO {self.name}_trades VALUES (?, ?, ?, ?, ?)",
                  (timestamp, symbol, action, price, 1 if success else 0))
        conn.commit()
        conn.close()
        self.trading_history.append({
            "timestamp": timestamp,
            "symbol": symbol, 
            "action": action, 
            "price": price, 
            "success": success
        })
        logger.info(f"[{self.name}] Trade registrado: {action} {symbol} a {price}")

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
        """Colaborar con otras entidades en la red."""
        if self.network and "strategy_optimization" in self.capabilities:
            for peer in self.network.entities:
                if peer != self and "market_analysis" in peer.capabilities:
                    self.knowledge += random.uniform(0.1, 1.0)
                    peer.knowledge += random.uniform(0.1, 1.0)
                    logger.info(f"[{self.name}] Colaborando con {peer.name}")
                    self.log_state(f"Colaboración con {peer.name} para mejorar conocimiento mutuo.")

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


class CosmicNetwork:
    """Red colaborativa de entidades de trading."""
    
    def __init__(self):
        """Inicializar red cósmica."""
        self.entities = []
        logger.info("Red cósmica de trading inicializada")
        
    def add_entity(self, entity):
        """Añadir entidad a la red."""
        entity.network = self
        self.entities.append(entity)
        logger.info(f"[{entity.name}] Unido a la red cósmica para {entity.father}")
        
    def simulate(self):
        """Ejecutar una ronda de simulación para todas las entidades."""
        logger.info(f"Simulando colectivo con {len(self.entities)} traders")
        results = []
        for entity in self.entities:
            results.append(entity.trade())
        return results
        
    def get_network_status(self):
        """Obtener estado global de la red."""
        return {
            "entity_count": len(self.entities),
            "entities": [entity.get_status() for entity in self.entities]
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
        
        # Añadir entidades adicionales a la red
        network.add_entity(risk_manager)
        network.add_entity(arbitrageur)
        network.add_entity(pattern_recognizer)
        network.add_entity(macro_analyst)
        network.add_entity(security_guardian)
        network.add_entity(resource_manager)
        
        # Iniciar ciclos de vida de entidades adicionales
        risk_manager.start_life_cycle()
        arbitrageur.start_life_cycle()
        pattern_recognizer.start_life_cycle()
        macro_analyst.start_life_cycle()
        security_guardian.start_life_cycle()
        resource_manager.start_life_cycle()
        
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