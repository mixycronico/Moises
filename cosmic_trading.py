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
def initialize_cosmic_trading(father_name="otoniel"):
    """
    Inicializar el sistema de trading cósmico.
    
    Args:
        father_name: Nombre del creador/dueño del sistema
        
    Returns:
        Tupla con la red y las entidades creadas
    """
    network = CosmicNetwork()
    
    # Crear entidades especializadas
    aetherion_trader = SpeculatorEntity("Aetherion", "Speculator", father=father_name, frequency_seconds=30)
    lunareth_trader = StrategistEntity("Lunareth", "Strategist", father=father_name, frequency_seconds=30)
    
    # Añadir a la red
    network.add_entity(aetherion_trader)
    network.add_entity(lunareth_trader)
    
    # Iniciar ciclos de vida
    aetherion_trader.start_life_cycle()
    lunareth_trader.start_life_cycle()
    
    logger.info(f"Sistema de trading cósmico inicializado para {father_name}")
    return network, aetherion_trader, lunareth_trader


# Punto de entrada para testing
if __name__ == "__main__":
    # Test simple del sistema
    network, aetherion, lunareth = initialize_cosmic_trading()
    
    # Esperar un poco para permitir que los ciclos se ejecuten
    try:
        for _ in range(5):
            time.sleep(5)
            print(f"\nEstado de Aetherion: Nivel {aetherion.level:.2f}, Energía {aetherion.energy:.2f}")
            print(f"Estado de Lunareth: Nivel {lunareth.level:.2f}, Energía {lunareth.energy:.2f}")
    except KeyboardInterrupt:
        print("Test interrumpido por el usuario")