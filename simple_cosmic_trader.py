"""
Sistema de trading cósmico simplificado.

Versión simplificada del sistema de trading cósmico con capacidades esenciales:
- Simulación de precios de mercado
- Sistema de colaboración básico
- Diferentes entidades especializadas
- Ciclo de vida con evolución automática
"""

import os
import random
import time
import json
import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from collections import deque

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simple_cosmic_trader")

class CosmicTrader(ABC):
    """Clase base para entidades cósmicas de trading con vida simulada."""
    
    def __init__(self, name, role, father="otoniel", energy_rate=0.1, frequency_seconds=15):
        """Inicializar trader cósmico con capacidades básicas."""
        # Atributos básicos
        self.name = name
        self.role = role
        self.father = father
        self.energy_rate = energy_rate
        self.frequency_seconds = frequency_seconds
        
        # Estado interno
        self.alive = True
        self.energy = 100.0
        self.level = 1.0
        self.knowledge = 0.0
        self.experience = 0.0
        self.capabilities = []
        self.memory = []
        self.trading_history = []
        self.network = None
        
        # Datos de mercado simulados
        self.price_history = deque(maxlen=100)
        self.last_price = None
        
        # Generar algunos precios iniciales simulados
        for _ in range(20):
            self.price_history.append(random.uniform(60000, 70000))
        self.last_price = self.price_history[-1]
        
        logger.info(f"[{self.name}] Inicialización completa como {self.role}")
    
    def fetch_market_data(self, symbol="BTCUSD"):
        """
        Obtener datos actuales del mercado (simulados).
        
        Args:
            symbol: Símbolo del activo
            
        Returns:
            Último precio simulado
        """
        # Generar un pequeño cambio basado en el último precio
        if self.last_price:
            change = self.last_price * random.uniform(-0.005, 0.005)
            new_price = self.last_price + change
            self.last_price = max(50000, min(80000, new_price))  # Mantener en rango
            self.price_history.append(self.last_price)
        else:
            self.last_price = random.uniform(60000, 70000)
            self.price_history.append(self.last_price)
            
        return self.last_price
    
    def predict_price(self):
        """
        Predecir precio futuro usando método simple.
        
        Returns:
            Predicción de precio simulada
        """
        if len(self.price_history) < 5:
            return None
            
        # Método simple: tendencia basada en últimos precios
        data = list(self.price_history)[-5:]
        last_price = data[-1]
        avg_price = sum(data) / len(data)
        
        # Tendencia simple
        if last_price > avg_price:
            return last_price * (1 + random.uniform(0.001, 0.005))
        else:
            return last_price * (1 - random.uniform(0.001, 0.005))
    
    def metabolize(self):
        """
        Gestionar ciclo de energía vital.
        
        Returns:
            bool: True si la entidad sigue con energía, False si se quedó sin energía
        """
        # Consumir energía (afectado por factores aleatorios)
        energy_change = -self.energy_rate * random.uniform(0.8, 1.2)
        
        # Modificadores basados en capacidades
        if "energy_harvesting" in self.capabilities:
            # Recuperar energía basado en nivel de conocimiento
            energy_change += 0.05 * (self.knowledge / 10.0)
        
        if "efficiency_optimization" in self.capabilities:
            # Reducir consumo de energía
            energy_change *= 0.8
            
        # Aplicar cambio de energía
        self.energy = max(0.0, min(200.0, self.energy + energy_change))
        
        # Si nos quedamos sin energía, pero estamos vivos, regenerar un poco
        if self.energy <= 0 and self.alive:
            self.energy = 10.0
            logger.warning(f"[{self.name}] Regeneración de emergencia: +10 energía")
            
        return self.energy > 0
    
    def evolve(self):
        """Evolucionar y aumentar capacidades."""
        # Convertir experiencia en conocimiento
        if self.experience > 0:
            knowledge_gain = self.experience * 0.2
            self.knowledge += knowledge_gain
            self.experience = 0
            
            # Limitar conocimiento máximo
            self.knowledge = min(200.0, self.knowledge)
        
        # Ganar algo de experiencia pasiva
        base_exp_gain = random.uniform(0.01, 0.05)
        
        # Modificadores por capacidades
        if "accelerated_learning" in self.capabilities:
            base_exp_gain *= 1.5
        
        # Aplicar ganancia de experiencia
        self.experience += base_exp_gain
        
        # Calcular nuevo nivel
        old_level = int(self.level)
        self.level = 1.0 + self.knowledge / 10.0
        
        # Si aumentamos de nivel, desbloquear capacidades
        if int(self.level) > old_level:
            self._unlock_capabilities()
            logger.info(f"[{self.name}] ¡Subió al nivel {int(self.level)}!")
    
    def _unlock_capabilities(self):
        """Desbloquear nuevas capacidades basadas en el nivel actual."""
        # Mapeo de capacidades por nivel
        capability_thresholds = {
            2: ["basic_analysis"],
            5: ["energy_harvesting"],
            8: ["price_prediction"],
            10: ["market_analysis"],
            12: ["network_collaboration"],
            15: ["advanced_pattern_recognition"],
            18: ["trend_prediction"],
            20: ["risk_assessment"],
            25: ["strategy_optimization"],
            30: ["accelerated_learning"],
            35: ["efficiency_optimization"],
            40: ["advanced_collaboration"],
            50: ["market_influence"],
            60: ["autonomous_trading"],
            75: ["system_synergy"],
            100: ["market_mastery"]
        }
        
        # Verificar capacidades a desbloquear
        current_level = int(self.level)
        for level, capabilities in capability_thresholds.items():
            if current_level >= level:
                for capability in capabilities:
                    if capability not in self.capabilities:
                        self.capabilities.append(capability)
                        logger.info(f"[{self.name}] ¡Nueva capacidad desbloqueada! {capability}")
    
    @abstractmethod
    def trade(self):
        """
        Método abstracto para realizar operaciones de trading.
        Debe ser implementado por las subclases especializadas.
        
        Returns:
            str: Mensaje descriptivo de la operación
        """
        pass
    
    def collaborate(self, peers=None):
        """
        Colaborar con otras entidades (versión simplificada).
        
        Args:
            peers: Lista de entidades con las que colaborar
            
        Returns:
            dict: Resultado de la colaboración
        """
        if not peers:
            return None
            
        knowledge_gained = 0
        messages = []
        
        # Intercambiar conocimientos con pares
        for peer in peers:
            synergy = random.uniform(0.1, 0.8)  # Simular sinergia
            knowledge_boost = synergy * min(5.0, peer.knowledge / 20.0)
            knowledge_gained += knowledge_boost
            
            message = f"Colaboración con {peer.name} ({peer.role}), ganancia: {knowledge_boost:.2f}"
            messages.append(message)
            
        # Aplicar ganancia de conocimiento
        if knowledge_gained > 0:
            self.experience += knowledge_gained
            logger.info(f"[{self.name}] {messages[0]}")
        
        return {
            "knowledge_gained": knowledge_gained,
            "messages": messages
        }
    
    def start_life_cycle(self):
        """Iniciar ciclo de vida en un hilo separado."""
        self.alive = True
        
        def cycle():
            while self.alive:
                try:
                    if self.metabolize():
                        self.evolve()
                        self.trade()
                    time.sleep(self.frequency_seconds)
                except Exception as e:
                    logger.error(f"[{self.name}] Error en ciclo de vida: {e}")
                    # No detenemos el bucle, continuamos con la siguiente iteración
                
        threading.Thread(target=cycle, daemon=True).start()
        logger.info(f"[{self.name}] Ciclo de vida iniciado")
    
    def get_status(self):
        """
        Obtener estado actual para mostrar en interfaz.
        
        Returns:
            dict: Información del estado
        """
        return {
            "name": self.name,
            "role": self.role,
            "level": float(self.level),
            "energy": float(self.energy),
            "knowledge": float(self.knowledge),
            "experience": float(self.experience),
            "capabilities": self.capabilities,
            "alive": self.alive,
            "trades": len(self.trading_history),
            "price": self.last_price,
            "timestamp": datetime.now().isoformat()
        }

class SimpleSpeculatorEntity(CosmicTrader):
    """Entidad especializada en trading especulativo."""
    
    def trade(self):
        """
        Realizar operación de trading especulativo.
        
        Returns:
            str: Mensaje descriptivo de la operación
        """
        price = self.fetch_market_data()
        if "autonomous_trading" in self.capabilities:
            # Predecir precio futuro
            predicted_price = self.predict_price()
            if predicted_price:
                # Tomar decisión basada en predicción
                current_trend = "alcista" if predicted_price > price else "bajista"
                decision = "comprar" if current_trend == "alcista" else "vender"
                # Simular resultado (mayor probabilidad de éxito con nivel alto)
                success_prob = 0.5 + min(0.4, self.level / 25.0)
                success = random.random() < success_prob
                
                # Registrar trade
                self.trading_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "symbol": "BTCUSD",
                    "action": decision,
                    "price": price,
                    "success": success
                })
                
                action = f"Trade autónomo para {self.father}: {decision} BTCUSD a {price:.2f} " \
                        f"con predicción {predicted_price:.2f} ({current_trend})."
            else:
                action = f"Esperando suficientes datos para predicción para {self.father}."
        elif "trend_prediction" in self.capabilities:
            # Análisis básico de tendencia
            trend = "alcista" if random.random() > 0.5 else "bajista"
            confidence = round(random.uniform(60, 95), 1)
            action = f"Predicción para {self.father}: BTCUSD {price:.2f}, tendencia {trend} " \
                     f"con {confidence}% de confianza."
        else:
            action = f"Sensando mercado para {self.father}: BTCUSD {price:.2f}."
        
        logger.info(f"[{self.name}] {action}")
        return action

class SimpleStrategistEntity(CosmicTrader):
    """Entidad especializada en estrategias a largo plazo y análisis de mercado."""
    
    def trade(self):
        """
        Realizar análisis de mercado y desarrollo de estrategias.
        
        Returns:
            str: Mensaje descriptivo del análisis
        """
        price = self.fetch_market_data()
        if "strategy_optimization" in self.capabilities:
            # Estrategia avanzada con múltiples factores
            market_state = random.choice(["alcista", "bajista", "lateral", "volátil"])
            timeframe = random.choice(["corto", "medio", "largo"])
            suggestion = random.choice([
                "compras escalonadas", "ventas parciales", 
                "hodl estratégico", "trading con tendencia", 
                "arbitraje entre exchanges"
            ])
            action = f"Estrategia optimizada para {self.father}: En mercado {market_state}, " \
                     f"plazo {timeframe}, {suggestion} para BTCUSD a {price:.2f}."
        elif "market_analysis" in self.capabilities:
            # Análisis básico
            pattern = random.choice(["soporte", "resistencia", "triángulo", "cabeza y hombros"])
            action = f"Análisis técnico para {self.father}: Formación de {pattern} " \
                     f"detectada en BTCUSD a {price:.2f}."
        else:
            action = f"Observando patrón de mercado para {self.father}: BTCUSD {price:.2f}."
        
        logger.info(f"[{self.name}] {action}")
        return action

class SimpleCosmicNetwork:
    """Red simplificada de entidades cósmicas."""
    
    def __init__(self, father="otoniel"):
        """Inicializar red cósmica simplificada."""
        self.father = father
        self.entities = []
        self.knowledge_pool = 0.0
        logger.info(f"Red cósmica simplificada inicializada para {father}")
    
    def add_entity(self, entity):
        """
        Añadir entidad a la red.
        
        Args:
            entity: Entidad a añadir
        """
        entity.network = self
        self.entities.append(entity)
        entity.start_life_cycle()
        logger.info(f"[{entity.name}] añadido a la red cósmica simplificada")
    
    def simulate_collaboration(self):
        """
        Ejecutar una ronda de colaboración entre entidades.
        
        Returns:
            Lista de resultados de colaboración
        """
        results = []
        
        for entity in self.entities:
            if entity.alive:
                # Seleccionar pares aleatorios para colaborar (al menos una entidad)
                peers = [e for e in self.entities if e != entity and e.alive]
                if peers:
                    collab_count = min(len(peers), random.randint(1, 3))
                    selected_peers = random.sample(peers, collab_count)
                    
                    result = entity.collaborate(selected_peers)
                    if result:
                        results.append({
                            "entity": entity.name,
                            "knowledge_gained": result["knowledge_gained"],
                            "message": result["messages"][0] if result["messages"] else "Colaboración realizada"
                        })
        
        # Aumentar el conocimiento colectivo
        self.knowledge_pool += sum(r["knowledge_gained"] for r in results) * 0.1
        
        return results
    
    def simulate(self):
        """
        Ejecutar una ronda de simulación para todas las entidades.
        
        Returns:
            Lista de resultados de simulación
        """
        logger.info(f"Simulando colectivo con {len(self.entities)} traders")
        results = []
        
        for entity in self.entities:
            if entity.alive:
                try:
                    result = entity.trade()
                    results.append({
                        "entity": entity.name,
                        "result": result
                    })
                except Exception as e:
                    logger.error(f"Error en simulación de {entity.name}: {e}")
                    results.append({
                        "entity": entity.name,
                        "error": str(e)
                    })
        
        return results
    
    def get_network_status(self):
        """
        Obtener estado global de la red.
        
        Returns:
            dict: Información del estado
        """
        return {
            "father": self.father,
            "entity_count": len(self.entities),
            "entities": [entity.get_status() for entity in self.entities if entity.alive],
            "knowledge_pool": float(self.knowledge_pool),
            "timestamp": datetime.now().isoformat()
        }

def initialize_simple_trading(father_name="otoniel"):
    """
    Inicializar el sistema de trading cósmico simplificado.
    
    Args:
        father_name: Nombre del creador/dueño del sistema
        
    Returns:
        Tupla (red, aetherion_trader, lunareth_trader)
    """
    network = SimpleCosmicNetwork(father=father_name)
    
    # Crear entidades especializadas principales
    aetherion_trader = SimpleSpeculatorEntity("Aetherion", "Speculator", father=father_name, frequency_seconds=30)
    lunareth_trader = SimpleStrategistEntity("Lunareth", "Strategist", father=father_name, frequency_seconds=30)
    
    # Añadir entidades principales a la red
    network.add_entity(aetherion_trader)
    network.add_entity(lunareth_trader)
    
    logger.info(f"Sistema de trading cósmico simplificado inicializado para {father_name}")
    return network, aetherion_trader, lunareth_trader

# Para testing directo del módulo
if __name__ == "__main__":
    print("\n===== INICIANDO PRUEBA DEL SISTEMA SIMPLIFICADO =====")
    print("Inicializando entidades principales (Aetherion y Lunareth)...")
    
    try:
        # Inicializar sistema básico
        network, aetherion, lunareth = initialize_simple_trading()
        
        # Ejecutar simulación por 60 segundos
        print("\nSimulando actividad del sistema por 60 segundos...")
        
        for i in range(6):
            time.sleep(10)
            print(f"\n[Ciclo {i+1}]")
            print(f"Aetherion: Nivel {aetherion.level:.2f}, Energía {aetherion.energy:.2f}")
            print(f"Lunareth: Nivel {lunareth.level:.2f}, Energía {lunareth.energy:.2f}")
            
            if i >= 2:  # Después de 30 segundos, simular colaboración
                print("\nSimulando colaboración en la red...")
                results = network.simulate_collaboration()
                for result in results:
                    print(f"{result['entity']}: {result['message']}")
                
                # Solicitar análisis
                price = aetherion.fetch_market_data()
                print(f"\nAnálisis de mercado: BTCUSD {price:.2f}")
                print(f"Aetherion: {aetherion.trade()}")
                print(f"Lunareth: {lunareth.trade()}")
    except KeyboardInterrupt:
        print("\nPrueba interrumpida por el usuario")
    except Exception as e:
        print(f"\nError en la prueba: {e}")
    
    print("\n===== PRUEBA FINALIZADA =====")