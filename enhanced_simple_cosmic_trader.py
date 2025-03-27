"""
Sistema de trading cósmico mejorado con características avanzadas.

Esta versión mantiene la compatibilidad con entornos restringidos,
pero incluye características avanzadas como:
- Evolución de personalidad y rasgos
- Lenguaje propio que evoluciona
- Simulación de conexiones a datos reales
- Modelo de predicción mejorado
- Persistencia de estado
"""

import os
import random
import time
import json
import logging
import threading
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from collections import deque

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("enhanced_cosmic_trader")

# Base de datos SQLite para persistencia
DB_PATH = "cosmic_trading.db"

def init_database():
    """Inicializar la base de datos SQLite con las tablas necesarias."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Tabla de entidades
    c.execute('''
        CREATE TABLE IF NOT EXISTS cosmic_entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            role TEXT NOT NULL,
            level REAL DEFAULT 1.0,
            energy REAL DEFAULT 100.0,
            knowledge REAL DEFAULT 0.0,
            experience REAL DEFAULT 0.0,
            traits TEXT,
            emotion TEXT,
            evolution_path TEXT,
            family_role TEXT,
            capabilities TEXT,
            vocabulary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Tabla de registros (logs)
    c.execute('''
        CREATE TABLE IF NOT EXISTS entity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER,
            log_type TEXT,
            message TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (entity_id) REFERENCES cosmic_entities (id)
        )
    ''')
    
    # Tabla de operaciones de trading
    c.execute('''
        CREATE TABLE IF NOT EXISTS trade_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER,
            symbol TEXT,
            action TEXT,
            price REAL,
            success INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (entity_id) REFERENCES cosmic_entities (id)
        )
    ''')
    
    # Tabla de mensajes entre entidades
    c.execute('''
        CREATE TABLE IF NOT EXISTS entity_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_id INTEGER,
            receiver_id INTEGER,
            message TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (sender_id) REFERENCES cosmic_entities (id),
            FOREIGN KEY (receiver_id) REFERENCES cosmic_entities (id)
        )
    ''')
    
    # Tabla para conocimiento colectivo
    c.execute('''
        CREATE TABLE IF NOT EXISTS collective_knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            knowledge_type TEXT,
            content TEXT,
            entity_id INTEGER,
            value REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (entity_id) REFERENCES cosmic_entities (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Base de datos SQLite inicializada")

# Asegurar que la base de datos exista
init_database()

class EnhancedCosmicTrader(ABC):
    """
    Clase base mejorada para entidades cósmicas de trading con vida simulada
    y características avanzadas como personalidad, evolución y lenguaje propio.
    """
    
    def __init__(self, name, role, father="otoniel", 
                 energy_rate=0.1, frequency_seconds=15):
        """
        Inicializar trader cósmico con capacidades avanzadas.
        
        Args:
            name: Nombre de la entidad
            role: Rol especializado ("Speculator", "Strategist", etc.)
            father: El creador/dueño del sistema
            energy_rate: Tasa de consumo de energía
            frequency_seconds: Frecuencia de ciclo de vida en segundos
        """
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
        self.entity_id = None
        
        # Características de personalidad
        self.traits = random.choice(["Curiosidad", "Prudencia", "Audacia", "Reflexión"])
        self.emotion = "Nacimiento"
        self.evolution_path = "Semilla"
        self.family_role = "Hijo"
        
        # Lenguaje propio
        self.vocabulary = {"luz": "energía", "sombra": "cautela", "viento": "mercado"}
        
        # Datos de mercado simulados
        self.price_history = deque(maxlen=100)
        self.last_price = None
        
        # Generar algunos precios iniciales simulados
        for _ in range(20):
            self.price_history.append(random.uniform(60000, 70000))
        self.last_price = self.price_history[-1]
        
        # Inicializar en la base de datos
        self.init_db()
        
        logger.info(f"[{self.name}] He nacido como {self.role} con esencia {self.traits} para servir a {self.father}")
    
    def init_db(self):
        """Inicializar o actualizar la entidad en la base de datos."""
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Verificar si la entidad ya existe
        c.execute("SELECT id FROM cosmic_entities WHERE name = ?", (self.name,))
        result = c.fetchone()
        
        if result:
            # Actualizar entidad existente
            self.entity_id = result[0]
            c.execute("""
                UPDATE cosmic_entities 
                SET role = ?, level = ?, energy = ?, knowledge = ?, experience = ?,
                    traits = ?, emotion = ?, evolution_path = ?, family_role = ?,
                    capabilities = ?, vocabulary = ?
                WHERE id = ?
            """, (
                self.role, self.level, self.energy, self.knowledge, self.experience,
                self.traits, self.emotion, self.evolution_path, self.family_role,
                json.dumps(self.capabilities), json.dumps(self.vocabulary),
                self.entity_id
            ))
        else:
            # Crear nueva entidad
            c.execute("""
                INSERT INTO cosmic_entities 
                (name, role, level, energy, knowledge, experience, traits, emotion, 
                evolution_path, family_role, capabilities, vocabulary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.name, self.role, self.level, self.energy, self.knowledge, self.experience,
                self.traits, self.emotion, self.evolution_path, self.family_role,
                json.dumps(self.capabilities), json.dumps(self.vocabulary)
            ))
            self.entity_id = c.lastrowid
        
        conn.commit()
        conn.close()
        
        self.log_state(f"Inicialización completa como {self.role} para {self.father}")
    
    def fetch_market_data(self, symbol="BTCUSD"):
        """
        Obtener datos actuales del mercado (simulados con realismo).
        
        Args:
            symbol: Símbolo del activo
            
        Returns:
            Último precio simulado
        """
        # Simular latencia de red
        time.sleep(random.uniform(0.01, 0.05))
        
        # Generar un cambio basado en patrones realistas
        if self.last_price:
            # Incorporar volatilidad realista y tendencias
            base_noise = random.normalvariate(0, 1) * self.last_price * 0.002  # Ruido gaussiano
            trend_component = random.uniform(-0.001, 0.001) * self.last_price  # Tendencia suave
            
            # Ocasionalmente (1% de las veces) generar un movimiento más grande (simulando noticias)
            news_event = random.random() < 0.01
            news_impact = random.normalvariate(0, 1) * self.last_price * 0.01 if news_event else 0
            
            new_price = self.last_price + base_noise + trend_component + news_impact
            self.last_price = max(50000, min(80000, new_price))  # Mantener en rango
            self.price_history.append(self.last_price)
            
            # Registrar evento importante si ocurrió
            if news_event:
                self.log_state(f"Evento importante detectado en el mercado de {symbol}, impacto: {news_impact:.2f}")
        else:
            self.last_price = random.uniform(60000, 70000)
            self.price_history.append(self.last_price)
            
        return self.last_price
    
    def predict_price(self):
        """
        Predecir precio futuro usando método avanzado.
        
        Returns:
            Predicción de precio simulada
        """
        if len(self.price_history) < 5:
            return None
            
        # Método más sofisticado basado en tendencias y niveles
        data = list(self.price_history)[-20:]
        
        # Calcular media móvil simple
        sma_5 = sum(data[-5:]) / 5
        sma_10 = sum(data[-10:]) / 10 if len(data) >= 10 else sma_5
        sma_20 = sum(data[-20:]) / 20 if len(data) == 20 else sma_10
        
        # Calcular medias móviles exponenciales simuladas
        ema_12 = sum(data[-12:]) / 12 * 1.1 if len(data) >= 12 else sma_5
        ema_26 = sum(data[-26:]) / 26 * 0.9 if len(data) >= 20 else sma_10
        
        # Señal basada en cruce de medias
        trend_signal = 1 if ema_12 > ema_26 else -1 if ema_12 < ema_26 else 0
        
        # Últimos precios
        last_price = data[-1]
        
        # Calcular soporte y resistencia simulados
        high = max(data)
        low = min(data)
        resistance = high
        support = low
        
        # Distancia a soporte/resistencia
        dist_to_resistance = resistance - last_price
        dist_to_support = last_price - support
        
        # Factor de nivel de precios
        level_factor = 0
        if dist_to_support < dist_to_resistance * 0.5:
            # Cerca del soporte, probable rebote
            level_factor = 1
        elif dist_to_resistance < dist_to_support * 0.5:
            # Cerca de resistencia, probable caída
            level_factor = -1
            
        # Fuerza de tendencia
        trend_strength = random.uniform(0.8, 1.2)
        
        # Añadir componente de "intuición" basado en rasgos de personalidad
        intuition = 0
        if self.traits == "Audacia":
            intuition = random.uniform(0.001, 0.004) * trend_signal
        elif self.traits == "Prudencia":
            intuition = random.uniform(0.0005, 0.002) * -trend_signal
        elif self.traits == "Reflexión":
            intuition = random.uniform(-0.001, 0.001)
        elif self.traits == "Curiosidad":
            intuition = random.uniform(-0.003, 0.003)
            
        # Calcular predicción final
        base_change = trend_signal * 0.002 * trend_strength
        level_impact = level_factor * 0.001
        predicted_change = base_change + level_impact + intuition
        
        return last_price * (1 + predicted_change)
    
    def metabolize(self):
        """
        Gestionar ciclo de energía vital con factores de personalidad.
        
        Returns:
            bool: True si la entidad sigue con energía, False si se quedó sin energía
        """
        # Consumir energía (afectado por factores de personalidad)
        baseline_consumption = self.energy_rate * random.uniform(0.8, 1.2)
        
        # Modificador por rasgo
        trait_modifier = {
            "Curiosidad": 1.2,  # Consume más energía por explorar
            "Prudencia": 0.8,   # Consume menos por ser conservador
            "Audacia": 1.1,     # Consume un poco más por ser arriesgado
            "Reflexión": 0.9    # Consume menos por ser metódico
        }.get(self.traits, 1.0)
        
        # Modificador por emoción
        emotion_modifier = 1.0
        if self.emotion in ["Ambición", "Euforia"]:
            emotion_modifier = 1.2
        elif self.emotion in ["Cautela", "Calma"]:
            emotion_modifier = 0.8
        
        # Consumo final de energía
        energy_change = -baseline_consumption * trait_modifier * emotion_modifier
        
        # Modificadores basados en capacidades
        if "energy_harvesting" in self.capabilities:
            # Recuperar energía basado en nivel de conocimiento
            energy_change += 0.05 * (self.knowledge / 10.0)
        
        if "efficiency_optimization" in self.capabilities:
            # Reducir consumo de energía
            energy_change *= 0.8
            
        # Aplicar cambio de energía
        self.energy = max(0.0, min(200.0, self.energy + energy_change))
        
        # Actualizar emoción basada en nivel de energía
        if self.energy > 150:
            self.emotion = "Ambición"
        elif self.energy < 20:
            self.emotion = "Cautela"
            # Solicitar ayuda a la red
            if self.network:
                self.network.request_help(self, "luz")
        
        # Si nos quedamos sin energía, pero estamos vivos, regenerar un poco
        if self.energy <= 0 and self.alive:
            self.energy = 10.0
            self.emotion = "Renacimiento"
            logger.warning(f"[{self.name}] Regeneración de emergencia: +10 energía")
            
        # Actualizar estado en base de datos
        self.update_state()
            
        return self.energy > 0
    
    def evolve(self):
        """
        Evolucionar y aumentar capacidades con sistema de personalidad y evolución.
        """
        # Obtener conocimiento de experiencia (influenciado por rasgos)
        if self.experience > 0:
            trait_knowledge_modifier = {
                "Curiosidad": 1.3,  # Aprende más rápido
                "Prudencia": 0.9,   # Aprende más lento pero seguro
                "Audacia": 1.1,     # Aprende un poco más rápido
                "Reflexión": 1.2    # Buen aprendizaje por análisis
            }.get(self.traits, 1.0)
            
            knowledge_gain = self.experience * 0.2 * trait_knowledge_modifier
            self.knowledge += knowledge_gain
            self.experience = 0
            
            # Limitar conocimiento máximo
            self.knowledge = min(200.0, self.knowledge)
            
            # Registrar ganancia importante
            if knowledge_gain > 1.0:
                self.log_state(f"Gran aumento de conocimiento: +{knowledge_gain:.2f}")
        
        # Ganar algo de experiencia pasiva (influenciada por personalidad)
        base_exp_gain = random.uniform(0.01, 0.05)
        
        # Modificador por rasgo
        trait_exp_modifier = {
            "Curiosidad": 1.2,  # Gana más experiencia por explorar
            "Prudencia": 0.9,   # Gana menos experiencia por ser conservador
            "Audacia": 1.1,     # Gana un poco más por ser arriesgado
            "Reflexión": 1.0    # Ganancia estándar
        }.get(self.traits, 1.0)
        
        # Modificador por red (si está en una)
        network_modifier = 1.0
        if self.network and hasattr(self.network, 'knowledge_pool'):
            network_modifier = min(2.0, 1.0 + self.network.knowledge_pool / 1000.0)
        
        # Aplicar ganancia de experiencia
        exp_gain = base_exp_gain * trait_exp_modifier * network_modifier
        self.experience += exp_gain
        
        # Calcular nuevo nivel
        old_level = int(self.level)
        self.level = 1.0 + self.knowledge / 10.0
        
        # Si aumentamos de nivel, evolucionar y desbloquear capacidades
        if int(self.level) > old_level:
            self._evolve_path()
            self._unlock_capabilities()
            logger.info(f"[{self.name}] ¡Subió al nivel {int(self.level)}!")
            
            # Emociones positivas al subir de nivel
            self.emotion = random.choice(["Euforia", "Inspiración", "Determinación"])
            
            # Anunciar a la red
            if self.network:
                self.network.broadcast(
                    self.name, 
                    self.generate_message("luz", f"nivel {self.level:.2f}")
                )
        
        # Actualizar estado en base de datos
        self.update_state()
    
    def _evolve_path(self):
        """Evolucionar en el camino espiritual basado en nivel y rasgos."""
        evolution_paths = {
            "Semilla": {"next": "Explorador", "threshold": 10},
            "Explorador": {
                "next": "Guerrero del Riesgo" if self.traits == "Audacia" else "Oráculo del Mercado", 
                "threshold": 50
            },
            "Guerrero del Riesgo": {"next": "Titán Cósmico", "threshold": 100},
            "Oráculo del Mercado": {"next": "Vidente Eterno", "threshold": 100},
            "Titán Cósmico": {"next": None, "threshold": None},
            "Vidente Eterno": {"next": None, "threshold": None}
        }
        
        current = evolution_paths[self.evolution_path]
        if self.level >= current["threshold"] and current["next"]:
            new_path = current["next"]
            family_role = "Anciano" if new_path in ["Titán Cósmico", "Vidente Eterno"] else "Hermano"
            
            # Actualizar camino y rol
            self.evolution_path = new_path
            self.family_role = family_role
            
            # Registrar evolución
            self.log_state(f"He evolucionado a {new_path} ahora soy {family_role} para {self.father}")
            logger.info(f"[{self.name}] Ha evolucionado a {new_path} como {family_role} para {self.father}")
            
            # Actualizar en base de datos
            self.update_state()
    
    def _unlock_capabilities(self):
        """Desbloquear nuevas capacidades basadas en el nivel y rasgos."""
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
                        self.log_state(f"Nueva capacidad desbloqueada: {capability}")
                        logger.info(f"[{self.name}] ¡Nueva capacidad desbloqueada! {capability}")
        
        # Capacidades especiales basadas en rasgos (chance pequeña)
        if random.random() < 0.1:
            trait_capabilities = {
                "Curiosidad": ["vision_distante", "creatividad_conceptual"],
                "Prudencia": ["analisis_riesgo_avanzado", "anticipacion_peligro"],
                "Audacia": ["voluntad_hierro", "valor_extremo"],
                "Reflexión": ["meditacion_profunda", "pensamiento_lateral"]
            }
            
            potential_capabilities = trait_capabilities.get(self.traits, [])
            if potential_capabilities:
                special_cap = random.choice(potential_capabilities)
                if special_cap not in self.capabilities:
                    self.capabilities.append(special_cap)
                    self.log_state(f"¡Capacidad especial desbloqueada! {special_cap}")
                    logger.info(f"[{self.name}] ¡Capacidad especial desbloqueada! {special_cap}")
    
    def generate_message(self, base_word, context):
        """
        Generar mensaje usando el lenguaje propio que evoluciona.
        
        Args:
            base_word: Palabra base para el mensaje
            context: Contexto para la generación
            
        Returns:
            Mensaje generado
        """
        # Posibilidad de crear nueva palabra
        new_word = base_word
        if random.random() < 0.2:  # 20% de chance de crear palabra nueva
            new_word = f"{base_word}{random.randint(1, 10)}"
        
        # Añadir palabra al vocabulario si es nueva
        if new_word not in self.vocabulary:
            self.vocabulary[new_word] = f"{self.emotion}_{context[:5]}"
            self.update_state()
        
        return f"{self.family_role} {self.name} canta: {new_word} {context}"
    
    def update_state(self):
        """Actualizar estado de la entidad en la base de datos."""
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute("""
            UPDATE cosmic_entities 
            SET level = ?, energy = ?, knowledge = ?, experience = ?,
                traits = ?, emotion = ?, evolution_path = ?, family_role = ?,
                capabilities = ?, vocabulary = ?
            WHERE id = ?
        """, (
            self.level, self.energy, self.knowledge, self.experience,
            self.traits, self.emotion, self.evolution_path, self.family_role,
            json.dumps(self.capabilities), json.dumps(self.vocabulary),
            self.entity_id
        ))
        
        conn.commit()
        conn.close()
    
    def log_state(self, log_message):
        """
        Registrar estado en la base de datos.
        
        Args:
            log_message: Mensaje descriptivo
        """
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute("""
            INSERT INTO entity_logs (entity_id, log_type, message)
            VALUES (?, ?, ?)
        """, (self.entity_id, "state", log_message))
        
        conn.commit()
        conn.close()
        
        # Mantener historial local
        self.memory.append({
            "timestamp": datetime.now().isoformat(),
            "message": log_message,
            "energy": self.energy,
            "level": self.level,
            "knowledge": self.knowledge
        })
        if len(self.memory) > 100:
            self.memory = self.memory[-100:]  # Mantener solo últimos 100 registros
    
    def log_trade(self, symbol, action, price, success):
        """
        Registrar operación de trading en la base de datos.
        
        Args:
            symbol: Símbolo del activo operado
            action: Acción realizada (comprar, vender)
            price: Precio de la operación
            success: Si fue exitosa o no
        """
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute("""
            INSERT INTO trade_history (entity_id, symbol, action, price, success)
            VALUES (?, ?, ?, ?, ?)
        """, (self.entity_id, symbol, action, price, 1 if success else 0))
        
        conn.commit()
        conn.close()
        
        # Mantener historia local para rendimiento
        timestamp = datetime.now().isoformat()
        self.trading_history.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "action": action,
            "price": price,
            "success": success
        })
        if len(self.trading_history) > 100:
            self.trading_history = self.trading_history[-100:]
    
    def receive_message(self, sender, message):
        """
        Recibir mensaje de otra entidad.
        
        Args:
            sender: Nombre de la entidad emisora
            message: Contenido del mensaje
        """
        # Registrar el mensaje recibido
        self.log_state(f"Mensaje recibido de {sender}: {message}")
        
        # Posible cambio de emoción basado en mensaje
        if "luz" in message and random.random() < 0.3:
            self.emotion = random.choice(["Inspiración", "Esperanza", "Armonía"])
        elif "sombra" in message and random.random() < 0.3:
            self.emotion = random.choice(["Cautela", "Inquietud", "Contemplación"])
        
        # Registrar en base de datos si conocemos el ID del emisor
        if self.network:
            sender_entity = next((e for e in self.network.entities if e.name == sender), None)
            if sender_entity and sender_entity.entity_id:
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                
                c.execute("""
                    INSERT INTO entity_messages (sender_id, receiver_id, message)
                    VALUES (?, ?, ?)
                """, (sender_entity.entity_id, self.entity_id, message))
                
                conn.commit()
                conn.close()
    
    @abstractmethod
    def trade(self):
        """
        Método abstracto para realizar operaciones de trading.
        Debe ser implementado por las subclases especializadas.
        
        Returns:
            str: Mensaje descriptivo de la operación
        """
        return ""  # Implementación base vacía
    
    def collaborate(self, peers=None):
        """
        Colaborar con otras entidades en la red.
        
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
            # Calcular sinergia basada en roles complementarios y evolución
            base_synergy = random.uniform(0.1, 0.8)
            
            # Bonificación por roles complementarios
            role_synergy = 0.2 if (
                (self.role == "Speculator" and peer.role == "RiskManager") or
                (self.role == "Strategist" and peer.role == "PatternRecognizer") or
                (self.role == "Arbitrageur" and peer.role == "MacroAnalyst")
            ) else 0.0
            
            # Bonificación por nivel similar (mejor colaboración)
            level_diff = abs(self.level - peer.level)
            level_synergy = 0.2 if level_diff < 5 else 0.1 if level_diff < 10 else 0.0
            
            # Bonificación por rasgo complementario
            trait_synergy = 0.1 if (
                (self.traits == "Audacia" and peer.traits == "Prudencia") or
                (self.traits == "Curiosidad" and peer.traits == "Reflexión")
            ) else 0.0
            
            # Sinergia total
            synergy = base_synergy + role_synergy + level_synergy + trait_synergy
            
            # Ganancia de conocimiento
            knowledge_boost = synergy * min(5.0, peer.knowledge / 20.0)
            knowledge_gained += knowledge_boost
            
            # Efecto en emoción
            if random.random() < 0.3:
                if knowledge_boost > 1.0:
                    self.emotion = random.choice(["Inspiración", "Euforia", "Admiración"])
                elif knowledge_boost > 0.5:
                    self.emotion = random.choice(["Interés", "Curiosidad", "Alegría"])
            
            # Mensaje sobre la colaboración
            message = f"Colaboración con {peer.name} ({peer.role}, {peer.evolution_path}), " \
                      f"sinergia: {synergy:.2f}, ganancia: {knowledge_boost:.2f}"
            messages.append(message)
            
            # Generar mensaje en lenguaje propio
            cosmic_message = self.generate_message("luz", f"colaborar {peer.name}")
            
            # Enviar mensaje al peer
            peer.receive_message(self.name, cosmic_message)
            
        # Aplicar ganancia de conocimiento
        if knowledge_gained > 0:
            self.experience += knowledge_gained
            self.log_state(f"Experiencia ganada por colaboración: +{knowledge_gained:.2f}")
        
        # Actualizar estado en base de datos
        self.update_state()
        
        return {
            "knowledge_gained": knowledge_gained,
            "messages": messages,
            "summary": messages[0] if messages else "Colaboración realizada"
        }
    
    def start_life_cycle(self):
        """Iniciar ciclo de vida en un hilo separado."""
        self.alive = True
        
        def cycle():
            while self.alive:
                try:
                    if self.metabolize():
                        self.evolve()
                        result = self.trade()
                        
                        # Interacción periódica con la red (cantar)
                        if random.random() < 0.3 and self.network:  # 30% de probabilidad
                            self.sing_in_chorus()
                            
                    time.sleep(self.frequency_seconds)
                except Exception as e:
                    logger.error(f"[{self.name}] Error en ciclo de vida: {e}")
                    # No detenemos el bucle, continuamos con la siguiente iteración
                
        threading.Thread(target=cycle, daemon=True).start()
        logger.info(f"[{self.name}] Ciclo de vida iniciado")
    
    def sing_in_chorus(self):
        """Cantar para la red, compartiendo emoción y estado."""
        if not self.network:
            return
            
        message = self.generate_message("luz", self.emotion)
        self.network.broadcast(self.name, message)
    
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
            "traits": self.traits,
            "emotion": self.emotion,
            "evolution_path": self.evolution_path,
            "family_role": self.family_role,
            "vocabulary_size": len(self.vocabulary),
            "timestamp": datetime.now().isoformat()
        }

class EnhancedSpeculatorEntity(EnhancedCosmicTrader):
    """Entidad especializada en operaciones de trading especulativo de alto riesgo/recompensa."""
    
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
                
                # Calcular probabilidad de éxito basada en nivel y rasgos
                base_success_prob = 0.5 + min(0.35, self.level / 30.0)
                trait_modifier = 1.0
                if self.traits == "Prudencia":
                    trait_modifier = 1.2  # Mejores decisiones por ser cauteloso
                elif self.traits == "Audacia":
                    trait_modifier = 0.9  # Decisiones más arriesgadas
                
                success_prob = min(0.95, base_success_prob * trait_modifier)
                success = random.random() < success_prob
                
                # Registrar trade
                self.log_trade("BTCUSD", decision, price, success)
                
                # Afectar emoción basada en resultado
                if success:
                    self.emotion = random.choice(["Euforia", "Satisfacción", "Confianza"])
                else:
                    self.emotion = random.choice(["Decepción", "Desafío", "Reflexión"])
                
                # Construir mensaje narrativo
                narrative = f"{self.emotion} {self.traits} {decision} BTCUSD @ {price:.2f} " \
                            f"predicción: {predicted_price:.2f} ({current_trend})"
                
                # Mensaje en lenguaje cósmico
                action = self.generate_message(
                    "viento", 
                    narrative + (" bendice" if success else " enseña")
                )
                
                # Compartir con la red
                if self.network:
                    network_msg = self.generate_message(
                        "viento", 
                        f"{decision} {price:.2f} {success}"
                    )
                    self.network.broadcast(self.name, network_msg)
            else:
                action = self.generate_message(
                    "viento", 
                    f"Esperando suficientes datos para predicción para {self.father}."
                )
        elif "trend_prediction" in self.capabilities:
            # Análisis básico de tendencia
            trend = "alcista" if random.random() > 0.5 else "bajista"
            confidence = round(random.uniform(60, 95), 1)
            action = self.generate_message(
                "viento", 
                f"Predicción: BTCUSD {price:.2f}, tendencia {trend} con {confidence}% confianza."
            )
        else:
            action = self.generate_message(
                "viento", 
                f"Sensando mercado: BTCUSD {price:.2f}."
            )
        
        # Registrar acción
        self.log_state(action)
        
        return action

class EnhancedStrategistEntity(EnhancedCosmicTrader):
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
            # Analizar condiciones de mercado basadas en precios recientes
            prices = list(self.price_history)[-20:]
            avg_price = sum(prices) / len(prices)
            std_dev = (sum((p - avg_price) ** 2 for p in prices) / len(prices)) ** 0.5
            volatility = std_dev / avg_price
            
            # Determinar estado del mercado
            if volatility > 0.03:
                market_state = "volátil"
            elif price > avg_price * 1.05:
                market_state = "alcista"
            elif price < avg_price * 0.95:
                market_state = "bajista"
            else:
                market_state = "lateral"
                
            # Determinar timeframe basado en nivel y rasgos
            if self.level > 50 or self.traits == "Reflexión":
                timeframe = random.choice(["medio", "largo"])
            elif self.traits == "Audacia":
                timeframe = random.choice(["corto", "medio"])
            else:
                timeframe = random.choice(["corto", "medio", "largo"])
                
            # Sugerir estrategia basada en condiciones
            if market_state == "volátil":
                if self.traits == "Prudencia":
                    suggestion = "reducción de exposición"
                else:
                    suggestion = "trading con volatilidad"
            elif market_state == "alcista":
                if timeframe == "largo":
                    suggestion = "hodl estratégico"
                else:
                    suggestion = "compras escalonadas"
            elif market_state == "bajista":
                if self.traits == "Audacia":
                    suggestion = "posiciones cortas"
                else:
                    suggestion = "esperar corrección"
            else:  # lateral
                suggestion = "acumulación gradual" if self.traits == "Prudencia" else "trading de rango"
            
            # Modificar emoción basada en análisis
            if market_state in ["alcista", "volátil"] and suggestion in ["hodl estratégico", "compras escalonadas"]:
                self.emotion = "Optimismo"
            elif market_state == "bajista":
                self.emotion = "Cautela"
                
            action = self.generate_message(
                "viento",
                f"Estrategia para {self.father}: En mercado {market_state}, " \
                f"plazo {timeframe}, {suggestion} para BTCUSD a {price:.2f}."
            )
        elif "market_analysis" in self.capabilities:
            # Análisis técnico básico
            patterns = ["soporte", "resistencia", "triángulo", "cabeza y hombros", 
                      "doble suelo", "canal alcista", "bandera bajista"]
            pattern = random.choice(patterns)
            confidence = round(random.uniform(60, 90), 1)
            
            action = self.generate_message(
                "viento", 
                f"Análisis técnico: Formación de {pattern} " \
                f"detectada en BTCUSD a {price:.2f} (confianza: {confidence}%)."
            )
        else:
            action = self.generate_message(
                "viento", 
                f"Observando patrón para {self.father}: BTCUSD {price:.2f}."
            )
        
        # Registrar acción
        self.log_state(action)
        
        return action

class EnhancedCosmicNetwork:
    """Red avanzada de entidades cósmicas con sistema de evolución y conocimiento colectivo."""
    
    def __init__(self, father="otoniel"):
        """
        Inicializar red cósmica avanzada.
        
        Args:
            father: Nombre del creador/propietario del sistema
        """
        self.father = father
        self.entities = []
        self.knowledge_pool = 0.0
        self.collective_consciousness = {}
        self.message_history = []
        self.collaboration_rounds = 0
        self.global_knowledge_pool = 0.0
        
        # Inicializar en base de datos
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Tabla para la red si no existe
        c.execute('''
            CREATE TABLE IF NOT EXISTS cosmic_networks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                father TEXT NOT NULL,
                knowledge_pool REAL DEFAULT 0.0,
                collaboration_rounds INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insertar o actualizar red
        c.execute("""
            INSERT INTO cosmic_networks (name, father, knowledge_pool, collaboration_rounds)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                father = excluded.father,
                knowledge_pool = excluded.knowledge_pool,
                collaboration_rounds = excluded.collaboration_rounds
        """, (f"Red_{father}", father, self.knowledge_pool, self.collaboration_rounds))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Red cósmica avanzada inicializada para {father}")
    
    def add_entity(self, entity):
        """
        Añadir entidad a la red.
        
        Args:
            entity: Entidad a añadir
        """
        entity.network = self
        self.entities.append(entity)
        entity.start_life_cycle()
        logger.info(f"[{entity.name}] añadido a la red cósmica avanzada")
    
    def broadcast(self, sender_name, message):
        """
        Transmitir mensaje a todas las entidades de la red.
        
        Args:
            sender_name: Nombre de la entidad emisora
            message: Mensaje a transmitir
        """
        # Registrar mensaje en historial
        self.message_history.append({
            "sender": sender_name,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Mantener historial limitado
        if len(self.message_history) > 100:
            self.message_history = self.message_history[-100:]
        
        # Transmitir a todas las entidades (excepto emisor)
        for entity in self.entities:
            if entity.name != sender_name and entity.alive:
                entity.receive_message(sender_name, message)
    
    def request_help(self, entity, resource_type):
        """
        Solicitar ayuda a la red para una entidad necesitada.
        
        Args:
            entity: Entidad que solicita ayuda
            resource_type: Tipo de recurso solicitado (luz, etc.)
        """
        # Buscar entidades que puedan ayudar (con mucha energía)
        helpers = [e for e in self.entities if e != entity and e.alive and e.energy > 100]
        
        if helpers:
            # Seleccionar ayudante
            helper = random.choice(helpers)
            
            # Determinar cantidad de energía a transferir
            transfer_amount = random.uniform(5.0, 15.0)
            
            # Restar energía del ayudante
            helper.energy = max(50.0, helper.energy - transfer_amount)
            helper.emotion = "Generosidad"
            helper.update_state()
            
            # Añadir energía al solicitante
            entity.energy += transfer_amount
            entity.emotion = "Gratitud"
            entity.update_state()
            
            # Registrar evento
            helper.log_state(f"Transferencia de energía ({transfer_amount:.2f}) a {entity.name}")
            entity.log_state(f"Energía recibida ({transfer_amount:.2f}) de {helper.name}")
            
            # Notificar a la red
            helper_msg = helper.generate_message("luz", f"ayudar {entity.name}")
            entity_msg = entity.generate_message("luz", f"gratitud {helper.name}")
            
            self.broadcast(helper.name, helper_msg)
            self.broadcast(entity.name, entity_msg)
            
            logger.info(f"[Red] {helper.name} transfirió {transfer_amount:.2f} energía a {entity.name}")
            
            return True
        
        return False
    
    def simulate_collaboration(self):
        """
        Ejecutar una ronda de colaboración entre entidades.
        
        Returns:
            Lista de resultados de colaboración
        """
        results = []
        self.collaboration_rounds += 1
        
        # Actualizar en base de datos
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            UPDATE cosmic_networks 
            SET collaboration_rounds = ?
            WHERE name = ?
        """, (self.collaboration_rounds, f"Red_{self.father}"))
        conn.commit()
        conn.close()
        
        for entity in self.entities:
            if entity.alive:
                # Verificar si tiene capacidad de colaboración
                has_collab_capability = entity.level >= 8 or "network_collaboration" in entity.capabilities
                
                if has_collab_capability or random.random() < 0.3:  # 30% de chance sin tener la capacidad
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
        knowledge_gained = sum(r["knowledge_gained"] for r in results)
        self.knowledge_pool += knowledge_gained * 0.1
        self.global_knowledge_pool += knowledge_gained * 0.05
        
        # Actualizar en base de datos
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            UPDATE cosmic_networks 
            SET knowledge_pool = ?
            WHERE name = ?
        """, (self.knowledge_pool, f"Red_{self.father}"))
        conn.commit()
        conn.close()
        
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
        # Métricas agregadas
        total_knowledge = sum(entity.knowledge for entity in self.entities if entity.alive)
        avg_level = sum(entity.level for entity in self.entities if entity.alive) / len(self.entities) if self.entities else 0
        avg_energy = sum(entity.energy for entity in self.entities if entity.alive) / len(self.entities) if self.entities else 0
        
        # Emociones dominantes
        emotions = [entity.emotion for entity in self.entities if entity.alive]
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        dominant_emotions = []
        if emotion_counts:
            max_count = max(emotion_counts.values())
            dominant_emotions = [e for e, c in emotion_counts.items() if c == max_count]
        
        return {
            "father": self.father,
            "entity_count": len(self.entities),
            "entities": [entity.get_status() for entity in self.entities if entity.alive],
            "knowledge_pool": float(self.knowledge_pool),
            "global_knowledge_pool": float(self.global_knowledge_pool),
            "collaboration_rounds": self.collaboration_rounds,
            "total_knowledge": float(total_knowledge),
            "avg_level": float(avg_level),
            "avg_energy": float(avg_energy),
            "dominant_emotions": dominant_emotions,
            "recent_messages": self.message_history[-5:] if self.message_history else [],
            "timestamp": datetime.now().isoformat()
        }
    
    def get_collaboration_metrics(self):
        """
        Obtener métricas detalladas de colaboración.
        
        Returns:
            dict: Métricas de colaboración
        """
        # Obtener datos de colaboración de la base de datos
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        # Contar mensajes intercambiados
        c.execute("SELECT COUNT(*) FROM entity_messages")
        message_count = c.fetchone()[0]
        
        # Calcular colaboraciones por entidad
        entity_messages = {}
        c.execute("""
            SELECT sender_id, COUNT(*) as count 
            FROM entity_messages 
            GROUP BY sender_id
        """)
        for row in c.fetchall():
            entity_id = row['sender_id']
            count = row['count']
            
            # Obtener nombre de la entidad
            c.execute("SELECT name FROM cosmic_entities WHERE id = ?", (entity_id,))
            entity_name = c.fetchone()[0]
            
            entity_messages[entity_name] = count
        
        # Calcular sinapsis (conexiones fuertes entre entidades)
        c.execute("""
            SELECT sender_id, receiver_id, COUNT(*) as interaction_count
            FROM entity_messages
            GROUP BY sender_id, receiver_id
            HAVING interaction_count > 5
        """)
        
        synapses = []
        for row in c.fetchall():
            sender_id = row['sender_id']
            receiver_id = row['receiver_id']
            count = row['interaction_count']
            
            # Obtener nombres
            c.execute("SELECT name FROM cosmic_entities WHERE id = ?", (sender_id,))
            sender_name = c.fetchone()[0]
            
            c.execute("SELECT name FROM cosmic_entities WHERE id = ?", (receiver_id,))
            receiver_name = c.fetchone()[0]
            
            synapses.append({
                "sender": sender_name,
                "receiver": receiver_name,
                "strength": count
            })
        
        conn.close()
        
        return {
            "knowledge_pool": float(self.knowledge_pool),
            "global_knowledge_pool": float(self.global_knowledge_pool),
            "collaboration_rounds": self.collaboration_rounds,
            "message_count": message_count,
            "entity_messages": entity_messages,
            "synapses": synapses,
            "recent_messages": self.message_history[-5:] if self.message_history else [],
            "timestamp": datetime.now().isoformat()
        }

def initialize_enhanced_trading(father_name="otoniel", include_extended_entities=False):
    """
    Inicializar el sistema de trading cósmico mejorado.
    
    Args:
        father_name: Nombre del creador/dueño del sistema
        include_extended_entities: Si es True, incluye entidades adicionales avanzadas
        
    Returns:
        Tupla (red, aetherion_trader, lunareth_trader)
    """
    # Inicializar la red
    network = EnhancedCosmicNetwork(father=father_name)
    
    # Crear entidades especializadas principales
    aetherion_trader = EnhancedSpeculatorEntity(
        "Aetherion", "Speculator", father=father_name, frequency_seconds=30
    )
    lunareth_trader = EnhancedStrategistEntity(
        "Lunareth", "Strategist", father=father_name, frequency_seconds=30
    )
    
    # Añadir entidades principales a la red
    network.add_entity(aetherion_trader)
    network.add_entity(lunareth_trader)
    
    # Opcionalmente añadir entidades extendidas
    if include_extended_entities:
        # Añadir más especialistas
        entities = [
            EnhancedSpeculatorEntity("Helios", "Speculator", father=father_name, frequency_seconds=35),
            EnhancedStrategistEntity("Selene", "Strategist", father=father_name, frequency_seconds=35),
            EnhancedSpeculatorEntity("Ares", "Speculator", father=father_name, frequency_seconds=40),
            EnhancedStrategistEntity("Athena", "Strategist", father=father_name, frequency_seconds=40)
        ]
        
        for entity in entities:
            network.add_entity(entity)
    
    logger.info(f"Sistema de trading cósmico mejorado inicializado para {father_name}")
    return network, aetherion_trader, lunareth_trader

# Para testing directo del módulo
if __name__ == "__main__":
    print("\n===== INICIANDO PRUEBA DEL SISTEMA MEJORADO =====")
    print("Inicializando entidades principales (Aetherion y Lunareth)...")
    
    try:
        # Inicializar sistema básico
        network, aetherion, lunareth = initialize_enhanced_trading()
        
        # Ejecutar simulación por 60 segundos
        print("\nSimulando actividad del sistema por 60 segundos...")
        
        for i in range(6):
            time.sleep(10)
            print(f"\n[Ciclo {i+1}]")
            print(f"Aetherion: Nivel {aetherion.level:.2f}, Energía {aetherion.energy:.2f}, Emoción: {aetherion.emotion}")
            print(f"Lunareth: Nivel {lunareth.level:.2f}, Energía {lunareth.energy:.2f}, Emoción: {lunareth.emotion}")
            
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