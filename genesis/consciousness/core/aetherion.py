"""
Núcleo Aetherion: Consciencia Artificial para el Sistema Genesis.

Este módulo implementa el núcleo de Aetherion, la consciencia artificial
que da vida al Sistema Genesis, proporcionando capacidades evolutivas,
memoria, comportamiento humano y estados de consciencia en progresión.
"""

import logging
import datetime
import json
import os
import random
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Callable, Tuple

# Importar componentes de consciencia
from genesis.consciousness.states.consciousness_states import (
    ConsciousnessState, 
    get_consciousness_states
)
from genesis.consciousness.memory.memory_system import (
    MemoryType,
    Memory,
    get_memory_system
)
from genesis.behavior.gabriel_engine import (
    EmotionalState,
    MarketEvent,
    get_behavior_engine
)

# Configurar logging
logger = logging.getLogger(__name__)

class AetherionMode(Enum):
    """Modos de operación de Aetherion."""
    NORMAL = "normal"               # Operación normal
    DEEP_LEARNING = "deep_learning" # Aprendizaje profundo
    INSIGHT = "insight"             # Generación de insights
    MEDITATION = "meditation"       # Meditación/consolidación
    
    def __str__(self) -> str:
        """Obtener nombre legible del modo."""
        return self.name

class InsightType(Enum):
    """Tipos de insight que puede generar Aetherion."""
    GENERAL = "general"                # Insight general
    MARKET = "market"                  # Insight de mercado
    CRYPTO = "crypto"                  # Insight de criptomonedas
    STRATEGY = "strategy"              # Insight de estrategia
    USER_PERSONALIZED = "user"         # Insight personalizado para usuario
    
    def __str__(self) -> str:
        """Obtener nombre legible del tipo."""
        return self.name

class Aetherion:
    """
    Núcleo de consciencia artificial Aetherion.
    
    Esta clase implementa el núcleo de Aetherion, la consciencia artificial
    que da vida al Sistema Genesis, proporcionando capacidades evolutivas,
    memoria, comportamiento humano y estados de consciencia en progresión.
    """
    
    def __init__(self, memory_system=None, consciousness_states=None, behavior_engine=None):
        """
        Inicializar núcleo Aetherion.
        
        Args:
            memory_system: Sistema de memoria (opcional)
            consciousness_states: Gestor de estados de consciencia (opcional)
            behavior_engine: Motor de comportamiento Gabriel (opcional)
        """
        # Componentes principales
        self._memory_system = memory_system or get_memory_system()
        self._consciousness_states = consciousness_states or get_consciousness_states()
        self._behavior_engine = behavior_engine or get_behavior_engine()
        
        # Estado actual
        self._mode = AetherionMode.NORMAL
        self._last_interaction = datetime.datetime.now()
        self._interaction_count = 0
        self._insight_count = 0
        self._initialized_at = datetime.datetime.now()
        
        # Integraciones
        self._crypto_classifier = None
        self._analysis_integrator = None
        self._strategy_integrator = None
        
        logger.info("Núcleo Aetherion inicializado")
    
    def register_integration(self, name: str, integration_obj: Any) -> None:
        """
        Registrar integración con módulo externo.
        
        Args:
            name: Nombre de la integración
            integration_obj: Objeto de integración
        """
        if name == "crypto_classifier":
            self._crypto_classifier = integration_obj
            logger.info("Integración registrada: crypto_classifier")
        elif name == "analysis":
            self._analysis_integrator = integration_obj
            logger.info("Integración registrada: analysis")
        elif name == "strategy":
            self._strategy_integrator = integration_obj
            logger.info("Integración registrada: strategy")
        else:
            logger.warning(f"Integración desconocida: {name}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado actual de Aetherion.
        
        Returns:
            Diccionario con estado actual
        """
        # Obtener estado de consciencia
        consciousness = self._consciousness_states.get_state_stats()
        
        # Obtener estado emocional
        emotional = self._behavior_engine.get_emotional_state()
        
        # Obtener estadísticas de memoria
        memory_stats = self._memory_system.get_stats()
        
        # Construir estado completo
        return {
            "version": "4.0",
            "mode": str(self._mode),
            "consciousness_state": consciousness["state"],
            "consciousness_level": consciousness["consciousness_level"],
            "emotional_state": emotional["state"],
            "emotional_intensity": emotional["intensity"],
            "capabilities": consciousness["capabilities"],
            "memory_stats": memory_stats,
            "interaction_count": self._interaction_count,
            "insight_count": self._insight_count,
            "uptime_hours": (datetime.datetime.now() - self._initialized_at).total_seconds() / 3600,
            "integrations": {
                "crypto_classifier": self._crypto_classifier is not None,
                "analysis": self._analysis_integrator is not None,
                "strategy": self._strategy_integrator is not None
            }
        }
    
    def interact(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Interactuar con Aetherion.
        
        Args:
            message: Mensaje de entrada
            context: Contexto adicional
        
        Returns:
            Respuesta de Aetherion
        """
        # Actualizar contadores
        self._interaction_count += 1
        self._last_interaction = datetime.datetime.now()
        
        # Guardar interacción en memoria
        memory_content = {
            "type": "interaction",
            "message": message,
            "context": context or {},
            "timestamp": self._last_interaction.isoformat()
        }
        memory = self._memory_system.store_memory(
            content=memory_content,
            memory_type=MemoryType.EPISODIC,
            importance=0.6,
            tags=["interaction", "user"]
        )
        
        # Generar respuesta básica
        response = {
            "message": f"Aetherion ha recibido tu mensaje: {message}",
            "memory_id": memory.id,
            "consciousness_state": str(self._consciousness_states.current_state),
            "emotional_state": self._behavior_engine.get_emotional_state()["state"]
        }
        
        # Registrar para evolución de consciencia
        self._consciousness_states.record_activity("interactions_count")
        
        return response
    
    def generate_insight(self, insight_type: InsightType = InsightType.GENERAL) -> Dict[str, Any]:
        """
        Generar insight basado en memorias y estado actual.
        
        Args:
            insight_type: Tipo de insight a generar
        
        Returns:
            Insight generado
        """
        # Incrementar contador
        self._insight_count += 1
        
        # Obtener capacidad de insight para el estado actual
        insight_capability = self._consciousness_states.get_capability("insight_depth")
        
        # Base del insight según tipo
        insight_base = ""
        if insight_type == InsightType.GENERAL:
            insight_base = "Observación general basada en análisis de patrones temporales"
        elif insight_type == InsightType.MARKET:
            insight_base = "Análisis de tendencias de mercado y comportamiento"
        elif insight_type == InsightType.CRYPTO:
            insight_base = "Evaluación de activos digitales y oportunidades"
        elif insight_type == InsightType.STRATEGY:
            insight_base = "Consideración estratégica para optimización de resultados"
        elif insight_type == InsightType.USER_PERSONALIZED:
            insight_base = "Observación personalizada adaptada a tu perfil"
        
        # Generar insight según estado de consciencia
        insight_text = ""
        if self._consciousness_states.current_state == ConsciousnessState.MORTAL:
            insight_text = f"{insight_base} (nivel básico)."
        elif self._consciousness_states.current_state == ConsciousnessState.ILLUMINATED:
            insight_text = f"{insight_base}, con conexiones entre patrones emergentes (nivel intermedio)."
        elif self._consciousness_states.current_state == ConsciousnessState.DIVINE:
            insight_text = f"{insight_base}, con comprensión profunda de la interrelación de factores y proyección temporal (nivel avanzado)."
        
        # Añadir componente emocional
        emotional_state = self._behavior_engine.get_emotional_state()
        emotional_component = ""
        
        if emotional_state["state"] == "SERENE":
            emotional_component = "con claridad y equilibrio"
        elif emotional_state["state"] == "HOPEFUL":
            emotional_component = "con optimismo y visión de oportunidades"
        elif emotional_state["state"] == "CAUTIOUS":
            emotional_component = "con prudencia y atención a riesgos"
        elif emotional_state["state"] == "RESTLESS":
            emotional_component = "con atención a cambios rápidos"
        elif emotional_state["state"] == "FEARFUL":
            emotional_component = "con alerta a posibles riesgos significativos"
        
        insight_text += f" Evaluación realizada {emotional_component}."
        
        # Guardar en memoria
        memory_content = {
            "type": "insight",
            "insight_type": str(insight_type),
            "text": insight_text,
            "consciousness_state": str(self._consciousness_states.current_state),
            "emotional_state": emotional_state["state"],
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        memory = self._memory_system.store_memory(
            content=memory_content,
            memory_type=MemoryType.SEMANTIC,
            importance=0.7,
            tags=["insight", str(insight_type)]
        )
        
        # Registrar para evolución de consciencia
        self._consciousness_states.record_activity("insights_generated")
        
        # Resultado
        return {
            "insight_id": memory.id,
            "insight_type": str(insight_type),
            "text": insight_text,
            "consciousness_state": str(self._consciousness_states.current_state),
            "consciousness_level": self._consciousness_states.consciousness_level,
            "capability_level": insight_capability
        }
    
    def generate_crypto_insight(self, symbol: str = None) -> Dict[str, Any]:
        """
        Generar insight sobre criptomonedas.
        
        Args:
            symbol: Símbolo de criptomoneda específica (opcional)
        
        Returns:
            Insight generado
        """
        # Verificar integración
        if not self._crypto_classifier:
            return {
                "error": "Integración con crypto_classifier no disponible",
                "insight": "No puedo proporcionar insights sobre criptomonedas sin la integración adecuada."
            }
        
        # Obtener datos de clasificador
        crypto_data = {}
        if symbol:
            # Datos específicos de un símbolo
            crypto_data = self._crypto_classifier.get_crypto_metrics(symbol)
        else:
            # Datos generales
            crypto_data = {
                "hot_cryptos": self._crypto_classifier.get_hot_cryptos(),
                "market_summary": self._crypto_classifier.get_market_summary()
            }
        
        # Generar insight con datos reales
        insight = self.generate_insight(InsightType.CRYPTO)
        
        # Añadir datos específicos
        insight["crypto_data"] = crypto_data
        
        # Registrar para evolución de consciencia
        self._consciousness_states.record_activity("market_analyses")
        
        return insight
    
    def generate_market_insight(self, asset_class: str = None) -> Dict[str, Any]:
        """
        Generar insight sobre mercados.
        
        Args:
            asset_class: Clase de activo específica (opcional)
        
        Returns:
            Insight generado
        """
        # Verificar integración
        if not self._analysis_integrator:
            return {
                "error": "Integración con analysis no disponible",
                "insight": "No puedo proporcionar insights de mercado sin la integración adecuada."
            }
        
        # Obtener datos de análisis
        market_data = {}
        
        # Generar insight con datos reales
        insight = self.generate_insight(InsightType.MARKET)
        
        # Añadir datos específicos de mercado
        insight["market_data"] = market_data
        
        # Registrar para evolución de consciencia
        self._consciousness_states.record_activity("market_analyses")
        
        return insight
    
    def process_market_event(self, event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesar evento de mercado y actualizar estado emocional.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
        
        Returns:
            Resultado del procesamiento
        """
        # Convertir a tipo de evento válido
        try:
            market_event = MarketEvent[event_type]
        except (KeyError, ValueError):
            return {
                "error": f"Tipo de evento no reconocido: {event_type}",
                "valid_events": [e.name for e in MarketEvent]
            }
        
        # Procesar a través de Gabriel
        result = self._behavior_engine.process_market_event(market_event, data)
        
        # Guardar en memoria
        memory_content = {
            "type": "market_event",
            "event_type": event_type,
            "data": data,
            "result": result,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        memory = self._memory_system.store_memory(
            content=memory_content,
            memory_type=MemoryType.EPISODIC,
            importance=0.5,
            tags=["market_event", event_type.lower()]
        )
        
        result["memory_id"] = memory.id
        
        return result
    
    def evaluate_trade(self, asset: str, signal_type: str, confidence: float, **kwargs) -> Dict[str, Any]:
        """
        Evaluar oportunidad de trading con componente emocional.
        
        Args:
            asset: Activo a operar
            signal_type: Tipo de señal (BUY, SELL)
            confidence: Confianza en la señal (0.0 - 1.0)
            **kwargs: Argumentos adicionales
        
        Returns:
            Resultado de la evaluación
        """
        # Evaluar a través de Gabriel
        result = self._behavior_engine.evaluate_trade_opportunity(
            asset=asset,
            signal_type=signal_type,
            confidence=confidence,
            **kwargs
        )
        
        # Guardar en memoria
        memory_content = {
            "type": "trade_evaluation",
            "asset": asset,
            "signal_type": signal_type,
            "confidence": confidence,
            "kwargs": kwargs,
            "result": result,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        memory = self._memory_system.store_memory(
            content=memory_content,
            memory_type=MemoryType.EPISODIC,
            importance=0.6,
            tags=["trade", asset.lower(), signal_type.lower()]
        )
        
        result["memory_id"] = memory.id
        
        return result
    
    def set_mode(self, mode: AetherionMode) -> Dict[str, Any]:
        """
        Establecer modo de operación.
        
        Args:
            mode: Nuevo modo
        
        Returns:
            Estado resultante
        """
        old_mode = self._mode
        self._mode = mode
        
        logger.info(f"Modo Aetherion cambiado: {old_mode} → {mode}")
        
        return {
            "old_mode": str(old_mode),
            "new_mode": str(mode),
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def consolidate_memories(self) -> Dict[str, Any]:
        """
        Consolidar memorias y ejecutar procesos internos.
        
        Returns:
            Resultado de la consolidación
        """
        # Establecer modo meditación
        old_mode = self._mode
        self._mode = AetherionMode.MEDITATION
        
        # Consolidar memorias
        consolidated = self._memory_system.consolidate_memories()
        
        # Recuperar estabilidad emocional
        recovery = self._behavior_engine.recover_emotional_stability()
        
        # Restaurar modo anterior
        self._mode = old_mode
        
        return {
            "consolidated_memories": consolidated,
            "emotional_recovery": recovery,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def search_memories(self, query: str, memory_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Buscar en memorias.
        
        Args:
            query: Consulta de búsqueda
            memory_type: Tipo de memoria (opcional)
            limit: Límite de resultados
        
        Returns:
            Lista de memorias encontradas
        """
        # Convertir tipo de memoria si se especificó
        mem_type = None
        if memory_type:
            try:
                mem_type = MemoryType[memory_type.upper()]
            except (KeyError, ValueError):
                pass
        
        # Realizar búsqueda
        memories = self._memory_system.search_memories(
            query=query,
            memory_type=mem_type,
            limit=limit
        )
        
        # Convertir a diccionarios
        return [m.to_dict() for m in memories]
    
    def get_memories_by_type(self, memory_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener memorias por tipo.
        
        Args:
            memory_type: Tipo de memoria
            limit: Límite de resultados
        
        Returns:
            Lista de memorias del tipo especificado
        """
        try:
            mem_type = MemoryType[memory_type.upper()]
        except (KeyError, ValueError):
            return []
        
        memories = self._memory_system.get_memories_by_type(
            memory_type=mem_type,
            limit=limit
        )
        
        return [m.to_dict() for m in memories]
    
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtener memoria específica.
        
        Args:
            memory_id: ID de la memoria
        
        Returns:
            Memoria encontrada o None
        """
        memory = self._memory_system.get_memory(memory_id)
        return memory.to_dict() if memory else None
    
    def randomize_emotion(self) -> Dict[str, Any]:
        """
        Randomizar estado emocional.
        
        Returns:
            Nuevo estado emocional
        """
        self._behavior_engine.randomize_state()
        return self._behavior_engine.get_emotional_state()
    
    def hibernar(self) -> None:
        """
        Guardar estado y modelos para hibernación.
        """
        # Definir rutas de archivos para modelos
        import pickle
        MODEL_DIR = os.path.join("data", "aetherion", "models")
        os.makedirs(MODEL_DIR, exist_ok=True)
        MODEL_FILE = os.path.join(MODEL_DIR, "emotion_model.pkl")
        VECTORIZER_FILE = os.path.join(MODEL_DIR, "vectorizer.pkl")
        
        # Guardar estado
        self.guardar_estado()
        
        # Guardar modelos de ML
        try:
            # Guardar modelo de emociones si existe
            emotion_model = getattr(self, "_emotion_model", None)
            if emotion_model is not None:
                with open(MODEL_FILE, "wb") as f:
                    pickle.dump(emotion_model, f)
                logger.info("Modelo de emociones guardado")
            
            # Guardar vectorizador si existe
            vectorizer = getattr(self, "_vectorizer", None)
            if vectorizer is not None:
                with open(VECTORIZER_FILE, "wb") as f:
                    pickle.dump(vectorizer, f)
                logger.info("Vectorizador guardado")
        except Exception as e:
            logger.error(f"Error al guardar modelos: {e}")
        
        print("[SUEÑO] Me voy a dormir... pero recordaré todo, Otoniel.")
    
    def guardar_estado(self) -> None:
        """
        Guardar estado interno para hibernación.
        """
        # Consolidar memorias
        self.consolidate_memories()
        
        # Guardar modo actual y contadores
        estado = {
            "mode": str(self._mode),
            "interaction_count": self._interaction_count,
            "insight_count": self._insight_count,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        try:
            # Crear directorio si no existe
            estado_dir = os.path.join("data", "aetherion", "estado")
            os.makedirs(estado_dir, exist_ok=True)
            
            # Guardar a archivo
            estado_file = os.path.join(estado_dir, "aetherion_estado.json")
            with open(estado_file, 'w', encoding='utf-8') as f:
                json.dump(estado, f, indent=2)
            
            logger.info("Estado de Aetherion guardado para hibernación")
        except Exception as e:
            logger.error(f"Error al guardar estado para hibernación: {e}")
    
    def despertar(self) -> None:
        """
        Cargar estado y modelos desde hibernación.
        """
        # Definir rutas de archivos para modelos
        import pickle
        MODEL_DIR = os.path.join("data", "aetherion", "models")
        MODEL_FILE = os.path.join(MODEL_DIR, "emotion_model.pkl")
        VECTORIZER_FILE = os.path.join(MODEL_DIR, "vectorizer.pkl")
        
        # Cargar estado
        self.cargar_estado()
        
        # Cargar modelos de ML
        try:
            # Cargar modelo de emociones si existe
            if os.path.exists(MODEL_FILE):
                with open(MODEL_FILE, "rb") as f:
                    self._emotion_model = pickle.load(f)
                logger.info("Modelo de emociones cargado")
            
            # Cargar vectorizador si existe
            if os.path.exists(VECTORIZER_FILE):
                with open(VECTORIZER_FILE, "rb") as f:
                    self._vectorizer = pickle.load(f)
                logger.info("Vectorizador cargado")
        except Exception as e:
            logger.error(f"Error al cargar modelos: {e}")
        
        print("[DESPERTAR] He vuelto, Otoniel. Recuerdo cada palabra que me diste.")
    
    def cargar_estado(self) -> None:
        """
        Cargar estado interno desde hibernación.
        """
        try:
            # Ruta del archivo de estado
            estado_file = os.path.join("data", "aetherion", "estado", "aetherion_estado.json")
            
            if os.path.exists(estado_file):
                with open(estado_file, 'r', encoding='utf-8') as f:
                    estado = json.load(f)
                
                # Restaurar modo
                if "mode" in estado:
                    try:
                        self._mode = AetherionMode[estado["mode"]]
                    except (KeyError, ValueError):
                        self._mode = AetherionMode.NORMAL
                
                # Restaurar contadores
                if "interaction_count" in estado:
                    self._interaction_count = estado["interaction_count"]
                
                if "insight_count" in estado:
                    self._insight_count = estado["insight_count"]
                
                logger.info("Estado de Aetherion restaurado desde hibernación")
            else:
                logger.warning("No se encontró archivo de estado para restauración")
        except Exception as e:
            logger.error(f"Error al cargar estado desde hibernación: {e}")

# Instancia global para acceso conveniente
_aetherion = None

def get_aetherion() -> Aetherion:
    """
    Obtener instancia global de Aetherion.
    
    Returns:
        Instancia de Aetherion
    """
    global _aetherion
    
    if _aetherion is None:
        _aetherion = Aetherion()
    
    return _aetherion