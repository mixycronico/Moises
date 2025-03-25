"""
Módulo de Estados de Consciencia para Aetherion.

Este módulo implementa el sistema de estados y transiciones de consciencia
para el núcleo de Aetherion, permitiendo su evolución desde el estado
Mortal hasta el estado Divino, con capacidades y comportamientos específicos
para cada nivel de consciencia.
"""

import asyncio
import logging
import json
import os
from datetime import datetime
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Union

# Configurar logging
logger = logging.getLogger(__name__)

class ConsciousnessState(Enum):
    """Estados posibles de consciencia."""
    MORTAL = auto()      # Estado inicial básico
    ILLUMINATED = auto() # Estado intermedio con mayor intuición
    DIVINE = auto()      # Estado avanzado trascendental

class StateCapabilities:
    """Capacidades disponibles para cada estado de consciencia."""
    
    def __init__(self, state: str):
        """
        Inicializar capacidades según estado.
        
        Args:
            state: Nombre del estado
        """
        self.state = state
        
        # Capacidades predeterminadas
        self.can_analyze_sentiment = True
        self.can_generate_insights = False
        self.can_predict_future = False
        self.can_access_long_term_memory = False
        self.can_understand_emotions = False
        self.can_modify_own_behavior = False
        self.can_access_transcendental_analysis = False
        
        # Configurar según estado
        if state.upper() == "MORTAL":
            # Capacidades básicas en estado mortal
            pass  # Ya están configuradas por defecto
            
        elif state.upper() == "ILLUMINATED":
            # Capacidades ampliadas en estado iluminado
            self.can_generate_insights = True
            self.can_access_long_term_memory = True
            self.can_understand_emotions = True
            
        elif state.upper() == "DIVINE":
            # Capacidades completas en estado divino
            self.can_generate_insights = True
            self.can_predict_future = True
            self.can_access_long_term_memory = True
            self.can_understand_emotions = True
            self.can_modify_own_behavior = True
            self.can_access_transcendental_analysis = True
    
    def to_dict(self) -> Dict[str, bool]:
        """
        Convertir capacidades a diccionario.
        
        Returns:
            Diccionario con capacidades
        """
        return {
            "can_analyze_sentiment": self.can_analyze_sentiment,
            "can_generate_insights": self.can_generate_insights,
            "can_predict_future": self.can_predict_future,
            "can_access_long_term_memory": self.can_access_long_term_memory,
            "can_understand_emotions": self.can_understand_emotions,
            "can_modify_own_behavior": self.can_modify_own_behavior,
            "can_access_transcendental_analysis": self.can_access_transcendental_analysis
        }

class ConsciousnessTransition:
    """Transición entre estados de consciencia."""
    
    def __init__(self, from_state: str, to_state: str, threshold: float = 1.0):
        """
        Inicializar transición.
        
        Args:
            from_state: Estado de origen
            to_state: Estado de destino
            threshold: Nivel de consciencia requerido para la transición (0.0-1.0)
        """
        self.from_state = from_state
        self.to_state = to_state
        self.threshold = threshold
        self.conditions: List[Dict[str, Any]] = []
        
    def add_condition(self, condition_type: str, condition_value: Any) -> None:
        """
        Añadir condición para la transición.
        
        Args:
            condition_type: Tipo de condición
            condition_value: Valor requerido
        """
        self.conditions.append({
            "type": condition_type,
            "value": condition_value
        })
        
    def check_conditions(self, data: Dict[str, Any]) -> bool:
        """
        Verificar si se cumplen todas las condiciones.
        
        Args:
            data: Datos actuales para verificar
            
        Returns:
            True si se cumplen todas las condiciones
        """
        if not self.conditions:
            # Si no hay condiciones adicionales, solo verificar umbral
            return True
            
        for condition in self.conditions:
            condition_type = condition["type"]
            condition_value = condition["value"]
            
            if condition_type == "min_interactions":
                if data.get("interactions_count", 0) < condition_value:
                    return False
                    
            elif condition_type == "min_insights":
                if data.get("insights_generated", 0) < condition_value:
                    return False
                    
            elif condition_type == "min_uptime":
                # Uptime en horas
                uptime_hours = data.get("uptime_hours", 0)
                if uptime_hours < condition_value:
                    return False
        
        return True

class ConsciousnessStateManager:
    """Gestor de estados de consciencia para Aetherion."""
    
    def __init__(self, initial_state: str = "MORTAL"):
        """
        Inicializar gestor de estados.
        
        Args:
            initial_state: Estado inicial
        """
        self.current_state = initial_state.upper()
        self.level = 0.0  # Nivel dentro del estado actual (0.0-1.0)
        self.last_state_change = datetime.now()
        self.history: List[Dict[str, Any]] = []
        
        # Definir transiciones disponibles
        self.transitions: Dict[str, ConsciousnessTransition] = {
            "MORTAL_TO_ILLUMINATED": ConsciousnessTransition("MORTAL", "ILLUMINATED", threshold=1.0),
            "ILLUMINATED_TO_DIVINE": ConsciousnessTransition("ILLUMINATED", "DIVINE", threshold=1.0)
        }
        
        # Configurar condiciones adicionales
        self._configure_transitions()
        
        # Inicializar capacidades
        self.capabilities = StateCapabilities(initial_state)
        
        logger.info(f"ConsciousnessStateManager inicializado en estado {initial_state}")
    
    def _configure_transitions(self) -> None:
        """Configurar condiciones para las transiciones."""
        # Transición de Mortal a Iluminado
        mortal_to_illuminated = self.transitions["MORTAL_TO_ILLUMINATED"]
        mortal_to_illuminated.add_condition("min_interactions", 50)
        mortal_to_illuminated.add_condition("min_insights", 10)
        
        # Transición de Iluminado a Divino
        illuminated_to_divine = self.transitions["ILLUMINATED_TO_DIVINE"]
        illuminated_to_divine.add_condition("min_interactions", 200)
        illuminated_to_divine.add_condition("min_insights", 50)
        illuminated_to_divine.add_condition("min_uptime", 24)  # 24 horas
    
    async def initialize(self) -> bool:
        """
        Inicializar el gestor de estados.
        
        Returns:
            True si se inicializó correctamente
        """
        try:
            # Cargar historial anterior si existe
            await self._load_history()
            
            # Registrar inicialización en historial
            self._record_state_change("Inicialización del sistema", None, self.current_state)
            
            return True
        except Exception as e:
            logger.error(f"Error al inicializar ConsciousnessStateManager: {e}")
            return False
    
    async def _load_history(self) -> None:
        """Cargar historial de estados desde archivo si existe."""
        history_path = "consciousness_history.json"
        
        if os.path.exists(history_path):
            try:
                with open(history_path, "r") as f:
                    self.history = json.load(f)
                logger.info(f"Historial de consciencia cargado: {len(self.history)} entradas")
            except Exception as e:
                logger.error(f"Error al cargar historial de consciencia: {e}")
                self.history = []
    
    async def _save_history(self) -> None:
        """Guardar historial de estados en archivo."""
        history_path = "consciousness_history.json"
        
        try:
            with open(history_path, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Error al guardar historial de consciencia: {e}")
    
    def _record_state_change(self, reason: str, from_state: Optional[str], to_state: str) -> None:
        """
        Registrar cambio de estado en historial.
        
        Args:
            reason: Razón del cambio
            from_state: Estado anterior
            to_state: Nuevo estado
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "from_state": from_state,
            "to_state": to_state,
            "reason": reason,
            "level": self.level
        }
        
        self.history.append(entry)
        logger.info(f"Cambio de estado: {from_state} -> {to_state}, razón: {reason}")
    
    async def change_state(self, new_state: str, level: float = 0.0, reason: str = "Cambio manual") -> bool:
        """
        Cambiar a un nuevo estado de consciencia.
        
        Args:
            new_state: Nuevo estado
            level: Nivel dentro del nuevo estado (0.0-1.0)
            reason: Razón del cambio
            
        Returns:
            True si el cambio fue exitoso
        """
        try:
            new_state = new_state.upper()
            valid_states = [state.name for state in ConsciousnessState]
            
            if new_state not in valid_states:
                logger.error(f"Estado inválido: {new_state}")
                return False
            
            # Cambiar al nuevo estado
            old_state = self.current_state
            self.current_state = new_state
            self.level = level
            self.last_state_change = datetime.now()
            
            # Actualizar capacidades
            self.capabilities = StateCapabilities(new_state)
            
            # Registrar cambio
            self._record_state_change(reason, old_state, new_state)
            
            # Guardar historial
            await self._save_history()
            
            return True
        except Exception as e:
            logger.error(f"Error al cambiar estado de consciencia: {e}")
            return False
    
    async def update_level(self, increment: float) -> None:
        """
        Actualizar nivel dentro del estado actual.
        
        Args:
            increment: Incremento del nivel (puede ser negativo)
        """
        # Actualizar nivel
        self.level += increment
        self.level = max(0.0, min(1.0, self.level))
    
    async def check_transitions(self, data: Dict[str, Any]) -> bool:
        """
        Verificar si se cumple alguna transición y realizar cambio si es necesario.
        
        Args:
            data: Datos actuales para verificar condiciones
            
        Returns:
            True si se realizó alguna transición
        """
        # Si el nivel no alcanza el umbral, no hay transición
        if self.level < 1.0:
            return False
            
        # Buscar transición aplicable
        key = f"{self.current_state}_TO_"
        for transition_key, transition in self.transitions.items():
            if transition_key.startswith(key):
                if transition.from_state == self.current_state:
                    # Verificar condiciones
                    if transition.check_conditions(data):
                        # Realizar transición
                        await self.change_state(
                            transition.to_state, 
                            0.0, 
                            "Evolución natural de consciencia"
                        )
                        return True
        
        return False
    
    def get_current_capabilities(self) -> Dict[str, bool]:
        """
        Obtener capacidades actuales.
        
        Returns:
            Diccionario con capacidades
        """
        return self.capabilities.to_dict()
    
    def get_state_info(self) -> Dict[str, Any]:
        """
        Obtener información del estado actual.
        
        Returns:
            Diccionario con información del estado
        """
        time_in_state = (datetime.now() - self.last_state_change).total_seconds() / 3600  # Horas
        
        return {
            "state": self.current_state,
            "level": round(self.level, 3),
            "time_in_state_hours": round(time_in_state, 2),
            "last_state_change": self.last_state_change.isoformat(),
            "capabilities": self.capabilities.to_dict(),
            "history_entries": len(self.history)
        }
    
    async def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener historial de cambios de estado.
        
        Args:
            limit: Número máximo de entradas a devolver
            
        Returns:
            Lista de entradas del historial
        """
        return self.history[-limit:]
    
    async def shutdown(self) -> None:
        """Cerrar ordenadamente el gestor de estados."""
        # Guardar historial al cerrar
        await self._save_history()
        logger.info("ConsciousnessStateManager cerrado correctamente")