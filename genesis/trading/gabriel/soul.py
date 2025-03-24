"""
El Alma de Gabriel - Núcleo emocional del comportamiento humano celestial

Este módulo captura la esencia emocional con una profundidad humana exquisita,
modelando estados de ánimo, estabilidad emocional y cambios de humor que
afectan profundamente las decisiones de trading.
"""

from enum import Enum, auto
from dataclasses import dataclass
from datetime import datetime
import random
import logging

logger = logging.getLogger(__name__)

class Mood(Enum):
    """Estados de ánimo que reflejan la complejidad emocional humana."""
    SERENE = auto()        # Paz interior, decisiones equilibradas
    HOPEFUL = auto()       # Alas de optimismo, busca el amanecer
    WARY = auto()          # Susurros de cautela, pasos cuidadosos
    RESTLESS = auto()      # Corazón acelerado, ansia por actuar
    BOLD = auto()          # Fuego de confianza, desafía al destino
    FRAUGHT = auto()       # Nudos de ansiedad, sombras en la mente
    DREAD = auto()         # Abismo de miedo, busca refugio (100% FEARFUL)
    PENSIVE = auto()       # Silencio reflexivo, tejiendo pensamientos

    @property
    def is_fearful(self) -> bool:
        """Determina si el estado actual es de miedo extremo."""
        return self == Mood.DREAD

    @property
    def is_positive(self) -> bool:
        """Determina si el estado actual es positivo."""
        return self in [Mood.SERENE, Mood.HOPEFUL, Mood.BOLD]

    @property
    def is_cautious(self) -> bool:
        """Determina si el estado actual es cauteloso."""
        return self in [Mood.WARY, Mood.FRAUGHT, Mood.PENSIVE]

@dataclass
class Soul:
    """El alma de Gabriel, centro de sus emociones y respuestas humanas."""
    mood: Mood = Mood.SERENE
    stability: float = 0.7    # Qué tan firme es el alma ante el viento
    whimsy: float = 0.2       # Capricho humano, chispa impredecible
    last_shift: datetime = datetime.now()
    mood_duration: float = 0.0  # Duración del estado actual en horas

    async def sway(self, whisper: str, intensity: float, echoes: dict) -> Mood:
        """
        El alma se mueve con los susurros del mundo.
        
        Args:
            whisper: El evento o estímulo emocional
            intensity: Intensidad del evento (0.0-1.0)
            echoes: Efectos emocionales de diferentes eventos
            
        Returns:
            El estado de ánimo resultante
        """
        old_mood = self.mood
        fates = {
            "cataclysm": Mood.DREAD,     # Catástrofe → Miedo absoluto
            "triumph": Mood.BOLD,        # Triunfo → Confianza
            "dawn": Mood.HOPEFUL,        # Amanecer → Esperanza
            "loss": Mood.FRAUGHT,        # Pérdida → Ansiedad
            "exhaustion": Mood.PENSIVE,  # Agotamiento → Reflexión
        }

        # Eventos contundentes siempre cambian el estado anímico
        if whisper in fates and (intensity >= 1.0 or random.random() < intensity):
            self.mood = fates[whisper]
            self.last_shift = datetime.now()
            self.mood_duration = 0.0
            
            # Registro con detalle poético
            logger.info(f"Alma transformada por {whisper}: {old_mood.name} → {self.mood.name} " +
                       f"(intensidad: {intensity:.2f})")
            return self.mood

        # La estabilidad reduce el efecto de los eventos menores
        resilience = self.stability * (1.0 + self.mood_duration/24.0)  # Mayor duración → mayor resistencia
        
        echo = echoes.get(f"on_{whisper}", echoes["natural_fade"]) * intensity * (1 - resilience)
        
        # Los estados de miedo son más persistentes
        if self.mood == Mood.DREAD and random.random() < 0.9:
            logger.debug(f"El miedo persiste a pesar de {whisper}")
            self.mood_duration += 0.1  # Incremento pequeño en la duración
            return self.mood
            
        # Cambio gradual en respuesta a estímulos
        moods = list(Mood)
        current = moods.index(self.mood)
        
        # Dirección del cambio basada en el estímulo
        shift = 1 if echo > 0 else -1
        new_idx = min(max(current + shift, 0), len(moods) - 1)
        
        # Capricho humano - a veces cambiamos sin razón aparente
        if random.random() < self.whimsy * (1.0 - intensity):  # Menos caprichoso en eventos intensos
            new_idx = random.randint(0, len(moods) - 1)
            reason = "capricho inexplicable"
        else:
            reason = f"respuesta a {whisper}"
            
        self.mood = moods[new_idx]
        self.last_shift = datetime.now()
        self.mood_duration = 0.0  # Reiniciar duración con el nuevo estado
        
        logger.debug(f"El alma danza: {old_mood.name} → {self.mood.name} por {reason}")
        return self.mood

    def reflect(self) -> Mood:
        """
        Un vistazo al estado del alma.
        
        Returns:
            El estado de ánimo actual, ocasionalmente fluctuante
        """
        # Actualizar duración del estado actual
        now = datetime.now()
        hours_since_shift = (now - self.last_shift).total_seconds() / 3600
        self.mood_duration = hours_since_shift
        
        # Un destello de duda humana - fluctuaciones aleatorias poco frecuentes
        if self.mood != Mood.DREAD and random.random() < 0.05:
            momentary_mood = random.choice(list(Mood))
            logger.debug(f"Fluctuación momentánea: {self.mood.name} → {momentary_mood.name}")
            return momentary_mood
            
        # El estado de miedo es absolutamente estable - sin fluctuaciones
        if self.mood == Mood.DREAD:
            return self.mood
            
        return self.mood
        
    def set_dread(self, reason: str = "danger_sensed") -> None:
        """
        Establece directamente el estado de miedo absoluto (DREAD).
        
        Args:
            reason: Motivo del cambio al estado de miedo
        """
        old_mood = self.mood
        self.mood = Mood.DREAD
        self.last_shift = datetime.now()
        self.mood_duration = 0.0
        
        logger.info(f"ATENCIÓN: Alma sumergida en miedo absoluto: {old_mood.name} → {self.mood.name} " +
                   f"(razón: {reason})")