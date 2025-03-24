"""
La Esencia de Gabriel - Configuraciones, principios y valores intrínsecos

Este módulo define los valores centrales, límites, umbrales y principios
que guían el comportamiento de Gabriel, dándole una personalidad coherente.
"""

from genesis.trading.human_behavior_engine import EmotionalState, RiskTolerance, DecisionStyle

__all__ = [
    'essence',
    'archetypes',
    'EmotionalState',
    'RiskTolerance',
    'DecisionStyle'
]

# Configuración principal de Gabriel
essence = {
    # === UMBRALES Y VALORES PARA DECISIONES ===
    
    # Umbral de confianza para entrar en operaciones según nivel de coraje
    "courage_thresholds": {
        "TIMID": 0.75,       # Necesita 75% de confianza para actuar
        "BALANCED": 0.60,    # Punto medio equilibrado
        "DARING": 0.45,      # Se atreve con solo 45% de confianza
    },
    
    # Objetivos de beneficio según nivel de coraje (porcentaje)
    "profit_targets": {
        "TIMID": 5.0,        # Toma beneficios pequeños pero seguros
        "BALANCED": 10.0,    # Objetivo moderado
        "DARING": 20.0,      # Busca grandes ganancias
    },
    
    # Límites de pérdidas según nivel de coraje (porcentaje negativo)
    "loss_limits": {
        "TIMID": -3.0,       # Corta pérdidas muy rápido
        "BALANCED": -8.0,    # Tolerancia moderada
        "DARING": -15.0,     # Aguanta pérdidas mayores esperando recuperación
    },
    
    # === EFECTOS EMOCIONALES DE EVENTOS ===
    
    # Cómo de intensamente afectan ciertos eventos al estado emocional
    "emotional_echoes": {
        "on_victory": 0.3,       # Efecto de una ganancia
        "on_defeat": -0.4,       # Efecto de una pérdida
        "on_stress": -0.3,       # Efecto de estrés en el mercado
        "on_opportunity": 0.25,  # Efecto de ver una oportunidad
        "on_relief": 0.2,        # Efecto de alivio (ej. recuperación)
        "natural_fade": 0.05,    # Desvanecimiento natural de emociones con el tiempo
    },
    
    # === PATRONES DE DECISIÓN ===
    
    # Influencia del estilo de decisión en el tiempo de reflexión
    "reflection_periods": {
        "THOUGHTFUL": {          # Estilo reflexivo
            "entry": 5.0,        # Tiempo (s) para decisiones de entrada 
            "exit": 3.0          # Tiempo (s) para decisiones de salida
        },
        "INSTINCTIVE": {         # Estilo instintivo
            "entry": 1.0,
            "exit": 0.5
        }, 
        "STEADFAST": {           # Estilo firme
            "entry": 3.0,
            "exit": 4.0          # Más meditación para salidas que entradas
        }
    },
    
    # === CARACTERÍSTICAS COMPORTAMENTALES ===
    
    # Tendencias de adaptación según estado emocional
    "adaptation_rates": {
        "SERENE": 1.0,           # Adaptación normal
        "DREAD": 5.0,            # Adaptación extremadamente rápida en miedo
        "BOLD": 0.7,             # Adaptación más lenta, más consistente
        "FRAUGHT": 2.0,          # Adaptación acelerada en ansiedad
    },
    
    # Disposición contraria (ir contra la tendencia) según estado
    "contrarian_bias": {
        "SERENE": 0.1,           # Ligeramente contrario
        "BOLD": 0.3,             # Moderadamente contrario
        "DREAD": -0.5,           # Sigue la manada en miedo (anti-contrario)
        "WARY": 0.2              # Moderadamente contrario en cautela
    },
    
    # === MEMORIA Y APRENDIZAJE ===
    
    # Pesos de recencia para diferentes tipos de experiencias
    "memory_weights": {
        "successes": 0.6,        # Recuerda 60% de los éxitos
        "failures": 0.8,         # Recuerda 80% de los fracasos
        "trauma": 0.95,          # Recuerda 95% de las experiencias traumáticas
        "baseline": 0.3          # Memoria general
    },
    
    # === CONDUCTAS ESPECÍFICAS PARA ESTADO DE MIEDO (100%) ===
    
    "fearful_behavior": {
        "buy_rejection_chance": 1.0,      # 100% de rechazo para compras
        "buy_size_multiplier": 0.1,       # Compras al 10% del tamaño normal
        "sell_size_multiplier": 1.5,      # Ventas al 150% del tamaño normal
        "profit_target_multiplier": 0.2,  # Objetivo de beneficio reducido al 20%
        "loss_limit_multiplier": 0.1,     # Stop loss al 10% del normal
        "confidence_threshold": 0.95,     # Requiere 95% de confianza (imposible)
        "market_deterioration_threshold": -0.001,  # Cualquier caída minúscula
        "max_holding_time_hours": 1.0     # Máximo 1 hora de mantenimiento
    }
}

# Características de personalidad predefinidas
archetypes = {
    "cautious_investor": {
        "courage": "TIMID",
        "resolve": "THOUGHTFUL",
        "base_mood": "WARY",
        "stability": 0.8,
        "whimsy": 0.1
    },
    "bold_trader": {
        "courage": "DARING",
        "resolve": "INSTINCTIVE", 
        "base_mood": "BOLD",
        "stability": 0.5,
        "whimsy": 0.3
    },
    "balanced_operator": {
        "courage": "BALANCED",
        "resolve": "STEADFAST",
        "base_mood": "SERENE",
        "stability": 0.7, 
        "whimsy": 0.15
    },
    "anxious_observer": {
        "courage": "TIMID",
        "resolve": "THOUGHTFUL",
        "base_mood": "FRAUGHT",
        "stability": 0.4,
        "whimsy": 0.2
    }
}