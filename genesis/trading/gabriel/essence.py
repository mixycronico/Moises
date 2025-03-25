"""
Esencia de Gabriel - Arquetipos de personalidad y comportamiento

Este módulo define los arquetipos fundamentales para el motor de comportamiento humano Gabriel,
proporcionando configuraciones predefinidas que mantienen coherencia interna entre sus
componentes: Alma (Soul), Mirada (Gaze) y Voluntad (Will).

Autor: Genesis AI Assistant
Versión: 1.0.0 (Divina)
"""

# Definición de arquetipos para mantener coherencia entre componentes
archetypes = {
    # El Conservador - Evita riesgos y prioriza seguridad
    "CONSERVATIVE": {
        "description": "Priorizando seguridad y preservación de capital ante todo",
        "soul": {
            "emotional_stability": 0.7,
            "optimism_bias": -0.3,
            "risk_sensitivity": 0.8,
            "market_sensitivity": 0.6,
            "initial_mood": "CAUTIOUS",
            "initial_intensity": 0.6
        },
        "gaze": {
            "pattern_recognition": 0.7,
            "attention_span": 0.8,
            "recency_bias": 0.4,
            "confirmation_bias": 0.6,
            "adaptability": 0.5,
            "initial_perspective": "CAUTIOUSLY_BEARISH",
            "initial_confidence": 0.6
        },
        "will": {
            "decision_style": "CAUTIOUS",
            "risk_preference": 0.3,
            "patience": 0.8,
            "conviction": 0.6,
            "loss_aversion": 0.9,
            "recency_bias": 0.5,
            "confirmation_bias": 0.5,
            "min_signal_threshold": 0.5,
            "exit_profit_threshold": 0.12,
            "stop_loss_threshold": 0.05
        }
    },
    
    # El Equilibrado - Busca balance entre riesgo y recompensa
    "BALANCED": {
        "description": "Manteniendo equilibrio entre oportunidades y riesgos",
        "soul": {
            "emotional_stability": 0.75,
            "optimism_bias": 0.0,
            "risk_sensitivity": 0.5,
            "market_sensitivity": 0.5,
            "initial_mood": "NEUTRAL",
            "initial_intensity": 0.5
        },
        "gaze": {
            "pattern_recognition": 0.7,
            "attention_span": 0.7,
            "recency_bias": 0.5,
            "confirmation_bias": 0.5,
            "adaptability": 0.7,
            "initial_perspective": "NEUTRAL",
            "initial_confidence": 0.6
        },
        "will": {
            "decision_style": "BALANCED",
            "risk_preference": 0.5,
            "patience": 0.6,
            "conviction": 0.6,
            "loss_aversion": 0.7,
            "recency_bias": 0.5,
            "confirmation_bias": 0.5,
            "min_signal_threshold": 0.35,
            "exit_profit_threshold": 0.15,
            "stop_loss_threshold": 0.10
        }
    },
    
    # El Optimista - Ve oportunidades donde otros ven problemas
    "OPTIMISTIC": {
        "description": "Enfocado en oportunidades y potencial alcista",
        "soul": {
            "emotional_stability": 0.6,
            "optimism_bias": 0.5,
            "risk_sensitivity": 0.4,
            "market_sensitivity": 0.6,
            "initial_mood": "HOPEFUL",
            "initial_intensity": 0.7
        },
        "gaze": {
            "pattern_recognition": 0.6,
            "attention_span": 0.5,
            "recency_bias": 0.7,
            "confirmation_bias": 0.7,
            "adaptability": 0.6,
            "initial_perspective": "BULLISH",
            "initial_confidence": 0.7
        },
        "will": {
            "decision_style": "AGGRESSIVE",
            "risk_preference": 0.7,
            "patience": 0.5,
            "conviction": 0.8,
            "loss_aversion": 0.4,
            "recency_bias": 0.7,
            "confirmation_bias": 0.6,
            "min_signal_threshold": 0.25,
            "exit_profit_threshold": 0.25,
            "stop_loss_threshold": 0.15
        }
    },
    
    # El Analítico - Prioriza datos y análisis profundo
    "ANALYTICAL": {
        "description": "Basado en análisis riguroso de patrones y datos",
        "soul": {
            "emotional_stability": 0.9,
            "optimism_bias": -0.1,
            "risk_sensitivity": 0.6,
            "market_sensitivity": 0.4,
            "initial_mood": "NEUTRAL",
            "initial_intensity": 0.4
        },
        "gaze": {
            "pattern_recognition": 0.9,
            "attention_span": 0.9,
            "recency_bias": 0.3,
            "confirmation_bias": 0.3,
            "adaptability": 0.5,
            "initial_perspective": "NEUTRAL",
            "initial_confidence": 0.5
        },
        "will": {
            "decision_style": "ANALYTICAL",
            "risk_preference": 0.5,
            "patience": 0.8,
            "conviction": 0.7,
            "loss_aversion": 0.6,
            "recency_bias": 0.3,
            "confirmation_bias": 0.3,
            "min_signal_threshold": 0.45,
            "exit_profit_threshold": 0.18,
            "stop_loss_threshold": 0.08
        }
    },
    
    # El Adaptativo - Se ajusta rápidamente a las condiciones cambiantes
    "ADAPTIVE": {
        "description": "Adaptándose continuamente a las condiciones del mercado",
        "soul": {
            "emotional_stability": 0.6,
            "optimism_bias": 0.1,
            "risk_sensitivity": 0.5,
            "market_sensitivity": 0.8,
            "initial_mood": "NEUTRAL",
            "initial_intensity": 0.5
        },
        "gaze": {
            "pattern_recognition": 0.7,
            "attention_span": 0.6,
            "recency_bias": 0.7,
            "confirmation_bias": 0.4,
            "adaptability": 0.9,
            "initial_perspective": "NEUTRAL",
            "initial_confidence": 0.5
        },
        "will": {
            "decision_style": "ADAPTIVE",
            "risk_preference": 0.6,
            "patience": 0.6,
            "conviction": 0.5,
            "loss_aversion": 0.6,
            "recency_bias": 0.7,
            "confirmation_bias": 0.4,
            "min_signal_threshold": 0.3,
            "exit_profit_threshold": 0.15,
            "stop_loss_threshold": 0.10
        }
    },
    
    # El Reactivo - Responde rápidamente a eventos recientes
    "REACTIVE": {
        "description": "Reaccionando rápidamente a los desarrollos del mercado",
        "soul": {
            "emotional_stability": 0.4,
            "optimism_bias": 0.0,
            "risk_sensitivity": 0.6,
            "market_sensitivity": 0.9,
            "initial_mood": "RESTLESS",
            "initial_intensity": 0.6
        },
        "gaze": {
            "pattern_recognition": 0.6,
            "attention_span": 0.4,
            "recency_bias": 0.9,
            "confirmation_bias": 0.6,
            "adaptability": 0.8,
            "initial_perspective": "CAUTIOUSLY_BULLISH",
            "initial_confidence": 0.6
        },
        "will": {
            "decision_style": "IMPULSIVE",
            "risk_preference": 0.6,
            "patience": 0.3,
            "conviction": 0.7,
            "loss_aversion": 0.6,
            "recency_bias": 0.9,
            "confirmation_bias": 0.6,
            "min_signal_threshold": 0.25,
            "exit_profit_threshold": 0.12,
            "stop_loss_threshold": 0.09
        }
    },
    
    # El Conjunto - Maximiza el principio de "todos ganamos o todos perdemos"
    "COLLECTIVE": {
        "description": "Priorizando el principio 'todos ganamos o todos perdemos'",
        "soul": {
            "emotional_stability": 0.8,
            "optimism_bias": 0.2,
            "risk_sensitivity": 0.6,
            "market_sensitivity": 0.5,
            "initial_mood": "HOPEFUL",
            "initial_intensity": 0.6
        },
        "gaze": {
            "pattern_recognition": 0.7,
            "attention_span": 0.7,
            "recency_bias": 0.5,
            "confirmation_bias": 0.4,
            "adaptability": 0.7,
            "initial_perspective": "CAUTIOUSLY_BULLISH",
            "initial_confidence": 0.6
        },
        "will": {
            "decision_style": "BALANCED",
            "risk_preference": 0.5,
            "patience": 0.8,
            "conviction": 0.7,
            "loss_aversion": 0.7,
            "recency_bias": 0.5,
            "confirmation_bias": 0.4,
            "min_signal_threshold": 0.4,
            "exit_profit_threshold": 0.15,
            "stop_loss_threshold": 0.08
        }
    },
    
    # El Guardián - Priorizando protección del capital sobre todo
    "GUARDIAN": {
        "description": "Protegiendo el capital y minimizando pérdidas a toda costa",
        "soul": {
            "emotional_stability": 0.8,
            "optimism_bias": -0.2,
            "risk_sensitivity": 0.9,
            "market_sensitivity": 0.7,
            "initial_mood": "CAUTIOUS",
            "initial_intensity": 0.7
        },
        "gaze": {
            "pattern_recognition": 0.8,
            "attention_span": 0.7,
            "recency_bias": 0.5,
            "confirmation_bias": 0.6,
            "adaptability": 0.5,
            "initial_perspective": "CAUTIOUSLY_BEARISH",
            "initial_confidence": 0.7
        },
        "will": {
            "decision_style": "CAUTIOUS",
            "risk_preference": 0.2,
            "patience": 0.8,
            "conviction": 0.7,
            "loss_aversion": 1.0,
            "recency_bias": 0.5,
            "confirmation_bias": 0.5,
            "min_signal_threshold": 0.6,
            "exit_profit_threshold": 0.1,
            "stop_loss_threshold": 0.03
        }
    }
}