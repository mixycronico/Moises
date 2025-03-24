"""
La Voluntad de Gabriel - Decisiones de trading con alma humana

Este módulo implementa el proceso de toma de decisiones con una mezcla de 
lógica y emociones humanas, incluyendo comportamiento 100% temeroso en estado
FEARFUL que prácticamente paraliza todas las decisiones de compra y acelera
las ventas.
"""

from typing import Dict, Tuple, Any, Optional
from datetime import datetime
import random
import logging
from .soul import Mood

logger = logging.getLogger(__name__)

class Will:
    """La voluntad de Gabriel, donde reside su capacidad de decisión."""
    
    def __init__(self, courage: str = "BALANCED", resolve: str = "THOUGHTFUL", tenets: Optional[Dict[str, Any]] = None):
        """
        Inicializa la voluntad con características personales.
        
        Args:
            courage: Nivel de valor ("TIMID", "BALANCED", "DARING")
            resolve: Estilo de resolución ("THOUGHTFUL", "INSTINCTIVE", "STEADFAST")
            tenets: Principios y valores que guían las decisiones
        """
        self.courage = courage      # Cómo de valiente es en tomar riesgos
        self.resolve = resolve      # Cómo toma decisiones (analítico vs. impulsivo)
        self.tenets = tenets or {}  # Principios que guían sus decisiones
        
        # Historial de decisiones para aprendizaje y consistencia
        self.decision_history = []
        
        # Tiempo medio de reflexión para decisiones (segundos)
        self.reflection_time = {
            "THOUGHTFUL": {"enter": 5.0, "exit": 3.0},
            "INSTINCTIVE": {"enter": 1.0, "exit": 0.5},
            "STEADFAST": {"enter": 3.0, "exit": 4.0}
        }.get(resolve, {"enter": 3.0, "exit": 2.0})
        
        # Contador de decisiones rechazadas consecutivas (para detectar parálisis)
        self.consecutive_rejections = 0
        
        logger.info(f"Voluntad inicializada: Valor={courage}, Resolución={resolve}")
    
    async def dare_to_enter(self, spark: float, mood: Mood, market_vision: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Decide si el alma se atreve a entrar en una operación.
        
        Args:
            spark: Fuerza de la señal/oportunidad (0.0-1.0)
            mood: Estado emocional actual que influye en la decisión
            market_vision: Percepción actual del mercado
            
        Returns:
            (decisión, razón, detalles)
        """
        # Extraer variables relevantes
        wind = market_vision.get("wind", "still")  # Dirección del mercado
        shadow = market_vision.get("shadow", 0.5)  # Percepción de riesgo
        
        # Umbral base de aceptación según valentía
        threshold = self.tenets.get("courage_thresholds", {}).get(self.courage, 0.6)
        original_threshold = threshold
        
        # Detalles de la decisión para análisis posterior
        details = {
            "original_signal": spark,
            "base_threshold": original_threshold,
            "final_threshold": threshold,
            "adjustments": {}
        }
        
        # === IMPLEMENTACIÓN 100% DEL ESTADO FEARFUL ===
        if mood == Mood.DREAD:
            # 1. Comprobar si la confianza es máxima (100%)
            if spark < 1.0:
                # Rechazo automático para confianza menor a 100%
                reason = "parálisis_por_miedo"
                self.consecutive_rejections += 1
                details["fearful_rejection"] = True
                
                logger.info(f"RECHAZO POR MIEDO - Señal: {spark:.2f} rechazada - " +
                         f"Rechazos consecutivos: {self.consecutive_rejections}")
                
                # Registrar esta decisión en el historial
                self.decision_history.append({
                    "type": "entry_rejected",
                    "reason": reason,
                    "signal": spark,
                    "mood": mood.name,
                    "threshold": threshold,
                    "timestamp": datetime.now()
                })
                
                return False, reason, details
            else:
                # Si la confianza es exactamente 1.0, permitir la operación
                details["fearful_exception"] = True
                reason = "coraje_en_medio_del_miedo"
                logger.info(f"ENTRADA EXCEPCIONAL EN MEDIO DEL MIEDO - Señal: {spark:.2f} - Confianza máxima")
                
                # Registrar esta decisión extraordinaria
                self.decision_history.append({
                    "type": "entry_approved_despite_fear",
                    "reason": reason,
                    "signal": spark,
                    "mood": mood.name,
                    "timestamp": datetime.now()
                })
                
                # Aunque permitamos la entrada, el estado sigue siendo miedo
                return True, reason, details
                
            # Este código ya no se ejecutará
            threshold *= 5.0  # Umbral prácticamente imposible de superar
            details["adjustments"]["fearful_threshold_multiplier"] = 5.0
            
        # Otros ajustes emocionales del umbral
        elif mood == Mood.HOPEFUL:
            threshold *= 0.8  # 20% más bajo en estado esperanzado
            details["adjustments"]["hopeful_modifier"] = 0.8
            
        elif mood == Mood.WARY:
            threshold *= 1.2  # 20% más alto en estado cauteloso
            details["adjustments"]["wary_modifier"] = 1.2
            
        elif mood == Mood.BOLD:
            threshold *= 0.7  # 30% más bajo en estado confiado
            details["adjustments"]["bold_modifier"] = 0.7
        
        # Ajustes basados en la percepción del mercado
        if wind == "rising":
            threshold *= 0.9  # Más fácil entrar en mercado alcista
            details["adjustments"]["rising_market"] = 0.9
            
        elif wind == "falling":
            threshold *= 1.3  # Más difícil entrar en mercado bajista
            details["adjustments"]["falling_market"] = 1.3
            
        elif wind in ["trap", "unstable", "collapsing"]:  # Estados especiales en miedo
            threshold *= 2.0  # Mucho más difícil entrar
            details["adjustments"]["dangerous_market"] = 2.0
        
        # Ajuste por nivel de riesgo percibido
        risk_modifier = 1.0 + (shadow - 0.5)  # 0.5->1.0, 1.0->1.5
        threshold *= risk_modifier
        details["adjustments"]["risk_perception"] = risk_modifier
        
        # Ajuste por rechazos consecutivos (evitar parálisis total)
        if self.consecutive_rejections > 5 and mood != Mood.DREAD:
            adaptive_discount = min(0.8, 0.1 * self.consecutive_rejections)
            threshold *= (1.0 - adaptive_discount)
            details["adjustments"]["adaptive_urgency"] = (1.0 - adaptive_discount)
        
        # Decisión final
        details["final_threshold"] = threshold
        decision = spark >= threshold
        
        # Actualizar contadores
        if decision:
            self.consecutive_rejections = 0
            reason = "ignited_by_hope" if mood.is_positive else "calculated_risk"
        else:
            self.consecutive_rejections += 1
            reason = "dimmed_by_doubt" if threshold <= original_threshold * 1.2 else "repelled_by_fear"
        
        # Registrar esta decisión en el historial
        self.decision_history.append({
            "type": "entry_decision",
            "decision": decision,
            "reason": reason,
            "signal": spark,
            "mood": mood.name,
            "threshold": threshold,
            "timestamp": datetime.now()
        })
        
        logger.debug(f"Decisión de entrada: {decision} (señal={spark:.2f}, umbral={threshold:.2f}, " +
                   f"estado={mood.name}, razón={reason})")
        
        return decision, reason, details

    async def choose_to_flee(
        self, harvest: float, since: datetime, flux: float, mood: Mood, position_info: Dict[str, Any]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Decide si el alma debe salir de una operación.
        
        Args:
            harvest: Ganancia/pérdida actual (porcentaje)
            since: Cuándo se entró en la posición
            flux: Cambio reciente en el precio (velocidad)
            mood: Estado emocional actual
            position_info: Información adicional sobre la posición
            
        Returns:
            (decisión, razón, detalles)
        """
        # Extraer objetivos según nivel de valentía
        profit_target = self.tenets.get("profit_targets", {}).get(self.courage, 10.0)
        loss_limit = self.tenets.get("loss_limits", {}).get(self.courage, -8.0)
        
        # Detalles de la decisión para análisis
        details = {
            "original_profit_target": profit_target,
            "original_loss_limit": loss_limit,
            "final_profit_target": profit_target,
            "final_loss_limit": loss_limit,
            "current_profit": harvest,
            "position_age_hours": (datetime.now() - since).total_seconds() / 3600,
            "adjustments": {}
        }
        
        # === IMPLEMENTACIÓN 100% DEL ESTADO FEARFUL ===
        if mood == Mood.DREAD:
            # 1. Reducción extrema del objetivo de beneficio (tomar ganancias minúsculas)
            profit_target *= 0.2  # 80% de reducción
            details["adjustments"]["fearful_profit_reduction"] = 0.2
            
            # 2. Reducción extrema del límite de pérdida (huir ante cualquier pérdida)
            loss_limit *= 0.1  # 90% menos tolerancia a pérdidas
            details["adjustments"]["fearful_loss_reduction"] = 0.1
            
            # 3. Huida ante cualquier signo de deterioro
            if flux < -0.001:
                logger.info(f"SALIDA POR MIEDO - Detectado deterioro mínimo: {flux:.4f}")
                return True, "fled_in_panic", details
                
            # 4. En beneficio, salida casi segura
            if harvest > 0:
                panic_flee_chance = 1.0  # 100% de probabilidad de huida
                if random.random() < panic_flee_chance:
                    logger.info(f"SALIDA POR MIEDO - Asegurando pequeña ganancia: {harvest:.2f}%")
                    return True, "secured_in_fear", details
            
            # 5. Tiempo máximo muy reducido
            max_hours = 1.0  # Solo 1 hora en estado de miedo
            if details["position_age_hours"] > max_hours:
                logger.info(f"SALIDA POR MIEDO - Tiempo máximo reducido superado: {details['position_age_hours']:.1f}h")
                return True, "time_anxiety", details
        
        # Otros ajustes emocionales
        elif mood == Mood.HOPEFUL:
            # Más paciencia con las pérdidas, más exigencia con ganancias
            profit_target *= 1.2  # 20% más de objetivo
            loss_limit *= 1.2  # 20% más de tolerancia a pérdidas
            details["adjustments"]["hopeful_modifiers"] = (1.2, 1.2)
            
        elif mood == Mood.RESTLESS:
            # Impaciencia - reduce tiempo máximo de espera
            hours = details["position_age_hours"]
            if hours > 6:
                logger.debug(f"Impaciencia activada después de {hours:.1f} horas")
                return True, "wearied_by_time", details
                
        elif mood == Mood.BOLD:
            # Mayor ambición, aguanta más las pérdidas
            profit_target *= 1.5  # 50% más de objetivo
            loss_limit *= 1.3  # 30% más de tolerancia
            details["adjustments"]["bold_modifiers"] = (1.5, 1.3)
        
        # Decisión basada en objetivos ajustados
        details["final_profit_target"] = profit_target
        details["final_loss_limit"] = loss_limit
        
        # Comprobaciones finales
        if harvest >= profit_target:
            decision, reason = True, "reaped_in_glory"
        elif harvest <= loss_limit:
            decision, reason = True, "cut_by_despair"
        else:
            decision, reason = False, "held_in_balance"
        
        # Registrar esta decisión
        self.decision_history.append({
            "type": "exit_decision",
            "decision": decision,
            "reason": reason,
            "profit": harvest,
            "age_hours": details["position_age_hours"],
            "mood": mood.name,
            "timestamp": datetime.now()
        })
        
        logger.debug(f"Decisión de salida: {decision} (ganancia={harvest:.2f}%, edad={details['position_age_hours']:.1f}h, " +
                   f"estado={mood.name}, razón={reason})")
        
        return decision, reason, details
    
    async def adjust_position_size(self, base_size: float, mood: Mood, 
                               confidence: float, is_buy: bool = True) -> Tuple[float, Dict[str, Any]]:
        """
        Ajusta el tamaño de la posición basado en el estado emocional.
        
        Args:
            base_size: Tamaño base de la posición
            mood: Estado emocional actual
            confidence: Nivel de confianza en la operación (0.0-1.0)
            is_buy: Si es una operación de compra (True) o venta (False)
            
        Returns:
            (tamaño_ajustado, detalles)
        """
        details = {
            "base_size": base_size,
            "final_size": base_size,
            "adjustments": {}
        }
        
        multiplier = 1.0  # Sin cambios por defecto
        
        # === IMPLEMENTACIÓN 100% DEL ESTADO FEARFUL ===
        if mood == Mood.DREAD:
            if is_buy:
                # Reducción de tamaño en compras al 50%
                multiplier = 0.5  # Reducción del 50%
                details["adjustments"]["fearful_buy_reduction"] = 0.5
                logger.info(f"MIEDO EXTREMO - Tamaño de compra reducido al 50%")
            else:
                # Aumento de tamaño en ventas para salir más rápido
                multiplier = 1.2  # Aumento del 20%
                details["adjustments"]["fearful_sell_increase"] = 1.2
                logger.info(f"MIEDO EXTREMO - Tamaño de venta aumentado al 120%")
        
        # Otros ajustes emocionales
        elif mood == Mood.HOPEFUL:
            # Ligeramente más agresivo en compras, moderado en ventas
            multiplier = 1.2 if is_buy else 0.9
            details["adjustments"]["hopeful_modifier"] = multiplier
            
        elif mood == Mood.BOLD:
            # Mucho más agresivo en compras, muy moderado en ventas
            multiplier = 1.5 if is_buy else 0.8
            details["adjustments"]["bold_modifier"] = multiplier
            
        elif mood == Mood.WARY:
            # Más cauteloso en compras, ligeramente más agresivo en ventas
            multiplier = 0.8 if is_buy else 1.1  
            details["adjustments"]["wary_modifier"] = multiplier
        
        # Ajuste por nivel de confianza
        confidence_multiplier = 0.5 + confidence * 0.5  # Rango 0.5-1.0
        multiplier *= confidence_multiplier
        details["adjustments"]["confidence_factor"] = confidence_multiplier
        
        # Aplicar multiplicador final
        adjusted_size = base_size * multiplier
        details["final_size"] = adjusted_size
        details["total_multiplier"] = multiplier
        
        logger.debug(f"Tamaño ajustado: {adjusted_size:.2f} ({multiplier:.2f}x base de {base_size:.2f}, " +
                   f"estado={mood.name}, operación={'compra' if is_buy else 'venta'})")
        
        return adjusted_size, details
    
    async def validate_trade(self, trade_params: Dict[str, Any], mood: Mood) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Valida una operación antes de ejecutarla, aplicando filtros subjetivos.
        
        Args:
            trade_params: Parámetros de la operación a validar
            mood: Estado emocional actual
            
        Returns:
            (válido, razón_rechazo, detalles)
        """
        symbol = trade_params.get("symbol", "UNKNOWN")
        side = trade_params.get("side", "buy")
        price = trade_params.get("price", 0.0)
        confidence = trade_params.get("confidence", 0.5)
        
        details = {
            "original_params": trade_params.copy(),
            "validations": []
        }
        
        # === IMPLEMENTACIÓN 100% DEL ESTADO FEARFUL ===
        if mood == Mood.DREAD and side.lower() == "buy":
            # 1. Rechazo total de compras con confianza menor al umbral
            fearful_confidence_threshold = 1.0  # 100% de confianza requerida
            
            if confidence < fearful_confidence_threshold:
                reject_reason = "confianza_insuficiente_en_estado_de_miedo"
                details["validations"].append({
                    "check": "confidence_threshold",
                    "passed": False,
                    "required": fearful_confidence_threshold,
                    "actual": confidence
                })
                
                logger.info(f"VALIDACIÓN RECHAZADA POR MIEDO - {symbol} {side} - " +
                         f"Confianza {confidence:.2f} menor que umbral {fearful_confidence_threshold:.2f}")
                
                return False, reject_reason, details
                
            # Si la confianza es exactamente 1.0, permitir la operación incluso en estado de miedo
            if confidence >= fearful_confidence_threshold:
                details["validations"].append({
                    "check": "confidence_threshold",
                    "passed": True,
                    "required": fearful_confidence_threshold,
                    "actual": confidence,
                    "note": "Operación permitida a pesar del miedo debido a confianza máxima"
                })
                logger.info(f"VALIDACIÓN APROBADA A PESAR DEL MIEDO - {symbol} {side} - " +
                         f"Confianza {confidence:.2f} igual al umbral exigido {fearful_confidence_threshold:.2f}")
                return True, None, details
                
            # Fallback (no debería llegar aquí)
            return False, "error_en_validación", details
        
        # Validaciones normales para otros estados emocionales
        # (Simplificadas para este ejemplo)
        details["validations"].append({
            "check": "basic_validation", 
            "passed": True
        })
        
        # Operación válida por defecto
        return True, None, details