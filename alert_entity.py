"""
Implementación de entidad de alertas y notificaciones para Sistema Genesis.

Este módulo implementa la entidad Sentinel, especializada en la detección, 
clasificación y notificación de alertas relevantes del sistema.
"""

import os
import logging
import random
import time
import threading
import json
from typing import Dict, Any, List, Optional, Tuple
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from enhanced_simple_cosmic_trader import EnhancedCosmicTrader
from enhanced_cosmic_entity_mixin import EnhancedCosmicEntityMixin

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlertEntity(EnhancedCosmicTrader, EnhancedCosmicEntityMixin):
    """
    Entidad especializada en la detección, clasificación y notificación de alertas.
    Proporciona servicios de monitoreo continuo y alerta proactiva.
    """
    
    # Niveles de alerta
    ALERT_LEVELS = ["INFO", "WARNING", "CRITICAL", "EMERGENCY"]
    
    # Categorías de alertas
    ALERT_CATEGORIES = [
        "MARKET", "SYSTEM", "SECURITY", "COMMUNICATION", 
        "DATABASE", "PERFORMANCE", "ANOMALY", "INTEGRATION"
    ]
    
    def __init__(self, name: str, role: str = "Alert", father: str = "otoniel", 
                 frequency_seconds: int = 15):
        """
        Inicializar entidad de alertas.
        
        Args:
            name: Nombre de la entidad
            role: Rol (siempre será "Alert")
            father: Nombre del creador/dueño
            frequency_seconds: Período de ciclo de vida en segundos (más frecuente para alertas)
        """
        super().__init__(name, role, father, frequency_seconds)
        
        # Configuración de alertas
        self.alert_thresholds = {
            "cpu_usage": 80,  # %
            "memory_usage": 85,  # %
            "api_latency": 2000,  # ms
            "error_rate": 10,  # %
            "transaction_anomaly": 3.0,  # desviación estándar
            "connection_drops": 5,  # count
            "price_volatility": 8,  # %
            "disk_space": 90  # %
        }
        
        # Cola de alertas
        self.alert_queue = []
        self.max_queue_size = 1000
        self.notification_channels = ["log", "system_message", "entity_broadcast"]
        
        # Canal de email configurado (sin credenciales reales)
        self.email_configured = False
        self.email_config = {
            "smtp_server": "",
            "smtp_port": 587,
            "username": "",
            "password": "",
            "from_address": "",
            "to_addresses": []
        }
        
        # Estadísticas específicas
        self.stats = {
            "alerts_detected": 0,
            "alerts_processed": 0,
            "alerts_by_level": {level: 0 for level in self.ALERT_LEVELS},
            "alerts_by_category": {category: 0 for category in self.ALERT_CATEGORIES},
            "alerts_notified": 0,
            "alerts_suppressed": 0,
            "false_positives": 0
        }
        
        # Personalidad y rasgos específicos
        self.personality_traits = ["Vigilante", "Cauteloso", "Protector"]
        self.emotional_volatility = 0.3  # Baja volatilidad emocional
        
        # Especializaciones
        self.specializations = {
            "Alert Detection": 0.9,
            "Anomaly Recognition": 0.8,
            "Pattern Matching": 0.9,
            "Risk Assessment": 0.8,
            "Notification Distribution": 0.7
        }
        
        # Estado de alerta del sistema
        self.system_alert_level = "NORMAL"  # NORMAL, ELEVATED, HIGH, CRITICAL
        
        # Historial de alertas compactado
        self.alert_history = []
        self.max_history_size = 100
        
        logger.info(f"[{self.name}] Entidad de alertas inicializada")
    
    def trade(self):
        """
        Implementar método trade requerido por la clase base abstracta.
        Para la entidad de alertas, esto representa el análisis de patrones
        y detección de anomalías que podrían indicar oportunidades o riesgos.
        """
        # Simular análisis de alertas y patrones
        insights = self.analyze_alert_patterns()
        
        trade_result = {
            "action": "alert_analysis",
            "insights": insights,
            "metrics": {
                "patterns_detected": len(insights),
                "alert_level": self.system_alert_level,
                "risk_coefficient": self.calculate_risk_coefficient()
            }
        }
        
        # Registrar actividad de trading
        self.last_trade_time = time.time()
        self.trades_count += 1
        
        return trade_result
    
    def analyze_alert_patterns(self) -> List[Dict[str, Any]]:
        """
        Analizar patrones en las alertas recientes.
        
        Returns:
            Lista de insights encontrados
        """
        insights = []
        
        # Verificar si hay suficientes alertas para análisis
        if len(self.alert_history) < 5:
            return insights
        
        # Contar alertas por categoría en las últimas 20
        recent_alerts = self.alert_history[-20:]
        category_counts = {}
        for alert in recent_alerts:
            category = alert.get("category")
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Buscar categorías con muchas alertas
        for category, count in category_counts.items():
            if count >= 3:  # 3 o más alertas de la misma categoría
                insights.append({
                    "type": "category_concentration",
                    "category": category,
                    "count": count,
                    "significance": count / len(recent_alerts)
                })
        
        # Buscar secuencias de alertas del mismo tipo
        alert_sequence = [alert.get("level") for alert in recent_alerts]
        for level in self.ALERT_LEVELS:
            if alert_sequence.count(level) >= 3:
                insights.append({
                    "type": "escalating_severity",
                    "level": level,
                    "count": alert_sequence.count(level),
                    "significance": 0.7
                })
        
        # Calcular frecuencia de alertas recientes
        if len(recent_alerts) >= 10:
            first_time = recent_alerts[0].get("timestamp", 0)
            last_time = recent_alerts[-1].get("timestamp", time.time())
            if last_time > first_time:
                alerts_per_minute = len(recent_alerts) / ((last_time - first_time) / 60)
                if alerts_per_minute > 5:  # Más de 5 alertas por minuto
                    insights.append({
                        "type": "high_frequency",
                        "alerts_per_minute": alerts_per_minute,
                        "significance": min(alerts_per_minute / 10, 1.0)
                    })
        
        return insights
    
    def calculate_risk_coefficient(self) -> float:
        """
        Calcular coeficiente de riesgo basado en alertas recientes.
        
        Returns:
            Coeficiente de riesgo (0-1)
        """
        if not self.alert_history:
            return 0.1
        
        # Factores para el cálculo
        recent_alerts = self.alert_history[-20:]
        
        # Factor de nivel de alerta
        level_weights = {
            "INFO": 0.2,
            "WARNING": 0.5,
            "CRITICAL": 0.8,
            "EMERGENCY": 1.0
        }
        
        level_factor = sum(level_weights.get(alert.get("level", "INFO"), 0.1) 
                            for alert in recent_alerts) / max(len(recent_alerts), 1)
        
        # Factor de categoría
        category_weights = {
            "SECURITY": 0.9,
            "SYSTEM": 0.7,
            "PERFORMANCE": 0.6,
            "DATABASE": 0.6,
            "COMMUNICATION": 0.5,
            "MARKET": 0.4,
            "INTEGRATION": 0.5,
            "ANOMALY": 0.8
        }
        
        category_factor = sum(category_weights.get(alert.get("category", "SYSTEM"), 0.5) 
                               for alert in recent_alerts) / max(len(recent_alerts), 1)
        
        # Factor de tiempo (más peso a alertas recientes)
        if len(recent_alerts) > 1:
            now = time.time()
            time_weights = [(now - alert.get("timestamp", now - 3600)) / 3600 
                             for alert in recent_alerts]
            time_factor = sum(max(0, min(1, 1 - w)) for w in time_weights) / len(time_weights)
        else:
            time_factor = 0.5
        
        # Combinar factores
        risk_coefficient = (level_factor * 0.4 + category_factor * 0.3 + time_factor * 0.3)
        
        return min(max(risk_coefficient, 0.0), 1.0)  # Asegurar rango 0-1
    
    def detect_alert(self, source: str, metric_name: str, metric_value: float, 
                    context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Detectar alerta basada en métricas.
        
        Args:
            source: Fuente de la métrica (entidad, componente, etc.)
            metric_name: Nombre de la métrica
            metric_value: Valor de la métrica
            context: Contexto adicional
            
        Returns:
            Detalles de la alerta si se detectó, None en caso contrario
        """
        # Contexto por defecto
        if context is None:
            context = {}
        
        # Verificar si la métrica supera el umbral
        threshold = self.alert_thresholds.get(metric_name)
        if threshold is None:
            return None
        
        # Determinar si hay alerta
        is_alert = False
        if "direction" in context and context["direction"] == "lower":
            is_alert = metric_value < threshold
        else:
            is_alert = metric_value > threshold
        
        if not is_alert:
            return None
        
        # Determinar nivel de alerta
        level_ratios = {
            "INFO": 1.0,
            "WARNING": 1.2,
            "CRITICAL": 1.5,
            "EMERGENCY": 2.0
        }
        
        level = "INFO"
        for potential_level, ratio in level_ratios.items():
            if "direction" in context and context["direction"] == "lower":
                if metric_value < threshold / ratio:
                    level = potential_level
            else:
                if metric_value > threshold * ratio:
                    level = potential_level
        
        # Determinar categoría
        category = context.get("category", "SYSTEM")
        if not category in self.ALERT_CATEGORIES:
            category = "SYSTEM"
        
        # Crear alerta
        alert = {
            "source": source,
            "metric": metric_name,
            "value": metric_value,
            "threshold": threshold,
            "level": level,
            "category": category,
            "timestamp": time.time(),
            "context": context,
            "id": f"ALT-{int(time.time())}-{random.randint(1000, 9999)}"
        }
        
        # Registrar estadísticas
        self.stats["alerts_detected"] += 1
        self.stats["alerts_by_level"][level] = self.stats["alerts_by_level"].get(level, 0) + 1
        self.stats["alerts_by_category"][category] = self.stats["alerts_by_category"].get(category, 0) + 1
        
        # Añadir a la cola
        self.enqueue_alert(alert)
        
        # Añadir al historial
        self.add_to_history(alert)
        
        # Actualizar nivel de alerta del sistema
        self.update_system_alert_level()
        
        return alert
    
    def enqueue_alert(self, alert: Dict[str, Any]):
        """
        Añadir alerta a la cola de procesamiento.
        
        Args:
            alert: Datos de la alerta
        """
        # Añadir a la cola
        self.alert_queue.append(alert)
        
        # Limitar tamaño de la cola
        if len(self.alert_queue) > self.max_queue_size:
            self.alert_queue.pop(0)
    
    def add_to_history(self, alert: Dict[str, Any]):
        """
        Añadir alerta al historial.
        
        Args:
            alert: Datos de la alerta
        """
        # Versión simplificada para historial
        history_entry = {
            "id": alert["id"],
            "source": alert["source"],
            "metric": alert["metric"],
            "level": alert["level"],
            "category": alert["category"],
            "timestamp": alert["timestamp"]
        }
        
        # Añadir al historial
        self.alert_history.append(history_entry)
        
        # Limitar tamaño del historial
        if len(self.alert_history) > self.max_history_size:
            self.alert_history.pop(0)
    
    def update_system_alert_level(self):
        """Actualizar nivel de alerta general del sistema."""
        if not self.alert_history:
            self.system_alert_level = "NORMAL"
            return
        
        # Obtener alertas recientes (últimas 20)
        recent_alerts = self.alert_history[-20:]
        
        # Contar niveles
        level_counts = {level: 0 for level in self.ALERT_LEVELS}
        for alert in recent_alerts:
            level = alert.get("level", "INFO")
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Determinar nivel general
        if level_counts.get("EMERGENCY", 0) >= 1:
            self.system_alert_level = "CRITICAL"
        elif level_counts.get("CRITICAL", 0) >= 2:
            self.system_alert_level = "HIGH"
        elif level_counts.get("WARNING", 0) >= 3:
            self.system_alert_level = "ELEVATED"
        else:
            self.system_alert_level = "NORMAL"
    
    def process_alert_queue(self):
        """Procesar cola de alertas."""
        if not self.alert_queue:
            return
        
        # Procesar hasta 10 alertas por ciclo
        alerts_to_process = self.alert_queue[:10]
        self.alert_queue = self.alert_queue[10:]
        
        for alert in alerts_to_process:
            # Decidir si notificar
            should_notify = self.should_notify_alert(alert)
            
            if should_notify:
                self.notify_alert(alert)
                self.stats["alerts_notified"] += 1
            else:
                self.stats["alerts_suppressed"] += 1
            
            self.stats["alerts_processed"] += 1
    
    def should_notify_alert(self, alert: Dict[str, Any]) -> bool:
        """
        Determinar si una alerta debe ser notificada.
        
        Args:
            alert: Datos de la alerta
            
        Returns:
            True si se debe notificar, False en caso contrario
        """
        # Siempre notificar EMERGENCY
        if alert.get("level") == "EMERGENCY":
            return True
        
        # Siempre notificar CRITICAL
        if alert.get("level") == "CRITICAL":
            return True
        
        # Para WARNING, verificar si hay duplicados recientes
        if alert.get("level") == "WARNING":
            # Buscar alertas similares en las últimas 10
            recent_similar = 0
            for hist_alert in reversed(self.alert_history[-10:]):
                if (hist_alert.get("source") == alert.get("source") and
                    hist_alert.get("metric") == alert.get("metric") and
                    hist_alert.get("level") == alert.get("level")):
                    recent_similar += 1
            
            # Si hay más de 3 similares recientes, suprimir
            if recent_similar >= 3:
                return False
            
            return True
        
        # Para INFO, ser más selectivo
        if alert.get("level") == "INFO":
            # Solo notificar 20% de INFO aleatoriamente
            return random.random() < 0.2
        
        return True
    
    def notify_alert(self, alert: Dict[str, Any]):
        """
        Notificar alerta por los canales configurados.
        
        Args:
            alert: Datos de la alerta
        """
        # Formatear mensaje de alerta
        message = self.format_alert_message(alert)
        
        # Notificar por canales configurados
        for channel in self.notification_channels:
            if channel == "log":
                level_name = alert.get("level", "INFO")
                if level_name == "INFO":
                    logger.info(f"[{self.name}] ALERTA: {message}")
                elif level_name == "WARNING":
                    logger.warning(f"[{self.name}] ALERTA: {message}")
                elif level_name in ["CRITICAL", "EMERGENCY"]:
                    logger.error(f"[{self.name}] ALERTA: {message}")
            
            elif channel == "system_message":
                # Esto enviaría a una cola de mensajes del sistema
                pass
            
            elif channel == "entity_broadcast":
                # Enviar mensaje a la red de entidades
                alert_message = self.generate_message("alerta", message)
                self.broadcast_message(alert_message)
            
            elif channel == "email" and self.email_configured:
                # Solo para alertas críticas o de emergencia
                if alert.get("level") in ["CRITICAL", "EMERGENCY"]:
                    self.send_email_alert(alert)
    
    def format_alert_message(self, alert: Dict[str, Any]) -> str:
        """
        Formatear mensaje de alerta para notificación.
        
        Args:
            alert: Datos de la alerta
            
        Returns:
            Mensaje formateado
        """
        level = alert.get("level", "INFO")
        category = alert.get("category", "SYSTEM")
        source = alert.get("source", "unknown")
        metric = alert.get("metric", "unknown")
        value = alert.get("value", 0)
        threshold = alert.get("threshold", 0)
        
        # Formatear mensaje básico
        message = f"[{level}] {category}: {source} - {metric} = {value} (umbral: {threshold})"
        
        # Añadir contexto si existe
        context = alert.get("context", {})
        if context:
            context_str = "; ".join(f"{k}: {v}" for k, v in context.items() 
                                   if k not in ["category", "direction"])
            if context_str:
                message += f" | {context_str}"
        
        return message
    
    def send_email_alert(self, alert: Dict[str, Any]) -> bool:
        """
        Enviar alerta por email.
        
        Args:
            alert: Datos de la alerta
            
        Returns:
            True si se envió correctamente, False en caso contrario
        """
        if not self.email_configured:
            return False
        
        try:
            # Formatear mensaje HTML
            html_message = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; }}
                    .alert {{ padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    .info {{ background-color: #d1ecf1; border: 1px solid #bee5eb; }}
                    .warning {{ background-color: #fff3cd; border: 1px solid #ffeeba; }}
                    .critical {{ background-color: #f8d7da; border: 1px solid #f5c6cb; }}
                    .emergency {{ background-color: #dc3545; color: white; border: 1px solid #bd2130; }}
                    .details {{ margin-top: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="alert {alert.get('level', 'info').lower()}">
                    <h2>Alerta {alert.get('level', 'INFO')} - Sistema Genesis</h2>
                    <p>{self.format_alert_message(alert)}</p>
                </div>
                
                <div class="details">
                    <h3>Detalles</h3>
                    <table>
                        <tr><th>ID</th><td>{alert.get('id', 'N/A')}</td></tr>
                        <tr><th>Fuente</th><td>{alert.get('source', 'N/A')}</td></tr>
                        <tr><th>Métrica</th><td>{alert.get('metric', 'N/A')}</td></tr>
                        <tr><th>Valor</th><td>{alert.get('value', 'N/A')}</td></tr>
                        <tr><th>Umbral</th><td>{alert.get('threshold', 'N/A')}</td></tr>
                        <tr><th>Categoría</th><td>{alert.get('category', 'N/A')}</td></tr>
                        <tr><th>Nivel</th><td>{alert.get('level', 'N/A')}</td></tr>
                        <tr><th>Fecha/Hora</th><td>{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.get('timestamp', time.time())))}</td></tr>
                    </table>
                </div>
            </body>
            </html>
            """
            
            # Crear mensaje
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert.get('level')}] Alerta Sistema Genesis - {alert.get('category')}"
            msg['From'] = self.email_config["from_address"]
            msg['To'] = ", ".join(self.email_config["to_addresses"])
            
            # Adjuntar parte HTML
            part = MIMEText(html_message, 'html')
            msg.attach(part)
            
            # Enviar email
            server = smtplib.SMTP(self.email_config["smtp_server"], self.email_config["smtp_port"])
            server.starttls()
            server.login(self.email_config["username"], self.email_config["password"])
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"[{self.name}] Error enviando email: {str(e)}")
            return False
    
    def configure_email(self, config: Dict[str, Any]) -> bool:
        """
        Configurar canal de notificación email.
        
        Args:
            config: Configuración del email
            
        Returns:
            True si se configuró correctamente, False en caso contrario
        """
        required_fields = ["smtp_server", "smtp_port", "username", "password", 
                         "from_address", "to_addresses"]
        
        # Verificar que están todos los campos requeridos
        if not all(field in config for field in required_fields):
            logger.warning(f"[{self.name}] Configuración de email incompleta")
            return False
        
        # Guardar configuración
        self.email_config = config
        
        # Verificar que to_addresses es una lista
        if isinstance(self.email_config["to_addresses"], str):
            self.email_config["to_addresses"] = [self.email_config["to_addresses"]]
        
        # Marcar como configurado
        self.email_configured = True
        
        # Añadir canal si no existe
        if "email" not in self.notification_channels:
            self.notification_channels.append("email")
        
        logger.info(f"[{self.name}] Canal de email configurado correctamente")
        return True
    
    def simulate_random_alert(self):
        """Simular alerta aleatoria para pruebas."""
        # Fuentes posibles
        sources = ["Aetherion", "Lunareth", "System", "NetworkModule", "DatabaseConnector", 
                 "TradeEngine", "ApiGateway", "SchedulerService"]
        
        # Métricas posibles con rangos
        metrics = {
            "cpu_usage": (10, 100),  # %
            "memory_usage": (20, 100),  # %
            "api_latency": (50, 5000),  # ms
            "error_rate": (0, 30),  # %
            "transaction_anomaly": (0, 10),  # desviación estándar
            "connection_drops": (0, 20),  # count
            "price_volatility": (1, 20),  # %
            "disk_space": (50, 100)  # %
        }
        
        # Elegir fuente y métrica aleatorias
        source = random.choice(sources)
        metric_name = random.choice(list(metrics.keys()))
        min_val, max_val = metrics[metric_name]
        metric_value = random.uniform(min_val, max_val)
        
        # Contexto aleatorio
        context = {
            "category": random.choice(self.ALERT_CATEGORIES),
            "subsystem": random.choice(["Core", "Trading", "Analysis", "Communication", "Storage"]),
            "instance": f"inst-{random.randint(1, 5)}"
        }
        
        # Algunas métricas son mejores cuando son más bajas
        if metric_name in ["api_latency", "error_rate", "connection_drops"]:
            context["direction"] = "lower"
        
        # Detectar alerta
        alert = self.detect_alert(source, metric_name, metric_value, context)
        
        return alert
    
    def process_cycle(self):
        """
        Procesar ciclo de vida de la entidad de alertas.
        Sobreescribe el método de la clase base.
        """
        if not self.is_alive:
            return
        
        # Actualizar ciclo base
        super().process_base_cycle()
        
        # Ciclo específico de entidad de alertas
        try:
            # Procesar cola de alertas
            self.process_alert_queue()
            
            # Simular alertas aleatorias (probabilidad del 10%)
            if random.random() < 0.1:
                self.simulate_random_alert()
            
            # Actualizar estado
            self.update_state()
            
            # Generar mensaje informativo ocasionalmente (5% de probabilidad)
            if random.random() < 0.05:
                insight = self.generate_alert_insight()
                self.broadcast_message(insight)
                
        except Exception as e:
            logger.error(f"[{self.name}] Error en ciclo de proceso: {str(e)}")
            self.handle_error(str(e))
    
    def generate_alert_insight(self) -> str:
        """
        Generar insight sobre el estado de alertas.
        
        Returns:
            Mensaje con insight
        """
        insights = [
            f"Sistema en estado de alerta {self.system_alert_level}. Coeficiente de riesgo: {self.calculate_risk_coefficient():.2f}",
            f"He procesado {self.stats['alerts_processed']} alertas desde mi activación, incluyendo {self.stats['alerts_by_level'].get('CRITICAL', 0)} críticas.",
            f"En la última hora, la categoría con más alertas ha sido {max(self.stats['alerts_by_category'].items(), key=lambda x: x[1])[0]}.",
            f"El sistema presenta un comportamiento {['normal', 'ligeramente anómalo', 'preocupante', 'crítico'][min(3, max(0, ['NORMAL', 'ELEVATED', 'HIGH', 'CRITICAL'].index(self.system_alert_level)))]}.",
            f"Mi esencia {self.dominant_trait} me permite detectar patrones sutiles en el comportamiento del sistema."
        ]
        
        # Elegir un insight aleatorio
        insight = random.choice(insights)
        
        # Formatear como mensaje
        return self.generate_message("insight", insight)
    
    def handle_error(self, error_message: str):
        """
        Manejar error de alertas.
        
        Args:
            error_message: Mensaje de error
        """
        # Registrar error
        logger.error(f"[{self.name}] Error detectado: {error_message}")
        
        # Considerar este error como alerta
        self.detect_alert(
            source=self.name,
            metric_name="error_rate",
            metric_value=100,  # Error garantizado
            context={
                "category": "SYSTEM",
                "error_message": error_message[:100]
            }
        )
    
    def update_state(self):
        """Actualizar estado interno basado en métricas de alertas."""
        # Simulación de variación de estado basado en actividad
        energy_variation = 0
        
        # Perder energía por alertas procesadas
        energy_loss = self.stats["alerts_processed"] * 0.001
        energy_variation -= energy_loss
        
        # Ganar energía por alertas críticas (mayor atención)
        critical_alerts = self.stats["alerts_by_level"].get("CRITICAL", 0) + self.stats["alerts_by_level"].get("EMERGENCY", 0)
        if critical_alerts > 0:
            energy_variation += 0.2 * critical_alerts
        
        # Ajustar nivel basado en estadísticas
        level_adjustment = (
            self.stats["alerts_detected"] * 0.0001 -
            self.stats["false_positives"] * 0.001
        )
        
        # Añadir factor basado en nivel de alerta del sistema
        system_level_factor = {
            "NORMAL": 0.0,
            "ELEVATED": 0.05,
            "HIGH": 0.1,
            "CRITICAL": 0.2
        }.get(self.system_alert_level, 0.0)
        
        level_adjustment += system_level_factor
        
        # Aplicar cambios
        self.adjust_energy(energy_variation)
        self.adjust_level(level_adjustment)
        
        # Actualizar emoción basada en estado de alertas
        if self.system_alert_level == "CRITICAL":
            self.emotion = "Alarma"
        elif self.system_alert_level == "HIGH":
            self.emotion = "Alerta"
        elif self.system_alert_level == "ELEVATED":
            self.emotion = "Vigilancia"
        else:
            emotions = ["Calma", "Atento", "Sereno", "Concentrado"]
            self.emotion = random.choice(emotions)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado actual de la entidad para mostrar en UI.
        Extiende el método base con información específica de alertas.
        
        Returns:
            Diccionario con estado
        """
        base_status = super().get_status()
        
        # Añadir métricas específicas de alertas
        alert_status = {
            "system_alert_level": self.system_alert_level,
            "risk_coefficient": self.calculate_risk_coefficient(),
            "alerts_stats": self.stats,
            "notification_channels": self.notification_channels,
            "email_configured": self.email_configured,
            "recent_alerts": self.alert_history[-5:] if self.alert_history else [],
            "specializations": self.specializations
        }
        
        # Combinar estados
        combined_status = {**base_status, **alert_status}
        return combined_status


def create_alert_entity(name="Sentinel", father="otoniel", frequency_seconds=15):
    """
    Crear y configurar una entidad de alertas.
    
    Args:
        name: Nombre de la entidad
        father: Nombre del creador/dueño
        frequency_seconds: Período de ciclo de vida en segundos
        
    Returns:
        Instancia de AlertEntity
    """
    return AlertEntity(name, "Alert", father, frequency_seconds)

if __name__ == "__main__":
    # Prueba básica de la entidad
    sentinel = create_alert_entity()
    print(f"Entidad {sentinel.name} creada con rol {sentinel.role}")
    
    # Iniciar ciclo de vida en un hilo separado
    thread = threading.Thread(target=sentinel.start_lifecycle)
    thread.daemon = True
    thread.start()
    
    # Mantener vivo por un tiempo
    try:
        # Simular algunas alertas
        for i in range(5):
            time.sleep(2)
            print(f"Estado de {sentinel.name}: Energía={sentinel.energy:.1f}, Nivel={sentinel.level:.1f}, Emoción={sentinel.emotion}")
            alert = sentinel.simulate_random_alert()
            if alert:
                print(f"Alerta simulada: {sentinel.format_alert_message(alert)}")
    
    except KeyboardInterrupt:
        print("Deteniendo prueba...")
    finally:
        # Detener ciclo de vida
        sentinel.stop_lifecycle()
        print(f"Entidad {sentinel.name} detenida")