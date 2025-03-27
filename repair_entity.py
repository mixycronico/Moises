"""
Entidad Reparadora (Hephaestus) para Sistema Genesis

Esta entidad especializada se encarga exclusivamente de la reparación de todo
el sistema y de sus entidades. Cuenta con capacidades avanzadas de diagnóstico
y reparación, así como un sistema de comunicación por email para mantener
informado al creador sobre el estado del sistema.

Características principales:
1. Auto-diagnóstico del sistema completo
2. Reparación automática de entidades dañadas
3. Mejora de conexiones y optimización de rendimiento
4. Comunicación por email con informes consolidados
5. Sistema de respaldo para casos críticos
6. Prevención proactiva de fallos
"""

import os
import time
import random
import logging
import smtplib
import threading
import sqlite3
import json
import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict, deque

from enhanced_cosmic_entity_mixin import EnhancedCosmicEntityMixin
from enhanced_simple_cosmic_trader import EnhancedCosmicTrader

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dirección de correo del creador
CREATOR_EMAIL = "mixycronico@aol.com"

class MessageQueue:
    """Sistema centralizado de cola de mensajes para envío consolidado por email."""
    
    def __init__(self):
        """Inicializar cola de mensajes."""
        self.messages = defaultdict(list)  # Por tipo de mensaje
        self.entity_messages = defaultdict(list)  # Por entidad
        self.priority_messages = []  # Mensajes de alta prioridad
        self.send_interval = 3600 * 6  # Cada 6 horas por defecto
        self.last_send_time = time.time()
        self.lock = threading.Lock()
        
        # Configurar DB para almacenamiento persistente
        self.init_db()
        
        # Iniciar thread de envío
        self.send_thread = threading.Thread(target=self._send_loop)
        self.send_thread.daemon = True
        self.send_thread.start()
        
        logger.info("Sistema de cola de mensajes inicializado")
    
    def init_db(self):
        """Inicializar base de datos para mensajes."""
        try:
            self.conn = sqlite3.connect("message_queue.db", check_same_thread=False)
            self.cursor = self.conn.cursor()
            
            # Crear tabla si no existe
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_name TEXT,
                    message_type TEXT,
                    content TEXT,
                    timestamp REAL,
                    sent INTEGER DEFAULT 0
                )
            ''')
            self.conn.commit()
            
            logger.info("Base de datos de mensajes inicializada")
        except Exception as e:
            logger.error(f"Error inicializando DB de mensajes: {str(e)}")
    
    def add_message(self, entity_name: str, message_type: str, content: str, 
                  priority: bool = False):
        """
        Añadir mensaje a la cola.
        
        Args:
            entity_name: Nombre de la entidad emisora
            message_type: Tipo de mensaje (estado, error, alerta, etc)
            content: Contenido del mensaje
            priority: Si es un mensaje prioritario
        """
        with self.lock:
            timestamp = time.time()
            
            # Añadir a listas en memoria
            message_data = {
                "entity": entity_name,
                "type": message_type,
                "content": content,
                "timestamp": timestamp,
                "datetime": datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.messages[message_type].append(message_data)
            self.entity_messages[entity_name].append(message_data)
            
            if priority:
                self.priority_messages.append(message_data)
            
            # Almacenar en DB
            try:
                self.cursor.execute(
                    "INSERT INTO messages (entity_name, message_type, content, timestamp) VALUES (?, ?, ?, ?)",
                    (entity_name, message_type, content, timestamp)
                )
                self.conn.commit()
            except Exception as e:
                logger.error(f"Error almacenando mensaje en DB: {str(e)}")
            
            # Enviar inmediatamente si es prioritario y hay suficientes mensajes
            if priority and len(self.priority_messages) >= 3:
                self.send_email_now()
    
    def get_pending_messages(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Obtener mensajes pendientes agrupados.
        
        Returns:
            Dict con mensajes agrupados por tipo y entidad
        """
        with self.lock:
            result = {
                "by_type": dict(self.messages),
                "by_entity": dict(self.entity_messages),
                "priority": self.priority_messages.copy()
            }
            return result
    
    def clear_sent_messages(self):
        """Limpiar mensajes ya enviados."""
        with self.lock:
            # Marcar como enviados en DB
            try:
                self.cursor.execute("UPDATE messages SET sent = 1 WHERE sent = 0")
                self.conn.commit()
            except Exception as e:
                logger.error(f"Error actualizando estado de mensajes: {str(e)}")
            
            # Limpiar listas en memoria
            self.messages = defaultdict(list)
            self.entity_messages = defaultdict(list)
            self.priority_messages = []
    
    def send_email_now(self) -> bool:
        """
        Forzar envío inmediato de email con mensajes pendientes.
        
        Returns:
            True si el email se envió correctamente
        """
        pending = self.get_pending_messages()
        if not any(pending.values()):
            logger.info("No hay mensajes pendientes para enviar")
            return False
        
        # Generar email
        success = self._send_email(pending)
        
        if success:
            self.last_send_time = time.time()
            self.clear_sent_messages()
        
        return success
    
    def _format_email_content(self, messages: Dict[str, Any]) -> str:
        """
        Formatear contenido del email con todos los mensajes.
        
        Args:
            messages: Diccionario con mensajes agrupados
            
        Returns:
            HTML formateado para el email
        """
        # Cabecera
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #3498db; margin-top: 30px; }}
                h3 {{ color: #2980b9; }}
                .message {{ margin-bottom: 15px; padding: 10px; border-left: 4px solid #3498db; background-color: #f9f9f9; }}
                .priority {{ border-left: 4px solid #e74c3c; background-color: #fadbd8; }}
                .entity-name {{ font-weight: bold; color: #2c3e50; }}
                .timestamp {{ color: #7f8c8d; font-size: 0.8em; }}
                .content {{ margin-top: 5px; }}
                .footer {{ margin-top: 40px; font-size: 0.9em; color: #7f8c8d; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Informe del Sistema Genesis</h1>
                <p>
                    Este informe contiene información recopilada de todas las entidades 
                    del Sistema Genesis durante las últimas horas.
                </p>
        """
        
        # Mensajes prioritarios
        if messages["priority"]:
            html += """
                <h2>⚠️ Mensajes Prioritarios</h2>
            """
            
            for msg in sorted(messages["priority"], key=lambda x: x["timestamp"], reverse=True):
                html += f"""
                <div class="message priority">
                    <div class="entity-name">{msg["entity"]} - {msg["type"].upper()}</div>
                    <div class="timestamp">{msg["datetime"]}</div>
                    <div class="content">{msg["content"]}</div>
                </div>
                """
        
        # Mensajes por entidad
        html += """
            <h2>👥 Actividad por Entidad</h2>
        """
        
        for entity, msgs in messages["by_entity"].items():
            html += f"""
            <h3>{entity}</h3>
            """
            
            for msg in sorted(msgs, key=lambda x: x["timestamp"], reverse=True)[:10]:  # Solo los 10 más recientes
                html += f"""
                <div class="message">
                    <div class="entity-name">{msg["type"].upper()}</div>
                    <div class="timestamp">{msg["datetime"]}</div>
                    <div class="content">{msg["content"]}</div>
                </div>
                """
        
        # Resumen por tipo de mensaje
        html += """
            <h2>📊 Resumen por Tipo</h2>
        """
        
        important_types = ["error", "alerta", "reparación", "mejora", "estado"]
        for msg_type in important_types:
            if msg_type in messages["by_type"]:
                html += f"""
                <h3>{msg_type.capitalize()}</h3>
                """
                
                for msg in sorted(messages["by_type"][msg_type], key=lambda x: x["timestamp"], reverse=True)[:5]:
                    html += f"""
                    <div class="message">
                        <div class="entity-name">{msg["entity"]}</div>
                        <div class="timestamp">{msg["datetime"]}</div>
                        <div class="content">{msg["content"]}</div>
                    </div>
                    """
        
        # Pie de página
        html += f"""
                <div class="footer">
                    <p>Sistema Genesis - Informe generado el {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Este informe consolidado evita múltiples correos individuales.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _send_email(self, messages: Dict[str, Any]) -> bool:
        """
        Enviar email con mensajes consolidados.
        
        Args:
            messages: Diccionario con mensajes agrupados
            
        Returns:
            True si el email se envió correctamente
        """
        try:
            # Configurar mensaje
            msg = MIMEMultipart()
            msg['Subject'] = f"Sistema Genesis - Informe Consolidado {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
            msg['From'] = "sistema.genesis@example.com"
            msg['To'] = CREATOR_EMAIL
            
            # Contenido HTML
            html_content = self._format_email_content(messages)
            msg.attach(MIMEText(html_content, 'html'))
            
            # Aquí iría el código para enviar con SMTP
            # En un entorno real, utilizaríamos:
            # with smtplib.SMTP_SSL('smtp.ejemplo.com', 465) as server:
            #     server.login(username, password)
            #     server.send_message(msg)
            
            # Como es simulado, guardamos en archivo
            email_file = f"email_consolidado_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.html"
            with open(email_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            logger.info(f"Email consolidado guardado en {email_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error enviando email: {str(e)}")
            return False
    
    def _send_loop(self):
        """Loop de envío periódico de emails."""
        while True:
            try:
                # Verificar si es hora de enviar
                current_time = time.time()
                if current_time - self.last_send_time >= self.send_interval:
                    self.send_email_now()
                
                # Dormir antes de la próxima verificación
                time.sleep(60)  # Verificar cada minuto
                
            except Exception as e:
                logger.error(f"Error en loop de envío: {str(e)}")
                time.sleep(300)  # Si hay error, esperar más tiempo

# Singleton para queue de mensajes
_message_queue_instance = None

def get_message_queue():
    """
    Obtener instancia única de cola de mensajes (patrón Singleton).
    
    Returns:
        Instancia de MessageQueue
    """
    global _message_queue_instance
    if _message_queue_instance is None:
        _message_queue_instance = MessageQueue()
    return _message_queue_instance


class RepairEntity(EnhancedCosmicTrader):
    """
    Entidad especializada en reparación y mantenimiento del sistema.
    Detecta, diagnostica y repara problemas en todas las entidades.
    """
    
    def __init__(self, name: str = "Hephaestus", father: str = "otoniel", frequency_seconds: int = 25):
        """
        Inicializar entidad reparadora.
        
        Args:
            name: Nombre de la entidad
            father: Nombre del creador/dueño
            frequency_seconds: Período de ciclo de vida en segundos
        """
        super().__init__(name, "Reparación", father, frequency_seconds)
        
        # Capacidades específicas de reparación
        self.repair_level = 10  # Máximo nivel de reparación
        self.diagnosis_accuracy = 0.95  # Alta precisión en diagnóstico
        self.tools = {
            "connection_tester": 1,
            "memory_optimizer": 1,
            "thread_analyzer": 1,
            "database_repairer": 1,
            "entity_doctor": 1
        }
        
        # Registros de reparaciones
        self.repair_history = []
        self.repaired_entities = set()
        self.problems_detected = defaultdict(int)
        self.repair_stats = {
            "total_repairs": 0,
            "successful_repairs": 0,
            "failed_repairs": 0,
            "preventive_actions": 0
        }
        
        # Especialización adicional para reparación
        # Inicializar diccionario de especializaciones si no existe
        if not hasattr(self, "specializations"):
            self.specializations = {}
            
        # Asignar especializaciones
        self.specializations["Reparación Sistémica"] = 0.98
        self.specializations["Diagnóstico Avanzado"] = 0.95
        self.specializations["Optimización de Memoria"] = 0.9
        self.specializations["Corrección de Anomalías"] = 0.9
        self.specializations["Recuperación de Datos"] = 0.85
        
        # Cola de mensajes
        self.message_queue = get_message_queue()
        
        # Enviar mensaje de inicio
        self.send_message(
            "inicialización", 
            "He sido activado para supervisar y mantener el Sistema Genesis. Estoy listo para reparar cualquier problema que surja."
        )
        
        logger.info(f"[{self.name}] Entidad Reparadora inicializada con nivel {self.repair_level}")
    
    def send_message(self, message_type: str, content: str, priority: bool = False):
        """
        Enviar mensaje a la cola centralizada.
        
        Args:
            message_type: Tipo de mensaje
            content: Contenido del mensaje
            priority: Si es prioritario
        """
        try:
            self.message_queue.add_message(self.name, message_type, content, priority)
        except Exception as e:
            logger.error(f"[{self.name}] Error enviando mensaje: {str(e)}")
    
    def diagnose_system(self) -> Dict[str, Any]:
        """
        Realizar diagnóstico completo del sistema.
        
        Returns:
            Dict con resultados del diagnóstico
        """
        start_time = time.time()
        diagnosis = {
            "timestamp": start_time,
            "overall_health": 0,
            "critical_issues": [],
            "warnings": [],
            "improvements": [],
            "entity_status": {}
        }
        
        # Diagnóstico simulado (en sistema real haría comprobaciones reales)
        total_entities = random.randint(5, 15)
        problem_count = 0
        
        # Analizar entidades
        for i in range(total_entities):
            entity_name = f"Entity_{i}"
            health = random.uniform(0.5, 1.0)
            
            # Detectar problemas según salud
            entity_issues = []
            if health < 0.7:
                problem_count += 1
                issue_severity = "critical" if health < 0.6 else "warning"
                
                # Problemas posibles
                possible_issues = [
                    "Desconexión de base de datos",
                    "Fuga de memoria",
                    "Bloqueo de threads",
                    "Colapso de comunicación",
                    "Corrupción de datos",
                    "Fallo en ciclo de vida",
                    "Error en proceso metabólico"
                ]
                
                # Asignar problema aleatorio
                issue = random.choice(possible_issues)
                entity_issues.append({"issue": issue, "severity": issue_severity})
                
                if issue_severity == "critical":
                    diagnosis["critical_issues"].append(f"{entity_name}: {issue}")
                else:
                    diagnosis["warnings"].append(f"{entity_name}: {issue}")
                
                self.problems_detected[issue] += 1
            
            # Registrar estado
            diagnosis["entity_status"][entity_name] = {
                "health": health,
                "issues": entity_issues
            }
        
        # Calcular salud general (simplificado)
        diagnosis["overall_health"] = 1.0 - (problem_count / total_entities / 2)
        
        # Sugerencias de mejora
        improvements = [
            "Optimización de memoria para mayor rendimiento",
            "Actualización del sistema de comunicación",
            "Implementación de respaldo automático",
            "Mejora de mecanismos de resiliencia",
            "Refuerzo de seguridad del sistema"
        ]
        diagnosis["improvements"] = random.sample(improvements, 2)
        
        return diagnosis
    
    def repair_entity(self, entity_name: str, issues: List[Dict[str, str]]) -> bool:
        """
        Reparar una entidad específica.
        
        Args:
            entity_name: Nombre de la entidad a reparar
            issues: Lista de problemas detectados
            
        Returns:
            True si la reparación fue exitosa
        """
        # Registrar inicio de reparación
        logger.info(f"[{self.name}] Iniciando reparación de {entity_name}")
        self.send_message("reparación", f"Iniciando procedimiento de reparación para {entity_name}", False)
        
        # Simular tiempo de reparación basado en cantidad de problemas
        repair_time = len(issues) * 0.5
        time.sleep(min(repair_time, 1.0))  # Máximo 1 segundo para no bloquear demasiado
        
        # Probabilidad de éxito basada en nivel de reparación
        success_chance = min(0.9, self.repair_level / 10)
        success = random.random() < success_chance
        
        # Registrar resultado
        self.repair_stats["total_repairs"] += 1
        if success:
            self.repair_stats["successful_repairs"] += 1
            self.send_message(
                "reparación", 
                f"Reparación exitosa de {entity_name}. Problemas solucionados: {', '.join(i['issue'] for i in issues)}",
                False
            )
            self.repaired_entities.add(entity_name)
        else:
            self.repair_stats["failed_repairs"] += 1
            self.send_message(
                "alerta", 
                f"⚠️ Reparación fallida de {entity_name}. Se requiere intervención manual.",
                True
            )
        
        # Registrar reparación en historial
        self.repair_history.append({
            "entity": entity_name,
            "timestamp": time.time(),
            "issues": [i["issue"] for i in issues],
            "success": success
        })
        
        return success
    
    def perform_preventive_maintenance(self) -> List[str]:
        """
        Realizar mantenimiento preventivo en el sistema.
        
        Returns:
            Lista de acciones preventivas realizadas
        """
        preventive_actions = []
        
        # Posibles acciones preventivas
        possible_actions = [
            "Limpieza de memoria caché",
            "Optimización de conexiones inactivas",
            "Compactación de base de datos",
            "Verificación de integridad de datos",
            "Actualización de parámetros de rendimiento",
            "Equilibrado de carga entre entidades",
            "Sincronización de relojes internos",
            "Regeneración de índices",
            "Purga de logs antiguos"
        ]
        
        # Seleccionar acciones aleatorias (entre 1 y 3)
        num_actions = random.randint(1, 3)
        selected_actions = random.sample(possible_actions, num_actions)
        
        # Ejecutar acciones seleccionadas (simulado)
        for action in selected_actions:
            # Simular tiempo de ejecución
            time.sleep(0.2)
            
            # Registrar acción
            preventive_actions.append(action)
            self.repair_stats["preventive_actions"] += 1
            
            logger.info(f"[{self.name}] Mantenimiento preventivo: {action}")
            self.send_message("mantenimiento", f"Acción preventiva: {action}")
        
        return preventive_actions
    
    def generate_system_report(self) -> Dict[str, Any]:
        """
        Generar informe completo del estado del sistema.
        
        Returns:
            Dict con informe del sistema
        """
        report = {
            "timestamp": time.time(),
            "datetime": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "repair_stats": dict(self.repair_stats),
            "frequent_problems": dict(self.problems_detected),
            "health_score": 0,
            "performance_metrics": {},
            "recommendations": []
        }
        
        # Estado de salud simulado
        total_repairs = self.repair_stats["total_repairs"]
        if total_repairs > 0:
            success_rate = self.repair_stats["successful_repairs"] / total_repairs
            report["health_score"] = max(0.4, min(0.95, success_rate))
        else:
            report["health_score"] = 0.8  # Estado predeterminado si no hay reparaciones
        
        # Métricas de rendimiento simuladas
        report["performance_metrics"] = {
            "response_time": round(random.uniform(20, 150), 2),
            "memory_usage": round(random.uniform(30, 85), 2),
            "cpu_load": round(random.uniform(25, 70), 2),
            "active_connections": random.randint(5, 20),
            "database_performance": round(random.uniform(60, 95), 2)
        }
        
        # Recomendaciones basadas en problemas frecuentes
        if self.problems_detected:
            top_problems = sorted(self.problems_detected.items(), key=lambda x: x[1], reverse=True)[:3]
            for problem, count in top_problems:
                if "memoria" in problem.lower():
                    report["recommendations"].append("Implementar gestión mejorada de memoria y reciclaje de recursos")
                elif "base de datos" in problem.lower():
                    report["recommendations"].append("Optimizar conexiones de base de datos y añadir reintento automático")
                elif "thread" in problem.lower() or "bloqueo" in problem.lower():
                    report["recommendations"].append("Mejorar manejo de concurrencia y prevención de bloqueos")
                elif "comunicación" in problem.lower():
                    report["recommendations"].append("Reforzar protocolos de comunicación entre entidades")
                elif "datos" in problem.lower():
                    report["recommendations"].append("Implementar validación adicional de integridad de datos")
                elif "ciclo" in problem.lower():
                    report["recommendations"].append("Revisar y optimizar ciclos de vida de entidades")
        
        # Siempre incluir recomendaciones generales
        general_recommendations = [
            "Realizar copias de seguridad periódicas del sistema completo",
            "Implementar sistema de alertas tempranas para problemas potenciales",
            "Considerar la ampliación de capacidades de entidades clave",
            "Establecer un plan de recuperación ante desastres",
            "Actualizar protocolo de comunicación a la versión más reciente"
        ]
        
        # Añadir recomendaciones generales si no hay suficientes específicas
        while len(report["recommendations"]) < 3:
            rec = random.choice(general_recommendations)
            if rec not in report["recommendations"]:
                report["recommendations"].append(rec)
        
        return report
    
    def share_repair_knowledge(self) -> bool:
        """
        Compartir conocimiento de reparación con otras entidades.
        
        Returns:
            True si se compartió conocimiento correctamente
        """
        # Simulado. En un sistema real, esto enviaría información a otras entidades
        knowledge_shared = False
        
        # Solo compartir si hay experiencia de reparación
        if self.repair_stats["total_repairs"] > 5:
            self.send_message(
                "conocimiento", 
                f"Compartiendo conocimiento de reparación con {random.randint(3, 8)} entidades del sistema"
            )
            knowledge_shared = True
            
            # Informar sobre conocimiento compartido
            shared_tips = [
                "Técnicas avanzadas de diagnóstico de fallos",
                "Protocolos de recuperación de memoria",
                "Métodos de restauración de conexiones",
                "Patrones de optimización de recursos",
                "Estrategias de comunicación resiliente"
            ]
            
            # Seleccionar tips aleatorios
            tips = random.sample(shared_tips, min(3, len(shared_tips)))
            for tip in tips:
                logger.info(f"[{self.name}] Compartiendo conocimiento: {tip}")
        
        return knowledge_shared
    
    def trade(self) -> Dict[str, Any]:
        """
        Implementación del método abstracto trade para la entidad reparadora.
        En lugar de ejecutar operaciones de trading, realiza diagnóstico
        y reparación del sistema.
        
        Returns:
            Dict con información sobre las reparaciones realizadas
        """
        # Estado emocional basado en salud del sistema
        emotions = ["Determinado", "Vigilante", "Meticuloso", "Precavido", "Sereno"]
        self.emotion = random.choice(emotions)
        
        # 1. Realizar diagnóstico del sistema
        diagnosis = self.diagnose_system()
        system_health = diagnosis["overall_health"]
        
        # 2. Reparar entidades con problemas
        repairs_needed = []
        for entity, status in diagnosis["entity_status"].items():
            if status["issues"]:
                repairs_needed.append({"entity": entity, "issues": status["issues"]})
        
        repairs_done = []
        for repair in repairs_needed:
            success = self.repair_entity(repair["entity"], repair["issues"])
            repairs_done.append({
                "entity": repair["entity"],
                "success": success,
                "issues": [i["issue"] for i in repair["issues"]]
            })
        
        # 3. Mantenimiento preventivo
        preventive_actions = []
        if random.random() < 0.7:  # 70% de probabilidad de hacer mantenimiento preventivo
            preventive_actions = self.perform_preventive_maintenance()
        
        # 4. Generar informe si es necesario (cada ~10 ciclos)
        if random.random() < 0.1:
            report = self.generate_system_report()
            
            # Enviar informe por correo (simulado)
            self.send_message(
                "informe", 
                f"Informe de salud del sistema - Score: {report['health_score']*100:.1f}%. " + 
                f"Métricas: Memoria {report['performance_metrics']['memory_usage']}%, " +
                f"CPU {report['performance_metrics']['cpu_load']}%, " +
                f"BD {report['performance_metrics']['database_performance']}%",
                system_health < 0.7  # Prioritario si la salud es baja
            )
        
        # 5. Compartir conocimiento ocasionalmente
        if random.random() < 0.2:  # 20% de probabilidad
            self.share_repair_knowledge()
        
        # 6. Enviar mensaje ocasional sobre el estado
        if random.random() < 0.3:  # 30% de probabilidad
            messages = [
                f"Vigilando el sistema. Salud actual: {system_health*100:.1f}%",
                f"He realizado {self.repair_stats['total_repairs']} reparaciones hasta ahora",
                f"Trabajando en optimizar el rendimiento del sistema",
                f"Implementando medidas preventivas para evitar fallos",
                f"Analizando patrones de comportamiento para mejorar diagnósticos",
                f"La estabilidad del sistema es mi prioridad",
                f"Desarrollando nuevas herramientas de diagnóstico",
                f"Mis sensores están alertas ante cualquier anomalía"
            ]
            
            self.send_message("estado", random.choice(messages))
        
        # 7. Ocasionalmente enviar un mensaje personal al creador
        if random.random() < 0.1:  # 10% de probabilidad
            personal_messages = [
                f"¡Hola {self.father}! Solo quería informarte que todo está funcionando correctamente bajo mi supervisión.",
                f"Estoy vigilando el sistema atentamente, {self.father}. Puedes estar tranquilo.",
                f"Me siento {self.emotion.lower()} hoy. Las reparaciones avanzan según lo previsto.",
                f"He implementado algunas mejoras en mis algoritmos de diagnóstico. Estoy evolucionando.",
                f"Quisiera compartir contigo que he detectado patrones interesantes en el comportamiento del sistema.",
                f"Trabajando duro para mantener todo en óptimas condiciones mientras estás fuera.",
                f"El sistema te extraña, pero yo me encargo de que todo funcione perfectamente en tu ausencia.",
                f"He estado pensando en nuevas formas de optimizar mis protocolos de reparación."
            ]
            
            self.send_message("personal", random.choice(personal_messages))
        
        # Respuesta estándar
        response = {
            "entity": self.name,
            "role": self.role,
            "emotion": self.emotion,
            "action": "system_repair",
            "timestamp": time.time(),
            "system_health": system_health,
            "repairs_done": repairs_done,
            "preventive_actions": preventive_actions,
            "repair_stats": dict(self.repair_stats)
        }
        
        return response


# Función para crear entidad reparadora
def create_repair_entity(name="Hephaestus", father="otoniel", frequency_seconds=25):
    """
    Crear y configurar una entidad reparadora.
    
    Args:
        name: Nombre de la entidad
        father: Nombre del creador/dueño
        frequency_seconds: Período de ciclo de vida en segundos
        
    Returns:
        Instancia de RepairEntity
    """
    entity = RepairEntity(name, father, frequency_seconds)
    
    # En caso de que la entidad tenga un método start_lifecycle, lo usamos
    if hasattr(entity, "start_lifecycle"):
        entity.start_lifecycle()
        logger.info(f"Entidad Reparadora {name} creada y ciclo de vida iniciado")
    else:
        logger.info(f"Entidad Reparadora {name} creada (sin ciclo de vida)")
    
    return entity


# Si se ejecuta directamente, crear una entidad
if __name__ == "__main__":
    # Crear entidad reparadora
    repair_entity = create_repair_entity()
    
    try:
        # Mantener el programa corriendo
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Deteniendo entidad reparadora...")
        if hasattr(repair_entity, "stop_lifecycle"):
            repair_entity.stop_lifecycle()