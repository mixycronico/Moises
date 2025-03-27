"""
Mixin para añadir métodos comunes a todas las entidades especializadas del Sistema Genesis.
Este módulo proporciona los métodos esenciales que las entidades especializadas deben implementar
para funcionar correctamente en el sistema de trading cósmico.
"""

import time
import random
import threading
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedCosmicEntityMixin:
    """
    Mixin que proporciona métodos comunes a todas las entidades especializadas.
    """
    
    def start_lifecycle(self):
        """Iniciar ciclo de vida de la entidad en un hilo separado."""
        logger.info(f"[{self.name}] Ciclo de vida iniciado")
        self.is_alive = True
        self.lifecycle_thread = threading.Thread(target=self._lifecycle_loop)
        self.lifecycle_thread.daemon = True
        self.lifecycle_thread.start()
        return True
    
    def stop_lifecycle(self):
        """Detener ciclo de vida de la entidad."""
        logger.info(f"[{self.name}] Ciclo de vida detenido")
        self.is_alive = False
        if hasattr(self, 'lifecycle_thread') and self.lifecycle_thread.is_alive():
            # Esperar a que termine el hilo (con timeout)
            self.lifecycle_thread.join(timeout=2.0)
        return True
    
    def _lifecycle_loop(self):
        """Bucle principal del ciclo de vida."""
        try:
            while self.is_alive:
                try:
                    # Procesar ciclo de vida
                    self.process_cycle()
                    
                    # Dormir hasta el siguiente ciclo
                    time.sleep(self.frequency_seconds)
                    
                except Exception as e:
                    logger.error(f"[{self.name}] Error en ciclo de vida: {str(e)}")
                    # Dar tiempo al sistema para recuperarse
                    logger.warning(f"[{self.name}] Regeneración de emergencia: +10 energía")
                    time.sleep(1)
        except Exception as e:
            logger.error(f"[{self.name}] Error fatal en ciclo de vida: {str(e)}")
    
    def process_base_cycle(self):
        """
        Procesar ciclo base común a todas las entidades.
        Este método debe ser llamado desde process_cycle() en cada entidad especializada.
        """
        # Consumo base de energía
        self.energy -= self.energy_rate
        
        # Verificar si la entidad sigue con vida
        if self.energy <= 0:
            self.energy = 0
            self.is_alive = False
            logger.warning(f"[{self.name}] Sin energía: ciclo de vida terminado")
            return False
        
        # Proceso continuo de consolidación de conocimiento (con baja probabilidad)
        if random.random() < 0.15:  # 15% de probabilidad en cada ciclo
            consolidation_result = self.consolidar_conocimiento()
            
            # Si hay descubrimiento importante, transmitir insight
            if consolidation_result.get("descubrimiento") and consolidation_result.get("mensaje_insight"):
                # Incremento de energía por descubrimiento
                energy_boost = random.uniform(1.0, 3.0)
                self.adjust_energy(energy_boost)
                
                # Log si el conocimiento supera ciertos umbrales
                if self.knowledge > 3.0 and random.random() < 0.3:
                    logger.info(f"[{self.name}] Conocimiento acumulado significativo: {self.knowledge:.2f}")
        
        # Evolución natural si se acumula suficiente conocimiento
        if hasattr(self, 'knowledge') and getattr(self, 'experience', 0) > 0:
            if self.knowledge > 5.0 and random.random() < 0.05:  # 5% de probabilidad con conocimiento > 5
                level_boost = random.uniform(0.05, 0.1)
                self.adjust_level(level_boost)
                logger.info(f"[{self.name}] Evolución natural por conocimiento acumulado: +{level_boost:.3f} niveles")
                
                # Notificar evolución
                if hasattr(self, 'generate_message') and hasattr(self, 'broadcast_message'):
                    evolution_message = self.generate_message(
                        "evolución", 
                        f"He evolucionado naturalmente gracias al conocimiento acumulado. Nivel: {self.level:.2f}"
                    )
                    self.broadcast_message(evolution_message)
        
        return True
    
    def adjust_energy(self, amount):
        """
        Ajustar nivel de energía de la entidad.
        
        Args:
            amount: Cantidad a ajustar (positiva o negativa)
        """
        self.energy += amount
        
        # Limitar a rango válido
        self.energy = max(0, min(self.energy, 150))
        
        # Verificar si murió por falta de energía
        if self.energy <= 0:
            self.is_alive = False
            logger.warning(f"[{self.name}] Sin energía: ciclo de vida terminado")
        
        return self.energy
    
    def adjust_level(self, amount):
        """
        Ajustar nivel de la entidad.
        
        Args:
            amount: Cantidad a ajustar (positiva o negativa)
        """
        self.level += amount
        
        # Limitar a rango válido (mínimo 1)
        self.level = max(1.0, self.level)
        
        return self.level
    
    def generate_message(self, tipo, contenido):
        """
        Generar mensaje formateado para comunicación.
        
        Args:
            tipo: Tipo de mensaje
            contenido: Contenido del mensaje
            
        Returns:
            Mensaje formateado
        """
        return {
            "sender": self.name,
            "type": tipo,
            "content": contenido,
            "timestamp": time.time(),
            "emotion": getattr(self, "emotion", "Neutral"),
            "level": self.level
        }
    
    def broadcast_message(self, message):
        """
        Enviar mensaje a la red cósmica.
        
        Args:
            message: Mensaje a enviar
            
        Returns:
            True si se envió correctamente, False en caso contrario
        """
        # Verificar si la entidad está conectada a una red
        if hasattr(self, "network") and self.network:
            try:
                self.network.broadcast(self.name, message)
                return True
            except Exception as e:
                logger.error(f"[{self.name}] Error al transmitir mensaje: {str(e)}")
                return False
        else:
            logger.warning(f"[{self.name}] No conectado a ninguna red, mensaje no enviado")
            return False

    def consolidar_conocimiento(self):
        """
        Consolida el conocimiento de la entidad basándose en su experiencia y memoria.
        Este proceso ocurre continuamente en el ciclo de vida de la entidad.
        
        Efectos:
        - Aumenta el nivel de conocimiento
        - Puede desbloquear nuevas capacidades
        - Mejora la eficiencia operativa
        - Posibilita insights y descubrimientos
        
        Returns:
            Diccionario con resultados del proceso de consolidación
        """
        # Base de conocimiento previo
        conocimiento_previo = self.knowledge
        
        # Factores que influyen en la consolidación
        memoria_factor = min(1.0, len(getattr(self, 'memory', [])) / 100) * 0.3
        experiencia_factor = min(1.0, getattr(self, 'experience', 0) / 1000) * 0.4
        nivel_factor = (self.level - 1) * 0.1
        especialidad_factor = random.uniform(0.05, 0.15)  # Factor aleatorio basado en especialidad
        
        # Calcular incremento de conocimiento
        incremento_base = 0.02 + random.uniform(0, 0.01)  # Base más variación aleatoria
        incremento_total = incremento_base * (1 + memoria_factor + experiencia_factor + nivel_factor + especialidad_factor)
        
        # Efecto de sinergia si hay otras entidades compartiendo conocimiento
        if hasattr(self, 'network') and self.network:
            entidades_conectadas = len(getattr(self.network, 'entities', []))
            if entidades_conectadas > 1:
                factor_sinergia = min(0.5, (entidades_conectadas - 1) * 0.05)
                incremento_total *= (1 + factor_sinergia)
        
        # Aplicar incremento
        self.knowledge += incremento_total
        
        # Posibilidad de descubrimiento importante (insight)
        descubrimiento = False
        mensaje_insight = None
        if random.random() < 0.05:  # 5% de probabilidad
            descubrimiento = True
            insights = [
                f"He descubierto un patrón sutil en los datos que procesamos.",
                f"La correlación entre {getattr(self, 'role', 'mi función')} y el rendimiento general parece significativa.",
                f"Mi especialidad en {getattr(self, 'role', 'proceso')} está evolucionando hacia un nivel más profundo de comprensión.",
                f"Las conexiones entre entidades muestran un potencial de optimización inexplorado.",
                f"He identificado una posible mejora en nuestros procesos de comunicación colectiva."
            ]
            mensaje_insight = random.choice(insights)
            
            # Compartir el insight con la red
            if hasattr(self, 'network') and self.network and mensaje_insight:
                mensaje = self.generate_message("insight", mensaje_insight)
                self.broadcast_message(mensaje)
        
        # Resultados
        resultados = {
            "conocimiento_previo": conocimiento_previo,
            "conocimiento_actual": self.knowledge,
            "incremento": incremento_total,
            "descubrimiento": descubrimiento,
            "mensaje_insight": mensaje_insight
        }
        
        return resultados