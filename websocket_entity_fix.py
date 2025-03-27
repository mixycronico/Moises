"""
Implementación de la corrección para entidades WebSocket del Sistema Genesis.

Este módulo corrige el error 'object has no attribute adjust_energy' 
que ocurre en las entidades LocalWebSocketEntity y ExternalWebSocketEntity.
"""

import logging
import time
from typing import Dict, Any

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def apply_websocket_entity_fix():
    """
    Aplica la corrección a las clases WebSocketEntity para agregar
    el método adjust_energy faltante.
    """
    try:
        # Importamos los módulos necesarios
        from websocket_entity import WebSocketEntity, LocalWebSocketEntity, ExternalWebSocketEntity
        
        # Verificamos que las clases existan
        if not hasattr(WebSocketEntity, 'adjust_energy'):
            # Agregamos el método adjust_energy a la clase base WebSocketEntity
            def adjust_energy(self, amount, reason=""):
                """
                Ajustar nivel de energía de la entidad.
                
                Args:
                    amount: Cantidad de energía a ajustar (positivo o negativo)
                    reason: Razón del ajuste de energía
                    
                Returns:
                    Nuevo nivel de energía
                """
                old_energy = self.energy
                
                # Aplicar ajuste de energía
                self.energy = max(0, min(100, self.energy + amount))
                
                # Registrar cambio significativo
                if abs(amount) >= 5:
                    direction = "ganó" if amount > 0 else "perdió"
                    abs_amount = abs(amount)
                    logger.info(f"[{self.name}] {direction} {abs_amount} de energía: {reason} [{old_energy} → {self.energy}]")
                    
                # Actualizar estado emocional basado en energía
                if self.energy < 20:
                    self.emotion = "Exhausto"
                    self.emotional_state = max(0.1, self.emotional_state - 0.1)
                elif self.energy > 80:
                    self.emotion = "Vigoroso"
                    self.emotional_state = min(1.0, self.emotional_state + 0.1)
                    
                return self.energy
            
            # Asignamos el método a la clase
            WebSocketEntity.adjust_energy = adjust_energy
            
            logger.info("✅ Método adjust_energy agregado correctamente a WebSocketEntity")
            
            return True
        else:
            logger.info("⚠️ El método adjust_energy ya existe en WebSocketEntity")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Error al importar módulos: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"❌ Error al aplicar la corrección: {str(e)}")
        return False

# Función para buscar la clase de entidad de reparación y corregirla si es necesario
def apply_repair_entity_fix():
    """
    Aplica la corrección a la clase RepairEntity para agregar
    el método start_lifecycle faltante.
    """
    try:
        # Importamos los módulos necesarios
        import importlib.util
        import os
        
        # Buscamos el archivo repair_entity.py
        repair_entity_file = None
        for root, dirs, files in os.walk('.'):
            if 'repair_entity.py' in files:
                repair_entity_file = os.path.join(root, 'repair_entity.py')
                break
        
        if not repair_entity_file:
            logger.error("❌ No se encontró el archivo repair_entity.py")
            return False
        
        # Cargamos el módulo dinámicamente
        spec = importlib.util.spec_from_file_location("repair_entity", repair_entity_file)
        repair_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(repair_module)
        
        # Verificamos que la clase exista
        if hasattr(repair_module, 'RepairEntity'):
            RepairEntity = repair_module.RepairEntity
            
            # Verificamos si ya tiene el método
            if not hasattr(RepairEntity, 'start_lifecycle'):
                def start_lifecycle(self):
                    """
                    Iniciar el ciclo de vida de la entidad de reparación.
                    Esta función inicia un hilo que ejecuta periódicamente
                    el ciclo de procesamiento de la entidad.
                    """
                    import threading
                    
                    if hasattr(self, '_lifecycle_thread') and self._lifecycle_thread.is_alive():
                        logger.warning(f"[{self.name}] El ciclo de vida ya está activo")
                        return False
                    
                    def lifecycle_loop():
                        logger.info(f"[{self.name}] Iniciando ciclo de vida")
                        while self.is_alive:
                            try:
                                self.process_cycle()
                            except Exception as e:
                                logger.error(f"[{self.name}] Error en ciclo de vida: {str(e)}")
                            time.sleep(self.frequency_seconds)
                    
                    self._lifecycle_thread = threading.Thread(target=lifecycle_loop)
                    self._lifecycle_thread.daemon = True
                    self._lifecycle_thread.start()
                    
                    logger.info(f"[{self.name}] Ciclo de vida iniciado")
                    return True
                
                # Asignamos el método a la clase
                RepairEntity.start_lifecycle = start_lifecycle
                
                logger.info("✅ Método start_lifecycle agregado correctamente a RepairEntity")
                return True
            else:
                logger.info("⚠️ El método start_lifecycle ya existe en RepairEntity")
                return False
        else:
            logger.error("❌ No se encontró la clase RepairEntity en el módulo")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error al aplicar la corrección a RepairEntity: {str(e)}")
        return False

if __name__ == "__main__":
    print("Aplicando correcciones al Sistema Genesis...")
    ws_fixed = apply_websocket_entity_fix()
    repair_fixed = apply_repair_entity_fix()
    
    if ws_fixed and repair_fixed:
        print("✅ Todas las correcciones aplicadas con éxito")
    elif ws_fixed:
        print("⚠️ Corrección de WebSocketEntity aplicada, pero hubo problemas con RepairEntity")
    elif repair_fixed:
        print("⚠️ Corrección de RepairEntity aplicada, pero hubo problemas con WebSocketEntity")
    else:
        print("❌ No se pudo aplicar ninguna corrección")