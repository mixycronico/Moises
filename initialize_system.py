"""
Script de inicialización del Sistema Genesis completo con:
- Entidad reparadora
- Sistema de mensajes centralizado
- Conectores para todas las entidades existentes

Ejecute este script para aplicar todas las mejoras a las entidades existentes
y habilitar las características de reparación automática y sistema de mensajes
consolidados por email.
"""

import os
import time
import logging
import threading
import importlib
from typing import List, Dict, Any, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importar módulos necesarios
from main_module_connector import (
    initialize_system, apply_connectors_to_entity, get_genesis_connector
)
from message_collector import send_system_message
from repair_entity import create_repair_entity

# Email del creador
CREATOR_EMAIL = "mixycronico@aol.com"

def locate_and_import_entities():
    """
    Localizar y importar todas las entidades existentes en el sistema.
    
    Returns:
        List de entidades encontradas
    """
    entities = []
    entity_modules = [
        # Lista de módulos donde buscar entidades
        "cosmic_trading",
        "enhanced_simple_cosmic_trader",
        "websocket_entity_fix",
        "database_entity",
        "integration_entity",
        "alert_entity"
    ]
    
    for module_name in entity_modules:
        try:
            # Importar módulo
            module = importlib.import_module(module_name)
            logger.info(f"Módulo {module_name} importado correctamente")
            
            # Buscar entidades en el módulo
            for attr_name in dir(module):
                # Filtrar atributos que parecen ser entidades
                if (attr_name.endswith("Entity") or 
                    attr_name.endswith("Trader") or 
                    "Entity" in attr_name or
                    "Trader" in attr_name):
                    try:
                        entity_class = getattr(module, attr_name)
                        
                        # Verificar si tiene función de creación
                        create_func_name = f"create_{attr_name.lower()}"
                        create_func_alt = f"create_{attr_name.replace('Entity', '').lower()}_entity"
                        
                        if hasattr(module, create_func_name):
                            # Usar función de creación si existe
                            create_func = getattr(module, create_func_name)
                            entity = create_func()
                            entities.append(entity)
                            logger.info(f"Entidad {attr_name} creada usando {create_func_name}()")
                            
                        elif hasattr(module, create_func_alt):
                            # Probar nombre alternativo
                            create_func = getattr(module, create_func_alt)
                            entity = create_func()
                            entities.append(entity)
                            logger.info(f"Entidad {attr_name} creada usando {create_func_alt}()")
                            
                        # También buscar instancias ya creadas
                        for potential_instance in dir(module):
                            if potential_instance.lower() == attr_name.lower() or (
                               attr_name.lower() in potential_instance.lower() and 
                               not potential_instance.startswith("_")):
                                instance = getattr(module, potential_instance)
                                # Verificar si es una instancia
                                if hasattr(instance, "__class__") and instance.__class__.__name__ == attr_name:
                                    entities.append(instance)
                                    logger.info(f"Instancia existente {potential_instance} encontrada")
                        
                    except Exception as e:
                        logger.warning(f"Error procesando entidad {attr_name}: {str(e)}")
            
        except ImportError as e:
            logger.warning(f"No se pudo importar módulo {module_name}: {str(e)}")
        except Exception as e:
            logger.error(f"Error procesando módulo {module_name}: {str(e)}")
    
    return entities

def initialize_all_entities():
    """
    Inicializar sistema y conectar todas las entidades.
    
    Returns:
        Dict con resultado de la inicialización
    """
    start_time = time.time()
    result = {
        "success": False,
        "entity_count": 0,
        "connected_count": 0,
        "errors": []
    }
    
    try:
        # 1. Inicializar el sistema
        logger.info("Inicializando sistema central...")
        connector = initialize_system()
        
        # 2. Localizar entidades existentes
        logger.info("Localizando entidades existentes...")
        entities = locate_and_import_entities()
        result["entity_count"] = len(entities)
        
        if not entities:
            logger.warning("No se encontraron entidades existentes")
            result["errors"].append("No se encontraron entidades existentes")
        
        # 3. Aplicar conectores a todas las entidades
        logger.info(f"Aplicando conectores a {len(entities)} entidades...")
        connected_count = 0
        
        for entity in entities:
            try:
                entity_name = getattr(entity, "name", str(entity))
                logger.info(f"Conectando entidad {entity_name}...")
                
                # Aplicar conectores
                repair_connector, message_connector = apply_connectors_to_entity(entity)
                
                # Contar éxito
                connected_count += 1
                
                # Enviar mensaje de estado inicial
                if message_connector:
                    message_connector.send_status_update()
                
            except Exception as e:
                error_msg = f"Error conectando entidad {getattr(entity, 'name', str(entity))}: {str(e)}"
                logger.error(error_msg)
                result["errors"].append(error_msg)
        
        result["connected_count"] = connected_count
        
        # 4. Enviar estado consolidado
        try:
            connector.send_consolidated_status()
        except Exception as e:
            logger.error(f"Error enviando estado consolidado: {str(e)}")
            result["errors"].append(f"Error enviando estado consolidado: {str(e)}")
        
        # 5. Enviar mensaje de inicialización exitosa
        send_system_message(
            "sistema",
            f"Sistema Genesis inicializado con éxito. {connected_count} entidades conectadas con capacidades de reparación y mensajería."
        )
        
        # Forzar envío inmediato
        connector.force_send_all_messages()
        
        # Marcar como exitoso
        result["success"] = connected_count > 0
        result["elapsed_time"] = time.time() - start_time
        
        logger.info(f"Inicialización completada en {result['elapsed_time']:.2f} segundos")
        
    except Exception as e:
        logger.error(f"Error en inicialización global: {str(e)}")
        result["errors"].append(f"Error en inicialización global: {str(e)}")
    
    return result

def apply_fixes_to_websocket_entities():
    """
    Aplicar correcciones específicas a entidades WebSocket.
    
    Returns:
        True si se aplicaron correctamente
    """
    try:
        # Importar módulo
        import websocket_entity_fix
        
        # Verificar si existen las funciones de creación
        if hasattr(websocket_entity_fix, "create_local_websocket_entity") and hasattr(websocket_entity_fix, "create_external_websocket_entity"):
            # Crear nuevas instancias con configuración correcta
            hermes = websocket_entity_fix.create_local_websocket_entity("Hermes", "otoniel", 30, False)
            apollo = websocket_entity_fix.create_external_websocket_entity("Apollo", "otoniel", 35, False)
            
            # Aplicar conectores
            for entity in [hermes, apollo]:
                repair_connector, message_connector = apply_connectors_to_entity(entity)
                
                # Añadir los métodos faltantes si no están presentes
                if not hasattr(entity, "adjust_energy"):
                    entity.adjust_energy = repair_connector.auto_repair
                
                # Iniciar ciclo de vida
                if hasattr(entity, "start_lifecycle") and not getattr(entity, "is_alive", False):
                    entity.start_lifecycle()
            
            logger.info("Correcciones aplicadas a entidades WebSocket")
            return True
        else:
            logger.warning("No se encontraron funciones de creación en websocket_entity_fix")
            return False
    
    except ImportError:
        logger.warning("No se pudo importar módulo websocket_entity_fix")
        return False
    except Exception as e:
        logger.error(f"Error aplicando correcciones a WebSocket: {str(e)}")
        return False

def run_system_maintenance_cycle():
    """
    Ejecutar un ciclo de mantenimiento completo del sistema.
    
    Returns:
        Dict con resultados del mantenimiento
    """
    result = {
        "start_time": time.time(),
        "actions_performed": []
    }
    
    # 1. Aplicar correcciones específicas
    if apply_fixes_to_websocket_entities():
        result["actions_performed"].append("Correcciones en entidades WebSocket")
    
    # 2. Obtener conector
    connector = get_genesis_connector()
    
    # 3. Verificar estado de todas las entidades
    for entity_name, entity in connector.entities.items():
        if hasattr(entity, "repair_connector"):
            # Verificar salud
            health = entity.repair_connector.check_health()
            
            # Aplicar reparación si es necesario
            if health < 0.7:
                entity.repair_connector.request_repair()
                result["actions_performed"].append(f"Reparación solicitada para {entity_name}")
            
            # Aplicar buff si está bajo de energía
            if hasattr(entity, "energy") and entity.energy < 50:
                entity.repair_connector.apply_buff("energy", 20)
                result["actions_performed"].append(f"Buff de energía aplicado a {entity_name}")
    
    # 4. Enviar informe consolidado
    connector.send_consolidated_status()
    result["actions_performed"].append("Informe de estado enviado")
    
    # 5. Forzar envío inmediato
    connector.force_send_all_messages()
    
    # Completar resultado
    result["end_time"] = time.time()
    result["elapsed_time"] = result["end_time"] - result["start_time"]
    
    return result

# Si se ejecuta como script principal
if __name__ == "__main__":
    logger.info("Iniciando Sistema Genesis con capacidades de reparación y mensajería...")
    
    # Inicializar todo el sistema
    result = initialize_all_entities()
    
    # Ejecutar ciclo de mantenimiento
    maintenance_result = run_system_maintenance_cycle()
    
    # Mostrar resultados
    if result["success"]:
        logger.info(f"Sistema inicializado exitosamente.")
        logger.info(f"Entidades encontradas: {result['entity_count']}")
        logger.info(f"Entidades conectadas: {result['connected_count']}")
        
        if maintenance_result["actions_performed"]:
            logger.info(f"Acciones de mantenimiento realizadas:")
            for action in maintenance_result["actions_performed"]:
                logger.info(f"  - {action}")
    else:
        logger.error("Error inicializando el sistema.")
        for error in result["errors"]:
            logger.error(f"  - {error}")
    
    logger.info("Proceso de inicialización completado. El sistema está funcionando en segundo plano.")
    logger.info(f"Los informes se enviarán periódicamente a {CREATOR_EMAIL}")
    
    # Mantener el script corriendo
    try:
        while True:
            time.sleep(60)
            
            # Ejecutar ciclo de mantenimiento cada hora
            if int(time.time()) % 3600 < 60:  # Aproximadamente cada hora
                run_system_maintenance_cycle()
    
    except KeyboardInterrupt:
        logger.info("Sistema detenido por el usuario.")
        print("Sistema detenido. Los servicios seguirán funcionando en segundo plano.")