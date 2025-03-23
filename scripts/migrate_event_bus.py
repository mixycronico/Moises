#!/usr/bin/env python
"""
Script para migrar de EventBus tradicional al nuevo TranscendentalEventBus.

Este script identifica y modifica los archivos que utilizan el EventBus tradicional,
reemplazándolo por el nuevo adaptador TranscendentalEventBus con capacidades WebSocket/API.

Uso:
    python migrate_event_bus.py [--dry-run]
    
Argumentos:
    --dry-run: Solo mostrar archivos y cambios, sin realizar modificaciones
"""

import os
import re
import sys
import logging
from typing import List, Tuple, Dict, Set

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("EventBusMigration")

# Patrones para buscar importaciones y usos de EventBus
EVENT_BUS_IMPORT_PATTERNS = [
    r'from\s+genesis\.events\s+import\s+EventBus',
    r'from\s+genesis\.core\.events\s+import\s+EventBus',
    r'import\s+EventBus\s+from\s+[\'"]genesis\.events[\'"]'
]

EVENT_BUS_USAGE_PATTERNS = [
    r'EventBus\(\)',
    r'EventBus\.get_instance\(\)',
    r'event_bus\s*=\s*EventBus\(\)',
    r'self\.event_bus\s*=\s*EventBus\(\)'
]

# Reemplazos para importaciones y usos
IMPORT_REPLACEMENT = 'from genesis.core.transcendental_event_bus import TranscendentalEventBus'
USAGE_REPLACEMENT = 'TranscendentalEventBus()'

def find_python_files(directory: str) -> List[str]:
    """
    Encuentra todos los archivos Python en el directorio y subdirectorios.
    
    Args:
        directory: Directorio raíz para la búsqueda
        
    Returns:
        Lista de rutas de archivos Python
    """
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def analyze_file(file_path: str) -> Tuple[bool, Dict[str, List[Tuple[int, str, str]]]]:
    """
    Analiza un archivo para detectar uso de EventBus.
    
    Args:
        file_path: Ruta del archivo a analizar
        
    Returns:
        Tupla (contiene_event_bus, {tipo_reemplazo: [(línea, texto_original, texto_reemplazo)]})
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    import_replacements = []
    usage_replacements = []
    
    # Buscar importaciones y usos
    for i, line in enumerate(lines):
        # Verificar patrones de importación
        for pattern in EVENT_BUS_IMPORT_PATTERNS:
            if re.search(pattern, line):
                import_replacements.append((i, line.strip(), IMPORT_REPLACEMENT))
                break
        
        # Verificar patrones de uso
        for pattern in EVENT_BUS_USAGE_PATTERNS:
            if re.search(pattern, line):
                # Mantener la indentación original
                indentation = len(line) - len(line.lstrip())
                original = line.strip()
                replacement = ' ' * indentation + line.strip().replace(
                    re.search(pattern, line).group(0), 
                    USAGE_REPLACEMENT
                )
                usage_replacements.append((i, original, replacement))
                break
    
    contains_event_bus = bool(import_replacements or usage_replacements)
    replacements = {
        'imports': import_replacements,
        'usages': usage_replacements
    }
    
    return contains_event_bus, replacements

def modify_file(file_path: str, replacements: Dict[str, List[Tuple[int, str, str]]]) -> bool:
    """
    Modifica un archivo aplicando los reemplazos.
    
    Args:
        file_path: Ruta del archivo a modificar
        replacements: Diccionario con reemplazos a aplicar
        
    Returns:
        True si se realizaron cambios, False en caso contrario
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = content
        
        # Aplicar reemplazos de importaciones
        for _, original, replacement in replacements['imports']:
            new_content = new_content.replace(original, replacement)
        
        # Aplicar reemplazos de usos
        for _, original, replacement in replacements['usages']:
            new_content = new_content.replace(original, replacement)
        
        # Escribir solo si hubo cambios
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        
        return False
    
    except Exception as e:
        logger.error(f"Error al modificar {file_path}: {str(e)}")
        return False

def migrate_event_bus(directory: str, dry_run: bool = False) -> None:
    """
    Migra todos los archivos Python en el directorio de EventBus a TranscendentalEventBus.
    
    Args:
        directory: Directorio raíz para la migración
        dry_run: Si es True, solo muestra los cambios sin aplicarlos
    """
    logger.info(f"Comenzando migración de EventBus {'(dry run)' if dry_run else ''}")
    
    python_files = find_python_files(directory)
    logger.info(f"Encontrados {len(python_files)} archivos Python para analizar")
    
    files_with_event_bus = []
    
    for file_path in python_files:
        contains_event_bus, replacements = analyze_file(file_path)
        
        if contains_event_bus:
            files_with_event_bus.append(file_path)
            
            logger.info(f"Archivo con EventBus: {file_path}")
            
            # Mostrar reemplazos de importaciones
            for line, original, replacement in replacements['imports']:
                logger.info(f"  Línea {line+1}: {original} -> {replacement}")
            
            # Mostrar reemplazos de usos
            for line, original, replacement in replacements['usages']:
                logger.info(f"  Línea {line+1}: {original} -> {replacement}")
            
            # Aplicar cambios si no es dry run
            if not dry_run:
                modified = modify_file(file_path, replacements)
                if modified:
                    logger.info(f"  Archivo modificado: {file_path}")
    
    logger.info(f"Total de archivos con EventBus: {len(files_with_event_bus)}")
    
    if files_with_event_bus:
        logger.info("Archivos que requieren migración:")
        for file_path in files_with_event_bus:
            logger.info(f"  {file_path}")

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    migrate_event_bus(".", dry_run)