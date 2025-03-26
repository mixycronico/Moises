#!/usr/bin/env python3
"""
Script para construir la aplicación React del Sistema Genesis.

Este script facilita la construcción de la aplicación React para
que pueda ser servida por Flask.
"""

import os
import subprocess
import shutil
import json
import sys
from pathlib import Path

def check_prerequisites():
    """Verificar que tenemos los prerrequisitos necesarios."""
    print("Verificando prerrequisitos...")
    
    # Verificar que existe la carpeta frontend
    if not os.path.exists('website/frontend'):
        print("Error: No se encontró el directorio website/frontend")
        return False
    
    # Verificar que existe package.json
    if not os.path.exists('website/frontend/package.json'):
        print("Error: No se encontró el archivo package.json en website/frontend")
        return False
    
    # Verificar que existe webpack.config.js
    if not os.path.exists('website/frontend/webpack.config.js'):
        print("Error: No se encontró el archivo webpack.config.js en website/frontend")
        return False
    
    return True

def create_basic_files():
    """Crear archivos básicos necesarios que podrían faltar."""
    print("Creando archivos básicos...")
    
    # Asegurarse de que existe la carpeta dist
    os.makedirs('website/frontend/dist', exist_ok=True)
    
    # Crear un index.html básico si no existe
    if not os.path.exists('website/frontend/dist/index.html'):
        with open('website/frontend/dist/index.html', 'w') as f:
            f.write('''<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="Sistema Genesis - Plataforma avanzada de trading e inversión">
  <title>Sistema Genesis</title>
</head>
<body>
  <div id="root"></div>
  <script src="bundle.js"></script>
</body>
</html>''')
    
    # Crear un bundle.js vacío si no existe
    if not os.path.exists('website/frontend/dist/bundle.js'):
        with open('website/frontend/dist/bundle.js', 'w') as f:
            f.write('// Placeholder para bundle.js real\nconsole.log("Placeholder bundle - necesita ser construido");')

def copy_public_files():
    """Copiar archivos estáticos de public a dist."""
    print("Copiando archivos estáticos...")
    
    # Verificar si existe el directorio public
    if os.path.exists('website/frontend/public'):
        # Copiar todos los archivos excepto index.html (que usará webpack)
        for item in os.listdir('website/frontend/public'):
            if item != 'index.html':
                src = os.path.join('website/frontend/public', item)
                dst = os.path.join('website/frontend/dist', item)
                
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                elif os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)

def main():
    """Función principal."""
    print("=== Proceso de construcción para la aplicación React del Sistema Genesis ===")
    
    # Verificar prerrequisitos
    if not check_prerequisites():
        print("No se cumplen los prerrequisitos para la construcción. Abortando.")
        return False
    
    # Crear archivos básicos
    create_basic_files()
    
    # Copiar archivos estáticos
    copy_public_files()
    
    print("\nAplicación React preparada para ser servida por Flask.")
    print("\nPara una construcción completa con webpack, ejecutar:")
    print("cd website/frontend && npm run build")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)