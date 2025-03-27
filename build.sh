#!/bin/bash

echo "Construyendo la aplicación React de forma optimizada..."

# Entrar al directorio del cliente
cd client

# Limpiar caché de npm para resolver problemas potenciales
echo "Limpiando caché..."
npm cache clean --force

# Instalar solo dependencias de producción con opción --no-optional 
echo "Instalando dependencias mínimas..."
npm install --production --no-optional --prefer-offline

# Construir la aplicación con optimizaciones
echo "Compilando el código..."
NODE_OPTIONS="--max-old-space-size=512" npm run build

echo "Construcción finalizada."

# Volver al directorio raíz
cd ..

# Asegurarse de que la carpeta static existe y tiene permisos correctos
mkdir -p static
chmod -R 755 static

echo "El frontend ha sido compilado en la carpeta static."