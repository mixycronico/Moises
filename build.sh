#!/bin/bash

echo "Construyendo la aplicación React..."

# Entrar al directorio del cliente
cd client

# Instalar dependencias si es necesario
if [ ! -d "node_modules" ]; then
  echo "Instalando dependencias..."
  npm install
fi

# Construir la aplicación
echo "Compilando el código..."
npm run build

echo "Construcción finalizada."

# Volver al directorio raíz
cd ..

echo "El frontend ha sido compilado en la carpeta static."