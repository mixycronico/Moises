#!/bin/bash

echo "🚀 Iniciando compilación optimizada de Vite..."

# Asegurarse de que node_modules está actualizado
echo "📦 Verificando dependencias..."
if [ ! -d "client/node_modules" ]; then
  echo "⚙️ Instalando dependencias del cliente..."
  cd client && npm install --no-audit --no-fund --loglevel=error
  cd ..
fi

# Establecer variables de entorno para optimización
export NODE_ENV=production
# Limitar el uso de memoria de Node
export NODE_OPTIONS="--max-old-space-size=2048"

echo "🛠️ Compilando aplicación React con Vite..."
cd client && npm run build
BUILD_SUCCESS=$?
cd ..

if [ $BUILD_SUCCESS -eq 0 ]; then
  echo "✅ Compilación Vite completada con éxito!"
  
  # Verificar si existen los archivos compilados
  if [ -d "static" ] && [ -f "static/index.html" ]; then
    echo "📦 Verificando archivos generados..."
    find static -type f | grep -v "node_modules" | sort
    echo "📏 Tamaño total de archivos:"
    du -sh static
  else
    echo "⚠️ Los archivos compilados no se encuentran en la ubicación esperada."
    echo "🔄 Ejecutando compilación rápida como respaldo..."
    ./quick_build.sh
  fi
else
  echo "❌ Error en la compilación de Vite."
  echo "🔄 Ejecutando compilación rápida como respaldo..."
  ./quick_build.sh
fi

# Mensaje final
echo "🌟 Proceso completado. El servidor puede ser iniciado ahora."