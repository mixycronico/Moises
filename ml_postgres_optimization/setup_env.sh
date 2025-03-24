#!/bin/bash
# Script para configurar el entorno para el Sistema de Optimización ML de PostgreSQL

# Crear directorio de logs si no existe
mkdir -p logs

# Configurar variables de entorno para conexión a PostgreSQL
export POSTGRES_DB=${POSTGRES_DB:-"postgres"}
export POSTGRES_USER=${POSTGRES_USER:-"postgres"}
export POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-""}
export POSTGRES_HOST=${POSTGRES_HOST:-"localhost"}
export POSTGRES_PORT=${POSTGRES_PORT:-"5432"}

# Verificar si PostgreSQL está disponible
echo "Verificando conexión a PostgreSQL..."
psql "postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@$POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB" -c "SELECT version();" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ PostgreSQL está disponible y accesible."
else
    echo "❌ No se pudo conectar a PostgreSQL. Verifique las credenciales y la disponibilidad del servidor."
    echo "   Puede configurar las variables de entorno para la conexión:"
    echo "   export POSTGRES_DB=nombre_bd"
    echo "   export POSTGRES_USER=usuario"
    echo "   export POSTGRES_PASSWORD=contraseña"
    echo "   export POSTGRES_HOST=host"
    echo "   export POSTGRES_PORT=puerto"
    exit 1
fi

# Verificar dependencias de Python
echo "Verificando dependencias de Python..."
python -c "import sklearn, pandas, numpy, psycopg2" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Todas las dependencias de Python están instaladas."
else
    echo "❌ Faltan algunas dependencias de Python."
    echo "   Por favor, instale las dependencias con:"
    echo "   pip install scikit-learn pandas psycopg2-binary numpy"
    echo "   o use el packager_tool en el entorno Replit."
    exit 1
fi

# Configurar base de datos (opcional)
if [ "$1" == "--setup-db" ]; then
    echo "Configurando base de datos PostgreSQL..."
    psql "postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@$POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB" -f setup_db.sql
    if [ $? -eq 0 ]; then
        echo "✅ Base de datos configurada correctamente."
    else
        echo "❌ Error al configurar la base de datos."
        exit 1
    fi
fi

# Mensaje de finalización
echo "✅ Entorno configurado correctamente para el Sistema de Optimización ML de PostgreSQL."
echo "   Puede ejecutar el sistema con: python run.py"