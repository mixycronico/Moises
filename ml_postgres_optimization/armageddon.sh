#!/bin/bash
# Script para desatar el ARMAGEDDON sobre PostgreSQL
# ABSOLUTAMENTE SIN PIEDAD

# Colores para mensajes dramáticos
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Función para mostrar advertencia
function show_warning {
    echo -e "${RED}"
    echo "█████████████████████████████████████████████████████████████████████"
    echo "█                                                                   █"
    echo "█                         ¡¡¡ ADVERTENCIA !!!                      █"
    echo "█                                                                   █"
    echo "█  ESTE SCRIPT EJECUTARÁ UN TEST DEVASTADOR SOBRE POSTGRESQL       █"
    echo "█  QUE PUEDE:                                                      █"
    echo "█   - LLEVAR CPU AL 100% EN TODOS LOS NÚCLEOS                      █"
    echo "█   - CONSUMIR TODA LA MEMORIA DISPONIBLE                          █"
    echo "█   - SATURAR EL SISTEMA DE ARCHIVOS                               █"
    echo "█   - PROVOCAR INESTABILIDAD GENERAL DEL SISTEMA                   █"
    echo "█                                                                   █"
    echo "█  SE RECOMIENDA EJECUTAR SOLO EN ENTORNOS DE PRUEBA AISLADOS      █"
    echo "█                                                                   █"
    echo "█████████████████████████████████████████████████████████████████████"
    echo -e "${NC}"
}

# Función para mostrar cuenta regresiva
function countdown {
    echo -e "${YELLOW}Tienes ${1} segundos para cancelar (CTRL+C)${NC}"
    for (( i=${1}; i>=1; i-- )); do
        echo -ne "${YELLOW}$i...${NC} "
        sleep 1
    done
    echo -e "\n${RED}¡COMENZANDO ARMAGEDDON!${NC}\n"
}

# Crear directorio de logs si no existe
mkdir -p logs

# Verificar que Python y las dependencias estén instaladas
echo "Verificando dependencias..."
python -c "import psycopg2" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Falta psycopg2. Instálalo con: pip install psycopg2-binary${NC}"
    exit 1
fi

# Mostrar advertencia y prepararse para el apocalipsis
show_warning
echo -e "${RED}Este test está diseñado para llevar PostgreSQL a sus límites ABSOLUTOS.${NC}"
echo -e "${RED}Se ejecutará múltiples pruebas devastadoras con configuraciones extremas.${NC}"
echo ""

# Verificar si el usuario desea continuar
# Para automatización, aceptar confirmación desde stdin o definir FORCE_ARMAGEDDON=1
if [ -n "$FORCE_ARMAGEDDON" ] && [ "$FORCE_ARMAGEDDON" -eq 1 ]; then
    echo -e "${RED}ARMAGEDDON forzado por variable de entorno${NC}"
    confirmation="ARMAGEDDON"
else
    read -p "¿Estás ABSOLUTAMENTE SEGURO de que quieres continuar? (escribe 'ARMAGEDDON' para confirmar): " confirmation
fi

if [ "$confirmation" != "ARMAGEDDON" ]; then
    echo -e "${GREEN}Test apocalíptico cancelado. Sistema a salvo.${NC}"
    exit 0
fi

# Iniciar cuenta regresiva
countdown 3

# Configurar variables de entorno para PostgreSQL
export POSTGRES_DB=${POSTGRES_DB:-"postgres"}
export POSTGRES_USER=${POSTGRES_USER:-"postgres"}
export POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-""}
export POSTGRES_HOST=${POSTGRES_HOST:-"localhost"}
export POSTGRES_PORT=${POSTGRES_PORT:-"5432"}

# Verificar conexión a PostgreSQL
echo "Verificando conexión a PostgreSQL..."
python -c "import psycopg2; conn = psycopg2.connect(dbname='$POSTGRES_DB', user='$POSTGRES_USER', password='$POSTGRES_PASSWORD', host='$POSTGRES_HOST', port='$POSTGRES_PORT'); conn.close()" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ No se pudo conectar a PostgreSQL. Verifique las credenciales.${NC}"
    exit 1
fi

# Parámetros de la prueba
DURATION_MINUTES=${1:-20}  # Permite pasar duración como argumento, por defecto 20 minutos

# Mensaje final antes del armageddon
echo -e "${RED}╔═════════════════════════════════════════════════╗${NC}"
echo -e "${RED}║             INICIANDO ARMAGEDDON                ║${NC}"
echo -e "${RED}║        POSTGRESQL ESTÁ EN GRAVE PELIGRO         ║${NC}"
echo -e "${RED}╚═════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Duración aproximada: ${YELLOW}${DURATION_MINUTES} minutos${NC}"
echo -e "Hora de inicio: ${YELLOW}$(date)${NC}"
echo ""
echo -e "${RED}Para detener, presiona CTRL+C (si es que puedes...)${NC}"
echo ""

# Ejecutar el test apocalíptico
python apocalyptic_test.py ${DURATION_MINUTES}

# Verificar resultado
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}¡INCREÍBLE! PostgreSQL ha sobrevivido al ARMAGEDDON.${NC}"
    echo -e "${GREEN}Esto prueba que el sistema de optimización ML es EXTREMADAMENTE eficaz.${NC}"
else
    echo -e "\n${YELLOW}PostgreSQL no ha superado todas las pruebas apocalípticas.${NC}"
    echo -e "${YELLOW}Revise los logs para analizar los fallos y mejorar la resiliencia.${NC}"
fi

# Mostrar ubicación de logs
echo ""
echo "Los resultados detallados se encuentran en:"
ls -t logs/apocalipsis_resultados_*.json | head -1