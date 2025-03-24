#!/bin/bash
# Script para ejecutar el test apocalíptico con parámetros configurables

# Colores para mensajes
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Valores por defecto
DURATION=5
INTENSITY=5
MEMORY_BOMB=false
CONNECTION_FLOOD=false
DATA_ATTACK=false

# Función de ayuda
function show_help {
    echo -e "${CYAN}Script para pruebas cuánticas de PostgreSQL${NC}"
    echo ""
    echo "Uso: $0 [opciones]"
    echo ""
    echo "Opciones:"
    echo "  --help                Muestra este mensaje de ayuda"
    echo "  --duration NUM        Duración del test en minutos (default: 5)"
    echo "  --intensity NUM       Intensidad del test del 1-10 (default: 5)"
    echo "  --memory              Incluir bombas de memoria"
    echo "  --flood               Incluir inundación de conexiones"
    echo "  --dataattack          Incluir ataques masivos de datos"
    echo "  --apocalypse          Modo apocalíptico (equivale a --memory --flood --dataattack con intensidad 10)"
    echo ""
    echo "Ejemplos:"
    echo "  $0 --duration 10 --intensity 7             # Test de 10 minutos con intensidad 7"
    echo "  $0 --intensity 3 --memory                  # Test de intensidad 3 con bombas de memoria"
    echo "  $0 --apocalypse                            # Test apocalíptico total"
    echo ""
}

# Procesar argumentos
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            show_help
            exit 0
            ;;
        --duration)
            DURATION="$2"
            shift
            shift
            ;;
        --intensity)
            INTENSITY="$2"
            shift
            shift
            ;;
        --memory)
            MEMORY_BOMB=true
            shift
            ;;
        --flood)
            CONNECTION_FLOOD=true
            shift
            ;;
        --dataattack)
            DATA_ATTACK=true
            shift
            ;;
        --apocalypse)
            MEMORY_BOMB=true
            CONNECTION_FLOOD=true
            DATA_ATTACK=true
            INTENSITY=10
            shift
            ;;
        *)
            echo "Opción desconocida: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validar intensidad
if [[ $INTENSITY -lt 1 || $INTENSITY -gt 10 ]]; then
    echo -e "${RED}Error: La intensidad debe estar entre 1 y 10${NC}"
    exit 1
fi

# Mostrar configuración
echo -e "${PURPLE}=== CONFIGURACIÓN DEL TEST CUÁNTICO ===${NC}"
echo -e "${BLUE}Duración:${NC} $DURATION minutos"
echo -e "${BLUE}Intensidad:${NC} $INTENSITY/10"
echo -e "${BLUE}Bombas de memoria:${NC} $([ "$MEMORY_BOMB" = true ] && echo "Activadas" || echo "Desactivadas")"
echo -e "${BLUE}Inundación de conexiones:${NC} $([ "$CONNECTION_FLOOD" = true ] && echo "Activada" || echo "Desactivada")"
echo -e "${BLUE}Ataques de datos:${NC} $([ "$DATA_ATTACK" = true ] && echo "Activados" || echo "Desactivados")"
echo ""

# Mostrar advertencia según nivel de intensidad
if [[ $INTENSITY -ge 8 ]]; then
    echo -e "${RED}⚠️ ADVERTENCIA: Has seleccionado una intensidad muy alta (${INTENSITY}/10)${NC}"
    echo -e "${RED}   Este nivel puede causar inestabilidad extrema en el sistema${NC}"
    echo ""
    read -p "¿Estás seguro de continuar? (s/n): " confirm
    if [[ $confirm != "s" && $confirm != "S" ]]; then
        echo -e "${GREEN}Test cancelado por el usuario.${NC}"
        exit 0
    fi
fi

# Preparar ambiente
mkdir -p logs

# Configurar variables de entorno para la prueba
export TEST_DURATION_MINUTES=$DURATION
export TEST_INTENSITY=$INTENSITY
export TEST_MEMORY_BOMB=$MEMORY_BOMB
export TEST_CONNECTION_FLOOD=$CONNECTION_FLOOD
export TEST_DATA_ATTACK=$DATA_ATTACK

# Mostrar mensaje de inicio
echo -e "${YELLOW}Iniciando test cuántico en 3 segundos...${NC}"
sleep 3

# Si es intensidad 10 y todas las opciones activadas, usar directamente armageddon.sh
if [[ $INTENSITY -eq 10 && "$MEMORY_BOMB" = true && "$CONNECTION_FLOOD" = true && "$DATA_ATTACK" = true ]]; then
    echo -e "${RED}¡INICIANDO MODO ARMAGEDDON!${NC}"
    cd "$(dirname "$0")"  # Cambiar al directorio del script
    ./armageddon.sh $DURATION
else
    # Ejecutar prueba normal
    echo -e "${YELLOW}Iniciando prueba cuántica...${NC}"
    python apocalyptic_test.py $DURATION
fi

# Mostrar resultados
echo -e "${GREEN}Prueba completada.${NC}"
echo -e "Resultados disponibles en: ${BLUE}logs/apocalipsis_resultados_*.json${NC}"