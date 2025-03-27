#!/bin/bash
# ARMAGEDÓN CÓSMICO - Script para ejecutar la prueba extrema del sistema de trading

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
COSMIC='\033[38;5;208m'
DIVINE='\033[38;5;141m'
TRANSCEND='\033[38;5;51m'
NC='\033[0m' # No Color

# Mostrar banner
echo -e "${DIVINE}╔══════════════════════════════════════════════════════════════════╗"
echo -e "║                   ${COSMIC}ARMAGEDÓN CÓSMICO${DIVINE}                           ║"
echo -e "║       ${TRANSCEND}La prueba definitiva para el Sistema Genesis de Trading${DIVINE}   ║"
echo -e "╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Este script facilita la ejecución de la prueba ARMAGEDÓN CÓSMICO.${NC}"
echo ""

# Función para mostrar ayuda
show_help() {
    echo -e "${WHITE}Uso:${NC}"
    echo -e "  ./armageddon_cosmic.sh [opciones]"
    echo ""
    echo -e "${WHITE}Opciones:${NC}"
    echo -e "  -m, --mode      : Modo de prueba (basic, extended). Default: extended"
    echo -e "  -d, --duration  : Duración máxima en minutos. Default: 5"
    echo -e "  -c, --cycles    : Número de ciclos de operaciones. Default: 3"
    echo -e "  -e, --events    : Número de eventos catastróficos. Default: 1"
    echo -e "  -f, --father    : Nombre del creador/dueño. Default: otoniel"
    echo -e "  -h, --help      : Mostrar esta ayuda"
    echo ""
    echo -e "${WHITE}Ejemplos:${NC}"
    echo -e "  ./armageddon_cosmic.sh"
    echo -e "  ./armageddon_cosmic.sh -m extended -d 10 -c 5 -e 2"
    echo -e "  ./armageddon_cosmic.sh --mode basic --duration 3"
    echo ""
}

# Valores por defecto
MODE="extended"
DURATION=5
CYCLES=3
EVENTS=1
FATHER="otoniel"

# Procesar parámetros
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -c|--cycles)
            CYCLES="$2"
            shift 2
            ;;
        -e|--events)
            EVENTS="$2"
            shift 2
            ;;
        -f|--father)
            FATHER="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Opción desconocida: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Validar modo
if [[ "$MODE" != "basic" && "$MODE" != "extended" ]]; then
    echo -e "${RED}Modo inválido: $MODE. Debe ser 'basic' o 'extended'.${NC}"
    exit 1
fi

# Validar números
if ! [[ "$DURATION" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Duración inválida: $DURATION. Debe ser un número entero.${NC}"
    exit 1
fi

if ! [[ "$CYCLES" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Número de ciclos inválido: $CYCLES. Debe ser un número entero.${NC}"
    exit 1
fi

if ! [[ "$EVENTS" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Número de eventos inválido: $EVENTS. Debe ser un número entero.${NC}"
    exit 1
fi

# Mostrar configuración
echo -e "${CYAN}Configuración seleccionada:${NC}"
echo -e "  - Modo: ${GREEN}$MODE${NC}"
echo -e "  - Duración máxima: ${GREEN}$DURATION minutos${NC}"
echo -e "  - Ciclos: ${GREEN}$CYCLES${NC}"
echo -e "  - Eventos catastróficos: ${GREEN}$EVENTS${NC}"
echo -e "  - Creador: ${GREEN}$FATHER${NC}"
echo ""

# Solicitar confirmación
read -p "$(echo -e ${YELLOW}"¿Iniciar la prueba ARMAGEDÓN CÓSMICO con esta configuración? (s/n): "${NC})" CONFIRM

if [[ "$CONFIRM" != "s" && "$CONFIRM" != "S" && "$CONFIRM" != "si" && "$CONFIRM" != "SI" && "$CONFIRM" != "y" && "$CONFIRM" != "Y" && "$CONFIRM" != "yes" && "$CONFIRM" != "YES" ]]; then
    echo -e "${RED}Prueba cancelada por el usuario.${NC}"
    exit 0
fi

echo -e "${COSMIC}Iniciando prueba ARMAGEDÓN CÓSMICO...${NC}"
echo ""

# Ejecutar prueba
python test_armageddon_cosmic_traders.py --mode "$MODE" --duration "$DURATION" --cycles "$CYCLES" --events "$EVENTS" --father "$FATHER"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo -e "${GREEN}Prueba ARMAGEDÓN CÓSMICO completada.${NC}"
    echo -e "${YELLOW}El informe detallado se ha guardado en: informe_armageddon_trading_cosmico.md${NC}"
else
    echo ""
    echo -e "${RED}La prueba ARMAGEDÓN CÓSMICO ha fallado con código de error: $exit_code${NC}"
    echo -e "${YELLOW}Revise el archivo de log: armageddon_cosmic_report.log${NC}"
fi

echo ""
echo -e "${DIVINE}╔══════════════════════════════════════════════════════════════════╗"
echo -e "║             ${COSMIC}ARMAGEDÓN CÓSMICO COMPLETADO${DIVINE}                       ║"
echo -e "╚══════════════════════════════════════════════════════════════════╝${NC}"