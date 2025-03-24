#!/bin/bash
# Script para iniciar la Prueba de Estrés Ultra-Cuántica para PostgreSQL

# Verificar que Python y las dependencias estén instaladas
echo "Verificando dependencias..."
python -c "import psycopg2, numpy" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Faltan dependencias. Por favor, instale los paquetes necesarios:"
    echo "   pip install psycopg2-binary numpy"
    exit 1
fi

# Crear directorio de logs si no existe
mkdir -p logs

# Configurar variables de entorno para conexión a PostgreSQL
echo "Configurando variables de entorno para PostgreSQL..."
export POSTGRES_DB=${POSTGRES_DB:-"postgres"}
export POSTGRES_USER=${POSTGRES_USER:-"postgres"}
export POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-""}
export POSTGRES_HOST=${POSTGRES_HOST:-"localhost"}
export POSTGRES_PORT=${POSTGRES_PORT:-"5432"}

# Verificar conexión a PostgreSQL
echo "Verificando conexión a PostgreSQL..."
python -c "import psycopg2; conn = psycopg2.connect(dbname='$POSTGRES_DB', user='$POSTGRES_USER', password='$POSTGRES_PASSWORD', host='$POSTGRES_HOST', port='$POSTGRES_PORT'); conn.close()" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ No se pudo conectar a PostgreSQL. Verifique las credenciales."
    exit 1
fi

# Función para mostrar ayuda
function show_help {
    echo "Uso: $0 [OPCIONES]"
    echo "Inicia la Prueba de Estrés Ultra-Cuántica para PostgreSQL"
    echo ""
    echo "Opciones:"
    echo "  -d, --duration MINUTOS   Duración de la prueba en minutos (default: 60)"
    echo "  -i, --intensity NIVEL    Intensidad de la prueba (1-10, default: 5)"
    echo "  -c, --connections NUM    Número máximo de conexiones (default: 100)"
    echo "  -h, --help               Muestra esta ayuda"
    echo ""
    echo "Ejemplo:"
    echo "  $0 --duration 30 --intensity 8 --connections 50"
}

# Valores predeterminados
DURATION=60
INTENSITY=5
CONNECTIONS=100

# Procesar argumentos
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -i|--intensity)
            INTENSITY="$2"
            shift 2
            ;;
        -c|--connections)
            CONNECTIONS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Opción desconocida: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validar argumentos
if ! [[ "$DURATION" =~ ^[0-9]+$ ]]; then
    echo "❌ La duración debe ser un número entero."
    exit 1
fi

if ! [[ "$INTENSITY" =~ ^[0-9]+$ ]] || [ "$INTENSITY" -lt 1 ] || [ "$INTENSITY" -gt 10 ]; then
    echo "❌ La intensidad debe ser un número entre 1 y 10."
    exit 1
fi

if ! [[ "$CONNECTIONS" =~ ^[0-9]+$ ]] || [ "$CONNECTIONS" -lt 1 ]; then
    echo "❌ El número de conexiones debe ser un número entero positivo."
    exit 1
fi

# Ajustar la configuración según la intensidad
# - Intensidad 1-3: Baja carga
# - Intensidad 4-7: Carga media
# - Intensidad 8-10: Carga extrema
if [ "$INTENSITY" -le 3 ]; then
    LOAD_DESC="baja"
    OPERATIONS_PER_CONN=50
    FAULT_PROB=0.01
elif [ "$INTENSITY" -le 7 ]; then
    LOAD_DESC="media"
    OPERATIONS_PER_CONN=100
    FAULT_PROB=0.05
else
    LOAD_DESC="extrema"
    OPERATIONS_PER_CONN=200
    FAULT_PROB=0.10
fi

# Mostrar configuración
echo "🚀 Iniciando Prueba de Estrés Ultra-Cuántica para PostgreSQL"
echo "   Duración: $DURATION minutos"
echo "   Intensidad: $INTENSITY/10 (carga $LOAD_DESC)"
echo "   Conexiones máximas: $CONNECTIONS"
echo "   Operaciones por conexión: $OPERATIONS_PER_CONN"
echo "   Probabilidad de fallos: $FAULT_PROB"
echo ""
echo "Los resultados se guardarán en el directorio 'logs/'"
echo ""

# Modificar temporalmente el archivo quantum_stress_test.py para ajustar la configuración
TMP_FILE=$(mktemp)
sed "s/'duration_minutes': [0-9][0-9]*/'duration_minutes': $DURATION/" quantum_stress_test.py > $TMP_FILE
sed -i "s/'max_connections': [0-9][0-9]*/'max_connections': $CONNECTIONS/" $TMP_FILE
sed -i "s/'operations_per_connection': [0-9][0-9]*/'operations_per_connection': $OPERATIONS_PER_CONN/" $TMP_FILE
sed -i "s/'probability': [0-9]\.[0-9]*/'probability': $FAULT_PROB/" $TMP_FILE

# Ejecutar la prueba
echo "Comenzando prueba..."
python $TMP_FILE
RESULT=$?

# Limpiar
rm $TMP_FILE

# Mostrar resultado
if [ $RESULT -eq 0 ]; then
    echo "✅ Prueba completada con éxito."
else
    echo "❌ La prueba falló o no cumplió con los criterios de éxito."
fi

# Mostrar ubicación de los resultados
echo "Revise los resultados detallados en el archivo de logs más reciente:"
ls -t logs/quantum_stress_results_*.json | head -1