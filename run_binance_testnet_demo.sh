#!/bin/bash

# Script para ejecutar la demostración de Binance Testnet con el adaptador ultra-cuántico
# Este script ejecuta la demo que demuestra la capacidad del sistema para trabajar
# con Binance Testnet, ya sea usando API keys reales o mediante transmutación cuántica.

echo "=========================================================="
echo "    DEMO BINANCE TESTNET - SISTEMA GENESIS ULTRA-CUÁNTICO"
echo "=========================================================="
echo ""

# Verificar si las credenciales API están disponibles en el entorno
if [ -n "$BINANCE_TESTNET_API_KEY" ] && [ -n "$BINANCE_TESTNET_API_SECRET" ]; then
  echo "🔑 Credenciales API detectadas"
  echo "Se está utilizando modo de conexión real + transmutación cuántica de respaldo"
else
  echo "⚠️ No se detectaron credenciales API completas"
  echo "Se utilizará modo de transmutación cuántica para simular la conexión"
fi

echo ""
echo "Iniciando demo..."
echo "------------------------------------------------------------------------"

# Ejecutar la demo
python test_binance_testnet_quantum.py

# Verificar si la ejecución fue exitosa
if [ $? -eq 0 ]; then
  echo "------------------------------------------------------------------------"
  echo "✅ Demo completada exitosamente"
  echo "El Sistema Cuántico Ultra-Divino funciona perfectamente"
else
  echo "------------------------------------------------------------------------"
  echo "❌ Error al ejecutar la demo"
  echo "Por favor, verifica los logs para más detalles"
fi