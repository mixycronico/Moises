"""
Demo interactiva del Buddha Trader: Estrategia de trading potenciada por Buddha AI.

Este script ejecuta una demo interactiva de la estrategia Buddha Trader,
permitiendo al usuario probar el sistema con parámetros personalizados.

La demo simula el ciclo completo de trading, mostrando:
- Análisis técnico tradicional
- Mejoras de Buddha AI
- Gestión dinámica de riesgo
- Reserva adaptativa
- Seguimiento de métricas

Uso:
    python run_buddha_trader.py
"""

import asyncio
import logging
import json
import os
from typing import Dict, Any

# Importar Buddha Trader
from genesis.strategies.buddha_trader import BuddhaTrader

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("run_buddha_trader")

async def run_interactive_demo():
    """Ejecutar demo interactiva del Buddha Trader."""
    print("\n" + "=" * 60)
    print("SISTEMA GENESIS: BUDDHA TRADER DEMO".center(60))
    print("Estrategia de Trading Trascendental".center(60))
    print("=" * 60)
    
    print("\nEsta demo permite simular tu estrategia de trading con ciclos de $150,")
    print("potenciada por Buddha AI para maximizar la precisión y reducir el riesgo.")
    
    # Solicitar parámetros iniciales
    try:
        print("\n--- CONFIGURACIÓN INICIAL ---")
        capital = float(input("Capital inicial (predeterminado: 150): ") or "150")
        emergency_fund = float(input("Fondo de emergencia (predeterminado: 20): ") or "20")
        next_cycle = float(input("Reserva próximo ciclo (predeterminado: 20): ") or "20")
        personal_use = float(input("Uso personal (predeterminado: 10): ") or "10")
        
        print("\n--- PARÁMETROS DE SIMULACIÓN ---")
        days = int(input("Días de simulación (predeterminado: 5): ") or "5")
        trades_per_day = int(input("Trades objetivo por día (predeterminado: 4): ") or "4")
        
        # Confirmar
        print("\n--- RESUMEN DE CONFIGURACIÓN ---")
        print(f"Capital inicial: ${capital:.2f}")
        print(f"Fondo emergencia: ${emergency_fund:.2f}")
        print(f"Próximo ciclo: ${next_cycle:.2f}")
        print(f"Uso personal: ${personal_use:.2f}")
        print(f"Días simulación: {days}")
        print(f"Trades por día: {trades_per_day}")
        
        confirm = input("\n¿Iniciar simulación con estos parámetros? (s/n): ").lower()
        if confirm != "s":
            print("Simulación cancelada.")
            return
        
        # Inicializar trader
        print("\nInicializando Buddha Trader...")
        trader = BuddhaTrader(
            capital=capital,
            emergency_fund=emergency_fund,
            next_cycle=next_cycle,
            personal_use=personal_use
        )
        
        # Verificar estado de Buddha
        print(f"Buddha AI {'habilitado' if trader.buddha_enabled else 'deshabilitado'}")
        if not trader.buddha_enabled:
            print("NOTA: Sin Buddha AI habilitado, se usarán solo indicadores técnicos tradicionales.")
        
        # Ejecutar simulación
        await trader.run_cycle_simulation(days=days, trades_per_day=trades_per_day)
        
        # Guardar estado
        save = input("\n¿Guardar estado actual? (s/n): ").lower()
        if save == "s":
            filename = input("Nombre del archivo (predeterminado: buddha_trader_state.json): ") or "buddha_trader_state.json"
            trader.save_state(filename)
            print(f"Estado guardado en {filename}")
        
        # Opciones para analizar resultados
        print("\n--- ANÁLISIS DE RESULTADOS ---")
        metrics = trader.get_metrics()
        print(f"Operaciones totales: {metrics['trades_total']}")
        print(f"Tasa de éxito: {metrics['success_rate']:.2f}%")
        print(f"Beneficio total: ${metrics['profit_total']:.2f}")
        print(f"Beneficio promedio por operación: ${metrics['avg_profit_per_trade']:.2f}")
        print(f"Insights de Buddha utilizados: {metrics['buddha_insights_used']}")
        
        print("\n¡Demo finalizada!")
        
    except KeyboardInterrupt:
        print("\nDemo interrumpida por el usuario.")
    except Exception as e:
        logger.error(f"Error durante la demo: {str(e)}")
        print(f"\nError durante la demo: {str(e)}")

async def main():
    """Función principal."""
    await run_interactive_demo()

if __name__ == "__main__":
    asyncio.run(main())