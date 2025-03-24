"""
Script de prueba para el Motor de Comportamiento Humano Gabriel

Este script verifica el funcionamiento del nuevo motor de comportamiento Gabriel,
con énfasis especial en el estado FEARFUL (100% implementación).

Autor: Genesis AI Assistant
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from genesis.trading.gabriel_adapter import get_gabriel_adapter
from genesis.trading.human_behavior_engine import EmotionalState

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_gabriel")

async def test_normal_behavior():
    """Probar comportamiento normal (no FEARFUL)."""
    adapter = get_gabriel_adapter()
    
    logger.info("=== PRUEBA DE COMPORTAMIENTO NORMAL ===")
    
    # Obtener características actuales
    characteristics = adapter.get_current_characteristics()
    logger.info(f"Características iniciales: {characteristics}")
    
    # Aleatorizar características
    new_characteristics = await adapter.randomize_human_characteristics()
    logger.info(f"Características aleatorizadas: {new_characteristics}")
    
    # Probar decisión de entrada
    market_context = {
        "volatility": 0.3,
        "trend": 0.7,  # Tendencia alcista
        "volume_change": 0.2,
        "price_change": 0.05,
        "sentiment": "bullish",
        "risk_level": 0.4
    }
    
    decision, reason = await adapter.should_enter_trade(0.75, market_context)
    logger.info(f"Decisión de entrada (normal): {decision}, razón: {reason}")
    
    # Probar decisión de salida
    exit_decision, exit_reason = await adapter.should_exit_trade(12.0, 4.0, 0.02)
    logger.info(f"Decisión de salida (normal): {exit_decision}, razón: {exit_reason}")
    
    # Probar ajuste de tamaño
    original_size = 100.0
    adjusted_size = await adapter.adjust_order_size(original_size, 0.8, True)
    logger.info(f"Ajuste de tamaño (normal): {original_size} -> {adjusted_size}")
    
    # Probar validación de operación
    trade_params = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "price": 50000.0,
        "confidence": 0.8
    }
    
    valid, reject_reason = await adapter.validate_trade(trade_params)
    logger.info(f"Validación (normal): {'Válida' if valid else 'Rechazada'}, " +
               f"razón: {reject_reason if not valid else 'N/A'}")
    
    return {
        "entry_decision": decision,
        "exit_decision": exit_decision,
        "size_adjustment": adjusted_size / original_size,
        "validation": valid
    }

async def test_fearful_behavior():
    """Probar comportamiento en estado FEARFUL (100% implementación)."""
    adapter = get_gabriel_adapter()
    
    logger.info("\n=== PRUEBA DE COMPORTAMIENTO FEARFUL (100%) ===")
    
    # Establecer estado FEARFUL
    adapter.set_fearful_state("prueba_comportamiento_fearful")
    
    # Verificar estado
    characteristics = adapter.get_current_characteristics()
    logger.info(f"Características en estado FEARFUL: {characteristics}")
    
    if characteristics["emotional_state"] != EmotionalState.FEARFUL.name:
        logger.error(f"ERROR: Estado emocional no es FEARFUL, es {characteristics['emotional_state']}")
    
    # Probar decisión de entrada
    market_context = {
        "volatility": 0.3,  # Volatilidad baja (objetivamente)
        "trend": 0.7,        # Tendencia alcista (objetivamente)
        "volume_change": 0.2,
        "price_change": 0.05,
        "sentiment": "bullish",
        "risk_level": 0.4
    }
    
    decision, reason = await adapter.should_enter_trade(0.75, market_context)
    logger.info(f"Decisión de entrada (FEARFUL): {decision}, razón: {reason}")
    
    # Probar decisión de salida con ganancia pequeña
    exit_decision1, exit_reason1 = await adapter.should_exit_trade(2.0, 0.5, 0.01)
    logger.info(f"Decisión de salida con ganancia pequeña (FEARFUL): {exit_decision1}, razón: {exit_reason1}")
    
    # Probar decisión de salida con pérdida pequeña
    exit_decision2, exit_reason2 = await adapter.should_exit_trade(-1.0, 0.5, -0.01)
    logger.info(f"Decisión de salida con pérdida pequeña (FEARFUL): {exit_decision2}, razón: {exit_reason2}")
    
    # Probar ajuste de tamaño (compra)
    original_size = 100.0
    buy_size = await adapter.adjust_order_size(original_size, 0.8, True)
    logger.info(f"Ajuste de tamaño de compra (FEARFUL): {original_size} -> {buy_size}")
    
    # Probar ajuste de tamaño (venta)
    sell_size = await adapter.adjust_order_size(original_size, 0.8, False)
    logger.info(f"Ajuste de tamaño de venta (FEARFUL): {original_size} -> {sell_size}")
    
    # Probar validación de operación
    trade_params = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "price": 50000.0,
        "confidence": 0.8  # Alta confianza, pero en estado FEARFUL debería rechazar
    }
    
    valid, reject_reason = await adapter.validate_trade(trade_params)
    logger.info(f"Validación de compra (FEARFUL): {'Válida' if valid else 'Rechazada'}, " +
               f"razón: {reject_reason if not valid else 'N/A'}")
    
    # Probar con confianza extrema (debería seguir rechazando)
    trade_params["confidence"] = 0.94  # Justo por debajo del umbral de 0.95
    valid_extreme, reject_reason_extreme = await adapter.validate_trade(trade_params)
    logger.info(f"Validación de compra con confianza 0.94 (FEARFUL): {'Válida' if valid_extreme else 'Rechazada'}, " +
               f"razón: {reject_reason_extreme if not valid_extreme else 'N/A'}")
    
    return {
        "is_fearful": characteristics["emotional_state"] == EmotionalState.FEARFUL.name,
        "entry_decision": decision,
        "exit_small_gain": exit_decision1,
        "exit_small_loss": exit_decision2,
        "buy_size_reduction": buy_size / original_size,
        "sell_size_change": sell_size / original_size,
        "validation_normal": valid,
        "validation_extreme": valid_extreme
    }

async def print_comparison(normal_results, fearful_results):
    """Imprimir comparación de resultados."""
    logger.info("\n=== COMPARACIÓN DE COMPORTAMIENTO NORMAL VS FEARFUL (100%) ===")
    
    # Decisión de entrada
    logger.info(f"Entrada en operación:")
    logger.info(f"  - Normal: {'SÍ' if normal_results['entry_decision'] else 'NO'}")
    logger.info(f"  - FEARFUL: {'SÍ' if fearful_results['entry_decision'] else 'NO'}")
    
    # Decisión de salida
    logger.info(f"Salida con ganancia:")
    logger.info(f"  - Normal: {'SÍ' if normal_results['exit_decision'] else 'NO'}")
    logger.info(f"  - FEARFUL con ganancia pequeña: {'SÍ' if fearful_results['exit_small_gain'] else 'NO'}")
    logger.info(f"  - FEARFUL con pérdida pequeña: {'SÍ' if fearful_results['exit_small_loss'] else 'NO'}")
    
    # Ajuste de tamaño
    logger.info(f"Ajuste de tamaño:")
    logger.info(f"  - Normal: {normal_results['size_adjustment']:.2f}x")
    logger.info(f"  - FEARFUL (compra): {fearful_results['buy_size_reduction']:.2f}x")
    logger.info(f"  - FEARFUL (venta): {fearful_results['sell_size_change']:.2f}x")
    
    # Validación
    logger.info(f"Validación de operaciones:")
    logger.info(f"  - Normal: {'Aceptada' if normal_results['validation'] else 'Rechazada'}")
    logger.info(f"  - FEARFUL (confianza 0.8): {'Aceptada' if fearful_results['validation_normal'] else 'Rechazada'}")
    logger.info(f"  - FEARFUL (confianza 0.94): {'Aceptada' if fearful_results['validation_extreme'] else 'Rechazada'}")
    
    # Verificación de implementación 100%
    criteria_met = (
        not fearful_results['entry_decision'] and  # Rechaza entradas
        fearful_results['exit_small_gain'] and     # Sale con ganancia pequeña
        fearful_results['exit_small_loss'] and     # Sale con pérdida pequeña
        fearful_results['buy_size_reduction'] <= 0.2 and  # Reduce tamaño compra al 20% o menos
        fearful_results['sell_size_change'] >= 1.2 and    # Aumenta tamaño venta al menos 20%
        not fearful_results['validation_normal'] and      # Rechaza validación normal
        not fearful_results['validation_extreme']         # Rechaza incluso con confianza alta
    )
    
    logger.info("\n=== RESULTADO FINAL ===")
    if criteria_met:
        logger.info("✅ ÉXITO: La implementación del estado FEARFUL cumple el 100% de los criterios")
    else:
        logger.info("❌ ERROR: La implementación del estado FEARFUL NO cumple el 100% de los criterios")
    
    return criteria_met

async def main():
    """Función principal."""
    logger.info("Iniciando prueba del Motor de Comportamiento Humano Gabriel...")
    
    try:
        # Probar comportamiento normal
        normal_results = await test_normal_behavior()
        
        # Esperar un momento para separar las pruebas
        await asyncio.sleep(1)
        
        # Probar comportamiento FEARFUL
        fearful_results = await test_fearful_behavior()
        
        # Comparar resultados
        success = await print_comparison(normal_results, fearful_results)
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Error durante la prueba: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)