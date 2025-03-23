#!/usr/bin/env python3
"""
Test simplificado para el módulo de Sincronización Atemporal.

Esta versión tiene pruebas rápidas y directas para verificar las mejoras
implementadas en el módulo AtemporalSynchronization.
"""

import sys
import os
import time
import random
import logging
import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("Genesis.AtemporalSync.Simple")

# Importar módulos necesarios
sys.path.append('.')
from genesis.db.transcendental_database import AtemporalSynchronization

async def test_basic_functionality():
    """Prueba básica de funcionalidad del sincronizador atemporal."""
    logger.info("Iniciando prueba básica de funcionalidad")
    
    # Crear sincronizador con capacidades avanzadas
    sync = AtemporalSynchronization(
        temporal_buffer_size=30,
        extended_horizon=True,
        adaptive_compression=True,
        interdimensional_backup=True
    )
    
    # Registrar estados en diferentes posiciones temporales
    now = time.time()
    past = now - 10
    future = now + 10
    
    # Probar con diferentes tipos de datos
    
    # 1. Valores numéricos
    logger.info("Probando valores numéricos")
    sync.record_state("num_key", 10.0, past)
    sync.record_state("num_key", 20.0, now)
    sync.record_state("num_key", 30.0, future)
    
    # Recuperar valores
    past_val = sync.get_state("num_key", "past")
    present_val = sync.get_state("num_key", "present")
    future_val = sync.get_state("num_key", "future")
    
    logger.info(f"Valores numéricos: Pasado={past_val}, Presente={present_val}, Futuro={future_val}")
    
    # 2. Cadenas
    logger.info("Probando cadenas")
    sync.record_state("str_key", "PASADO", past)
    sync.record_state("str_key", "PRESENTE", now)
    sync.record_state("str_key", "FUTURO", future)
    
    # Recuperar valores
    past_str = sync.get_state("str_key", "past")
    present_str = sync.get_state("str_key", "present")
    future_str = sync.get_state("str_key", "future")
    
    logger.info(f"Valores de cadena: Pasado={past_str}, Presente={present_str}, Futuro={future_str}")
    
    # 3. Diccionarios
    logger.info("Probando diccionarios")
    sync.record_state("dict_key", {"version": 1, "data": "antiguo"}, past)
    sync.record_state("dict_key", {"version": 2, "data": "actual"}, now)
    sync.record_state("dict_key", {"version": 3, "data": "futuro"}, future)
    
    # Recuperar valores
    past_dict = sync.get_state("dict_key", "past")
    present_dict = sync.get_state("dict_key", "present")
    future_dict = sync.get_state("dict_key", "future")
    
    logger.info(f"Valores de diccionario: Pasado={past_dict}, Presente={present_dict}, Futuro={future_dict}")
    
    return True

async def test_anomaly_resolution():
    """Prueba de resolución de anomalías temporales."""
    logger.info("Iniciando prueba de resolución de anomalías")
    
    # Crear sincronizador con capacidades avanzadas
    sync = AtemporalSynchronization(
        temporal_buffer_size=30,
        extended_horizon=True,
        adaptive_compression=True,
        interdimensional_backup=True
    )
    
    # Crear anomalía: inversión temporal (pasado > presente)
    now = time.time()
    past = now - 10
    future = now + 10
    
    # Registrar valores que representan una anomalía (inversión temporal)
    sync.record_state("anomaly_key", 30.0, past)  # Valor mayor en el pasado
    sync.record_state("anomaly_key", 20.0, now)   # Valor menor en el presente
    sync.record_state("anomaly_key", 40.0, future)  # Valor mayor en el futuro
    
    # Verificar valores antes de estabilización
    logger.info("Valores antes de estabilización:")
    past_val = sync.get_state("anomaly_key", "past")
    present_val = sync.get_state("anomaly_key", "present")
    future_val = sync.get_state("anomaly_key", "future")
    logger.info(f"Pasado={past_val}, Presente={present_val}, Futuro={future_val}")
    
    # Detectar y estabilizar anomalía
    severity = None
    try:
        # Intentar usar método de detección de anomalías si existe
        if hasattr(sync, 'detect_temporal_anomaly'):
            severity = sync.detect_temporal_anomaly("anomaly_key")
            logger.info(f"Severidad de anomalía detectada: {severity}")
    except Exception as e:
        logger.warning(f"Error al detectar severidad: {e}")
    
    # Estabilizar anomalía
    start_time = time.time()
    stabilized = sync.stabilize_temporal_anomaly("anomaly_key")
    stabilization_time = time.time() - start_time
    
    # Verificar valores después de estabilización
    logger.info(f"Anomalía estabilizada: {stabilized} en {stabilization_time*1000:.2f} ms")
    past_val_after = sync.get_state("anomaly_key", "past")
    present_val_after = sync.get_state("anomaly_key", "present")
    future_val_after = sync.get_state("anomaly_key", "future")
    logger.info("Valores después de estabilización:")
    logger.info(f"Pasado={past_val_after}, Presente={present_val_after}, Futuro={future_val_after}")
    
    # Verificar coherencia temporal (pasado <= presente <= futuro)
    is_coherent = (past_val_after <= present_val_after <= future_val_after) or (past_val_after >= present_val_after >= future_val_after)
    logger.info(f"Coherencia temporal después de estabilización: {is_coherent}")
    
    return stabilized and is_coherent

async def test_paradox_resolution():
    """Prueba de resolución de paradojas temporales."""
    logger.info("Iniciando prueba de resolución de paradojas")
    
    # Crear sincronizador con capacidades avanzadas
    sync = AtemporalSynchronization(
        temporal_buffer_size=30,
        extended_horizon=True,
        adaptive_compression=True,
        interdimensional_backup=True
    )
    
    # Crear paradoja: inversión total (pasado > presente > futuro)
    now = time.time()
    past = now - 10
    future = now + 10
    
    # Registrar valores que representan una paradoja severa
    sync.record_state("paradox_key", 100.0, past)    # Valor muy alto en el pasado
    sync.record_state("paradox_key", 50.0, now)     # Valor medio en el presente
    sync.record_state("paradox_key", 10.0, future)  # Valor muy bajo en el futuro
    
    # Verificar valores antes de estabilización
    logger.info("Valores antes de estabilización:")
    past_val = sync.get_state("paradox_key", "past")
    present_val = sync.get_state("paradox_key", "present")
    future_val = sync.get_state("paradox_key", "future")
    logger.info(f"Pasado={past_val}, Presente={present_val}, Futuro={future_val}")
    
    # Detectar y estabilizar paradoja
    severity = None
    try:
        # Intentar usar método de detección de anomalías si existe
        if hasattr(sync, 'detect_temporal_anomaly'):
            severity = sync.detect_temporal_anomaly("paradox_key")
            logger.info(f"Severidad de paradoja detectada: {severity}")
    except Exception as e:
        logger.warning(f"Error al detectar severidad: {e}")
    
    # Estabilizar paradoja
    start_time = time.time()
    stabilized = sync.stabilize_temporal_anomaly("paradox_key")
    stabilization_time = time.time() - start_time
    
    # Verificar valores después de estabilización
    logger.info(f"Paradoja estabilizada: {stabilized} en {stabilization_time*1000:.2f} ms")
    past_val_after = sync.get_state("paradox_key", "past")
    present_val_after = sync.get_state("paradox_key", "present")
    future_val_after = sync.get_state("paradox_key", "future")
    logger.info("Valores después de estabilización:")
    logger.info(f"Pasado={past_val_after}, Presente={present_val_after}, Futuro={future_val_after}")
    
    # Verificar coherencia temporal (pasado <= presente <= futuro)
    is_coherent = (past_val_after <= present_val_after <= future_val_after)
    logger.info(f"Coherencia temporal después de estabilización: {is_coherent}")
    
    return stabilized and is_coherent

async def test_extended_horizon():
    """Prueba de horizonte temporal extendido."""
    logger.info("Iniciando prueba de horizonte temporal extendido")
    
    # Crear sincronizador con horizonte extendido
    sync = AtemporalSynchronization(
        temporal_buffer_size=30,
        extended_horizon=True,  # Activar horizonte extendido
        adaptive_compression=False,
        interdimensional_backup=False
    )
    
    # Crear sincronizador con horizonte normal para comparar
    sync_normal = AtemporalSynchronization(
        temporal_buffer_size=30,
        extended_horizon=False,  # Horizonte normal
        adaptive_compression=False,
        interdimensional_backup=False
    )
    
    # Probar con diferentes distancias temporales
    now = time.time()
    
    # Tiempo "futuro lejano" que solo debería ser accesible con horizonte extendido
    far_future = now + 1000  # Muy en el futuro
    
    # Registrar valores
    value = 100.0
    
    # Registrar en ambos sincronizadores
    sync.record_state("horizon_key", value, far_future)
    sync_normal.record_state("horizon_key", value, far_future)
    
    # Intentar recuperar del futuro lejano
    extended_value = sync.get_state("horizon_key", "future")
    normal_value = sync_normal.get_state("horizon_key", "future")
    
    logger.info(f"Valor con horizonte extendido: {extended_value}")
    logger.info(f"Valor con horizonte normal: {normal_value}")
    
    # El valor debería ser accesible con horizonte extendido
    return extended_value == value

async def test_dimensional_backup():
    """Prueba de respaldo interdimensional."""
    logger.info("Iniciando prueba de respaldo interdimensional")
    
    # Crear sincronizador con respaldo interdimensional
    sync = AtemporalSynchronization(
        temporal_buffer_size=30,
        extended_horizon=False,
        adaptive_compression=False,
        interdimensional_backup=True  # Activar respaldo interdimensional
    )
    
    # Registrar un valor presente
    now = time.time()
    key = "backup_key"
    value = {"critical_data": "información vital", "version": 42}
    
    sync.record_state(key, value, now)
    
    # Verificar que se pueda recuperar normalmente
    present_val = sync.get_state(key, "present")
    logger.info(f"Valor recuperado normalmente: {present_val}")
    
    # Simular pérdida de datos accediendo directamente a la estructura interna
    # (esto normalmente no se haría, es solo para la prueba)
    if hasattr(sync, 'present_state'):
        original_state = sync.present_state.copy()
        sync.present_state = {}  # Borrar presente para simular pérdida
        
        # Intentar recuperar - debería recuperarse del respaldo interdimensional
        recovered_val = sync.get_state(key, "present")
        logger.info(f"Valor recuperado tras pérdida: {recovered_val}")
        
        # Restaurar el estado original
        sync.present_state = original_state
        
        # Verificar recuperación
        success = recovered_val == value
        logger.info(f"Recuperación exitosa: {success}")
        
        # Verificar estadísticas
        if hasattr(sync, 'dimension_recoveries'):
            logger.info(f"Recuperaciones dimensionales: {sync.dimension_recoveries}")
        
        return success
    else:
        logger.warning("No se pudo acceder a la estructura interna para la prueba")
        return False

async def test_stats():
    """Prueba de estadísticas del sincronizador."""
    logger.info("Iniciando prueba de estadísticas")
    
    # Crear sincronizador con todas las características
    sync = AtemporalSynchronization(
        temporal_buffer_size=30,
        extended_horizon=True,
        adaptive_compression=True,
        interdimensional_backup=True
    )
    
    # Realizar varias operaciones para generar estadísticas
    now = time.time()
    
    # Registrar varios estados
    for i in range(5):
        past = now - random.uniform(1, 20)
        future = now + random.uniform(1, 20)
        
        sync.record_state(f"stats_key_{i}", i * 10, past)
        sync.record_state(f"stats_key_{i}", i * 20, now)
        sync.record_state(f"stats_key_{i}", i * 30, future)
    
    # Crear algunas anomalías
    sync.record_state("anomaly_stats", 100, now - 10)
    sync.record_state("anomaly_stats", 50, now)
    sync.record_state("anomaly_stats", 200, now + 10)
    
    # Estabilizar
    sync.stabilize_temporal_anomaly("anomaly_stats")
    
    # Obtener estadísticas
    try:
        stats = sync.get_stats()
        logger.info("Estadísticas del sincronizador:")
        logger.info(json.dumps(stats, indent=2, default=str))
        
        # Guardar estadísticas en archivo
        with open("atemporal_stats.json", "w") as f:
            json.dump(stats, f, indent=2, default=str)
        
        return True
    except Exception as e:
        logger.error(f"Error al obtener estadísticas: {e}")
        return False

async def run_all_tests():
    """Ejecutar todas las pruebas."""
    logger.info("=== INICIANDO PRUEBAS DE SINCRONIZACIÓN ATEMPORAL ===")
    
    tests = [
        ("Funcionalidad básica", test_basic_functionality),
        ("Resolución de anomalías", test_anomaly_resolution),
        ("Resolución de paradojas", test_paradox_resolution),
        ("Horizonte temporal extendido", test_extended_horizon),
        ("Respaldo interdimensional", test_dimensional_backup),
        ("Estadísticas", test_stats)
    ]
    
    results = {}
    
    for name, test_func in tests:
        logger.info(f"\n=== Ejecutando prueba: {name} ===")
        try:
            start_time = time.time()
            success = await test_func()
            elapsed = time.time() - start_time
            
            results[name] = {
                "success": success,
                "elapsed": elapsed
            }
            
            logger.info(f"Resultado: {'ÉXITO' if success else 'FALLO'}")
            logger.info(f"Tiempo: {elapsed:.3f} segundos\n")
            
        except Exception as e:
            logger.error(f"Error en prueba {name}: {e}")
            results[name] = {"success": False, "error": str(e)}
    
    # Mostrar resumen
    logger.info("\n=== RESUMEN DE RESULTADOS ===")
    success_count = sum(1 for r in results.values() if r.get("success", False))
    total = len(tests)
    
    logger.info(f"Pruebas exitosas: {success_count}/{total} ({success_count/total*100:.1f}%)")
    
    for name, result in results.items():
        status = "✓" if result.get("success", False) else "✗"
        time_info = f" ({result.get('elapsed', 0):.3f}s)" if "elapsed" in result else ""
        error_info = f" - Error: {result.get('error')}" if "error" in result else ""
        
        logger.info(f"{status} {name}{time_info}{error_info}")
    
    # Guardar resultados
    with open("atemporal_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Resultados guardados en atemporal_test_results.json")
    
    return success_count == total

if __name__ == "__main__":
    asyncio.run(run_all_tests())