#!/usr/bin/env python3
"""
Test minimalista para el módulo de Sincronización Atemporal.

Esta versión ejecuta una única prueba básica y rápida para verificar
la funcionalidad del módulo AtemporalSynchronization mejorado.
"""
import sys
import time
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("Genesis.AtemporalSync.Minimal")

# Importar módulos necesarios
sys.path.append('.')
from genesis.db.transcendental_database import AtemporalSynchronization

def run_minimal_test():
    """Ejecutar prueba mínima de funcionalidad del sincronizador atemporal."""
    logger.info("=== INICIANDO PRUEBA MÍNIMA DE SINCRONIZACIÓN ATEMPORAL ===")
    
    # Crear sincronizador con capacidades avanzadas
    sync = AtemporalSynchronization(
        temporal_buffer_size=10,          # Tamaño mínimo
        extended_horizon=True,           # Horizonte extendido activado
        adaptive_compression=True,       # Compresión adaptativa activada
        interdimensional_backup=True     # Respaldo dimensional activado
    )
    
    # 1. Probar registro y recuperación básica
    logger.info("1. Probando registro y recuperación básica")
    
    now = time.time()
    past = now - 5.0
    future = now + 5.0
    
    # Registrar valores
    sync.record_state("test_key", 10.0, past)
    sync.record_state("test_key", 20.0, now)
    sync.record_state("test_key", 30.0, future)
    
    # Recuperar valores
    past_val = sync.get_state("test_key", "past")
    present_val = sync.get_state("test_key", "present")
    future_val = sync.get_state("test_key", "future")
    
    logger.info(f"Valores recuperados: Pasado={past_val}, Presente={present_val}, Futuro={future_val}")
    basic_test = (past_val == 10.0 and present_val == 20.0 and future_val == 30.0)
    logger.info(f"Prueba básica: {'ÉXITO' if basic_test else 'FALLO'}")
    
    # 2. Probar estabilización de anomalía simple
    logger.info("2. Probando estabilización de anomalía")
    
    # Crear una anomalía (inversión temporal)
    sync.record_state("anomaly_key", 30.0, past)    # Mayor en el pasado
    sync.record_state("anomaly_key", 20.0, now)     # Menor en el presente
    sync.record_state("anomaly_key", 40.0, future)  # Mayor en el futuro
    
    # Valores antes de estabilización
    before_past = sync.get_state("anomaly_key", "past")
    before_present = sync.get_state("anomaly_key", "present")
    before_future = sync.get_state("anomaly_key", "future")
    logger.info(f"Antes de estabilización: P={before_past}, Pr={before_present}, F={before_future}")
    
    # Estabilizar
    stabilized = sync.stabilize_temporal_anomaly("anomaly_key")
    
    # Valores después de estabilización
    after_past = sync.get_state("anomaly_key", "past")
    after_present = sync.get_state("anomaly_key", "present")
    after_future = sync.get_state("anomaly_key", "future")
    logger.info(f"Después de estabilización: P={after_past}, Pr={after_present}, F={after_future}")
    
    # Verificar coherencia
    is_coherent = (after_past <= after_present <= after_future) or (after_past >= after_present >= after_future)
    anomaly_test = stabilized and is_coherent
    logger.info(f"Prueba de anomalía: {'ÉXITO' if anomaly_test else 'FALLO'}")
    
    # 3. Verificar estadísticas
    logger.info("3. Verificando estadísticas")
    
    try:
        stats = sync.get_stats()
        logger.info(f"Estadísticas disponibles: {list(stats.keys())}")
        stats_test = len(stats) > 0
    except Exception as e:
        logger.error(f"Error al obtener estadísticas: {e}")
        stats_test = False
    
    logger.info(f"Prueba de estadísticas: {'ÉXITO' if stats_test else 'FALLO'}")
    
    # Resultado general
    all_passed = basic_test and anomaly_test and stats_test
    logger.info(f"\nResultado general: {'ÉXITO' if all_passed else 'FALLO'}")
    logger.info(f"- Prueba básica: {'✓' if basic_test else '✗'}")
    logger.info(f"- Prueba de anomalía: {'✓' if anomaly_test else '✗'}")
    logger.info(f"- Prueba de estadísticas: {'✓' if stats_test else '✗'}")
    
    return all_passed

if __name__ == "__main__":
    result = run_minimal_test()
    sys.exit(0 if result else 1)