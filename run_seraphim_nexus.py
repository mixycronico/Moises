#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejecutor del Sistema Genesis Ultra-Divino Trading Nexus 10M

Este script inicia el Sistema Genesis con la estrategia Seraphim Pool,
implementando el concepto completo de HumanPoolTrader con nombres celestiales
y capacidades divinas.

El sistema integra:
- Comportamiento humano simulado (Gabriel)
- Análisis de mercado superior (Buddha)
- Clasificación trascendental de activos
- Gestión de riesgo adaptativa
- Ciclos de trading con capital limitado
- Distribución equitativa de ganancias

Autor: Genesis AI Assistant
Versión: 1.0.0 (Divina)
"""

import os
import sys
import json
import logging
import asyncio
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import argparse
import traceback

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("seraphim_nexus.log", mode="a")
    ]
)
logger = logging.getLogger("seraphim.nexus")


class SeraphimNexusRunner:
    """Ejecutor del Sistema Genesis Ultra-Divino Trading Nexus."""
    
    def __init__(self):
        """Inicializar ejecutor del Sistema Genesis Ultra-Divino Trading Nexus."""
        self.config = None
        self.orchestrator = None
        self.behavior_engine = None
        self.buddha_integrator = None
        self.classifier = None
        self.cloud_circuit_breaker = None
        self.distributed_checkpoint_manager = None
        self.cloud_load_balancer = None
        self.transcendental_database = None
        self.oracle = None
        self.alert_manager = None
        self.initialized = False
        self.auto_mode_running = False
        
    async def load_configuration(self, config_path: str = "seraphim_config.json") -> bool:
        """
        Cargar configuración desde archivo.
        
        Args:
            config_path: Ruta al archivo de configuración
            
        Returns:
            True si la configuración se cargó correctamente
        """
        try:
            if not os.path.exists(config_path):
                logger.warning(f"Archivo de configuración {config_path} no encontrado")
                # Crear configuración por defecto
                self.config = {
                    "system_name": "Genesis Ultra-Divino Trading Nexus 10M",
                    "version": "1.0.0",
                    "pool_size": 5,
                    "participants": ["Metatron", "Gabriel", "Uriel", "Rafael", "Miguel"],
                    "total_capital": 10000.0,
                    "capital_per_cycle": 150.0,
                    "buddha_enabled": True,
                    "max_positions": 5,
                    "risk_level": "MODERADO",
                    "mode": "DIVINE"
                }
                # Guardar configuración
                with open(config_path, 'w') as f:
                    json.dump(self.config, f, indent=4)
                logger.info(f"Configuración por defecto creada en {config_path}")
            else:
                # Cargar configuración existente
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Configuración cargada desde {config_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error al cargar/crear configuración: {e}")
            logger.error(traceback.format_exc())
            return False

    async def initialize_components(self) -> bool:
        """
        Inicializar todos los componentes del sistema.
        
        Returns:
            True si todos los componentes se inicializaron correctamente
        """
        try:
            # Importar componentes necesarios
            try:
                from genesis.trading.seraphim_orchestrator import SeraphimOrchestrator
                from genesis.trading.human_behavior_engine import GabrielBehaviorEngine
                from genesis.trading.buddha_integrator import BuddhaIntegrator
                from genesis.analysis.transcendental_crypto_classifier import TranscendentalCryptoClassifier
                from genesis.risk.adaptive_risk_manager import AdaptiveRiskManager
                from genesis.cloud.circuit_breaker_v4 import CloudCircuitBreakerV4
                from genesis.cloud.distributed_checkpoint_v4 import DistributedCheckpointManagerV4
                from genesis.cloud.load_balancer_v4 import CloudLoadBalancerV4
                from genesis.db.transcendental_database import TranscendentalDatabase
                from genesis.trading.quantum_oracle import QuantumOracle
                from genesis.notifications.alert_manager import AlertManager
                
                # Inicializar componentes
                self.orchestrator = SeraphimOrchestrator()
                self.behavior_engine = GabrielBehaviorEngine()
                self.buddha_integrator = BuddhaIntegrator()
                self.classifier = TranscendentalCryptoClassifier()
                self.risk_manager = AdaptiveRiskManager()
                self.cloud_circuit_breaker = CloudCircuitBreakerV4()
                self.distributed_checkpoint_manager = DistributedCheckpointManagerV4()
                self.cloud_load_balancer = CloudLoadBalancerV4()
                self.transcendental_database = TranscendentalDatabase()
                self.oracle = QuantumOracle()
                self.alert_manager = AlertManager()
                
                logger.info("Componentes importados correctamente")
            except ImportError as e:
                logger.error(f"Error al importar componentes: {e}")
                logger.error(traceback.format_exc())
                return False
            
            # Verificar salud del sistema
            health_status = await self.check_system_health()
            if not health_status["all_healthy"]:
                logger.warning(f"No todos los componentes están saludables: {health_status}")
            
            # Inicializar orquestador
            await self.orchestrator.initialize()
            
            # Marcar como inicializado
            self.initialized = True
            logger.info("Sistema Genesis Ultra-Divino Trading Nexus 10M inicializado correctamente")
            
            return True
        except Exception as e:
            logger.error(f"Error al inicializar componentes: {e}")
            logger.error(traceback.format_exc())
            return False

    async def start_system(self) -> bool:
        """
        Iniciar el sistema completo.
        
        Returns:
            True si el sistema se inició correctamente
        """
        try:
            if not self.initialized:
                success = await self.initialize_components()
                if not success:
                    logger.error("No se pudo inicializar el sistema")
                    return False
            
            # Mostrar información del sistema
            await self.display_system_info()
            
            logger.info("Sistema iniciado correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al iniciar sistema: {e}")
            logger.error(traceback.format_exc())
            return False

    async def display_system_info(self) -> None:
        """Mostrar información del sistema."""
        try:
            if not self.initialized:
                logger.warning("Sistema no inicializado")
                return
            
            # Obtener información del sistema
            system_overview = await self.orchestrator.get_system_overview()
            
            # Imprimir información
            logger.info(f"=== Información del Sistema ===")
            logger.info(f"Nombre: {self.config['system_name']}")
            logger.info(f"Versión: {self.config['version']}")
            logger.info(f"Estado: {system_overview.get('orchestrator_state', 'Desconocido')}")
            logger.info(f"Salud: {system_overview.get('health_score', 0.0) * 100:.2f}%")
            logger.info(f"Participantes: {', '.join(self.config['participants'])}")
            logger.info(f"Capital total: ${self.config['total_capital']:.2f}")
            logger.info(f"Capital por ciclo: ${self.config['capital_per_cycle']:.2f}")
            logger.info(f"Estado Buddha: {'Activado' if self.config['buddha_enabled'] else 'Desactivado'}")
            logger.info(f"Ciclos completados: {system_overview.get('completed_cycles_count', 0)}")
            logger.info(f"Ganancia total: ${system_overview.get('total_profit', 0.0):.2f}")
            logger.info(f"================================")
        except Exception as e:
            logger.error(f"Error al mostrar información del sistema: {e}")
            logger.error(traceback.format_exc())

    async def run_interactive(self) -> None:
        """Ejecutar el sistema en modo interactivo."""
        try:
            if not self.initialized:
                success = await self.start_system()
                if not success:
                    logger.error("No se pudo iniciar el sistema")
                    return
            
            logger.info("Modo interactivo iniciado")
            
            while True:
                print("\n=== Genesis Ultra-Divino Trading Nexus 10M ===")
                print("1. Iniciar nuevo ciclo de trading")
                print("2. Procesar ciclo activo")
                print("3. Ver estado del sistema")
                print("4. Ver información del ciclo activo")
                print("5. Aleatorizar comportamiento humano")
                print("6. Iniciar operación autónoma")
                print("7. Detener operación autónoma")
                print("0. Salir")
                
                option = input("\nSeleccione una opción: ")
                
                if option == "1":
                    await self.orchestrator.start_trading_cycle()
                    logger.info("Ciclo de trading iniciado")
                elif option == "2":
                    await self.orchestrator.process_cycle()
                    logger.info("Ciclo procesado")
                elif option == "3":
                    system_overview = await self.orchestrator.get_system_overview()
                    print("\n=== Estado del Sistema ===")
                    print(f"Estado: {system_overview.get('orchestrator_state', 'Desconocido')}")
                    print(f"Salud: {system_overview.get('health_score', 0.0) * 100:.2f}%")
                    print(f"Ciclos completados: {system_overview.get('completed_cycles_count', 0)}")
                    print(f"Ganancia total: ${system_overview.get('total_profit', 0.0):.2f}")
                elif option == "4":
                    cycle_status = await self.orchestrator.get_cycle_status()
                    if cycle_status:
                        print("\n=== Información del Ciclo Activo ===")
                        print(f"ID: {cycle_status.get('cycle_id', 'N/A')}")
                        print(f"Estado: {cycle_status.get('status', 'Desconocido')}")
                        print(f"Fase: {cycle_status.get('phase', 'N/A')}")
                        
                        performance = cycle_status.get('performance', {})
                        if performance:
                            print(f"Capital inicial: ${performance.get('starting_capital', 0.0):.2f}")
                            print(f"Capital actual: ${performance.get('current_capital', 0.0):.2f}")
                            print(f"ROI: {performance.get('roi_percentage', 0.0):.2f}%")
                    else:
                        print("No hay ciclo activo")
                elif option == "5":
                    new_characteristics = await self.behavior_engine.randomize()
                    print("\n=== Nuevo Comportamiento Humano ===")
                    print(f"Estado emocional: {new_characteristics.get('emotional_state', 'N/A')}")
                    print(f"Tolerancia al riesgo: {new_characteristics.get('risk_tolerance', 'N/A')}")
                    print(f"Estilo de decisión: {new_characteristics.get('decision_style', 'N/A')}")
                    
                    market_perceptions = new_characteristics.get('market_perceptions', {})
                    if market_perceptions:
                        print(f"Sentimiento de mercado: {market_perceptions.get('market_sentiment', 'N/A')}")
                elif option == "6":
                    duration_hours = input("Duración en horas (deje en blanco para indefinido): ")
                    duration = int(duration_hours) if duration_hours.strip() else None
                    await self.orchestrator.run_autonomous_operation(duration)
                    self.auto_mode_running = True
                    logger.info(f"Operación autónoma iniciada por {duration} horas" if duration else "Operación autónoma iniciada indefinidamente")
                elif option == "7":
                    if self.auto_mode_running:
                        await self.orchestrator.stop_autonomous_operation()
                        self.auto_mode_running = False
                        logger.info("Operación autónoma detenida")
                    else:
                        print("Operación autónoma no está en ejecución")
                elif option == "0":
                    if self.auto_mode_running:
                        await self.orchestrator.stop_autonomous_operation()
                        self.auto_mode_running = False
                    await self.shutdown()
                    break
                else:
                    print("Opción no válida")
                
                # Pequeña pausa para no saturar la consola
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Ejecución interrumpida por el usuario")
            if self.auto_mode_running:
                await self.orchestrator.stop_autonomous_operation()
            await self.shutdown()
        except Exception as e:
            logger.error(f"Error en modo interactivo: {e}")
            logger.error(traceback.format_exc())
            await self.shutdown()

    async def run_autonomous(self, duration_hours: Optional[int] = None) -> None:
        """
        Ejecutar el sistema en modo autónomo.
        
        Args:
            duration_hours: Duración en horas, o None para indefinido
        """
        try:
            if not self.initialized:
                success = await self.start_system()
                if not success:
                    logger.error("No se pudo iniciar el sistema")
                    return
            
            # Iniciar operación autónoma
            await self.orchestrator.run_autonomous_operation(duration_hours)
            self.auto_mode_running = True
            
            if duration_hours:
                logger.info(f"Modo autónomo iniciado por {duration_hours} horas")
            else:
                logger.info("Modo autónomo iniciado indefinidamente")
            
            try:
                # Si es por tiempo definido, esperar
                if duration_hours:
                    end_time = datetime.now() + timedelta(hours=duration_hours)
                    while datetime.now() < end_time and self.auto_mode_running:
                        time_left = end_time - datetime.now()
                        hours_left = time_left.total_seconds() / 3600
                        logger.info(f"Tiempo restante: {hours_left:.2f} horas")
                        await asyncio.sleep(300)  # Revisar cada 5 minutos
                else:
                    # Si es indefinido, simplemente mantener vivo
                    while self.auto_mode_running:
                        system_overview = await self.orchestrator.get_system_overview()
                        logger.info(f"Estado del sistema: {system_overview.get('orchestrator_state', 'Desconocido')}")
                        logger.info(f"Salud: {system_overview.get('health_score', 0.0) * 100:.2f}%")
                        logger.info(f"Ciclos completados: {system_overview.get('completed_cycles_count', 0)}")
                        logger.info(f"Ganancia total: ${system_overview.get('total_profit', 0.0):.2f}")
                        await asyncio.sleep(600)  # Revisar cada 10 minutos
            except KeyboardInterrupt:
                logger.info("Ejecución interrumpida por el usuario")
            finally:
                # Asegurarse de detener la operación autónoma
                if self.auto_mode_running:
                    await self.orchestrator.stop_autonomous_operation()
                    self.auto_mode_running = False
                    logger.info("Operación autónoma detenida")
        except Exception as e:
            logger.error(f"Error en modo autónomo: {e}")
            logger.error(traceback.format_exc())
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Cerrar el sistema de forma segura."""
        try:
            if self.initialized:
                logger.info("Cerrando el sistema...")
                
                # Detener operación autónoma si está activa
                if self.auto_mode_running:
                    await self.orchestrator.stop_autonomous_operation()
                    self.auto_mode_running = False
                
                # Cerrar componentes
                if hasattr(self.orchestrator, 'shutdown'):
                    await self.orchestrator.shutdown()
                
                logger.info("Sistema cerrado correctamente")
        except Exception as e:
            logger.error(f"Error al cerrar el sistema: {e}")
            logger.error(traceback.format_exc())


async def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Genesis Ultra-Divino Trading Nexus 10M')
    parser.add_argument('-c', '--config', type=str, default='seraphim_config.json',
                        help='Ruta al archivo de configuración')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Ejecutar en modo interactivo')
    parser.add_argument('-d', '--duration', type=int, default=None,
                        help='Duración en horas para modo autónomo (None para indefinido)')
    
    args = parser.parse_args()
    
    try:
        # Crear y ejecutar el runner
        runner = SeraphimNexusRunner()
        
        # Cargar configuración
        config_loaded = await runner.load_configuration(args.config)
        if not config_loaded:
            logger.error("No se pudo cargar la configuración")
            return
        
        # Ejecutar en modo interactivo o autónomo
        if args.interactive:
            await runner.run_interactive()
        else:
            await runner.run_autonomous(args.duration)
    except KeyboardInterrupt:
        logger.info("Ejecución interrumpida por el usuario")
    except Exception as e:
        logger.error(f"Error en ejecución principal: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    try:
        # Ejecutar el bucle de eventos
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Programa terminado por el usuario")
    except Exception as e:
        logger.error(f"Error fatal: {e}")
        logger.error(traceback.format_exc())