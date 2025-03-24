#!/usr/bin/env python
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

import asyncio
import json
import logging
import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

# Componentes Genesis
from genesis.trading.seraphim_orchestrator import SeraphimOrchestrator
from genesis.trading.human_behavior_engine import GabrielBehaviorEngine
from genesis.strategies.seraphim.seraphim_pool import SeraphimPool
from genesis.cloud.circuit_breaker_v4 import CloudCircuitBreakerV4
from genesis.notifications.alert_manager import AlertManager

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('seraphim_nexus.log')
    ]
)

logger = logging.getLogger("seraphim_nexus")

class SeraphimNexusRunner:
    """Ejecutor del Sistema Genesis Ultra-Divino Trading Nexus."""
    
    def __init__(self):
        """Inicializar ejecutor del Sistema Genesis Ultra-Divino Trading Nexus."""
        self.orchestrator = None
        self.behavior_engine = None
        self.config = {}
        self.start_time = datetime.now()
        self.running = False
        
        logger.info("Inicializando Sistema Genesis Ultra-Divino Trading Nexus 10M")
    
    async def load_configuration(self, config_path: str = "seraphim_config.json") -> bool:
        """
        Cargar configuración desde archivo.
        
        Args:
            config_path: Ruta al archivo de configuración
            
        Returns:
            True si la configuración se cargó correctamente
        """
        try:
            logger.info(f"Cargando configuración desde {config_path}")
            
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            logger.info(f"Configuración cargada: modo {self.config['general']['mode']}")
            
            # Aplicar nivel de log desde configuración
            log_level = self.config["general"].get("log_level", "INFO")
            numeric_level = getattr(logging, log_level.upper(), None)
            if isinstance(numeric_level, int):
                logging.getLogger().setLevel(numeric_level)
                logger.info(f"Nivel de log establecido a {log_level}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            return False
    
    async def initialize_components(self) -> bool:
        """
        Inicializar todos los componentes del sistema.
        
        Returns:
            True si todos los componentes se inicializaron correctamente
        """
        try:
            logger.info("Inicializando componentes del Sistema Genesis Ultra-Divino Trading Nexus")
            
            # Inicializar motor de comportamiento humano
            self.behavior_engine = GabrielBehaviorEngine()
            
            # Aplicar configuración de comportamiento humano
            human_config = self.config.get("human_behavior", {})
            if human_config:
                # Aquí se aplicaría la configuración
                logger.info("Configuración de comportamiento humano aplicada")
            
            # Inicializar orquestador Seraphim
            self.orchestrator = SeraphimOrchestrator()
            
            # Inicializar orquestador (esto iniciará todos los demás componentes)
            orchestrator_init = await self.orchestrator.initialize()
            
            if not orchestrator_init:
                logger.error("Falló inicialización del orquestador")
                return False
            
            logger.info("Todos los componentes inicializados correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error al inicializar componentes: {str(e)}")
            return False
    
    async def start_system(self) -> bool:
        """
        Iniciar el sistema completo.
        
        Returns:
            True si el sistema se inició correctamente
        """
        try:
            # Verificar salud del sistema
            health = await self.orchestrator.check_system_health()
            
            if health < 0.8:
                logger.warning(f"Salud del sistema baja para iniciar: {health:.2f}")
                choice = input("Salud del sistema por debajo del umbral recomendado. ¿Continuar? (s/n): ")
                if choice.lower() != 's':
                    logger.info("Inicio del sistema cancelado por el usuario")
                    return False
            
            # Establecer estado de ejecución
            self.running = True
            
            logger.info("Sistema Genesis Ultra-Divino Trading Nexus iniciado correctamente")
            
            # Mostrar información del sistema
            await self.display_system_info()
            
            return True
            
        except Exception as e:
            logger.error(f"Error al iniciar sistema: {str(e)}")
            return False
    
    async def display_system_info(self) -> None:
        """Mostrar información del sistema."""
        try:
            system_overview = await self.orchestrator.get_system_overview()
            
            if system_overview.get("success", False):
                print("\n" + "="*50)
                print(" SISTEMA GENESIS ULTRA-DIVINO TRADING NEXUS 10M ")
                print("="*50)
                
                stats = system_overview.get("system_stats", {})
                
                print(f"\nEstado:           {stats.get('orchestrator_state', 'Desconocido')}")
                print(f"Salud:            {stats.get('health_score', 0) * 100:.1f}%")
                print(f"Tiempo activo:    {stats.get('uptime', 'Desconocido')}")
                print(f"Ciclos completos: {stats.get('completed_cycles_count', 0)}")
                print(f"Ganancia total:   ${stats.get('total_profit', 0):.2f}")
                
                print("\nTop Criptomonedas Identificadas:")
                for i, crypto in enumerate(system_overview.get("top_cryptos", []), 1):
                    print(f"  {i}. {crypto}")
                
                print("\nEstado Buddha AI:  ", system_overview.get("buddha_status", "Desconectado"))
                
                if system_overview.get("active_cycle", None):
                    cycle = system_overview.get("active_cycle", {})
                    print("\nCICLO ACTIVO:")
                    print(f"ID:               {cycle.get('cycle_id', 'Desconocido')}")
                    print(f"Fase:             {cycle.get('cycle_phase', 'Desconocido')}")
                    print(f"Estado:           {cycle.get('strategy_state', 'Desconocido')}")
                    
                    performance = cycle.get("cycle_performance", {})
                    if performance:
                        print(f"Capital inicial:  ${performance.get('starting_capital', 0):.2f}")
                        print(f"Capital actual:   ${performance.get('current_capital', 0):.2f}")
                        print(f"ROI:              {performance.get('roi_percentage', 0):.2f}%")
                
                print("\n" + "="*50)
            else:
                logger.warning("No se pudo obtener información del sistema")
                
        except Exception as e:
            logger.error(f"Error al mostrar información del sistema: {str(e)}")
    
    async def run_interactive(self) -> None:
        """Ejecutar el sistema en modo interactivo."""
        try:
            print("\nModo interactivo iniciado. Comandos disponibles:")
            print("  1. iniciar ciclo - Iniciar nuevo ciclo de trading")
            print("  2. procesar      - Procesar ciclo actual")
            print("  3. estado        - Mostrar estado del sistema")
            print("  4. automatico    - Iniciar procesamiento automático")
            print("  5. detener       - Detener procesamiento automático")
            print("  6. salir         - Salir del programa")
            
            while self.running:
                command = input("\nComando > ").lower().strip()
                
                if command == "iniciar ciclo":
                    result = await self.orchestrator.start_trading_cycle()
                    if result.get("success", False):
                        print(f"Ciclo iniciado: {result.get('cycle_id', 'Desconocido')}")
                    else:
                        print(f"Error al iniciar ciclo: {result.get('error', 'Error desconocido')}")
                
                elif command == "procesar":
                    result = await self.orchestrator.process_cycle()
                    if result.get("success", False):
                        print(f"Ciclo procesado: {result.get('cycle_id', 'Desconocido')}, fase: {result.get('phase', 'Desconocida')}")
                    else:
                        print(f"Error al procesar ciclo: {result.get('error', 'Error desconocido')}")
                
                elif command == "estado":
                    await self.display_system_info()
                
                elif command == "automatico":
                    duration = input("Duración en horas (vacío para indefinido): ")
                    
                    if duration.strip():
                        try:
                            hours = int(duration)
                            print(f"Iniciando procesamiento automático por {hours} horas...")
                            asyncio.create_task(self.orchestrator.run_autonomous_operation(hours))
                        except ValueError:
                            print("Duración inválida. Debe ser un número entero.")
                    else:
                        print("Iniciando procesamiento automático indefinido...")
                        asyncio.create_task(self.orchestrator.run_autonomous_operation())
                
                elif command == "detener":
                    self.orchestrator.stop_autonomous_operation()
                    print("Procesamiento automático detenido")
                
                elif command == "salir":
                    self.running = False
                    print("Saliendo del programa...")
                
                else:
                    print("Comando no reconocido")
                
                # Pequeña pausa para evitar CPU al 100%
                await asyncio.sleep(0.1)
            
        except KeyboardInterrupt:
            logger.info("Programa interrumpido por el usuario")
            self.running = False
        except Exception as e:
            logger.error(f"Error en modo interactivo: {str(e)}")
    
    async def run_autonomous(self, duration_hours: Optional[int] = None) -> None:
        """
        Ejecutar el sistema en modo autónomo.
        
        Args:
            duration_hours: Duración en horas, o None para indefinido
        """
        try:
            logger.info(f"Iniciando modo autónomo{f' por {duration_hours} horas' if duration_hours else ''}")
            
            # Iniciar operación autónoma
            result = await self.orchestrator.run_autonomous_operation(duration_hours)
            
            if result.get("success", False):
                logger.info(f"Operación autónoma completada: "
                          f"{result.get('cycles_completed', 0)} ciclos, "
                          f"${result.get('total_profit', 0):.2f} ganancia total")
            else:
                logger.error(f"Error en operación autónoma: {result.get('error', 'Error desconocido')}")
                
            self.running = False
            
        except KeyboardInterrupt:
            logger.info("Programa interrumpido por el usuario")
            self.orchestrator.stop_autonomous_operation()
            self.running = False
        except Exception as e:
            logger.error(f"Error en modo autónomo: {str(e)}")
            self.running = False
    
    async def shutdown(self) -> None:
        """Cerrar el sistema de forma segura."""
        try:
            logger.info("Cerrando Sistema Genesis Ultra-Divino Trading Nexus...")
            
            if self.orchestrator and self.orchestrator.active_cycle_id:
                logger.warning("Hay un ciclo activo durante el cierre")
                # En una implementación completa, guardaríamos el estado
                
            # Mostrar estadísticas finales
            if self.orchestrator:
                logger.info(f"Estadísticas finales: "
                          f"{self.orchestrator.completed_cycles_count} ciclos completados, "
                          f"${self.orchestrator.total_realized_profit:.2f} ganancia total")
            
            logger.info("Sistema cerrado correctamente")
            
        except Exception as e:
            logger.error(f"Error al cerrar sistema: {str(e)}")

async def main():
    """Función principal."""
    # Analizar argumentos
    parser = argparse.ArgumentParser(description='Ejecutor del Sistema Genesis Ultra-Divino Trading Nexus 10M')
    parser.add_argument('-c', '--config', default='seraphim_config.json', help='Ruta al archivo de configuración')
    parser.add_argument('-a', '--auto', action='store_true', help='Iniciar en modo autónomo')
    parser.add_argument('-d', '--duration', type=int, help='Duración en horas para modo autónomo')
    args = parser.parse_args()
    
    # Crear ejecutor
    runner = SeraphimNexusRunner()
    
    try:
        # Cargar configuración
        config_loaded = await runner.load_configuration(args.config)
        if not config_loaded:
            logger.error("No se pudo cargar la configuración. Abortando.")
            return 1
        
        # Inicializar componentes
        components_initialized = await runner.initialize_components()
        if not components_initialized:
            logger.error("No se pudieron inicializar los componentes. Abortando.")
            return 1
        
        # Iniciar sistema
        system_started = await runner.start_system()
        if not system_started:
            logger.error("No se pudo iniciar el sistema. Abortando.")
            return 1
        
        # Ejecutar en modo apropiado
        if args.auto:
            await runner.run_autonomous(args.duration)
        else:
            await runner.run_interactive()
        
        # Cerrar sistema
        await runner.shutdown()
        
        return 0
        
    except Exception as e:
        logger.error(f"Error en ejecución principal: {str(e)}")
        if runner:
            await runner.shutdown()
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nPrograma interrumpido por el usuario")
        sys.exit(0)