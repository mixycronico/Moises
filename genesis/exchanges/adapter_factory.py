"""
Fábrica de adaptadores para exchanges tanto simulados como reales.

Este módulo proporciona una fábrica que selecciona y crea el adaptador
adecuado según la configuración, permitiendo elegir fácilmente entre
simulador, exchange real y Binance Testnet.
"""

import logging
import json
from typing import Dict, Any, Optional, Union

from genesis.core.transcendental_exchange_integrator import TranscendentalExchangeIntegrator
from genesis.exchanges.simulated_exchange_adapter import SimulatedExchangeAdapter
from genesis.exchanges.binance_testnet_adapter import BinanceTestnetAdapter

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Genesis.AdapterFactory")

class AdapterType:
    """Tipos de adaptadores disponibles."""
    REAL = "real"              # Exchange real mediante API
    SIMULATED = "simulated"    # Exchange simulado localmente
    BINANCE_TESTNET = "binance_testnet"  # Adaptador específico para Binance Testnet

class ExchangeAdapterFactory:
    """Fábrica para crear adaptadores de exchange."""
    
    @staticmethod
    async def create_adapter(
        exchange_id: str,
        adapter_type: str = AdapterType.REAL,
        config: Optional[Dict[str, Any]] = None
    ) -> Union[TranscendentalExchangeIntegrator, SimulatedExchangeAdapter, BinanceTestnetAdapter]:
        """
        Crear un adaptador para un exchange específico.
        
        Args:
            exchange_id: Identificador del exchange
            adapter_type: Tipo de adaptador (real, simulado o binance_testnet)
            config: Configuración opcional
            
        Returns:
            Instancia del adaptador creado
            
        Raises:
            ValueError: Si se especifica un tipo de adaptador desconocido
        """
        logger.info(f"Creando adaptador {adapter_type} para exchange {exchange_id}")
        
        config = config or {}
        
        if adapter_type.lower() == AdapterType.REAL:
            # Crear adaptador para exchange real
            adapter = TranscendentalExchangeIntegrator()
            await adapter.initialize()
            await adapter.add_exchange(exchange_id)
            
            logger.info(f"Adaptador real creado para {exchange_id}")
            return adapter
            
        elif adapter_type.lower() == AdapterType.SIMULATED:
            # Crear adaptador para exchange simulado
            adapter = SimulatedExchangeAdapter(exchange_id, config)
            await adapter.initialize()
            
            logger.info(f"Adaptador simulado creado para {exchange_id}")
            return adapter
            
        elif adapter_type.lower() == AdapterType.BINANCE_TESTNET:
            # Crear adaptador específico para Binance Testnet
            adapter = BinanceTestnetAdapter()
            await adapter.initialize()
            
            logger.info(f"Adaptador Binance Testnet creado con éxito")
            return adapter
            
        else:
            error_msg = f"Tipo de adaptador desconocido: {adapter_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
    @staticmethod
    def get_adapter_config_from_json(config_path: str) -> Dict[str, Any]:
        """
        Cargar configuración de adaptador desde archivo JSON.
        
        Args:
            config_path: Ruta al archivo JSON
            
        Returns:
            Dict con configuración cargada
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            return config
            
        except Exception as e:
            logger.error(f"Error cargando configuración desde {config_path}: {e}")
            return {}