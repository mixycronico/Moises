"""
Tests avanzados para el RiskManager.

Este módulo prueba funcionalidades avanzadas del RiskManager,
incluyendo resiliencia en condiciones extremas de mercado, 
adaptabilidad dinámica de parámetros, gestión avanzada del riesgo
por correlación de activos, y simulación de eventos catastróficos.
"""

import pytest
import asyncio
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from copy import deepcopy

from genesis.risk.risk_manager import RiskManager
from genesis.risk.position_sizer import PositionSizer
from genesis.risk.stop_loss import StopLossCalculator
from genesis.risk.correlation_manager import CorrelationManager
from genesis.risk.risk_metrics import RiskMetricsCalculator
from genesis.risk.drawdown_monitor import DrawdownMonitor
from genesis.core.event_bus import EventBus
from genesis.core.config import Settings


class MarketScenarioSimulator:
    """Simulador de escenarios de mercado para pruebas de estrés."""
    
    def __init__(self, base_volatility=0.02, extreme_factor=5.0):
        """
        Inicializar simulador de escenarios.
        
        Args:
            base_volatility: Volatilidad base normalizada (diaria)
            extreme_factor: Factor multiplicador para escenarios extremos
        """
        self.base_volatility = base_volatility
        self.extreme_factor = extreme_factor
        
        # Precios base para diferentes activos
        self.base_prices = {
            "BTC/USDT": 45000.0,
            "ETH/USDT": 3000.0,
            "XRP/USDT": 0.5,
            "LTC/USDT": 150.0,
            "ADA/USDT": 1.2,
            "DOT/USDT": 30.0,
            "SOL/USDT": 100.0,
            "LINK/USDT": 25.0,
            "AAVE/USDT": 300.0,
            "UNI/USDT": 20.0
        }
        
        # Matriz de correlación entre activos (simplificada)
        self.correlations = {
            ("BTC/USDT", "ETH/USDT"): 0.8,
            ("BTC/USDT", "LTC/USDT"): 0.7,
            ("BTC/USDT", "XRP/USDT"): 0.5,
            ("ETH/USDT", "LINK/USDT"): 0.6,
            ("ETH/USDT", "AAVE/USDT"): 0.7,
            ("SOL/USDT", "ADA/USDT"): 0.6,
            ("DOT/USDT", "KSM/USDT"): 0.9,
        }
        
        # Inicializar generador aleatorio
        np.random.seed(int(time.time()))
    
    def get_correlation(self, symbol1, symbol2):
        """Obtener correlación entre dos símbolos."""
        if symbol1 == symbol2:
            return 1.0
            
        # Buscar correlación directa
        key = (symbol1, symbol2)
        if key in self.correlations:
            return self.correlations[key]
            
        # Buscar correlación invertida
        key_inv = (symbol2, symbol1)
        if key_inv in self.correlations:
            return self.correlations[key_inv]
            
        # Valor predeterminado para pares sin correlación definida
        return 0.4  # Correlación moderada por defecto
    
    def generate_normal_market_data(self, days=30, symbols=None):
        """
        Generar datos de mercado en condiciones normales.
        
        Args:
            days: Número de días a simular
            symbols: Lista de símbolos a generar (por defecto, todos)
            
        Returns:
            DataFrame con datos de mercado
        """
        return self._generate_market_data(days, symbols, is_extreme=False)
    
    def generate_extreme_market_data(self, days=30, symbols=None, crash_day=15):
        """
        Generar datos de mercado en condiciones extremas (e.g., crash).
        
        Args:
            days: Número de días a simular
            symbols: Lista de símbolos a generar (por defecto, todos)
            crash_day: Día en que ocurre el evento extremo (desde 0)
            
        Returns:
            DataFrame con datos de mercado
        """
        return self._generate_market_data(days, symbols, is_extreme=True, crash_day=crash_day)
    
    def _generate_market_data(self, days, symbols=None, is_extreme=False, crash_day=None):
        """
        Implementación del generador de datos de mercado.
        
        Args:
            days: Número de días a simular
            symbols: Lista de símbolos a generar
            is_extreme: Si es True, simula condiciones extremas
            crash_day: Día en que ocurre el evento extremo
            
        Returns:
            DataFrame con datos de mercado
        """
        if symbols is None:
            symbols = list(self.base_prices.keys())
        
        dates = [datetime.now() - timedelta(days=days-i) for i in range(days)]
        
        # Crear matriz de movimientos correlacionados
        price_changes = {}
        
        # Generar cambios porcentuales correlacionados para todos los activos
        # Para cada día
        for i in range(days):
            is_crash_day = is_extreme and crash_day is not None and i == crash_day
            
            # Generar un cambio común (factor de mercado)
            market_change = np.random.normal(0, self.base_volatility)
            
            # Si es día de crash, aplicar un gran movimiento bajista
            if is_crash_day:
                market_change = -self.base_volatility * self.extreme_factor
            
            # Para cada símbolo
            for symbol in symbols:
                if symbol not in price_changes:
                    price_changes[symbol] = []
                
                # Generar cambio específico del activo
                symbol_volatility = self.base_volatility * (1.5 if "BTC" in symbol else 1.0)
                symbol_change = np.random.normal(0, symbol_volatility)
                
                # Combinar cambio de mercado y cambio específico (correlación)
                combined_change = market_change * 0.7 + symbol_change * 0.3
                
                # Si es condición extrema pero no crash, aumentar volatilidad
                if is_extreme and not is_crash_day:
                    # 20% de probabilidad de movimientos extremos
                    if random.random() < 0.2:
                        combined_change *= random.uniform(1.5, self.extreme_factor)
                
                price_changes[symbol].append(combined_change)
        
        # Convertir cambios a precios absolutos
        prices = {}
        for symbol in symbols:
            # Empezar con el precio base
            base_price = self.base_prices.get(symbol, 100.0)
            
            # Calcular precios diarios basados en cambios porcentuales
            symbol_prices = [base_price]
            for change in price_changes[symbol]:
                new_price = symbol_prices[-1] * (1 + change)
                symbol_prices.append(new_price)
            
            # Eliminar el primer precio (usado solo como base)
            prices[symbol] = symbol_prices[1:]
        
        # Crear DataFrame con todos los datos
        market_data = []
        
        for i in range(days):
            date = dates[i]
            for symbol in symbols:
                market_data.append({
                    'date': date,
                    'symbol': symbol,
                    'price': prices[symbol][i],
                    'volume': random.uniform(1000, 10000) * (5 if is_extreme else 1),
                    'is_extreme': is_extreme
                })
        
        return pd.DataFrame(market_data)


class AdvancedRiskManager(RiskManager):
    """Versión avanzada del RiskManager para pruebas con funcionalidades adicionales."""
    
    def __init__(self, event_bus=None, correlation_manager=None, drawdown_monitor=None):
        """Inicializar el gestor de riesgos avanzado."""
        super().__init__(event_bus=event_bus)
        
        # Componentes adicionales para gestión avanzada de riesgos
        self._correlation_manager = correlation_manager or CorrelationManager()
        self._drawdown_monitor = drawdown_monitor or DrawdownMonitor()
        
        # Parámetros avanzados
        self._max_portfolio_var = 0.02  # 2% Value at Risk máximo
        self._max_exposure_per_sector = 0.25  # 25% máximo por sector
        self._sector_mappings = {
            "BTC/USDT": "store_of_value",
            "ETH/USDT": "smart_contract",
            "SOL/USDT": "smart_contract",
            "ADA/USDT": "smart_contract",
            "XRP/USDT": "payment",
            "LINK/USDT": "oracle",
            "AAVE/USDT": "defi",
            "UNI/USDT": "defi"
        }
        
        # Estado avanzado
        self._portfolio_exposure = {}
        self._sector_exposure = {}
        self._correlation_matrix = {}
        self._is_high_volatility_mode = False
        self._drawdown_protection_active = False
    
    async def start(self):
        """Iniciar el gestor de riesgos avanzado."""
        await super().start()
        
        # Inicializar componentes adicionales
        if hasattr(self._correlation_manager, 'start'):
            await self._correlation_manager.start()
            
        if hasattr(self._drawdown_monitor, 'start'):
            await self._drawdown_monitor.start()
            
        # Inicializar exposiciones
        self._portfolio_exposure = {}
        self._sector_exposure = {sector: 0.0 for sector in set(self._sector_mappings.values())}
    
    async def stop(self):
        """Detener el gestor de riesgos avanzado."""
        await super().stop()
        
        # Detener componentes adicionales
        if hasattr(self._correlation_manager, 'stop'):
            await self._correlation_manager.stop()
            
        if hasattr(self._drawdown_monitor, 'stop'):
            await self._drawdown_monitor.stop()
    
    async def handle_market_volatility_change(self, data):
        """Manejar cambios en la volatilidad de mercado."""
        self._is_high_volatility_mode = data.get('is_high_volatility', False)
        
        # Ajustar parámetros de riesgo según volatilidad
        if self._is_high_volatility_mode:
            # Reducir exposición en alta volatilidad
            self._position_sizer.set_risk_percentage(1.0)  # 1% por operación
            self._max_portfolio_var = 0.015  # 1.5% VaR
        else:
            # Configuración normal
            self._position_sizer.set_risk_percentage(2.0)  # 2% por operación
            self._max_portfolio_var = 0.02  # 2% VaR
        
        # Emitir evento de ajuste de riesgo
        await self._event_bus.emit(
            "risk.parameters_adjusted",
            {
                "high_volatility_mode": self._is_high_volatility_mode,
                "risk_percentage": self._position_sizer.get_risk_percentage(),
                "max_portfolio_var": self._max_portfolio_var
            },
            source="risk_manager"
        )
    
    async def handle_significant_drawdown(self, data):
        """Manejar un drawdown significativo en el portfolio."""
        drawdown_pct = data.get('drawdown_percentage', 0.0)
        
        if drawdown_pct > 0.15:  # 15% de drawdown
            # Activar protección extrema
            self._drawdown_protection_active = True
            
            # Reducir significativamente la exposición
            self._position_sizer.set_risk_percentage(0.5)  # 0.5% por operación
            
            # Emitir evento de protección contra drawdown
            await self._event_bus.emit(
                "risk.drawdown_protection_activated",
                {
                    "drawdown_percentage": drawdown_pct,
                    "risk_percentage": self._position_sizer.get_risk_percentage()
                },
                source="risk_manager"
            )
        elif drawdown_pct > 0.10:  # 10% de drawdown
            # Activar protección moderada
            self._drawdown_protection_active = True
            
            # Reducir moderadamente la exposición
            self._position_sizer.set_risk_percentage(1.0)  # 1% por operación
            
            # Emitir evento de protección contra drawdown
            await self._event_bus.emit(
                "risk.drawdown_protection_activated",
                {
                    "drawdown_percentage": drawdown_pct,
                    "risk_percentage": self._position_sizer.get_risk_percentage()
                },
                source="risk_manager"
            )
        elif self._drawdown_protection_active and drawdown_pct < 0.05:  # 5% de drawdown
            # Desactivar protección
            self._drawdown_protection_active = False
            
            # Restaurar configuración normal
            self._position_sizer.set_risk_percentage(2.0)  # 2% por operación
            
            # Emitir evento de desactivación de protección
            await self._event_bus.emit(
                "risk.drawdown_protection_deactivated",
                {
                    "drawdown_percentage": drawdown_pct,
                    "risk_percentage": self._position_sizer.get_risk_percentage()
                },
                source="risk_manager"
            )
    
    async def calculate_adjusted_position_size(self, symbol, base_size, price):
        """
        Calcular tamaño de posición ajustado teniendo en cuenta
        correlaciones y exposición sectorial.
        """
        # Obtener sector del símbolo
        sector = self._sector_mappings.get(symbol, "other")
        
        # Calcular ajuste por correlación
        correlation_adjustment = 1.0
        
        # Si ya tenemos posiciones, ajustar según correlación
        if self._portfolio_exposure:
            total_correlation = 0.0
            for existing_symbol, exposure in self._portfolio_exposure.items():
                # Obtener correlación entre símbolos
                correlation = self._correlation_matrix.get(
                    (symbol, existing_symbol),
                    self._correlation_matrix.get((existing_symbol, symbol), 0.5)
                )
                
                # Sumar correlación ponderada por exposición
                total_correlation += correlation * exposure
            
            # Convertir correlación total a un factor de ajuste
            # Alta correlación = menor tamaño (para diversificar)
            correlation_adjustment = 1.0 - (total_correlation * 0.5)
            correlation_adjustment = max(0.2, min(1.0, correlation_adjustment))
        
        # Calcular ajuste por sector
        sector_exposure = self._sector_exposure.get(sector, 0.0)
        sector_adjustment = 1.0
        
        if sector_exposure > self._max_exposure_per_sector * 0.8:
            # Si estamos cerca del límite sectorial, reducir tamaño
            sector_adjustment = 0.5
        elif sector_exposure > self._max_exposure_per_sector * 0.5:
            # Si estamos a medio camino, reducir un poco
            sector_adjustment = 0.8
        
        # Aplicar ajustes
        adjusted_size = base_size * correlation_adjustment * sector_adjustment
        
        # Reducir tamaño en alta volatilidad o drawdown
        if self._is_high_volatility_mode:
            adjusted_size *= 0.7
            
        if self._drawdown_protection_active:
            adjusted_size *= 0.5
        
        return adjusted_size
    
    async def update_portfolio_exposure(self, symbol, amount, price):
        """Actualizar la exposición del portfolio para un símbolo."""
        position_value = amount * price
        total_portfolio = sum([
            self._portfolio_exposure.get(sym, 0) 
            for sym in self._portfolio_exposure
        ])
        
        # Añadir nuevo valor de posición
        total_portfolio += position_value
        
        # Actualizar exposiciones
        self._portfolio_exposure[symbol] = position_value
        
        # Actualizar exposición sectorial
        sector = self._sector_mappings.get(symbol, "other")
        sector_value = sum([
            self._portfolio_exposure.get(sym, 0) 
            for sym, sec in self._sector_mappings.items() 
            if sec == sector
        ])
        
        self._sector_exposure[sector] = sector_value / total_portfolio if total_portfolio > 0 else 0
        
        return {
            "symbol": symbol,
            "exposure": position_value / total_portfolio if total_portfolio > 0 else 0,
            "sector": sector,
            "sector_exposure": self._sector_exposure[sector]
        }


@pytest.fixture
def market_simulator():
    """Proporciona un simulador de escenarios de mercado para pruebas."""
    return MarketScenarioSimulator()


@pytest.fixture
def event_bus():
    """Proporciona un bus de eventos para pruebas."""
    bus = Mock(spec=EventBus)
    bus.emit = AsyncMock()
    bus.subscribe = AsyncMock()
    bus.unsubscribe = AsyncMock()
    return bus


@pytest.fixture
def correlation_manager():
    """Proporciona un gestor de correlaciones simulado."""
    manager = Mock(spec=CorrelationManager)
    manager.start = AsyncMock()
    manager.stop = AsyncMock()
    manager.get_correlation = Mock(return_value=0.5)
    manager.update_correlation_matrix = AsyncMock()
    
    return manager


@pytest.fixture
def drawdown_monitor():
    """Proporciona un monitor de drawdown simulado."""
    monitor = Mock(spec=DrawdownMonitor)
    monitor.start = AsyncMock()
    monitor.stop = AsyncMock()
    monitor.check_drawdown = Mock(return_value=0.05)  # 5% drawdown por defecto
    
    return monitor


@pytest.fixture
def risk_metrics_calculator():
    """Proporciona un calculador de métricas de riesgo simulado."""
    calculator = Mock(spec=RiskMetricsCalculator)
    calculator.calculate_var = Mock(return_value=0.015)  # 1.5% VaR por defecto
    calculator.calculate_expected_shortfall = Mock(return_value=0.025)  # 2.5% ES por defecto
    calculator.calculate_sharpe_ratio = Mock(return_value=1.8)  # Sharpe 1.8 por defecto
    
    return calculator


@pytest.fixture
def advanced_risk_manager(event_bus, correlation_manager, drawdown_monitor):
    """Proporciona un gestor de riesgos avanzado para pruebas."""
    manager = AdvancedRiskManager(
        event_bus=event_bus,
        correlation_manager=correlation_manager,
        drawdown_monitor=drawdown_monitor
    )
    
    # Inicializar correlaciones para pruebas
    manager._correlation_matrix = {
        ("BTC/USDT", "ETH/USDT"): 0.8,
        ("BTC/USDT", "LTC/USDT"): 0.7,
        ("ETH/USDT", "AAVE/USDT"): 0.6,
        ("SOL/USDT", "ADA/USDT"): 0.7,
    }
    
    return manager


@pytest.mark.asyncio
async def test_risk_manager_volatility_adaptation(advanced_risk_manager, event_bus):
    """Prueba la adaptación del gestor de riesgos a cambios en volatilidad de mercado."""
    # Iniciar el gestor
    await advanced_risk_manager.start()
    
    # Verificar valores iniciales
    initial_risk = advanced_risk_manager._position_sizer.get_risk_percentage()
    initial_var = advanced_risk_manager._max_portfolio_var
    
    # Simular aumento de volatilidad
    high_volatility_data = {
        "is_high_volatility": True,
        "market_volatility": 0.03  # 3% volatilidad
    }
    
    # Manejar cambio de volatilidad
    await advanced_risk_manager.handle_market_volatility_change(high_volatility_data)
    
    # Verificar que se hayan ajustado los parámetros
    high_vol_risk = advanced_risk_manager._position_sizer.get_risk_percentage()
    high_vol_var = advanced_risk_manager._max_portfolio_var
    
    assert high_vol_risk < initial_risk, "El riesgo no se redujo en alta volatilidad"
    assert high_vol_var < initial_var, "El VaR no se redujo en alta volatilidad"
    
    # Verificar que se emitió evento de ajuste
    event_bus.emit.assert_called_with(
        "risk.parameters_adjusted",
        {
            "high_volatility_mode": True,
            "risk_percentage": high_vol_risk,
            "max_portfolio_var": high_vol_var
        },
        source="risk_manager"
    )
    
    # Simular normalización de volatilidad
    normal_volatility_data = {
        "is_high_volatility": False,
        "market_volatility": 0.015  # 1.5% volatilidad
    }
    
    # Resetear el mock para verificar la segunda llamada
    event_bus.emit.reset_mock()
    
    # Manejar cambio de volatilidad
    await advanced_risk_manager.handle_market_volatility_change(normal_volatility_data)
    
    # Verificar que se hayan restaurado los parámetros
    normal_vol_risk = advanced_risk_manager._position_sizer.get_risk_percentage()
    normal_vol_var = advanced_risk_manager._max_portfolio_var
    
    assert normal_vol_risk > high_vol_risk, "El riesgo no se aumentó en volatilidad normal"
    assert normal_vol_var > high_vol_var, "El VaR no se aumentó en volatilidad normal"
    
    # Verificar que se emitió evento de ajuste
    event_bus.emit.assert_called_with(
        "risk.parameters_adjusted",
        {
            "high_volatility_mode": False,
            "risk_percentage": normal_vol_risk,
            "max_portfolio_var": normal_vol_var
        },
        source="risk_manager"
    )


@pytest.mark.asyncio
async def test_risk_manager_drawdown_protection(advanced_risk_manager, event_bus):
    """Prueba la protección contra drawdowns significativos."""
    # Iniciar el gestor
    await advanced_risk_manager.start()
    
    # Verificar valores iniciales
    initial_risk = advanced_risk_manager._position_sizer.get_risk_percentage()
    assert not advanced_risk_manager._drawdown_protection_active, "La protección no debe estar activa al inicio"
    
    # Simular un drawdown moderado (10%)
    moderate_drawdown_data = {
        "drawdown_percentage": 0.12,  # 12% drawdown
        "current_equity": 88000,
        "peak_equity": 100000
    }
    
    # Manejar drawdown
    await advanced_risk_manager.handle_significant_drawdown(moderate_drawdown_data)
    
    # Verificar que se haya activado la protección moderada
    assert advanced_risk_manager._drawdown_protection_active, "La protección no se activó con drawdown moderado"
    moderate_risk = advanced_risk_manager._position_sizer.get_risk_percentage()
    assert moderate_risk < initial_risk, "El riesgo no se redujo con drawdown moderado"
    
    # Verificar que se emitió evento de protección
    event_bus.emit.assert_called_with(
        "risk.drawdown_protection_activated",
        {
            "drawdown_percentage": 0.12,
            "risk_percentage": moderate_risk
        },
        source="risk_manager"
    )
    
    # Simular un drawdown severo (18%)
    event_bus.emit.reset_mock()
    severe_drawdown_data = {
        "drawdown_percentage": 0.18,  # 18% drawdown
        "current_equity": 82000,
        "peak_equity": 100000
    }
    
    # Manejar drawdown severo
    await advanced_risk_manager.handle_significant_drawdown(severe_drawdown_data)
    
    # Verificar que se haya activado la protección extrema
    assert advanced_risk_manager._drawdown_protection_active, "La protección no se activó con drawdown severo"
    severe_risk = advanced_risk_manager._position_sizer.get_risk_percentage()
    assert severe_risk < moderate_risk, "El riesgo no se redujo aún más con drawdown severo"
    
    # Verificar que se emitió evento de protección
    event_bus.emit.assert_called_with(
        "risk.drawdown_protection_activated",
        {
            "drawdown_percentage": 0.18,
            "risk_percentage": severe_risk
        },
        source="risk_manager"
    )
    
    # Simular una recuperación (drawdown reducido a 3%)
    event_bus.emit.reset_mock()
    recovery_data = {
        "drawdown_percentage": 0.03,  # 3% drawdown
        "current_equity": 97000,
        "peak_equity": 100000
    }
    
    # Manejar recuperación
    await advanced_risk_manager.handle_significant_drawdown(recovery_data)
    
    # Verificar que se haya desactivado la protección
    assert not advanced_risk_manager._drawdown_protection_active, "La protección no se desactivó tras la recuperación"
    recovery_risk = advanced_risk_manager._position_sizer.get_risk_percentage()
    assert recovery_risk > severe_risk, "El riesgo no se aumentó tras la recuperación"
    
    # Verificar que se emitió evento de desactivación
    event_bus.emit.assert_called_with(
        "risk.drawdown_protection_deactivated",
        {
            "drawdown_percentage": 0.03,
            "risk_percentage": recovery_risk
        },
        source="risk_manager"
    )


@pytest.mark.asyncio
async def test_position_size_correlation_adjustment(advanced_risk_manager):
    """Prueba el ajuste de tamaño de posición basado en correlaciones entre activos."""
    # Iniciar el gestor
    await advanced_risk_manager.start()
    
    # Escenario 1: Portfolio vacío, sin ajuste por correlación
    # Establecer un portfolio vacío
    advanced_risk_manager._portfolio_exposure = {}
    
    # Calcular tamaño con portfolio vacío
    base_size = 1.0  # 1 BTC
    symbol = "BTC/USDT"
    price = 45000
    
    size1 = await advanced_risk_manager.calculate_adjusted_position_size(symbol, base_size, price)
    
    # No debería haber ajuste por correlación
    assert size1 == base_size, "Hubo ajuste de correlación con portfolio vacío"
    
    # Escenario 2: Portfolio con activos correlacionados
    # Añadir exposición a ETH (altamente correlacionado con BTC)
    advanced_risk_manager._portfolio_exposure = {
        "ETH/USDT": 100000  # $100,000 en ETH
    }
    
    # Calcular tamaño
    size2 = await advanced_risk_manager.calculate_adjusted_position_size(symbol, base_size, price)
    
    # Debería haber reducción por correlación
    assert size2 < base_size, "No hubo reducción por correlación con ETH"
    
    # Escenario 3: Portfolio diversificado
    # Añadir exposiciones diversificadas
    advanced_risk_manager._portfolio_exposure = {
        "SOL/USDT": 50000,  # $50,000 en SOL
        "XRP/USDT": 25000   # $25,000 en XRP
    }
    
    # Ajustar correlaciones para este test
    advanced_risk_manager._correlation_matrix = {
        ("BTC/USDT", "SOL/USDT"): 0.3,  # Baja correlación
        ("BTC/USDT", "XRP/USDT"): 0.2   # Muy baja correlación
    }
    
    # Calcular tamaño
    size3 = await advanced_risk_manager.calculate_adjusted_position_size(symbol, base_size, price)
    
    # Debería haber menor reducción por baja correlación
    assert size3 > size2, "No hubo menor reducción con portfolio diversificado"
    assert size3 < base_size, "No hubo ninguna reducción con portfolio diversificado"


@pytest.mark.asyncio
async def test_position_size_sector_exposure_adjustment(advanced_risk_manager):
    """Prueba el ajuste de tamaño de posición basado en exposición sectorial."""
    # Iniciar el gestor
    await advanced_risk_manager.start()
    
    # Establecer un portfolio vacío
    advanced_risk_manager._portfolio_exposure = {}
    advanced_risk_manager._sector_exposure = {
        "store_of_value": 0.0,
        "smart_contract": 0.0,
        "payment": 0.0,
        "oracle": 0.0,
        "defi": 0.0
    }
    
    # Base size y precio
    base_size = 1.0  # 1 ETH
    price = 3000     # $3000 por ETH
    
    # Escenario 1: Sin exposición previa al sector smart_contract
    symbol1 = "ETH/USDT"  # smart_contract
    
    size1 = await advanced_risk_manager.calculate_adjusted_position_size(symbol1, base_size, price)
    
    # No debería haber ajuste por sector
    assert size1 == base_size, "Hubo ajuste sectorial sin exposición previa"
    
    # Escenario 2: Alta exposición al sector smart_contract
    # Simular alta exposición a smart_contract (20%)
    advanced_risk_manager._sector_exposure = {
        "store_of_value": 0.1,      # 10%
        "smart_contract": 0.2,      # 20%
        "payment": 0.05,            # 5%
        "oracle": 0.05,             # 5%
        "defi": 0.1                 # 10%
    }
    
    size2 = await advanced_risk_manager.calculate_adjusted_position_size(symbol1, base_size, price)
    
    # Debería haber reducción por exposición sectorial
    assert size2 < base_size, "No hubo reducción por alta exposición sectorial"
    
    # Escenario 3: Exposición extrema al sector smart_contract
    # Simular exposición extrema a smart_contract (24%, casi en el límite de 25%)
    advanced_risk_manager._sector_exposure = {
        "store_of_value": 0.1,      # 10%
        "smart_contract": 0.24,     # 24%
        "payment": 0.05,            # 5%
        "oracle": 0.05,             # 5%
        "defi": 0.1                 # 10%
    }
    
    size3 = await advanced_risk_manager.calculate_adjusted_position_size(symbol1, base_size, price)
    
    # Debería haber reducción significativa por exposición sectorial extrema
    assert size3 < size2, "No hubo reducción adicional por exposición sectorial extrema"
    
    # Escenario 4: Comprobación con otro sector (store_of_value)
    # Mantenemos las mismas exposiciones sectoriales
    symbol4 = "BTC/USDT"  # store_of_value
    
    size4 = await advanced_risk_manager.calculate_adjusted_position_size(symbol4, base_size, price)
    
    # No debería haber reducción significativa por sector
    assert size4 > size3, "Hubo reducción excesiva para un sector diferente"


@pytest.mark.asyncio
async def test_portfolio_exposure_update(advanced_risk_manager):
    """Prueba la actualización de exposición de portfolio y sectorial."""
    # Iniciar el gestor
    await advanced_risk_manager.start()
    
    # Establecer un portfolio vacío
    advanced_risk_manager._portfolio_exposure = {}
    advanced_risk_manager._sector_exposure = {sector: 0.0 for sector in set(advanced_risk_manager._sector_mappings.values())}
    
    # Añadir la primera posición: 2 BTC a $45,000
    symbol1 = "BTC/USDT"
    amount1 = 2.0
    price1 = 45000
    
    result1 = await advanced_risk_manager.update_portfolio_exposure(symbol1, amount1, price1)
    
    # Calcular valores esperados
    position_value1 = amount1 * price1  # $90,000
    
    # Verificar resultado
    assert result1["symbol"] == symbol1
    assert result1["exposure"] == 1.0  # 100% de exposición
    assert result1["sector"] == "store_of_value"
    assert result1["sector_exposure"] == 1.0  # 100% del sector
    
    # Verificar estado interno del portfolio
    assert advanced_risk_manager._portfolio_exposure[symbol1] == position_value1
    assert advanced_risk_manager._sector_exposure["store_of_value"] == 1.0
    
    # Añadir una segunda posición: 30 ETH a $3,000
    symbol2 = "ETH/USDT"
    amount2 = 30.0
    price2 = 3000
    
    result2 = await advanced_risk_manager.update_portfolio_exposure(symbol2, amount2, price2)
    
    # Calcular valores esperados
    position_value2 = amount2 * price2  # $90,000
    total_value = position_value1 + position_value2  # $180,000
    expected_exposure_btc = position_value1 / total_value  # 0.5
    expected_exposure_eth = position_value2 / total_value  # 0.5
    
    # Verificar resultado
    assert result2["symbol"] == symbol2
    assert abs(result2["exposure"] - expected_exposure_eth) < 1e-6
    assert result2["sector"] == "smart_contract"
    assert abs(result2["sector_exposure"] - 0.5) < 1e-6  # 50% del sector
    
    # Verificar estado interno del portfolio
    assert advanced_risk_manager._portfolio_exposure[symbol1] == position_value1
    assert advanced_risk_manager._portfolio_exposure[symbol2] == position_value2
    assert abs(advanced_risk_manager._sector_exposure["store_of_value"] - 0.5) < 1e-6
    assert abs(advanced_risk_manager._sector_exposure["smart_contract"] - 0.5) < 1e-6
    
    # Añadir una tercera posición del mismo sector: 100 SOL a $100
    symbol3 = "SOL/USDT"
    amount3 = 100.0
    price3 = 100
    
    result3 = await advanced_risk_manager.update_portfolio_exposure(symbol3, amount3, price3)
    
    # Calcular valores esperados
    position_value3 = amount3 * price3  # $10,000
    total_value = position_value1 + position_value2 + position_value3  # $190,000
    expected_exposure_btc = position_value1 / total_value  # ~0.4737
    expected_exposure_eth = position_value2 / total_value  # ~0.4737
    expected_exposure_sol = position_value3 / total_value  # ~0.0526
    expected_store_of_value = position_value1 / total_value  # ~0.4737
    expected_smart_contract = (position_value2 + position_value3) / total_value  # ~0.5263
    
    # Verificar resultado
    assert result3["symbol"] == symbol3
    assert abs(result3["exposure"] - expected_exposure_sol) < 1e-6
    assert result3["sector"] == "smart_contract"
    assert abs(result3["sector_exposure"] - expected_smart_contract) < 1e-6
    
    # Verificar estado interno del portfolio
    assert abs(advanced_risk_manager._sector_exposure["store_of_value"] - expected_store_of_value) < 1e-6
    assert abs(advanced_risk_manager._sector_exposure["smart_contract"] - expected_smart_contract) < 1e-6


@pytest.mark.asyncio
async def test_risk_manager_stress_test_market_crash(advanced_risk_manager, market_simulator, event_bus):
    """
    Prueba el comportamiento del gestor de riesgos durante un crash de mercado.
    
    Este test simula un escenario de crash de mercado y verifica que el 
    sistema de gestión de riesgos responda adecuadamente ajustando parámetros
    y protegiendo el capital.
    """
    # Iniciar el gestor
    await advanced_risk_manager.start()
    
    # Generar escenario de crash
    crash_data = market_simulator.generate_extreme_market_data(
        days=30,
        symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        crash_day=15
    )
    
    # Registrar valores iniciales
    initial_risk = advanced_risk_manager._position_sizer.get_risk_percentage()
    
    # Procesar datos día a día
    dates = sorted(crash_data["date"].unique())
    
    for date in dates:
        # Filtrar datos para este día
        day_data = crash_data[crash_data["date"] == date]
        
        # Verificar volatilidad
        price_changes = []
        for symbol in day_data["symbol"].unique():
            symbol_data = day_data[day_data["symbol"] == symbol]
            price = symbol_data["price"].values[0]
            
            # Añadir precios al historial para calcular volatilidad
            if not hasattr(advanced_risk_manager, "_price_history"):
                advanced_risk_manager._price_history = {}
            
            if symbol not in advanced_risk_manager._price_history:
                advanced_risk_manager._price_history[symbol] = []
            
            advanced_risk_manager._price_history[symbol].append(price)
            
            # Calcular cambio porcentual si tenemos suficientes datos
            if len(advanced_risk_manager._price_history[symbol]) > 1:
                prev_price = advanced_risk_manager._price_history[symbol][-2]
                change = (price - prev_price) / prev_price
                price_changes.append(change)
        
        # Si tenemos suficientes datos, calcular volatilidad
        if price_changes:
            volatility = np.std(price_changes)
            high_volatility = volatility > 0.02  # Umbral de 2%
            
            # Manejar cambio de volatilidad
            await advanced_risk_manager.handle_market_volatility_change({
                "is_high_volatility": high_volatility,
                "market_volatility": volatility
            })
        
        # Simular cálculo de drawdown basado en precios de BTC
        btc_data = day_data[day_data["symbol"] == "BTC/USDT"]
        if len(btc_data) > 0 and hasattr(advanced_risk_manager, "_price_history"):
            btc_prices = advanced_risk_manager._price_history.get("BTC/USDT", [])
            
            if btc_prices:
                peak_price = max(btc_prices)
                current_price = btc_prices[-1]
                drawdown = (peak_price - current_price) / peak_price if peak_price > 0 else 0
                
                # Manejar drawdown
                await advanced_risk_manager.handle_significant_drawdown({
                    "drawdown_percentage": drawdown,
                    "peak_equity": peak_price,
                    "current_equity": current_price
                })
    
    # Verificar que los parámetros de riesgo se hayan ajustado
    final_risk = advanced_risk_manager._position_sizer.get_risk_percentage()
    
    # En un crash, esperamos que el riesgo se haya reducido
    assert final_risk < initial_risk, "El riesgo no se redujo después del crash"
    
    # Verificar que se activó la protección contra drawdown
    event_bus.emit.assert_any_call(
        "risk.drawdown_protection_activated",
        {
            "drawdown_percentage": pytest.approx(0.12, abs=0.1),  # Aproximadamente 12% drawdown +/- 10%
            "risk_percentage": pytest.approx(1.0, abs=0.5)        # Aproximadamente 1% de riesgo +/- 0.5%
        },
        source="risk_manager"
    )
    
    # Verificar que en algún momento se detectó alta volatilidad
    event_bus.emit.assert_any_call(
        "risk.parameters_adjusted",
        {
            "high_volatility_mode": True,
            "risk_percentage": pytest.approx(1.0, abs=0.5),
            "max_portfolio_var": pytest.approx(0.015, abs=0.005)
        },
        source="risk_manager"
    )


@pytest.mark.asyncio
async def test_risk_manager_recovery_after_crash(advanced_risk_manager, market_simulator, event_bus):
    """
    Prueba el comportamiento del gestor de riesgos durante la recuperación post-crash.
    
    Este test simula un escenario de crash seguido de recuperación y verifica
    que el sistema de gestión de riesgos reajuste los parámetros apropiadamente.
    """
    # Iniciar el gestor
    await advanced_risk_manager.start()
    
    # Generar escenario de crash y recuperación
    # (crash en el día 10, recuperación desde el día 20)
    market_data = market_simulator.generate_extreme_market_data(
        days=30,
        symbols=["BTC/USDT", "ETH/USDT"],
        crash_day=10
    )
    
    # Modificar los datos para simular una recuperación
    dates = sorted(market_data["date"].unique())
    recovery_start = dates[20]  # Día 20
    
    # Crear un punto de recuperación
    for i, date in enumerate(dates[20:], 20):
        # Filtrar datos para este día
        day_data = market_data[market_data["date"] == date]
        
        for idx in day_data.index:
            # Aumentar precios gradualmente para simular recuperación
            recovery_factor = 1.0 + ((i - 20) * 0.01)  # +1% por día desde día 20
            market_data.at[idx, "price"] *= recovery_factor
    
    # Registrar el gestor de riesgos para seguir su comportamiento
    risk_states = []
    
    # Procesar datos día a día
    for date in dates:
        # Filtrar datos para este día
        day_data = market_data[market_data["date"] == date]
        
        # Verificar volatilidad
        price_changes = []
        prices_by_symbol = {}
        
        for symbol in day_data["symbol"].unique():
            symbol_data = day_data[day_data["symbol"] == symbol]
            price = symbol_data["price"].values[0]
            prices_by_symbol[symbol] = price
            
            # Añadir precios al historial para calcular volatilidad
            if not hasattr(advanced_risk_manager, "_price_history"):
                advanced_risk_manager._price_history = {}
            
            if symbol not in advanced_risk_manager._price_history:
                advanced_risk_manager._price_history[symbol] = []
            
            advanced_risk_manager._price_history[symbol].append(price)
            
            # Calcular cambio porcentual si tenemos suficientes datos
            if len(advanced_risk_manager._price_history[symbol]) > 1:
                prev_price = advanced_risk_manager._price_history[symbol][-2]
                change = (price - prev_price) / prev_price
                price_changes.append(change)
        
        # Si tenemos suficientes datos, calcular volatilidad
        if price_changes:
            volatility = np.std(price_changes)
            high_volatility = volatility > 0.02  # Umbral de 2%
            
            # Manejar cambio de volatilidad
            await advanced_risk_manager.handle_market_volatility_change({
                "is_high_volatility": high_volatility,
                "market_volatility": volatility
            })
        
        # Simular cálculo de drawdown basado en precios de BTC
        if "BTC/USDT" in prices_by_symbol and hasattr(advanced_risk_manager, "_price_history"):
            btc_prices = advanced_risk_manager._price_history.get("BTC/USDT", [])
            
            if btc_prices:
                peak_price = max(btc_prices)
                current_price = btc_prices[-1]
                drawdown = (peak_price - current_price) / peak_price if peak_price > 0 else 0
                
                # Manejar drawdown
                await advanced_risk_manager.handle_significant_drawdown({
                    "drawdown_percentage": drawdown,
                    "peak_equity": peak_price,
                    "current_equity": current_price
                })
        
        # Guardar estado del gestor de riesgos
        current_state = {
            "date": date,
            "volatility_mode": advanced_risk_manager._is_high_volatility_mode,
            "drawdown_protection": advanced_risk_manager._drawdown_protection_active,
            "risk_percentage": advanced_risk_manager._position_sizer.get_risk_percentage(),
            "prices": prices_by_symbol.copy()
        }
        risk_states.append(current_state)
    
    # Verificar el comportamiento durante y después del crash
    crash_state = None
    recovery_state = None
    final_state = None
    
    for state in risk_states:
        # Encontrar el estado durante el crash (día ~10-15)
        if crash_state is None and state["date"] >= dates[15]:
            crash_state = state
        
        # Encontrar el estado durante la recuperación (día ~25)
        if recovery_state is None and state["date"] >= dates[25]:
            recovery_state = state
            
        # Guardar el estado final
        if state["date"] == dates[-1]:
            final_state = state
    
    # Durante el crash, debería haber protección y parámetros conservadores
    assert crash_state["volatility_mode"], "No se activó el modo de alta volatilidad durante el crash"
    assert crash_state["drawdown_protection"], "No se activó la protección contra drawdown durante el crash"
    assert crash_state["risk_percentage"] < 2.0, "El riesgo no se redujo durante el crash"
    
    # Verificar que en la recuperación se han ajustado los parámetros
    # Nota: puede que la protección aún esté activa si el drawdown no bajó lo suficiente
    assert final_state["risk_percentage"] > crash_state["risk_percentage"], \
        "El riesgo no se aumentó después de la recuperación"
    
    # Verificar que en algún momento se emitió evento de desactivación 
    # de protección contra drawdown
    found_deactivation = False
    for call in event_bus.emit.call_args_list:
        if call[0][0] == "risk.drawdown_protection_deactivated":
            found_deactivation = True
            break
            
    assert found_deactivation, "No se emitió evento de desactivación de protección durante la recuperación"


@pytest.mark.asyncio
async def test_risk_manager_concurrent_operations(advanced_risk_manager):
    """Prueba el manejo de múltiples operaciones concurrentes en el gestor de riesgos."""
    # Iniciar el gestor
    await advanced_risk_manager.start()
    
    # Preparar múltiples símbolos para prueba concurrente
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT"]
    prices = [45000, 3000, 100, 1.2, 0.5]
    
    # Crear tareas concurrentes para actualizar exposición
    tasks = []
    
    for symbol, price in zip(symbols, prices):
        amount = random.uniform(0.1, 1.0)
        tasks.append(advanced_risk_manager.update_portfolio_exposure(symbol, amount, price))
    
    # Ejecutar concurrentemente
    results = await asyncio.gather(*tasks)
    
    # Verificar que todos los símbolos se actualizaron
    for symbol, result in zip(symbols, results):
        assert result["symbol"] == symbol, f"Resultado no coincide para símbolo {symbol}"
        assert result["exposure"] > 0, f"Exposición no calculada para {symbol}"
        
    # Verificar que la suma de exposiciones sectoriales no supera el 100%
    total_sector_exposure = sum(advanced_risk_manager._sector_exposure.values())
    assert abs(total_sector_exposure - 1.0) < 1e-6, \
        f"La suma de exposiciones sectoriales no es 100%: {total_sector_exposure}"
    
    # Crear tareas concurrentes para calcular tamaños de posición
    size_tasks = []
    
    for symbol, price in zip(symbols, prices):
        base_size = 1.0
        size_tasks.append(advanced_risk_manager.calculate_adjusted_position_size(symbol, base_size, price))
    
    # Ejecutar concurrentemente
    size_results = await asyncio.gather(*size_tasks)
    
    # Verificar que todos los cálculos se completaron
    for size, symbol in zip(size_results, symbols):
        assert size > 0, f"Tamaño no calculado para {symbol}"


@pytest.mark.asyncio
async def test_risk_manager_with_incomplete_data(advanced_risk_manager, event_bus):
    """Prueba el manejo de datos incompletos o malformados en el gestor de riesgos."""
    # Iniciar el gestor
    await advanced_risk_manager.start()
    
    # Caso 1: Datos de volatilidad incompletos
    incomplete_volatility = {
        # Falta is_high_volatility
        "market_volatility": 0.03
    }
    
    # Esto no debería fallar
    await advanced_risk_manager.handle_market_volatility_change(incomplete_volatility)
    
    # Verificar que se usó un valor predeterminado y no hubo error
    assert not advanced_risk_manager._is_high_volatility_mode, "Se activó incorrectamente el modo de alta volatilidad"
    
    # Caso 2: Datos de drawdown incompletos
    incomplete_drawdown = {
        "drawdown_percentage": None,  # Valor None
        "current_equity": 90000
        # Falta peak_equity
    }
    
    # Esto no debería fallar
    await advanced_risk_manager.handle_significant_drawdown(incomplete_drawdown)
    
    # Verificar que no se activó la protección por datos incompletos
    assert not advanced_risk_manager._drawdown_protection_active, "Se activó incorrectamente la protección contra drawdown"
    
    # Caso 3: Actualización de portfolio con símbolo desconocido
    unknown_symbol = "UNKNOWN/USDT"
    
    # Esto debería manejar el símbolo desconocido sin sector definido
    result = await advanced_risk_manager.update_portfolio_exposure(unknown_symbol, 1.0, 100)
    
    # Verificar que se asignó al sector "other"
    assert result["sector"] == "other", "No se asignó el sector 'other' al símbolo desconocido"
    
    # Caso 4: Calcular tamaño de posición con símbolo desconocido
    size = await advanced_risk_manager.calculate_adjusted_position_size(unknown_symbol, 1.0, 100)
    
    # Verificar que se devolvió un valor razonable a pesar del símbolo desconocido
    assert size > 0, "No se calculó tamaño para símbolo desconocido"


@pytest.mark.asyncio
async def test_risk_manager_performance_stress_test(advanced_risk_manager, market_simulator):
    """Prueba de rendimiento del gestor de riesgos bajo alta carga."""
    # Iniciar el gestor
    await advanced_risk_manager.start()
    
    # Generar un conjunto grande de datos
    large_data = market_simulator.generate_normal_market_data(
        days=30,
        symbols=list(advanced_risk_manager._sector_mappings.keys())  # Todos los símbolos conocidos
    )
    
    # Medir tiempo de cálculo de tamaños para múltiples símbolos
    symbols = list(advanced_risk_manager._sector_mappings.keys())
    num_calculations = 100
    
    start_time = time.time()
    
    # Ejecutar muchos cálculos de tamaño de posición
    tasks = []
    for _ in range(num_calculations):
        symbol = random.choice(symbols)
        base_size = random.uniform(0.1, 2.0)
        price = random.uniform(100, 50000)
        
        tasks.append(advanced_risk_manager.calculate_adjusted_position_size(symbol, base_size, price))
    
    # Ejecutar concurrentemente
    await asyncio.gather(*tasks)
    
    end_time = time.time()
    
    # Calcular throughput
    time_taken = end_time - start_time
    calculations_per_second = num_calculations / time_taken
    
    # El throughput debe ser razonable (al menos 10 cálculos por segundo)
    assert calculations_per_second > 10, f"Rendimiento insuficiente: {calculations_per_second:.2f} cálculos/s"
    
    # Medir tiempo de actualización de portfolio para múltiples símbolos
    num_updates = 50
    
    start_time = time.time()
    
    # Ejecutar muchas actualizaciones de portfolio
    update_tasks = []
    for _ in range(num_updates):
        symbol = random.choice(symbols)
        amount = random.uniform(0.1, 10.0)
        price = random.uniform(100, 50000)
        
        update_tasks.append(advanced_risk_manager.update_portfolio_exposure(symbol, amount, price))
    
    # Ejecutar concurrentemente
    await asyncio.gather(*update_tasks)
    
    end_time = time.time()
    
    # Calcular throughput
    time_taken = end_time - start_time
    updates_per_second = num_updates / time_taken
    
    # El throughput debe ser razonable (al menos 10 actualizaciones por segundo)
    assert updates_per_second > 10, f"Rendimiento insuficiente: {updates_per_second:.2f} actualizaciones/s"


@pytest.mark.asyncio
async def test_risk_manager_zero_division_protection(advanced_risk_manager):
    """Prueba que el gestor de riesgos maneja correctamente casos que podrían causar división por cero."""
    # Iniciar el gestor
    await advanced_risk_manager.start()
    
    # Caso 1: División por cero en cálculo de exposición
    # Esto podría ocurrir si tenemos posiciones con valor cero
    advanced_risk_manager._portfolio_exposure = {
        "BTC/USDT": 0,
        "ETH/USDT": 0
    }
    
    # Actualizar exposición - esto no debería fallar
    result = await advanced_risk_manager.update_portfolio_exposure("BTC/USDT", 0, 45000)
    
    # Verificar que se manejó correctamente
    assert result["exposure"] == 0, "No se manejó correctamente el caso de valor de portfolio cero"
    
    # Caso 2: Cálculo de tamaño con precio cero
    symbol = "BTC/USDT"
    base_size = 1.0
    zero_price = 0
    
    # Esto no debería fallar (aunque podría devolver cero o un valor predeterminado)
    size = await advanced_risk_manager.calculate_adjusted_position_size(symbol, base_size, zero_price)
    
    # Verificar que manejó el caso sin error
    assert size is not None, "Falló al manejar precio cero"
"""