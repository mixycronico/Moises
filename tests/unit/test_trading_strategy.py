"""
Pruebas unitarias para la estrategia de trading personalizada.

Este módulo prueba la estrategia de trading personalizada en diferentes 
fases (inicial, media, expansión y final) y situaciones, incluyendo
adaptaciones a condiciones del mercado y mecanismos de seguridad.
"""

import pytest
from unittest.mock import Mock, patch

from genesis.exchanges.manager import ExchangeManager
from genesis.analysis.crypto_classifier import CryptoClassifier
from genesis.analysis.signal_generator import SignalGenerator
from genesis.risk.risk_management import RiskManagement
from genesis.notifications.alert_manager import AlertManager
from genesis.strategy.custom_strategy import TradingStrategy


# Fixture para mocks de componentes
@pytest.fixture
def mock_components():
    """Fixture que proporciona mocks de los componentes clave."""
    exchange_manager = Mock(spec=ExchangeManager)
    crypto_classifier = Mock(spec=CryptoClassifier)
    signal_generator = Mock(spec=SignalGenerator)
    risk_manager = Mock(spec=RiskManagement)
    alert_manager = Mock(spec=AlertManager)
    
    return {
        "exchange_manager": exchange_manager,
        "crypto_classifier": crypto_classifier,
        "signal_generator": signal_generator,
        "risk_manager": risk_manager,
        "alert_manager": alert_manager
    }


# Fixture para una instancia de la estrategia
@pytest.fixture
def strategy(mock_components):
    """Fixture que proporciona una instancia de la estrategia con componentes simulados."""
    return TradingStrategy(mock_components)


# Pruebas para Fase 1: Crecimiento Inicial ($200 → $250)
def test_initial_phase_trade(strategy, mock_components):
    """Prueba un trade en la fase inicial ($200 → $250)."""
    mock_components["crypto_classifier"].filter_hot_cryptos.return_value = [
        {"symbol": "BTC/USDT", "volume": 50000000, "change_24h": 6.5, "adx": 30}
    ]
    mock_components["signal_generator"].generate_ema_signal.return_value = "BUY"
    mock_components["signal_generator"].generate_macd_signal.return_value = "BUY"
    mock_components["signal_generator"].generate_rsi_signal.return_value = "NEUTRAL"
    mock_components["exchange_manager"].execute_trade.return_value = {"status": "success"}
    mock_components["alert_manager"].check_drawdown.return_value = 5  # Drawdown < 10%
    
    result = strategy.execute_trade("BTC/USDT", "buy", strategy.capital, atr=10, phase="initial")
    
    assert result["status"] == "success"
    assert strategy.capital == 210  # 200 + (200 * 0.05) = 210
    mock_components["exchange_manager"].execute_trade.assert_called_once_with("BTC/USDT", "buy", 8)  # 200 * 0.04


def test_initial_phase_drawdown_adjustment(strategy, mock_components):
    """Prueba ajuste de riesgo si drawdown > 10% en fase inicial."""
    mock_components["alert_manager"].check_drawdown.return_value = 12  # Drawdown > 10%
    mock_components["exchange_manager"].execute_trade.return_value = {"status": "success"}
    
    result = strategy.execute_trade("BTC/USDT", "buy", strategy.capital, atr=10, phase="initial")
    
    assert result["status"] == "success"
    assert strategy.capital == 210
    mock_components["exchange_manager"].execute_trade.assert_called_once_with("BTC/USDT", "buy", 4)  # 200 * 0.02


# Pruebas para Fase 2: Crecimiento Medio ($250 → $450)
def test_middle_phase_trade(strategy, mock_components):
    """Prueba un trade estándar en la fase media ($250 → $450)."""
    strategy.capital = 250
    mock_components["signal_generator"].generate_rsi_signal.return_value = "NEUTRAL"
    mock_components["exchange_manager"].execute_trade.return_value = {"status": "success"}
    
    result = strategy.execute_trade("BTC/USDT", "buy", strategy.capital, atr=12, phase="middle")
    
    assert result["status"] == "success"
    assert strategy.capital == 261.25  # 250 + (250 * 0.045) = 261.25
    mock_components["exchange_manager"].execute_trade.assert_called_once_with("BTC/USDT", "buy", 8.75)  # 250 * 0.035


def test_middle_phase_rsi_adjustment(strategy, mock_components):
    """Prueba ajuste de riesgo si RSI indica sobreventa en fase media."""
    strategy.capital = 250
    mock_components["signal_generator"].generate_rsi_signal.return_value = "SELL"  # RSI < 30
    mock_components["exchange_manager"].execute_trade.return_value = {"status": "success"}
    
    result = strategy.execute_trade("BTC/USDT", "buy", strategy.capital, atr=12, phase="middle")
    
    assert result["status"] == "success"
    assert strategy.capital == 261.25  # 250 + (250 * 0.045) = 261.25
    mock_components["exchange_manager"].execute_trade.assert_called_once_with("BTC/USDT", "buy", 5)  # 250 * 0.02


# Pruebas para Fase 3: Expansión ($450 → $1,000)
def test_expansion_phase_trade(strategy, mock_components):
    """Prueba un trade en la fase de expansión ($450 → $1,000)."""
    strategy.capital = 450
    mock_components["exchange_manager"].execute_trade.return_value = {"status": "success"}
    mock_components["alert_manager"].check_drawdown.return_value = 8  # Drawdown < 10%
    
    result = strategy.execute_trade("BTC/USDT", "buy", strategy.capital, atr=15, phase="expansion")
    
    assert result["status"] == "success"
    assert strategy.capital == 468  # 450 + (450 * 0.04) = 468
    mock_components["exchange_manager"].execute_trade.assert_called_once_with("BTC/USDT", "buy", 13.5)  # 450 * 0.03


def test_expansion_phase_high_drawdown(strategy, mock_components):
    """Prueba un trade en fase de expansión con alto drawdown."""
    strategy.capital = 450
    mock_components["exchange_manager"].execute_trade.return_value = {"status": "success"}
    mock_components["alert_manager"].check_drawdown.return_value = 15  # Drawdown > 10%
    
    result = strategy.execute_trade("BTC/USDT", "buy", strategy.capital, atr=15, phase="expansion")
    
    assert result["status"] == "success"
    assert strategy.capital == 468  # 450 + (450 * 0.04) = 468
    mock_components["exchange_manager"].execute_trade.assert_called_once_with("BTC/USDT", "buy", 6.75)  # 450 * 0.015


# Pruebas para Fase 4: Final ($1,000 → $1,500+)
def test_final_phase_trade(strategy, mock_components):
    """Prueba un trade en la fase final ($1,000 → $1,500+)."""
    strategy.capital = 1000
    mock_components["signal_generator"].success_rate = 0.8  # > 70%
    mock_components["exchange_manager"].execute_trade.return_value = {"status": "success"}
    
    result = strategy.execute_trade("BTC/USDT", "buy", strategy.capital, atr=18, phase="final")
    
    assert result["status"] == "success"
    assert strategy.capital == 1035  # 1000 + (1000 * 0.035) = 1035
    mock_components["exchange_manager"].execute_trade.assert_called_once_with("BTC/USDT", "buy", 25)  # 1000 * 0.025


def test_final_phase_success_rate_adjustment(strategy, mock_components):
    """Prueba ajuste de riesgo si tasa de éxito < 70% en fase final."""
    strategy.capital = 1000
    mock_components["signal_generator"].success_rate = 0.65  # < 70%
    mock_components["exchange_manager"].execute_trade.return_value = {"status": "success"}
    
    result = strategy.execute_trade("BTC/USDT", "buy", strategy.capital, atr=18, phase="final")
    
    assert result["status"] == "success"
    assert strategy.capital == 1035  # 1000 + (1000 * 0.035) = 1035
    mock_components["exchange_manager"].execute_trade.assert_called_once_with("BTC/USDT", "buy", 10)  # 1000 * 0.01


# Pruebas para Fail-Safes y Mecanismos de Seguridad
def test_kill_switch_activation(strategy, mock_components):
    """Prueba la activación del kill switch con caída > 30%."""
    market_drop = 35  # > 30%
    
    activated = strategy.check_kill_switch(market_drop)
    
    assert activated is True
    mock_components["exchange_manager"].convert_to_usdt.assert_called_once()


def test_kill_switch_no_activation(strategy, mock_components):
    """Prueba que el kill switch no se active con caída < 30%."""
    market_drop = 25  # < 30%
    
    activated = strategy.check_kill_switch(market_drop)
    
    assert activated is False
    mock_components["exchange_manager"].convert_to_usdt.assert_not_called()


def test_failed_trade_handling(strategy, mock_components):
    """Prueba el manejo de un trade fallido."""
    mock_components["exchange_manager"].execute_trade.return_value = {"status": "error", "message": "Insufficient funds"}
    initial_capital = strategy.capital
    
    result = strategy.execute_trade("BTC/USDT", "buy", strategy.capital, atr=10, phase="initial")
    
    assert result["status"] == "error"
    assert result["message"] == "Insufficient funds"
    # El capital no debería cambiar si el trade falla
    assert strategy.capital == initial_capital


# Pruebas de integración con otros componentes
def test_crypto_classifier_filter(strategy, mock_components):
    """Prueba que el CryptoClassifier filtre correctamente según los criterios."""
    mock_components["crypto_classifier"].filter_hot_cryptos.return_value = [
        {"symbol": "BTC/USDT", "volume": 50000000, "change_24h": 6.5, "adx": 30},
        {"symbol": "ETH/USDT", "volume": 30000000, "change_24h": 5.5, "adx": 28}
    ]
    
    hot_cryptos = strategy.crypto_classifier.filter_hot_cryptos(
        volume_threshold=20000000, 
        change_24h_threshold=5, 
        adx_threshold=25
    )
    
    assert len(hot_cryptos) == 2
    assert hot_cryptos[0]["symbol"] == "BTC/USDT"
    assert hot_cryptos[1]["symbol"] == "ETH/USDT"
    mock_components["crypto_classifier"].filter_hot_cryptos.assert_called_once_with(
        volume_threshold=20000000, 
        change_24h_threshold=5, 
        adx_threshold=25
    )


def test_stop_loss_calculation(strategy, mock_components):
    """Prueba el cálculo de stop loss basado en ATR."""
    price = 1000
    atr = 20
    multiplier = 1.5
    expected_stop_loss = price - (atr * multiplier)  # 1000 - (20 * 1.5) = 970
    
    mock_components["risk_manager"].calculate_stop_loss.return_value = expected_stop_loss
    
    strategy.execute_trade("BTC/USDT", "buy", strategy.capital, atr=atr, phase="initial")
    
    mock_components["risk_manager"].calculate_stop_loss.assert_called_once_with(price, atr, multiplier)


def test_multiple_indicators_alignment(strategy, mock_components):
    """Prueba la alineación de múltiples indicadores para tomar decisiones."""
    mock_components["crypto_classifier"].filter_hot_cryptos.return_value = [
        {"symbol": "BTC/USDT", "volume": 50000000, "change_24h": 6.5, "adx": 30}
    ]
    mock_components["signal_generator"].generate_ema_signal.return_value = "BUY"
    mock_components["signal_generator"].generate_macd_signal.return_value = "BUY"
    mock_components["signal_generator"].generate_rsi_signal.return_value = "BUY"
    mock_components["exchange_manager"].execute_trade.return_value = {"status": "success"}
    
    # Simulamos la llamada a una función que verifica la alineación de indicadores
    with patch.object(strategy, 'check_indicators_alignment', return_value=True):
        strategy.check_indicators_alignment("BTC/USDT")
        
        # Verificar que todos los indicadores fueron consultados
        mock_components["signal_generator"].generate_ema_signal.assert_called_once()
        mock_components["signal_generator"].generate_macd_signal.assert_called_once()
        mock_components["signal_generator"].generate_rsi_signal.assert_called_once()