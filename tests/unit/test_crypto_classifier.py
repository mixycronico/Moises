"""
Pruebas unitarias para el clasificador de criptomonedas.

Este módulo prueba las funcionalidades del clasificador de criptomonedas,
que se encarga de filtrar criptomonedas basándose en varios indicadores
como volumen, variación de precio y fuerza de tendencia (ADX).
"""

import pytest
from unittest.mock import Mock

from genesis.analysis.crypto_classifier import CryptoClassifier


# Fixture para datos simulados
@pytest.fixture
def mock_market_data():
    """Fixture que proporciona datos simulados de criptomonedas."""
    return [
        {"symbol": "BTC/USDT", "volume": 50000000, "change_24h": 6.5, "adx": 30},
        {"symbol": "ETH/USDT", "volume": 30000000, "change_24h": 7.0, "adx": 28},
        {"symbol": "DOGE/USDT", "volume": 10000000, "change_24h": 2.0, "adx": 15},
        {"symbol": "ADA/USDT", "volume": 20000000, "change_24h": 5.5, "adx": 27},
        {"symbol": "XRP/USDT", "volume": 25000000, "change_24h": 4.5, "adx": 20},
    ]


@pytest.fixture
def classifier(mock_market_data):
    """Fixture que proporciona una instancia de CryptoClassifier con datos simulados."""
    classifier = CryptoClassifier()
    classifier.get_market_data = Mock(return_value=mock_market_data)
    return classifier


# Pruebas principales
def test_crypto_classifier_filters(classifier, mock_market_data):
    """Prueba que los filtros de volumen, cambio 24h y ADX seleccionen las criptos adecuadas."""
    # Filtros: volumen >= 20M, cambio 24h >= 5%, ADX >= 25
    hot_cryptos = classifier.filter_hot_cryptos(
        volume_threshold=20000000,
        change_24h_threshold=5,
        adx_threshold=25
    )

    # Verificamos el resultado
    expected = [
        {"symbol": "BTC/USDT", "volume": 50000000, "change_24h": 6.5, "adx": 30},
        {"symbol": "ETH/USDT", "volume": 30000000, "change_24h": 7.0, "adx": 28},
        {"symbol": "ADA/USDT", "volume": 20000000, "change_24h": 5.5, "adx": 27},
    ]
    assert len(hot_cryptos) == 3
    assert hot_cryptos == expected  # Comparación exacta de la lista
    classifier.get_market_data.assert_called_once()


def test_crypto_classifier_strict_filters(classifier):
    """Prueba filtros más estrictos donde pocas criptos deberían pasar."""
    # Filtros estrictos: volumen >= 40M, cambio 24h >= 6%, ADX >= 29
    hot_cryptos = classifier.filter_hot_cryptos(
        volume_threshold=40000000,
        change_24h_threshold=6,
        adx_threshold=29
    )

    # Solo BTC debería pasar
    expected = [{"symbol": "BTC/USDT", "volume": 50000000, "change_24h": 6.5, "adx": 30}]
    assert len(hot_cryptos) == 1
    assert hot_cryptos == expected


def test_crypto_classifier_no_results(classifier):
    """Prueba cuando ninguna cripto cumple los criterios."""
    # Filtros imposibles: volumen >= 100M, cambio 24h >= 10%, ADX >= 50
    hot_cryptos = classifier.filter_hot_cryptos(
        volume_threshold=100000000,
        change_24h_threshold=10,
        adx_threshold=50
    )

    assert len(hot_cryptos) == 0
    assert hot_cryptos == []


def test_crypto_classifier_empty_data(classifier):
    """Prueba el comportamiento con datos vacíos."""
    classifier.get_market_data = Mock(return_value=[])
    hot_cryptos = classifier.filter_hot_cryptos(
        volume_threshold=20000000,
        change_24h_threshold=5,
        adx_threshold=25
    )

    assert len(hot_cryptos) == 0
    assert hot_cryptos == []


def test_crypto_classifier_missing_fields(classifier):
    """Prueba el manejo de datos con campos faltantes."""
    mock_data_with_missing = [
        {"symbol": "BTC/USDT", "volume": 50000000, "change_24h": 6.5},  # Sin ADX
        {"symbol": "ETH/USDT", "volume": 30000000, "adx": 28},          # Sin change_24h
        {"symbol": "DOGE/USDT", "change_24h": 2.0, "adx": 15},          # Sin volume
    ]
    classifier.get_market_data = Mock(return_value=mock_data_with_missing)

    # Suponiendo que filter_hot_cryptos ignora o maneja registros incompletos
    hot_cryptos = classifier.filter_hot_cryptos(
        volume_threshold=20000000,
        change_24h_threshold=5,
        adx_threshold=25
    )

    assert len(hot_cryptos) == 0  # Ninguno cumple todos los criterios completos


def test_crypto_classifier_invalid_thresholds(classifier):
    """Prueba el manejo de umbrales inválidos."""
    with pytest.raises(ValueError, match="Thresholds must be non-negative"):
        classifier.filter_hot_cryptos(
            volume_threshold=-20000000,
            change_24h_threshold=5,
            adx_threshold=25
        )

    with pytest.raises(ValueError, match="Thresholds must be non-negative"):
        classifier.filter_hot_cryptos(
            volume_threshold=20000000,
            change_24h_threshold=-5,
            adx_threshold=25
        )

    with pytest.raises(ValueError, match="Thresholds must be non-negative"):
        classifier.filter_hot_cryptos(
            volume_threshold=20000000,
            change_24h_threshold=5,
            adx_threshold=-25
        )


def test_crypto_classifier_edge_case_thresholds(classifier):
    """Prueba umbrales en los límites (exactamente igual a los valores)."""
    hot_cryptos = classifier.filter_hot_cryptos(
        volume_threshold=50000000,  # Exactamente el volumen de BTC
        change_24h_threshold=6.5,  # Exactamente el cambio de BTC
        adx_threshold=30           # Exactamente el ADX de BTC
    )

    expected = [{"symbol": "BTC/USDT", "volume": 50000000, "change_24h": 6.5, "adx": 30}]
    assert len(hot_cryptos) == 1
    assert hot_cryptos == expected


def test_crypto_classifier_get_market_data_called_once(classifier):
    """Prueba que get_market_data se llame solo una vez por ejecución."""
    classifier.get_market_data.reset_mock()  # Reseteamos el conteo de llamadas
    classifier.filter_hot_cryptos(
        volume_threshold=20000000,
        change_24h_threshold=5,
        adx_threshold=25
    )
    classifier.get_market_data.assert_called_once()


def test_crypto_classifier_multiple_filters(classifier):
    """Prueba aplicar múltiples filtros secuencialmente."""
    # Primer filtro: volumen >= 20M
    volume_filter = classifier.filter_by_volume(20000000)
    assert len(volume_filter) == 4  # BTC, ETH, ADA, XRP
    
    # Segundo filtro: cambio 24h >= 5%
    change_filter = classifier.filter_by_change(volume_filter, 5)
    assert len(change_filter) == 3  # BTC, ETH, ADA
    
    # Tercer filtro: ADX >= 25
    adx_filter = classifier.filter_by_adx(change_filter, 25)
    assert len(adx_filter) == 3  # BTC, ETH, ADA todos tienen ADX >= 25


def test_crypto_classifier_returns_copy(classifier):
    """Prueba que el clasificador devuelve copias de los datos, no referencias."""
    hot_cryptos = classifier.filter_hot_cryptos(
        volume_threshold=20000000,
        change_24h_threshold=5,
        adx_threshold=25
    )
    
    # Modificar los datos devueltos no debería afectar los datos originales
    hot_cryptos[0]["volume"] = 999
    
    # Volver a solicitar los datos y comprobar que son los originales
    new_hot_cryptos = classifier.filter_hot_cryptos(
        volume_threshold=20000000,
        change_24h_threshold=5,
        adx_threshold=25
    )
    
    assert new_hot_cryptos[0]["volume"] == 50000000  # Valor original, no 999


def test_crypto_classifier_sorting(classifier):
    """Prueba ordenar criptomonedas por diferentes criterios."""
    # Obtener todas las criptos 
    all_cryptos = classifier.get_market_data()
    
    # Ordenar por volumen (descendente)
    volume_sorted = classifier.sort_by_volume(all_cryptos)
    assert volume_sorted[0]["symbol"] == "BTC/USDT"  # Mayor volumen
    assert volume_sorted[-1]["symbol"] == "DOGE/USDT"  # Menor volumen
    
    # Ordenar por cambio 24h (descendente)
    change_sorted = classifier.sort_by_change(all_cryptos)
    assert change_sorted[0]["symbol"] == "ETH/USDT"  # Mayor cambio
    assert change_sorted[-1]["symbol"] == "DOGE/USDT"  # Menor cambio
    
    # Ordenar por ADX (descendente)
    adx_sorted = classifier.sort_by_adx(all_cryptos)
    assert adx_sorted[0]["symbol"] == "BTC/USDT"  # Mayor ADX
    assert adx_sorted[-1]["symbol"] == "DOGE/USDT"  # Menor ADX