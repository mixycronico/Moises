"""
API REST para el sistema Genesis.

Este módulo proporciona una API REST para integración con sistemas externos,
permitiendo el acceso programático a los datos y funcionalidades del sistema.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import logging
import uuid
import asyncio
from functools import wraps
from flask import request, jsonify, Blueprint, current_app, abort
from werkzeug.security import check_password_hash

from genesis.core.base import Component
from genesis.utils.logger import setup_logging
from genesis.security.manager import SecurityUtils


# Crear el Blueprint para la API REST
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')
logger = setup_logging('api_rest')


# Decorador para autenticación con API key
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Obtener API key y firma del request
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({
                'success': False,
                'error': 'API key requerida'
            }), 401
        
        # Aquí se verificaría la API key y firma con el SecurityManager
        # Esta implementación es simplificada, en producción deberíamos usar el EventBus
        # para comunicarnos con el SecurityManager
        
        # Verificar si la API key es válida (implementación de ejemplo)
        # En un sistema real, esto debería verificarse contra el SecurityManager
        valid_api_key = True  # Placeholder
        
        if not valid_api_key:
            return jsonify({
                'success': False,
                'error': 'API key inválida'
            }), 403
        
        return f(*args, **kwargs)
    return decorated_function


# Decorador para autenticación con token JWT
def require_auth_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Obtener token de autorización
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({
                'success': False,
                'error': 'Token de autorización requerido'
            }), 401
        
        # Extraer token
        token = auth_header.split(' ')[1]
        
        # Aquí se verificaría el token con el SecurityManager
        # Esta implementación es simplificada
        
        # Verificar si el token es válido (implementación de ejemplo)
        valid_token = True  # Placeholder
        
        if not valid_token:
            return jsonify({
                'success': False,
                'error': 'Token inválido o expirado'
            }), 403
        
        return f(*args, **kwargs)
    return decorated_function


# Rutas de autenticación
@api_bp.route('/auth/login', methods=['POST'])
def login():
    """Iniciar sesión y obtener token de autorización."""
    data = request.get_json()
    
    # Verificar datos requeridos
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({
            'success': False,
            'error': 'Nombre de usuario y contraseña requeridos'
        }), 400
    
    username = data['username']
    password = data['password']
    
    # Aquí se autenticaría al usuario con el SecurityManager
    # Esta implementación es simplificada
    
    # Autenticación de ejemplo (en un sistema real, esto se haría a través del EventBus)
    if username == 'admin' and password == 'admin':  # SOLO PARA DESARROLLO
        # Generar token
        token = str(uuid.uuid4())
        expiry = datetime.now() + timedelta(days=1)
        
        return jsonify({
            'success': True,
            'token': token,
            'expiry': expiry.isoformat(),
            'user': {
                'username': username,
                'role': 'admin'
            }
        })
    
    return jsonify({
        'success': False,
        'error': 'Credenciales inválidas'
    }), 401


@api_bp.route('/auth/logout', methods=['POST'])
@require_auth_token
def logout():
    """Cerrar sesión e invalidar token."""
    # Aquí se invalidaría el token en el SecurityManager
    
    return jsonify({
        'success': True,
        'message': 'Sesión cerrada correctamente'
    })


@api_bp.route('/auth/refresh', methods=['POST'])
@require_auth_token
def refresh_token():
    """Renovar token de autorización."""
    # Aquí se renovaría el token en el SecurityManager
    
    # Generar nuevo token
    token = str(uuid.uuid4())
    expiry = datetime.now() + timedelta(days=1)
    
    return jsonify({
        'success': True,
        'token': token,
        'expiry': expiry.isoformat()
    })


# Rutas de market data
@api_bp.route('/market/symbols', methods=['GET'])
@require_api_key
def get_symbols():
    """Obtener lista de símbolos disponibles."""
    # Aquí se obtendría la lista de símbolos disponibles
    # Esta implementación es simplificada
    
    symbols = [
        'BTC/USDT',
        'ETH/USDT',
        'XRP/USDT',
        'SOL/USDT'
    ]
    
    return jsonify({
        'success': True,
        'data': {
            'symbols': symbols
        }
    })


@api_bp.route('/market/ticker/<symbol>', methods=['GET'])
@require_api_key
def get_ticker(symbol):
    """Obtener datos de ticker para un símbolo."""
    # Aquí se obtendrían los datos de ticker desde MarketDataManager
    # Esta implementación es simplificada
    
    # Verificar símbolo
    if '/' not in symbol:
        return jsonify({
            'success': False,
            'error': 'Formato de símbolo inválido'
        }), 400
    
    # Datos de ticker de ejemplo
    ticker = {
        'symbol': symbol,
        'last': 50000.0,
        'bid': 49995.0,
        'ask': 50005.0,
        'volume': 1000.0,
        'timestamp': datetime.now().isoformat(),
        'change_24h': 2.5,
        'high_24h': 51000.0,
        'low_24h': 49000.0
    }
    
    return jsonify({
        'success': True,
        'data': ticker
    })


@api_bp.route('/market/candles/<symbol>', methods=['GET'])
@require_api_key
def get_candles(symbol):
    """Obtener datos OHLCV para un símbolo."""
    # Parámetros opcionales
    timeframe = request.args.get('timeframe', '1h')
    limit = min(int(request.args.get('limit', 100)), 1000)  # Limitar a máximo 1000
    
    # Aquí se obtendrían los datos OHLCV desde MarketDataManager
    # Esta implementación es simplificada
    
    # Verificar símbolo
    if '/' not in symbol:
        return jsonify({
            'success': False,
            'error': 'Formato de símbolo inválido'
        }), 400
    
    # Datos OHLCV de ejemplo
    now = datetime.now()
    candles = []
    
    for i in range(limit):
        timestamp = now - timedelta(hours=i)
        candles.append({
            'timestamp': timestamp.isoformat(),
            'open': 50000.0 + (i % 10) * 100,
            'high': 50100.0 + (i % 10) * 100,
            'low': 49900.0 + (i % 10) * 100,
            'close': 50050.0 + (i % 10) * 100,
            'volume': 100.0 + i * 10
        })
    
    # Ordenar cronológicamente (más antiguo primero)
    candles.reverse()
    
    return jsonify({
        'success': True,
        'data': {
            'symbol': symbol,
            'timeframe': timeframe,
            'candles': candles
        }
    })


@api_bp.route('/market/orderbook/<symbol>', methods=['GET'])
@require_api_key
def get_orderbook(symbol):
    """Obtener libro de órdenes para un símbolo."""
    # Parámetro opcional
    depth = min(int(request.args.get('depth', 20)), 100)  # Limitar a máximo 100
    
    # Aquí se obtendría el libro de órdenes desde MarketDataManager
    # Esta implementación es simplificada
    
    # Verificar símbolo
    if '/' not in symbol:
        return jsonify({
            'success': False,
            'error': 'Formato de símbolo inválido'
        }), 400
    
    # Datos de orderbook de ejemplo
    bids = []
    asks = []
    
    base_price = 50000.0
    
    for i in range(depth):
        bid_price = base_price - (i * 10)
        ask_price = base_price + (i * 10)
        
        bids.append([bid_price, 1.0 - (i * 0.01)])
        asks.append([ask_price, 1.0 - (i * 0.01)])
    
    orderbook = {
        'symbol': symbol,
        'bids': bids,
        'asks': asks,
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify({
        'success': True,
        'data': orderbook
    })


# Rutas de backtesting
@api_bp.route('/backtest', methods=['POST'])
@require_auth_token
def run_backtest():
    """Ejecutar un backtest con parámetros específicos."""
    data = request.get_json()
    
    # Verificar datos requeridos
    if not data or 'symbol' not in data or 'strategy' not in data:
        return jsonify({
            'success': False,
            'error': 'Símbolo y estrategia requeridos'
        }), 400
    
    # Extraer parámetros
    symbol = data['symbol']
    strategy = data['strategy']
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    params = data.get('params', {})
    
    # Aquí se ejecutaría el backtest a través del BacktestEngine
    # Esta implementación es simplificada
    
    # ID de backtest único
    backtest_id = str(uuid.uuid4())
    
    # Respuesta con ID del backtest
    return jsonify({
        'success': True,
        'data': {
            'backtest_id': backtest_id,
            'status': 'submitted',
            'params': {
                'symbol': symbol,
                'strategy': strategy,
                'start_date': start_date,
                'end_date': end_date,
                'params': params
            },
            'eta_seconds': 30
        }
    })


@api_bp.route('/backtest/<backtest_id>', methods=['GET'])
@require_auth_token
def get_backtest_result(backtest_id):
    """Obtener resultado de un backtest por ID."""
    # Aquí se obtendría el resultado del backtest desde el BacktestEngine
    # Esta implementación es simplificada
    
    # Datos de ejemplo
    result = {
        'backtest_id': backtest_id,
        'status': 'completed',
        'symbol': 'BTC/USDT',
        'strategy': 'ma_crossover',
        'start_date': '2023-01-01T00:00:00',
        'end_date': '2023-12-31T23:59:59',
        'params': {
            'fast_period': 20,
            'slow_period': 50
        },
        'metrics': {
            'total_trades': 120,
            'win_rate': 0.65,
            'profit_factor': 2.1,
            'max_drawdown': 0.12,
            'net_profit': 0.32,
            'sharpe_ratio': 1.8
        },
        'trades': [
            {
                'timestamp': '2023-01-15T10:30:00',
                'type': 'buy',
                'price': 42000.0,
                'amount': 0.1,
                'pnl': 0.0
            },
            {
                'timestamp': '2023-01-20T14:45:00',
                'type': 'sell',
                'price': 44000.0,
                'amount': 0.1,
                'pnl': 0.048  # (44000 - 42000) / 42000
            }
            # etc.
        ]
    }
    
    return jsonify({
        'success': True,
        'data': result
    })


# Rutas de análisis
@api_bp.route('/analysis/indicators/<symbol>', methods=['GET'])
@require_api_key
def get_indicators(symbol):
    """Obtener indicadores técnicos para un símbolo."""
    # Parámetros
    timeframe = request.args.get('timeframe', '1h')
    indicators = request.args.get('indicators', 'rsi,macd,bb').split(',')
    limit = min(int(request.args.get('limit', 50)), 200)  # Limitar a máximo 200
    
    # Aquí se calcularían los indicadores a través del MarketAnalyzer
    # Esta implementación es simplificada
    
    # Verificar símbolo
    if '/' not in symbol:
        return jsonify({
            'success': False,
            'error': 'Formato de símbolo inválido'
        }), 400
    
    # Datos de ejemplo
    now = datetime.now()
    result = {
        'symbol': symbol,
        'timeframe': timeframe,
        'data': []
    }
    
    for i in range(limit):
        timestamp = now - timedelta(hours=i)
        
        data_point = {
            'timestamp': timestamp.isoformat(),
            'price': 50000.0 + (i % 30) * 100
        }
        
        # Añadir indicadores solicitados
        if 'rsi' in indicators:
            data_point['rsi'] = 50 + (i % 40) - 20
        
        if 'macd' in indicators:
            data_point['macd'] = {
                'line': (i % 20) - 10,
                'signal': (i % 15) - 7.5,
                'histogram': ((i % 20) - 10) - ((i % 15) - 7.5)
            }
        
        if 'bb' in indicators:
            middle = 50000.0 + (i % 30) * 100
            data_point['bollinger_bands'] = {
                'upper': middle + 500,
                'middle': middle,
                'lower': middle - 500
            }
        
        result['data'].append(data_point)
    
    # Ordenar cronológicamente (más antiguo primero)
    result['data'].reverse()
    
    return jsonify({
        'success': True,
        'data': result
    })


@api_bp.route('/analysis/anomalies/<symbol>', methods=['GET'])
@require_api_key
def get_anomalies(symbol):
    """Obtener anomalías detectadas para un símbolo."""
    # Parámetros
    days = min(int(request.args.get('days', 7)), 30)  # Limitar a máximo 30 días
    
    # Aquí se obtendrían las anomalías desde el AnomalyDetector
    # Esta implementación es simplificada
    
    # Verificar símbolo
    if '/' not in symbol:
        return jsonify({
            'success': False,
            'error': 'Formato de símbolo inválido'
        }), 400
    
    # Datos de ejemplo
    now = datetime.now()
    anomalies = []
    
    # Generar algunas anomalías de ejemplo
    anomaly_types = [
        'price_spike', 'volume_spike', 'price_gap',
        'volatility_surge', 'liquidity_change', 'price_volume_divergence'
    ]
    
    for i in range(5):  # 5 anomalías aleatorias
        timestamp = now - timedelta(days=i*2)
        anomaly_type = anomaly_types[i % len(anomaly_types)]
        
        anomalies.append({
            'timestamp': timestamp.isoformat(),
            'type': anomaly_type,
            'score': 0.7 + (i * 0.05),
            'details': {
                'description': f'Detected {anomaly_type} anomaly',
                'threshold': 0.65,
                'value': 0.7 + (i * 0.05),
                'z_score': 2.8 + (i * 0.2)
            }
        })
    
    return jsonify({
        'success': True,
        'data': {
            'symbol': symbol,
            'period_days': days,
            'anomalies': anomalies
        }
    })


# Rutas de gestión de balances
@api_bp.route('/accounting/balance', methods=['GET'])
@require_auth_token
def get_balance():
    """Obtener balance total de la cuenta."""
    # Aquí se obtendría el balance desde el BalanceManager
    # Esta implementación es simplificada
    
    # Datos de ejemplo
    balances = {
        'total_usd': 125000.0,
        'assets': [
            {
                'asset': 'BTC',
                'free': 1.5,
                'locked': 0.1,
                'total': 1.6,
                'price_usd': 50000.0,
                'value_usd': 80000.0
            },
            {
                'asset': 'ETH',
                'free': 15.0,
                'locked': 0.0,
                'total': 15.0,
                'price_usd': 3000.0,
                'value_usd': 45000.0
            },
            {
                'asset': 'USDT',
                'free': 0.0,
                'locked': 0.0,
                'total': 0.0,
                'price_usd': 1.0,
                'value_usd': 0.0
            }
        ],
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify({
        'success': True,
        'data': balances
    })


@api_bp.route('/accounting/transactions', methods=['GET'])
@require_auth_token
def get_transactions():
    """Obtener historial de transacciones."""
    # Parámetros
    asset = request.args.get('asset')
    transaction_type = request.args.get('type')
    limit = min(int(request.args.get('limit', 20)), 100)  # Limitar a máximo 100
    
    # Aquí se obtendrían las transacciones desde el BalanceManager
    # Esta implementación es simplificada
    
    # Datos de ejemplo
    now = datetime.now()
    transactions = []
    
    types = ['deposit', 'withdrawal', 'trade', 'fee', 'transfer']
    assets = ['BTC', 'ETH', 'USDT']
    
    for i in range(limit):
        timestamp = now - timedelta(hours=i*5)
        tx_type = types[i % len(types)]
        tx_asset = assets[i % len(assets)]
        
        # Filtrar si se solicitó un asset específico
        if asset and tx_asset != asset:
            continue
        
        # Filtrar si se solicitó un tipo específico
        if transaction_type and tx_type != transaction_type:
            continue
        
        transactions.append({
            'transaction_id': f'tx_{uuid.uuid4()}',
            'timestamp': timestamp.isoformat(),
            'type': tx_type,
            'asset': tx_asset,
            'amount': 0.1 + (i * 0.02),
            'price': 50000.0 if tx_asset == 'BTC' else (3000.0 if tx_asset == 'ETH' else 1.0),
            'value_usd': (0.1 + (i * 0.02)) * (50000.0 if tx_asset == 'BTC' else (3000.0 if tx_asset == 'ETH' else 1.0)),
            'status': 'completed'
        })
    
    return jsonify({
        'success': True,
        'data': {
            'transactions': transactions,
            'count': len(transactions)
        }
    })


# Rutas para reportes
@api_bp.route('/reports/performance', methods=['GET'])
@require_auth_token
def get_performance_report():
    """Obtener reporte de rendimiento."""
    # Parámetros
    period = request.args.get('period', 'daily')  # daily, weekly, monthly
    
    # Aquí se generaría el reporte desde el ReportGenerator
    # Esta implementación es simplificada
    
    # Datos de ejemplo
    now = datetime.now()
    
    # Determinar puntos de datos según periodo
    if period == 'daily':
        days = 30
        interval = 'day'
    elif period == 'weekly':
        days = 90
        interval = 'week'
    else:  # monthly
        days = 365
        interval = 'month'
    
    # Generar datos
    data_points = []
    
    start_val = 100000.0
    current_val = start_val
    
    for i in range(days):
        if period == 'daily' or (period == 'weekly' and i % 7 == 0) or (period == 'monthly' and i % 30 == 0):
            timestamp = now - timedelta(days=i)
            
            # Fluctuación aleatoria pero con tendencia al alza
            change_pct = 0.005 + (0.01 * (i % 5)) - 0.02
            current_val = current_val * (1 + change_pct)
            
            data_points.append({
                'timestamp': timestamp.isoformat(),
                'portfolio_value': current_val,
                'change_pct': change_pct,
                'benchmark_value': start_val * (1 + (0.0003 * i))
            })
    
    # Ordenar cronológicamente (más antiguo primero)
    data_points.reverse()
    
    # Calcular métricas
    total_return = (current_val - start_val) / start_val
    
    report = {
        'period': period,
        'start_date': data_points[0]['timestamp'],
        'end_date': data_points[-1]['timestamp'],
        'start_value': start_val,
        'end_value': current_val,
        'total_return': total_return,
        'annualized_return': total_return * (365 / days),
        'volatility': 0.12,
        'sharpe_ratio': 1.8,
        'max_drawdown': 0.09,
        'data': data_points
    }
    
    return jsonify({
        'success': True,
        'data': report
    })


@api_bp.route('/reports/strategies', methods=['GET'])
@require_auth_token
def get_strategies_report():
    """Obtener reporte de rendimiento por estrategia."""
    # Aquí se generaría el reporte desde el ReportGenerator
    # Esta implementación es simplificada
    
    # Datos de ejemplo
    strategies = [
        {
            'name': 'MA Crossover',
            'id': 'ma_crossover',
            'metrics': {
                'trades': 85,
                'win_rate': 0.68,
                'profit_factor': 2.3,
                'total_return': 0.18,
                'max_drawdown': 0.08
            }
        },
        {
            'name': 'RSI Strategy',
            'id': 'rsi',
            'metrics': {
                'trades': 120,
                'win_rate': 0.62,
                'profit_factor': 1.9,
                'total_return': 0.15,
                'max_drawdown': 0.11
            }
        },
        {
            'name': 'MACD Strategy',
            'id': 'macd',
            'metrics': {
                'trades': 95,
                'win_rate': 0.59,
                'profit_factor': 1.7,
                'total_return': 0.12,
                'max_drawdown': 0.09
            }
        },
        {
            'name': 'Bollinger Bands',
            'id': 'bollinger_bands',
            'metrics': {
                'trades': 68,
                'win_rate': 0.72,
                'profit_factor': 2.5,
                'total_return': 0.22,
                'max_drawdown': 0.12
            }
        }
    ]
    
    return jsonify({
        'success': True,
        'data': {
            'strategies': strategies,
            'period': 'last_90_days',
            'timestamp': datetime.now().isoformat()
        }
    })


# Rutas para logs y alertas
@api_bp.route('/logs', methods=['GET'])
@require_auth_token
def get_logs():
    """Obtener logs del sistema."""
    # Parámetros
    level = request.args.get('level', 'info')  # debug, info, warning, error, critical
    component = request.args.get('component')
    limit = min(int(request.args.get('limit', 50)), 200)  # Limitar a máximo 200
    
    # Aquí se obtendrían los logs desde el LogManager
    # Esta implementación es simplificada
    
    # Datos de ejemplo
    now = datetime.now()
    logs = []
    
    levels = ['debug', 'info', 'warning', 'error', 'critical']
    components = ['system', 'market_data', 'strategy', 'risk_manager', 'api']
    
    for i in range(limit):
        timestamp = now - timedelta(minutes=i*10)
        log_level = levels[i % len(levels)]
        log_component = components[i % len(components)]
        
        # Filtrar por nivel
        if level == 'debug':
            pass  # mostrar todos
        elif level == 'info' and log_level == 'debug':
            continue
        elif level == 'warning' and log_level in ['debug', 'info']:
            continue
        elif level == 'error' and log_level in ['debug', 'info', 'warning']:
            continue
        elif level == 'critical' and log_level != 'critical':
            continue
        
        # Filtrar por componente
        if component and log_component != component:
            continue
        
        logs.append({
            'timestamp': timestamp.isoformat(),
            'level': log_level,
            'component': log_component,
            'message': f'Log message {i} for {log_component}',
            'details': {
                'request_id': str(uuid.uuid4()),
                'user': 'system',
                'source_ip': '127.0.0.1'
            }
        })
    
    # Ordenar cronológicamente (más reciente primero)
    logs.reverse()
    
    return jsonify({
        'success': True,
        'data': {
            'logs': logs,
            'count': len(logs),
            'filter': {
                'level': level,
                'component': component
            }
        }
    })


@api_bp.route('/alerts', methods=['GET'])
@require_auth_token
def get_alerts():
    """Obtener alertas activas del sistema."""
    # Aquí se obtendrían las alertas desde el AlertManager
    # Esta implementación es simplificada
    
    # Datos de ejemplo
    now = datetime.now()
    alerts = [
        {
            'id': str(uuid.uuid4()),
            'timestamp': (now - timedelta(minutes=15)).isoformat(),
            'type': 'price_alert',
            'level': 'info',
            'symbol': 'BTC/USDT',
            'message': 'BTC price exceeded 50000 USD',
            'status': 'active'
        },
        {
            'id': str(uuid.uuid4()),
            'timestamp': (now - timedelta(hours=2)).isoformat(),
            'type': 'system_alert',
            'level': 'warning',
            'component': 'exchange_connector',
            'message': 'Connection rate limited by exchange',
            'status': 'active'
        },
        {
            'id': str(uuid.uuid4()),
            'timestamp': (now - timedelta(hours=6)).isoformat(),
            'type': 'security_alert',
            'level': 'error',
            'component': 'api_server',
            'message': 'Multiple failed login attempts',
            'details': {
                'ip': '192.168.1.100',
                'attempts': 5
            },
            'status': 'active'
        }
    ]
    
    return jsonify({
        'success': True,
        'data': {
            'alerts': alerts,
            'count': len(alerts)
        }
    })


@api_bp.route('/alerts/<alert_id>/acknowledge', methods=['POST'])
@require_auth_token
def acknowledge_alert(alert_id):
    """Marcar una alerta como reconocida."""
    # Aquí se marcaría la alerta como reconocida en el AlertManager
    # Esta implementación es simplificada
    
    return jsonify({
        'success': True,
        'data': {
            'alert_id': alert_id,
            'status': 'acknowledged'
        }
    })


# Manejadores de errores
@api_bp.errorhandler(400)
def bad_request(error):
    return jsonify({
        'success': False,
        'error': 'Bad Request',
        'message': str(error)
    }), 400

@api_bp.errorhandler(401)
def unauthorized(error):
    return jsonify({
        'success': False,
        'error': 'Unauthorized',
        'message': 'Authentication required'
    }), 401

@api_bp.errorhandler(403)
def forbidden(error):
    return jsonify({
        'success': False,
        'error': 'Forbidden',
        'message': 'You don\'t have permission to access this resource'
    }), 403

@api_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Not Found',
        'message': 'The requested URL was not found on the server'
    }), 404

@api_bp.errorhandler(500)
def server_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred'
    }), 500


class RESTAPIManager(Component):
    """
    Gestor de API REST para el sistema Genesis.
    
    Este componente gestiona la API REST para integración con sistemas externos.
    """
    
    def __init__(self, name: str = "rest_api_manager"):
        """
        Inicializar el gestor de API REST.
        
        Args:
            name: Nombre del componente
        """
        super().__init__(name)
        self.logger = setup_logging(name)
    
    async def start(self) -> None:
        """Iniciar el gestor de API REST."""
        await super().start()
        self.logger.info("Gestor de API REST iniciado")
    
    async def stop(self) -> None:
        """Detener el gestor de API REST."""
        await super().stop()
        self.logger.info("Gestor de API REST detenido")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente de origen
        """
        # La comunicación con la API REST se gestiona a través de Flask
        # Este método recibe eventos de otros componentes que podrían ser relevantes
        # para la API REST, como actualizaciones de estado, alertas, etc.
        pass


# Exportación para uso fácil
rest_api_manager = RESTAPIManager()

# Función de inicialización para registrar el blueprint en la aplicación Flask
def init_api(app):
    """
    Inicializar la API REST en la aplicación Flask.
    
    Args:
        app: Aplicación Flask
    """
    app.register_blueprint(api_bp)
    logger.info("API REST inicializada en la aplicación Flask")