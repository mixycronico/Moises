"""
Documentación Swagger para la API REST de Genesis.

Este módulo proporciona la documentación de la API REST según las especificaciones
de Swagger/OpenAPI, facilitando el desarrollo y consumo de la API.
"""

import os
from flask import Blueprint, jsonify, render_template, Flask, current_app

# Crear el Blueprint para Swagger
swagger_bp = Blueprint('swagger', __name__, url_prefix='/api/docs')


# Definición de la documentación OpenAPI
OPENAPI_SPEC = {
    "openapi": "3.0.0",
    "info": {
        "title": "Genesis Trading System API",
        "description": "API REST para interactuar con el sistema de trading Genesis",
        "version": "1.0.0",
        "contact": {
            "name": "Genesis Trading Team",
            "email": "support@genesis-trading.com",
            "url": "https://genesis-trading.com"
        },
        "license": {
            "name": "Private",
            "url": "https://genesis-trading.com/license"
        }
    },
    "servers": [
        {
            "url": "/api/v1",
            "description": "Servidor principal"
        }
    ],
    "tags": [
        {
            "name": "auth",
            "description": "Autenticación y gestión de sesiones"
        },
        {
            "name": "market",
            "description": "Datos de mercado y precios"
        },
        {
            "name": "backtest",
            "description": "Operaciones de backtesting"
        },
        {
            "name": "analysis",
            "description": "Análisis de mercado e indicadores"
        },
        {
            "name": "accounting",
            "description": "Gestión de balances y transacciones"
        },
        {
            "name": "reports",
            "description": "Generación de reportes"
        },
        {
            "name": "logs",
            "description": "Logs y alertas del sistema"
        }
    ],
    "components": {
        "securitySchemes": {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key"
            },
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer"
            }
        },
        "schemas": {
            "Error": {
                "type": "object",
                "required": ["success", "error"],
                "properties": {
                    "success": {
                        "type": "boolean",
                        "example": False
                    },
                    "error": {
                        "type": "string",
                        "example": "Error message"
                    },
                    "message": {
                        "type": "string",
                        "example": "Detailed error message"
                    }
                }
            },
            "LoginRequest": {
                "type": "object",
                "required": ["username", "password"],
                "properties": {
                    "username": {
                        "type": "string",
                        "example": "admin"
                    },
                    "password": {
                        "type": "string",
                        "example": "password123"
                    }
                }
            },
            "LoginResponse": {
                "type": "object",
                "required": ["success", "token", "expiry", "user"],
                "properties": {
                    "success": {
                        "type": "boolean",
                        "example": True
                    },
                    "token": {
                        "type": "string",
                        "example": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
                    },
                    "expiry": {
                        "type": "string",
                        "format": "date-time",
                        "example": "2023-12-31T23:59:59"
                    },
                    "user": {
                        "type": "object",
                        "properties": {
                            "username": {
                                "type": "string",
                                "example": "admin"
                            },
                            "role": {
                                "type": "string",
                                "example": "admin"
                            }
                        }
                    }
                }
            },
            "Ticker": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "example": "BTC/USDT"
                    },
                    "last": {
                        "type": "number",
                        "format": "float",
                        "example": 50000.0
                    },
                    "bid": {
                        "type": "number",
                        "format": "float",
                        "example": 49995.0
                    },
                    "ask": {
                        "type": "number",
                        "format": "float",
                        "example": 50005.0
                    },
                    "volume": {
                        "type": "number",
                        "format": "float",
                        "example": 1000.0
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "example": "2023-01-01T12:00:00"
                    },
                    "change_24h": {
                        "type": "number",
                        "format": "float",
                        "example": 2.5
                    },
                    "high_24h": {
                        "type": "number",
                        "format": "float",
                        "example": 51000.0
                    },
                    "low_24h": {
                        "type": "number",
                        "format": "float",
                        "example": 49000.0
                    }
                }
            },
            "Candle": {
                "type": "object",
                "properties": {
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "example": "2023-01-01T12:00:00"
                    },
                    "open": {
                        "type": "number",
                        "format": "float",
                        "example": 50000.0
                    },
                    "high": {
                        "type": "number",
                        "format": "float",
                        "example": 50100.0
                    },
                    "low": {
                        "type": "number",
                        "format": "float",
                        "example": 49900.0
                    },
                    "close": {
                        "type": "number",
                        "format": "float",
                        "example": 50050.0
                    },
                    "volume": {
                        "type": "number",
                        "format": "float",
                        "example": 100.0
                    }
                }
            },
            "OrderBook": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "example": "BTC/USDT"
                    },
                    "bids": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "number",
                                "format": "float"
                            },
                            "example": [50000.0, 1.0]
                        }
                    },
                    "asks": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "number",
                                "format": "float"
                            },
                            "example": [50100.0, 1.0]
                        }
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "example": "2023-01-01T12:00:00"
                    }
                }
            },
            "BacktestRequest": {
                "type": "object",
                "required": ["symbol", "strategy"],
                "properties": {
                    "symbol": {
                        "type": "string",
                        "example": "BTC/USDT"
                    },
                    "strategy": {
                        "type": "string",
                        "example": "ma_crossover"
                    },
                    "start_date": {
                        "type": "string",
                        "format": "date-time",
                        "example": "2023-01-01T00:00:00"
                    },
                    "end_date": {
                        "type": "string",
                        "format": "date-time",
                        "example": "2023-12-31T23:59:59"
                    },
                    "params": {
                        "type": "object",
                        "example": {
                            "fast_period": 20,
                            "slow_period": 50
                        }
                    }
                }
            },
            "BacktestResult": {
                "type": "object",
                "properties": {
                    "backtest_id": {
                        "type": "string",
                        "example": "12345678-1234-5678-1234-567812345678"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["submitted", "running", "completed", "failed"],
                        "example": "completed"
                    },
                    "symbol": {
                        "type": "string",
                        "example": "BTC/USDT"
                    },
                    "strategy": {
                        "type": "string",
                        "example": "ma_crossover"
                    },
                    "start_date": {
                        "type": "string",
                        "format": "date-time",
                        "example": "2023-01-01T00:00:00"
                    },
                    "end_date": {
                        "type": "string",
                        "format": "date-time",
                        "example": "2023-12-31T23:59:59"
                    },
                    "params": {
                        "type": "object",
                        "example": {
                            "fast_period": 20,
                            "slow_period": 50
                        }
                    },
                    "metrics": {
                        "type": "object",
                        "properties": {
                            "total_trades": {
                                "type": "integer",
                                "example": 120
                            },
                            "win_rate": {
                                "type": "number",
                                "format": "float",
                                "example": 0.65
                            },
                            "profit_factor": {
                                "type": "number",
                                "format": "float",
                                "example": 2.1
                            },
                            "max_drawdown": {
                                "type": "number",
                                "format": "float",
                                "example": 0.12
                            },
                            "net_profit": {
                                "type": "number",
                                "format": "float",
                                "example": 0.32
                            },
                            "sharpe_ratio": {
                                "type": "number",
                                "format": "float",
                                "example": 1.8
                            }
                        }
                    },
                    "trades": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "timestamp": {
                                    "type": "string",
                                    "format": "date-time",
                                    "example": "2023-01-15T10:30:00"
                                },
                                "type": {
                                    "type": "string",
                                    "enum": ["buy", "sell"],
                                    "example": "buy"
                                },
                                "price": {
                                    "type": "number",
                                    "format": "float",
                                    "example": 42000.0
                                },
                                "amount": {
                                    "type": "number",
                                    "format": "float",
                                    "example": 0.1
                                },
                                "pnl": {
                                    "type": "number",
                                    "format": "float",
                                    "example": 0.0
                                }
                            }
                        }
                    }
                }
            },
            "Balance": {
                "type": "object",
                "properties": {
                    "total_usd": {
                        "type": "number",
                        "format": "float",
                        "example": 125000.0
                    },
                    "assets": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "asset": {
                                    "type": "string",
                                    "example": "BTC"
                                },
                                "free": {
                                    "type": "number",
                                    "format": "float",
                                    "example": 1.5
                                },
                                "locked": {
                                    "type": "number",
                                    "format": "float",
                                    "example": 0.1
                                },
                                "total": {
                                    "type": "number",
                                    "format": "float",
                                    "example": 1.6
                                },
                                "price_usd": {
                                    "type": "number",
                                    "format": "float",
                                    "example": 50000.0
                                },
                                "value_usd": {
                                    "type": "number",
                                    "format": "float",
                                    "example": 80000.0
                                }
                            }
                        }
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "example": "2023-01-01T12:00:00"
                    }
                }
            },
            "Transaction": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "example": "tx_12345678-1234-5678-1234-567812345678"
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "example": "2023-01-01T12:00:00"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["deposit", "withdrawal", "trade", "fee", "transfer"],
                        "example": "trade"
                    },
                    "asset": {
                        "type": "string",
                        "example": "BTC"
                    },
                    "amount": {
                        "type": "number",
                        "format": "float",
                        "example": 0.1
                    },
                    "price": {
                        "type": "number",
                        "format": "float",
                        "example": 50000.0
                    },
                    "value_usd": {
                        "type": "number",
                        "format": "float",
                        "example": 5000.0
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "completed", "failed"],
                        "example": "completed"
                    }
                }
            },
            "IndicatorData": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "example": "BTC/USDT"
                    },
                    "timeframe": {
                        "type": "string",
                        "example": "1h"
                    },
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "timestamp": {
                                    "type": "string",
                                    "format": "date-time",
                                    "example": "2023-01-01T12:00:00"
                                },
                                "price": {
                                    "type": "number",
                                    "format": "float",
                                    "example": 50000.0
                                },
                                "rsi": {
                                    "type": "number",
                                    "format": "float",
                                    "example": 50
                                },
                                "macd": {
                                    "type": "object",
                                    "properties": {
                                        "line": {
                                            "type": "number",
                                            "format": "float",
                                            "example": 100
                                        },
                                        "signal": {
                                            "type": "number",
                                            "format": "float",
                                            "example": 50
                                        },
                                        "histogram": {
                                            "type": "number",
                                            "format": "float",
                                            "example": 50
                                        }
                                    }
                                },
                                "bollinger_bands": {
                                    "type": "object",
                                    "properties": {
                                        "upper": {
                                            "type": "number",
                                            "format": "float",
                                            "example": 52000.0
                                        },
                                        "middle": {
                                            "type": "number",
                                            "format": "float",
                                            "example": 50000.0
                                        },
                                        "lower": {
                                            "type": "number",
                                            "format": "float",
                                            "example": 48000.0
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "Anomaly": {
                "type": "object",
                "properties": {
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "example": "2023-01-01T12:00:00"
                    },
                    "type": {
                        "type": "string",
                        "example": "price_spike"
                    },
                    "score": {
                        "type": "number",
                        "format": "float",
                        "example": 0.8
                    },
                    "details": {
                        "type": "object",
                        "example": {
                            "description": "Detected price_spike anomaly",
                            "threshold": 0.65,
                            "value": 0.8,
                            "z_score": 3.2
                        }
                    }
                }
            },
            "Log": {
                "type": "object",
                "properties": {
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "example": "2023-01-01T12:00:00"
                    },
                    "level": {
                        "type": "string",
                        "enum": ["debug", "info", "warning", "error", "critical"],
                        "example": "info"
                    },
                    "component": {
                        "type": "string",
                        "example": "market_data"
                    },
                    "message": {
                        "type": "string",
                        "example": "Log message for market_data"
                    },
                    "details": {
                        "type": "object",
                        "example": {
                            "request_id": "12345678-1234-5678-1234-567812345678",
                            "user": "system",
                            "source_ip": "127.0.0.1"
                        }
                    }
                }
            },
            "Alert": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "example": "12345678-1234-5678-1234-567812345678"
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "example": "2023-01-01T12:00:00"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["price_alert", "system_alert", "security_alert"],
                        "example": "price_alert"
                    },
                    "level": {
                        "type": "string",
                        "enum": ["info", "warning", "error", "critical"],
                        "example": "info"
                    },
                    "symbol": {
                        "type": "string",
                        "example": "BTC/USDT"
                    },
                    "message": {
                        "type": "string",
                        "example": "BTC price exceeded 50000 USD"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["active", "acknowledged", "resolved"],
                        "example": "active"
                    },
                    "details": {
                        "type": "object",
                        "example": {}
                    }
                }
            }
        }
    },
    "paths": {
        "/auth/login": {
            "post": {
                "tags": ["auth"],
                "summary": "Iniciar sesión y obtener token de autorización",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/LoginRequest"
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Login exitoso",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/LoginResponse"
                                }
                            }
                        }
                    },
                    "401": {
                        "description": "Credenciales inválidas",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/auth/logout": {
            "post": {
                "tags": ["auth"],
                "summary": "Cerrar sesión e invalidar token",
                "security": [
                    {"BearerAuth": []}
                ],
                "responses": {
                    "200": {
                        "description": "Sesión cerrada correctamente",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {
                                            "type": "boolean",
                                            "example": True
                                        },
                                        "message": {
                                            "type": "string",
                                            "example": "Sesión cerrada correctamente"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "401": {
                        "description": "No autorizado",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/auth/refresh": {
            "post": {
                "tags": ["auth"],
                "summary": "Renovar token de autorización",
                "security": [
                    {"BearerAuth": []}
                ],
                "responses": {
                    "200": {
                        "description": "Token renovado correctamente",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {
                                            "type": "boolean",
                                            "example": True
                                        },
                                        "token": {
                                            "type": "string",
                                            "example": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
                                        },
                                        "expiry": {
                                            "type": "string",
                                            "format": "date-time",
                                            "example": "2023-12-31T23:59:59"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "401": {
                        "description": "No autorizado",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/market/symbols": {
            "get": {
                "tags": ["market"],
                "summary": "Obtener lista de símbolos disponibles",
                "security": [
                    {"ApiKeyAuth": []}
                ],
                "responses": {
                    "200": {
                        "description": "Lista de símbolos",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {
                                            "type": "boolean",
                                            "example": True
                                        },
                                        "data": {
                                            "type": "object",
                                            "properties": {
                                                "symbols": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "string"
                                                    },
                                                    "example": ["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT"]
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "401": {
                        "description": "No autorizado",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/market/ticker/{symbol}": {
            "get": {
                "tags": ["market"],
                "summary": "Obtener datos de ticker para un símbolo",
                "security": [
                    {"ApiKeyAuth": []}
                ],
                "parameters": [
                    {
                        "name": "symbol",
                        "in": "path",
                        "required": True,
                        "schema": {
                            "type": "string"
                        },
                        "example": "BTC/USDT",
                        "description": "Símbolo de trading (par)"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Datos de ticker",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {
                                            "type": "boolean",
                                            "example": True
                                        },
                                        "data": {
                                            "$ref": "#/components/schemas/Ticker"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Parámetros inválidos",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    },
                    "401": {
                        "description": "No autorizado",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/market/candles/{symbol}": {
            "get": {
                "tags": ["market"],
                "summary": "Obtener datos OHLCV para un símbolo",
                "security": [
                    {"ApiKeyAuth": []}
                ],
                "parameters": [
                    {
                        "name": "symbol",
                        "in": "path",
                        "required": True,
                        "schema": {
                            "type": "string"
                        },
                        "example": "BTC/USDT",
                        "description": "Símbolo de trading (par)"
                    },
                    {
                        "name": "timeframe",
                        "in": "query",
                        "required": False,
                        "schema": {
                            "type": "string",
                            "enum": ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
                        },
                        "default": "1h",
                        "description": "Intervalo de tiempo"
                    },
                    {
                        "name": "limit",
                        "in": "query",
                        "required": False,
                        "schema": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 1000
                        },
                        "default": 100,
                        "description": "Cantidad máxima de velas a retornar"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Datos OHLCV",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {
                                            "type": "boolean",
                                            "example": True
                                        },
                                        "data": {
                                            "type": "object",
                                            "properties": {
                                                "symbol": {
                                                    "type": "string",
                                                    "example": "BTC/USDT"
                                                },
                                                "timeframe": {
                                                    "type": "string",
                                                    "example": "1h"
                                                },
                                                "candles": {
                                                    "type": "array",
                                                    "items": {
                                                        "$ref": "#/components/schemas/Candle"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Parámetros inválidos",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    },
                    "401": {
                        "description": "No autorizado",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/market/orderbook/{symbol}": {
            "get": {
                "tags": ["market"],
                "summary": "Obtener libro de órdenes para un símbolo",
                "security": [
                    {"ApiKeyAuth": []}
                ],
                "parameters": [
                    {
                        "name": "symbol",
                        "in": "path",
                        "required": True,
                        "schema": {
                            "type": "string"
                        },
                        "example": "BTC/USDT",
                        "description": "Símbolo de trading (par)"
                    },
                    {
                        "name": "depth",
                        "in": "query",
                        "required": False,
                        "schema": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100
                        },
                        "default": 20,
                        "description": "Profundidad del libro de órdenes"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Libro de órdenes",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {
                                            "type": "boolean",
                                            "example": True
                                        },
                                        "data": {
                                            "$ref": "#/components/schemas/OrderBook"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Parámetros inválidos",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    },
                    "401": {
                        "description": "No autorizado",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/backtest": {
            "post": {
                "tags": ["backtest"],
                "summary": "Ejecutar un backtest con parámetros específicos",
                "security": [
                    {"BearerAuth": []}
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/BacktestRequest"
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Backtest iniciado",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {
                                            "type": "boolean",
                                            "example": True
                                        },
                                        "data": {
                                            "type": "object",
                                            "properties": {
                                                "backtest_id": {
                                                    "type": "string",
                                                    "example": "12345678-1234-5678-1234-567812345678"
                                                },
                                                "status": {
                                                    "type": "string",
                                                    "example": "submitted"
                                                },
                                                "params": {
                                                    "type": "object",
                                                    "example": {
                                                        "symbol": "BTC/USDT",
                                                        "strategy": "ma_crossover",
                                                        "start_date": "2023-01-01T00:00:00",
                                                        "end_date": "2023-12-31T23:59:59",
                                                        "params": {
                                                            "fast_period": 20,
                                                            "slow_period": 50
                                                        }
                                                    }
                                                },
                                                "eta_seconds": {
                                                    "type": "integer",
                                                    "example": 30
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Parámetros inválidos",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    },
                    "401": {
                        "description": "No autorizado",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/backtest/{backtest_id}": {
            "get": {
                "tags": ["backtest"],
                "summary": "Obtener resultado de un backtest por ID",
                "security": [
                    {"BearerAuth": []}
                ],
                "parameters": [
                    {
                        "name": "backtest_id",
                        "in": "path",
                        "required": True,
                        "schema": {
                            "type": "string"
                        },
                        "example": "12345678-1234-5678-1234-567812345678",
                        "description": "ID del backtest"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Resultado del backtest",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {
                                            "type": "boolean",
                                            "example": True
                                        },
                                        "data": {
                                            "$ref": "#/components/schemas/BacktestResult"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "404": {
                        "description": "Backtest no encontrado",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    },
                    "401": {
                        "description": "No autorizado",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/analysis/indicators/{symbol}": {
            "get": {
                "tags": ["analysis"],
                "summary": "Obtener indicadores técnicos para un símbolo",
                "security": [
                    {"ApiKeyAuth": []}
                ],
                "parameters": [
                    {
                        "name": "symbol",
                        "in": "path",
                        "required": True,
                        "schema": {
                            "type": "string"
                        },
                        "example": "BTC/USDT",
                        "description": "Símbolo de trading (par)"
                    },
                    {
                        "name": "timeframe",
                        "in": "query",
                        "required": False,
                        "schema": {
                            "type": "string",
                            "enum": ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
                        },
                        "default": "1h",
                        "description": "Intervalo de tiempo"
                    },
                    {
                        "name": "indicators",
                        "in": "query",
                        "required": False,
                        "schema": {
                            "type": "string"
                        },
                        "default": "rsi,macd,bb",
                        "description": "Indicadores a calcular (separados por coma)"
                    },
                    {
                        "name": "limit",
                        "in": "query",
                        "required": False,
                        "schema": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 200
                        },
                        "default": 50,
                        "description": "Cantidad máxima de puntos a retornar"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Indicadores técnicos",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {
                                            "type": "boolean",
                                            "example": True
                                        },
                                        "data": {
                                            "$ref": "#/components/schemas/IndicatorData"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Parámetros inválidos",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    },
                    "401": {
                        "description": "No autorizado",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/analysis/anomalies/{symbol}": {
            "get": {
                "tags": ["analysis"],
                "summary": "Obtener anomalías detectadas para un símbolo",
                "security": [
                    {"ApiKeyAuth": []}
                ],
                "parameters": [
                    {
                        "name": "symbol",
                        "in": "path",
                        "required": True,
                        "schema": {
                            "type": "string"
                        },
                        "example": "BTC/USDT",
                        "description": "Símbolo de trading (par)"
                    },
                    {
                        "name": "days",
                        "in": "query",
                        "required": False,
                        "schema": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 30
                        },
                        "default": 7,
                        "description": "Número de días a analizar"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Anomalías detectadas",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {
                                            "type": "boolean",
                                            "example": True
                                        },
                                        "data": {
                                            "type": "object",
                                            "properties": {
                                                "symbol": {
                                                    "type": "string",
                                                    "example": "BTC/USDT"
                                                },
                                                "period_days": {
                                                    "type": "integer",
                                                    "example": 7
                                                },
                                                "anomalies": {
                                                    "type": "array",
                                                    "items": {
                                                        "$ref": "#/components/schemas/Anomaly"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Parámetros inválidos",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    },
                    "401": {
                        "description": "No autorizado",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/accounting/balance": {
            "get": {
                "tags": ["accounting"],
                "summary": "Obtener balance total de la cuenta",
                "security": [
                    {"BearerAuth": []}
                ],
                "responses": {
                    "200": {
                        "description": "Balance de la cuenta",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {
                                            "type": "boolean",
                                            "example": True
                                        },
                                        "data": {
                                            "$ref": "#/components/schemas/Balance"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "401": {
                        "description": "No autorizado",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/accounting/transactions": {
            "get": {
                "tags": ["accounting"],
                "summary": "Obtener historial de transacciones",
                "security": [
                    {"BearerAuth": []}
                ],
                "parameters": [
                    {
                        "name": "asset",
                        "in": "query",
                        "required": False,
                        "schema": {
                            "type": "string"
                        },
                        "example": "BTC",
                        "description": "Filtrar por activo"
                    },
                    {
                        "name": "type",
                        "in": "query",
                        "required": False,
                        "schema": {
                            "type": "string",
                            "enum": ["deposit", "withdrawal", "trade", "fee", "transfer"]
                        },
                        "description": "Filtrar por tipo de transacción"
                    },
                    {
                        "name": "limit",
                        "in": "query",
                        "required": False,
                        "schema": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100
                        },
                        "default": 20,
                        "description": "Cantidad máxima de transacciones a retornar"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Historial de transacciones",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {
                                            "type": "boolean",
                                            "example": True
                                        },
                                        "data": {
                                            "type": "object",
                                            "properties": {
                                                "transactions": {
                                                    "type": "array",
                                                    "items": {
                                                        "$ref": "#/components/schemas/Transaction"
                                                    }
                                                },
                                                "count": {
                                                    "type": "integer",
                                                    "example": 20
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "401": {
                        "description": "No autorizado",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/reports/performance": {
            "get": {
                "tags": ["reports"],
                "summary": "Obtener reporte de rendimiento",
                "security": [
                    {"BearerAuth": []}
                ],
                "parameters": [
                    {
                        "name": "period",
                        "in": "query",
                        "required": False,
                        "schema": {
                            "type": "string",
                            "enum": ["daily", "weekly", "monthly"]
                        },
                        "default": "daily",
                        "description": "Periodo del reporte"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Reporte de rendimiento",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {
                                            "type": "boolean",
                                            "example": True
                                        },
                                        "data": {
                                            "type": "object",
                                            "properties": {
                                                "period": {
                                                    "type": "string",
                                                    "example": "daily"
                                                },
                                                "start_date": {
                                                    "type": "string",
                                                    "format": "date-time",
                                                    "example": "2023-01-01T00:00:00"
                                                },
                                                "end_date": {
                                                    "type": "string",
                                                    "format": "date-time",
                                                    "example": "2023-12-31T23:59:59"
                                                },
                                                "start_value": {
                                                    "type": "number",
                                                    "format": "float",
                                                    "example": 100000.0
                                                },
                                                "end_value": {
                                                    "type": "number",
                                                    "format": "float",
                                                    "example": 150000.0
                                                },
                                                "total_return": {
                                                    "type": "number",
                                                    "format": "float",
                                                    "example": 0.5
                                                },
                                                "annualized_return": {
                                                    "type": "number",
                                                    "format": "float",
                                                    "example": 0.6
                                                },
                                                "volatility": {
                                                    "type": "number",
                                                    "format": "float",
                                                    "example": 0.12
                                                },
                                                "sharpe_ratio": {
                                                    "type": "number",
                                                    "format": "float",
                                                    "example": 1.8
                                                },
                                                "max_drawdown": {
                                                    "type": "number",
                                                    "format": "float",
                                                    "example": 0.09
                                                },
                                                "data": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "timestamp": {
                                                                "type": "string",
                                                                "format": "date-time",
                                                                "example": "2023-01-01T00:00:00"
                                                            },
                                                            "portfolio_value": {
                                                                "type": "number",
                                                                "format": "float",
                                                                "example": 100000.0
                                                            },
                                                            "change_pct": {
                                                                "type": "number",
                                                                "format": "float",
                                                                "example": 0.005
                                                            },
                                                            "benchmark_value": {
                                                                "type": "number",
                                                                "format": "float",
                                                                "example": 100000.0
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "401": {
                        "description": "No autorizado",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/reports/strategies": {
            "get": {
                "tags": ["reports"],
                "summary": "Obtener reporte de rendimiento por estrategia",
                "security": [
                    {"BearerAuth": []}
                ],
                "responses": {
                    "200": {
                        "description": "Reporte de estrategias",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {
                                            "type": "boolean",
                                            "example": True
                                        },
                                        "data": {
                                            "type": "object",
                                            "properties": {
                                                "strategies": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "name": {
                                                                "type": "string",
                                                                "example": "MA Crossover"
                                                            },
                                                            "id": {
                                                                "type": "string",
                                                                "example": "ma_crossover"
                                                            },
                                                            "metrics": {
                                                                "type": "object",
                                                                "properties": {
                                                                    "trades": {
                                                                        "type": "integer",
                                                                        "example": 85
                                                                    },
                                                                    "win_rate": {
                                                                        "type": "number",
                                                                        "format": "float",
                                                                        "example": 0.68
                                                                    },
                                                                    "profit_factor": {
                                                                        "type": "number",
                                                                        "format": "float",
                                                                        "example": 2.3
                                                                    },
                                                                    "total_return": {
                                                                        "type": "number",
                                                                        "format": "float",
                                                                        "example": 0.18
                                                                    },
                                                                    "max_drawdown": {
                                                                        "type": "number",
                                                                        "format": "float",
                                                                        "example": 0.08
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                },
                                                "period": {
                                                    "type": "string",
                                                    "example": "last_90_days"
                                                },
                                                "timestamp": {
                                                    "type": "string",
                                                    "format": "date-time",
                                                    "example": "2023-01-01T12:00:00"
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "401": {
                        "description": "No autorizado",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/logs": {
            "get": {
                "tags": ["logs"],
                "summary": "Obtener logs del sistema",
                "security": [
                    {"BearerAuth": []}
                ],
                "parameters": [
                    {
                        "name": "level",
                        "in": "query",
                        "required": False,
                        "schema": {
                            "type": "string",
                            "enum": ["debug", "info", "warning", "error", "critical"]
                        },
                        "default": "info",
                        "description": "Nivel mínimo de logs"
                    },
                    {
                        "name": "component",
                        "in": "query",
                        "required": False,
                        "schema": {
                            "type": "string"
                        },
                        "description": "Filtrar por componente"
                    },
                    {
                        "name": "limit",
                        "in": "query",
                        "required": False,
                        "schema": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 200
                        },
                        "default": 50,
                        "description": "Cantidad máxima de logs a retornar"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Logs del sistema",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {
                                            "type": "boolean",
                                            "example": True
                                        },
                                        "data": {
                                            "type": "object",
                                            "properties": {
                                                "logs": {
                                                    "type": "array",
                                                    "items": {
                                                        "$ref": "#/components/schemas/Log"
                                                    }
                                                },
                                                "count": {
                                                    "type": "integer",
                                                    "example": 50
                                                },
                                                "filter": {
                                                    "type": "object",
                                                    "properties": {
                                                        "level": {
                                                            "type": "string",
                                                            "example": "info"
                                                        },
                                                        "component": {
                                                            "type": "string",
                                                            "example": "market_data"
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "401": {
                        "description": "No autorizado",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/alerts": {
            "get": {
                "tags": ["logs"],
                "summary": "Obtener alertas activas del sistema",
                "security": [
                    {"BearerAuth": []}
                ],
                "responses": {
                    "200": {
                        "description": "Alertas del sistema",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {
                                            "type": "boolean",
                                            "example": True
                                        },
                                        "data": {
                                            "type": "object",
                                            "properties": {
                                                "alerts": {
                                                    "type": "array",
                                                    "items": {
                                                        "$ref": "#/components/schemas/Alert"
                                                    }
                                                },
                                                "count": {
                                                    "type": "integer",
                                                    "example": 3
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "401": {
                        "description": "No autorizado",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/alerts/{alert_id}/acknowledge": {
            "post": {
                "tags": ["logs"],
                "summary": "Marcar una alerta como reconocida",
                "security": [
                    {"BearerAuth": []}
                ],
                "parameters": [
                    {
                        "name": "alert_id",
                        "in": "path",
                        "required": True,
                        "schema": {
                            "type": "string"
                        },
                        "example": "12345678-1234-5678-1234-567812345678",
                        "description": "ID de la alerta"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Alerta reconocida",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {
                                            "type": "boolean",
                                            "example": True
                                        },
                                        "data": {
                                            "type": "object",
                                            "properties": {
                                                "alert_id": {
                                                    "type": "string",
                                                    "example": "12345678-1234-5678-1234-567812345678"
                                                },
                                                "status": {
                                                    "type": "string",
                                                    "example": "acknowledged"
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "404": {
                        "description": "Alerta no encontrada",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    },
                    "401": {
                        "description": "No autorizado",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


# Ruta para la documentación Swagger
@swagger_bp.route('/')
def swagger_ui():
    """Render Swagger UI."""
    return render_template('swagger_ui.html', title="Genesis API Documentation")


# Ruta para obtener la especificación OpenAPI
@swagger_bp.route('/openapi.json')
def openapi_spec():
    """Get OpenAPI specification."""
    return jsonify(OPENAPI_SPEC)


# Plantilla HTML para Swagger UI
SWAGGER_UI_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4.3.0/swagger-ui.css" >
    <style>
        html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
        *, *:before, *:after { box-sizing: inherit; }
        body { margin: 0; background: #222; }
        #swagger-ui { margin: 0 auto; }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4.3.0/swagger-ui-bundle.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4.3.0/swagger-ui-standalone-preset.js"></script>
    <script>
    window.onload = function() {
        const ui = SwaggerUIBundle({
            url: "{{ url_for('swagger.openapi_spec') }}",
            dom_id: '#swagger-ui',
            deepLinking: true,
            presets: [
                SwaggerUIBundle.presets.apis,
                SwaggerUIStandalonePreset
            ],
            plugins: [
                SwaggerUIBundle.plugins.DownloadUrl
            ],
            layout: "BaseLayout",
            supportedSubmitMethods: ['get', 'post', 'put', 'delete', 'patch'],
            syntaxHighlight: {
                activated: true,
                theme: "monokai"
            }
        });
        window.ui = ui;
    };
    </script>
</body>
</html>
"""


# Función de inicialización para registrar el blueprint de swagger
def init_swagger(app):
    """
    Inicializar la documentación Swagger en la aplicación Flask.
    
    Args:
        app: Aplicación Flask
    """
    # Registrar el blueprint
    app.register_blueprint(swagger_bp)
    
    # Registrar la plantilla
    app.jinja_env.globals['swagger_ui_template'] = SWAGGER_UI_TEMPLATE
    
    # Crear un directorio de plantillas en memoria
    @app.route('/api/docs/swagger_ui.html')
    def render_swagger_ui():
        """Render Swagger UI template."""
        return SWAGGER_UI_TEMPLATE