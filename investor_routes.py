"""
Rutas de API para gestión de inversionistas en el Sistema Genesis.

Este módulo implementa los endpoints necesarios para que los administradores
puedan gestionar inversionistas y sus balances, y para que los inversionistas
puedan ver sus propios datos y realizar operaciones.
"""

import logging
import json
from flask import jsonify, request, session
from decimal import Decimal

# Configuración de logging
logger = logging.getLogger('genesis_website')

# Datos simulados para demostración
# En producción, estos datos vendrían de la base de datos
INVESTOR_CATEGORIES = {
    "platinum": {
        "min_investment": 100000,
        "bonus_rate": 0.07,  # 7%
        "access_level": 3,
        "features": ["Acceso a todas las estrategias", "Asesoría personalizada 24/7", "Acceso a DeepSeek Premium", "Informes trimestrales detallados"]
    },
    "gold": {
        "min_investment": 50000,
        "bonus_rate": 0.05,  # 5%
        "access_level": 2,
        "features": ["Acceso a estrategias avanzadas", "Asesoría personalizada", "Informes mensuales"]
    },
    "silver": {
        "min_investment": 10000,
        "bonus_rate": 0.03,  # 3%
        "access_level": 1,
        "features": ["Acceso a estrategias básicas", "Soporte por email", "Informes trimestrales"]
    },
    "bronze": {
        "min_investment": 1000,
        "bonus_rate": 0.02,  # 2%
        "access_level": 0,
        "features": ["Acceso a estrategias básicas", "Soporte por email"]
    }
}

# Datos simulados de inversionistas
INVESTORS = {
    "1": {
        "id": "1",
        "user_id": "investor",
        "name": "Usuario Inversor",
        "email": "investor@genesis.ai",
        "balance": 12500.75,
        "invested": 10000.00,
        "available": 2500.75,
        "status": "active",
        "category": "gold",
        "join_date": "2024-01-15",
        "referral_code": "INV001",
        "total_profit": 2500.75,
        "performance": {
            "daily": 0.5,
            "weekly": 2.3,
            "monthly": 8.1,
            "yearly": 25.0
        },
        "investments": [
            {
                "id": "inv_1",
                "name": "Bitcoin",
                "symbol": "BTC",
                "amount": 0.12,
                "value_usd": 6500.00,
                "change_24h": 1.8,
                "allocation": 65.0
            },
            {
                "id": "inv_2",
                "name": "Ethereum",
                "symbol": "ETH",
                "amount": 2.5,
                "value_usd": 3500.00,
                "change_24h": -0.7,
                "allocation": 35.0
            }
        ],
        "transactions": [
            {
                "id": "tx_1",
                "type": "deposit",
                "amount": 10000.00,
                "status": "completed",
                "date": "2024-01-15",
                "description": "Depósito inicial"
            },
            {
                "id": "tx_2",
                "type": "bonus",
                "amount": 500.00,
                "status": "completed",
                "date": "2024-02-15",
                "description": "Bono mensual"
            },
            {
                "id": "tx_3",
                "type": "profit",
                "amount": 2000.75,
                "status": "completed",
                "date": "2024-03-15",
                "description": "Ganancia de inversiones"
            }
        ]
    },
    "2": {
        "id": "2",
        "user_id": "investor2",
        "name": "Ana Martínez",
        "email": "ana.martinez@example.com",
        "balance": 55000.00,
        "invested": 50000.00,
        "available": 5000.00,
        "status": "active",
        "category": "platinum",
        "join_date": "2023-12-01",
        "referral_code": "INV002",
        "total_profit": 5000.00,
        "performance": {
            "daily": 0.6,
            "weekly": 2.8,
            "monthly": 9.2,
            "yearly": 32.0
        },
        "investments": [
            {
                "id": "inv_3",
                "name": "Bitcoin",
                "symbol": "BTC",
                "amount": 0.5,
                "value_usd": 25000.00,
                "change_24h": 1.8,
                "allocation": 50.0
            },
            {
                "id": "inv_4",
                "name": "Ethereum",
                "symbol": "ETH",
                "amount": 10.0,
                "value_usd": 15000.00,
                "change_24h": -0.7,
                "allocation": 30.0
            },
            {
                "id": "inv_5",
                "name": "Cardano",
                "symbol": "ADA",
                "amount": 10000.0,
                "value_usd": 10000.00,
                "change_24h": 3.2,
                "allocation": 20.0
            }
        ],
        "transactions": [
            {
                "id": "tx_4",
                "type": "deposit",
                "amount": 50000.00,
                "status": "completed",
                "date": "2023-12-01",
                "description": "Depósito inicial"
            },
            {
                "id": "tx_5",
                "type": "bonus",
                "amount": 2500.00,
                "status": "completed",
                "date": "2024-01-01",
                "description": "Bono mensual"
            },
            {
                "id": "tx_6",
                "type": "profit",
                "amount": 2500.00,
                "status": "completed",
                "date": "2024-02-01",
                "description": "Ganancia de inversiones"
            }
        ]
    },
    "3": {
        "id": "3",
        "user_id": "investor3",
        "name": "Carlos Rodríguez",
        "email": "carlos.rodriguez@example.com",
        "balance": 5500.00,
        "invested": 5000.00,
        "available": 500.00,
        "status": "active",
        "category": "silver",
        "join_date": "2024-02-10",
        "referral_code": "INV003",
        "total_profit": 500.00,
        "performance": {
            "daily": 0.4,
            "weekly": 1.9,
            "monthly": 7.5,
            "yearly": 20.0
        },
        "investments": [
            {
                "id": "inv_6",
                "name": "Bitcoin",
                "symbol": "BTC",
                "amount": 0.05,
                "value_usd": 2500.00,
                "change_24h": 1.8,
                "allocation": 50.0
            },
            {
                "id": "inv_7",
                "name": "Ethereum",
                "symbol": "ETH",
                "amount": 1.5,
                "value_usd": 2500.00,
                "change_24h": -0.7,
                "allocation": 50.0
            }
        ],
        "transactions": [
            {
                "id": "tx_7",
                "type": "deposit",
                "amount": 5000.00,
                "status": "completed",
                "date": "2024-02-10",
                "description": "Depósito inicial"
            },
            {
                "id": "tx_8",
                "type": "bonus",
                "amount": 150.00,
                "status": "completed",
                "date": "2024-03-10",
                "description": "Bono mensual"
            },
            {
                "id": "tx_9",
                "type": "profit",
                "amount": 350.00,
                "status": "completed",
                "date": "2024-03-15",
                "description": "Ganancia de inversiones"
            }
        ]
    },
    "4": {
        "id": "4",
        "user_id": "investor4",
        "name": "Laura Gómez",
        "email": "laura.gomez@example.com",
        "balance": 1200.00,
        "invested": 1000.00,
        "available": 200.00,
        "status": "active",
        "category": "bronze",
        "join_date": "2024-03-01",
        "referral_code": "INV004",
        "total_profit": 200.00,
        "performance": {
            "daily": 0.3,
            "weekly": 1.5,
            "monthly": 6.0,
            "yearly": 18.0
        },
        "investments": [
            {
                "id": "inv_8",
                "name": "Bitcoin",
                "symbol": "BTC",
                "amount": 0.02,
                "value_usd": 1000.00,
                "change_24h": 1.8,
                "allocation": 100.0
            }
        ],
        "transactions": [
            {
                "id": "tx_10",
                "type": "deposit",
                "amount": 1000.00,
                "status": "completed",
                "date": "2024-03-01",
                "description": "Depósito inicial"
            },
            {
                "id": "tx_11",
                "type": "bonus",
                "amount": 20.00,
                "status": "completed",
                "date": "2024-03-15",
                "description": "Bono inicial"
            },
            {
                "id": "tx_12",
                "type": "profit",
                "amount": 180.00,
                "status": "completed",
                "date": "2024-03-20",
                "description": "Ganancia de inversiones"
            }
        ]
    },
    "5": {
        "id": "5",
        "user_id": "mixycronico",
        "name": "Moises Alvarenga",
        "email": "mixycronico@aol.com",
        "balance": 350000.00,
        "invested": 300000.00,
        "available": 50000.00,
        "status": "active",
        "category": "platinum",
        "join_date": "2023-10-01",
        "referral_code": "FOUNDER001",
        "total_profit": 50000.00,
        "performance": {
            "daily": 0.7,
            "weekly": 3.2,
            "monthly": 10.5,
            "yearly": 38.0
        },
        "investments": [
            {
                "id": "inv_9",
                "name": "Bitcoin",
                "symbol": "BTC",
                "amount": 3.0,
                "value_usd": 150000.00,
                "change_24h": 1.8,
                "allocation": 50.0
            },
            {
                "id": "inv_10",
                "name": "Ethereum",
                "symbol": "ETH",
                "amount": 50.0,
                "value_usd": 75000.00,
                "change_24h": -0.7,
                "allocation": 25.0
            },
            {
                "id": "inv_11",
                "name": "Cardano",
                "symbol": "ADA",
                "amount": 50000.0,
                "value_usd": 45000.00,
                "change_24h": 3.2,
                "allocation": 15.0
            },
            {
                "id": "inv_12",
                "name": "Solana",
                "symbol": "SOL",
                "amount": 300.0,
                "value_usd": 30000.00,
                "change_24h": 5.1,
                "allocation": 10.0
            }
        ],
        "transactions": [
            {
                "id": "tx_13",
                "type": "deposit",
                "amount": 300000.00,
                "status": "completed",
                "date": "2023-10-01",
                "description": "Inversión inicial"
            },
            {
                "id": "tx_14",
                "type": "bonus",
                "amount": 15000.00,
                "status": "completed",
                "date": "2024-01-01",
                "description": "Bono trimestral"
            },
            {
                "id": "tx_15",
                "type": "profit",
                "amount": 35000.00,
                "status": "completed",
                "date": "2024-03-01",
                "description": "Ganancia de inversiones"
            }
        ]
    }
}

# Datos simulados del sistema
SYSTEM_STATS = {
    "total_investors": len(INVESTORS),
    "active_investors": sum(1 for i in INVESTORS.values() if i["status"] == "active"),
    "total_capital": sum(i["balance"] for i in INVESTORS.values()),
    "total_invested": sum(i["invested"] for i in INVESTORS.values()),
    "total_profit_distributed": sum(i["total_profit"] for i in INVESTORS.values()),
    "maintenance_fund": 15000.00,
    "daily_stats": {
        "bonus_enabled": True,
        "loans_enabled": True,
        "commissions_allowed": True,
        "transfer_limit": 10000.00,
        "notes": "Sistema funcionando correctamente"
    },
    "performance": {
        "daily": 0.55,
        "weekly": 2.7,
        "monthly": 8.9,
        "yearly": 30.0
    }
}

def get_investor_dashboard(investor_id):
    """Obtener datos del panel de inversionista."""
    if investor_id not in INVESTORS:
        return jsonify({
            "success": False,
            "message": "Inversionista no encontrado"
        }), 404
    
    investor = INVESTORS[investor_id]
    category = INVESTOR_CATEGORIES.get(investor["category"], {})
    
    # Enriquecer datos con información de categoría
    response = {
        "success": True,
        "investor": investor,
        "category_info": category,
        "system_stats": {
            "total_investors": SYSTEM_STATS["total_investors"],
            "total_capital": SYSTEM_STATS["total_capital"],
            "performance": SYSTEM_STATS["performance"]
        }
    }
    
    return jsonify(response)

def get_all_investors():
    """Obtener lista de todos los inversionistas (solo para admin)."""
    # Verificar que usuario sea admin o super_admin
    user = session.get('user', {})
    if user.get('role') not in ['admin', 'super_admin']:
        return jsonify({
            "success": False,
            "message": "No autorizado"
        }), 403
    
    return jsonify({
        "success": True,
        "investors": list(INVESTORS.values()),
        "system_stats": SYSTEM_STATS,
        "categories": INVESTOR_CATEGORIES
    })

def get_investor_transactions(investor_id):
    """Obtener historial de transacciones de un inversionista."""
    if investor_id not in INVESTORS:
        return jsonify({
            "success": False,
            "message": "Inversionista no encontrado"
        }), 404
    
    transactions = INVESTORS[investor_id]["transactions"]
    
    return jsonify({
        "success": True,
        "transactions": transactions
    })

def get_investor_investments(investor_id):
    """Obtener inversiones actuales de un inversionista."""
    if investor_id not in INVESTORS:
        return jsonify({
            "success": False,
            "message": "Inversionista no encontrado"
        }), 404
    
    investments = INVESTORS[investor_id]["investments"]
    
    return jsonify({
        "success": True,
        "investments": investments
    })

def create_investor():
    """Crear un nuevo inversionista (solo para admin)."""
    user = session.get('user', {})
    if user.get('role') not in ['admin', 'super_admin']:
        return jsonify({
            "success": False,
            "message": "No autorizado"
        }), 403
    
    data = request.json
    required_fields = ['name', 'email', 'initial_balance', 'category']
    
    for field in required_fields:
        if field not in data:
            return jsonify({
                "success": False,
                "message": f"Campo requerido: {field}"
            }), 400
    
    # En una implementación real, aquí crearíamos el inversionista en la base de datos
    # Para la demo, simplemente devolvemos éxito
    
    return jsonify({
        "success": True,
        "message": "Inversionista creado correctamente",
        "investor": {
            "id": str(len(INVESTORS) + 1),
            "name": data["name"],
            "email": data["email"],
            "balance": data["initial_balance"],
            "category": data["category"],
            "status": "active"
        }
    })

def update_investor(investor_id):
    """Actualizar datos de un inversionista."""
    user = session.get('user', {})
    if user.get('role') not in ['admin', 'super_admin']:
        return jsonify({
            "success": False,
            "message": "No autorizado"
        }), 403
    
    if investor_id not in INVESTORS:
        return jsonify({
            "success": False,
            "message": "Inversionista no encontrado"
        }), 404
    
    data = request.json
    
    # En una implementación real, aquí actualizaríamos el inversionista en la base de datos
    # Para la demo, simplemente devolvemos éxito
    
    return jsonify({
        "success": True,
        "message": "Inversionista actualizado correctamente"
    })

def get_current_investor():
    """Obtener datos del inversionista actual basado en la sesión."""
    user = session.get('user', {})
    user_id = user.get('username')
    
    # Buscar inversionista por user_id
    investor = next((inv for inv in INVESTORS.values() if inv["user_id"] == user_id), None)
    
    if not investor:
        return jsonify({
            "success": False,
            "message": "Inversionista no encontrado para el usuario actual"
        }), 404
    
    category = INVESTOR_CATEGORIES.get(investor["category"], {})
    
    return jsonify({
        "success": True,
        "investor": investor,
        "category_info": category
    })

def register_investor_routes(app):
    """Registrar rutas de inversionistas en la aplicación Flask."""
    app.add_url_rule('/api/investor/dashboard/<investor_id>', 'get_investor_dashboard', get_investor_dashboard)
    app.add_url_rule('/api/investors', 'get_all_investors', get_all_investors)
    app.add_url_rule('/api/investor/transactions/<investor_id>', 'get_investor_transactions', get_investor_transactions)
    app.add_url_rule('/api/investor/investments/<investor_id>', 'get_investor_investments', get_investor_investments)
    app.add_url_rule('/api/investor', 'create_investor', create_investor, methods=['POST'])
    app.add_url_rule('/api/investor/<investor_id>', 'update_investor', update_investor, methods=['PUT'])
    app.add_url_rule('/api/investor/current', 'get_current_investor', get_current_investor)