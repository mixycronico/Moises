"""
Módulo REST API para el sistema Genesis.

Este módulo implementa una API REST para comunicación con sistemas externos
y acceso a las funcionalidades del sistema Genesis.
"""

import os
import logging
import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable

# FastAPI para la API REST
from fastapi import FastAPI, Depends, HTTPException, Security, status, Request, Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Importaciones del sistema Genesis
from genesis.core.base import Component
from genesis.core.engine import Engine
from genesis.utils.helpers import generate_id, format_timestamp
from genesis.security.manager import SecurityManager
from genesis.utils.log_manager import get_logger

# Definir logger
logger = get_logger("api")

# Configurar esquema OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
api_key_header = APIKeyHeader(name="X-API-Key")

# Modelos para la API
class Token(BaseModel):
    """Modelo para token de acceso."""
    access_token: str
    token_type: str = "bearer"
    expires_at: int


class UserCredentials(BaseModel):
    """Modelo para credenciales de usuario."""
    username: str
    password: str


class ApiKeyRequest(BaseModel):
    """Modelo para solicitud de API key."""
    name: str
    permissions: List[str] = Field(default_factory=list)
    expires_in_days: Optional[int] = None


class ApiKeyResponse(BaseModel):
    """Modelo para respuesta de API key."""
    api_key: str
    api_secret: str
    name: str
    permissions: List[str]
    created_at: str
    expires_at: Optional[str] = None


class TradingPair(BaseModel):
    """Modelo para par de trading."""
    symbol: str
    base_asset: str
    quote_asset: str
    min_qty: float
    max_qty: float
    price_precision: int
    qty_precision: int
    min_notional: float
    status: str = "active"


class MarketData(BaseModel):
    """Modelo para datos de mercado."""
    symbol: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    num_trades: Optional[int] = None
    taker_buy_volume: Optional[float] = None
    taker_sell_volume: Optional[float] = None


class TradeSignal(BaseModel):
    """Modelo para señal de trading."""
    symbol: str
    timestamp: str
    strategy: str
    signal_type: str
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    timeframe: str
    confidence: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TradeRequest(BaseModel):
    """Modelo para solicitud de operación."""
    symbol: str
    side: str
    order_type: str = "market"
    quantity: Optional[float] = None
    amount: Optional[float] = None
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    strategy_id: Optional[str] = None
    reduce_only: bool = False


class TradeResponse(BaseModel):
    """Modelo para respuesta de operación."""
    trade_id: str
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: float
    status: str
    created_at: str
    fee: Optional[float] = None
    fee_currency: Optional[str] = None


class AccountBalance(BaseModel):
    """Modelo para saldo de cuenta."""
    asset: str
    free: float
    locked: float
    total: float


class ErrorResponse(BaseModel):
    """Modelo para respuesta de error."""
    error: str
    code: str
    detail: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class BacktestRequest(BaseModel):
    """Modelo para solicitud de backtest."""
    strategy: str
    symbol: str
    timeframe: str = "1h"
    start_date: str
    end_date: str
    initial_capital: float = 10000.0
    params: Dict[str, Any] = Field(default_factory=dict)


class BacktestResponse(BaseModel):
    """Modelo para respuesta de backtest."""
    backtest_id: str
    strategy: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    profit_loss: float
    profit_loss_pct: float
    max_drawdown: float
    win_rate: float
    sharpe_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int


class RestAPI(Component):
    """
    Implementación de API REST para el sistema Genesis.
    
    Esta clase proporciona endpoints para interactuar con el sistema
    a través de una API REST, incluyendo autenticación, operaciones de trading,
    consulta de datos de mercado, y más.
    """
    
    def __init__(
        self,
        name: str = "rest_api",
        host: str = "0.0.0.0",
        port: int = 5000,
        enable_cors: bool = True,
        enable_docs: bool = True,
        log_requests: bool = True,
        api_prefix: str = "/api/v1"
    ):
        """
        Inicializar la API REST.
        
        Args:
            name: Nombre del componente
            host: Host para el servidor
            port: Puerto para el servidor
            enable_cors: Habilitar CORS
            enable_docs: Habilitar documentación (Swagger/ReDoc)
            log_requests: Registrar todas las solicitudes
            api_prefix: Prefijo para las rutas de la API
        """
        super().__init__(name)
        self.host = host
        self.port = port
        self.enable_cors = enable_cors
        self.enable_docs = enable_docs
        self.log_requests = log_requests
        self.api_prefix = api_prefix
        
        # Componentes del sistema
        self.engine = None
        self.security_manager = None
        
        # Crear aplicación FastAPI
        self.app = FastAPI(
            title="Genesis Trading API",
            description="API para el sistema de trading algorítmico Genesis",
            version="1.0.0",
            docs_url="/docs" if enable_docs else None,
            redoc_url="/redoc" if enable_docs else None
        )
        
        # Configurar CORS si está habilitado
        if enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
        # Inicializar rutas
        self._init_routes()
        
    async def start(self) -> None:
        """Iniciar la API REST."""
        await super().start()
        
        # Obtener referencia al motor y componentes
        self.engine = self.context.get("engine")
        if not self.engine:
            logger.error("No se encontró el motor en el contexto")
            return
            
        self.security_manager = self.context.get("security_manager")
        if not self.security_manager:
            logger.error("No se encontró el gestor de seguridad en el contexto")
            return
            
        logger.info(f"API REST iniciada en http://{self.host}:{self.port}")
        
        # No iniciamos el servidor aquí. Eso lo hará el método run()
        
    async def stop(self) -> None:
        """Detener la API REST."""
        await super().stop()
        logger.info("API REST detenida")
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente origen
        """
        pass
        
    def _init_routes(self) -> None:
        """Inicializar las rutas de la API."""
        app = self.app
        prefix = self.api_prefix
        
        # Middleware para logging
        @app.middleware("http")
        async def log_requests(request: Request, call_next):
            if self.log_requests:
                start_time = time.time()
                method = request.method
                url = str(request.url)
                try:
                    response = await call_next(request)
                    process_time = time.time() - start_time
                    status_code = response.status_code
                    logger.info(f"{method} {url} {status_code} - {process_time:.4f}s")
                    return response
                except Exception as e:
                    process_time = time.time() - start_time
                    logger.error(f"{method} {url} ERROR - {process_time:.4f}s - {str(e)}")
                    raise
            else:
                return await call_next(request)
                
        # Rutas de autenticación
        @app.post(f"{prefix}/auth/token", response_model=Token, tags=["Autenticación"])
        async def login(form_data: OAuth2PasswordRequestForm = Depends()):
            """
            Obtener token de acceso con credenciales.
            """
            if not self.security_manager:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Servicio de autenticación no disponible"
                )
                
            try:
                result = await self.security_manager.authenticate_user(form_data.username, form_data.password)
                if not result or not result.get("success"):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Credenciales incorrectas",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                    
                return {
                    "access_token": result.get("token"),
                    "token_type": "bearer",
                    "expires_at": result.get("expires_at")
                }
            except Exception as e:
                logger.error(f"Error en autenticación: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error en autenticación: {str(e)}"
                )
                
        @app.post(f"{prefix}/auth/apikey", response_model=ApiKeyResponse, tags=["Autenticación"])
        async def create_api_key(
            request: ApiKeyRequest,
            token: str = Depends(oauth2_scheme)
        ):
            """
            Crear una nueva clave API.
            """
            if not self.security_manager:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Servicio de seguridad no disponible"
                )
                
            try:
                # Verificar token
                user = await self._get_current_user(token)
                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Token inválido o expirado"
                    )
                    
                # Crear API key
                result = await self.security_manager.create_api_key(
                    user_id=user.get("id"),
                    name=request.name,
                    permissions=request.permissions,
                    expires_in_days=request.expires_in_days
                )
                
                if not result or not result.get("success"):
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Error al crear la clave API"
                    )
                    
                return {
                    "api_key": result.get("api_key"),
                    "api_secret": result.get("api_secret"),
                    "name": request.name,
                    "permissions": request.permissions,
                    "created_at": datetime.now().isoformat(),
                    "expires_at": result.get("expires_at")
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error al crear API key: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error al crear API key: {str(e)}"
                )
                
        # Rutas de mercado
        @app.get(f"{prefix}/market/pairs", tags=["Mercado"])
        async def get_trading_pairs(
            exchange: Optional[str] = None,
            token: str = Security(oauth2_scheme)
        ):
            """
            Obtener pares de trading disponibles.
            """
            await self._validate_token(token)
            
            if not self.engine or not hasattr(self.engine, "exchange_manager"):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Servicio de exchange no disponible"
                )
                
            try:
                exchange_manager = self.engine.exchange_manager
                if exchange:
                    result = await exchange_manager.get_trading_pairs(exchange)
                else:
                    result = await exchange_manager.get_all_trading_pairs()
                    
                return {"pairs": result}
                
            except Exception as e:
                logger.error(f"Error al obtener pares de trading: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error al obtener pares de trading: {str(e)}"
                )
                
        @app.get(f"{prefix}/market/data", tags=["Mercado"])
        async def get_market_data(
            symbol: str,
            timeframe: str = "1h",
            limit: int = 100,
            start_time: Optional[str] = None,
            end_time: Optional[str] = None,
            token: str = Security(oauth2_scheme)
        ):
            """
            Obtener datos históricos de mercado.
            """
            await self._validate_token(token)
            
            if not self.engine or not hasattr(self.engine, "market_data"):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Servicio de datos de mercado no disponible"
                )
                
            try:
                market_data = self.engine.market_data
                
                # Convertir fechas si se proporcionan
                start_datetime = None
                end_datetime = None
                
                if start_time:
                    start_datetime = datetime.fromisoformat(start_time)
                if end_time:
                    end_datetime = datetime.fromisoformat(end_time)
                    
                # Obtener datos
                result = await market_data.fetch_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                    start_time=start_datetime,
                    end_time=end_datetime
                )
                
                return {"data": result}
                
            except Exception as e:
                logger.error(f"Error al obtener datos de mercado: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error al obtener datos de mercado: {str(e)}"
                )
                
        @app.get(f"{prefix}/market/ticker", tags=["Mercado"])
        async def get_ticker(
            symbol: Optional[str] = None,
            token: str = Security(oauth2_scheme)
        ):
            """
            Obtener ticker de mercado.
            """
            await self._validate_token(token)
            
            if not self.engine or not hasattr(self.engine, "exchange_manager"):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Servicio de exchange no disponible"
                )
                
            try:
                exchange_manager = self.engine.exchange_manager
                if symbol:
                    result = await exchange_manager.get_ticker(symbol)
                else:
                    result = await exchange_manager.get_all_tickers()
                    
                return {"tickers": result}
                
            except Exception as e:
                logger.error(f"Error al obtener ticker: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error al obtener ticker: {str(e)}"
                )
                
        # Rutas de trading
        @app.post(f"{prefix}/trading/order", response_model=TradeResponse, tags=["Trading"])
        async def create_order(
            trade_request: TradeRequest,
            token: str = Security(oauth2_scheme)
        ):
            """
            Crear una nueva orden de trading.
            """
            user = await self._validate_token(token)
            
            if not self.engine or not hasattr(self.engine, "exchange_manager"):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Servicio de exchange no disponible"
                )
                
            try:
                exchange_manager = self.engine.exchange_manager
                
                # Validar solicitud
                if not trade_request.quantity and not trade_request.amount:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Debe especificar quantity o amount"
                    )
                    
                # Procesar parámetros adicionales según el tipo de orden
                params = {}
                if trade_request.reduce_only:
                    params["reduce_only"] = True
                    
                if trade_request.order_type == "limit" and not trade_request.price:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Las órdenes limit requieren un precio"
                    )
                    
                if trade_request.order_type == "stop" and not trade_request.stop_price:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Las órdenes stop requieren un stop_price"
                    )
                    
                # Crear la orden
                result = await exchange_manager.create_order(
                    symbol=trade_request.symbol,
                    order_type=trade_request.order_type,
                    side=trade_request.side,
                    amount=trade_request.amount,
                    quantity=trade_request.quantity,
                    price=trade_request.price,
                    params=params
                )
                
                if not result or not result.get("success"):
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Error al crear orden: {result.get('error', 'Error desconocido')}"
                    )
                    
                # Formatear respuesta
                order_data = result.get("order", {})
                response = {
                    "trade_id": generate_id(),
                    "order_id": order_data.get("id", ""),
                    "symbol": trade_request.symbol,
                    "side": trade_request.side,
                    "order_type": trade_request.order_type,
                    "quantity": order_data.get("amount", 0.0),
                    "price": order_data.get("price", 0.0),
                    "status": order_data.get("status", "unknown"),
                    "created_at": format_timestamp(datetime.now()),
                    "fee": order_data.get("fee", None),
                    "fee_currency": order_data.get("fee_currency", None)
                }
                
                # Registrar la operación
                await self.engine.event_bus.emit("trade.created", {
                    "user_id": user.get("id"),
                    "trade_id": response["trade_id"],
                    "order_id": response["order_id"],
                    "symbol": response["symbol"],
                    "side": response["side"],
                    "order_type": response["order_type"],
                    "quantity": response["quantity"],
                    "price": response["price"],
                    "status": response["status"],
                    "strategy_id": trade_request.strategy_id
                })
                
                return response
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error al crear orden: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error al crear orden: {str(e)}"
                )
                
        @app.get(f"{prefix}/trading/orders", tags=["Trading"])
        async def get_orders(
            symbol: Optional[str] = None,
            status: Optional[str] = None,
            limit: int = 50,
            token: str = Security(oauth2_scheme)
        ):
            """
            Obtener órdenes del usuario.
            """
            user = await self._validate_token(token)
            
            if not self.engine or not hasattr(self.engine, "exchange_manager"):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Servicio de exchange no disponible"
                )
                
            try:
                exchange_manager = self.engine.exchange_manager
                
                # Obtener órdenes
                params = {}
                if status:
                    params["status"] = status
                    
                result = await exchange_manager.fetch_orders(
                    symbol=symbol,
                    limit=limit,
                    params=params
                )
                
                return {"orders": result}
                
            except Exception as e:
                logger.error(f"Error al obtener órdenes: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error al obtener órdenes: {str(e)}"
                )
                
        @app.get(f"{prefix}/trading/balance", tags=["Trading"])
        async def get_balance(
            token: str = Security(oauth2_scheme)
        ):
            """
            Obtener saldo de la cuenta.
            """
            user = await self._validate_token(token)
            
            if not self.engine or not hasattr(self.engine, "exchange_manager"):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Servicio de exchange no disponible"
                )
                
            try:
                exchange_manager = self.engine.exchange_manager
                
                # Obtener saldo
                result = await exchange_manager.fetch_balance()
                
                # Formatear respuesta
                formatted_balances = []
                for asset, data in result.get("balances", {}).items():
                    if data.get("total", 0) > 0:
                        formatted_balances.append({
                            "asset": asset,
                            "free": data.get("free", 0),
                            "locked": data.get("used", 0),
                            "total": data.get("total", 0)
                        })
                        
                return {"balances": formatted_balances}
                
            except Exception as e:
                logger.error(f"Error al obtener saldo: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error al obtener saldo: {str(e)}"
                )
                
        # Rutas de estrategias
        @app.get(f"{prefix}/strategies", tags=["Estrategias"])
        async def get_strategies(
            token: str = Security(oauth2_scheme)
        ):
            """
            Obtener estrategias disponibles.
            """
            await self._validate_token(token)
            
            if not self.engine or not hasattr(self.engine, "strategy_manager"):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Servicio de estrategias no disponible"
                )
                
            try:
                strategy_manager = self.engine.strategy_manager
                
                # Obtener estrategias
                strategies = strategy_manager.get_available_strategies()
                
                return {"strategies": strategies}
                
            except Exception as e:
                logger.error(f"Error al obtener estrategias: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error al obtener estrategias: {str(e)}"
                )
                
        @app.get(f"{prefix}/strategies/{strategy_id}/performance", tags=["Estrategias"])
        async def get_strategy_performance(
            strategy_id: str,
            timeframe: str = "1d",
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            token: str = Security(oauth2_scheme)
        ):
            """
            Obtener rendimiento de una estrategia.
            """
            await self._validate_token(token)
            
            if not self.engine or not hasattr(self.engine, "performance_analyzer"):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Servicio de análisis de rendimiento no disponible"
                )
                
            try:
                performance_analyzer = self.engine.performance_analyzer
                
                # Convertir fechas si se proporcionan
                start_datetime = None
                end_datetime = None
                
                if start_date:
                    start_datetime = datetime.fromisoformat(start_date)
                else:
                    # Por defecto, último mes
                    start_datetime = datetime.now() - timedelta(days=30)
                    
                if end_date:
                    end_datetime = datetime.fromisoformat(end_date)
                    
                # Obtener rendimiento
                result = await performance_analyzer.get_strategy_performance(
                    strategy_id=strategy_id,
                    start_date=start_datetime,
                    end_date=end_datetime,
                    timeframe=timeframe
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Error al obtener rendimiento de estrategia: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error al obtener rendimiento de estrategia: {str(e)}"
                )
                
        # Rutas de backtest
        @app.post(f"{prefix}/backtest", response_model=BacktestResponse, tags=["Backtest"])
        async def run_backtest(
            request: BacktestRequest,
            token: str = Security(oauth2_scheme)
        ):
            """
            Ejecutar un backtest de estrategia.
            """
            await self._validate_token(token)
            
            if not self.engine or not hasattr(self.engine, "backtest_engine"):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Servicio de backtest no disponible"
                )
                
            try:
                backtest_engine = self.engine.backtest_engine
                
                # Ejecutar backtest
                result = await backtest_engine.run_backtest(
                    strategy_name=request.strategy,
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    initial_capital=request.initial_capital,
                    params=request.params
                )
                
                # Formatear respuesta
                return {
                    "backtest_id": result.get("backtest_id", ""),
                    "strategy": request.strategy,
                    "symbol": request.symbol,
                    "timeframe": request.timeframe,
                    "start_date": request.start_date,
                    "end_date": request.end_date,
                    "initial_capital": request.initial_capital,
                    "final_capital": result.get("final_capital", 0.0),
                    "profit_loss": result.get("net_profit", 0.0),
                    "profit_loss_pct": result.get("profit_pct", 0.0),
                    "max_drawdown": result.get("max_drawdown", 0.0),
                    "win_rate": result.get("win_rate", 0.0),
                    "sharpe_ratio": result.get("sharpe_ratio", 0.0),
                    "total_trades": result.get("total_trades", 0),
                    "winning_trades": result.get("winning_trades", 0),
                    "losing_trades": result.get("losing_trades", 0)
                }
                
            except Exception as e:
                logger.error(f"Error al ejecutar backtest: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error al ejecutar backtest: {str(e)}"
                )
                
        @app.get(f"{prefix}/backtest/{backtest_id}", tags=["Backtest"])
        async def get_backtest(
            backtest_id: str,
            token: str = Security(oauth2_scheme)
        ):
            """
            Obtener resultados de un backtest.
            """
            await self._validate_token(token)
            
            if not self.engine or not hasattr(self.engine, "backtest_engine"):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Servicio de backtest no disponible"
                )
                
            try:
                backtest_engine = self.engine.backtest_engine
                
                # Obtener resultados
                result = await backtest_engine.get_backtest_result(backtest_id)
                
                if not result:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Backtest {backtest_id} no encontrado"
                    )
                    
                return result
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error al obtener resultados de backtest: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error al obtener resultados de backtest: {str(e)}"
                )
                
        # Rutas administrativas
        @app.get(f"{prefix}/admin/status", tags=["Admin"])
        async def get_system_status(
            token: str = Security(oauth2_scheme)
        ):
            """
            Obtener estado del sistema.
            """
            user = await self._validate_token(token, require_admin=True)
            
            if not self.engine:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Motor no disponible"
                )
                
            try:
                # Obtener estado de componentes
                components = {}
                for component_name, component in self.engine.components.items():
                    components[component_name] = {
                        "name": component.name,
                        "status": "running" if component.is_running else "stopped",
                        "started_at": format_timestamp(component.started_at) if component.started_at else None
                    }
                    
                return {
                    "uptime": time.time() - self.engine.started_at if hasattr(self.engine, "started_at") else 0,
                    "version": "1.0.0",
                    "components": components,
                    "memory_usage_mb": None,  # Implementar si es necesario
                    "system_load": None       # Implementar si es necesario
                }
                
            except Exception as e:
                logger.error(f"Error al obtener estado del sistema: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error al obtener estado del sistema: {str(e)}"
                )
                
        @app.get(f"{prefix}/admin/logs", tags=["Admin"])
        async def get_logs(
            level: Optional[str] = None,
            component: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            limit: int = 100,
            token: str = Security(oauth2_scheme)
        ):
            """
            Obtener logs del sistema.
            """
            await self._validate_token(token, require_admin=True)
            
            if not self.engine or not hasattr(self.engine, "log_manager"):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Servicio de logs no disponible"
                )
                
            try:
                from genesis.utils.log_manager import query_logs
                
                # Obtener logs
                logs = query_logs(
                    start_date=start_date,
                    end_date=end_date,
                    level=level,
                    component=component,
                    limit=limit
                )
                
                return {"logs": logs}
                
            except Exception as e:
                logger.error(f"Error al obtener logs: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error al obtener logs: {str(e)}"
                )
                
    async def _get_current_user(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Obtener usuario actual a partir del token.
        
        Args:
            token: Token JWT
            
        Returns:
            Datos del usuario o None
        """
        if not self.security_manager:
            return None
            
        try:
            user_data = await self.security_manager.validate_token(token)
            if not user_data or not user_data.get("success"):
                return None
                
            return user_data.get("user")
                
        except Exception as e:
            logger.error(f"Error al validar token: {str(e)}")
            return None
            
    async def _validate_token(self, token: str, require_admin: bool = False) -> Dict[str, Any]:
        """
        Validar token y devolver usuario.
        
        Args:
            token: Token JWT
            require_admin: Si se requiere rol de administrador
            
        Returns:
            Datos del usuario
            
        Raises:
            HTTPException: Si el token no es válido
        """
        user = await self._get_current_user(token)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token inválido o expirado",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        if require_admin and user.get("role") != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Se requiere rol de administrador"
            )
            
        return user
        
    def run(self) -> None:
        """
        Iniciar el servidor HTTP.
        
        Este método bloquea el hilo actual.
        """
        import uvicorn
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port
        )