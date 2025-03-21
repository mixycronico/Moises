"""
API REST para el sistema Genesis.

Este módulo proporciona una API REST para interacción con el sistema
desde aplicaciones externas, permitiendo consultar datos, ejecutar
operaciones y administrar el sistema.
"""

# Función de inicialización para integrarse con Flask
def init_api(flask_app):
    """
    Inicializa la API REST con una aplicación Flask existente.
    
    Args:
        flask_app: Aplicación Flask a la que se montará la API
    """
    from fastapi import FastAPI
    from fastapi.middleware.wsgi import WSGIMiddleware
    
    # Obtener la instancia de la aplicación FastAPI
    app = get_app()
    
    # Montar FastAPI en Flask bajo /api/v1
    flask_app.wsgi_app = WSGIMiddleware(app)
    
    return app


def get_app():
    """
    Obtener instancia de la aplicación FastAPI.
    Útil para testing o montaje en otro servidor.
    
    Returns:
        Aplicación FastAPI configurada
    """
    # La aplicación FastAPI ya está configurada en este módulo
    return app

import os
import json
import logging
import asyncio
import datetime
from typing import List, Dict, Any, Optional
from jose import JWTError, jwt
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from uuid import uuid4

from genesis.utils.logger import setup_logging
from genesis.utils.log_manager import get_logger, query_logs, get_log_stats


# Modelos Pydantic para la API
class Token(BaseModel):
    """Modelo para token de autenticación."""
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Modelo para datos de token."""
    username: Optional[str] = None


class User(BaseModel):
    """Modelo para usuario."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None


class UserInDB(User):
    """Modelo para usuario en base de datos."""
    hashed_password: str


class StrategyRequest(BaseModel):
    """Modelo para solicitud de ejecución de estrategia."""
    strategy_name: str
    symbol: str
    params: Dict[str, Any] = Field(default_factory=dict)


class PerformanceRequest(BaseModel):
    """Modelo para solicitud de rendimiento."""
    strategy_name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class LogQueryRequest(BaseModel):
    """Modelo para consulta de logs."""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    level: Optional[str] = None
    component: Optional[str] = None
    limit: int = 100
    offset: int = 0


# Configuración de la API
app = FastAPI(
    title="Genesis Trading API",
    description="API REST para el sistema de trading Genesis",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configurar según necesidades de seguridad
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de autenticación
SECRET_KEY = os.getenv("API_SECRET_KEY", "development_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Logger
logger = get_logger("api")

# Sistema de autenticación simple para ejemplo
# En un sistema real, esto se conectaría a una base de datos
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Administrator",
        "email": "admin@example.com",
        "hashed_password": "fakehashedsecret",
        "disabled": False,
    },
    "user": {
        "username": "user",
        "full_name": "Regular User",
        "email": "user@example.com",
        "hashed_password": "fakehashedsecret",
        "disabled": False,
    },
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_password(plain_password, hashed_password):
    """Verificar contraseña."""
    # Simulación simple, en producción usar algoritmos de hash
    return plain_password == "secret"


def get_user(db, username: str):
    """Obtener usuario de la base de datos."""
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def authenticate_user(fake_db, username: str, password: str):
    """Autenticar usuario."""
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None):
    """Crear token de acceso JWT."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Obtener usuario actual a partir del token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Credenciales inválidas",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    """Verificar que el usuario esté activo."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Usuario inactivo")
    return current_user


# Endpoints de autenticación
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Endpoint para obtener token de acceso."""
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario o contraseña incorrectos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    logger.info(f"Usuario {user.username} ha iniciado sesión", extra={
        "correlation_id": str(uuid4()),
        "user": user.username,
        "action": "login"
    })
    return {"access_token": access_token, "token_type": "bearer"}


# Endpoints de sistema
@app.get("/health")
async def health_check():
    """Verificar estado del sistema."""
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "api_version": "1.0.0"
    }


@app.get("/system/status")
async def system_status(current_user: User = Depends(get_current_active_user)):
    """Obtener estado del sistema."""
    # Aquí se conectaría con componentes reales del sistema
    return {
        "status": "running",
        "components": {
            "market_data": "active",
            "strategy_orchestrator": "active",
            "risk_manager": "active"
        },
        "memory_usage_mb": 128.5,  # Simulado
        "uptime_seconds": 3600,     # Simulado
        "timestamp": datetime.datetime.utcnow().isoformat()
    }


# Endpoints de estrategias
@app.post("/strategies/execute")
async def execute_strategy(
    request: StrategyRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Ejecutar una estrategia de trading."""
    logger.info(f"Ejecutando estrategia {request.strategy_name} para {request.symbol}", extra={
        "correlation_id": str(uuid4()),
        "user": current_user.username,
        "action": "execute_strategy",
        "strategy": request.strategy_name,
        "symbol": request.symbol
    })
    
    # Aquí se conectaría con el componente de estrategias
    # Simulación de resultado
    return {
        "strategy": request.strategy_name,
        "symbol": request.symbol,
        "signal": "buy",  # Simulado
        "confidence": 0.85,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "execution_id": str(uuid4())
    }


@app.get("/strategies/list")
async def list_strategies(current_user: User = Depends(get_current_active_user)):
    """Listar estrategias disponibles."""
    # Simulación de lista de estrategias
    return {
        "strategies": [
            {"name": "ma_crossover", "description": "Moving Average Crossover"},
            {"name": "rsi", "description": "Relative Strength Index"},
            {"name": "macd", "description": "MACD"},
            {"name": "bollinger", "description": "Bollinger Bands"}
        ]
    }


@app.post("/performance/query")
async def query_performance(
    request: PerformanceRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Consultar rendimiento de estrategias."""
    logger.info(f"Consultando rendimiento", extra={
        "correlation_id": str(uuid4()),
        "user": current_user.username,
        "action": "query_performance",
        "strategy": request.strategy_name,
        "start_date": request.start_date,
        "end_date": request.end_date
    })
    
    # Aquí se conectaría con el sistema de rendimiento
    # Simulación de respuesta
    return {
        "strategy": request.strategy_name,
        "performance": {
            "profit_pct": 8.5,
            "max_drawdown": 2.1,
            "sharpe_ratio": 1.2,
            "win_rate": 0.65
        },
        "trades": [
            {
                "timestamp": "2023-01-01T10:00:00",
                "symbol": "BTC/USDT",
                "action": "buy",
                "price": 50000.0,
                "volume": 0.1
            },
            {
                "timestamp": "2023-01-02T14:30:00",
                "symbol": "BTC/USDT",
                "action": "sell",
                "price": 52000.0,
                "volume": 0.1
            }
        ]
    }


# Endpoints de logs
@app.post("/logs/query")
async def api_query_logs(
    request: LogQueryRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Consultar logs del sistema."""
    # Verificar privilegios (sólo admin)
    if current_user.username != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tienes permiso para acceder a los logs"
        )
    
    logger.info(f"Consultando logs", extra={
        "correlation_id": str(uuid4()),
        "user": current_user.username,
        "action": "query_logs",
        "params": request.dict()
    })
    
    # Usar la función de consulta del sistema de logs
    logs = query_logs(
        start_date=request.start_date,
        end_date=request.end_date,
        level=request.level,
        component=request.component,
        limit=request.limit,
        offset=request.offset
    )
    
    return {"logs": logs, "total": len(logs)}


@app.get("/logs/stats")
async def get_log_statistics(current_user: User = Depends(get_current_active_user)):
    """Obtener estadísticas de logs."""
    # Verificar privilegios (sólo admin)
    if current_user.username != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tienes permiso para acceder a las estadísticas de logs"
        )
    
    # Obtener estadísticas
    stats = get_log_stats()
    
    return stats


# Middleware para logging de solicitudes
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware para logging de solicitudes HTTP."""
    start_time = datetime.datetime.utcnow()
    correlation_id = str(uuid4())
    
    # Extraer información de la solicitud
    path = request.url.path
    method = request.method
    client = request.client.host if request.client else "unknown"
    
    logger.info(f"{method} {path}", extra={
        "correlation_id": correlation_id,
        "method": method,
        "path": path,
        "client_ip": client,
        "action": "http_request"
    })
    
    # Procesar solicitud
    response = await call_next(request)
    
    # Calcular tiempo de procesamiento
    process_time = (datetime.datetime.utcnow() - start_time).total_seconds()
    
    logger.info(f"Completed {method} {path} in {process_time:.6f}s", extra={
        "correlation_id": correlation_id,
        "method": method,
        "path": path,
        "status_code": response.status_code,
        "processing_time": process_time,
        "action": "http_response"
    })
    
    return response


# Manejador de excepciones global
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Manejador global de excepciones."""
    correlation_id = str(uuid4())
    
    logger.error(f"Error no controlado: {str(exc)}", extra={
        "correlation_id": correlation_id,
        "method": request.method,
        "path": request.url.path,
        "error": str(exc),
        "error_type": type(exc).__name__,
        "action": "error"
    })
    
    return {
        "error": "Error interno del servidor",
        "detail": str(exc) if app.debug else "Contacte al administrador",
        "correlation_id": correlation_id
    }


# Si se ejecuta como script
if __name__ == "__main__":
    import uvicorn
    
    # Configuración para desarrollo
    uvicorn.run("genesis.api.rest:app", host="0.0.0.0", port=8000, reload=True)