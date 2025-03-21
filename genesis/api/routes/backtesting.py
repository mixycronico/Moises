"""
Rutas de API para el sistema de backtesting.

Este módulo proporciona rutas para integrar el sistema de backtesting
con la API REST principal del sistema Genesis.
"""

from fastapi import APIRouter, Request, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from typing import Dict, List, Any, Optional, Union
import os
import json

from genesis.backtesting.api import (
    get_available_strategies,
    get_available_symbols,
    run_backtest,
    optimize_strategy,
    walk_forward_analysis,
    get_task_status,
    get_tasks,
    get_results_list,
    get_result,
    delete_task,
    get_chart
)

# Crear router
router = APIRouter(prefix="/backtesting", tags=["backtesting"])

# Templates para renderizar vistas
templates = Jinja2Templates(directory="templates")


@router.get("/")
async def backtesting_home(request: Request):
    """
    Página principal del sistema de backtesting.
    
    Args:
        request: Solicitud HTTP
        
    Returns:
        Respuesta HTML
    """
    # Obtener estrategias disponibles
    strategies_result = await get_available_strategies()
    strategies = strategies_result.get("strategies", [])
    
    # Obtener símbolos disponibles
    symbols_result = await get_available_symbols()
    symbols = symbols_result.get("symbols", [])
    
    # Obtener resultados
    results_data = await get_results_list()
    results = results_data.get("results", [])
    
    # Obtener tareas activas
    tasks_data = await get_tasks(limit=5)
    tasks = tasks_data.get("tasks", [])
    
    # Renderizar template
    return templates.TemplateResponse(
        "backtesting/index.html",
        {
            "request": request,
            "strategies": strategies,
            "symbols": symbols,
            "results": results,
            "tasks": tasks
        }
    )


@router.get("/strategies")
async def get_strategies():
    """
    Obtener estrategias disponibles.
    
    Returns:
        Lista de estrategias
    """
    return await get_available_strategies()


@router.get("/symbols")
async def get_symbols():
    """
    Obtener símbolos disponibles.
    
    Returns:
        Lista de símbolos
    """
    return await get_available_symbols()


@router.post("/run")
async def start_backtest(
    background_tasks: BackgroundTasks,
    strategy_name: str,
    strategy_module: str,
    symbol: str,
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    initial_balance: float = 10000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    risk_per_trade: float = 0.02,
    params: Optional[Dict[str, Any]] = None
):
    """
    Iniciar un backtest.
    
    Args:
        background_tasks: Tareas en segundo plano
        strategy_name: Nombre de la estrategia
        strategy_module: Módulo que contiene la estrategia
        symbol: Símbolo de trading
        timeframe: Timeframe para los datos
        start_date: Fecha de inicio (ISO format)
        end_date: Fecha de fin (ISO format)
        initial_balance: Balance inicial
        commission: Comisión por operación
        slippage: Slippage por operación
        risk_per_trade: Riesgo por operación
        params: Parámetros de la estrategia
        
    Returns:
        ID de la tarea y estado
    """
    return await run_backtest(
        background_tasks,
        strategy_name,
        strategy_module,
        symbol,
        timeframe,
        start_date,
        end_date,
        initial_balance,
        commission,
        slippage,
        risk_per_trade,
        params
    )


@router.post("/optimize")
async def start_optimization(
    background_tasks: BackgroundTasks,
    strategy_name: str,
    strategy_module: str,
    symbol: str,
    timeframe: str,
    param_grid: Dict[str, List[Any]],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    initial_balance: float = 10000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    risk_per_trade: float = 0.02,
    metric: str = "profit_loss",
    n_jobs: int = -1
):
    """
    Iniciar optimización de estrategia.
    
    Args:
        background_tasks: Tareas en segundo plano
        strategy_name: Nombre de la estrategia
        strategy_module: Módulo que contiene la estrategia
        symbol: Símbolo de trading
        timeframe: Timeframe para los datos
        param_grid: Grid de parámetros a probar
        start_date: Fecha de inicio (ISO format)
        end_date: Fecha de fin (ISO format)
        initial_balance: Balance inicial
        commission: Comisión por operación
        slippage: Slippage por operación
        risk_per_trade: Riesgo por operación
        metric: Métrica a optimizar
        n_jobs: Número de jobs para paralelización
        
    Returns:
        ID de la tarea y estado
    """
    return await optimize_strategy(
        background_tasks,
        strategy_name,
        strategy_module,
        symbol,
        timeframe,
        param_grid,
        start_date,
        end_date,
        initial_balance,
        commission,
        slippage,
        risk_per_trade,
        metric,
        n_jobs
    )


@router.post("/walk-forward")
async def start_walk_forward(
    background_tasks: BackgroundTasks,
    strategy_name: str,
    strategy_module: str,
    symbol: str,
    timeframe: str,
    param_grid: Dict[str, List[Any]],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    window_size: int = 30,
    test_size: int = 10,
    step_size: int = 10,
    initial_balance: float = 10000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    risk_per_trade: float = 0.02,
    metric: str = "profit_loss"
):
    """
    Iniciar análisis Walk-Forward.
    
    Args:
        background_tasks: Tareas en segundo plano
        strategy_name: Nombre de la estrategia
        strategy_module: Módulo que contiene la estrategia
        symbol: Símbolo de trading
        timeframe: Timeframe para los datos
        param_grid: Grid de parámetros a probar
        start_date: Fecha de inicio (ISO format)
        end_date: Fecha de fin (ISO format)
        window_size: Tamaño de la ventana de entrenamiento (días)
        test_size: Tamaño de la ventana de prueba (días)
        step_size: Tamaño del paso entre ventanas (días)
        initial_balance: Balance inicial
        commission: Comisión por operación
        slippage: Slippage por operación
        risk_per_trade: Riesgo por operación
        metric: Métrica a optimizar
        
    Returns:
        ID de la tarea y estado
    """
    return await walk_forward_analysis(
        background_tasks,
        strategy_name,
        strategy_module,
        symbol,
        timeframe,
        param_grid,
        start_date,
        end_date,
        window_size,
        test_size,
        step_size,
        initial_balance,
        commission,
        slippage,
        risk_per_trade,
        metric
    )


@router.get("/task/{task_id}")
async def task_status(task_id: str):
    """
    Obtener estado de una tarea.
    
    Args:
        task_id: ID de la tarea
        
    Returns:
        Estado de la tarea
    """
    return await get_task_status(task_id)


@router.get("/tasks")
async def list_tasks(
    status: Optional[str] = None,
    type: Optional[str] = None,
    limit: int = 10
):
    """
    Obtener lista de tareas.
    
    Args:
        status: Filtrar por estado
        type: Filtrar por tipo
        limit: Límite de resultados
        
    Returns:
        Lista de tareas
    """
    return await get_tasks(status, type, limit)


@router.get("/results")
async def list_results():
    """
    Obtener lista de resultados de backtest.
    
    Returns:
        Lista de resultados
    """
    return await get_results_list()


@router.get("/result/{result_id}")
async def result_details(result_id: str):
    """
    Obtener detalles de un resultado.
    
    Args:
        result_id: ID del resultado
        
    Returns:
        Detalles del resultado
    """
    return await get_result(result_id)


@router.delete("/task/{task_id}")
async def remove_task(task_id: str):
    """
    Eliminar una tarea.
    
    Args:
        task_id: ID de la tarea
        
    Returns:
        Confirmación
    """
    return await delete_task(task_id)


@router.get("/chart/{chart_id}")
async def chart_details(chart_id: str):
    """
    Obtener detalles de un gráfico.
    
    Args:
        chart_id: ID del gráfico
        
    Returns:
        Detalles del gráfico
    """
    chart_info = await get_chart(chart_id)
    
    # Verificar si existe el archivo
    if "path" in chart_info and os.path.exists(chart_info["path"]):
        return FileResponse(chart_info["path"])
    
    raise HTTPException(status_code=404, detail="Gráfico no encontrado")


@router.get("/ui/result/{result_id}")
async def result_view(request: Request, result_id: str):
    """
    Vista de resultado de backtest.
    
    Args:
        request: Solicitud HTTP
        result_id: ID del resultado
        
    Returns:
        Respuesta HTML
    """
    try:
        # Obtener resultado
        result = await get_result(result_id)
        
        # Renderizar template
        return templates.TemplateResponse(
            "backtesting/result.html",
            {
                "request": request,
                "result": result,
                "result_id": result_id
            }
        )
    except HTTPException as e:
        # Renderizar página de error
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "error": e.detail,
                "status_code": e.status_code
            }
        )


@router.get("/ui/task/{task_id}")
async def task_view(request: Request, task_id: str):
    """
    Vista de seguimiento de tarea.
    
    Args:
        request: Solicitud HTTP
        task_id: ID de la tarea
        
    Returns:
        Respuesta HTML
    """
    try:
        # Obtener tarea
        task = await get_task_status(task_id)
        
        # Renderizar template
        return templates.TemplateResponse(
            "backtesting/task.html",
            {
                "request": request,
                "task": task,
                "task_id": task_id
            }
        )
    except HTTPException as e:
        # Renderizar página de error
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "error": e.detail,
                "status_code": e.status_code
            }
        )


@router.get("/ui/new")
async def new_backtest_view(request: Request):
    """
    Vista para nuevo backtest.
    
    Args:
        request: Solicitud HTTP
        
    Returns:
        Respuesta HTML
    """
    # Obtener estrategias disponibles
    strategies_result = await get_available_strategies()
    strategies = strategies_result.get("strategies", [])
    
    # Obtener símbolos disponibles
    symbols_result = await get_available_symbols()
    symbols = symbols_result.get("symbols", [])
    
    # Renderizar template
    return templates.TemplateResponse(
        "backtesting/new.html",
        {
            "request": request,
            "strategies": strategies,
            "symbols": symbols
        }
    )


@router.get("/ui/optimize")
async def new_optimization_view(request: Request):
    """
    Vista para nueva optimización.
    
    Args:
        request: Solicitud HTTP
        
    Returns:
        Respuesta HTML
    """
    # Obtener estrategias disponibles
    strategies_result = await get_available_strategies()
    strategies = strategies_result.get("strategies", [])
    
    # Obtener símbolos disponibles
    symbols_result = await get_available_symbols()
    symbols = symbols_result.get("symbols", [])
    
    # Renderizar template
    return templates.TemplateResponse(
        "backtesting/optimize.html",
        {
            "request": request,
            "strategies": strategies,
            "symbols": symbols
        }
    )


@router.get("/ui/walk-forward")
async def new_wfa_view(request: Request):
    """
    Vista para nuevo análisis Walk-Forward.
    
    Args:
        request: Solicitud HTTP
        
    Returns:
        Respuesta HTML
    """
    # Obtener estrategias disponibles
    strategies_result = await get_available_strategies()
    strategies = strategies_result.get("strategies", [])
    
    # Obtener símbolos disponibles
    symbols_result = await get_available_symbols()
    symbols = symbols_result.get("symbols", [])
    
    # Renderizar template
    return templates.TemplateResponse(
        "backtesting/walk_forward.html",
        {
            "request": request,
            "strategies": strategies,
            "symbols": symbols
        }
    )