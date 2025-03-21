"""
API para el sistema de backtesting.

Este módulo proporciona endpoints de API para interactuar con el motor
de backtesting, permitiendo ejecutar backtests, optimizar estrategias,
y obtener resultados.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Type
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
import importlib
import inspect
import os
import json
import asyncio

from genesis.backtesting.engine import BacktestEngine, BacktestResult
from genesis.strategies.base import Strategy


# Crear router de FastAPI
router = APIRouter(prefix="/api/backtest", tags=["backtesting"])

# Obtener instancia del motor de backtesting
backtest_engine = BacktestEngine()

# Diccionario para almacenar tareas en ejecución
active_tasks: Dict[str, Dict[str, Any]] = {}


@router.get("/strategies")
async def get_available_strategies() -> Dict[str, Any]:
    """
    Obtener lista de estrategias disponibles.
    
    Returns:
        Lista de estrategias con sus parámetros
    """
    strategies = []
    
    try:
        # Importar módulo de estrategias
        strategies_module = importlib.import_module("genesis.strategies")
        
        # Obtener directorio del módulo
        module_dir = os.path.dirname(strategies_module.__file__)
        
        # Buscar archivos Python en el directorio
        strategy_files = [f[:-3] for f in os.listdir(module_dir) 
                         if f.endswith('.py') and f != '__init__.py' and f != 'base.py']
        
        # Importar cada módulo y buscar estrategias
        for file_name in strategy_files:
            try:
                module = importlib.import_module(f"genesis.strategies.{file_name}")
                
                # Buscar clases que heredan de Strategy
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, Strategy) and 
                        obj != Strategy and 
                        not inspect.isabstract(obj)):
                        
                        # Obtener parámetros del __init__
                        params = {}
                        if hasattr(obj, '__init__'):
                            signature = inspect.signature(obj.__init__)
                            for param_name, param in signature.parameters.items():
                                if param_name not in ('self', 'args', 'kwargs'):
                                    param_type = str(param.annotation).replace("<class '", "").replace("'>", "")
                                    default_value = param.default if param.default != inspect.Parameter.empty else None
                                    
                                    params[param_name] = {
                                        "type": param_type,
                                        "default": default_value
                                    }
                        
                        # Añadir estrategia
                        strategies.append({
                            "name": name,
                            "module": f"genesis.strategies.{file_name}",
                            "description": obj.__doc__,
                            "parameters": params
                        })
            
            except Exception as e:
                print(f"Error al cargar módulo {file_name}: {e}")
    
    except Exception as e:
        print(f"Error al buscar estrategias: {e}")
    
    return {"strategies": strategies}


@router.get("/symbols")
async def get_available_symbols() -> Dict[str, Any]:
    """
    Obtener lista de símbolos disponibles para backtest.
    
    Returns:
        Lista de símbolos con información adicional
    """
    symbols = []
    
    # Obtener directorio de datos
    data_dir = "data/historical"
    
    # Verificar si existe
    if not os.path.exists(data_dir):
        return {"symbols": symbols}
    
    # Buscar archivos CSV
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv'):
            # Obtener símbolo y timeframe
            parts = file_name[:-4].split('_')
            
            if len(parts) >= 2:
                symbol = parts[0]
                timeframe = parts[1]
                
                # Verificar si ya existe
                existing = next((s for s in symbols if s["symbol"] == symbol), None)
                
                if existing:
                    existing["timeframes"].append(timeframe)
                else:
                    symbols.append({
                        "symbol": symbol,
                        "timeframes": [timeframe]
                    })
    
    return {"symbols": symbols}


@router.post("/run")
async def run_backtest(
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
) -> Dict[str, Any]:
    """
    Ejecutar un backtest para una estrategia.
    
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
    # Generar ID de tarea
    task_id = f"backtest_{int(datetime.now().timestamp())}"
    
    try:
        # Importar módulo y clase de estrategia
        module = importlib.import_module(strategy_module)
        strategy_class = getattr(module, strategy_name)
        
        # Parámetros de la estrategia
        if params is None:
            params = {}
        
        # Registrar tarea
        active_tasks[task_id] = {
            "id": task_id,
            "type": "backtest",
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "params": {
                "strategy_name": strategy_name,
                "strategy_module": strategy_module,
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "initial_balance": initial_balance,
                "commission": commission,
                "slippage": slippage,
                "risk_per_trade": risk_per_trade,
                "strategy_params": params
            }
        }
        
        # Iniciar tarea en segundo plano
        background_tasks.add_task(
            _run_backtest_task,
            task_id,
            strategy_class,
            params,
            symbol,
            timeframe,
            start_date,
            end_date,
            initial_balance,
            commission,
            slippage,
            risk_per_trade
        )
        
        return {
            "task_id": task_id,
            "status": "accepted",
            "message": "Tarea de backtest iniciada"
        }
    
    except ImportError:
        raise HTTPException(status_code=404, detail=f"Módulo de estrategia no encontrado: {strategy_module}")
    
    except AttributeError:
        raise HTTPException(status_code=404, detail=f"Estrategia no encontrada: {strategy_name}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al iniciar backtest: {str(e)}")


async def _run_backtest_task(
    task_id: str,
    strategy_class: Type[Strategy],
    params: Dict[str, Any],
    symbol: str,
    timeframe: str,
    start_date: Optional[str],
    end_date: Optional[str],
    initial_balance: float,
    commission: float,
    slippage: float,
    risk_per_trade: float
) -> None:
    """
    Ejecutar tarea de backtest en segundo plano.
    
    Args:
        task_id: ID de la tarea
        strategy_class: Clase de la estrategia
        params: Parámetros de la estrategia
        symbol: Símbolo de trading
        timeframe: Timeframe para los datos
        start_date: Fecha de inicio
        end_date: Fecha de fin
        initial_balance: Balance inicial
        commission: Comisión por operación
        slippage: Slippage por operación
        risk_per_trade: Riesgo por operación
    """
    try:
        # Actualizar estado
        active_tasks[task_id]["status"] = "running"
        
        # Ejecutar backtest
        result = await backtest_engine.run_backtest(
            strategy_class,
            params,
            symbol,
            timeframe,
            start_date,
            end_date,
            initial_balance,
            commission,
            slippage,
            risk_per_trade
        )
        
        # Actualizar estado
        if result:
            active_tasks[task_id]["status"] = "completed"
            active_tasks[task_id]["result"] = result.to_dict()
            active_tasks[task_id]["completed_at"] = datetime.now().isoformat()
        else:
            active_tasks[task_id]["status"] = "failed"
            active_tasks[task_id]["error"] = "No se pudo completar el backtest"
    
    except Exception as e:
        active_tasks[task_id]["status"] = "failed"
        active_tasks[task_id]["error"] = str(e)


@router.post("/optimize")
async def optimize_strategy(
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
) -> Dict[str, Any]:
    """
    Optimizar parámetros de una estrategia.
    
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
    # Generar ID de tarea
    task_id = f"optimize_{int(datetime.now().timestamp())}"
    
    try:
        # Importar módulo y clase de estrategia
        module = importlib.import_module(strategy_module)
        strategy_class = getattr(module, strategy_name)
        
        # Registrar tarea
        active_tasks[task_id] = {
            "id": task_id,
            "type": "optimize",
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "params": {
                "strategy_name": strategy_name,
                "strategy_module": strategy_module,
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "initial_balance": initial_balance,
                "commission": commission,
                "slippage": slippage,
                "risk_per_trade": risk_per_trade,
                "param_grid": param_grid,
                "metric": metric,
                "n_jobs": n_jobs
            }
        }
        
        # Iniciar tarea en segundo plano
        background_tasks.add_task(
            _run_optimize_task,
            task_id,
            strategy_class,
            param_grid,
            symbol,
            timeframe,
            start_date,
            end_date,
            initial_balance,
            commission,
            slippage,
            risk_per_trade,
            metric,
            n_jobs
        )
        
        return {
            "task_id": task_id,
            "status": "accepted",
            "message": "Tarea de optimización iniciada"
        }
    
    except ImportError:
        raise HTTPException(status_code=404, detail=f"Módulo de estrategia no encontrado: {strategy_module}")
    
    except AttributeError:
        raise HTTPException(status_code=404, detail=f"Estrategia no encontrada: {strategy_name}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al iniciar optimización: {str(e)}")


async def _run_optimize_task(
    task_id: str,
    strategy_class: Type[Strategy],
    param_grid: Dict[str, List[Any]],
    symbol: str,
    timeframe: str,
    start_date: Optional[str],
    end_date: Optional[str],
    initial_balance: float,
    commission: float,
    slippage: float,
    risk_per_trade: float,
    metric: str,
    n_jobs: int
) -> None:
    """
    Ejecutar tarea de optimización en segundo plano.
    
    Args:
        task_id: ID de la tarea
        strategy_class: Clase de la estrategia
        param_grid: Grid de parámetros
        symbol: Símbolo de trading
        timeframe: Timeframe para los datos
        start_date: Fecha de inicio
        end_date: Fecha de fin
        initial_balance: Balance inicial
        commission: Comisión por operación
        slippage: Slippage por operación
        risk_per_trade: Riesgo por operación
        metric: Métrica a optimizar
        n_jobs: Número de jobs para paralelización
    """
    try:
        # Actualizar estado
        active_tasks[task_id]["status"] = "running"
        
        # Ejecutar optimización
        result = await backtest_engine.optimize_strategy(
            strategy_class,
            param_grid,
            symbol,
            timeframe,
            start_date,
            end_date,
            initial_balance,
            commission,
            slippage,
            risk_per_trade,
            n_jobs,
            metric
        )
        
        # Actualizar estado
        if result:
            active_tasks[task_id]["status"] = "completed"
            active_tasks[task_id]["result"] = result
            active_tasks[task_id]["completed_at"] = datetime.now().isoformat()
        else:
            active_tasks[task_id]["status"] = "failed"
            active_tasks[task_id]["error"] = "No se pudo completar la optimización"
    
    except Exception as e:
        active_tasks[task_id]["status"] = "failed"
        active_tasks[task_id]["error"] = str(e)


@router.post("/walk-forward")
async def walk_forward_analysis(
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
) -> Dict[str, Any]:
    """
    Realizar análisis Walk-Forward.
    
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
    # Generar ID de tarea
    task_id = f"wfa_{int(datetime.now().timestamp())}"
    
    try:
        # Importar módulo y clase de estrategia
        module = importlib.import_module(strategy_module)
        strategy_class = getattr(module, strategy_name)
        
        # Registrar tarea
        active_tasks[task_id] = {
            "id": task_id,
            "type": "walk-forward",
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "params": {
                "strategy_name": strategy_name,
                "strategy_module": strategy_module,
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "window_size": window_size,
                "test_size": test_size,
                "step_size": step_size,
                "initial_balance": initial_balance,
                "commission": commission,
                "slippage": slippage,
                "risk_per_trade": risk_per_trade,
                "param_grid": param_grid,
                "metric": metric
            }
        }
        
        # Iniciar tarea en segundo plano
        background_tasks.add_task(
            _run_wfa_task,
            task_id,
            strategy_class,
            param_grid,
            symbol,
            timeframe,
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
        
        return {
            "task_id": task_id,
            "status": "accepted",
            "message": "Tarea de análisis Walk-Forward iniciada"
        }
    
    except ImportError:
        raise HTTPException(status_code=404, detail=f"Módulo de estrategia no encontrado: {strategy_module}")
    
    except AttributeError:
        raise HTTPException(status_code=404, detail=f"Estrategia no encontrada: {strategy_name}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al iniciar análisis Walk-Forward: {str(e)}")


async def _run_wfa_task(
    task_id: str,
    strategy_class: Type[Strategy],
    param_grid: Dict[str, List[Any]],
    symbol: str,
    timeframe: str,
    start_date: Optional[str],
    end_date: Optional[str],
    window_size: int,
    test_size: int,
    step_size: int,
    initial_balance: float,
    commission: float,
    slippage: float,
    risk_per_trade: float,
    metric: str
) -> None:
    """
    Ejecutar tarea de Walk-Forward Analysis en segundo plano.
    
    Args:
        task_id: ID de la tarea
        strategy_class: Clase de la estrategia
        param_grid: Grid de parámetros
        symbol: Símbolo de trading
        timeframe: Timeframe para los datos
        start_date: Fecha de inicio
        end_date: Fecha de fin
        window_size: Tamaño de ventana de entrenamiento
        test_size: Tamaño de ventana de prueba
        step_size: Tamaño de paso
        initial_balance: Balance inicial
        commission: Comisión por operación
        slippage: Slippage por operación
        risk_per_trade: Riesgo por operación
        metric: Métrica a optimizar
    """
    try:
        # Actualizar estado
        active_tasks[task_id]["status"] = "running"
        
        # Ejecutar WFA
        result = await backtest_engine.walk_forward_analysis(
            strategy_class,
            param_grid,
            symbol,
            timeframe,
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
        
        # Actualizar estado
        if result:
            active_tasks[task_id]["status"] = "completed"
            active_tasks[task_id]["result"] = result
            active_tasks[task_id]["completed_at"] = datetime.now().isoformat()
        else:
            active_tasks[task_id]["status"] = "failed"
            active_tasks[task_id]["error"] = "No se pudo completar el análisis Walk-Forward"
    
    except Exception as e:
        active_tasks[task_id]["status"] = "failed"
        active_tasks[task_id]["error"] = str(e)


@router.get("/task/{task_id}")
async def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Obtener estado de una tarea.
    
    Args:
        task_id: ID de la tarea
        
    Returns:
        Estado de la tarea
    """
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail=f"Tarea no encontrada: {task_id}")
    
    return active_tasks[task_id]


@router.get("/tasks")
async def get_tasks(
    status: Optional[str] = None,
    type: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Obtener lista de tareas.
    
    Args:
        status: Filtrar por estado
        type: Filtrar por tipo
        limit: Límite de resultados
        
    Returns:
        Lista de tareas
    """
    tasks = list(active_tasks.values())
    
    # Filtrar por estado
    if status:
        tasks = [t for t in tasks if t.get("status") == status]
    
    # Filtrar por tipo
    if type:
        tasks = [t for t in tasks if t.get("type") == type]
    
    # Ordenar por fecha (más recientes primero)
    tasks.sort(key=lambda t: t.get("created_at", ""), reverse=True)
    
    # Limitar resultados
    tasks = tasks[:limit]
    
    return {"tasks": tasks}


@router.get("/results")
async def get_results_list() -> Dict[str, Any]:
    """
    Obtener lista de resultados de backtest guardados.
    
    Returns:
        Lista de resultados
    """
    results = []
    
    # Directorio de resultados
    results_dir = "data/backtest_results"
    
    # Verificar si existe
    if not os.path.exists(results_dir):
        return {"results": results}
    
    # Buscar archivos JSON
    for file_name in os.listdir(results_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(results_dir, file_name)
            
            try:
                # Leer archivo
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Añadir a la lista
                results.append({
                    "id": file_name[:-5],
                    "file_name": file_name,
                    "strategy_name": data.get("strategy_name", ""),
                    "symbol": data.get("symbol", ""),
                    "timeframe": data.get("timeframe", ""),
                    "start_date": data.get("start_date", ""),
                    "end_date": data.get("end_date", ""),
                    "metrics": data.get("metrics", {}),
                    "file_path": file_path
                })
            
            except Exception as e:
                print(f"Error al leer archivo {file_name}: {e}")
    
    # Ordenar por fecha (más recientes primero)
    results.sort(key=lambda r: os.path.getmtime(r["file_path"]), reverse=True)
    
    return {"results": results}


@router.get("/result/{result_id}")
async def get_result(result_id: str) -> Dict[str, Any]:
    """
    Obtener un resultado de backtest específico.
    
    Args:
        result_id: ID del resultado
        
    Returns:
        Datos completos del resultado
    """
    # Buscar en tareas activas
    for task in active_tasks.values():
        if task.get("status") == "completed" and task.get("id") == result_id:
            return task.get("result", {})
    
    # Buscar en archivos guardados
    file_path = f"data/backtest_results/{result_id}.json"
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al leer archivo: {str(e)}")
    
    raise HTTPException(status_code=404, detail=f"Resultado no encontrado: {result_id}")


@router.delete("/task/{task_id}")
async def delete_task(task_id: str) -> Dict[str, Any]:
    """
    Eliminar una tarea.
    
    Args:
        task_id: ID de la tarea
        
    Returns:
        Confirmación
    """
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail=f"Tarea no encontrada: {task_id}")
    
    del active_tasks[task_id]
    
    return {"message": f"Tarea {task_id} eliminada"}


# Endpoints adicionales

@router.get("/charts/{chart_id}")
async def get_chart(chart_id: str) -> Dict[str, Any]:
    """
    Obtener datos de un gráfico.
    
    Args:
        chart_id: ID del gráfico
        
    Returns:
        Datos del gráfico
    """
    # Buscar en resultados de tareas
    for task in active_tasks.values():
        if task.get("status") == "completed" and "result" in task:
            result = task.get("result", {})
            charts = result.get("charts", {})
            
            for chart_key, chart_path in charts.items():
                if chart_key == chart_id or chart_path.endswith(f"{chart_id}.png"):
                    return {
                        "id": chart_id,
                        "path": chart_path,
                        "type": chart_key
                    }
    
    # Buscar en directorio de gráficos
    plots_dir = "data/backtest_results/plots"
    
    if os.path.exists(plots_dir):
        for file_name in os.listdir(plots_dir):
            if file_name.endswith(f"{chart_id}.png"):
                return {
                    "id": chart_id,
                    "path": os.path.join(plots_dir, file_name),
                    "type": file_name.split('_')[0]
                }
    
    raise HTTPException(status_code=404, detail=f"Gráfico no encontrado: {chart_id}")