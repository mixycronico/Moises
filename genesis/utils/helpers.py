"""
Helper functions module.

This module contains utility functions used throughout the system.
"""

import time
import asyncio
from typing import Any, Callable, TypeVar, Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import json
import hashlib
import uuid
import os
import pandas as pd
import numpy as np

T = TypeVar('T')


def generate_id() -> str:
    """
    Generate a unique identifier.
    
    Returns:
        Unique string identifier
    """
    return str(uuid.uuid4())


def generate_trade_id(symbol: str) -> str:
    """
    Generate a trade ID with symbol prefix.
    
    Args:
        symbol: Trading pair symbol
        
    Returns:
        Trade ID string
    """
    symbol_prefix = symbol.split('/')[0].lower()
    random_id = uuid.uuid4().hex[:8]
    timestamp = int(time.time())
    return f"{symbol_prefix}_{timestamp}_{random_id}"


def format_price(price: float, precision: int = 8) -> str:
    """
    Format a price with appropriate precision.
    
    Args:
        price: Price value
        precision: Decimal precision
        
    Returns:
        Formatted price string
    """
    return f"{price:.{precision}f}".rstrip('0').rstrip('.') if '.' in f"{price:.{precision}f}" else f"{price:.{precision}f}"


def format_timestamp(timestamp: Union[int, float, str, datetime], fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a timestamp into a human-readable string.
    
    Args:
        timestamp: Timestamp (Unix timestamp or datetime)
        fmt: Format string
        
    Returns:
        Formatted datetime string
    """
    if isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(timestamp / 1000 if timestamp > 1e10 else timestamp)
    elif isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp)
        except ValueError:
            dt = datetime.fromtimestamp(float(timestamp) / 1000 if float(timestamp) > 1e10 else float(timestamp))
    elif isinstance(timestamp, datetime):
        dt = timestamp
    else:
        raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")
    
    return dt.strftime(fmt)


def truncate_float(value: float, precision: int) -> float:
    """
    Truncate a float to a specific precision.
    
    This is useful for respecting exchange precision requirements.
    
    Args:
        value: Float value to truncate
        precision: Number of decimal places
        
    Returns:
        Truncated float
    """
    factor = 10 ** precision
    return int(value * factor) / factor


def calculate_profit_percentage(
    entry_price: float, 
    exit_price: float, 
    side: str
) -> float:
    """
    Calculate profit percentage for a trade.
    
    Args:
        entry_price: Entry price
        exit_price: Exit price
        side: Trade side ('long' or 'short')
        
    Returns:
        Profit percentage (positive for profit, negative for loss)
    """
    if side.lower() == 'long':
        return (exit_price / entry_price - 1) * 100
    else:  # short
        return (entry_price / exit_price - 1) * 100


async def retry_async(
    func: Callable[..., Any],
    *args: Any,
    retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Exception, ...] = (Exception,)
) -> Any:
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: Async function to call
        *args: Arguments to pass to the function
        retries: Maximum number of retries
        delay: Initial delay between retries (seconds)
        backoff_factor: Multiplicative factor for backoff
        exceptions: Exception types to catch and retry
        
    Returns:
        Result of the function call
        
    Raises:
        Last exception encountered if all retries fail
    """
    last_exception = None
    for attempt in range(retries):
        try:
            return await func(*args)
        except exceptions as e:
            last_exception = e
            if attempt < retries - 1:
                # Calculate backoff delay
                backoff_delay = delay * (backoff_factor ** attempt)
                # Wait before retry
                await asyncio.sleep(backoff_delay)
    
    # If we get here, all retries failed
    if last_exception:
        raise last_exception
    else:
        raise Exception("All retries failed with unknown error")


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning a default value if division by zero.
    
    Args:
        a: Numerator
        b: Denominator
        default: Default value to return if division by zero
        
    Returns:
        Division result or default value
    """
    return a / b if b != 0 else default


def load_json_file(file_path: str, default: Any = None) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        default: Default value to return if file not found or invalid
        
    Returns:
        Loaded data or default value
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return default
    except Exception:
        return default


def save_json_file(file_path: str, data: Any, indent: int = 2) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        file_path: Path to the JSON file
        data: Data to save
        indent: JSON indentation level
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
        return True
    except Exception:
        return False


def convert_to_dataframe(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of dictionaries to a pandas DataFrame.
    
    Args:
        data: List of dictionaries
        
    Returns:
        Pandas DataFrame
    """
    df = pd.DataFrame(data)
    
    # Convert timestamp strings to datetime objects if present
    if 'timestamp' in df.columns and df['timestamp'].dtype == 'object':
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


def merge_dictionaries(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries, with later dictionaries overriding earlier ones.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result

