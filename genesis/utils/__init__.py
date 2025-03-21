"""
Utility modules for the Genesis trading system.

This package contains utility functions and classes used throughout
the system, including logging, helper functions, and common tools.
"""

from genesis.utils.helpers import (
    generate_id, generate_trade_id, format_price, format_timestamp,
    truncate_float, calculate_profit_percentage, retry_async, safe_divide,
    load_json_file, save_json_file, convert_to_dataframe, merge_dictionaries
)
from genesis.utils.logger import setup_logging, JsonFormatter
from genesis.utils.log_manager import (
    LogManager, get_logger, log_event, query_logs, get_log_stats
)
from genesis.utils.advanced_logger import (
    setup_advanced_logging, log_action, get_audit_logs, JsonFormatter as AdvancedJsonFormatter,
    SQLiteHandler, init_log_db
)

__all__ = [
    # From helpers.py
    "generate_id", "generate_trade_id", "format_price", "format_timestamp",
    "truncate_float", "calculate_profit_percentage", "retry_async", "safe_divide",
    "load_json_file", "save_json_file", "convert_to_dataframe", "merge_dictionaries",
    
    # From logger.py
    "setup_logging", "JsonFormatter",
    
    # From log_manager.py
    "LogManager", "get_logger", "log_event", "query_logs", "get_log_stats",
    
    # From advanced_logger.py
    "setup_advanced_logging", "log_action", "get_audit_logs", "AdvancedJsonFormatter",
    "SQLiteHandler", "init_log_db"
]

