"""
Logger module for Genesis system.

This module provides a centralized logging configuration for the entire system,
ensuring consistent log formats, levels, and destinations.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Configure logger name for the Genesis system
LOGGER_NAME = "genesis"

class Logger:
    """
    Central logging configuration for the Genesis system.
    
    Provides static methods to set up logging with appropriate handlers,
    formatters, and levels based on configuration.
    """
    
    @staticmethod
    def setup_logging(log_dir: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
        """
        Set up logging for the application.
        
        Args:
            log_dir: Directory to store log files
            level: Logging level (default: INFO)
            
        Returns:
            Configured logger instance
        """
        # Get or create logger
        logger = logging.getLogger(LOGGER_NAME)
        logger.setLevel(level)
        
        # Clear existing handlers to avoid duplicates
        if logger.handlers:
            logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Add console handler to logger
        logger.addHandler(console_handler)
        
        # Add file handler if log directory is provided
        if log_dir:
            try:
                # Create log directory if it doesn't exist
                os.makedirs(log_dir, exist_ok=True)
                
                # Create file handler
                log_file = log_dir / 'genesis.log'
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
                # Create separate error log
                error_log_file = log_dir / 'error.log'
                error_file_handler = logging.FileHandler(error_log_file)
                error_file_handler.setLevel(logging.ERROR)
                error_file_handler.setFormatter(formatter)
                logger.addHandler(error_file_handler)
            except Exception as e:
                # Don't fail if log directory can't be created, just log to console
                logger.error(f"Failed to set up file logging: {e}")
        
        return logger
    
    @staticmethod
    def get_logger() -> logging.Logger:
        """
        Get the Genesis logger instance.
        
        Returns:
            Logger instance
        """
        return logging.getLogger(LOGGER_NAME)