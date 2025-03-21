"""
Task processor implementation.

This module provides a processor for handling asynchronous background tasks
such as data processing, analysis, and other CPU-intensive operations.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable, Coroutine, Union, Set
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import traceback

from genesis.core.base import Component
from genesis.utils.logger import setup_logging


class TaskProcessor(Component):
    """
    Processor for asynchronous background tasks.
    
    This component manages a pool of workers for executing CPU-intensive
    tasks without blocking the main event loop.
    """
    
    def __init__(
        self, 
        name: str = "task_processor",
        max_workers: int = 5,
        use_processes: bool = False
    ):
        """
        Initialize the task processor.
        
        Args:
            name: Component name
            max_workers: Maximum number of worker threads/processes
            use_processes: Whether to use processes instead of threads
        """
        super().__init__(name)
        self.logger = setup_logging(name)
        
        self.max_workers = max_workers
        self.use_processes = use_processes
        
        # Task queue
        self.task_queue: asyncio.Queue = None
        
        # Executor
        self.executor = None
        
        # Active tasks
        self.active_tasks: Set[asyncio.Task] = set()
    
    async def start(self) -> None:
        """Start the task processor."""
        await super().start()
        
        # Initialize queue
        self.task_queue = asyncio.Queue()
        
        # Initialize executor
        if self.use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
            self.logger.info(f"Using process pool with {self.max_workers} workers")
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
            self.logger.info(f"Using thread pool with {self.max_workers} workers")
        
        # Start consumer task
        self.consumer_task = asyncio.create_task(self._consume_tasks())
        
        self.logger.info("Task processor started")
    
    async def stop(self) -> None:
        """Stop the task processor."""
        # Cancel consumer
        if hasattr(self, 'consumer_task') and self.consumer_task:
            self.consumer_task.cancel()
            try:
                await self.consumer_task
            except asyncio.CancelledError:
                pass
        
        # Cancel active tasks
        for task in self.active_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
        
        # Shutdown executor
        if self.executor:
            self.executor.shutdown()
        
        await super().stop()
        self.logger.info("Task processor stopped")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Handle events from the event bus.
        
        Args:
            event_type: Event type
            data: Event data
            source: Source component
        """
        if event_type == "task.submit":
            # Extract task information
            task_name = data.get("task_name")
            task_func = data.get("task_func")
            task_args = data.get("args", [])
            task_kwargs = data.get("kwargs", {})
            
            if task_name and task_func:
                # Add task to queue
                await self.submit_task(task_name, task_func, task_args, task_kwargs)
    
    async def submit_task(
        self, 
        task_name: str,
        task_func: Callable,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Submit a task for processing.
        
        Args:
            task_name: Name of the task
            task_func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
        """
        if not self.task_queue:
            raise RuntimeError("Task processor not started")
        
        task_data = {
            "name": task_name,
            "func": task_func,
            "args": args or [],
            "kwargs": kwargs or {},
            "submit_time": time.time()
        }
        
        await self.task_queue.put(task_data)
        self.logger.debug(f"Submitted task: {task_name}")
    
    async def _consume_tasks(self) -> None:
        """Consumer loop that processes tasks from the queue."""
        if not self.task_queue:
            raise RuntimeError("Task queue not initialized")
        
        while self.running:
            try:
                # Get task from queue
                task_data = await self.task_queue.get()
                
                # Process the task
                task = asyncio.create_task(self._execute_task(task_data))
                self.active_tasks.add(task)
                task.add_done_callback(self.active_tasks.discard)
                
                # Mark task as done in the queue
                self.task_queue.task_done()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in task consumer: {e}")
                await asyncio.sleep(1.0)
    
    async def _execute_task(self, task_data: Dict[str, Any]) -> None:
        """
        Execute a task in the thread/process pool.
        
        Args:
            task_data: Task data dictionary
        """
        task_name = task_data["name"]
        task_func = task_data["func"]
        task_args = task_data["args"]
        task_kwargs = task_data["kwargs"]
        start_time = time.time()
        
        try:
            self.logger.info(f"Executing task: {task_name}")
            
            # Submit to executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._run_in_executor,
                task_func,
                task_args,
                task_kwargs
            )
            
            elapsed = time.time() - start_time
            self.logger.info(f"Task {task_name} completed in {elapsed:.2f} seconds")
            
            # Emit completion event
            await self.emit_event("task.completed", {
                "task_name": task_name,
                "elapsed_time": elapsed,
                "result": result,
                "success": True
            })
        
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"Error executing task {task_name}: {e}")
            
            # Emit error event
            await self.emit_event("task.error", {
                "task_name": task_name,
                "elapsed_time": elapsed,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            })
    
    def _run_in_executor(
        self, 
        func: Callable, 
        args: List[Any], 
        kwargs: Dict[str, Any]
    ) -> Any:
        """
        Run a function in the executor.
        
        Args:
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        return func(*args, **kwargs)

