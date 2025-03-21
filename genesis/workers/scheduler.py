"""
Task scheduler implementation.

This module provides a scheduler for running periodic tasks
such as data synchronization, reporting, and maintenance.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Coroutine, Union
from datetime import datetime, timedelta
import time

from genesis.core.base import Component
from genesis.utils.logger import setup_logging


class Task:
    """
    Task definition for the scheduler.
    
    Represents a scheduled task with timing and execution details.
    """
    
    def __init__(
        self, 
        name: str, 
        coro_func: Callable[..., Coroutine],
        interval: Optional[int] = None,  # Seconds
        cron: Optional[str] = None,      # Cron-like expression "minute hour day month weekday"
        start_time: Optional[datetime] = None,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        enabled: bool = True
    ):
        """
        Initialize a scheduled task.
        
        Args:
            name: Task name
            coro_func: Coroutine function to execute
            interval: Time interval in seconds
            cron: Cron-like schedule expression
            start_time: Specific start time
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            enabled: Whether the task is enabled
        """
        self.name = name
        self.coro_func = coro_func
        self.interval = interval
        self.cron = self._parse_cron(cron) if cron else None
        self.start_time = start_time
        self.args = args or []
        self.kwargs = kwargs or {}
        self.enabled = enabled
        self.last_run: Optional[datetime] = None
        self.next_run: Optional[datetime] = None
        self._calculate_next_run()
    
    def _parse_cron(self, cron_str: str) -> Dict[str, List[int]]:
        """
        Parse a cron expression.
        
        Args:
            cron_str: Cron-like expression "minute hour day month weekday"
            
        Returns:
            Dictionary of parsed values
        """
        parts = cron_str.split()
        if len(parts) != 5:
            raise ValueError("Cron expression must have 5 parts: minute hour day month weekday")
        
        # Simple cron parsing
        cron = {
            "minute": self._parse_cron_field(parts[0], 0, 59),
            "hour": self._parse_cron_field(parts[1], 0, 23),
            "day": self._parse_cron_field(parts[2], 1, 31),
            "month": self._parse_cron_field(parts[3], 1, 12),
            "weekday": self._parse_cron_field(parts[4], 0, 6)
        }
        
        return cron
    
    def _parse_cron_field(self, field: str, min_val: int, max_val: int) -> List[int]:
        """
        Parse a single cron field.
        
        Args:
            field: Cron field value (e.g., '1-5', '*/2', '1,3,5')
            min_val: Minimum valid value
            max_val: Maximum valid value
            
        Returns:
            List of values
        """
        if field == '*':
            return list(range(min_val, max_val + 1))
        
        result = []
        for part in field.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                result.extend(range(start, end + 1))
            elif '/' in part:
                if part.startswith('*/'):
                    start, step = min_val, int(part.split('/')[1])
                    result.extend(range(start, max_val + 1, step))
                else:
                    range_val, step = part.split('/')
                    if '-' in range_val:
                        start, end = map(int, range_val.split('-'))
                    else:
                        start, end = int(range_val), max_val
                    result.extend(range(start, end + 1, int(step)))
            else:
                result.append(int(part))
        
        return sorted(set(result))
    
    def _calculate_next_run(self) -> None:
        """Calculate the next run time for the task."""
        now = datetime.now()
        
        if self.start_time and self.start_time > now:
            self.next_run = self.start_time
            return
        
        if self.interval:
            # If never run, schedule now, otherwise schedule after interval
            if self.last_run is None:
                self.next_run = now
            else:
                self.next_run = self.last_run + timedelta(seconds=self.interval)
            
            # If next run is in the past, schedule for now
            if self.next_run < now:
                self.next_run = now
            
            return
        
        if self.cron:
            # Calculate next run based on cron
            self.next_run = self._next_cron_run(now)
            return
        
        # Default: run now
        self.next_run = now
    
    def _next_cron_run(self, now: datetime) -> datetime:
        """
        Calculate the next run time based on cron expression.
        
        Args:
            now: Current datetime
            
        Returns:
            Next run datetime
        """
        # Simple implementation - in a real system this would be more robust
        candidate = now.replace(second=0, microsecond=0)
        
        # Try the next minute
        candidate += timedelta(minutes=1)
        
        # Find the next matching time
        for _ in range(525600):  # Max 1 year of minutes
            if (candidate.minute in self.cron["minute"] and
                candidate.hour in self.cron["hour"] and
                candidate.day in self.cron["day"] and
                candidate.month in self.cron["month"] and
                candidate.weekday() in self.cron["weekday"]):
                return candidate
            
            candidate += timedelta(minutes=1)
        
        # Fallback - should not happen
        return now + timedelta(minutes=1)
    
    def update_after_run(self) -> None:
        """Update timing after task execution."""
        self.last_run = datetime.now()
        self._calculate_next_run()
    
    def seconds_until_next_run(self) -> float:
        """
        Calculate seconds until next scheduled run.
        
        Returns:
            Seconds until next run
        """
        if not self.next_run:
            return 0
        
        now = datetime.now()
        if self.next_run <= now:
            return 0
        
        return (self.next_run - now).total_seconds()


class Scheduler(Component):
    """
    Task scheduler component.
    
    This component manages scheduled tasks, executing them
    at the appropriate times.
    """
    
    def __init__(self, name: str = "scheduler"):
        """
        Initialize the scheduler.
        
        Args:
            name: Component name
        """
        super().__init__(name)
        self.logger = setup_logging(name)
        self.tasks: Dict[str, Task] = {}
    
    async def start(self) -> None:
        """Start the scheduler."""
        await super().start()
        
        # Start scheduler loop
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        self.logger.info(f"Scheduler started with {len(self.tasks)} tasks")
    
    async def stop(self) -> None:
        """Stop the scheduler."""
        # Cancel scheduler loop
        if hasattr(self, 'scheduler_task') and self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        await super().stop()
        self.logger.info("Scheduler stopped")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Handle events from the event bus.
        
        Args:
            event_type: Event type
            data: Event data
            source: Source component
        """
        # The scheduler doesn't respond to events directly
        pass
    
    def add_task(self, task: Task) -> None:
        """
        Add a task to the scheduler.
        
        Args:
            task: Task to add
        """
        if task.name in self.tasks:
            self.logger.warning(f"Task {task.name} already exists, replacing")
        
        self.tasks[task.name] = task
        self.logger.info(f"Added task: {task.name}, next run: {task.next_run}")
    
    def remove_task(self, task_name: str) -> bool:
        """
        Remove a task from the scheduler.
        
        Args:
            task_name: Name of the task to remove
            
        Returns:
            True if the task was removed, False otherwise
        """
        if task_name in self.tasks:
            del self.tasks[task_name]
            self.logger.info(f"Removed task: {task_name}")
            return True
        return False
    
    def enable_task(self, task_name: str) -> bool:
        """
        Enable a task.
        
        Args:
            task_name: Name of the task to enable
            
        Returns:
            True if the task was enabled, False otherwise
        """
        if task_name in self.tasks:
            self.tasks[task_name].enabled = True
            self.logger.info(f"Enabled task: {task_name}")
            return True
        return False
    
    def disable_task(self, task_name: str) -> bool:
        """
        Disable a task.
        
        Args:
            task_name: Name of the task to disable
            
        Returns:
            True if the task was disabled, False otherwise
        """
        if task_name in self.tasks:
            self.tasks[task_name].enabled = False
            self.logger.info(f"Disabled task: {task_name}")
            return True
        return False
    
    async def _scheduler_loop(self) -> None:
        """Main scheduler loop that checks and executes tasks."""
        while self.running:
            try:
                now = datetime.now()
                
                # Find tasks that need to run
                for task_name, task in list(self.tasks.items()):
                    if not task.enabled:
                        continue
                    
                    if task.next_run and task.next_run <= now:
                        # Execute task in the background
                        asyncio.create_task(self._execute_task(task))
                        
                        # Update timing
                        task.update_after_run()
                        self.logger.debug(f"Scheduled next run for {task_name}: {task.next_run}")
                
                # Sleep until next task or for a maximum of 1 second
                sleep_time = min(
                    (task.seconds_until_next_run() for task in self.tasks.values() if task.enabled),
                    default=1.0
                )
                await asyncio.sleep(min(max(0.1, sleep_time), 1.0))
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _execute_task(self, task: Task) -> None:
        """
        Execute a task.
        
        Args:
            task: Task to execute
        """
        try:
            self.logger.info(f"Executing task: {task.name}")
            start_time = time.time()
            
            # Execute the task
            await task.coro_func(*task.args, **task.kwargs)
            
            elapsed = time.time() - start_time
            self.logger.info(f"Task {task.name} completed in {elapsed:.2f} seconds")
            
            # Emit task completion event
            await self.emit_event("scheduler.task_completed", {
                "task_name": task.name,
                "elapsed_time": elapsed,
                "success": True
            })
        
        except Exception as e:
            self.logger.error(f"Error executing task {task.name}: {e}")
            
            # Emit task error event
            await self.emit_event("scheduler.task_error", {
                "task_name": task.name,
                "error": str(e),
                "error_type": type(e).__name__
            })

