"""
API server implementation.

This module provides a FastAPI server for the Genesis trading system API.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
import os
import json
import time
from datetime import datetime, timedelta

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Query, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from genesis.config.settings import settings
from genesis.core.base import Component
from genesis.utils.logger import setup_logging


class APIServer(Component):
    """
    API server component for the Genesis trading system.
    
    This component provides a REST API for monitoring, configuration,
    and control of the trading system.
    """
    
    def __init__(
        self, 
        name: str = "api_server",
        host: Optional[str] = None,
        port: Optional[int] = None
    ):
        """
        Initialize the API server.
        
        Args:
            name: Component name
            host: API server host
            port: API server port
        """
        super().__init__(name)
        self.logger = setup_logging(name)
        
        self.host = host or settings.get('api.host', '0.0.0.0')
        self.port = port or settings.get('api.port', 8000)
        self.debug = settings.get('api.debug', False)
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Genesis Trading API",
            description="API for the Genesis automated cryptocurrency trading system",
            version="0.1.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Set up API routes
        self._setup_routes()
        
        # Server instance
        self.server = None
        
        # Event bus message cache
        self.event_cache: Dict[str, List[Dict[str, Any]]] = {
            "strategy.signal": [],
            "trade.opened": [],
            "trade.closed": [],
            "performance.metrics": [],
            "market.analyzed": [],
            "market.anomalies": []
        }
        self.max_cache_size = 100  # Maximum number of events to keep per type
    
    def _setup_routes(self) -> None:
        """Set up API routes."""
        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "ok",
                "timestamp": datetime.now().isoformat(),
                "version": "0.1.0"
            }
        
        # System status endpoint
        @self.app.get("/status")
        async def system_status():
            return {
                "status": "running" if self.running else "stopped",
                "uptime": self._get_uptime(),
                "components": self._get_component_statuses(),
                "timestamp": datetime.now().isoformat()
            }
        
        # Get recent events
        @self.app.get("/events/{event_type}")
        async def get_events(event_type: str, limit: int = Query(10, ge=1, le=100)):
            if event_type not in self.event_cache and event_type != "all":
                raise HTTPException(status_code=404, detail=f"Event type '{event_type}' not found")
            
            if event_type == "all":
                # Combine all event types
                all_events = []
                for events in self.event_cache.values():
                    all_events.extend(events)
                
                # Sort by timestamp if available
                all_events.sort(
                    key=lambda x: x.get("timestamp", ""),
                    reverse=True
                )
                
                return {"events": all_events[:limit]}
            
            return {"events": self.event_cache[event_type][:limit]}
    
    def _get_uptime(self) -> str:
        """
        Get system uptime.
        
        Returns:
            Uptime string
        """
        # Placeholder for actual uptime calculation
        return "0 days, 0 hours, 0 minutes"
    
    def _get_component_statuses(self) -> Dict[str, Any]:
        """
        Get status of all components.
        
        Returns:
            Component status dictionary
        """
        # This would be populated with actual component statuses
        return {}
    
    async def start(self) -> None:
        """Start the API server."""
        await super().start()
        
        # Start server in a background task
        self.server_task = asyncio.create_task(self._run_server())
        
        self.logger.info(f"API server starting on {self.host}:{self.port}")
    
    async def stop(self) -> None:
        """Stop the API server."""
        # Cancel the server task
        if hasattr(self, 'server_task') and self.server_task:
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass
        
        await super().stop()
        self.logger.info("API server stopped")
    
    async def _run_server(self) -> None:
        """Run the uvicorn server."""
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            loop="asyncio"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Handle events from the event bus.
        
        Caches recent events for API endpoints.
        
        Args:
            event_type: Event type
            data: Event data
            source: Source component
        """
        # Add timestamp if not present
        if "timestamp" not in data:
            data["timestamp"] = datetime.now().isoformat()
        
        # Add source component
        data["source"] = source
        
        # Cache the event if it's a type we're interested in
        if event_type in self.event_cache:
            # Add to front of list (newest first)
            self.event_cache[event_type].insert(0, data)
            
            # Trim to max size
            if len(self.event_cache[event_type]) > self.max_cache_size:
                self.event_cache[event_type] = self.event_cache[event_type][:self.max_cache_size]

