#!/usr/bin/env python3
"""
Enhanced MCP Client - Client for Model Context Protocol with configuration support.
"""
from typing import Any, Dict, List, Optional, Union
import asyncio
import json
import logging
import os
import uuid

import aiohttp
import httpx
import websockets
from pydantic import BaseModel, Field

from .mcp_server_manager import MCPServerManager

# Configure logging
logger = logging.getLogger(__name__)


class MCPClientRequest(BaseModel):
    """Model for MCP client requests."""
    client_id: str = Field(default_factory=lambda: f"mcp-client-{uuid.uuid4().hex[:8]}")
    query: str
    context: Dict[str, Any] = Field(default_factory=dict)


class MCPClientResponse(BaseModel):
    """Model for MCP client responses."""
    client_id: str
    response: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EnhancedMCPClient:
    """
    Enhanced MCP client with configuration support.
    
    This client can:
    1. Use server configurations from a config file
    2. Connect to both internal and external MCP servers
    3. Support both HTTP and WebSocket connections
    4. Handle server startup and monitoring
    """
    
    def __init__(
        self,
        server_name: Optional[str] = None,
        config_path: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        use_websocket: bool = False,
    ):
        """
        Initialize the MCP client.
        
        Args:
            server_name: Name of MCP server to use from config
            config_path: Path to MCP server configuration file
            host: Host for direct connection (if not using config)
            port: Port for direct connection (if not using config)
            use_websocket: Whether to use WebSocket instead of HTTP
        """
        # Initialize server manager if using configuration
        self.server_manager = None if (host and port) else MCPServerManager(config_path=config_path)
        
        self.server_name = server_name
        self.use_websocket = use_websocket
        self.client_id = f"mcp-client-{uuid.uuid4().hex[:8]}"
        
        # Determine connection details
        if self.server_manager:
            server_config = self.server_manager.get_server_config(server_name)
            self.host = server_config.host
            self.port = server_config.port
            self.external_server = not server_config.internal
        else:
            self.host = host or "localhost"
            self.port = port or 8000
            self.external_server = False
        
        # Initialize connection
        self._ws_connection = None
    
    async def ensure_server_running(self) -> bool:
        """Ensure the MCP server is running."""
        if not self.server_manager or not self.external_server:
            # No server management needed
            return True
            
        return self.server_manager.start_external_server(self.server_name)
    
    async def connect_websocket(self) -> bool:
        """Connect to the WebSocket endpoint."""
        if self._ws_connection:
            return True
            
        try:
            uri = f"ws://{self.host}:{self.port}/ws"
            self._ws_connection = await websockets.connect(uri)
            logger.info(f"Connected to MCP server WebSocket at {uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False
    
    async def disconnect_websocket(self):
        """Disconnect from the WebSocket endpoint."""
        if self._ws_connection:
            await self._ws_connection.close()
            self._ws_connection = None
    
    async def query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        client_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a query to the MCP server.
        
        Args:
            query: The query to process
            context: Additional context for the query
            client_id: Client identifier (generated if not provided)
            
        Returns:
            Dictionary containing response and metadata
        """
        # Ensure server is running if using external server
        await self.ensure_server_running()
        
        # Prepare request
        request = MCPClientRequest(
            client_id=client_id or self.client_id,
            query=query,
            context=context or {}
        )
        
        # Send request using appropriate method
        if self.use_websocket:
            return await self._query_websocket(request)
        else:
            return await self._query_http(request)
    
    async def _query_http(self, request: MCPClientRequest) -> Dict[str, Any]:
        """Send query using HTTP POST."""
        url = f"http://{self.host}:{self.port}/query"
        logger.debug(f"Sending HTTP query to {url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=request.model_dump()) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"HTTP Error: {response.status}, {error_text}")
                        return {
                            "client_id": request.client_id,
                            "response": f"Error: {response.status}, {error_text}",
                            "metadata": {
                                "error": True,
                                "status_code": response.status
                            }
                        }
        except Exception as e:
            logger.error(f"HTTP request failed: {e}")
            return {
                "client_id": request.client_id,
                "response": f"Error: {str(e)}",
                "metadata": {"error": True}
            }
    
    async def _query_websocket(self, request: MCPClientRequest) -> Dict[str, Any]:
        """Send query using WebSocket."""
        # Connect to WebSocket if not connected
        if not self._ws_connection:
            success = await self.connect_websocket()
            if not success:
                return {
                    "client_id": request.client_id,
                    "response": "Error: Failed to connect to WebSocket",
                    "metadata": {"error": True}
                }
        
        try:
            # Send request
            await self._ws_connection.send(json.dumps(request.model_dump()))
            
            # Receive response
            response_text = await self._ws_connection.recv()
            response_data = json.loads(response_text)
            
            return response_data
            
        except Exception as e:
            logger.error(f"WebSocket communication failed: {e}")
            
            # Try to reconnect and retry once
            try:
                await self.disconnect_websocket()
                await self.connect_websocket()
                
                # Retry request
                await self._ws_connection.send(json.dumps(request.model_dump()))
                response_text = await self._ws_connection.recv()
                response_data = json.loads(response_text)
                
                return response_data
                
            except Exception as retry_error:
                logger.error(f"WebSocket retry failed: {retry_error}")
                return {
                    "client_id": request.client_id,
                    "response": f"Error: {str(e)}, Retry failed: {str(retry_error)}",
                    "metadata": {"error": True}
                }
    
    def query_sync(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        client_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a query synchronously.
        
        Args:
            query: The query to process
            context: Additional context for the query
            client_id: Client identifier (generated if not provided)
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(
            self.query(query=query, context=context, client_id=client_id)
        )
    
    async def close(self):
        """Close any open connections."""
        await self.disconnect_websocket()
    
    def close_sync(self):
        """Close any open connections synchronously."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        loop.run_until_complete(self.close())
