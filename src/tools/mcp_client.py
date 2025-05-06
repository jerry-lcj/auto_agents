#!/usr/bin/env python3
"""
MCP Client Tool - Provides client functionality for Model Context Protocol.
"""
from typing import Any, Dict, List, Optional, Union
import asyncio
import json
import logging
import uuid

import aiohttp
import httpx
import websockets
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)


class MCPMessage(BaseModel):
    """Base model for MCP messages."""
    client_id: str
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=asyncio.get_event_loop().time)


class MCPQuery(MCPMessage):
    """Model for MCP query messages."""
    query: str
    context: Dict[str, Any] = Field(default_factory=dict)


class MCPResponse(MCPMessage):
    """Model for MCP response messages."""
    response: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MCPClientTool:
    """
    Tool for connecting to MCP-compatible services.
    
    This tool provides:
    1. HTTP and WebSocket connections to MCP servers
    2. Standardized message formatting
    3. Handling of MCP responses
    4. Management of connection state and context
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        server_url: Optional[str] = None,
    ):
        """
        Initialize the MCPClientTool.
        
        Args:
            client_id: Unique identifier for this client
            server_url: URL of the MCP server to connect to
        """
        self.client_id = client_id or f"mcp-client-{uuid.uuid4()}"
        self.server_url = server_url
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.websocket = None
        self.context: Dict[str, Any] = {}
        
    async def connect(self, server_url: Optional[str] = None) -> bool:
        """
        Connect to an MCP server via WebSocket.
        
        Args:
            server_url: URL of the MCP server to connect to
            
        Returns:
            True if connection successful, False otherwise
        """
        if server_url:
            self.server_url = server_url
            
        if not self.server_url:
            logger.error("No server URL provided")
            return False
            
        ws_url = self.server_url
        if not ws_url.startswith(("ws://", "wss://")):
            # Convert HTTP URL to WebSocket URL if needed
            ws_url = ws_url.replace("http://", "ws://").replace("https://", "wss://")
            if ws_url.endswith("/"):
                ws_url += "ws"
            else:
                ws_url += "/ws"
                
        try:
            self.websocket = await websockets.connect(ws_url)
            logger.info(f"Connected to MCP server at {ws_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            return False
            
    async def disconnect(self) -> bool:
        """
        Disconnect from the MCP server.
        
        Returns:
            True if successful, False otherwise
        """
        if self.websocket:
            try:
                await self.websocket.close()
                self.websocket = None
                logger.info("Disconnected from MCP server")
                return True
            except Exception as e:
                logger.error(f"Error disconnecting from MCP server: {e}")
                
        return False
        
    async def send_query_ws(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[MCPResponse]:
        """
        Send a query to the MCP server via WebSocket.
        
        Args:
            query: Query to send
            context: Additional context to include
            
        Returns:
            MCP response if successful, None otherwise
        """
        if not self.websocket:
            connected = await self.connect()
            if not connected:
                logger.error("Not connected to MCP server")
                return None
                
        # Create query message
        msg_context = self.context.copy()
        if context:
            msg_context.update(context)
            
        message = MCPQuery(
            client_id=self.client_id,
            query=query,
            context=msg_context
        )
        
        try:
            # Send query
            await self.websocket.send(message.model_dump_json())
            
            # Wait for response
            response_text = await self.websocket.recv()
            response_data = json.loads(response_text)
            
            return MCPResponse(**response_data)
        except Exception as e:
            logger.error(f"Error communicating with MCP server: {e}")
            return None
            
    async def send_query_http(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[MCPResponse]:
        """
        Send a query to the MCP server via HTTP.
        
        Args:
            query: Query to send
            context: Additional context to include
            
        Returns:
            MCP response if successful, None otherwise
        """
        if not self.server_url:
            logger.error("No server URL provided")
            return None
            
        # Ensure server URL has /query endpoint
        api_url = self.server_url
        if api_url.endswith("/"):
            api_url += "query"
        else:
            api_url += "/query"
            
        # Create query message
        msg_context = self.context.copy()
        if context:
            msg_context.update(context)
            
        message = {
            "client_id": self.client_id,
            "query": query,
            "context": msg_context
        }
        
        try:
            # Send query
            response = await self.http_client.post(
                api_url,
                json=message
            )
            response.raise_for_status()
            
            return MCPResponse(**response.json())
        except Exception as e:
            logger.error(f"Error communicating with MCP server: {e}")
            return None
            
    def add_context(self, key: str, value: Any) -> None:
        """
        Add an item to the persistent context.
        
        Args:
            key: Context key
            value: Context value
        """
        self.context[key] = value
        
    def remove_context(self, key: str) -> None:
        """
        Remove an item from the persistent context.
        
        Args:
            key: Context key to remove
        """
        if key in self.context:
            del self.context[key]
            
    def clear_context(self) -> None:
        """Clear all persistent context."""
        self.context = {}
        
    async def close(self) -> None:
        """Close all connections and clean up resources."""
        await self.disconnect()
        await self.http_client.aclose()
        
    # Synchronous versions of async methods using asyncio
    def send_query(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        use_websocket: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Send a query to the MCP server (synchronous version).
        
        Args:
            query: Query to send
            context: Additional context to include
            use_websocket: Whether to use WebSocket instead of HTTP
            
        Returns:
            Response dictionary if successful, None otherwise
        """
        if use_websocket:
            coro = self.send_query_ws(query, context)
        else:
            coro = self.send_query_http(query, context)
            
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        try:
            response = loop.run_until_complete(coro)
            if response:
                return response.model_dump()
            return None
        except Exception as e:
            logger.error(f"Error in send_query: {e}")
            return None