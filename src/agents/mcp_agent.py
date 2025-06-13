#!/usr/bin/env python3
"""
MCP Agent - Responsible for Model Context Protocol communication.
"""
from typing import Any, Dict, List, Optional, Union
import asyncio
import json
import logging
import os
import requests
import aiohttp

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import StructuredMessage
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from pydantic import BaseModel, Field

from tools.mcp_server_manager import MCPServerManager

# Configure logging
logger = logging.getLogger(__name__)


class MCPRequest(BaseModel):
    """Model for MCP requests."""
    client_id: str
    query: str
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    

class MCPResponse(BaseModel):
    """Model for MCP responses."""
    client_id: str
    response: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class MCPAgent:
    """
    MCPAgent handles communication using the Model Context Protocol.
    
    This agent is responsible for:
    1. Serving as an MCP server or client
    2. Processing MCP requests from external applications
    3. Formatting agent responses according to the MCP standard
    4. Managing persistent connections with MCP clients
    """

    def __init__(
        self,
        name: str = "mcp_agent",
        llm_config: Optional[Dict[str, Any]] = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        server_name: Optional[str] = None,
        config_path: Optional[str] = None,
        use_external_server: bool = False,
    ):
        """
        Initialize the MCPAgent.
        
        Args:
            name: Name identifier for the agent
            llm_config: Configuration for the language model
            host: Host for the MCP server (ignored if using server from config)
            port: Port for the MCP server (ignored if using server from config)
            server_name: Name of MCP server to use from config
            config_path: Path to MCP server configuration file
            use_external_server: Whether to use an external MCP server
        """
        self.name = name
        self.llm_config = llm_config or {}
        self.active_connections: List[WebSocket] = []
        self.use_external_server = use_external_server
        self.server_name = server_name
        
        # Initialize server manager
        self.server_manager = MCPServerManager(config_path=config_path)
        
        # Determine which server to use
        if use_external_server or server_name:
            # Use specified server from config
            server_config = self.server_manager.get_server_config(server_name)
            self.host = server_config.host
            self.port = server_config.port
            self.external_server = not server_config.internal
            self.server_name = server_config.name
        else:
            # Use provided host/port
            self.host = host
            self.port = port
            self.external_server = False
        
        # Setup LLM client with the new API
        llm_client = OpenAIChatCompletionClient(**self.llm_config)
        
        # Setup the underlying AutoGen agent - fixed initialization
        # For AutoGen 0.5.6, we need to use config_list instead of llm_config
        self.agent = AssistantAgent(
            name=self.name,
            model_client=llm_client,
            system_message="""You are a Model Context Protocol agent that specializes in 
            interfacing with advanced language models using the MCP standard. You excel 
            at formulating precise queries, interpreting model responses, and managing 
            context effectively.""",
            # Use a format compatible with AutoGen 0.5.6
            
        )
        
        # Create FastAPI app for internal MCP server (if not using external)
        if not self.external_server:
            self.app = FastAPI(title="MCP Agent Server")
            self._setup_routes()
        else:
            self.app = None
        
        # Client registry
        self.clients = {}
        
    def _setup_routes(self):
        """Set up API routes for the MCP server."""
        
        @self.app.get("/")
        async def root():
            return {"status": "ok", "message": "MCP Agent server is running"}
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self._handle_websocket(websocket)
        
        @self.app.post("/query")
        async def query(request: MCPRequest):
            return await self._handle_query(request)
    
    async def _handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    request_data = json.loads(data)
                    request = MCPRequest(**request_data)
                    response = await self._process_request(request)
                    await websocket.send_text(response.model_dump_json())
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
                    await websocket.send_text(json.dumps({"error": str(e)}))
        except WebSocketDisconnect:
            self.active_connections.remove(websocket)
    
    async def _handle_query(self, request: MCPRequest) -> MCPResponse:
        """Handle HTTP query request."""
        return await self._process_request(request)
    
    async def _process_request(self, request: MCPRequest) -> MCPResponse:
        """
        Process an MCP request and generate a response.
        
        Args:
            request: MCP request object
            
        Returns:
            MCP response object
        """
        try:
            # Store client information if new
            if request.client_id not in self.clients:
                self.clients[request.client_id] = {
                    "last_active": asyncio.get_event_loop().time(),
                    "context": {}
                }
                
            # Update client context with new information
            self.clients[request.client_id]["context"].update(request.context)
            self.clients[request.client_id]["last_active"] = asyncio.get_event_loop().time()
            
            # Process the query
            result = await self.run(request.query, client_id=request.client_id, context=request.context)
            
            return MCPResponse(
                client_id=request.client_id,
                response=result["response"],
                metadata=result.get("metadata", {})
            )
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return MCPResponse(
                client_id=request.client_id,
                response=f"Error: {str(e)}",
                metadata={"error": True}
            )
    
    async def run(
        self, 
        query: str, 
        client_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a query using the MCP protocol.
        
        Args:
            query: User query to process
            client_id: Identifier for the client
            context: Additional context information
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing response and metadata
        """
        # If using external server, forward the query to it
        if self.external_server:
            return await self._forward_to_external_server(query, client_id, context)
        
        # TODO: Implement MCP processing logic with new API
        # In the new API, we would use structured messages for communication
        
        # For now, return a simple response from the internal implementation
        response = {
            "response": f"Processed query: {query}",
            "metadata": {
                "client_id": client_id,
                "timestamp": asyncio.get_event_loop().time(),
                "request_context_keys": list(context.keys()) if context else [],
                "server": "internal",
                "server_name": self.server_name
            }
        }
        
        return response
    
    async def _forward_to_external_server(
        self,
        query: str,
        client_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Forward a query to an external MCP server.
        
        Args:
            query: User query to process
            client_id: Identifier for the client
            context: Additional context information
            
        Returns:
            Response from the external server
        """
        try:
            # Ensure the external server is running
            if not self.server_manager.get_server_config(self.server_name).running:
                logger.info(f"Starting external MCP server: {self.server_name}")
                self.server_manager.start_external_server(self.server_name)
            
            # Construct the request payload
            payload = {
                "client_id": client_id or "auto_agent",
                "query": query,
                "context": context or {}
            }
            
            # Send the request to the external server
            url = f"http://{self.host}:{self.port}/query"
            logger.info(f"Forwarding query to external MCP server at {url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "response": result.get("response", "No response from external server"),
                            "metadata": {
                                **result.get("metadata", {}),
                                "server": "external",
                                "server_name": self.server_name
                            }
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"Error from external MCP server: {error_text}")
                        return {
                            "response": f"Error from external MCP server: {response.status}",
                            "metadata": {
                                "error": True,
                                "status_code": response.status,
                                "server": "external",
                                "server_name": self.server_name
                            }
                        }
        except Exception as e:
            logger.error(f"Error forwarding to external MCP server: {e}")
            return {
                "response": f"Failed to connect to external MCP server: {str(e)}",
                "metadata": {
                    "error": True,
                    "server": "external",
                    "server_name": self.server_name
                }
            }
        
    async def start_server(self):
        """Start the MCP server."""
        if self.external_server:
            # For external servers, just make sure it's running
            if self.server_manager.start_external_server(self.server_name):
                logger.info(f"External MCP server '{self.server_name}' started")
            else:
                logger.error(f"Failed to start external MCP server '{self.server_name}'")
        else:
            # For internal servers, start the FastAPI app
            config = uvicorn.Config(
                app=self.app, 
                host=self.host, 
                port=self.port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
        
    def start_server_sync(self):
        """Start the MCP server synchronously."""
        if self.external_server:
            # For external servers, just make sure it's running
            if self.server_manager.start_external_server(self.server_name):
                logger.info(f"External MCP server '{self.server_name}' started")
            else:
                logger.error(f"Failed to start external MCP server '{self.server_name}'")
        else:
            # For internal servers, start the FastAPI app
            import uvicorn
            uvicorn.run(self.app, host=self.host, port=self.port)
            
    def stop_server(self):
        """Stop the MCP server."""
        if self.external_server:
            # Stop external server
            if self.server_manager.stop_external_server(self.server_name):
                logger.info(f"External MCP server '{self.server_name}' stopped")
            else:
                logger.error(f"Failed to stop external MCP server '{self.server_name}'")
                
        
    async def connect_to_client(self, client_url: str):
        """
        Connect to an MCP client.
        
        Args:
            client_url: URL of the MCP client
            
        Returns:
            True if successful, False otherwise
        """
        if self.external_server:
            logger.info(f"Using external MCP server '{self.server_name}' for client connection to {client_url}")
            # External server handles client connections directly
            return True
            
        # TODO: Implement client connection logic for internal server
        logger.info(f"Connecting to MCP client at {client_url}")
        return True