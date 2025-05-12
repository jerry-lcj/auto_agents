#!/usr/bin/env python3
"""
MCP Agent - Responsible for Model Context Protocol communication.
"""
from typing import Any, Dict, List, Optional, Union
import asyncio
import json
import logging
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import StructuredMessage
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from pydantic import BaseModel, Field

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
    ):
        """
        Initialize the MCPAgent.
        
        Args:
            name: Name identifier for the agent
            llm_config: Configuration for the language model
            host: Host for the MCP server
            port: Port for the MCP server
        """
        self.name = name
        self.llm_config = llm_config or {}
        self.host = host
        self.port = port
        self.active_connections: List[WebSocket] = []
        
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
        
        # Create FastAPI app for MCP server
        self.app = FastAPI(title="MCP Agent Server")
        self._setup_routes()
        
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
        # TODO: Implement MCP processing logic with new API
        # In the new API, we would use structured messages for communication
        
        # For now, return a simple response
        response = {
            "response": f"Processed query: {query}",
            "metadata": {
                "client_id": client_id,
                "timestamp": asyncio.get_event_loop().time(),
                "request_context_keys": list(context.keys()) if context else []
            }
        }
        
        return response
        
    async def start_server(self):
        """Start the MCP server."""
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
        import uvicorn
        uvicorn.run(self.app, host=self.host, port=self.port)
        
    async def connect_to_client(self, client_url: str):
        """
        Connect to an MCP client.
        
        Args:
            client_url: URL of the MCP client
            
        Returns:
            True if successful, False otherwise
        """
        # TODO: Implement client connection logic
        return True