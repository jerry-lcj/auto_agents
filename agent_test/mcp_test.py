#!/usr/bin/env python3
"""
Test script for MCP (Model Context Protocol) integration.
"""
import os
import sys
import logging
import asyncio
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.main import MultiAgentManager
from src.tools.mcp_client import MCPClientTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def test_mcp_server():
    """Test the MCP server functionality."""
    logger.info("Starting MCP server test")
    
    # Start the Multi-Agent Manager with MCP enabled
    manager = MultiAgentManager(
        enable_mcp=True,
        mcp_port=8000
    )
    
    # Give the server time to start
    await asyncio.sleep(2)
    
    # Create an MCP client to connect to the server
    client = MCPClientTool(
        client_id="test-client-1",
        server_url="http://localhost:8000"
    )
    
    # Add some context data
    client.add_context("source", "test_script")
    client.add_context("user", "test_user")
    
    # Test HTTP-based query
    logger.info("Sending HTTP query to MCP server")
    response = await client.send_query_http(
        query="What are the latest trends in AI?",
        context={"domain": "artificial intelligence"}
    )
    
    if response:
        logger.info(f"Received response: {response.response}")
        logger.info(f"Response metadata: {response.metadata}")
    else:
        logger.error("No response received from HTTP query")
    
    # Test WebSocket-based query
    logger.info("Sending WebSocket query to MCP server")
    try:
        # Connect to WebSocket
        connected = await client.connect()
        if connected:
            ws_response = await client.send_query_ws(
                query="How can machine learning improve healthcare?",
                context={"domain": "healthcare", "perspective": "technical"}
            )
            
            if ws_response:
                logger.info(f"Received WebSocket response: {ws_response.response}")
                logger.info(f"WebSocket response metadata: {ws_response.metadata}")
            else:
                logger.error("No response received from WebSocket query")
                
            # Disconnect
            await client.disconnect()
        else:
            logger.error("Failed to connect via WebSocket")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    
    # Clean up
    await client.close()
    logger.info("MCP test completed")


def test_mcp_sync():
    """Test the synchronous MCP functionality."""
    logger.info("Starting synchronous MCP test")
    
    # Start the Multi-Agent Manager with MCP enabled
    manager = MultiAgentManager(
        enable_mcp=True,
        mcp_port=8001  # Use a different port for this test
    )
    
    # Use the manager's synchronous MCP interface
    response = manager.run_mcp_task_sync(
        query="Give me a summary of recent developments in quantum computing",
        context={
            "max_length": 200,
            "technical_level": "intermediate",
            "include_references": True
        }
    )
    
    logger.info(f"MCP response: {json.dumps(response, indent=2)}")
    
    # Test using the client in synchronous mode
    client = MCPClientTool(
        client_id="test-sync-client",
        server_url="http://localhost:8001"
    )
    
    client_response = client.send_query(
        query="What are the practical applications of GPT models?",
        context={"domain": "natural language processing"},
        use_websocket=False  # Use HTTP
    )
    
    if client_response:
        logger.info(f"Client response: {json.dumps(client_response, indent=2)}")
    else:
        logger.error("No response received from client")
        
    logger.info("Synchronous MCP test completed")


if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_mcp_server())
    
    # Run the sync test
    test_mcp_sync()