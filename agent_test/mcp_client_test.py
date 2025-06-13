#!/usr/bin/env python3
"""
Test script for MCP client functionality.
"""
import argparse
import asyncio
import json
import os
import sys
from typing import Dict, Any, Optional

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.mcp_agent import MCPAgent


async def test_mcp_client(
    query: str, 
    server_name: Optional[str] = None,
    config_path: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
):
    """Test MCP client functionality by sending a query to an MCP server."""
    # Initialize MCP agent
    use_external = bool(server_name)
    
    agent = MCPAgent(
        name="mcp_test_client",
        server_name=server_name,
        config_path=config_path,
        use_external_server=use_external
    )
    
    # Send query to MCP server
    print(f"Sending query to MCP server '{server_name or 'default'}':")
    print(f"  Query: {query}")
    print(f"  Context: {context}")
    
    try:
        result = await agent.run(
            query=query,
            client_id="mcp_client_test",
            context=context or {}
        )
        
        print("\nResponse:")
        print(json.dumps(result, indent=2))
        
        return result
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="MCP Client Test")
    parser.add_argument("--query", type=str, required=True, help="Query to send to MCP server")
    parser.add_argument("--server", type=str, default=None, help="Name of MCP server to use")
    parser.add_argument("--config", type=str, default=None, help="Path to MCP configuration file")
    parser.add_argument("--platform", type=str, default="twitter", help="Platform for context")
    parser.add_argument("--time-range", type=str, default="last_week", help="Time range for context")
    args = parser.parse_args()
    
    # Create context
    context = {
        "platform": args.platform,
        "time_range": args.time_range
    }
    
    # Run test
    asyncio.run(test_mcp_client(
        query=args.query,
        server_name=args.server,
        config_path=args.config,
        context=context
    ))


if __name__ == "__main__":
    main()
