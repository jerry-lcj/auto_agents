#!/usr/bin/env python3
"""
Example script showing how to use the MCP system with configuration.
"""
import argparse
import asyncio
import json
import os
import sys
from typing import Dict, Any, Optional

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import MultiAgentManager
from src.tools.enhanced_mcp_client import EnhancedMCPClient


async def demo_mcp_with_config():
    """Demonstrate MCP usage with configuration."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="MCP Demo with Config")
    parser.add_argument("--server", type=str, default=None, help="Name of MCP server from config")
    parser.add_argument("--config", type=str, default=None, help="Path to MCP configuration file")
    parser.add_argument("--query", type=str, default="What are the trending AI topics on social media?", 
                        help="Query to send to the MCP server")
    args = parser.parse_args()
    
    # Method 1: Using MultiAgentManager
    print("=== Method 1: Using MultiAgentManager ===")
    
    # Initialize with server from config
    manager = MultiAgentManager(
        enable_mcp=True,
        mcp_server_name=args.server,
        mcp_config_path=args.config
    )
    
    # Wait for server to start
    await asyncio.sleep(2)
    
    # Run a query
    result = manager.run_mcp_task_sync(
        args.query,
        context={"platform": "twitter", "time_range": "last_week"}
    )
    
    print(f"Query: {args.query}")
    print(f"Result: {json.dumps(result, indent=2)}")
    
    print("\n=== Method 2: Using EnhancedMCPClient ===")
    
    # Method 2: Using EnhancedMCPClient directly
    client = EnhancedMCPClient(
        server_name=args.server,
        config_path=args.config
    )
    
    # Send a query
    response = await client.query(
        query=args.query,
        context={"platform": "twitter", "time_range": "last_week"}
    )
    
    print(f"Query: {args.query}")
    print(f"Response: {json.dumps(response, indent=2)}")
    
    # Close the client
    await client.close()


if __name__ == "__main__":
    asyncio.run(demo_mcp_with_config())
