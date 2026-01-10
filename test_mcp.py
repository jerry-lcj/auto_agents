#!/usr/bin/env python3
"""
Test script for MCP (Model Context Protocol) integration.

This script tests the MCP client implementation by sending requests to configured
MCP servers. By default, it will attempt to connect to all configured servers in
the src/config/mcp_servers.json file.

Before running:
1. Make sure to update the API keys in src/config/mcp_servers.json
2. For CLI-based MCP servers, ensure you have the necessary tools installed
   (e.g., npx for Node.js based tools)

Usage:
    python test_mcp.py
"""
import asyncio
import json
import logging
import os
from pathlib import Path

# Add project root to python path if running as a script
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent))

from src.agents.mcp_agent import MCPAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Run the MCP test."""
    try:
        logger.info("Initializing MCP agent")
        agent = MCPAgent()
        
        available_tools = agent.get_available_tools()
        logger.info(f"Available MCP tools: {available_tools}")
        
        payload = {
            "messages": [{"role": "user", "content": "Find me some steak houses in San Francisco"}],
        }
        
        
        # Test Seekeasy MCP if available
        if "seekeasy" in available_tools:
            logger.info("Sending request to Seekeasy MCP (CLI)")
            try:
                result2 = await agent.run("seekeasy", payload)
                logger.info("Seekeasy result:")
                logger.info(json.dumps(result2, indent=2))
            except Exception as e:
                logger.error(f"Error running Seekeasy MCP: {e}")
        else:
            logger.info("Seekeasy MCP not available in configuration")
        
        #Close agent connections
        await agent.close()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
