#!/usr/bin/env python3
"""
Test script for MCP server configuration and management.
"""
import argparse
import os
import sys
import time

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tools.mcp_server_manager import MCPServerManager


def main():
    parser = argparse.ArgumentParser(description="MCP Server Manager Test")
    parser.add_argument("--config", type=str, default=None, help="Path to MCP configuration file")
    parser.add_argument("--server", type=str, default=None, help="Name of MCP server to test")
    parser.add_argument("--list", action="store_true", help="List available servers")
    parser.add_argument("--start", action="store_true", help="Start specified server")
    parser.add_argument("--stop", action="store_true", help="Stop specified server")
    args = parser.parse_args()
    
    # Create server manager
    manager = MCPServerManager(config_path=args.config)
    
    # List available servers
    if args.list:
        print("Available MCP servers:")
        for name, server in manager.servers.items():
            status = "running" if server.running else "stopped"
            server_type = "internal" if server.internal else "external"
            default_mark = " (default)" if server == manager.default_server else ""
            print(f"  - {name}{default_mark}: {server.host}:{server.port} ({server_type}, {status})")
        
        print(f"\nDefault server: {manager.default_server.name}")
        return
    
    # Get server configuration
    server_name = args.server
    server_config = manager.get_server_config(server_name)
    print(f"Using server: {server_config.name}")
    print(f"Host: {server_config.host}")
    print(f"Port: {server_config.port}")
    print(f"Type: {'internal' if server_config.internal else 'external'}")
    
    # Start server
    if args.start:
        if server_config.internal:
            print("Cannot start internal server directly")
        else:
            print(f"Starting server '{server_config.name}'...")
            success = manager.start_external_server(server_config.name)
            if success:
                print("Server started successfully")
                print("Press Ctrl+C to stop")
                try:
                    # Keep running until interrupted
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("Stopping server...")
                    manager.stop_external_server(server_config.name)
                    print("Server stopped")
            else:
                print("Failed to start server")
    
    # Stop server
    if args.stop:
        if server_config.internal:
            print("Cannot stop internal server directly")
        else:
            print(f"Stopping server '{server_config.name}'...")
            success = manager.stop_external_server(server_config.name)
            if success:
                print("Server stopped successfully")
            else:
                print("Failed to stop server")


if __name__ == "__main__":
    main()
