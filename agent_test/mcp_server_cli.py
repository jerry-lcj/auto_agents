#!/usr/bin/env python3
"""
MCP Server Management CLI - Command line tool for managing MCP servers.
"""
import argparse
import os
import sys
import json
import time
import logging

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tools.mcp_server_manager import MCPServerManager, MCPServerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mcp-cli")


def list_servers(manager: MCPServerManager):
    """List all available MCP servers."""
    print("Available MCP servers:")
    for name, server in manager.servers.items():
        status = "running" if server.running else "stopped"
        server_type = "internal" if server.internal else "external"
        default_mark = " (default)" if server == manager.default_server else ""
        command = f"{server.command} {' '.join(server.args)}" if server.command else "N/A"
        
        print(f"  - {name}{default_mark}: {server.host}:{server.port}")
        print(f"    Type: {server_type}, Status: {status}")
        if not server.internal:
            print(f"    Command: {command}")
    
    print(f"\nDefault server: {manager.default_server.name}")


def start_server(manager: MCPServerManager, name: str, keep_running: bool = False):
    """Start an MCP server."""
    server_config = manager.get_server_config(name)
    
    if server_config.internal:
        print(f"Server '{name}' is internal and cannot be started directly.")
        return
    
    print(f"Starting MCP server '{name}'...")
    success = manager.start_external_server(name)
    
    if success:
        print(f"MCP server '{name}' started successfully.")
        
        if keep_running:
            print("Press Ctrl+C to stop the server...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping server...")
                manager.stop_external_server(name)
                print(f"MCP server '{name}' stopped.")
    else:
        print(f"Failed to start MCP server '{name}'.")


def stop_server(manager: MCPServerManager, name: str):
    """Stop an MCP server."""
    server_config = manager.get_server_config(name)
    
    if server_config.internal:
        print(f"Server '{name}' is internal and cannot be stopped directly.")
        return
    
    print(f"Stopping MCP server '{name}'...")
    success = manager.stop_external_server(name)
    
    if success:
        print(f"MCP server '{name}' stopped successfully.")
    else:
        print(f"Failed to stop MCP server '{name}'.")


def add_server(manager: MCPServerManager, config_path: str, name: str, host: str, port: int, 
               command: str = None, args: str = None, internal: bool = False):
    """Add a new MCP server to the configuration."""
    # Load current config
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        config = {"mcpServers": {}, "defaultServer": None}
    
    # Add new server
    server_config = {
        "host": host,
        "port": port
    }
    
    if internal:
        server_config["internal"] = True
    elif command:
        server_config["command"] = command
        if args:
            server_config["args"] = args.split()
    
    # Update config
    config["mcpServers"][name] = server_config
    
    # Set as default if first server
    if not config["defaultServer"]:
        config["defaultServer"] = name
    
    # Save config
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Added MCP server '{name}' to configuration.")
    if not config["defaultServer"]:
        print(f"Set '{name}' as default server.")


def remove_server(manager: MCPServerManager, config_path: str, name: str):
    """Remove an MCP server from the configuration."""
    # Load current config
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Configuration file not found: {config_path}")
        return
    
    # Check if server exists
    if name not in config.get("mcpServers", {}):
        print(f"Server '{name}' not found in configuration.")
        return
    
    # Remove server
    del config["mcpServers"][name]
    
    # Update default server if needed
    if config.get("defaultServer") == name:
        if config["mcpServers"]:
            config["defaultServer"] = next(iter(config["mcpServers"]))
        else:
            config["defaultServer"] = None
    
    # Save config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Removed MCP server '{name}' from configuration.")


def set_default(manager: MCPServerManager, config_path: str, name: str):
    """Set the default MCP server."""
    # Load current config
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Configuration file not found: {config_path}")
        return
    
    # Check if server exists
    if name not in config.get("mcpServers", {}):
        print(f"Server '{name}' not found in configuration.")
        return
    
    # Set default
    config["defaultServer"] = name
    
    # Save config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Set '{name}' as default MCP server.")


def main():
    parser = argparse.ArgumentParser(description="MCP Server Management CLI")
    parser.add_argument("--config", type=str, help="Path to MCP configuration file")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List servers command
    list_parser = subparsers.add_parser("list", help="List available servers")
    
    # Start server command
    start_parser = subparsers.add_parser("start", help="Start a server")
    start_parser.add_argument("name", type=str, help="Name of server to start")
    start_parser.add_argument("--keep-running", "-k", action="store_true", help="Keep running until Ctrl+C")
    
    # Stop server command
    stop_parser = subparsers.add_parser("stop", help="Stop a server")
    stop_parser.add_argument("name", type=str, help="Name of server to stop")
    
    # Add server command
    add_parser = subparsers.add_parser("add", help="Add a new server")
    add_parser.add_argument("name", type=str, help="Name of server to add")
    add_parser.add_argument("--host", type=str, default="localhost", help="Host address")
    add_parser.add_argument("--port", type=int, default=8000, help="Port number")
    add_parser.add_argument("--command", type=str, help="Command to start server")
    add_parser.add_argument("--args", type=str, help="Command arguments as a space-separated string")
    add_parser.add_argument("--internal", action="store_true", help="Mark as internal server")
    
    # Remove server command
    remove_parser = subparsers.add_parser("remove", help="Remove a server")
    remove_parser.add_argument("name", type=str, help="Name of server to remove")
    
    # Set default command
    default_parser = subparsers.add_parser("set-default", help="Set default server")
    default_parser.add_argument("name", type=str, help="Name of server to set as default")
    
    args = parser.parse_args()
    
    # Get default config path if not specified
    if not args.config:
        args.config = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            "mcp_config.json"
        )
    
    # Initialize manager
    manager = MCPServerManager(config_path=args.config)
    
    # Execute command
    if args.command == "list":
        list_servers(manager)
    elif args.command == "start":
        start_server(manager, args.name, args.keep_running)
    elif args.command == "stop":
        stop_server(manager, args.name)
    elif args.command == "add":
        add_server(
            manager, 
            args.config, 
            args.name, 
            args.host, 
            args.port, 
            args.command, 
            args.args, 
            args.internal
        )
    elif args.command == "remove":
        remove_server(manager, args.config, args.name)
    elif args.command == "set-default":
        set_default(manager, args.config, args.name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
