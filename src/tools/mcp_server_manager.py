#!/usr/bin/env python3
"""
MCP Server Manager - Handles MCP server configuration and management.
"""
import json
import os
import subprocess
import logging
from typing import Dict, Any, Optional, List, Tuple
import threading

# Configure logging
logger = logging.getLogger(__name__)

class MCPServerConfig:
    """Configuration for an MCP server."""
    def __init__(
        self,
        name: str,
        host: str = "localhost",
        port: int = 8000,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        internal: bool = False,
    ):
        """
        Initialize MCP server configuration.
        
        Args:
            name: Name identifier for the server
            host: Host address for the server
            port: Port number for the server
            command: Command to start an external server (None for internal servers)
            args: Arguments for the command
            internal: Whether this is an internal server (managed by the application)
        """
        self.name = name
        self.host = host
        self.port = port
        self.command = command
        self.args = args or []
        self.internal = internal
        self.process = None
        self.running = False

class MCPServerManager:
    """
    Manages MCP servers based on configuration.
    
    This manager is responsible for:
    1. Loading MCP server configurations from files
    2. Starting and stopping external MCP servers
    3. Managing server processes
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
    ):
        """
        Initialize the MCP Server Manager.
        
        Args:
            config_path: Path to the MCP configuration file
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            "mcp_config.json"
        )
        self.servers = {}
        self.default_server = None
        self._load_config()
        
    def _load_config(self):
        """Load MCP server configurations from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                # Load server configurations
                servers_config = config.get('mcpServers', {})
                for name, server_config in servers_config.items():
                    self.servers[name] = MCPServerConfig(
                        name=name,
                        host=server_config.get('host', 'localhost'),
                        port=server_config.get('port', 8000),
                        command=server_config.get('command'),
                        args=server_config.get('args', []),
                        internal=server_config.get('internal', False)
                    )
                
                # Set default server
                default_server_name = config.get('defaultServer')
                if default_server_name and default_server_name in self.servers:
                    self.default_server = self.servers[default_server_name]
                elif self.servers:
                    # Use first server as default if not specified
                    self.default_server = list(self.servers.values())[0]
                
                logger.info(f"Loaded {len(self.servers)} MCP server configurations")
            else:
                logger.warning(f"MCP config file not found at {self.config_path}, using default configuration")
                # Create a default internal server
                default_server = MCPServerConfig(
                    name="default",
                    host="0.0.0.0",
                    port=8000,
                    internal=True
                )
                self.servers["default"] = default_server
                self.default_server = default_server
                
        except Exception as e:
            logger.error(f"Error loading MCP configurations: {e}")
            # Create a default configuration on error
            default_server = MCPServerConfig(
                name="default",
                host="0.0.0.0",
                port=8000,
                internal=True
            )
            self.servers["default"] = default_server
            self.default_server = default_server
    
    def get_server_config(self, name: Optional[str] = None) -> MCPServerConfig:
        """
        Get MCP server configuration by name.
        
        Args:
            name: Server name (or None for default)
            
        Returns:
            Server configuration
        """
        if name and name in self.servers:
            return self.servers[name]
        return self.default_server
    
    def start_external_server(self, name: str) -> bool:
        """
        Start an external MCP server.
        
        Args:
            name: Server name
            
        Returns:
            True if server was started successfully, False otherwise
        """
        if name not in self.servers:
            logger.error(f"Server '{name}' not found in configuration")
            return False
            
        server = self.servers[name]
        
        if server.internal:
            logger.info(f"Server '{name}' is marked as internal, no need to start externally")
            return True
            
        if not server.command:
            logger.error(f"Server '{name}' has no command specified")
            return False
            
        if server.running:
            logger.info(f"Server '{name}' is already running")
            return True
            
        try:
            # Start the server process
            cmd = [server.command] + server.args
            logger.info(f"Starting external MCP server '{name}' with command: {' '.join(cmd)}")
            
            # Start in a new process group so it can be terminated properly
            server.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True
            )
            
            # Start a thread to monitor the process output
            def monitor_output():
                for line in server.process.stdout:
                    logger.info(f"[{name}] {line.strip()}")
                for line in server.process.stderr:
                    logger.error(f"[{name}] {line.strip()}")
            
            threading.Thread(target=monitor_output, daemon=True).start()
            
            server.running = True
            logger.info(f"External MCP server '{name}' started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting external MCP server '{name}': {e}")
            return False
    
    def stop_external_server(self, name: str) -> bool:
        """
        Stop an external MCP server.
        
        Args:
            name: Server name
            
        Returns:
            True if server was stopped successfully, False otherwise
        """
        if name not in self.servers:
            logger.error(f"Server '{name}' not found in configuration")
            return False
            
        server = self.servers[name]
        
        if server.internal:
            logger.info(f"Server '{name}' is marked as internal, no need to stop externally")
            return True
            
        if not server.running or not server.process:
            logger.info(f"Server '{name}' is not running")
            return True
            
        try:
            # Terminate the server process
            import os
            import signal
            
            # Send termination signal to process group
            os.killpg(os.getpgid(server.process.pid), signal.SIGTERM)
            
            # Wait for process to terminate
            server.process.wait(timeout=5)
            server.running = False
            server.process = None
            
            logger.info(f"External MCP server '{name}' stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping external MCP server '{name}': {e}")
            
            # Force kill if termination failed
            try:
                server.process.kill()
                server.running = False
                server.process = None
            except:
                pass
                
            return False
    
    def get_server_endpoint(self, name: Optional[str] = None) -> Tuple[str, int]:
        """
        Get the host and port for a server.
        
        Args:
            name: Server name (or None for default)
            
        Returns:
            Tuple of (host, port)
        """
        server = self.get_server_config(name)
        return server.host, server.port
    
    def stop_all_servers(self):
        """Stop all running external servers."""
        for name, server in self.servers.items():
            if server.running and not server.internal:
                self.stop_external_server(name)
