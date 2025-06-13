#!/usr/bin/env python3
"""
Main entry point for the Auto Agents platform.
Contains the MultiAgentManager that orchestrates all agent interactions.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Union
import asyncio
import threading

# Try to handle imports correctly regardless of how the script is called
try:
    # Try direct imports first - newer AutoGen version
    from autogen.agentchat.assistant import AssistantAgent
    from autogen.agentchat.user_proxy import UserProxyAgent
    from autogen.oai import OpenAIWrapper
except ImportError:
    # Fall back to autogen_agentchat - older versions or different package name
    try:
        from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
        from autogen_ext.models.openai import OpenAIChatCompletionClient as OpenAIWrapper
    except ImportError:
        # If still failing, raise a helpful error
        raise ImportError("Could not import AutoGen components. Please ensure AutoGen is installed correctly.")

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from agents.critic_agent import CriticAgent
from agents.executor_agent import ExecutorAgent
from agents.planner_agent import PlannerAgent
from agents.retriever_agent import RetrieverAgent
from agents.ui_tool_agent import UIToolAgent
from agents.mcp_agent import MCPAgent

# Configure logging
logging_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, logging_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MultiAgentManager:
    """
    Manages multiple specialized agents and their interactions for
    data analysis and automation tasks.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        enable_mcp: bool = False,
        mcp_host: Optional[str] = None,
        mcp_port: Optional[int] = None,
        mcp_server_name: Optional[str] = None,
        mcp_config_path: Optional[str] = None,
    ):
        """
        Initialize the multi-agent system with configuration options.

        Args:
            config_path: Path to configuration file (default: None)
            llm_config: Configuration for the language model (default: None)
            enable_mcp: Whether to enable the MCP server (default: False)
            mcp_host: Host for the MCP server (default: from env or 0.0.0.0)
            mcp_port: Port for the MCP server (default: from env or 8000)
            mcp_server_name: Name of MCP server from config to use (default: None)
            mcp_config_path: Path to MCP server configuration file (default: None)
        """
        # Load config from file if provided
        self.config = self._load_config(config_path) if config_path else {}
        
        # Use provided LLM config or set default from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables. LLM functionality may be limited.")
        
        # Prepare configuration for LLM
        client_config = {
            "api_key": api_key,
            "model": "gpt-4-turbo",
            "temperature": 0.1,
        }
        
        # Override with any provided config
        if llm_config:
            client_config.update(llm_config)
            
        self.llm_config = client_config

        # Initialize the agent instances
        logger.info("Initializing agents...")
        self.planner = PlannerAgent(llm_config=self.llm_config)
        self.retriever = RetrieverAgent(llm_config=self.llm_config)
        self.executor = ExecutorAgent(llm_config=self.llm_config)
        self.critic = CriticAgent(llm_config=self.llm_config)
        self.ui_tool = UIToolAgent(llm_config=self.llm_config)
        
        # Create a user proxy agent for interactive usage with flexible initialization 
        try:
            # Try newer AutoGen version approach
            self.user_proxy = UserProxyAgent(name="user_proxy")
        except (TypeError, ValueError):
            # Fall back to a more basic approach for older versions
            self.user_proxy = UserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER"
            )
        
        # Initialize the MCP agent if enabled
        self.enable_mcp = enable_mcp
        self.mcp_host = mcp_host or os.getenv("MCP_HOST", "0.0.0.0")
        self.mcp_port = mcp_port or int(os.getenv("MCP_PORT", "8000"))
        self.mcp_server_name = mcp_server_name or os.getenv("MCP_SERVER", None)
        self.mcp_config_path = mcp_config_path
        self.mcp_server_thread = None
        
        if self.enable_mcp:
            # Check if we're using a named server configuration or direct host/port
            use_external_server = bool(self.mcp_server_name)
            
            if use_external_server:
                logger.info(f"Initializing MCP agent with configured server '{self.mcp_server_name}'...")
            else:
                logger.info(f"Initializing MCP agent on {self.mcp_host}:{self.mcp_port}...")
                
            self.mcp = MCPAgent(
                llm_config=self.llm_config,
                host=self.mcp_host,
                port=self.mcp_port,
                server_name=self.mcp_server_name,
                config_path=self.mcp_config_path,
                use_external_server=use_external_server
            )
        else:
            self.mcp = None

        # Create agent-to-agent connections
        self._setup_agent_connections()

        logger.info("Multi-agent system initialized successfully")
        
        # Start MCP server if enabled
        if self.enable_mcp:
            self._start_mcp_server()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        # TODO: Implement config loading from JSON/YAML
        return {}
    
    def _setup_agent_connections(self) -> None:
        """Set up connections and communication between agents."""
        # In the new AutoGen API, we connect agents using register_reply
        # This allows structured communication between agents
        
        # Example connection setup (to be expanded based on your workflow)
        # self.user_proxy.register_reply(
        #    [self.planner.agent],
        #    lambda msg: True  # Accept all messages
        # )
        
        # For now, we'll rely on the MultiAgentManager to coordinate
        pass
        
    def _start_mcp_server(self) -> None:
        """Start the MCP server in a separate thread."""
        # Check if we're using an external server
        if hasattr(self.mcp, 'external_server') and self.mcp.external_server:
            logger.info(f"Using external MCP server '{self.mcp.server_name}'")
            # Start external server if needed
            if self.mcp.server_manager.start_external_server(self.mcp.server_name):
                logger.info(f"External MCP server '{self.mcp.server_name}' started")
            else:
                logger.error(f"Failed to start external MCP server '{self.mcp.server_name}'")
        else:
            # Start internal server in a separate thread
            def run_server():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.mcp.start_server())
                
            self.mcp_server_thread = threading.Thread(
                target=run_server,
                daemon=True,
                name="mcp-server"
            )
            self.mcp_server_thread.start()
            logger.info(f"Internal MCP server started on {self.mcp_host}:{self.mcp_port}")
        
    def run_task(self, task_description: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a task using the multi-agent system.

        Args:
            task_description: Natural language description of the task
            **kwargs: Additional task parameters

        Returns:
            Dictionary containing task results and metadata
        """
        logger.info(f"Starting task: {task_description}")
        
        # 1. Planner creates task plan
        plan = self.planner.run(task_description, **kwargs)
        
        # 2. Retriever collects necessary data
        data = self.retriever.run(plan=plan, **kwargs)
        
        # 3. Executor performs the task
        result = self.executor.run(plan=plan, data=data, **kwargs)
        
        # 4. Critic evaluates the results
        evaluation = self.critic.run(
            plan=plan, data=data, result=result, **kwargs
        )
        
        return {
            "task": task_description,
            "plan": plan,
            "data": data,
            "result": result,
            "evaluation": evaluation,
        }
    
    def run_chat_task(self, task_description: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a task using the AutoGen chat protocol.
        
        This uses a more basic approach that should be compatible with various AutoGen versions.
        
        Args:
            task_description: Natural language description of the task
            **kwargs: Additional task parameters
            
        Returns:
            Dictionary containing chat results
        """
        logger.info(f"Starting chat task: {task_description}")
        
        try:
            # In the current AutoGen version, we need to use the chat method
            # The method returns a response that we can use directly
            response = self.planner.agent.generate(task_description)
            
            # Get the final result and format it
            result = {
                "task": task_description,
                "conversation": response.get("content", "No content received"),
                "status": "success" 
            }
            
        except Exception as e:
            logger.error(f"Error in chat task: {e}")
            result = {
                "task": task_description,
                "conversation": f"Error: {str(e)}",
                "status": "error"
            }
            
        return result
        
    async def run_mcp_task(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a task via MCP protocol.
        
        Args:
            query: MCP query to process
            context: Additional context for the query
            
        Returns:
            Dictionary containing task results formatted for MCP
        """
        if not self.mcp:
            raise RuntimeError("MCP is not enabled. Initialize with enable_mcp=True.")
            
        # Process the query through the MCP agent
        result = await self.mcp.run(query=query, context=context or {})
        return result
        
    def run_mcp_task_sync(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a task via MCP protocol (synchronous version).
        
        Args:
            query: MCP query to process
            context: Additional context for the query
            
        Returns:
            Dictionary containing task results formatted for MCP
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(self.run_mcp_task(query, context))


if __name__ == "__main__":
    # Example usage with command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto Agents Platform")
    parser.add_argument("--enable-mcp", action="store_true", help="Enable MCP server")
    parser.add_argument("--mcp-host", type=str, help="MCP server host")
    parser.add_argument("--mcp-port", type=int, help="MCP server port")
    parser.add_argument("--mcp-server", type=str, help="MCP server name from config")
    parser.add_argument("--mcp-config", type=str, help="Path to MCP configuration file")
    parser.add_argument("--query", type=str, default="What are the trending AI topics on social media?",
                      help="Query to send to the system")
    args = parser.parse_args()
    
    # Create manager with provided arguments
    manager = MultiAgentManager(
        enable_mcp=args.enable_mcp or True,
        mcp_host=args.mcp_host,
        mcp_port=args.mcp_port,
        mcp_server_name=args.mcp_server,
        mcp_config_path=args.mcp_config
    )
    
    # Run a regular task
    result = manager.run_task(
        "Analyze the trending topics on Twitter related to AI"
    )
    print("Regular task result:", result)
    
    # Run an MCP task synchronously
    mcp_result = manager.run_mcp_task_sync(
        args.query,
        context={"platform": "twitter", "time_range": "last_week"}
    )
    print("MCP task result:", mcp_result)