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

import autogen
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
    ):
        """
        Initialize the multi-agent system with configuration options.

        Args:
            config_path: Path to configuration file (default: None)
            llm_config: Configuration for the language model (default: None)
            enable_mcp: Whether to enable the MCP server (default: False)
            mcp_host: Host for the MCP server (default: from env or 0.0.0.0)
            mcp_port: Port for the MCP server (default: from env or 8000)
        """
        # Load config from file if provided
        self.config = self._load_config(config_path) if config_path else {}
        
        # Use provided LLM config or set default from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables. LLM functionality may be limited.")
            
        self.llm_config = llm_config or {
            "config_list": [{"model": "gpt-4-turbo", "api_key": api_key}],
            "temperature": 0.1,
        }

        # Initialize the agent instances
        logger.info("Initializing agents...")
        self.planner = PlannerAgent(llm_config=self.llm_config)
        self.retriever = RetrieverAgent(llm_config=self.llm_config)
        self.executor = ExecutorAgent(llm_config=self.llm_config)
        self.critic = CriticAgent(llm_config=self.llm_config)
        self.ui_tool = UIToolAgent(llm_config=self.llm_config)
        
        # Initialize the MCP agent if enabled
        self.enable_mcp = enable_mcp
        self.mcp_host = mcp_host or os.getenv("MCP_HOST", "0.0.0.0")
        self.mcp_port = mcp_port or int(os.getenv("MCP_PORT", "8000"))
        self.mcp_server_thread = None
        
        if self.enable_mcp:
            logger.info(f"Initializing MCP agent on {self.mcp_host}:{self.mcp_port}...")
            self.mcp = MCPAgent(
                llm_config=self.llm_config,
                host=self.mcp_host,
                port=self.mcp_port
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
        # TODO: Implement agent-to-agent connections
        pass
        
    def _start_mcp_server(self) -> None:
        """Start the MCP server in a separate thread."""
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
        logger.info(f"MCP server started on {self.mcp_host}:{self.mcp_port}")
        
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
    # Example usage
    manager = MultiAgentManager(enable_mcp=True)
    
    # Run a regular task
    result = manager.run_task(
        "Analyze the trending topics on Twitter related to AI"
    )
    print("Regular task result:", result)
    
    # Run an MCP task synchronously
    mcp_result = manager.run_mcp_task_sync(
        "What are the trending AI topics on social media?",
        context={"platform": "twitter", "time_range": "last_week"}
    )
    print("MCP task result:", mcp_result)