#!/usr/bin/env python3
"""
GPT-4o Agent Implementation
This agent utilizes OpenAI's GPT-4o model for advanced reasoning capabilities.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Union

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import StructuredMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class GPT4OAgent:
    """
    Agent implementation that uses OpenAI's GPT-4o model.
    Provides a high-performance model for complex reasoning tasks.
    """

    def __init__(
        self,
        name: str = "gpt4o",
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        system_message: str = "You are a helpful AI assistant that uses tools to solve tasks.",
        tools: Optional[List[Any]] = None,
        api_key: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the GPT-4o agent.

        Args:
            name: Name of the agent (default: "gpt4o")
            model: Model to use (default: "gpt-4o")
            temperature: Temperature for model sampling (default: 0.1)
            max_tokens: Maximum tokens for response (default: None)
            system_message: System message for the agent (default: "You are a helpful AI assistant...")
            tools: List of tools the agent can use (default: None)
            api_key: OpenAI API key (default: None, will use environment variable)
            llm_config: Additional LLM configuration (default: None)
        """
        # Use provided API key or get from environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided and OPENAI_API_KEY not found in environment variables")

        # Prepare model client configuration
        client_config = {
            "api_key": self.api_key,
            "model": model,
            "temperature": temperature,
        }
        if max_tokens:
            client_config["max_tokens"] = max_tokens

        # Override with provided config
        if llm_config:
            client_config.update(llm_config)

        # Initialize OpenAI model client
        self.model_client = OpenAIChatCompletionClient(**client_config)

        # Initialize the agent with the model client
        self.agent = AssistantAgent(
            name=name,
            model_client=self.model_client,
            system_message=system_message,
            tools=tools or [],
        )

        logger.info(f"GPT-4o agent '{name}' initialized with model '{model}'")

    def run(self, task_description: str, **kwargs):
        """
        Run the agent on a task.

        Args:
            task_description: Natural language description of the task
            **kwargs: Additional task parameters

        Returns:
            Response from the agent
        """
        logger.info(f"Running GPT-4o agent on task: {task_description}")
        
        # Process the task with the agent
        # The implementation depends on how you want to structure interactions
        # This is a simplified version
        response = self.agent.chat(task_description)
        
        return response

    # Add any additional methods for more complex interaction patterns