#!/usr/bin/env python3
"""
Test script for GPT-4o Agent
This script demonstrates how to use the GPT-4o agent for various tasks.
"""
import os
import sys
import logging
from dotenv import load_dotenv

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.gpt4o_agent import GPT4OAgent
from autogen_ext.agentchat.webutils import WebSearchTool

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def main():
    """Main function to demonstrate GPT-4o agent capabilities"""
    
    # Create a web search tool
    web_search = WebSearchTool()
    
    # Create the GPT-4o agent with web search capability
    agent = GPT4OAgent(
        name="gpt4o_assistant",
        model="gpt-4o",  # Using the GPT-4o model
        temperature=0.1,
        system_message="You are a helpful AI assistant with access to web search. Use tools to provide accurate, up-to-date information.",
        tools=[web_search.as_tool()],
    )
    
    # Example task to test the agent
    task = "What were the major technology announcements in the first quarter of 2025? Provide a brief summary."
    
    print(f"\n\n{'='*50}")
    print(f"Running GPT-4o agent on task: {task}")
    print(f"{'='*50}\n")
    
    # Run the agent
    response = agent.run(task)
    
    print(f"\n\n{'='*50}")
    print("Agent response:")
    print(f"{'='*50}")
    print(response)

if __name__ == "__main__":
    main()