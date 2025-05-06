#!/usr/bin/env python3
"""
Test script for the multi-agent automation system.
"""
import os
import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.main import MultiAgentManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_workflow():
    """Test the basic workflow of the multi-agent system."""
    logger.info("Testing basic multi-agent workflow")
    
    # Initialize the multi-agent manager
    manager = MultiAgentManager()
    
    # Run a simple data analysis task
    result = manager.run_task(
        "Analyze trending topics on Twitter related to artificial intelligence"
    )
    
    # Print the results
    logger.info("Task completed with results:")
    logger.info(f"Status: {result.get('result', {}).get('status', 'unknown')}")
    
    # Access some parts of the result
    findings = result.get("result", {}).get("results", {}).get("key_findings", [])
    logger.info(f"Key findings: {findings}")
    
    return result


def test_web_scraping():
    """Test the web scraping capabilities."""
    logger.info("Testing web scraping workflow")
    
    # Initialize with custom config
    manager = MultiAgentManager()
    
    # Run a web scraping task
    result = manager.run_task(
        "Collect the latest news headlines about machine learning from TechCrunch"
    )
    
    return result


def test_ui_automation():
    """Test the UI automation capabilities."""
    logger.info("Testing UI automation workflow")
    
    # Initialize with custom config
    manager = MultiAgentManager()
    
    # Run a UI automation task
    result = manager.run_task(
        "Log into example.com and extract user profile information"
    )
    
    return result


if __name__ == "__main__":
    # Run the tests
    test_result = test_basic_workflow()
    
    # Uncomment to run additional tests
    # test_web_scraping()
    # test_ui_automation()
    
    print("Tests completed successfully")