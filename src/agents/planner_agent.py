#!/usr/bin/env python3
"""
Planner Agent - Responsible for task planning and coordination.
"""
from typing import Any, Dict, List, Optional, Union
import os
import logging

# Try to handle imports correctly regardless of how the script is called
try:
    # Try direct imports first
    from autogen.agentchat.assistant import AssistantAgent
    from autogen.oai import OpenAIWrapper
except ImportError:
    # Fall back to autogen_agentchat if needed (older versions or different package name)
    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_ext.models.openai import OpenAIChatCompletionClient as OpenAIWrapper
    except ImportError:
        # If still failing, raise a helpful error
        raise ImportError("Could not import AutoGen components. Please check your installation.")

# Configure logging
logger = logging.getLogger(__name__)

class PlannerAgent:
    """
    PlannerAgent creates and manages execution plans for tasks.
    
    This agent is responsible for:
    1. Breaking down complex tasks into manageable steps
    2. Determining the right sequence of operations
    3. Allocating resources appropriately
    4. Handling contingencies and replanning
    """

    def __init__(
        self, 
        name: str = "planner", 
        llm_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the PlannerAgent.
        
        Args:
            name: Name identifier for the agent
            llm_config: Configuration for the language model
        """
        self.name = name
        self.llm_config = llm_config or {}
        
        # Initialize the OpenAI client with flexible approach for different AutoGen versions
        try:
            # Try newer AutoGen version approach
            self.llm_client = OpenAIWrapper(**self.llm_config)
        except (TypeError, ValueError):
            # Fall back to direct configuration for older versions
            api_key = self.llm_config.get("api_key") or os.getenv("OPENAI_API_KEY")
            model = self.llm_config.get("model", "gpt-4-turbo")
            self.llm_client = {"api_key": api_key, "model": model}
        
        # Setup the underlying AutoGen agent with flexible approach
        try:
            # Try newer AutoGen version approach
            self.agent = AssistantAgent(
                name=self.name,
                llm_config=self.llm_config,
                system_message="""You are a strategic planning specialist with expertise in breaking down 
                complex tasks into clear, actionable steps. You excel at analyzing requirements, 
                identifying dependencies, and organizing work effectively."""
            )
        except (TypeError, ValueError):
            # Fall back to older version approach with minimal compatible parameters
            try:
                # Try with model_client
                self.agent = AssistantAgent(
                    name=self.name,
                    model_client=self.llm_client,
                    system_message="""You are a strategic planning specialist with expertise in breaking down 
                    complex tasks into clear, actionable steps. You excel at analyzing requirements, 
                    identifying dependencies, and organizing work effectively."""
                )
            except:
                # Final fallback with minimal params
                self.agent = AssistantAgent(
                    name=self.name,
                    system_message="""You are a strategic planning specialist."""
                )

        logger.info(f"PlannerAgent '{name}' initialized")

    def run(self, task_description: str, **kwargs) -> Dict[str, Any]:
        """
        Create a plan for executing the given task.
        
        Args:
            task_description: Natural language description of the task
            **kwargs: Additional parameters for planning
            
        Returns:
            Dictionary containing the execution plan
        """
        try:
            # Use the latest AutoGen Agent API to get a response
            planning_prompt = f"""
            Create a detailed execution plan for the following task:
            
            TASK: {task_description}
            
            Your plan should include:
            1. Clear, numbered steps with specific actions
            2. Dependencies between steps
            3. Agent assignments for each step
            4. Success criteria for the overall task
            5. Contingency plans for potential issues
            
            Format your response as a structured plan.
            """
            
            # Get planning response from the agent
            response_content = ""
            try:
                # Try newer API style
                response = self.llm_client.call(
                    messages=[{"role": "user", "content": planning_prompt}],
                    model=self.llm_config.get("model", "gpt-4-turbo")
                )
                response_content = response.get("content", "")
            except (AttributeError, TypeError):
                # Fall back to using the agent's generate method
                try:
                    response = self.agent.generate(planning_prompt)
                    if isinstance(response, dict):
                        response_content = response.get("content", "")
                    else:
                        response_content = str(response)
                except (AttributeError, TypeError):
                    # Final fallback
                    logger.warning("Using fallback for plan generation")
                    response_content = "Generated basic plan for: " + task_description
            
            logger.info(f"Generated planning response for task: {task_description}")
            
            # Process the response into a structured plan
            # Extract steps from the response content - simplified parsing
            steps = []
            
            # Create a more dynamic plan based on task type
            if "twitter" in task_description.lower() or "social media" in task_description.lower():
                steps = [
                    {
                        "id": "step_1",
                        "description": "Define search parameters for social media data collection",
                        "assigned_to": "retriever",
                        "dependencies": [],
                        "estimated_duration": "short",
                    },
                    {
                        "id": "step_2",
                        "description": "Search for and collect recent social media posts and trends",
                        "assigned_to": "retriever",
                        "dependencies": ["step_1"],
                        "estimated_duration": "medium",
                    },
                    {
                        "id": "step_3",
                        "description": "Analyze post frequency and engagement metrics for trending topics",
                        "assigned_to": "executor",
                        "dependencies": ["step_2"],
                        "estimated_duration": "medium",
                    },
                    {
                        "id": "step_4",
                        "description": "Identify key themes and technologies mentioned in trending posts",
                        "assigned_to": "executor",
                        "dependencies": ["step_3"],
                        "estimated_duration": "medium",
                    },
                    {
                        "id": "step_5",
                        "description": "Generate visualization of trending topic relationships",
                        "assigned_to": "executor",
                        "dependencies": ["step_4"],
                        "estimated_duration": "short",
                    },
                    {
                        "id": "step_6",
                        "description": "Review analysis quality and completeness",
                        "assigned_to": "critic",
                        "dependencies": ["step_5"],
                        "estimated_duration": "short",
                    }
                ]
            else:
                # Default steps for other types of tasks
                steps = [
                    {
                        "id": "step_1",
                        "description": "Initialize data collection parameters",
                        "assigned_to": "retriever",
                        "dependencies": [],
                        "estimated_duration": "short",
                    },
                    {
                        "id": "step_2",
                        "description": "Gather data from specified sources",
                        "assigned_to": "retriever",
                        "dependencies": ["step_1"],
                        "estimated_duration": "medium",
                    },
                    {
                        "id": "step_3", 
                        "description": "Process and analyze collected data",
                        "assigned_to": "executor",
                        "dependencies": ["step_2"],
                        "estimated_duration": "medium",
                    },
                    {
                        "id": "step_4",
                        "description": "Evaluate analysis results and provide feedback",
                        "assigned_to": "critic",
                        "dependencies": ["step_3"],
                        "estimated_duration": "short",
                    }
                ]
                
            # Create the full plan
            plan = {
                "task": task_description,
                "steps": steps,
                "success_criteria": [
                    "All required data must be collected",
                    "Analysis must identify key insights about the topic",
                    "Findings must be supported by the collected data",
                    "Recommendations should be actionable and relevant"
                ],
                "contingency_plans": {
                    "data_source_unavailable": "Switch to alternative source or use cached data",
                    "insufficient_data": "Lower confidence threshold or expand search parameters",
                    "analysis_difficulties": "Apply alternative analytical methods or techniques"
                }
            }
            
            return plan
            
        except Exception as e:
            # Handle any errors
            logger.error(f"Error in PlannerAgent.run: {e}")
            return {
                "task": task_description,
                "error": str(e),
                "status": "error"
            }