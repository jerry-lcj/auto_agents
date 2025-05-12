#!/usr/bin/env python3
"""
Planner Agent - Responsible for task planning and coordination.
"""
from typing import Any, Dict, List, Optional, Union

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import StructuredMessage


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
        
        # Setup the underlying AutoGen agent with a compatible initialization approach
        # For AutoGen 0.5.6, we need to use config_list instead of llm_config
        self.llm_client = OpenAIChatCompletionClient(**self.llm_config)

        self.agent = AssistantAgent(
            name=self.name,
            model_client=self.llm_client,
            system_message="""You are a strategic planning specialist with expertise in breaking down 
            complex tasks into clear, actionable steps. You excel at analyzing requirements, 
            identifying dependencies, and organizing work effectively.""",
            
        )

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
            # We'll use a direct approach that should work across versions
            response = None
            try:
                # Try using the generate method if available in this version
                response = self.agent.generate(planning_prompt)
            except (AttributeError, TypeError):
                # Fallback to a more basic approach
                response = {"content": "Sample plan - using fallback planning mechanism"}
            
            # Process the response into a structured plan
            # In a real implementation, this would parse the agent's response
            
            # Sample plan structure (to be generated dynamically)
            plan = {
                "task": task_description,
                "steps": [
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
                    # Additional steps would be generated here
                ],
                "success_criteria": [
                    "All required data must be collected",
                    "Analysis must identify key insights about the topic",
                ],
                "contingency_plans": {
                    "data_source_unavailable": "Switch to alternative source",
                    "insufficient_data": "Lower confidence threshold or expand search parameters",
                }
            }
            
            return plan
            
        except Exception as e:
            # Handle any errors
            return {
                "task": task_description,
                "error": str(e),
                "status": "error"
            }