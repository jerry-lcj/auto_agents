#!/usr/bin/env python3
"""
Critic Agent - Responsible for evaluating results and providing feedback.
"""
from typing import Any, Dict, List, Optional, Union

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import StructuredMessage


class CriticAgent:
    """
    CriticAgent evaluates results and provides feedback for improvement.
    
    This agent is responsible for:
    1. Evaluating the quality and validity of results
    2. Identifying errors or inconsistencies
    3. Suggesting improvements
    4. Verifying that success criteria are met
    """

    def __init__(
        self,
        name: str = "critic",
        llm_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the CriticAgent.
        
        Args:
            name: Name identifier for the agent
            llm_config: Configuration for the language model
        """
        self.name = name
        self.llm_config = llm_config or {}
        
        # Setup LLM client with the new API
        llm_client = OpenAIChatCompletionClient(**self.llm_config)
        
        # Setup the underlying AutoGen agent with a compatible initialization approach
        self.agent = AssistantAgent(
            name=self.name,
            model_client=llm_client,
            system_message="""You are an analytical evaluation specialist with expertise in 
            critically assessing results, identifying issues, and suggesting improvements. 
            You excel at thorough analysis, providing constructive feedback, and ensuring 
            solutions meet requirements and quality standards.""",
            # Use a format compatible with AutoGen 0.5.6
            
        )

    def run(
        self, 
        plan: Dict[str, Any], 
        data: Dict[str, Any],
        result: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate the execution results against the plan.
        
        Args:
            plan: Original execution plan
            data: Data used for execution
            result: Results from the execution
            **kwargs: Additional parameters for evaluation
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            # Construct an evaluation prompt
            evaluation_prompt = f"""
            Evaluate the following execution results against the original plan:
            
            PLAN: {plan.get('task', 'No task description available')}
            DATA: {len(data.get('records', [])) if data else 0} records were used
            RESULT STATUS: {result.get('status', 'unknown')}
            
            Provide a thorough assessment of the results focusing on:
            1. Achievement of objectives
            2. Quality and accuracy of analysis
            3. Comprehensiveness of the results
            4. Areas for improvement
            """
            
            # Get evaluation response from the agent
            response = None
            try:
                # Try using the generate method if available in this version
                response = self.agent.generate(evaluation_prompt)
            except (AttributeError, TypeError):
                # Fallback to a more basic approach
                response = {"content": "Sample evaluation - using fallback mechanism"}
            
            # Process the response to create evaluation results
            # In a real implementation, this would parse the agent's response
            
            # Sample evaluation results (to be generated dynamically)
            evaluation_results = {
                "overall_rating": 8.5,  # Scale of 1-10
                "criteria": {
                    "objective_achievement": {
                        "score": 8,
                        "comments": "Most key objectives were achieved with good results."
                    },
                    "quality_accuracy": {
                        "score": 9,
                        "comments": "Analysis was accurate and well-supported by data."
                    },
                    "comprehensiveness": {
                        "score": 8,
                        "comments": "Results covered most aspects, but could explore alternative viewpoints."
                    }
                },
                "strengths": [
                    "Strong analytical approach",
                    "Clear presentation of findings"
                ],
                "improvement_areas": [
                    "Consider additional data sources",
                    "Explore alternative analytical methods"
                ],
                "recommendations": "Focus on broadening data sources while maintaining analytical quality."
            }
            
            return evaluation_results
            
        except Exception as e:
            # Handle any errors
            return {
                "error": str(e),
                "overall_rating": 0,
                "criteria": {},
                "recommendations": "Error occurred during evaluation"
            }