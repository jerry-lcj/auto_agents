#!/usr/bin/env python3
"""
Critic Agent - Responsible for evaluating results and providing feedback.
"""
from typing import Any, Dict, List, Optional, Union

import autogen
from autogen import AssistantAgent


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
        
        # Setup the underlying AutoGen agent
        self.agent = AssistantAgent(
            name=self.name,
            system_message="""You are an evaluation specialist with strong analytical skills. 
            Your role is to critically assess results, identify potential issues or inconsistencies, 
            and provide constructive feedback for improvement. You excel at detecting flaws in 
            analyses and suggesting better approaches.""",
            llm_config=self.llm_config,
        )

    def run(
        self,
        plan: Dict[str, Any],
        data: Dict[str, Any],
        result: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate the results of task execution against the plan and data.
        
        Args:
            plan: Original execution plan
            data: Data used for execution
            result: Results from task execution
            **kwargs: Additional parameters for evaluation
            
        Returns:
            Dictionary containing evaluation and feedback
        """
        # TODO: Implement evaluation logic
        
        # Sample evaluation results (to be dynamically generated)
        evaluation = {
            "overall_assessment": "satisfactory",  # options: excellent, satisfactory, needs_improvement, failed
            "success_criteria_met": [
                {"criterion": "All required data collected", "met": True, "comments": "Complete dataset obtained"},
                {"criterion": "Analysis identifies key insights", "met": True, "comments": "Good insights, but could be more detailed"}
            ],
            "issues_identified": [
                {"severity": "low", "description": "Minor inconsistency in data visualization", "recommendation": "Normalize data before plotting"}
            ],
            "improvement_suggestions": [
                "Consider additional data sources for more comprehensive analysis",
                "Apply statistical significance tests to validate findings"
            ],
            "confidence_score": 0.85
        }
        
        return evaluation