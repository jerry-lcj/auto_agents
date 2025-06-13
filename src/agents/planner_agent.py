#!/usr/bin/env python3
"""
Planner Agent - Responsible for task planning and coordination.
"""
from typing import Any, Dict, List, Optional, Union
import os
import re
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
            
            ## STEPS
            Please provide a numbered list of steps, each with a clear description. Format as:
            1. [Step 1 description]
            2. [Step 2 description]
            ...
            
            ## SUCCESS CRITERIA
            Please provide a bulleted list of criteria that would indicate the task has been successfully completed. Format as:
            - [Criterion 1]
            - [Criterion 2]
            ...
            
            ## CONTINGENCY PLANS
            For each potential issue, provide a solution. Format as:
            - If [problem 1]: [solution 1]
            - If [problem 2]: [solution 2]
            ...
            
            Format your response clearly with these exact section headings to facilitate parsing.
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
            # Extract steps from the response content
            steps = []
            
            try:
                # 基于响应内容提取步骤
                import re
                
                # 尝试查找步骤编号/列表的模式
                step_pattern = re.compile(r'(?:Step|STEP|^\d+[\)\.:]|^-|\b\d+[\)\.:])\s*(.*?)(?=(?:Step|STEP|\n\d+[\)\.:]|\n-|\n\b\d+[\)\.:])|\Z)', re.MULTILINE | re.DOTALL)
                # 替代模式，如果第一种匹配失败
                alt_pattern = re.compile(r'(\d+[\.\):]\s*.+?)(?=\n\d+[\.\):]|\Z)', re.MULTILINE | re.DOTALL)
                
                # 尝试匹配步骤
                matches = step_pattern.findall(response_content)
                if not matches:
                    matches = alt_pattern.findall(response_content)
                
                # 处理匹配到的步骤
                for i, step_text in enumerate(matches):
                    step_text = step_text.strip()
                    if not step_text:
                        continue
                        
                    # 从步骤描述中推断角色分配
                    assigned_to = "executor"  # 默认执行者
                    if any(word in step_text.lower() for word in ["collect", "gather", "search", "retrieve", "find"]):
                        assigned_to = "retriever"
                    elif any(word in step_text.lower() for word in ["evaluate", "assess", "review", "critique"]):
                        assigned_to = "critic"
                    
                    # 推断依赖关系
                    dependencies = []
                    if i > 0:  # 如果不是第一步，通常依赖于前一步
                        dependencies = [f"step_{i}"]
                    
                    # 推断预计持续时间
                    estimated_duration = "medium"
                    if len(step_text) < 50:  # 简短描述可能是简单任务
                        estimated_duration = "short"
                    
                    # 创建步骤字典
                    step = {
                        "id": f"step_{i+1}",
                        "description": step_text,
                        "assigned_to": assigned_to,
                        "dependencies": dependencies,
                        "estimated_duration": estimated_duration,
                    }
                    steps.append(step)
                
                logger.info(f"Extracted {len(steps)} steps from LLM response")
                
                # 如果没有提取到任何步骤，提供一些默认步骤
                if not steps:
                    raise ValueError("Could not extract steps from response")
                    
            except Exception as extraction_error:
                logger.warning(f"Error extracting steps from response: {extraction_error}")
                logger.warning("Using default step structure")
                
                # 基于任务类型提供默认步骤
                if "twitter" in task_description.lower() or "social media" in task_description.lower():
                    # 社交媒体相关任务的默认步骤
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
                            "description": "Analyze collected data for trending topics",
                            "assigned_to": "executor",
                            "dependencies": ["step_2"],
                            "estimated_duration": "medium",
                        }
                    ]
                else:
                    # 通用任务的默认步骤
                    steps = [
                        {
                            "id": "step_1",
                            "description": "Collect relevant data",
                            "assigned_to": "retriever",
                            "dependencies": [],
                            "estimated_duration": "short",
                        },
                        {
                            "id": "step_2", 
                            "description": "Process and analyze collected data",
                            "assigned_to": "executor",
                            "dependencies": ["step_1"],
                            "estimated_duration": "medium",
                        },
                        {
                            "id": "step_3",
                            "description": "Evaluate results",
                            "assigned_to": "critic",
                            "dependencies": ["step_2"],
                            "estimated_duration": "short",
                        }
                    ]
                
            # 从响应中提取成功标准
            success_criteria = []
            try:
                # 查找成功标准部分
                criteria_match = re.search(r'(?:Success Criteria|SUCCESS CRITERIA|Success Metrics|COMPLETION CRITERIA).*?(?:\n\n|\Z)', 
                                          response_content, re.IGNORECASE | re.DOTALL)
                if criteria_match:
                    criteria_text = criteria_match.group(0)
                    # 尝试查找列出的标准
                    criteria_items = re.findall(r'(?:-|\d+[\.\):])\s*(.*?)(?=\n\s*(?:-|\d+[\.\):]|\n|\Z))', criteria_text, re.DOTALL)
                    if criteria_items:
                        success_criteria = [item.strip() for item in criteria_items if item.strip()]
            except Exception as e:
                logger.warning(f"Error extracting success criteria: {e}")
                
            # 如果没有提取到成功标准，使用默认标准
            if not success_criteria:
                success_criteria = [
                    "All required data must be collected",
                    "Analysis must identify key insights about the topic",
                    "Findings must be supported by the collected data",
                    "Recommendations should be actionable and relevant"
                ]
                
            # 从响应中提取应急计划
            contingency_plans = {}
            try:
                # 查找应急计划部分
                contingency_match = re.search(r'(?:Contingency Plans|CONTINGENCY PLANS|Backup Plans|Error Handling).*?(?:\n\n|\Z)', 
                                             response_content, re.IGNORECASE | re.DOTALL)
                if contingency_match:
                    contingency_text = contingency_match.group(0)
                    # 尝试提取问题和解决方案
                    contingency_items = re.findall(r'(?:-|\d+[\.\):]|If|When)\s*(.*?):(.*?)(?=\n\s*(?:-|\d+[\.\):]|If|When|\n|\Z))', 
                                                  contingency_text, re.IGNORECASE | re.DOTALL)
                    if contingency_items:
                        for problem, solution in contingency_items:
                            problem = problem.strip().lower().replace(" ", "_")
                            solution = solution.strip()
                            if problem and solution:
                                contingency_plans[problem] = solution
            except Exception as e:
                logger.warning(f"Error extracting contingency plans: {e}")
                
            # 如果没有提取到应急计划，使用默认应急计划
            if not contingency_plans:
                contingency_plans = {
                    "data_source_unavailable": "Switch to alternative source or use cached data",
                    "insufficient_data": "Lower confidence threshold or expand search parameters",
                    "analysis_difficulties": "Apply alternative analytical methods or techniques"
                }
            
            # 创建完整计划
            plan = {
                "task": task_description,
                "steps": steps,
                "success_criteria": success_criteria,
                "contingency_plans": contingency_plans,
                "raw_plan_text": response_content  # 保存原始响应以供参考
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