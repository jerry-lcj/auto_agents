#!/usr/bin/env python3
"""
Executor Agent - Responsible for task execution based on plans and data.
"""
from typing import Any, Dict, List, Optional, Union
import os
import logging
import importlib.util
import sys
from pathlib import Path
import json
import re
from datetime import datetime

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


class ExecutorAgent:
    """
    ExecutorAgent carries out planned tasks using retrieved data.
    
    This agent is responsible for:
    1. Executing analysis and processing tasks
    2. Running code and computations
    3. Generating outputs and visualizations
    4. Handling operation sequences
    """

    def __init__(
        self,
        name: str = "executor",
        llm_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the ExecutorAgent.
        
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
                system_message="""You are an execution specialist with expertise in carrying out 
                complex tasks based on well-defined plans. You excel at implementing processes, 
                handling details, and delivering results according to specifications."""
            )
        except (TypeError, ValueError):
            # Fall back to older version approach with minimal compatible parameters
            self.agent = AssistantAgent(
                name=self.name,
                model_client=self.llm_client,
                system_message="""You are an execution specialist with expertise in carrying out 
                complex tasks based on well-defined plans."""
            )
        
        logger.info(f"ExecutorAgent '{name}' initialized")
        
        # Tool registry - can be extended with additional processing tools
        self.tools = {
            "data_processing": {
                "extract_keywords": self._extract_keywords,
                "summarize_text": self._summarize_text,
                "analyze_sentiment": self._analyze_sentiment,
            },
            "visualization": {},  # Will be implemented later
            "insight_generation": {},  # Will be implemented later
        }

    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract key terms from text using simple frequency analysis."""
        if not text:
            return []
            
        # Simple keyword extraction using word frequency
        words = re.findall(r'\b[a-zA-Z]{3,15}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {"the", "and", "is", "in", "to", "of", "for", "on", "that", "this", 
                     "with", "by", "at", "from", "are", "was", "were", "be", "been", 
                     "being", "have", "has", "had", "do", "does", "did", "but", "or",
                     "as", "what", "when", "where", "how", "all", "any", "both", "each",
                     "few", "more", "most", "other", "some", "such", "no", "nor", "not",
                     "only", "own", "same", "so", "than", "too", "very"}
        
        filtered_words = [w for w in words if w not in stop_words]
        
        # Count word frequency
        from collections import Counter
        word_counts = Counter(filtered_words)
        
        # Return the most common words
        return [word for word, count in word_counts.most_common(max_keywords)]

    def _summarize_text(self, text: str, max_sentences: int = 3) -> str:
        """Create a simple extractive summary of text."""
        if not text:
            return ""
            
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) <= max_sentences:
            return text
            
        # Very basic extractive summarization - choose sentences with keywords
        keywords = self._extract_keywords(text)
        sentence_scores = []
        
        for sentence in sentences:
            score = sum(1 for keyword in keywords if keyword in sentence.lower())
            sentence_scores.append((sentence, score))
            
        # Sort sentences by score and pick the top ones
        top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:max_sentences]
        
        # Restore original order
        original_order = []
        for i, sentence in enumerate(sentences):
            for top_sentence, score in top_sentences:
                if sentence == top_sentence:
                    original_order.append((i, sentence))
                    
        summary = " ".join(sentence for _, sentence in sorted(original_order))
        return summary

    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Perform basic sentiment analysis on text."""
        if not text:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
            
        # Simple lexicon-based sentiment analysis
        positive_words = {"good", "great", "excellent", "positive", "wonderful", "best", "love", 
                        "happy", "beneficial", "success", "successful", "advantage", "advantages", 
                        "beneficial", "perfect", "recommend", "impressive", "improvement"}
        
        negative_words = {"bad", "awful", "terrible", "poor", "negative", "worst", "hate",
                        "problem", "difficult", "trouble", "fail", "failed", "failure", 
                        "disadvantage", "disadvantages", "unfortunately", "weak", "weakness"}
        
        text_lower = text.lower()
        words = set(re.findall(r'\b[a-zA-Z]{3,15}\b', text_lower))
        
        positive_count = sum(1 for word in positive_words if word in words)
        negative_count = sum(1 for word in negative_words if word in words)
        total = positive_count + negative_count or 1  # Avoid division by zero
        
        positive_score = positive_count / total
        negative_score = negative_count / total
        
        if positive_count == 0 and negative_count == 0:
            neutral_score = 1.0
        else:
            neutral_score = 1.0 - (positive_score + negative_score)
            neutral_score = max(0, neutral_score)  # Ensure non-negative
            
        return {
            "positive": round(positive_score, 2),
            "negative": round(negative_score, 2),
            "neutral": round(neutral_score, 2)
        }

    def run(
        self, 
        plan: Dict[str, Any], 
        data: Dict[str, Any], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the tasks defined in the plan using the provided data.
        
        Args:
            plan: Execution plan from the planner
            data: Data from the retriever
            **kwargs: Additional parameters for task execution
            
        Returns:
            Dictionary containing execution results
        """
        try:
            start_time = datetime.now()
            
            # Extract task from plan - handle different plan formats
            task = plan
            if isinstance(plan, dict):
                task = plan.get('task', plan)
            
            # Convert task to string if it's an object
            if not isinstance(task, str):
                task = str(task)
                
            # Extract records from data
            records = []
            if isinstance(data, dict):
                records = data.get("records", [])
                
            # Combine all text content from records
            all_content = ""
            for record in records:
                if isinstance(record, dict) and "content" in record:
                    all_content += record["content"] + " "
            
            # Construct an execution prompt based on the plan and data
            num_records = len(records)
            data_summary = self._summarize_text(all_content[:10000])  # Limit to first 10,000 chars
            
            execution_prompt = f"""
            Execute the following task using the provided data:
            
            TASK: {task}
            
            DATA SUMMARY:
            - Number of records: {num_records}
            - Content summary: {data_summary}
            
            Perform comprehensive analysis and generate insightful results.
            Focus on extracting meaningful patterns and insights.
            Structure your response with clear sections:
            1. Key Findings
            2. Analysis
            3. Recommendations
            """
            
            # Get execution response from the LLM
            response_content = ""
            try:
                # Try newer API style
                response = self.llm_client.call(
                    messages=[{"role": "user", "content": execution_prompt}],
                    model=self.llm_config.get("model", "gpt-4-turbo")
                )
                response_content = response.get("content", "")
            except (AttributeError, TypeError):
                # Fall back to a basic approach for older versions
                logger.warning("Using fallback for execution response generation")
                
                # Process the data directly to generate insights
                keywords = self._extract_keywords(all_content)
                sentiment = self._analyze_sentiment(all_content)
                
                # Generate a simple report
                response_content = f"""
                # Analysis Results
                
                ## Key Findings
                - Analyzed {num_records} data records
                - Key topics: {', '.join(keywords[:5])}
                - Overall sentiment: {max(sentiment, key=sentiment.get)}
                
                ## Analysis
                {data_summary}
                
                ## Recommendations
                Based on the provided data, continue exploring these key topics and themes.
                """
            
            # Parse the response to extract key sections
            key_findings = []
            analysis = ""
            recommendations = []
            
            # Extract key findings using regex
            findings_match = re.search(r'(?:Key\s+Findings|Findings)[\s:]*\n(.*?)(?:\n\s*##|\n\s*$)', 
                                     response_content, re.DOTALL | re.IGNORECASE)
            if findings_match:
                findings_text = findings_match.group(1).strip()
                key_findings = [item.strip().strip('*-') for item in findings_text.split('\n') 
                              if item.strip() and item.strip() not in ['', '-', '*']]
            
            # Extract analysis section
            analysis_match = re.search(r'(?:Analysis)[\s:]*\n(.*?)(?:\n\s*##|\n\s*$)', 
                                     response_content, re.DOTALL | re.IGNORECASE)
            if analysis_match:
                analysis = analysis_match.group(1).strip()
            
            # Extract recommendations
            recommendations_match = re.search(r'(?:Recommendations)[\s:]*\n(.*?)(?:\n\s*##|\n\s*$)', 
                                           response_content, re.DOTALL | re.IGNORECASE)
            if recommendations_match:
                recommendations_text = recommendations_match.group(1).strip()
                recommendations = [item.strip().strip('*-') for item in recommendations_text.split('\n') 
                                 if item.strip() and item.strip() not in ['', '-', '*']]
            
            # If we didn't extract anything, use simple default values
            if not key_findings:
                key_findings = ["No specific findings could be extracted"]
            
            if not analysis:
                analysis = "Analysis could not be extracted from the response."
            
            if not recommendations:
                recommendations = ["Further analysis recommended"]
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create execution results
            execution_results = {
                "status": "success",
                "metrics": {
                    "records_processed": num_records,
                    "execution_time": execution_time,
                    "success_rate": 1.0,
                },
                "results": {
                    "key_findings": key_findings,
                    "analysis": analysis,
                    "recommendations": recommendations,
                    "keywords": self._extract_keywords(all_content, max_keywords=20),
                    "sentiment": self._analyze_sentiment(all_content),
                },
                "log": [
                    {"timestamp": start_time.isoformat(), "level": "INFO", "message": "Execution started"},
                    {"timestamp": datetime.now().isoformat(), "level": "INFO", "message": "Execution completed"},
                ]
            }
            
            logger.info(f"Task execution completed successfully with {len(key_findings)} findings")
            return execution_results
            
        except Exception as e:
            # Handle any errors
            logger.error(f"Error in ExecutorAgent.run: {e}")
            return {
                "status": "error",
                "error": str(e),
                "results": {}
            }