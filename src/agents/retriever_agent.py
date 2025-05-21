#!/usr/bin/env python3
"""
Retriever Agent - Responsible for gathering and processing data from various sources.
"""
from typing import Any, Dict, List, Optional, Union
import os
import logging
import importlib.util
import sys
from pathlib import Path
import time

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

# Import tools
# First determine if the module is being run directly or imported
if __name__ == "__main__":
    # Adjust path if run directly
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.tools.web_scraper import WebScraperTool
    from src.tools.db_tool import DatabaseTool
    from src.tools.mcp_client import MCPClientTool
else:
    # Try relative imports
    try:
        from ..tools.web_scraper import WebScraperTool
        from ..tools.db_tool import DatabaseTool
        from ..tools.mcp_client import MCPClientTool
    except ImportError:
        # Fall back to absolute imports
        current_dir = Path(__file__).resolve().parent
        tools_dir = current_dir.parent / "tools"
        
        # Add to path if needed
        if str(tools_dir.parent) not in sys.path:
            sys.path.append(str(tools_dir.parent))
            
        from tools.web_scraper import WebScraperTool
        from tools.db_tool import DatabaseTool
        from tools.mcp_client import MCPClientTool

# Configure logging
logger = logging.getLogger(__name__)


class RetrieverAgent:
    """
    RetrieverAgent gathers, processes, and manages data from various sources.
    
    This agent is responsible for:
    1. Collecting data from databases, APIs, web pages, etc.
    2. Filtering and preprocessing data
    3. Ensuring data quality and relevance
    4. Handling data storage and retrieval
    """

    def __init__(
        self,
        name: str = "retriever",
        llm_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the RetrieverAgent.
        
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
                system_message="""You are a data retrieval specialist with expertise in finding and 
                collecting information from various sources. You excel at formulating effective 
                search queries, identifying relevant data sources, and extracting high-quality 
                information efficiently."""
            )
        except (TypeError, ValueError):
            # Fall back to older version approach with minimal compatible parameters
            self.agent = AssistantAgent(
                name=self.name,
                model_client=self.llm_client,
                system_message="""You are a data retrieval specialist with expertise in finding and 
                collecting information from various sources."""
            )
        
        logger.info(f"RetrieverAgent '{name}' initialized")
        
        # Initialize data sources with actual tool instances
        self.data_sources = {
            "database": None,  # Will be initialized when needed
            "web_scraper": None,  # Initialize when needed
            "api_clients": {},  # Will store API client instances
        }
        
        # Try to initialize web scraper
        try:
            self.data_sources["web_scraper"] = WebScraperTool()
            logger.info("Web scraper initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize web scraper: {e}")
        
        # Try to initialize database if connection string exists in environment
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            try:
                self.data_sources["database"] = DatabaseTool(connection_string=db_url)
                logger.info("Database connection initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize database connection: {e}")

    def run(
        self, 
        plan: Dict[str, Any], 
        query: Optional[str] = None, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        根据执行计划收集数据。
        
        Args:
            plan: 来自规划者的执行计划
            query: 可选搜索查询，用于优化数据收集
            **kwargs: 用于数据检索的额外参数
            
        Returns:
            包含收集数据的字典
        """
        try:
            import time
            start_time = time.time()
            # 从计划中提取任务 - 处理不同的计划格式
            task = plan
            if isinstance(plan, dict):
                task = plan.get('task', plan)
            
            # 如果任务是对象，转换为字符串
            if not isinstance(task, str):
                task = str(task)
                
            # 检测任务类型，为不同类型的任务添加专门的处理
            task_lower = task.lower()
            is_twitter_task = "twitter" in task_lower
            is_social_media_task = is_twitter_task or any(kw in task_lower for kw in ["social media", "facebook", "instagram", "linkedin", "reddit", "tiktok"])
            
            # 基于计划构建检索提示，添加特定于任务类型的指导
            retrieval_prompt = f"""
            基于以下计划和查询检索信息:
            
            计划: {task}
            查询: {query if query else '未提供特定查询'}
            
            请关注收集相关、高质量的数据。
            考虑多种数据类型和格式。
            """
            
            # 为社交媒体任务添加特定提示
            if is_social_media_task:
                if is_twitter_task:
                    retrieval_prompt += f"""
                    这是一个关于Twitter的任务。请特别关注:
                    1. Twitter上的趋势话题和标签
                    2. 与AI相关的Twitter讨论
                    3. 有影响力的AI相关Twitter账号
                    4. Twitter上AI话题的参与度和情绪分析
                    """
                else:
                    retrieval_prompt += f"""
                    这是一个关于社交媒体的任务。请特别关注:
                    1. 社交媒体平台上的趋势话题
                    2. 与内容相关的用户参与度指标
                    3. 跨平台的话题比较
                    4. 情绪分析和用户反应
                    """
            
            # 获取检索指导 - 处理不同的 OpenAIWrapper 版本
            retrieval_strategy = ""
            try:
                # 尝试较新的 API 风格
                response = self.llm_client.call(
                    messages=[{"role": "user", "content": retrieval_prompt}],
                    model=self.llm_config.get("model", "gpt-4-turbo")
                )
                retrieval_strategy = response.get("content", "")
            except (AttributeError, TypeError):
                # 回退到较旧版本的基本方法
                logger.warning("为检索策略生成使用回退方法")
                retrieval_strategy = f"正在为以下任务检索信息: {task}"
                
            logger.info(f"已为任务生成检索策略: {task}")
            
            # 初始化数据收集结果
            retrieved_data = {
                "sources": [],
                "records": [],
                "metadata": {
                    "total_records": 0,
                    "query_time": 0,
                    "filters_applied": [],
                    "task": task,
                    "task_type": "social_media" if is_social_media_task else "general"
                }
            }
            
            # 确定需要使用的数据源
            web_scraper = self.data_sources.get("web_scraper")
            db_tool = self.data_sources.get("database")
            
            # 检查是否需要从 Web 获取数据
            if web_scraper:
                # 构建优化的搜索查询，为不同类型的任务添加关键词
                search_query = task
                
                # 为不同任务类型增强查询
                if is_twitter_task:
                    # 为Twitter任务强化查询
                    search_terms = ["twitter trending AI topics", "AI hashtags twitter", "popular AI accounts twitter"]
                    search_query = f"{search_query} {search_terms[0]}"
                elif is_social_media_task:
                    # 为一般社交媒体任务增强查询
                    search_terms = ["social media AI trends", "AI social media analytics", "AI discussions online"]
                    search_query = f"{search_query} {search_terms[0]}"
                    
                if query:
                    search_query = f"{search_query} {query}"
                    
                # 使用 Web 搜索收集信息
                logger.info(f"执行网络搜索: {search_query}")
                
                # 使用 search_and_scrape 函数
                search_results = web_scraper.search_and_scrape(search_query, max_pages=3)
                
                # 处理搜索结果
                if search_results and search_results.get("scraped_data"):
                    scraped_data = search_results.get("scraped_data", [])
                    logger.info(f"从 {len(scraped_data)} 个网站检索数据")
                    
                    # 处理每个抓取的页面
                    for page_data in scraped_data:
                        # 添加来源
                        retrieved_data["sources"].append({
                            "name": page_data.get("title", page_data.get("url", "未知来源")),
                            "type": "web",
                            "url": page_data.get("url", ""),
                            "status": "success"
                        })
                        
                        # 添加内容项
                        content_items = page_data.get("content", [])
                        for item in content_items:
                            if item.strip():  # 跳过空内容
                                retrieved_data["records"].append({
                                    "id": len(retrieved_data["records"]) + 1,
                                    "content": item,
                                    "source": page_data.get("url", "web search"),
                                    "type": "text"
                                })
                    
                    logger.info(f"从网络搜索添加了 {len(retrieved_data['records'])} 个内容项")
                else:
                    logger.warning("网络搜索未返回有用结果")
                    
                    # 如果第一次搜索失败，尝试使用不同的查询
                    if is_twitter_task and len(retrieved_data["records"]) == 0:
                        # 尝试使用替代查询
                        alternative_queries = [
                            "twitter AI analytics recent trends",
                            "AI machine learning trending topics twitter",
                            "artificial intelligence twitter discussions"
                        ]
                        
                        for alt_query in alternative_queries:
                            logger.info(f"尝试替代查询: {alt_query}")
                            alt_results = web_scraper.search_and_scrape(alt_query, max_pages=2)
                            
                            if alt_results and alt_results.get("scraped_data"):
                                scraped_data = alt_results.get("scraped_data", [])
                                logger.info(f"从替代查询中检索到 {len(scraped_data)} 个数据源")
                                
                                # 处理每个抓取的页面
                                for page_data in scraped_data:
                                    # 添加来源
                                    retrieved_data["sources"].append({
                                        "name": page_data.get("title", page_data.get("url", "替代查询来源")),
                                        "type": "web",
                                        "url": page_data.get("url", ""),
                                        "status": "success"
                                    })
                                    
                                    # 添加内容项
                                    content_items = page_data.get("content", [])
                                    for item in content_items:
                                        if item.strip():  # 跳过空内容
                                            retrieved_data["records"].append({
                                                "id": len(retrieved_data["records"]) + 1,
                                                "content": item,
                                                "source": page_data.get("url", f"替代查询: {alt_query}"),
                                                "type": "text"
                                            })
                                
                                # 如果找到足够的内容项，停止尝试其他替代查询
                                if len(retrieved_data["records"]) > 10:
                                    logger.info("从替代查询中找到足够的内容，停止进一步搜索")
                                    break
            
            # 检查是否需要从数据库获取数据
            # 通过分析任务内容来确定是否需要数据库数据
            db_keywords = ["database", "storage", "record", "table", "query", "sql", 
                         "dataset", "数据库", "存储", "记录", "表", "查询"]
            need_db_data = any(keyword in task.lower() for keyword in db_keywords) and db_tool is not None
            
            if need_db_data and db_tool:
                try:
                    # 获取表列表
                    tables = db_tool.get_table_names()
                    
                    # 示例：查询第一个表
                    if tables:
                        first_table = tables[0]
                        query_result = db_tool.execute_query(f"SELECT * FROM {first_table} LIMIT 100")
                        
                        # 添加来源
                        retrieved_data["sources"].append({
                            "name": "database",
                            "type": "database",
                            "table": first_table,
                            "status": "success"
                        })
                        
                        # 添加查询结果作为记录
                        for row in query_result:
                            retrieved_data["records"].append({
                                "id": len(retrieved_data["records"]) + 1,
                                "content": str(row),
                                "source": "database",
                                "type": "database_row"
                            })
                            
                        logger.info(f"从数据库表 {first_table} 检索到 {len(query_result)} 行")
                except Exception as e:
                    logger.error(f"从数据库检索数据时出错: {e}")
                    retrieved_data["sources"].append({
                        "name": "database",
                        "type": "database",
                        "status": "error",
                        "error": str(e)
                    })
            
            # 更新元数据
            end_time = time.time()
            retrieved_data["metadata"]["total_records"] = len(retrieved_data["records"])
            retrieved_data["metadata"]["query_time"] = end_time - start_time
            
            # 如果没有收集到任何数据，给出提示
            if not retrieved_data["records"]:
                logger.warning("未从来源收集到数据，使用提示")
                
                retrieved_data["records"] = [
                    {
                        "id": 1, 
                        "content": "未能为此查询找到特定数据。建议尝试使用不同的搜索词或提供更具体的查询。",
                        "source": "system",
                        "type": "message"
                    }
                ]
                retrieved_data["sources"].append({
                    "name": "system",
                    "type": "message",
                    "status": "warning",
                    "message": "数据来源不匹配查询要求"
                })
            
            # 添加用于分析的任务描述
            if not any(record.get("content") == task for record in retrieved_data["records"]):
                # 确保原始任务也被包含在记录中，这有助于后续分析
                retrieved_data["records"].append({
                    "id": len(retrieved_data["records"]) + 1,
                    "content": task,
                    "source": "original_task",
                    "type": "task_description"
                })
            
            return retrieved_data
            
        except Exception as e:
            # 处理任何错误
            logger.error(f"RetrieverAgent.run 中的错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "status": "error",
                "records": []
            }