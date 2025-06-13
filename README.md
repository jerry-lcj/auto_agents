# Auto Agents

A multi-agent data-analysis & automation platform built on AutoGen.

## Project Overview

This platform combines multiple specialized AI agents to perform complex data analysis and automation tasks. The system includes:

- **Planner Agent**: Coordinates tasks and creates execution plans
- **Retriever Agent**: Gathers and processes data from various sources
- **Executor Agent**: Runs actions and processes based on plans
- **Critic Agent**: Evaluates results and suggests improvements
- **UI Tool Agent**: Handles UI automation tasks

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd auto_agents

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development
pip install -e ".[dev]"
```

## Usage Example

```python
from src.main import MultiAgentManager

# Initialize the multi-agent system
manager = MultiAgentManager()

# Run a simple data analysis task
result = manager.run_task(
    "Analyze the stock price trends of AAPL from the last quarter and generate a summary report"
)

# Run a web automation task
result = manager.run_task(
    "Log into example.com and extract the latest user statistics"
)
```

## Project Structure

```
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── src/
    ├── agents/
    │   ├── planner_agent.py - Task planning and coordination
    │   ├── retriever_agent.py - Data gathering and processing
    │   ├── executor_agent.py - Task execution
    │   ├── critic_agent.py - Result evaluation
    │   └── ui_tool_agent.py - UI automation
    ├── tools/
    │   ├── db_tool.py - Database operations
    │   ├── web_scraper.py - Web scraping utilities
    │   └── ui_automation.py - Browser and UI automation
    └── main.py - Main entry point and manager
```

## Key Features

- Multi-agent orchestration with AutoGen
- Specialized agents for different aspects of data analysis
- Flexible tool integration (database, web scraping, UI automation)
- Task planning and execution framework

## License

MIT




nl_ui_agent.py input the NL command and then use ui_use_from_command.py to execute the command.
