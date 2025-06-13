# MCP 服务器配置和使用指南

这个指南介绍如何使用增强型 MCP (Model Context Protocol) 系统，该系统支持从配置文件加载和管理外部 MCP 服务器。

## 配置文件格式

MCP 配置文件使用 JSON 格式，结构如下：

```json
{
  "mcpServers": {
    "context7-mcp": {
      "command": "npx",
      "args": [
        "-y",
        "@smithery/cli@latest",
        "run",
        "@upstash/context7-mcp",
        "--key",
        "13c09f31-aea6-449e-801b-26fa75825341"
      ],
      "host": "localhost",
      "port": 3000
    },
    "local-mcp": {
      "host": "0.0.0.0",
      "port": 8000,
      "internal": true
    }
  },
  "defaultServer": "local-mcp"
}
```

配置文件说明:
- `mcpServers`: 包含所有可用的 MCP 服务器配置
  - 每个服务器配置项包含:
    - `host`: 服务器主机地址
    - `port`: 服务器端口
    - `command`: (可选) 启动外部服务器的命令
    - `args`: (可选) 启动命令的参数数组
    - `internal`: (可选) 如果为 true，表示这是一个内部服务器
- `defaultServer`: 默认使用的服务器名称

## 命令行工具

项目包含多个命令行工具用于管理和测试 MCP 服务器:

### MCP 服务器管理 CLI

```bash
python agent_test/mcp_server_cli.py [OPTIONS] COMMAND

Commands:
  list         列出所有可用的服务器
  start        启动服务器 
  stop         停止服务器
  add          添加新服务器到配置中
  remove       从配置中移除服务器
  set-default  设置默认服务器

选项:
  --config CONFIG  MCP 配置文件路径
```

示例:

```bash
# 列出所有服务器
python agent_test/mcp_server_cli.py list

# 启动一个服务器
python agent_test/mcp_server_cli.py start context7-mcp

# 添加新服务器
python agent_test/mcp_server_cli.py add new-mcp-server --host localhost --port 3001 --command npx --args "--yes other-mcp-server"
```

### MCP 客户端测试

```bash
python agent_test/mcp_client_test.py --query "你的查询" [--server SERVER_NAME] [--config CONFIG_PATH]
```

### MCP 配置演示

```bash
python agent_test/mcp_config_demo.py [--server SERVER_NAME] [--config CONFIG_PATH] [--query "你的查询"]
```

## 在代码中使用

### 方法 1: 使用 MultiAgentManager

```python
from src.main import MultiAgentManager

# 使用配置的服务器初始化
manager = MultiAgentManager(
    enable_mcp=True,
    mcp_server_name="context7-mcp",  # 配置中的服务器名称
    mcp_config_path="/path/to/config.json"  # 可选，默认使用 src/config/mcp_config.json
)

# 运行查询
result = manager.run_mcp_task_sync(
    "你的查询",
    context={"platform": "twitter", "time_range": "last_week"}
)
```

### 方法 2: 使用 EnhancedMCPClient

```python
import asyncio
from src.tools.enhanced_mcp_client import EnhancedMCPClient

async def run_query():
    # 初始化客户端
    client = EnhancedMCPClient(
        server_name="context7-mcp",  # 配置中的服务器名称
        config_path="/path/to/config.json"  # 可选
    )
    
    # 发送查询
    response = await client.query(
        query="你的查询",
        context={"platform": "twitter", "time_range": "last_week"}
    )
    
    # 处理响应
    print(response)
    
    # 关闭客户端
    await client.close()

# 运行异步函数
asyncio.run(run_query())

# 或者使用同步版本
client = EnhancedMCPClient(server_name="context7-mcp")
result = client.query_sync("你的查询")
client.close_sync()
```

## 关键组件

- `mcp_server_manager.py`: 管理 MCP 服务器配置和服务器进程
- `enhanced_mcp_client.py`: 增强版 MCP 客户端，支持配置和外部服务器
- `mcp_agent.py`: 支持外部服务器的 MCP 代理
- `mcp_config.json`: MCP 服务器配置文件
