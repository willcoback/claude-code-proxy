# Claude Code Proxy

一个 Python 代理服务器，用于将 Claude Code CLI 的请求转换为其他大语言模型（如 Gemini、Grok）的请求格式，实现使用其他 LLM 作为 Claude Code 后端。

## 功能特性

- 支持 Claude API 格式与其他 LLM API 格式的双向转换
- 支持流式（Streaming）和非流式响应
- 支持工具调用（Tool/Function Calling）
- 支持多轮对话
- 支持图片处理
- 每日轮转日志，记录请求时间、模型名称、Token 消耗
- 策略模式架构，易于扩展新的模型提供商

## 支持的模型提供商

| 提供商    | 状态    | 说明                      |
|--------|-------|-------------------------|
| Gemini | ✅ 已完成 | 完整支持 Google Gemini 系列模型 |
| Grok   | ⏳ 待实现 | Xai Grok 模型（预留接口）       |

## 项目结构

```
claude-code-proxy/
├── main.py                 # FastAPI 应用入口
├── requirements.txt        # Python 依赖
├── run.sh                  # 启动脚本
├── config/
│   └── config.yaml         # 配置文件
├── logs/                   # 日志目录（每日一个文件）
├── proxy/
│   ├── base/
│   │   └── strategy.py     # 策略模式基类
│   ├── gemini/
│   │   └── converter.py    # Gemini 策略实现
│   ├── grok/               # Grok 策略（待实现）
│   └── utils/
│       ├── config.py       # 配置管理器
│       └── logger.py       # 日志工具
└── tests/                  # 测试文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API 密钥

创建 `.env` 文件并添加 API 密钥：

```bash
# Gemini API 密钥
GEMINI_API_KEY=your_gemini_api_key_here

# Grok API 密钥（可选）
GROK_API_KEY=your_grok_api_key_here
```

### 3. 修改配置文件

编辑 `config/config.yaml`：

```yaml
server:
  host: "0.0.0.0"
  port: 8080

provider:
  name: "gemini"            # 当前使用的提供商：gemini 或 grok

gemini:
  api_key: "${GEMINI_API_KEY}"
  model: "gemini-2.0-flash" # 使用的模型
  base_url: "https://generativelanguage.googleapis.com/v1beta"
  timeout: 300

grok:
  api_key: "${GROK_API_KEY}"
  model: "grok-beta"
  base_url: "https://api.x.ai/v1"
  timeout: 300

logging:
  level: "INFO"
  dir: "./logs"
```

### 4. 启动代理服务

```bash
# 方式一：使用启动脚本（推荐）
./run.sh

# 方式二：直接运行
python main.py
```

服务默认运行在 `http://0.0.0.0:8080`

### 5. 配置 Claude Code 使用代理

设置 Claude Code 的 API 端点指向本代理：

```bash
# 方式一：设置环境变量（推荐）
export ANTHROPIC_BASE_URL=http://localhost:8080

# 方式二：每次运行时指定
ANTHROPIC_BASE_URL=http://localhost:8080 claude

# 方式三：添加到 shell 配置文件使其永久生效
echo 'export ANTHROPIC_BASE_URL=http://localhost:8080' >> ~/.zshrc
source ~/.zshrc
```

## API 端点

| 端点                         | 方法   | 说明                    |
|----------------------------|------|-----------------------|
| `/health`                  | GET  | 健康检查，返回服务状态和当前提供商     |
| `/v1/messages`             | POST | 主要代理端点，接收 Claude 格式请求 |
| `/v1/models`               | GET  | 返回可用模型列表              |
| `/api/event_logging/batch` | POST | 处理遥测事件（空实现）           |

## 使用示例

### 直接调用代理 API

```bash
curl -X POST http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello, world!"}
    ]
  }'
```

### 流式请求

```bash
curl -X POST http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "stream": true,
    "messages": [
      {"role": "user", "content": "写一首短诗"}
    ]
  }'
```

## 日志

日志文件存储在 `./logs/` 目录下，每天生成一个新文件，命名格式为 `proxy_YYYY-MM-DD.log`。

日志内容包括：

- 请求时间
- 代理目标模型名称
- Token 消耗（输入/输出/总计）
- 请求状态

日志示例：

```
2026-01-22 15:30:45 | INFO | REQUEST | model=gemini-2.0-flash | input_tokens=10 | output_tokens=20 | total_tokens=30 | status=success | request_id=a1b2c3d4
```

## 运行测试

```bash
# 单元测试
python tests/test_gemini_converter.py

# 端到端测试
python tests/test_e2e.py
```

## 扩展新的模型提供商

1. 在 `proxy/` 目录下创建新的提供商目录，如 `proxy/openai/`

2. 继承 `BaseModelStrategy` 基类并实现所有抽象方法：

```python
from proxy.base.strategy import BaseModelStrategy, ProxyResponse


class OpenAIStrategy(BaseModelStrategy):
    @property
    def provider_name(self) -> str:
        return "openai"

    def convert_request(self, claude_request: dict) -> dict:
        # 将 Claude 请求格式转换为 OpenAI 格式
        pass

    def convert_response(self, model_response: dict) -> dict:
        # 将 OpenAI 响应格式转换为 Claude 格式
        pass

    async def send_request(self, request: dict) -> ProxyResponse:
        # 发送请求到 OpenAI API
        pass

    async def stream_request(self, request: dict):
        # 处理流式请求
        pass
```

3. 在 `config/config.yaml` 中添加新提供商的配置

4. 注册策略到 `StrategyFactory`

## 架构图

```
Claude Code CLI
      ↓
FastAPI Server (main.py)
      ↓
StrategyFactory
      ↓
┌─────────────────────┐
│   Provider Strategy  │
├─────────────────────┤
│  • Gemini (已实现)   │
│  • Grok (待实现)     │
│  • OpenAI (可扩展)   │
└─────────────────────┘
      ↓
Target LLM API
      ↓
Response Conversion
      ↓
Claude Code CLI
```

## 注意事项

1. 确保 API 密钥已正确配置在 `.env` 文件中
2. 不同模型的能力可能有所不同，某些 Claude 特有功能可能无法完全映射
3. Token 计算方式可能因模型提供商而异
4. 建议在生产环境中配置适当的超时时间和错误处理

## License

MIT
