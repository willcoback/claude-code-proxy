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
| Grok   | ✅ 已完成 | 支持 Xai Grok 系列模型 |
| DeepSeek | ✅ 已完成 | 支持 DeepSeek Anthropic 兼容 API |

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
│   ├── __init__.py         # 自动发现机制入口
│   ├── base/
│   │   └── strategy.py     # 策略模式基类
│   ├── gemini/
│   │   └── converter.py    # Gemini 策略实现
│   ├── grok/
│   │   └── converter.py    # Grok 策略实现
│   ├── deepseek/
│   │   └── converter.py    # DeepSeek 策略实现
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

# DeepSeek API 密钥（可选）
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

### 3. 修改配置文件

编辑 `config/config.yaml`：

```yaml
server:
  host: "0.0.0.0"
  port: 8080

provider:
  name: "deepseek"          # 当前使用的提供商：gemini、grok 或 deepseek
  fallback_providers: ["gemini"]  # 备用提供商列表（主提供商失败时按顺序尝试）

gemini:
  api_key: "${GEMINI_API_KEY}"
  model: "gemini-2.5-flash" # 使用的模型
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai"
  timeout: 600

grok:
  api_key: "${GROK_API_KEY}"
  model: "grok-4-1-fast-reasoning"
  base_url: "https://api.x.ai/v1"
  timeout: 600

deepseek:
  api_key: "${DEEPSEEK_API_KEY}"
  model: "deepseek-reasoner"
  base_url: "https://api.deepseek.com/anthropic"
  timeout: 600

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

### 4.1. 全局命令启动/停止服务 (推荐)

为了方便在任意终端直接使用 `claude-proxy-start` 和 `claude-proxy-stop` 命令，你可以将脚本创建为全局软链接。

**重要说明**：由于 `run.sh` 和 `stop.sh` 脚本在内部已更新以支持软链接的真实路径解析，无论软链接在哪里创建，脚本都能正确找到其所需的文件。

**推荐步骤**：

1.  **停止可能正在运行的服务**：
    ```bash
    ./stop.sh
    ```

2.  **创建全局软链接**：将 `run.sh` 和 `stop.sh` 链接到系统 PATH 包含的目录（如 `/usr/local/bin`）。这需要 `sudo` 权限。
    ```bash
    sudo ln -s "$(pwd)/run.sh" /usr/local/bin/claude-proxy-start
    sudo ln -s "$(pwd)/stop.sh" /usr/local/bin/claude-proxy-stop
    ```

3.  **赋予执行权限**（通常软链接会继承，但为了确保）：
    ```bash
    sudo chmod +x /usr/local/bin/claude-proxy-start /usr/local/bin/claude-proxy-stop
    ```

4.  **在任意终端使用命令**：现在你可以在任何终端直接输入命令来启动或停止服务。
    ```bash
    claude-proxy-start
    # ... 等待服务启动 ...
    claude-proxy-stop
    ```

**可选方案：在项目内部创建软链接**

如果你不希望在系统级别安装，也可以在项目根目录创建 `bin/` 目录，并在其中创建软链接。这种方式更适用于项目内部管理，但需要将 `bin/` 目录添加到 PATH 或使用相对路径执行（例如 `./bin/start.sh`）。

```bash
# 在项目根目录创建 bin/ 目录
mkdir -p bin
# 创建软链接
ln -s "$(pwd)/run.sh" bin/start.sh
ln -s "$(pwd)/stop.sh" bin/stop.sh
# 赋予执行权限
chmod +x bin/start.sh bin/stop.sh
# 执行方式
./bin/start.sh
./bin/stop.sh
```

**注意事项**：
-   在 `.gitignore` 文件中添加 `bin/` 目录，避免将项目内部的软链接提交到版本控制中。
-   Windows 系统请使用 `mklink` 命令代替 `ln -s`。

---

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

### 自动发现机制

项目实现了提供者自动发现机制，无需在 main.py 中手动导入新增的提供者策略。当 proxy 包被导入时，会自动执行以下操作：

1. **扫描 proxy/ 目录**：查找所有包含 converter.py 文件的子目录（如 gemini/, grok/, deepseek/ 等）
2. **动态导入**：自动导入每个 provider 目录下的 converter.py 模块
3. **自动注册**：每个 converter.py 模块中的 `StrategyFactory.register()` 调用会在导入时执行，将提供者注册到策略工厂中

### 添加新提供者步骤

1. **创建目录**：在 `proxy/` 目录下创建新的提供商目录，如 `proxy/{provider_name}/`

2. **实现策略**：创建 `converter.py` 文件，继承 `BaseModelStrategy` 并实现所有抽象方法：

```python
from proxy.base.strategy import BaseModelStrategy, StrategyFactory, ProxyResponse


class NewProviderStrategy(BaseModelStrategy):
    @property
    def provider_name(self) -> str:
        return "newprovider"  # 必须与 config.yaml 中的配置节名称一致

    def convert_request(self, claude_request: dict) -> dict:
        # 将 Claude 请求格式转换为目标模型格式
        pass

    def convert_response(self, model_response: dict) -> dict:
        # 将目标模型响应格式转换为 Claude 格式
        pass

    async def send_request(self, request: dict) -> ProxyResponse:
        # 发送请求到目标模型 API
        pass

    async def stream_request(self, request: dict):
        # 处理流式请求
        pass

# 自动注册策略（无需修改 main.py）
StrategyFactory.register("newprovider", NewProviderStrategy)
```

3. **添加配置**：在 `config/config.yaml` 中添加对应提供商的配置节：

```yaml
newprovider:
  api_key: "${NEWPROVIDER_API_KEY}"
  model: "newprovider-model"
  base_url: "https://api.newprovider.com"
  timeout: 600
```

4. **配置环境变量**：在 `.env` 文件中添加对应的 API 密钥：

```bash
NEWPROVIDER_API_KEY=your_api_key_here
```

5. **重启服务**：重启代理服务即可生效，无需修改 main.py 或其他代码

### 注意事项

- 提供者名称需与 config.yaml 中的配置节名称一致
- 配置中的 `provider.name` 可随时热切换，无需重启服务
- 日志会自动显示当前使用的提供商和模型名称
- 系统支持配置热加载，修改 config.yaml 后新请求会立即生效

## 架构图

```
Claude Code CLI
      ↓
FastAPI Server (main.py)
      ↓
StrategyFactory
      ↓
┌─────────────────────┐
│   Provider Strategy │
├─────────────────────┤
│  • Gemini (已实现)   │
│  • Grok (已实现)     │
│  • DeepSeek (已实现) │
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
