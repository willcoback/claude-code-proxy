# 项目介绍

这是一个python项目，该项目的作用是，接收claude-code的请求接，对其参数格式进行转换，并将其转变为其他大语言模型的请求参数格式，然后发送给指定大语言模型；并将其他大语言模型的返回结果参数格式进行转换，转为claude-code的结果格式，并返回给claude-code；
以gemini为例：
claude-code 发送请求 claude-code-proxy
claude-code-proxy 解析claude-code请求参数，从配置中获取当前目标大模型厂商（gemini），再次从配置中获取要使用的模型(model=gemini-3-flash-preview),根据这两个参数使用策略模式路由到对应模型的参数解析器中
参数解析器将参数解析并重新封装成指定模型请求需要的参数格式
发送请求到gemini
gemini返回数据给claude-code-proxy
claude-code-proxy解析gemini返回的参数，并重新封装成claude-code所需的参数格式
claude-code-proxy 返回数据给claude-code

# 其他要求

每次请求代理过程中都需要打印日志，并将日志保存在./logs/目录下，日志要求至少包含：请求时间，代理目标模型名称，以及当次请求消耗token数；日志文档每天创建一份

# 项目架构

基础包路径com.claude.proxy
在包下区分模型厂商创建策略实现，如
com.claude.proxy.gemini
com.claude.proxy.grok

## 自动发现机制

项目实现了提供者自动发现机制，无需在 main.py 中手动导入新增的提供者策略。当 proxy 包被导入时，会自动执行以下操作：

1. **扫描 proxy/ 目录**：查找所有包含 converter.py 文件的子目录（如 gemini/, grok/, deepseek/ 等）
2. **动态导入**：自动导入每个 provider 目录下的 converter.py 模块
3. **自动注册**：每个 converter.py 模块中的 StrategyFactory.register() 调用会在导入时执行，将提供者注册到策略工厂中

## 扩展新的模型提供商

新增模型提供商只需遵循以下规则：

1. **创建目录**：在 `proxy/` 目录下创建新的提供商目录，如 `proxy/{provider_name}/`
2. **实现策略**：创建 `converter.py` 文件，继承 `BaseModelStrategy` 并实现所有抽象方法
3. **注册策略**：在 `converter.py` 末尾调用 `StrategyFactory.register("{provider_name}", {StrategyClassName})`
4. **添加配置**：在 `config/config.yaml` 中添加对应提供商的配置节
5. **无需修改代码**：不需要修改 main.py 或其他文件，重启服务即可生效

示例目录结构：
```
proxy/
├── base/
│   └── strategy.py         # 策略模式基类
├── gemini/
│   └── converter.py        # Gemini 策略实现
├── grok/
│   └── converter.py        # Grok 策略实现
├── deepseek/
│   └── converter.py        # DeepSeek 策略实现
└── __init__.py             # 自动发现逻辑
```

注意事项：
- 提供者名称需与 config.yaml 中的配置节名称一致
- 配置中的 `provider.name` 可随时热切换，无需重启服务
- 日志会自动显示当前使用的提供商和模型名称
