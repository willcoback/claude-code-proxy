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
