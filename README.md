# Simple Python Orchestrator

一个 Python 多智能体后端，包含独立的意图识别服务和编排服务。
两个服务位于不同目录，分别运行在不同的 Python 进程和端口，通过 HTTP 传递结构化意图。

## 请求处理方式

编排服务保留前端入口，调用意图服务识别请求后，再选择处理路径：

```text
用户请求 → 编排服务入口（18002）
             ├─ HTTP 调用意图识别服务（18003）→ 返回结构化意图
             └─ 根据意图分流
                  ├─ 明确单一任务 → 直接调用业务智能体
                  ├─ 已有标准步骤 → 固定业务流程
                  └─ 需要动态拆解 → 动态规划、调度与复盘
```

信息不足时先返回澄清问题。直接调用和固定流程跳过模型规划与复盘；
Agent Registry 保留后续接入点，默认关闭。

## 文件夹说明

| 目录 | 用途 |
| --- | --- |
| [src/](src/) | Python 源码目录，实际业务代码集中在 `simple_orchestrator` 包中。 |
| [src/simple_orchestrator/intent/](src/simple_orchestrator/intent/) | 意图识别智能体及独立 HTTP 服务，默认端口 `18003`；`client.py` 供编排服务调用。 |
| [src/simple_orchestrator/orchestration/](src/simple_orchestrator/orchestration/) | 编排服务：请求分流、固定流程、动态规划、执行调度和 CLI / HTTP 入口，默认 HTTP 端口 `18002`。 |
| [src/simple_orchestrator/](src/simple_orchestrator/) 下的公共模块 | 数据模型、接口、Ollama 连接、配置读取、业务服务适配、Fake 和日志。 |
| [src/simple_orchestrator/skills/](src/simple_orchestrator/skills/) | Fake USRP 流程使用的操作指导演示文件，用于验证 Skill 检索与执行前置检查。 |
| [agents/](agents/) | 智能体能力描述草稿，包括 USRP 能力 JSON 和格式定义。当前程序尚未加载这些文件，完整 Agent Registry 后续建立。 |
| [tests/](tests/) | 自动化测试，覆盖请求分流、规划、执行、HTTP/TCP 接口和日志，主要使用 Fake 与模拟网络响应。 |
| [examples/](examples/) | 接口消息示例，目前提供 TCP 任务请求 JSON，供理解消息格式和联调参考。 |
| [docs/](docs/) | [开发总路线](docs/开发总路线.md) 是当前详细指南，旧文档保存在 `archive/`。 |
| `logs/` | 运行时生成的模型响应和智能体收发日志；默认不提交 Git。 |
| `.venv/` | 本项目的 Python 虚拟环境，保存解释器和依赖，不是业务源码。 |
| `.agents/`、`.codex/` | 当前为空的开发工具预留目录，服务代码不依赖它们。 |

## 核心代码从哪里看

先沿请求入口阅读，不必一次读完所有文件：

```text
orchestration/main.py（CLI） / orchestration/api.py（HTTP）
    → orchestration/router.py（请求分流）
        → intent/client.py → HTTP → intent/service.py → intent/agent.py
        → orchestration/workflows.py → orchestration/execution.py（直接 / 固定流程）
        → orchestration/graph.py → orchestration/planner.py（动态编排）
```

## 启动两个服务

以下命令在两个 Ubuntu 终端或 SSH 终端中分别执行。
如果旧的 `./serve` 还在运行，先在原终端按 `Ctrl+C` 停止，释放 `18002` 端口。

两个服务默认复用现有 `.env` 中的 `OLLAMA_*` 和 `ORCHESTRATOR_API_TOKEN`。
首次配置参考 `.env.example`；意图服务也可单独设置 `INTENT_OLLAMA_*` 和 `INTENT_API_TOKEN`。

### 终端一：意图识别服务（18003）

```bash
cd /home/wenchaoxia/Desktop/simple-python-orchestrator
source .venv/bin/activate
./serve-intent --host 192.168.1.106 --port 18003
```

看到“意图识别服务：http://192.168.1.106:18003”后，保持该终端运行，再启动编排服务。

### 终端二：编排服务（18002）

```bash
cd /home/wenchaoxia/Desktop/simple-python-orchestrator
source .venv/bin/activate
./serve-orchestrator --host 192.168.1.106 --port 18002 --intent-service-url http://192.168.1.106:18003
```

看到“编排 HTTP 服务”启动信息后，即可从前端提交任务。
本机当前局域网 IP 为 `192.168.1.106`，两个服务只绑定该网卡地址：

- 编排服务：`http://192.168.1.106:18002`，前端继续使用 `/api/orchestrator/*` 接口。
- 意图识别服务：`http://192.168.1.106:18003`，提供 `/api/intent/analyze`。

局域网内的外部程序使用上述地址访问，接口仍需要 Bearer 认证。
网络或机器变化后，用服务器的新 IP 替换启动命令和配置中的地址。

原来的 `./serve` 等效于 `./serve-orchestrator`，只启动编排服务。
意图服务需要单独启动；两个终端需同时保持运行。

### 自定义地址和端口

上面的命令显式指定端口和意图服务地址，命令行参数优先于 `.env`。
需要使用 `.env` 中的自定义配置时，分别运行 `./serve-intent` 和 `./serve-orchestrator` 即可。

```dotenv
ORCHESTRATOR_API_HOST=192.168.1.106
ORCHESTRATOR_API_PORT=18002
INTENT_API_HOST=192.168.1.106
INTENT_API_PORT=18003
INTENT_SERVICE_URL=http://192.168.1.106:18003
```

`INTENT_SERVICE_URL` 是编排服务调用意图服务的地址，也需与意图服务的实际监听地址一致。
跨机器部署时，将其设为意图服务所在服务器的可访问地址。
两端意图接口密钥需要一致；未单独设置 `INTENT_API_TOKEN` 时复用 `ORCHESTRATOR_API_TOKEN`。

## 不调用模型的双服务联调

先停止已运行的两个服务，再在各自终端的项目目录中分别执行：

终端一：

```bash
./serve-intent --backend fixed --host 192.168.1.106 --port 18003
```

终端二：

```bash
./serve-orchestrator --planner fixed --host 192.168.1.106 --port 18002 --intent-service-url http://192.168.1.106:18003
```

这会真实经过两个进程和端口，意图识别及规划使用离线实现。
业务接口仍由 `.env` 决定，配置了 `KNOWLEDGE_API_URL` 时仍会调用真实知识服务。

## 命令行运行

意图服务启动后，可以在另一个终端的项目目录中运行 CLI。
CLI 在当前进程执行编排，并通过 HTTP 调用意图服务：

```bash
./run "查询有哪些可用设备"
```

仅调试本地代码、不启动服务时，显式使用单进程模式：

```bash
./run "查询有哪些可用设备" --local-intent --planner fixed --fake-outcome succeeded
```

默认不会在意图服务不可用时自动切换为本地识别，而是返回错误。

## 停止服务与退出环境

在两个服务终端分别按 `Ctrl+C` 停止对应进程，再分别执行：

```bash
deactivate
```
