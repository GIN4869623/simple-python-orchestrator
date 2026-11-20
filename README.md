# Simple Python Orchestrator

一个 Python 多智能体后端，包含独立的意图识别服务和编排服务。
两个服务位于不同目录，分别运行在不同的 Python 进程和端口，通过 HTTP 传递结构化意图。

## 请求处理方式

意图服务是网页业务请求的统一入口，识别一次后按实际处理路径调用：

```text
用户请求 → 意图识别与分流服务（18003）
             ├─ 明确单一任务 → 直接调用对应业务智能体
             ├─ 已有标准步骤 → 执行固定业务流程
             └─ 需要动态拆解 → HTTP 请求编排服务（18002）
                                   → 动态规划、调用业务智能体、复盘
```

信息不足时先返回澄清问题，不调用业务智能体。直接调用和固定流程不请求编排服务。
转交动态编排时携带已经识别的意图和同一个请求编号，编排服务不重复识别意图。
实际业务适配器仍由原有配置决定；网络监管和 USRP 未接真实接口时继续使用 Fake。

正式请求入口为 `POST /api/intent/runs`，正文为 `{"request":"任务文本"}`，
立即返回 HTTP 202 与请求编号；通过 `GET /api/intent/runs/<request_id>` 查询识别、
分流、执行阶段、实际调用目标和完整结果。动态任务的 `orchestration_request_id`
可用于查询编排服务的同一请求。服务各保留最近 20 条记录，不自动重试业务请求。

`POST /api/intent/analyze` 保留为纯识别调试接口，不执行任务。
`POST /api/orchestrator/dispatch` 是服务间动态转交接口，需要 Bearer 认证和已就绪的
`orchestrated` 意图；不允许重复请求编号、缺失参数、演示默认参数或未明确指定的硬件执行。
原有 `/api/orchestrator/run(s)` 和 CLI 保留兼容，网页正常入口不经过该兼容调用链。

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
    → intent/service.py → intent/agent.py（识别一次）
        → orchestration/router.py → workflows.py → execution.py（直接 / 固定流程）
        → orchestration/client.py → HTTP → orchestration/api.py
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
./serve-intent --host 192.168.1.106 --port 18003 --orchestration-service-url http://192.168.1.106:18002
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

- 编排服务：`http://192.168.1.106:18002`，接收意图服务转交的动态任务，前端用于跟踪这些任务。
- 意图识别与分流服务：`http://192.168.1.106:18003`，网页通过 `/api/intent/runs` 提交任务。

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
ORCHESTRATION_SERVICE_URL=http://192.168.1.106:18002
```

`ORCHESTRATION_SERVICE_URL` 是意图服务转交动态任务的地址。
`INTENT_SERVICE_URL` 保留给编排器兼容入口和 CLI 的纯意图分析调用。
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

## RIS 文字回复

意图服务与编排服务可通过 `RIS_AGENT_CHAT_URL` 调用 RIS 的 `/api/chat/stream`，
使用现有 `think` 模式接收 SSE，最终以 `done.reply` 作为完整答案。
每个任务使用独立的 `session_id`，`runId` 对应请求、任务和尝试编号。
回复保存在任务结果的 `data.answer` 中，直接显示在意图识别或编排跟踪面板；
Agent 业务服务层维持原有任务展示。这里的“回复已收到”仅表示收到了文字，
不将一段回复当作硬件动作成功的独立证明。流中断且没有最终回复时记录为结果未知，
不自动重新提交 RIS 任务。`RIS_AGENT_TIMEOUT` 默认 180 秒；空的接口地址禁用 RIS。
