# Simple Python Orchestrator

这是一个用于学习的最小 LLM 编排智能体骨架。

当前版本先分析业务意图，再由本地 Ollama 模型生成结构化计划并交给编排器执行。
能力注册表相关代码暂时保留，但默认不参与规划：

```text
用户请求
  → OllamaPlanner.analyze_intent()
  → OllamaPlanner.create_plan()
  → Orchestrator 按依赖执行任务
  → Knowledge / DeviceManager / USRP Port
  → OllamaPlanner.review() 评估每个返回
  → 继续 / 重试 / 重规划 / 成功 / 停止
  → Orchestrator 汇总结果
```

其中：

- Planner 使用 Ollama `/api/chat` 接口分别生成结构化意图和动态任务计划。
- 能力注册表与严格计划校验暂时默认关闭；USRP 任务可直接交给 `usrp`，不再强制插入 `device.select` 或引用 `device_id`。
- 如需恢复旧的能力匹配和前置条件校验，可使用 `--enable-agent-registry` 或设置 `ORCHESTRATOR_ENABLE_AGENT_REGISTRY=1`。
- 每个后端返回 `AgentResult` 后，LLM 使用结构化决策判断下一步。
- `FixedPlanner` 仍用于离线运行和单元测试。
- 知识智能体只保留一个 `KnowledgeAgentPort.invoke()` 接口。
- 设备管理智能体查询设备信息、状态与能力，并为 USRP 任务选择可用设备，只保留一个 `DeviceManagerPort.invoke()` 接口。
- USRP 硬件智能体只执行真实 USRP 配置、发射、接收、停止和状态查询；软件仿真及非 USRP 硬件不会启用它。
- 三个依赖使用 Fake 运行，因此无需模型、数据库或真实硬件即可阅读和运行。
- 当前 Fake 环境默认采用 `demo_defaults`：允许模糊 USRP 请求，并明确记录由 Fake 默认补全的参数。接入真实硬件时应切换为 `strict`。

## 现在没有什么

第一版故意不加入：

- LangGraph
- MCP / A2A
- 并发
- checkpoint 和数据库
- RAG 与各专业智能体内部实现
- 真实 USRP 操作

这些功能只有在当前固定闭环通过测试后，才按需增加。

## 项目结构

```text
simple-python-orchestrator/
├── README.md
├── pyproject.toml
├── docs/
│   └── 开发总路线.md
├── src/simple_orchestrator/
│   ├── __init__.py
│   ├── models.py          # 先写：数据和状态
│   ├── agent_registry.py  # 智能体能力注册表
│   ├── validation.py      # 计划业务边界校验
│   ├── ports.py           # 再写：三个外部接口
│   ├── fakes.py           # 固定响应的测试实现
│   ├── planner.py         # FixedPlanner 与 OllamaPlanner
│   ├── orchestrator.py    # 编排主循环
│   └── main.py            # 命令行入口
└── tests/
    └── test_orchestrator.py
```

详细编写顺序见 [`docs/开发总路线.md`](docs/开发总路线.md)。

## 使用 Ollama 规划

Python 代码只使用标准库，要求 Python 3.11 或更高版本。程序会自动读取项目根
目录的 `.env`，真实的系统环境变量具有更高优先级。请先在 `.env` 中填写本地
Ollama 配置，并确保所选模型支持结构化输出：

无需使用 pip 安装，直接运行项目启动器：

```bash
./run
```

也可以直接把任务附在后面：

```bash
./run "解释 QPSK 原理"
```

```bash
cd ~/Desktop/simple-python-orchestrator
PYTHONPATH=src python3 -m simple_orchestrator.main "查询网络中有哪些可用设备"
```

也可以先启动程序，再输入一次任务；执行完成后程序会自动退出：

```bash
PYTHONPATH=src python3 -m simple_orchestrator.main
```

```text
请输入任务请求：如何使用 USRP 发送 QPSK 信号
```

如果已经执行过 `python3 -m pip install -e .`，还可以使用更短的入口命令：

```bash
simple-orchestrator
```

终端会把 LLM 返回的结构化计划解析成中文任务分工和依赖说明，并将它与
Fake 后端的执行结果分区显示。

`.env` 可配置项：

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=<你的本地模型名>
OLLAMA_TIMEOUT=120
OLLAMA_TEMPERATURE=0.2
OLLAMA_LOG_PATH=logs/ollama_responses.json
FAKE_DEVICE_ID=USRP-01
ORCHESTRATOR_PARAMETER_POLICY=demo_defaults
ORCHESTRATOR_ENABLE_AGENT_REGISTRY=0
ORCHESTRATOR_MAX_RETRIES=1
ORCHESTRATOR_MAX_REPLANS=1
```

仓库提供 `.env.example` 作为模板；包含本机配置的 `.env` 已被 `.gitignore` 排除。

每次 Ollama 返回后，程序会更新 `OLLAMA_LOG_PATH` 指向的格式化 JSON 日志。日志
包含 UTC 时间、模型名、对应的用户请求和 Ollama 原始响应，并自动只保留最新
20 条。旧版 `ollama_responses.jsonl` 中的记录会在首次写入时自动迁移到新文件。
日志中的 `phase` 用于区分 `intent_analysis`、`planning` 和执行后的 `review`。

每次任务执行后，LLM 的结构化决策只能是：`continue`、`retry`、`replan`、
`complete` 或 `stop`。整个工作流最多重新执行一次、最多重新规划一次；上面的两个
`ORCHESTRATOR_MAX_RETRIES` 和 `ORCHESTRATOR_MAX_REPLANS` 只能设置为 `0` 或 `1`。

当前三个执行后端都是 Fake，因此参数策略默认是 `demo_defaults`。例如“使用 USRP
发送信号”会继续完成模拟执行，缺少的频率、采样率、增益、时长和波形只会由
Fake 默认补全，不会被 Planner 编造。未来接入真实 USRP 后应配置：

```dotenv
ORCHESTRATOR_PARAMETER_POLICY=strict
```

严格模式发现缺失参数时会在当前终端继续询问，并把用户补充内容合并到原始请求
后重新分析。直接回车可以结束本次请求。

每次运行时，终端会要求用一个序号选择本轮所有 Fake 的回复方案：

```text
1. 成功
2. 拒绝
3. 未知
4. 失败
```

也可以使用 `--fake-outcome succeeded|rejected|unknown|failed` 跳过选择菜单。

也可以使用命令行参数覆盖环境变量：

```bash
PYTHONPATH=src python3 -m simple_orchestrator.main \
  "使用 USRP 以中心频率 2.4GHz、采样率 1MHz、增益 10dB，发射 QPSK 信号 5 秒" \
  --ollama-url "http://localhost:11434" \
  --ollama-model "<你的本地模型名>" \
  --ollama-temperature "0.2" \
  --ollama-log-path "logs/ollama_responses.json"
```

`OLLAMA_TEMPERATURE` 可设置为 `0` 到 `2`：较低值让规划更稳定，较高值让计划
更有变化。任务编排建议从 `0.2` 开始调整。

暂时不调用 Ollama、只验证原来的固定计划：

```bash
PYTHONPATH=src python3 -m simple_orchestrator.main \
  "解释 QPSK 原理" \
  --planner fixed
```

不显示菜单，直接模拟拒绝回复：

```bash
PYTHONPATH=src python3 -m simple_orchestrator.main \
  "解释 QPSK 原理" \
  --planner fixed \
  --fake-outcome rejected
```

## 运行测试

使用标准库运行全部测试：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## 推荐阅读方式

按照 `models.py → agent_registry.py → validation.py → ports.py → planner.py → fakes.py → orchestrator.py → main.py → tests` 的顺序阅读，并配合 `docs/开发总路线.md` 理解每个模块。
