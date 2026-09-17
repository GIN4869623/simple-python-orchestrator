# Simple Python Orchestrator

这是一个用于学习 LangGraph 的多智能体主编排器。主图负责任务规划、动态分发、
依赖检查、批次 Review、重试与重规划；Knowledge、DeviceManager、USRP 作为独立
专业智能体，通过 Port/Adapter 与主图隔离。

当前专业智能体仍是本地 Fake，不会操作真实硬件。将来它们部署到不同设备时，
可用静态配置的 HTTP + JSON-RPC 2.0 Adapter 替换 Fake。远程智能体可在各自服务
内部使用自己的 LangGraph 和私有 State；它们不是嵌在主图进程里的子图。

下面仅是 **USRP 物理操作** 的路径，不代表每次请求都会经过这些节点：

```text
物理操作请求
  → analyze_intent
  → create_plan
  → dispatch（按依赖动态 Send）
      ├─ Knowledge：检索可信 Skill ID ─┐
      └─ DeviceManager：选择可用设备 ───┴→ review_batch
                                           → USRP（条件进入）
                                           → review_batch → 完成/重试/重规划/停止
```

对于 USRP 物理任务，主图强制要求 Knowledge 与 DeviceManager 前置任务。
两个前置任务可并行；没有可用设备时直接停止。Knowledge 只返回可信 Markdown
Skill 的 ID、版本和摘要。USRP 智能体自行二次读取并校验正文；读取或验证失败时
不得执行物理操作。当前 Fake 模拟这条调用链。

只读请求走更短的路径：查询可用设备或设备状态时，仅调用 DeviceManager；
USRP 专属知识可以调用 USRP 的 `knowledge.explain` 只读能力，不触发 Skill 前置任务、
设备选择或硬件执行。通用通信知识仍可由 Knowledge 回答。Ollama 将请求识别为
`device_query` 后，仍由 LLM 按固定 JSON Schema 填写任务字段；主图只允许单个
DeviceManager 只读查询任务，不允许把设备查询指派给 USRP。若计划违反约束，会
明确报规划错误，不会悄悄改写模型的任务内容。

## 安装与运行

要求 Python 3.11+。项目依赖 LangGraph：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
./run "解释 QPSK 原理" --planner fixed --fake-outcome succeeded
```

`./run` 优先使用项目 `.venv/bin/python`。默认 Planner 为本地 Ollama；如需使用，
参照 `.env.example` 设置 `OLLAMA_MODEL`，并确保 Ollama 服务已启动：

```bash
./run "查询有哪些可用的 USRP 设备" --fake-outcome succeeded
```

Fake 环境默认使用 `demo_defaults`，可演示参数不完整的 USRP 请求。真实远程硬件
模式强制要求 `--parameter-policy strict`，缺参时会在终端继续询问用户。真实硬件
接入前，还必须由 USRP 服务自行实现权限、安全检查和幂等执行。

```bash
./run "使用 USRP 以中心频率 2.4GHz、采样率 1MHz、增益 10dB，发射 QPSK 信号 5 秒" \
  --planner fixed --fake-outcome succeeded
```

## 从启动到结束

1. `./run` 定位项目目录，把 `src` 加入 `PYTHONPATH`，优先用 `.venv/bin/python`，
   否则用系统 `python3` 启动 `simple_orchestrator.main`。依赖仍须先安装到所用解释器。
2. `main.py` 读取 `.env` 和命令行参数；若未提供请求文本，则在终端询问。默认使用
   Ollama Planner，需配置 `OLLAMA_MODEL`；学习流程可显式传 `--planner fixed`，无需
   启动 Ollama。未指定 `--fake-outcome` 且未启用远程模式时，会出现 Fake 结果选择菜单。
3. 程序构建 Planner、Knowledge、DeviceManager、USRP 的 Port，然后构建
   `GraphOrchestrator`。默认 Port 是本地 Fake。仅 `--remote-agents` 会改用静态配置的
   JSON-RPC 地址；该模式要求三个地址齐全、参数策略为 `strict`，且不能指定
   `--fake-outcome`。
4. `orchestrator.run()` 为本次运行生成 `request_id`、初始化 State，并调用编译后的
   LangGraph。`analyze_intent` 先识别意图，`create_plan` 再生成任务；需要补参或不支持
   的请求直接到 `finalize`，不执行智能体。设备查询只能规划一个 DeviceManager 任务；
   USRP 物理任务才会补齐 Knowledge Skill 查询与设备选择两项前置任务。
5. `dispatch` 按依赖选出本批可执行任务，并用 `Send` 发给固定的智能体节点。
   Knowledge 与 DeviceManager 在无相互依赖时同批并行；USRP 只读知识任务可以直接
   进入，USRP 物理任务只在前置结果成功且通过校验后进入，且每批最多发送一个物理
   USRP 任务。每批结果合并到 `results`，随后
   统一经过 `review_batch → apply_review`，可能继续、只重试失败任务、重规划一次，
   或停止。没有可用设备、明确拒绝、前置校验失败及 USRP 状态未知会直接停止。
6. 图进入 `finalize` 后，CLI 输出意图、当前版本计划、任务状态、历次批次 Review 和
   工作流汇总。成功退出码为 `0`，工作流未成功为 `1`，规划异常为 `2`。在 `strict`
   模式下若需补参，CLI 会询问补充内容，并把它追加到请求后**重新运行一张新图**；
   这不是从原图的 checkpoint 继续。

例如，上面的 USRP Fake 命令会走
`入口 → 意图分析 → 规划 → Knowledge ∥ DeviceManager → 批次 Review → USRP → 批次 Review → 汇总`。
若没有可用设备，流程停在前置任务的批次 Review，不会进入 USRP。

## 远程接口边界

远程服务可用后，通过静态地址切换到同步 HTTP + JSON-RPC 2.0：

```dotenv
KNOWLEDGE_RPC_URL=http://knowledge-host:8000/rpc
DEVICE_MANAGER_RPC_URL=http://device-host:8000/rpc
USRP_RPC_URL=http://usrp-host:8000/rpc
ORCHESTRATOR_RPC_TIMEOUT=30
ORCHESTRATOR_PARAMETER_POLICY=strict
```

```bash
./run "查询有哪些可用设备" --planner fixed --remote-agents --parameter-policy strict
```

只有传入 `--remote-agents` 才会调用这些地址；默认始终使用本地 Fake。实际远程
服务及 Knowledge 的 Skill 读取接口还未在本仓库实现，接口契约可在
`src/simple_orchestrator/remote.py` 中查看。Planner 只产生 agent ID 和能力，不得
决定远程 URL。第一版**不放行远程 USRP 物理执行**：在真实服务具备审批、幂等和
操作状态查询前，主图会返回拒绝结果；`--remote-agents` 当前只适合只读接口联调。

## State 与执行规则

- `TaskSpec` 只记录业务任务定义；`task_statuses` 记录当前运行状态。
- `results[task_id]` 保留该任务每次尝试的结果与错误，不覆盖旧记录。
- `plan_history` 保存历次完整计划；调度器只执行当前版本。
- 每个任务最多重试一次，成功任务不随失败任务重跑。
- 整个工作流最多重规划一次。重规划保留成功任务，替换失败及未执行任务，并使用
  新 Task ID；新任务重新获得一次重试机会。
- USRP 返回 `UNKNOWN` 时不盲目重试，以免重复物理操作。
- 第一版不含 checkpoint、人工中断和真实专业智能体内部实现。

## 测试与阅读

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

建议按 `graph_orchestrator.py → models.py → planner.py → ports.py → fakes.py →
remote.py → main.py → tests/test_graph_orchestrator.py` 阅读。
`orchestrator.py` 保留旧串行实现，供迁移对照；CLI 已切换到 LangGraph 主图。
早期开发路线在 `docs/开发总路线.md`，讨论中的架构决策在
`docs/LangGraph迁移决策记录.md`。本次由助手代定的实现细节、代码位置和复盘问题在
`docs/LangGraph实现默认决策与复盘清单.md`；这份清单不是你已经认可的架构决定。
