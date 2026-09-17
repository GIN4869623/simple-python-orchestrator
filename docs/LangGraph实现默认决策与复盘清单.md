# LangGraph 实现默认决策与复盘清单

记录日期：2026-09-17。本文只记录这次迁移中**由助手为落地代码选定、并非用户逐项决定**的做法，供以后学习 LangGraph 时质疑和改进。用户明确确认过的业务流程见 [迁移决策记录](LangGraph迁移决策记录.md)；实际行为以代码和测试为准。这里的“待复盘”不表示当前程序必然有 bug。

## 主图、State 与任务

| 助手采用的做法 | 当时的理由 | 位置与待复盘点 |
| --- | --- | --- |
| 主图集中写在一个 `graph_orchestrator.py`，用 `TypedDict` 描述 `MainState`，用冻结的 `TaskSpec`、`PlanSnapshot`、`AttemptResult` 记录任务、计划和尝试。 | 第一版便于逐段对照旧编排器。 | `graph_orchestrator.py` 的类型定义和 `_build_graph`。图变大后是否拆成 State、Node、Router 模块？运行时输入校验是否需要更强的类型模型？ |
| `results` 按 Task ID 追加全部尝试，并以 `attempt_id` 防重；`task_statuses` 使用字典合并 reducer；`reviews` 追加列表。 | 让并行分支的更新能汇合，同时保留错误定位信息。 | `merge_results`、`merge_statuses`、`append_reviews`。目前状态合并是后写覆盖；未来若同一 Task ID 被并发更新，是否应该直接报冲突？ |
| `plan_history` 保存每版完整快照，调度只看最后一版；每版重新规划后的任务用新 ID，成功任务沿用原 ID。 | 便于把旧任务定义与旧失败记录关联，并复用成功结果。 | `_apply_replan`。若计划数和结果数变大，是否改用持久化存储或事件日志？ |
| 每次 `run` 生成 `req-<uuid>`，内存中调用 `graph.invoke(..., recursion_limit=100)`；未接 checkpointer、会话 ID 或持久化 State。 | 与第一版无 checkpoint 的范围一致，先保证本地流程可运行。 | `run`。`100` 只是防止异常循环的实现值，不是业务上限；恢复、审计、跨进程续跑要另行设计。 |

## 分发与 Review

| 助手采用的做法 | 当时的理由 | 位置与待复盘点 |
| --- | --- | --- |
| 每轮选出所有依赖已成功的待执行任务，用 `Send` 发给固定节点；同轮若出现可执行 USRP 物理任务，则只发送第一个物理任务。 | 实现 Knowledge/DeviceManager 的并行前置，同时保守限制物理操作；USRP 只读知识任务不受物理前置约束。 | `_dispatch`、`_route_dispatch`。目前没有通用并发上限；未来多个设备/智能体如何限流、隔离物理任务？ |
| 对 USRP 计划在主图中自动补 Knowledge Skill 查询与 DeviceManager 设备选择；若 Planner 已给出匹配任务则复用。 | 避免 Planner 漏掉必需的安全前置。 | `_prepare_tasks`。这是代码强制补计划，不完全由 Planner 显式规划；以后是否要求 Planner 明确输出前置任务，并由图只负责校验？ |
| Ollama 已识别为 `device_query` 时，代码把意图能力规范为 `device.list`/`device.inspect`，并把可选 agent 限定为 DeviceManager；Task 内容仍由 LLM 填写，主图验证。此前助手曾直接生成固定 Task，现已按用户要求撤销。 | 使只读设备查询不被升级为 USRP 任务，同时保留 LLM 填写任务内容的学习目标。 | `OllamaPlanner.analyze_intent`、`create_plan_from_intent`、`_prepare_tasks`。意图能力规范化仍是代码选择；以后是否改为只校验、不修正意图能力，值得复盘。 |
| USRP 专属知识暂借用 `knowledge.explain` 能力名，在 USRP 节点走 `invoke` 只读接口；物理任务才走 `execute`。 | 最少改动地表达“同一智能体既能答知识，也能操作硬件”，并确保只读查询不触发物理前置条件。 | `agent_registry.py`、`_usrp`、`FakeUsrpAgent.invoke`。以后是否给 USRP 知识查询独立能力名及独立远程接口契约，需要结合真实智能体设计。 |
| 仅允许主图固定注册的 `knowledge`、`device_manager`、`usrp` 节点；USRP 物理任务要求硬件意图目标为 `usrp`，原始请求中还必须显式出现 `usrp`。设备查询只允许单个 DeviceManager 任务。 | 防止 LLM 擅自增加执行目标或把只读请求升级为物理执行。 | `_prepare_tasks`、`_create_plan`、`_apply_replan`。字符串检查较粗糙；加入 RIS 或用户使用别名时要重新定义授权表达。 |
| 每批任务汇合后统一 Review；无设备、明确拒绝、Skill/设备前置无效、USRP 状态未知先由代码判定 STOP，其他交给 Planner。 | 防止 LLM 覆盖关键安全边界。 | `_review_batch`、`_apply_review`。哪些规则应属于主图、哪些应下沉到设备智能体？代码与 Planner 决策冲突时如何解释？ |
| `FixedPlanner` 的 replan 只是给失败和未执行任务换 ID、保留原定义及成功结果；Ollama 可产生真正的新计划，但主图会校验。 | 给“retry 耗尽后最多一次 replan”提供可重复演示路径。 | `planner.py:FixedPlanner.review_batch`、`_apply_replan`。Fixed 版并不会找新策略，若故障是永久性的，只会再次失败。 |

## Skill、远程接口与安全边界

| 助手采用的做法 | 当时的理由 | 位置与待复盘点 |
| --- | --- | --- |
| Fake Knowledge 使用包内 Markdown 演示文件，先返回 Skill ID、版本、SHA-256；Fake USRP 再 `read_skill` 读取并核对摘要。主图不读取正文。 | 模拟“Skill 索引先返回，业务智能体二次读取”的约定。 | `fakes.py`、`skills/usrp-operation-demo.md`、`_usrp`。本地摘要校验不等于真正的可信发布、签名、权限与版本治理；远程 Skill 读取接口还未实现。 |
| 主图在调用 USRP 前校验 Skill 索引字段和设备状态、能力；Fake USRP 再校验 Skill 正文。 | 尽量让失败在模拟物理执行前暴露。 | `_usrp`、`FakeUsrpAgent.execute`。真实硬件端必须独立实施安全检查，不能只信任主图。 |
| 使用同步 HTTP POST + JSON-RPC 2.0 的 `tasks.send`，按智能体 ID 静态绑定地址；请求带 request/task/attempt 标识，响应校验 JSON-RPC ID、Task ID 和状态。三个服务共用一个可选 Bearer Token 与超时配置。 | 给不同设备部署的智能体留一个最小远程契约，避免 Planner 决定 URL。 | `remote.py`、`main.py`。当前仓库没有服务端；认证、鉴权、密钥分离、重放保护、异步任务查询都需在真正部署前设计。 |
| 远程 USRP 物理执行默认拒绝；调用异常对 USRP 记 `UNKNOWN`，对 Knowledge/DeviceManager 记 `FAILED`。 | 丢失响应时不清楚物理动作是否发生，不能自动重试。 | `_usrp`、`_invoke_branch`。真实端需要审批、幂等键、操作状态查询和人工处置流程；`UNKNOWN` 不等于“未执行”。 |

## CLI、诊断与验证

| 助手采用的做法 | 当时的理由 | 位置与待复盘点 |
| --- | --- | --- |
| 保留原先默认 Ollama 与 Fake 结果菜单；Fake 默认 `demo_defaults`，远程模式强制 `strict`。 | 兼顾原有使用体验与远程硬件安全边界。 | `main.py`、`.env.example`。真实部署前应把安全策略作为服务端能力校验，而不仅是 CLI 参数。 |
| `strict` 缺参时，CLI 把补充文本追加到请求后重新 `run`，会获得新的 request ID。 | 第一版没有 checkpoint/interrupt，也能提供可用的补参流程。 | `main.py:main`。这不是从原图恢复；若需会话级追踪和断点续跑，设计会不同。 |
| 终端只展示当前计划及其状态、历次批次判断与最终摘要；完整 `plan_history`/`results` 留在本次返回的内存 State，没有自动落盘。 | 保持演示输出简洁。 | `main.py`、`GraphOrchestrator.run`。进程结束后只能看到终端文字（以及 Ollama 最近响应日志）；若要长期排查，应添加脱敏持久化或导出命令。 |
| 测试覆盖 Fake 图路径、并行汇合、重试/重规划和远程 Adapter 的模拟响应；没有真实远程服务、Ollama 或硬件端到端测试。 | 当前外部智能体仍是假的。 | `tests/test_graph_orchestrator.py`。接入真实服务前需契约测试与安全失败演练。 |

## 后续复盘顺序建议

先从 `README.md` 的启动流程走一遍，再在 `graph_orchestrator.py` 中对照 `MainState → _build_graph → _dispatch → _review_batch → _apply_review`。接着选上表一项，写下“当前假设、反例、替代方案、需要新增的测试”，不要把这些默认实现直接当成 LangGraph 的标准答案。
