> 已归档：本文保留原有记录，当前项目的阅读与开发入口见 [开发总路线](../开发总路线.md)。

# LangGraph 迁移决策记录

本文件记录用户确认的流程选择和此前讨论。助手在本次实现时自行选定的具体做法、
对应代码位置及待复盘问题，另见 [实现默认决策与复盘清单](LangGraph实现默认决策与复盘清单.md)。

## 2026-09-17 实现进度

补充路由决定：查询可用 USRP 设备是只读 `device_query`，仅由 DeviceManager
独立完成，不进入 USRP 物理节点，也无需 Knowledge。USRP 专属知识允许由 USRP
智能体以只读 `knowledge.explain` 回答；仅真实物理操作需要 Knowledge Skill 与
DeviceManager 前置结果。用户进一步明确：Task 字段结构固定，但具体任务内容
由 LLM 填写，代码只负责硬性边界校验。因此 Ollama 在 `device_query` 时仍执行
`planning`；若其任务指向 USRP 或其他非 DeviceManager 节点，代码拒绝该计划，
不擅自替换为固定 Task。代码目前仍会把设备查询意图的能力规范为只读
`device.list`/`device.inspect`，并限制 Planner 可见的执行目标为 DeviceManager。

补充：用户明确要求 `batch_review` 在非 `replan`（正常继续、重试、完成或停止）时
返回 `revised_tasks: []`；只有真正 `replan` 才填写新任务。Ollama 评审提示已写明
该规则，代码继续拒绝动作与任务列表不一致的响应。

用户授权助手按已确认决策和保守默认值直接修改代码；仅在业务流程需要改变时再
集中询问，不继续逐个追问细小字段。当前 CLI 已改为运行 `GraphOrchestrator`：
使用 LangGraph `StateGraph`、条件路由、`Send` 动态分发与 State reducer。
Knowledge 与 DeviceManager 在物理路径可并行，USRP 物理执行只在前置任务成功后进入。执行状态与
Task 定义分离；结果按 Task ID 保存全部尝试，计划版本保留历史。批次统一 Review，
失败任务单独 retry 一次，支持最多一次 replan。无设备直接停止，USRP Skill 验证
失败禁止物理操作，USRP `UNKNOWN` 禁止盲目重试。

`FixedPlanner` 在普通任务 retry 耗尽后会确定性地产生一次新计划；新任务使用新 ID，
已成功前置任务及其结果复用。明确无设备、拒绝执行、可信 Skill 校验失败和 USRP
状态未知不会通过自动 retry/replan 绕过。远程 USRP 服务尚缺审批、幂等和状态查询
契约，因此第一版默认拒绝远程物理执行。

本地 Fake 仍是默认运行方式。`remote.py` 提供同步 HTTP + JSON-RPC 2.0 客户端
Adapter，`--remote-agents` 才启用；本仓库尚未实现各远程服务端和正式可信 Skill 库。
当前 Markdown Skill 文件仅用于 Fake 演示。
旧 `orchestrator.py` 暂留作新旧架构对照。下文保留讨论过程；若与此节冲突，
以此节和最新代码为准。

> 用途：持久记录本项目的架构讨论，使后续会话或上下文压缩后能够恢复关键细节。
> 每轮讨论后更新“已确认决策”“未决问题”和“下一步”。实现时以已确认决策为准，
> 不把仍在讨论中的方案误当成最终设计。

## 项目背景

用户正在学习 LangGraph，希望通过把 `simple-python-orchestrator` 从普通 Python
编排循环迁移到 LangGraph 来巩固知识。State、Node 和 Edge 的最终定义由用户决定；
助手负责提出设计问题、检查选择是否自洽、指出遗漏，并标明需要补充学习的概念。
用户希望后续把剩余架构问题一次性集中提出，按编号批量回答；不要每轮只追问一个
小问题。对暂不理解的概念，用户可以标记“待补课”，助手再针对性解释。

当前项目已有以下行为：

- Ollama 或 Fixed Planner 分析意图并创建动态任务计划。
- Orchestrator 按计划列表顺序逐个执行 Knowledge、DeviceManager、USRP，并在执行
  当前任务前检查其依赖是否已成功。
- 每个任务执行后由 Planner review，返回 `continue`、`retry`、`replan`、
  `complete` 或 `stop`。
- 支持缺参澄清、不支持请求、失败、拒绝、未知结果、重试上限和重规划上限。
- 当前执行是同步、串行的；外部智能体由 Port 和 Fake 隔离。

## 已确认决策

记录日期：2026-09-15

1. 第一版保留现有业务能力和终止语义，但不再要求执行过程严格等价；任务执行模型
   将从串行任务下标循环升级为动态任务分发。
2. 总体架构采用“主编排图 + 远程专业智能体”。主图负责全局编排；专业智能体
   可以在各自设备上建立内部图和私有 State。它们不是嵌入主图进程的 LangGraph 子图。
3. Knowledge、DeviceManager、USRP 在主图中采用独立节点，不使用单一的
   `execute_task` 节点统一分发。
4. 第一版允许动态分发任务；Knowledge 与 DeviceManager 是否在所有无依赖场景下
   并行，以及每批任务的并发边界和结果聚合规则尚未确定。
5. USRP 是编译图时存在的固定节点，通过条件边决定是否进入，不在运行时创建节点。
6. `analyze_intent` 和 `create_plan` 拆成两个节点。
7. 第一版暂不加入 checkpoint 和人工中断；由于选择动态分发，第一版需要明确是否
   启用实际并发以及并发上限。
8. 动态 `Task` 采用动态分发形式：调度路由根据运行时计划，把不同任务及其局部
   上下文发送给预先注册的 Knowledge、DeviceManager 或 USRP 节点。节点仍然固定，
   动态的是分发次数、目标和每次调用的数据。
9. 每个专业智能体使用自己的内部 State。主图只传入任务上下文，远程智能体返回
   统一结果；主图 State 不直接暴露给远程智能体。

## 已核实的当前实现语义

### Knowledge 与 DeviceManager 不是固定串行链

`FixedPlanner` 的意图类型是互斥的：

- `knowledge_query` 只生成一个 Knowledge Task。
- `device_query` 只生成一个 DeviceManager Task。
- 默认的 USRP 硬件请求只生成一个 USRP Task。
- 启用 AgentRegistry 后，USRP 请求生成 `DeviceManager -> USRP` 两个 Task，后者
  显式依赖前者。

`OllamaPlanner` 可以生成包含多个智能体的动态任务列表。如果同一计划里同时存在
Knowledge 和 DeviceManager，当前 Orchestrator 不会并行，也不会主动寻找任意一个
依赖已满足的任务，而是严格按照 `tasks[0]、tasks[1]……` 的列表顺序处理：

1. 取当前下标的 Task。
2. 检查其全部依赖是否已经成功；不满足就立即标记 `SKIPPED`。
3. 执行对应 Port。
4. 保存结果并调用 Planner review。
5. review 为 `continue` 时才将下标加一。

因此当前所谓“串行”是“计划列表顺序 + 每项执行后 review”，不是完整的 DAG
就绪任务调度器。即使两个 Task 都没有依赖，也仍然按列表顺序逐个执行。

## 迁移预期效果

迁移的目的不是仅用 LangGraph API 重写当前 `while` 循环，而是获得以下可观察效果：

- Planner 生成的无依赖任务可动态分发，互不依赖的专业智能体可同时工作，依赖任务
  按批次推进。
- 调度依据从“任务列表下标”升级为“任务依赖是否满足”，独立任务不会因为列表中
  较早的无关任务失败而必然失去执行机会；具体失败传播策略仍由用户定义。
- Knowledge、DeviceManager、USRP 作为主图中的固定调用节点，远程服务内部可以
  拥有私有 State 和自己的图；主图仍只依赖统一输入、输出契约。
- retry、replan、失败传播、结果汇合和终止条件在图的节点与边中显式表达，执行路径
  更容易观察和针对性测试。
- 多分支结果通过显式 reducer/聚合规则写回主 State，能够保存每个任务的独立结果
  和尝试记录，避免依赖共享可变对象的隐式更新。
- 后续加入 checkpoint 后，可从图状态恢复而不是从头重新执行；加入 interrupt 后，
  可在 USRP 物理操作前暂停审批。这两项不是第一版的即时收益。

这些收益不会因为安装 LangGraph 自动出现。如果把整个原有 `Orchestrator.run()` 包装
成单个节点，行为和可维护性几乎不会改变；收益来自当前正在设计的动态分发、依赖
  调度、隔离远程智能体、显式 State 合并和条件路由。

## 当前需要澄清的架构问题

## 2026-09-16 用户提出的两阶段编排设想（待确认）

用户当前设想的执行结构是：

1. 主图先判断原始业务请求是否需要 USRP。
2. 将前置任务同时分发给 Knowledge 和 DeviceManager。
3. 等待两个分支返回并汇合。
4. 只有原始请求需要 USRP 时，才通过条件边进入固定的 USRP 专业节点/子图。
5. USRP 子图读取 Knowledge 与 DeviceManager 返回的信息，然后自行规划 USRP 内部
   任务并执行。
6. Review 的位置、粒度和决策冲突规则尚未确定。

这一设想不再是简单地把主 Planner 已生成的所有 Task 一次性动态分发，而是“两阶段
规划”：主图负责前置信息收集和是否委派 USRP；USRP 子图负责收到上下文后的内部
规划。仍需决定 Knowledge 与 DeviceManager 是否对所有请求都执行，还是只对 USRP
硬件请求执行；也需决定它们各自必须提供什么输出，以及某一前置分支失败时能否进入
USRP。

### 2026-09-16 后续确认

1. Knowledge 在 USRP 硬件路径中的主要职责，是提供类似 Skill/SOP 的操作流程、
   工具使用指导和规划参考，使 USRP 智能体更容易制定或落实工具调用步骤。
2. DeviceManager 没有找到可用设备时，工作流直接终止，不进入 USRP。
3. 主 Planner 在第一次规划时就需要包含 USRP 要完成的任务，而不是等 Knowledge 和
   DeviceManager 返回以后才首次决定 USRP 的业务目标。
4. 仍待区分两层规划的粒度：主 Planner 是只定义 USRP 的目标、约束和验收标准，
   还是连具体工具步骤也一起生成；若主 Planner 已生成完整工具步骤，USRP 子图是否
   还有内部规划职责需要重新定义。

### 2026-09-16 两层规划边界已确认

1. 主 Planner 只规划业务级 Task，包括任务目标、目标专业智能体、依赖、用户明确
   提供的输入、预期输出和完成标准；它不生成 USRP 内部的具体工具调用步骤。
2. Knowledge 不临时编造 USRP 操作流程，而是从可信 Skill/SOP 库检索与当前业务
   Task 匹配的操作流程。
3. USRP 子图接收业务级 USRP Task、可信 Skill/SOP、选定设备信息和用户明确参数，
   在自己的内部 State 中规划具体工具步骤并执行。
4. 主图只接收 USRP 子图返回的统一结果，不读取 USRP 子图的内部规划 State。

当前业务级依赖模型可以表达为：Knowledge 流程检索 Task 与 DeviceManager 设备选择
Task 同批动态分发；USRP Task 同时依赖二者。DeviceManager 无可用设备时，USRP Task
不分发并直接终止工作流。

### 2026-09-16 主 State 与消息格式决策

已确认：

1. Task 的业务定义与运行状态分离。业务计划使用不可变或不在执行阶段修改的
   `TaskSpec`；状态、尝试和时间等信息放入独立的执行记录。
2. retry 不覆盖旧结果。每个 Task 保留所有 attempt 及其结果，便于定位失败原因。
3. Skill/SOP 和设备信息不提升为主 State 的专用字段，统一保存在按 Task 关联的
   `results` 中；依赖任务通过 Task ID 读取所需结果。
4. 是否区分 `DISPATCHED` 与 `RUNNING` 尚未决定。

用户提供了一版 HTTP + JSON-RPC 2.0 消息格式设想。该设想需要先解决：

- Knowledge 查询输入包含 DeviceManager 返回的 `selected_devices`，与二者“同时动态
  分发”的设想冲突。若坚持并行，Knowledge 只能根据初始请求中的设备类型、操作和
  capability 检索通用 Skill；若要检索具体设备型号的 Skill，则必须串行或增加第二次
  精确检索。
- 需要区分 LangGraph 内部 State/分支输入与远程传输 DTO。只有智能体作为独立服务
  部署时才需要 HTTP + JSON-RPC；同一 Python 进程中的子图可以直接使用类型化对象。
- 若存在远程投递和接收确认，`DISPATCHED` 与 `RUNNING` 有实际观测意义；若第一版
  只有进程内 `Send`，可以先从 `PENDING` 直接进入 `RUNNING`。
- 示例使用 RIS/CAR，但当前项目真实硬件边界只支持 USRP；迁移示例需要换成 USRP，
  除非另行决定扩展业务范围。

### 2026-09-16 消息语义进一步确认

1. 原消息中的 `selected_devices` 并非 DeviceManager 返回的具体设备实例 ID，而是指
   后续可能委派的专业智能体/设备类型，例如 `usrp`，未来可能扩展 `ris`。为避免与
   `USRP-01` 一类设备实例混淆，应改名为 `target_agent_types`、`target_resource_types`
   或以 `required_capabilities` 表达。
2. 因为 Knowledge 只依赖主 Planner 已知的目标专业智能体、设备类型、操作和能力，
   它可以与 DeviceManager 并行；Knowledge 不需要等待实际设备选择结果。
3. 可信 Skill/SOP 设想为 Markdown 文件。Knowledge 结果返回 Skill ID/索引，而非
   Markdown 全文；USRP 服务再向 Knowledge 的 Skill 读取接口获取 Markdown。
4. 第一版任务状态不区分 `DISPATCHED`，采用 `PENDING -> RUNNING -> 终态`。若未来
   改成带接收确认的异步远程任务，再增加 `DISPATCHED`。
5. 已确认后续多个专业智能体运行在不同设备上，因此目标接口采用同步 HTTP +
   JSON-RPC 2.0。当前专业智能体尚未完成，开发阶段由本地 Fake 直接返回结果；
   主图通过 Port/Adapter 隔离调用方式，不能把 Fake 调用描述成实际 HTTP 请求。
6. RIS 仅作为未来可扩展的专业智能体示例；是否纳入第一版范围尚未改变，当前实现
   仍只支持 USRP 真实硬件任务。

### 2026-09-16 分布式部署边界确认

1. 主 LangGraph 中的 Knowledge、DeviceManager、USRP 是调用节点/适配器节点；
   当前接本地 Fake，远程服务可用后切换为同步 HTTP + JSON-RPC Adapter。
2. 每个远程专业智能体可以在自己的服务内部拥有独立 LangGraph 和内部 State，但主
   图不直接挂载或读取该远程图，只通过稳定输入/输出契约交互。
3. 第一版仍不增加 `DISPATCHED`：任务从 `PENDING` 进入 `RUNNING`，收到结果后
   转为终态。未来远程请求超时或连接中断时保留 `UNKNOWN` 语义，因为远端可能已
   执行但响应丢失。
4. Knowledge 任务结果保存 Skill ID、版本等索引信息。USRP 服务使用自行配置的
   Knowledge Skill 读取地址二次获取 Markdown；不能返回仅 Knowledge 设备本地可见
   的文件路径。Fake 阶段应模拟该契约，而非要求真实网络访问。
5. 主编排器静态配置各远程专业智能体的服务地址；Planner 只能选择 agent ID 和
   capability，不能生成或修改 endpoint。USRP 服务访问 Knowledge Skill 读取接口
   所需的地址也由可信配置提供，而非由 Planner 生成。

### 2026-09-16 最新部署与 Skill 决策

1. 目标通信接口：同步 HTTP + JSON-RPC 2.0。
2. 当前运行实现：尚未建成的专业智能体由本地 Fake 直接返回；保留可替换为远程
   HTTP Adapter 的 Port 边界。
3. Knowledge 查询只返回可信 Markdown Skill 的 ID/索引；USRP 智能体自行二次读取
   Skill 正文，内部规划工具步骤。Knowledge 的 Skill 读取接口需要单独定义。
4. 远程专业智能体地址由主编排器静态配置。USRP 服务读取 Skill 的地址由其自身的
   可信配置提供。
5. 第一版不区分 `DISPATCHED` 与 `RUNNING`。
6. USRP 在获取并验证可信 Skill Markdown 正文前禁止任何物理执行。Skill 二次读取
   失败、版本不匹配或内容校验失败时，USRP 返回明确的业务失败结果，不继续调用硬件
   工具。由于此时物理操作尚未开始，不能把它误报为“可能已执行”的 `UNKNOWN`。
7. Review 采用批次汇合模式：Knowledge 与 DeviceManager 的并行任务全部返回后，
   主图进行一次统一的前置 Review；若准入条件满足，USRP 执行完成后再做一次最终
   Review。第一版不让 Knowledge 和 DeviceManager 各自并行产生相互冲突的 Review
   决策。
8. 这要求重新设计当前 `PlannerPort.review(state, current_task, result)` 的单任务
   输入契约，使前置 Review 能看见一个批次的多个任务结果；具体输入、动作和 retry/
   replan 规则仍由用户定义。DeviceManager 无可用设备以及 USRP 未通过可信 Skill
   验证，属于不可由 LLM Review 覆盖的物理执行硬性门禁。
9. 前置批次发生部分失败且决定 retry 时，仅重新分发失败的任务；已成功任务不重跑，
   其结果保留并在下一次汇合时复用。例如 Knowledge 失败而 DeviceManager 成功，
   只重试 Knowledge。所有尝试记录继续保留，不覆盖旧结果。
10. “没有可用设备”已经确定为直接终止，不属于上一条的自动 retry 场景；远程调用
    超时、服务暂不可用等其他失败类型的可重试性仍待定义。
11. 第一版每个 Task 独立计算重试次数，上限为一次；不再让并行任务共用当前项目的
    全局 `retry_count`。原始执行和重试执行的结果都保存在该 Task 的尝试历史中。
    当前使用 Fake，这个上限足以验证重试路由；未来接入真实远程服务后再结合错误
    分类和物理操作幂等性重新评估。
12. 对普通可重试任务，前置任务用完一次 retry 后若仍失败，允许进入一次 replan，
    而不是立即终止；replan 后仍失败且新一轮 retry 用尽时，终止并返回具体错误。
    两轮所有尝试结果与错误均保留。DeviceManager 明确报告“无可用设备”仍按更早
    决策直接终止，无须消耗 retry 或 replan 次数。
13. 第一版每次任务尝试的最小错误定位信息先采用：`request_id`、`task_id`、
    `agent_id`、尝试序号、错误类型/代码、错误消息。该结构是可扩展的初版；用户后续
    可再补字段。每次尝试还需要关联该次业务结果和状态，但具体 State schema、键名
    及 reducer 尚未决定。详细异常堆栈不直接作为最终用户答复内容。
14. 主图 `results` 采用按 `task_id` 索引的尝试历史：
    `dict[task_id, list[AttemptResult]]`。每个并行分支仅提交自己的新增尝试记录，
    通过合并规则汇入主 State；retry 追加记录，不覆盖旧尝试。Knowledge 返回的
    Skill ID 与 DeviceManager 返回的设备信息保存在相应尝试的业务结果中。
    合并规则对重复 `attempt_id` 的处理尚未决定。
15. 主图另设 `task_statuses: dict[task_id, TaskStatus]`，只表达每个任务当前所处状态：
    `PENDING`、`RUNNING`、`SUCCEEDED`、`FAILED`、`REJECTED`、`UNKNOWN`、
    `SKIPPED`。`TaskSpec` 不存运行状态；`results` 是尝试历史，不能代替状态表，
    因为尚无结果时需要区分待执行、运行中和已跳过。
16. 同一调度批次不得重复分发相同 `task_id`。若出现重复，视为编排错误并明确报错；
    `results` 或 `task_statuses` 的合并规则不得静默覆盖这种冲突。不同 Task 对各自
    ID 的并行更新可以合并。跨批次的 retry 仍使用原 `task_id`，但创建新的尝试记录。
17. 第一版保留 `replan`，整个工作流最多一次。普通任务在当前计划中的一次 retry
    耗尽后可以触发 replan；新计划任务重新获得各自的一次 retry 机会。若已用完
    replan 机会，后续任务 retry 耗尽则进入失败终态并返回具体错误。此规则取代
    上一轮对“retry 耗尽即终止”的临时理解。`plan_version` 因此需要保留。
18. replan 保留已成功任务及其结果，不重新分发它们；替换当前失败任务与尚未执行
    的下游任务。新计划允许引用保留下来的成功任务结果。旧的失败尝试历史保留供
    排查，但不能被当成成功依赖。Skill 读取/校验失败是否允许 replan 尚需单独
    决定；任何 replan 都不能绕过设备可用性和
    可信 Skill 验证的物理执行门禁。
19. replan 生成的每个新任务必须使用本次工作流内此前未出现过的新 `task_id`；
    被保留的成功任务沿用原 ID。新任务的依赖允许引用这些保留的成功 Task ID，
    不允许引用已失败或被替换任务作为成功前置。这样 `results[task_id]` 的尝试历史
    不会混合不同业务任务，旧计划仍可供审计和排错。
20. 主 State 保留历次完整计划，不在 replan 时覆盖旧计划。每个计划版本应能对应
    当时的业务级 Task 定义，方便将历史尝试和错误定位到原计划。调度器只从当前
    有效计划选取待执行任务，不能再次分发旧版本已被替换的任务。具体 `PlanSnapshot`
    schema 与当前版本指针字段仍待定义。

### 1. 动态分发批次

需要决定调度器每次分发一个 Task，还是一次分发当前所有依赖已满足的 Task。后者
形成按依赖层级推进的并行批次，需要定义并发限制以及 USRP 是否允许进入并行批次。

### 2. 分支输入 State

需要为每个动态任务定义局部输入，只携带该专业节点真正需要的数据。至少要决定是否
包含原始请求、当前 Task、其依赖结果、尝试次数、计划版本和运行标识。

### 3. 并行结果聚合

多个分支可能在同一轮返回任务结果。需要定义 reducer/合并规则、结果唯一键、重复
结果处理方式和最终展示顺序，避免并发更新相互覆盖。

### 4. Review 决策规则

已选择一批任务全部结束后统一 Review。仍需定义前置批次 Review 的输入、允许动作、
部分失败时的 retry 粒度、replan 语义，以及 USRP 后最终 Review 的终止条件。

### 5. 依赖和失败传播

需要决定失败、拒绝或未知结果是否阻止其所有后继任务，独立分支是否继续，以及等待
一个失败依赖的任务是标记 `SKIPPED` 还是保留为不可运行状态。

### 6. Retry 与 Replan

需要决定 retry 是只重新分发失败任务还是重跑整个批次；replan 发生时如何处理同一
批次已经成功、失败或仍在运行的任务；新计划能否复用旧结果；任务 ID 如何保持唯一。

### 7. USRP 副作用边界

需要决定 USRP 是否参与并行分发、是否同一时刻最多一个 USRP 任务、动态重试如何
防止重复物理操作，以及未来人工确认应该插在分发前还是专业子图内部。

### 8. 主图与未来子图的契约

已确定采用隔离 State：主图通过包装节点把主图中的任务上下文转换成专业子图输入，
调用子图后再把其输出转换成主图的统一结果。下一步仍要定义任务上下文和统一结果
各自包含哪些字段。当前 Port 的共同输出类型是 `AgentResult`，但 Knowledge 和
DeviceManager 使用 `invoke()`，USRP 使用 `execute()`。

## 下一轮

1. 定义动态分发批次：单任务分发，还是所有就绪任务并行分发。
2. 定义批次 Review 的输入与动作，尤其是部分失败时只重试失败任务还是整批重试。
3. 定义分支输入、统一结果和主 State reducer。
4. 定义依赖失败、retry、replan 和 USRP 副作用规则。
5. 在上述选择稳定后，由用户定义主图 State 的字段、字段所有者和更新规则。
6. 再定义各节点的输入、输出和条件边。

## 做出动态分发决策前的学习路线

用户决定先掌握必要概念，再回答动态批次、USRP 并行和 Review 粒度问题。学习顺序如下：

1. `StateGraph` 基础：State、Node、Edge、`START`、`END`、节点返回局部更新。
2. State channel 与 reducer：默认覆盖、追加/字典合并、并发写冲突，以及 reducer
   应具备的确定性合并性质。
3. super-step、fan-out 与 fan-in：同一轮并行分支、汇合屏障、批次的含义。
4. 条件边与 `Command`：只路由和“更新 State 后立即路由”的区别；固定节点与动态
   路由的区别。
5. `Send`：根据运行时数据向固定节点发送数量不确定的局部 State；它不是动态创建
   节点。
6. DAG 任务调度：依赖图、就绪任务集合、拓扑层、失败传播。这不是 LangGraph
   专属概念，却是本项目正确动态分发的基础。
7. 隔离 State 的子图：父图包装节点负责父子 State 映射，子图只遵守输入输出契约。
8. 并发错误与副作用：业务失败和异常的区别、retry 粒度、幂等，以及 USRP 等物理
   副作用节点的并发边界。
9. Review 协调：逐任务 Review、批次 Review、多个决策的仲裁规则。

暂缓学习：checkpoint、interrupt、持久化、部署。它们不影响当前三个架构决定，待
第一版动态分发流程稳定后再加入。

建议练习顺序：

1. 单条件分支：`A -> B/C -> END`。
2. 固定 fan-out/fan-in：`A -> [B, C] -> D`，用 reducer 合并结果。
3. 动态 map：Planner 生成若干简单数据，使用 `Send` 调用 worker 并聚合结果。
4. 两层依赖 DAG：先动态分发两个无依赖任务，成功后再分发依赖它们的第三个任务。
5. 隔离 State 子图：父图传入任务上下文，子图返回统一结果。

## 恢复会话提示

新会话或上下文压缩后，请先阅读本文件，再阅读：

1. `src/simple_orchestrator/models.py`
2. `src/simple_orchestrator/orchestrator.py`
3. `src/simple_orchestrator/ports.py`

然后从“当前需要澄清的架构问题”继续，不要跳过用户尚未决定的内容，也不要直接
替用户定义 State、Node 或 Edge。
