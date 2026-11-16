# 智能体能力仓库

每个智能体一个 JSON 文件，供编排器了解其能力并分配业务任务。

- `agent_id` / `name`：智能体身份。
- `description`：一句话概括能力。
- `capabilities`：各项能力、必要输入和影响任务安排的限制。
- `constraints`：该智能体的共同限制。

能力条目只要求 `id` 和 `description`；必要时补充 `required_inputs`、`constraints`。
能力 ID 是业务标签，具体工具调用和参数校验由专业智能体处理。
文件路径、源码指纹、接口定义和内部 Skill 匹配规则不进入能力摘要。

`usrp.json` 根据 Agent_SDR 代码整理；`schema.json` 用于格式校验，无需放入 LLM 提示词。
当前规划器尚未加载这些 JSON。
