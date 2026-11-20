# 智能体能力仓库

`registry.json` 集中保存 Agent 记录。每个 Agent ID 对应一条记录，包含独立的版本号和完整的能力描述。`usrp.json` 保留为初始资料，后续在总仓库中维护 USRP。`schema.json` 描述一条 Agent 草稿的字段，仓库工具另行检查记录版本、索引一致性和能力 ID 唯一性。

仓库可在 Linux 上通过 Python 3 独立运行。

## 查询

在项目根目录执行以下命令。

```bash
python3 agents/registry.py validate
python3 agents/registry.py summaries
python3 agents/registry.py details usrp
```

`summaries` 从完整记录中提取 Agent ID 和简述。`details` 接收一个或多个候选 ID，按 ID 返回完整的 `profile` 和该记录的 `revision`。索引和详情来自同一份仓库数据。

## 服务位置与接口

Agent 的完整记录可以包含一组 `interfaces`。每个接口记录用途、协议和完整地址。HTTP 和 SSE 还应注明请求方法，请把主机、端口与调用路径写在同一个地址里。TCP 使用 `tcp://主机:端口`。可用 `verification` 区分配置登记和人工验证，`configured` 表示已登记配置，`verified` 表示已做过验证。该标记只记录核对结果。前端可直接维护这些字段，粗筛索引仍只提供 ID 和简述。

## 新增和更新

先准备一份单 Agent JSON 草稿。可以参考 `usrp.json` 的结构，或从 `details` 的结果复制目标 Agent 的 `profile`。草稿需声明自己的 `agent_id`，并与命令指定的目标一致。能力可以显式写 `required_inputs: []` 表示没有必填参数，省略则表示仓库尚未声明。

```bash
python3 agents/registry.py add 新AgentID 草稿.json
python3 agents/registry.py update usrp 草稿.json --expect-revision 1
```

替换示例中的 ID、草稿路径和版本号。新增时 ID 必须是新的。更新时使用编辑前查询到的版本号，每次更新只替换该 Agent 的完整记录。如果出现版本冲突，重新读取并合并修改后再提交。日常写入通过这两个命令完成，工具会加锁并基于最新仓库保存。需要撤回一条记录的变更时，可将上一版的 `profile` 当作草稿再次更新，并提供当前版本号。

## 验收

```bash
python3 -m unittest discover -s agents -p 'test_registry.py' -v
```

测试在临时仓库内演练重复新增、错误 ID、并发更新、旧草稿冲突、索引与详情对应关系，以及保存失败时保留原仓库。
