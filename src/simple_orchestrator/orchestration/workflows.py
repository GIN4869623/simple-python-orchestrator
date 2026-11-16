"""Predefined business workflows; no model-based task planning."""

from ..models import ExecutionMode, IntentAnalysis, IntentType, Task


def _build_fixed_tasks(
    intent: IntentAnalysis,
    *,
    require_device_query: bool = False,
) -> list[Task]:
    if intent.intent_type is IntentType.KNOWLEDGE_QUERY:
        return [
            Task(
                "answer_knowledge",
                "knowledge",
                "回答用户的通信知识、指导或仿真问题",
                capability="knowledge.explain",
                inputs={"request": intent.evidence},
                expected_output=["knowledge", "assumptions"],
                completion_criteria=["回答覆盖用户问题且不声称执行真实硬件"],
            )
        ]
    if intent.intent_type is IntentType.DEVICE_QUERY:
        capability = intent.required_capabilities[0]
        return [
            Task(
                "query_devices",
                "device_manager",
                "查询网络中的设备清单和在线状态",
                capability=capability,
                inputs={"target_type": intent.target_type},
                expected_output=["network_devices"],
                completion_criteria=["返回匹配设备及其当前状态"],
            )
        ]

    usrp_capability = next(
        capability
        for capability in intent.required_capabilities
        if capability.startswith("usrp.")
    )
    if not require_device_query:
        return [
            Task(
                "execute_usrp_operation",
                "usrp",
                f"执行 USRP {intent.action} 操作",
                capability=usrp_capability,
                inputs={
                    "operation": intent.action,
                    **intent.provided_parameters,
                },
                expected_output=[
                    "operation_id",
                    "actual_configuration",
                    "execution_status",
                ],
                completion_criteria=["操作完成并返回可追踪的 operation_id"],
            )
        ]
    return [
        Task(
            "query_usrp",
            "device_manager",
            "查询在线 USRP 设备及其状态",
            capability="device.list",
            inputs={"target_type": "usrp", "required_capability": usrp_capability},
            expected_output=["network_devices"],
            completion_criteria=["返回匹配设备及其在线状态"],
        ),
        Task(
            "execute_usrp_operation",
            "usrp",
            f"在选定 USRP 上执行 {intent.action} 操作",
            dependencies=["query_usrp"],
            capability=usrp_capability,
            inputs={
                "device_candidates_from": "query_usrp.network_devices",
                "operation": intent.action,
                **intent.provided_parameters,
            },
            expected_output=[
                "operation_id",
                "device_id",
                "actual_configuration",
                "execution_status",
            ],
            completion_criteria=["操作成功且返回可追踪的 operation_id"],
        ),
    ]




def build_direct_task(user_request: str, intent: IntentAnalysis) -> Task | None:
    """Bind a complete single-agent request to an explicit local handler."""
    capabilities = set(intent.required_capabilities)
    if (
        intent.intent_type is IntentType.KNOWLEDGE_QUERY
        and intent.execution_mode is ExecutionMode.INFORMATIONAL
        and capabilities == {"knowledge.explain"}
    ):
        return Task(
            "answer_knowledge", "usrp" if intent.target_type == "usrp" else "knowledge",
            user_request, capability="knowledge.explain",
            inputs={**intent.provided_parameters, "request": user_request},
            expected_output=["knowledge"],
            completion_criteria=["回答覆盖用户问题且不声称执行真实硬件"],
        )
    if (
        intent.intent_type is IntentType.DEVICE_QUERY
        and intent.execution_mode is ExecutionMode.READ_ONLY
        and len(capabilities) == 1
        and capabilities <= {"device.list", "device.inspect"}
    ):
        return Task(
            "query_devices", "device_manager", user_request,
            capability=next(iter(capabilities)),
            inputs={**intent.provided_parameters, "target_type": intent.target_type},
            expected_output=["network_devices"],
            completion_criteria=["返回匹配设备及其当前状态"],
        )
    return None


def build_predefined_workflow(
    intent: IntentAnalysis, *, require_device_query: bool = False,
) -> list[Task] | None:
    """Only the existing single-operation USRP workflow is predefined today."""
    capabilities = set(intent.required_capabilities) - {"device.list"}
    if (
        intent.intent_type is IntentType.HARDWARE_EXECUTION
        and intent.execution_mode is ExecutionMode.PHYSICAL
        and intent.target_type == "usrp"
        and len(capabilities) == 1
        and capabilities <= {
            "usrp.configure", "usrp.transmit", "usrp.receive", "usrp.stop", "usrp.get_status",
        }
    ):
        return _build_fixed_tasks(intent, require_device_query=require_device_query)
    return None
