"""Natural-language business requests shared by local and remote agents."""

from __future__ import annotations

from typing import Any, Mapping


_LABELS = {
    "request": "用户需求", "target_type": "目标设备类型",
    "required_capability": "所需能力", "required_capabilities": "所需能力",
    "device_id": "设备编号", "device_id_from": "设备编号来源",
    "device_type": "设备类型", "device_status": "设备状态",
    "connection_status": "连接状态", "status": "状态", "model": "型号",
    "capabilities": "支持的能力", "site": "所在地点", "waveform": "波形",
    "duration_seconds": "持续时间（秒）", "frequency_hz": "频率（Hz）",
    "center_frequency_hz": "中心频率（Hz）", "sample_rate_sps": "采样率（Sps）",
    "gain_db": "增益（dB）", "skill_id": "操作指导编号", "version": "版本",
    "sha256": "SHA-256 摘要", "knowledge": "知识说明", "assumptions": "假设条件",
    "recommendations": "建议", "network_devices": "设备清单", "answer": "查询结果",
    "selected_device": "选定设备", "device_candidates_from": "设备查询结果来源",
    "usrp.transmit": "USRP 信号发射", "usrp.receive": "USRP 信号接收",
    "usrp.configure": "USRP 参数配置", "usrp.stop": "停止 USRP 操作",
    "device.list": "查询设备清单", "device.inspect": "查询设备状态",
    "knowledge.lookup_skill": "检索可信操作指导",
    "knowledge.explain": "解释相关知识",
}


def _describe(value: Any) -> str:
    if isinstance(value, Mapping):
        parts = []
        for key, item in value.items():
            text = _describe(item)
            if isinstance(item, (Mapping, list, tuple)):
                text = f"（{text}）"
            parts.append(f"{_LABELS.get(key, key)}为{text}")
        return "；".join(parts) or "无"
    if isinstance(value, (list, tuple)):
        return "、".join(_describe(item) for item in value) or "无"
    if value is None:
        return "未提供"
    if isinstance(value, bool):
        return "是" if value else "否"
    text = str(value)
    return _LABELS.get(text, text)


def make_knowledge_request(
    context: Mapping[str, Any], *, use_user_question: bool = False,
) -> dict[str, str]:
    """Send one question with relevant facts, leaving review criteria internal."""
    task = context["current_task"]
    inputs = task.get("inputs", {})
    user_request = context.get("user_request", "")
    candidates = [inputs.get("question"), inputs.get("query"), task.get("description"), user_request]
    if use_user_question:
        candidates.insert(0, user_request)
    question = next((value.strip() for value in candidates if isinstance(value, str) and value.strip()), "")
    if task.get("capability") == "knowledge.lookup_skill" and "可信" not in question:
        question = f"请检索可信操作指导：{question}"
    parts = [question]
    conditions = {
        key: value for key, value in inputs.items()
        if key not in {"question", "query"}
        and not (key == "request" and value in (user_request, question))
    }
    if conditions:
        parts.append(f"补充条件：{_describe(conditions)}")
    for key, prefix in (
        ("previous_results", "参考结果"),
        ("skill_ref", "操作指导"),
        ("selected_device", "设备"),
    ):
        if context.get(key):
            parts.append(f"{prefix}：{_describe(context[key])}")
    return {"message": " ".join(parts)}


def make_agent_request(context: Mapping[str, Any], *, include_user_request: bool = True) -> dict[str, str]:
    """Only this one-field object crosses the business agent boundary."""
    task = context["current_task"]
    sentences = [f"请完成以下任务：{task.get('description', '')}"]
    user_request = context.get("user_request")
    if include_user_request and user_request and user_request != task.get("description"):
        sentences.append(f"用户的需求是：{user_request}")
    capability = task.get("capability")
    if capability == "knowledge.lookup_skill":
        sentences.append("请检索可信操作指导")
    for key, prefix in (
        ("inputs", "已知条件是"),
        ("expected_output", "请提供"),
        ("completion_criteria", "完成时应满足"),
    ):
        if task.get(key):
            sentences.append(f"{prefix}：{_describe(task[key])}")
    for key, prefix in (
        ("previous_results", "可参考的前置任务结果是"),
        ("skill_ref", "可信操作指导引用是"),
        ("selected_device", "已选定的设备是"),
    ):
        if key in context and context[key]:
            sentences.append(f"{prefix}：{_describe(context[key])}")
    paragraph = "。".join(" ".join(text.rstrip("。；\n").splitlines()) for text in sentences)
    return {"message": paragraph + "。"}


def invocation_metadata(context: Mapping[str, Any]) -> dict[str, Any]:
    task = context["current_task"]
    request_id = context.get("request_id")
    task_id = task.get("task_id")
    attempt = task.get("attempt")
    return {
        "request_id": request_id, "task_id": task_id, "attempt": attempt,
        "attempt_id": f"{request_id}:{task_id}:{attempt}",
        "capability": task.get("capability"),
    }
