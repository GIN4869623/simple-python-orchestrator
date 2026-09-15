"""Deterministic test doubles for all external dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .models import AgentResult, ResultStatus


@dataclass
class FakeKnowledgePort:
    should_fail: bool = False
    outcome: ResultStatus = ResultStatus.SUCCEEDED
    call_count: int = 0

    def invoke(self, request: Mapping[str, Any]) -> AgentResult:
        self.call_count += 1
        if self.should_fail or self.outcome is ResultStatus.FAILED:
            return AgentResult(
                ResultStatus.FAILED,
                "知识接口不可用",
                error="knowledge unavailable",
            )
        if self.outcome is ResultStatus.REJECTED:
            return AgentResult(
                ResultStatus.REJECTED,
                "知识智能体拒绝处理模拟请求",
                {"reason": "fake knowledge rejection"},
                error="knowledge request rejected",
            )
        if self.outcome is ResultStatus.UNKNOWN:
            return AgentResult(
                ResultStatus.UNKNOWN,
                "知识智能体无法确认查询结果",
                {"reason": "fake knowledge result unknown"},
            )
        return AgentResult(
            ResultStatus.SUCCEEDED,
            "已取得模拟知识",
            {"knowledge": "QPSK 是一种四相位数字调制方式。"},
        )


@dataclass
class FakeDeviceManagerPort:
    selection: str = "selected"
    device_id: str = "USRP-01"
    outcome: ResultStatus | None = None
    call_count: int = 0

    def invoke(self, request: Mapping[str, Any]) -> AgentResult:
        self.call_count += 1
        current_task = request.get("current_task", {})
        capability = (
            current_task.get("capability")
            if isinstance(current_task, Mapping)
            else None
        )
        if self.outcome is ResultStatus.REJECTED:
            return AgentResult(
                ResultStatus.REJECTED,
                "设备管理智能体拒绝本次模拟设备请求",
                {"selection": "rejected"},
                error="device request rejected",
            )
        if self.outcome is ResultStatus.FAILED or self.selection == "unavailable":
            return AgentResult(
                ResultStatus.FAILED,
                "设备选择失败：当前没有可用硬件",
                {"selection": "unavailable"},
                error="no device available",
            )
        if self.outcome is ResultStatus.UNKNOWN or self.selection == "unknown":
            return AgentResult(
                ResultStatus.UNKNOWN,
                "无法从网络资源信息中确认可用设备",
                {"selection": "unknown"},
            )
        selected_device = {
            "device_id": self.device_id,
            "device_type": "usrp",
            "status": "online",
            "capabilities": [
                "usrp.configure",
                "usrp.transmit",
                "usrp.receive",
                "signal_transmit",
                "signal_receive",
            ],
        }
        if capability != "device.select":
            return AgentResult(
                ResultStatus.SUCCEEDED,
                "已获取模拟设备清单和状态",
                {
                    "selection": "not_requested",
                    "network_devices": [selected_device],
                },
            )
        return AgentResult(
            ResultStatus.SUCCEEDED,
            f"已获取设备信息并选择模拟 USRP：{self.device_id}",
            {
                "selection": "selected",
                "device_id": self.device_id,
                "network_devices": [selected_device],
                "selected_device": selected_device,
            },
        )


@dataclass
class FakeUsrpAgent:
    outcome: ResultStatus = ResultStatus.SUCCEEDED
    call_count: int = 0
    received_requests: list[Mapping[str, Any]] = field(default_factory=list)

    def execute(self, request: Mapping[str, Any]) -> AgentResult:
        self.call_count += 1
        self.received_requests.append(request)
        current_task = request.get("current_task", {})
        inputs = (
            current_task.get("inputs", {})
            if isinstance(current_task, Mapping)
            else {}
        )
        if not isinstance(inputs, Mapping):
            inputs = {}
        if self.outcome is ResultStatus.REJECTED:
            return AgentResult(
                ResultStatus.REJECTED,
                "USRP 智能体拒绝本次模拟执行请求",
                {"reason": "fake USRP rejection"},
                error="USRP request rejected",
            )
        if self.outcome is ResultStatus.FAILED:
            return AgentResult(
                ResultStatus.FAILED,
                "USRP 模拟执行失败",
                error="fake USRP failure",
            )
        if self.outcome is ResultStatus.UNKNOWN:
            return AgentResult(ResultStatus.UNKNOWN, "USRP 执行状态未知")
        raw_defaulted_parameters = inputs.get("unresolved_parameters", [])
        defaulted_parameters = (
            raw_defaulted_parameters
            if isinstance(raw_defaulted_parameters, list)
            and all(isinstance(item, str) for item in raw_defaulted_parameters)
            else []
        )
        uses_defaults = inputs.get("parameter_policy") == "use_backend_defaults"
        summary = "USRP 模拟执行完成"
        if uses_defaults:
            defaults = "、".join(defaulted_parameters) or "后端配置"
            summary = f"USRP 模拟执行完成（Fake 默认补全：{defaults}）"
        return AgentResult(
            ResultStatus.SUCCEEDED,
            summary,
            {
                "operation_id": f"fake-operation-{self.call_count}",
                "execution_status": "succeeded",
                "actual_configuration": {
                    "source": (
                        "fake_defaults"
                        if uses_defaults
                        else "request"
                    ),
                    "defaulted_parameters": defaulted_parameters,
                },
            },
        )
