"""Deterministic test doubles for all external dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from importlib.resources import files
from typing import Any, Mapping

from .models import AgentResult, ResultStatus


@dataclass
class FakeKnowledgePort:
    should_fail: bool = False
    outcome: ResultStatus = ResultStatus.SUCCEEDED
    call_count: int = 0
    skill_read_failure: bool = False

    SKILL_ID = "usrp-operation-demo"
    SKILL_VERSION = "1.0"

    @staticmethod
    def _skill_content() -> str:
        return files("simple_orchestrator").joinpath(
            "skills/usrp-operation-demo.md"
        ).read_text(encoding="utf-8")

    def read_skill(self, skill_id: str, version: str) -> dict[str, str]:
        """Fake of the remote Knowledge skill-reading endpoint."""
        if self.skill_read_failure or skill_id != self.SKILL_ID or version != self.SKILL_VERSION:
            raise ValueError("可信 Skill 不可读取或版本不匹配")
        content = self._skill_content()
        return {
            "skill_id": skill_id,
            "version": version,
            "content": content,
            "sha256": sha256(content.encode("utf-8")).hexdigest(),
        }

    def invoke(
        self, request: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None,
    ) -> AgentResult:
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
        if (metadata or {}).get("capability") == "knowledge.lookup_skill":
            content = self._skill_content()
            return AgentResult(
                ResultStatus.SUCCEEDED,
                "已找到可信 USRP 操作 Skill",
                {
                    "skill_id": self.SKILL_ID,
                    "version": self.SKILL_VERSION,
                    "sha256": sha256(content.encode("utf-8")).hexdigest(),
                },
            )
        return AgentResult(
            ResultStatus.SUCCEEDED,
            "已取得模拟知识",
            {"knowledge": "QPSK 是一种四相位数字调制方式。"},
        )


@dataclass
class FakeDeviceManagerPort:
    available: bool = True
    device_id: str = "USRP-01"
    outcome: ResultStatus | None = None
    call_count: int = 0

    @property
    def device_capabilities(self) -> dict[str, tuple[str, ...]]:
        """Local inventory for preflight checks, separate from device responses."""
        return {
            self.device_id: (
                "usrp.configure", "usrp.transmit", "usrp.receive",
                "signal_transmit", "signal_receive",
            ),
        }

    def invoke(
        self, request: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None,
    ) -> AgentResult:
        self.call_count += 1
        if self.outcome is ResultStatus.REJECTED:
            return AgentResult(
                ResultStatus.REJECTED,
                "网络监管智能体拒绝本次模拟设备请求",
                {},
                error="device request rejected",
            )
        if self.outcome is ResultStatus.FAILED or not self.available:
            return AgentResult(
                ResultStatus.FAILED,
                "设备查询未取得可用设备",
                {"network_devices": []},
                error="no device available",
            )
        if self.outcome is ResultStatus.UNKNOWN:
            return AgentResult(
                ResultStatus.UNKNOWN,
                "无法从网络资源信息中确认可用设备",
                {},
            )
        device = {
            "device_id": self.device_id,
            "device_type": "usrp",
            "status": "online",
        }
        return AgentResult(
            ResultStatus.SUCCEEDED,
            "已获取模拟设备清单和状态",
            {"network_devices": [device]},
        )


@dataclass
class FakeUsrpAgent:
    outcome: ResultStatus = ResultStatus.SUCCEEDED
    call_count: int = 0
    physical_execution_count: int = 0
    received_requests: list[Mapping[str, Any]] = field(default_factory=list)
    skill_reader: Any = None

    def invoke(self, request: Mapping[str, Any]) -> AgentResult:
        """Simulate a read-only USRP knowledge answer."""
        self.call_count += 1
        self.received_requests.append(request)
        if self.outcome is ResultStatus.REJECTED:
            return AgentResult(ResultStatus.REJECTED, "USRP 知识查询被拒绝")
        if self.outcome is ResultStatus.FAILED:
            return AgentResult(ResultStatus.FAILED, "USRP 知识查询失败", error="fake usrp knowledge failure")
        if self.outcome is ResultStatus.UNKNOWN:
            return AgentResult(ResultStatus.UNKNOWN, "USRP 知识查询结果未知")
        return AgentResult(
            ResultStatus.SUCCEEDED,
            "已取得模拟 USRP 专属知识（未操作硬件）",
            {"knowledge": "USRP 相关操作应先确认设备能力和所需参数。"},
        )

    def execute(self, request: Mapping[str, Any]) -> AgentResult:
        self.call_count += 1
        self.received_requests.append(request)
        skill_ref = request.get("skill_ref")
        if skill_ref is not None:
            try:
                if not isinstance(skill_ref, Mapping) or self.skill_reader is None:
                    raise ValueError("缺少可信 Skill 读取接口")
                skill = self.skill_reader.read_skill(
                    skill_ref["skill_id"], skill_ref["version"]
                )
                if skill["sha256"] != skill_ref["sha256"]:
                    raise ValueError("可信 Skill 摘要校验失败")
                if sha256(skill["content"].encode("utf-8")).hexdigest() != skill["sha256"]:
                    raise ValueError("可信 Skill 内容校验失败")
            except (KeyError, TypeError, ValueError, OSError) as exc:
                return AgentResult(
                    ResultStatus.FAILED,
                    f"USRP 未执行：{exc}",
                    error="skill_validation_failed",
                )
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
        self.physical_execution_count += 1
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
