"""Machine-readable agent capabilities used for routing and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .models import IntentAnalysis


@dataclass(frozen=True)
class AgentProfile:
    agent_id: str
    name: str
    description: str
    capabilities: tuple[str, ...]
    supported_intents: tuple[str, ...]
    supported_targets: tuple[str, ...]
    side_effect_level: str
    required_inputs: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()
    output_contract: tuple[str, ...] = ()

    def as_prompt_dict(self) -> dict[str, object]:
        return asdict(self)


class AgentRegistry:
    def __init__(self, profiles: tuple[AgentProfile, ...]) -> None:
        self._profiles = {profile.agent_id: profile for profile in profiles}
        if len(self._profiles) != len(profiles):
            raise ValueError("智能体注册表中存在重复 agent_id")

    @property
    def agent_ids(self) -> set[str]:
        return set(self._profiles)

    def get(self, agent_id: str) -> AgentProfile:
        try:
            return self._profiles[agent_id]
        except KeyError as exc:
            raise KeyError(f"智能体未注册：{agent_id}") from exc

    def match(self, intent: IntentAnalysis) -> list[AgentProfile]:
        """Filter agents by declared intent, target, and requested capability."""

        requested = set(intent.required_capabilities)
        matches: list[AgentProfile] = []
        for profile in self._profiles.values():
            if intent.intent_type.value not in profile.supported_intents:
                continue
            if (
                intent.target_type
                and "*" not in profile.supported_targets
                and intent.target_type not in profile.supported_targets
            ):
                continue
            if requested and not requested.intersection(profile.capabilities):
                continue
            matches.append(profile)
        return matches


def build_default_registry() -> AgentRegistry:
    return AgentRegistry(
        (
            AgentProfile(
                agent_id="knowledge",
                name="知识智能体",
                description="提供通信知识、参数建议和操作指导，不接触实时设备。",
                capabilities=(
                    "knowledge.explain",
                    "knowledge.recommend_parameters",
                    "knowledge.create_procedure",
                ),
                supported_intents=("knowledge_query",),
                supported_targets=("*",),
                side_effect_level="none",
                exclusions=("实时设备查询", "设备选择", "真实硬件操作"),
                output_contract=("knowledge", "assumptions", "recommendations"),
            ),
            AgentProfile(
                agent_id="device_manager",
                name="网络监管智能体",
                description="只读查询网络中的设备清单、在线状态及实时传感器读数。",
                capabilities=(
                    "device.list",
                    "device.inspect",
                ),
                supported_intents=("device_query", "hardware_execution"),
                supported_targets=("*",),
                side_effect_level="none",
                exclusions=("法规审批", "通信原理回答", "设备选择或占用", "真实硬件操作"),
                output_contract=("network_devices", "answer"),
            ),
            AgentProfile(
                agent_id="ris", name="RIS 智能体",
                description="接收 RIS 自然语言任务并返回完整文字回复。",
                capabilities=("ris.chat", "knowledge.explain", "device.list", "device.inspect"),
                supported_intents=("knowledge_query", "device_query", "hardware_execution"),
                supported_targets=("ris",), side_effect_level="remote_agent",
                output_contract=("answer",),
            ),
            AgentProfile(
                agent_id="usrp",
                name="USRP 硬件智能体",
                description="回答 USRP 专属知识；仅在明确的硬件执行意图下配置、发射、接收或停止设备。",
                capabilities=(
                    "knowledge.explain",
                    "usrp.configure",
                    "usrp.transmit",
                    "usrp.receive",
                    "usrp.stop",
                    "usrp.get_status",
                ),
                supported_intents=("knowledge_query", "hardware_execution"),
                supported_targets=("usrp",),
                side_effect_level="physical_hardware",
                required_inputs=(),
                preconditions=(
                    "知识查询必须只读；物理操作必须由用户明确要求",
                    "物理操作前网络监管智能体已确认设备在线且能力匹配",
                ),
                exclusions=(
                    "软件仿真",
                    "RIS 操作",
                    "通用硬件操作",
                    "与 USRP 无关的通用通信原理问答",
                ),
                output_contract=(
                    "operation_id",
                    "device_id",
                    "actual_configuration",
                    "execution_status",
                ),
            ),
        )
    )
