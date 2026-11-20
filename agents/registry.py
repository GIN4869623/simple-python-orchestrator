#!/usr/bin/env python3
"""Manage the standalone Agent capability repository."""

import argparse
import fcntl
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit


DEFAULT_PATH = Path(__file__).resolve().with_name("registry.json")
AGENT_ID = re.compile(r"[a-z][a-z0-9_]*")
CAPABILITY_ID = re.compile(r"[a-z][a-z0-9_.]*")
PROFILE_FIELDS = {"agent_id", "name", "description", "capabilities"}
CAPABILITY_FIELDS = {"id", "description"}
INTERFACE_PROTOCOLS = {"http", "sse", "tcp"}
HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}


class RegistryConflict(ValueError):
    pass


class AgentNotFound(ValueError):
    pass


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 中存在重复键：{key}")
        result[key] = value
    return result


def read_json(path):
    with path.open(encoding="utf-8") as source:
        return json.load(source, object_pairs_hook=unique_object)


def check_fields(value, required, optional, where):
    if not isinstance(value, dict):
        raise ValueError(f"{where} 必须是对象")
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise ValueError(f"{where} 缺少字段：{', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{where} 含有未知字段：{', '.join(sorted(unknown))}")


def check_text(value, where):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} 必须是非空字符串")


def check_text_list(value, where):
    if not isinstance(value, list):
        raise ValueError(f"{where} 必须是字符串列表")
    for index, item in enumerate(value):
        check_text(item, f"{where}[{index}]")
    if len(set(value)) != len(value):
        raise ValueError(f"{where} 含有重复值")


def validate_interfaces(interfaces, where):
    if not isinstance(interfaces, list):
        raise ValueError(f"{where} 必须是列表")
    for index, interface in enumerate(interfaces):
        item = f"{where}[{index}]"
        check_fields(interface, {"protocol", "url", "description"}, {"method", "verification"}, item)
        check_text(interface["description"], f"{item}.description")
        protocol = interface["protocol"]
        if not isinstance(protocol, str) or protocol not in INTERFACE_PROTOCOLS:
            raise ValueError(f"{item}.protocol 必须为 http、sse 或 tcp")
        if "verification" in interface and interface["verification"] not in ("configured", "verified"):
            raise ValueError(f"{item}.verification 必须为 configured 或 verified")
        url = interface["url"]
        check_text(url, f"{item}.url")
        try:
            address = urlsplit(url)
            port = address.port
        except ValueError as error:
            raise ValueError(f"{item}.url 的主机或端口无效") from error
        if (not url.startswith(f"{address.scheme}://") or not address.hostname
                or address.username is not None or address.password is not None
                or "#" in url or any(character.isspace() or ord(character) < 32 for character in url)
                or port == 0):
            raise ValueError(f"{item}.url 必须是有效的服务地址，且不能包含账号、密码或片段")
        if protocol == "tcp":
            if address.scheme != "tcp" or port is None or address.path or "?" in url or "method" in interface:
                raise ValueError(f"{item} 的 TCP 接口需要 tcp://主机:端口")
        else:
            method = interface.get("method")
            if address.scheme not in {"http", "https"} or not isinstance(method, str) or method not in HTTP_METHODS:
                raise ValueError(f"{item} 的 HTTP 或 SSE 接口需要 HTTP(S) 地址与请求方法")


def validate_profile(agent_id, profile):
    where = f"Agent {agent_id}"
    if not AGENT_ID.fullmatch(agent_id):
        raise ValueError(f"Agent ID 格式无效：{agent_id}")
    check_fields(profile, PROFILE_FIELDS, {"constraints", "interfaces"}, where)
    if profile["agent_id"] != agent_id:
        raise ValueError(f"{where} 的记录 ID 与目标 ID 不一致：{profile['agent_id']!r}")
    check_text(profile["name"], f"{where}.name")
    check_text(profile["description"], f"{where}.description")
    if "constraints" in profile:
        check_text_list(profile["constraints"], f"{where}.constraints")
    if "interfaces" in profile:
        validate_interfaces(profile["interfaces"], f"{where}.interfaces")

    capabilities = profile["capabilities"]
    if not isinstance(capabilities, list) or not capabilities:
        raise ValueError(f"{where}.capabilities 必须是非空列表")
    seen = set()
    for index, capability in enumerate(capabilities):
        item = f"{where}.capabilities[{index}]"
        check_fields(capability, CAPABILITY_FIELDS, {"required_inputs", "constraints"}, item)
        capability_id = capability["id"]
        if not isinstance(capability_id, str) or not CAPABILITY_ID.fullmatch(capability_id):
            raise ValueError(f"{item}.id 格式无效")
        if capability_id in seen:
            raise ValueError(f"{where} 含有重复能力 ID：{capability_id}")
        seen.add(capability_id)
        check_text(capability["description"], f"{item}.description")
        for field in ("required_inputs", "constraints"):
            if field in capability:
                check_text_list(capability[field], f"{item}.{field}")


def load_registry(path):
    registry = read_json(path)
    check_fields(registry, {"schema_version", "agents"}, set(), "仓库")
    version = registry["schema_version"]
    if type(version) is not int or version != 1:
        raise ValueError("仓库 schema_version 必须为 1")
    agents = registry["agents"]
    if not isinstance(agents, dict):
        raise ValueError("agents 必须是以 Agent ID 为键的对象")
    for agent_id, record in agents.items():
        check_fields(record, {"revision", "profile"}, set(), f"Agent {agent_id} 的记录")
        revision = record["revision"]
        if type(revision) is not int or revision < 1:
            raise ValueError(f"Agent {agent_id} 的版本号必须是正整数")
        validate_profile(agent_id, record["profile"])
    return registry


def write_registry(path, registry):
    # Readers see one complete snapshot while the writer prepares the next one.
    mode = stat.S_IMODE(path.stat().st_mode)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as target:
            temp_path = Path(target.name)
            os.fchmod(target.fileno(), mode)
            json.dump(registry, target, ensure_ascii=False, allow_nan=False, indent=2)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def save_agent(path, agent_id, profile, expected_revision=None):
    validate_profile(agent_id, profile)
    if expected_revision is not None and (type(expected_revision) is not int or expected_revision < 1):
        raise ValueError("预期版本号必须是正整数")
    # The lock keeps the same inode when the JSON snapshot is replaced.
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        registry = load_registry(path)
        agents = registry["agents"]
        current = agents.get(agent_id)
        if expected_revision is None:
            if current is not None:
                raise RegistryConflict(f"Agent 已存在：{agent_id}")
            revision = 1
        else:
            if current is None:
                raise AgentNotFound(f"找不到 Agent：{agent_id}")
            if current["revision"] != expected_revision:
                raise RegistryConflict(f"Agent {agent_id} 版本冲突，当前是第 {current['revision']} 版")
            revision = expected_revision + 1
        agents[agent_id] = {"revision": revision, "profile": profile}
        write_registry(path, registry)
    return {"agent_id": agent_id, "revision": revision}


def summarize_agents(agents):
    return [
        {"agent_id": agent_id, "description": agents[agent_id]["profile"]["description"]}
        for agent_id in sorted(agents)
    ]


def main():
    parser = argparse.ArgumentParser(description="管理独立的 Agent 能力仓库")
    parser.add_argument("--file", type=Path, default=DEFAULT_PATH, help="指定独立仓库文件")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="校验整个仓库")
    commands.add_parser("summaries", help="读取 Agent ID 和简述")
    details = commands.add_parser("details", help="按候选 Agent ID 读取完整记录")
    details.add_argument("agent_ids", metavar="AGENT_ID", nargs="+")
    for command, help_text in (("add", "新增一个 Agent"), ("update", "更新一个 Agent")):
        writer = commands.add_parser(command, help=help_text)
        writer.add_argument("agent_id", metavar="AGENT_ID")
        writer.add_argument("profile", metavar="PROFILE_JSON", type=Path)
        if command == "update":
            writer.add_argument("--expect-revision", required=True, type=int, help="编辑前看到的记录版本")
    args = parser.parse_args()

    try:
        path = args.file.resolve()
        if args.command in {"add", "update"}:
            output = save_agent(
                path, args.agent_id, read_json(args.profile), getattr(args, "expect_revision", None)
            )
        else:
            agents = load_registry(path)["agents"]
            if args.command == "validate":
                output = {"agent_count": len(agents)}
            elif args.command == "summaries":
                output = summarize_agents(agents)
            else:
                missing = [agent_id for agent_id in args.agent_ids if agent_id not in agents]
                if missing:
                    raise ValueError(f"找不到候选 Agent：{', '.join(missing)}")
                output = {agent_id: agents[agent_id] for agent_id in args.agent_ids}
        print(json.dumps(output, ensure_ascii=False, indent=2))
    except (OSError, ValueError) as error:
        print(f"仓库操作失败：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
