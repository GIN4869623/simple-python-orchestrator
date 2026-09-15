"""Small .env loader used by the command-line entry point."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None) -> Path | None:
    """Load KEY=VALUE pairs without overriding real environment variables."""

    candidates = [Path(path)] if path is not None else _default_env_candidates()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        for line_number, raw_line in enumerate(
            candidate.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line.removeprefix("export ").lstrip()
            if "=" not in line:
                raise ValueError(
                    f"{candidate}:{line_number} 不是有效的 KEY=VALUE 配置"
                )
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or not key.replace("_", "a").isalnum():
                raise ValueError(f"{candidate}:{line_number} 包含无效的配置名称")
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            os.environ.setdefault(key, value)
        return candidate
    return None


def _default_env_candidates() -> list[Path]:
    working_directory_env = Path.cwd() / ".env"
    project_env = Path(__file__).resolve().parents[2] / ".env"
    if working_directory_env == project_env:
        return [working_directory_env]
    return [working_directory_env, project_env]
