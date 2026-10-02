from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_RELEASE = PACKAGE_ROOT / "data"
DEFAULT_TASK_REGISTRY = DEFAULT_DATA_RELEASE / "tasks.json"
DEFAULT_PROTOCOL_REGISTRY = PACKAGE_ROOT / "configs" / "release" / "protocols.json"
