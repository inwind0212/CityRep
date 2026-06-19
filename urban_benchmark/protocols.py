from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json
from .paths import DEFAULT_PROTOCOL_REGISTRY


def load_protocol(protocol_id: str, registry_path: str | Path = DEFAULT_PROTOCOL_REGISTRY) -> dict[str, Any]:
    registry = read_json(registry_path)
    protocols = registry.get("protocols", registry)
    if isinstance(protocols, list):
        by_id = {p["protocol_id"]: p for p in protocols}
    else:
        by_id = protocols
    if protocol_id not in by_id:
        raise KeyError(f"Protocol '{protocol_id}' not found in {registry_path}")
    protocol = dict(by_id[protocol_id])
    protocol.setdefault("protocol_id", protocol_id)
    return protocol
