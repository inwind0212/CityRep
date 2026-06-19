from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "tasks.json"


def main() -> int:
    if not REGISTRY.is_file():
        print(f"missing registry: {REGISTRY}")
        return 1

    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    specs = payload.get("tasks", payload)
    if isinstance(specs, list):
        rows = specs
    else:
        rows = list(specs.values())

    missing: list[str] = []
    cities = set()
    tasks = set()
    labels_expected = 0

    for spec in rows:
        task_id = str(spec["task_id"])
        cities.add(str(spec.get("city") or task_id.split(".")[0]))
        tasks.add(str(spec.get("task") or task_id.split(".")[1]))
        for key in ["samples_path", "task_meta_path", "raster_path"]:
            value = spec.get(key)
            if value and not (REGISTRY.parent / str(value)).is_file():
                missing.append(f"{task_id}: {key} -> {value}")
        labels_path = spec.get("labels_path")
        if labels_path:
            labels_expected += 1
            if not (REGISTRY.parent / str(labels_path)).is_file():
                missing.append(f"{task_id}: labels_path -> {labels_path}")

    print(f"registry: {REGISTRY}")
    print(f"tasks: {len(rows)}")
    print(f"cities: {len(cities)} ({', '.join(sorted(cities))})")
    print(f"task names: {len(tasks)} ({', '.join(sorted(tasks))})")
    print(f"label rasters referenced: {labels_expected}")

    if missing:
        print("\nmissing files:")
        for item in missing[:50]:
            print(f"  {item}")
        if len(missing) > 50:
            print(f"  ... {len(missing) - 50} more")
        return 1

    print("status: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
