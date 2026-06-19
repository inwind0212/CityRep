from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from .tasks import Task


@dataclass
class Split:
    seed: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    meta: dict


def make_splits(task: Task, split_cfg: dict) -> list[Split]:
    seeds = [int(s) for s in split_cfg.get("seeds", [42])]
    method = str(split_cfg.get("method", "spatial_block")).lower()
    return [_make_split(task, split_cfg, seed, method) for seed in seeds]


def _make_split(task: Task, cfg: dict, seed: int, method: str) -> Split:
    test_ratio = float(cfg.get("test_ratio", 0.2))
    val_ratio = float(cfg.get("val_ratio_on_train", 0.1))
    if method in {"random", "random_sample"}:
        return _random_split(task, seed, test_ratio, val_ratio, bool(cfg.get("stratify_classification", True)))
    if method in {"spatial_block", "block", "blocked"}:
        return _block_split(task, seed, test_ratio, val_ratio, int(cfg.get("n_blocks_x", cfg.get("n_blocks", 10))), int(cfg.get("n_blocks_y", cfg.get("n_blocks", 10))))
    raise ValueError(f"Unsupported split method: {method}")


def _random_split(task: Task, seed: int, test_ratio: float, val_ratio: float, stratify_classification: bool) -> Split:
    idx = np.arange(task.n_samples)
    stratify = task.y.astype(np.int64) if task.task_type == "classification" and stratify_classification else None
    idx_train, idx_test = train_test_split(idx, test_size=test_ratio, random_state=seed, stratify=stratify)
    stratify_train = stratify[idx_train] if stratify is not None else None
    if val_ratio > 0:
        idx_train, idx_val = train_test_split(idx_train, test_size=val_ratio, random_state=seed, stratify=stratify_train)
    else:
        idx_val = np.empty((0,), dtype=np.int64)
    return Split(seed, idx_train, idx_val, idx_test, {"method": "random", "test_ratio": test_ratio, "val_ratio_on_train": val_ratio})


def _block_ids(task: Task, n_blocks_x: int, n_blocks_y: int) -> np.ndarray:
    if {"row", "col"}.issubset(task.samples.columns):
        x = task.samples["col"].to_numpy(dtype=np.float64)
        y = task.samples["row"].to_numpy(dtype=np.float64)
    else:
        x = task.samples.geometry.x.to_numpy(dtype=np.float64)
        y = task.samples.geometry.y.to_numpy(dtype=np.float64)
    x_span = max(float(np.nanmax(x) - np.nanmin(x)), 1e-12)
    y_span = max(float(np.nanmax(y) - np.nanmin(y)), 1e-12)
    bx = np.floor((x - float(np.nanmin(x))) / x_span * n_blocks_x).astype(np.int64)
    by = np.floor((y - float(np.nanmin(y))) / y_span * n_blocks_y).astype(np.int64)
    bx = np.clip(bx, 0, n_blocks_x - 1)
    by = np.clip(by, 0, n_blocks_y - 1)
    return by * n_blocks_x + bx


def _block_split(task: Task, seed: int, test_ratio: float, val_ratio: float, n_blocks_x: int, n_blocks_y: int) -> Split:
    idx = np.arange(task.n_samples)
    block_ids = _block_ids(task, n_blocks_x, n_blocks_y)
    blocks = np.unique(block_ids)
    if len(blocks) < 3:
        raise ValueError(f"Need at least 3 non-empty spatial blocks, got {len(blocks)}")
    train_blocks, test_blocks = train_test_split(blocks, test_size=test_ratio, random_state=seed)
    if val_ratio > 0:
        train_blocks, val_blocks = train_test_split(train_blocks, test_size=val_ratio, random_state=seed)
    else:
        val_blocks = np.empty((0,), dtype=blocks.dtype)
    train_idx = idx[np.isin(block_ids, train_blocks)]
    val_idx = idx[np.isin(block_ids, val_blocks)]
    test_idx = idx[np.isin(block_ids, test_blocks)]
    return Split(
        seed,
        train_idx,
        val_idx,
        test_idx,
        {
            "method": "spatial_block",
            "n_blocks_x": n_blocks_x,
            "n_blocks_y": n_blocks_y,
            "test_ratio": test_ratio,
            "val_ratio_on_train": val_ratio,
            "train_blocks": sorted(int(x) for x in train_blocks),
            "val_blocks": sorted(int(x) for x in val_blocks),
            "test_blocks": sorted(int(x) for x in test_blocks),
        },
    )
