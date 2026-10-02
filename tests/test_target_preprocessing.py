from __future__ import annotations

import numpy as np

from urban_benchmark.evaluation import _prepare_target_for_split
from urban_benchmark.splits import Split
from urban_benchmark.tasks import Task, _prepare_regression_target


def _split() -> Split:
    return Split(
        seed=42,
        train_idx=np.asarray([0, 1, 2], dtype=np.int64),
        val_idx=np.asarray([3], dtype=np.int64),
        test_idx=np.asarray([4], dtype=np.int64),
        meta={},
    )


def _task(values: np.ndarray, normalization: str = "zscore") -> Task:
    return Task(
        task_id="demo.regression.2026",
        task_type="regression",
        samples=None,  # The split-specific target helper does not access geometry.
        y=np.asarray(values, dtype=np.float32).reshape(-1, 1),
        label_columns=["label"],
        meta={"normalization": normalization, "target_transform": "identity"},
    )


def test_zscore_is_applied_during_evaluation() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    prepared, meta = _prepare_regression_target(values, "zscore")
    np.testing.assert_array_equal(prepared, values)
    assert meta == {"normalization": "zscore", "target_transform": "identity"}


def test_log1p_is_applied_before_split_specific_zscore() -> None:
    values = np.asarray([0.0, 1.0, 3.0], dtype=np.float32)
    prepared, meta = _prepare_regression_target(values, "log1p_zscore")
    np.testing.assert_allclose(prepared, np.log1p(values))
    assert meta == {"normalization": "log1p_zscore", "target_transform": "log1p"}


def test_zscore_uses_training_partition_statistics() -> None:
    base = _task(np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]))
    changed_test = _task(np.asarray([1.0, 2.0, 3.0, 4.0, 5_000_000.0]))

    base_y, base_meta = _prepare_target_for_split(base, _split())
    changed_y, changed_meta = _prepare_target_for_split(changed_test, _split())

    np.testing.assert_allclose(base_y[:4], changed_y[:4])
    assert base_meta == changed_meta
    assert base_meta["fitted_on"] == "train"
    assert base_meta["train_size"] == 3
    np.testing.assert_allclose(base_y[_split().train_idx].mean(), 0.0, atol=2e-6)
    np.testing.assert_allclose(base_y[_split().train_idx].std(), 1.0, atol=2e-6)


def test_none_mode_preserves_regression_targets() -> None:
    task = _task(np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]), normalization="none")
    prepared, meta = _prepare_target_for_split(task, _split())
    np.testing.assert_array_equal(prepared, task.y)
    assert meta == {"normalization": "none", "fitted_on": "none"}
