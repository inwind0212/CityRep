from __future__ import annotations

from dataclasses import dataclass, field
import ast
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import xy as transform_xy
from shapely.geometry import Point

from .io import read_json, read_table
from .paths import DEFAULT_TASK_REGISTRY


@dataclass
class Task:
    task_id: str
    task_type: str
    samples: gpd.GeoDataFrame
    y: np.ndarray
    label_columns: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(len(self.samples))

    @property
    def output_dim(self) -> int:
        if self.task_type == "classification":
            return int(np.max(self.y)) + 1
        if self.y.ndim == 1:
            return 1
        return int(self.y.shape[1])


def load_task(task_id: str, registry_path: str | Path = DEFAULT_TASK_REGISTRY) -> Task:
    registry_file = Path(registry_path)
    base_dir = registry_file.resolve().parent
    by_id = load_task_specs(registry_file)
    if task_id not in by_id:
        raise KeyError(f"Task '{task_id}' not found in {registry_path}")
    return build_task(by_id[task_id], base_dir=base_dir)


def load_task_specs(registry_path: str | Path = DEFAULT_TASK_REGISTRY) -> dict[str, dict[str, Any]]:
    path = Path(registry_path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path).where(pd.notna, None)
        specs: dict[str, dict[str, Any]] = {}
        for record in df.to_dict(orient="records"):
            spec = {k: v for k, v in record.items() if v is not None and str(v) != ""}
            for key in ["label_cols", "bands", "class_names"]:
                if key in spec and isinstance(spec[key], str):
                    spec[key] = _parse_list_field(spec[key])
            for key in ["renormalize_distribution", "drop_zeros"]:
                if key in spec:
                    spec[key] = str(spec[key]).strip().lower() in {"1", "true", "yes", "y"}
            specs[str(spec["task_id"])] = spec
        return specs
    registry = read_json(path)
    specs = registry.get("tasks", registry)
    if isinstance(specs, list):
        return {str(spec["task_id"]): dict(spec) for spec in specs}
    return {str(k): dict(v) for k, v in specs.items()}


def _parse_list_field(value: str) -> list[Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def build_task(spec: dict[str, Any], base_dir: str | Path | None = None) -> Task:
    source_type = str(spec.get("source_type", "samples")).lower()
    base = Path(base_dir or ".")
    if source_type == "samples":
        return _task_from_samples(spec, base)
    if source_type == "raster":
        return _task_from_raster(spec, base)
    raise ValueError(f"Unsupported task source_type: {source_type}")


def _resolve(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def _task_from_samples(spec: dict[str, Any], base_dir: Path) -> Task:
    path = _resolve(base_dir, spec["samples_path"])
    assert path is not None
    df = read_table(path)
    task_type = str(spec["task_type"]).lower()
    sample_id_col = str(spec.get("sample_id_col", "sample_id"))
    if sample_id_col not in df.columns:
        raise KeyError(f"Missing sample id column '{sample_id_col}' in {path}")

    label_cols = spec.get("label_cols")
    if label_cols is None:
        label_col = spec.get("label_col", "label")
        label_cols = [label_col]
    label_cols = [str(c) for c in label_cols]
    missing = [c for c in label_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing label columns in {path}: {missing}")

    meta = dict(spec)
    if spec.get("labels_path") is not None:
        meta["labels_path"] = str(_resolve(base_dir, spec.get("labels_path")))
    if spec.get("task_meta_path") is not None:
        meta["task_meta_path"] = str(_resolve(base_dir, spec.get("task_meta_path")))
    if task_type == "classification":
        y = pd.to_numeric(df[label_cols[0]], errors="raise").to_numpy(dtype=np.int64)
    else:
        y = df[label_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
        if task_type == "regression" and y.shape[1] == 1:
            y_values, norm_meta = _normalize_regression(y.reshape(-1), str(spec.get("normalization", "zscore")))
            y = y_values.reshape(-1, 1)
            meta.update(norm_meta)
        if task_type == "distribution":
            row_sum = np.clip(y.sum(axis=1, keepdims=True), 1e-12, None)
            if bool(spec.get("renormalize_distribution", True)):
                y = y / row_sum

    crs = spec.get("crs", "EPSG:4326")
    if "geometry" in df.columns:
        geo = gpd.GeoDataFrame(df.copy(), geometry=gpd.GeoSeries.from_wkt(df["geometry"]), crs=crs)
    elif {"x", "y"}.issubset(df.columns):
        geo = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df["x"], df["y"]), crs=crs)
    else:
        raise KeyError("Sample table must include either WKT 'geometry' or numeric 'x'/'y' columns.")
    geo = geo.rename(columns={sample_id_col: "sample_id"})
    geo["sample_id"] = geo["sample_id"].astype(str)
    return Task(
        task_id=str(spec["task_id"]),
        task_type=task_type,
        samples=geo.reset_index(drop=True),
        y=y,
        label_columns=label_cols,
        meta={**meta, "samples_path": str(path)},
    )


def _normalize_regression(values: np.ndarray, mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    values = values.astype(np.float32, copy=False)
    if mode == "none":
        return values, {"normalization": mode}
    if mode == "zscore":
        mean = float(values.mean())
        std = float(values.std() + 1e-6)
        return ((values - mean) / std).astype(np.float32), {"normalization": mode, "y_mean": mean, "y_std": std}
    if mode == "log1p_zscore":
        logv = np.log1p(values)
        mean = float(logv.mean())
        std = float(logv.std() + 1e-6)
        return ((logv - mean) / std).astype(np.float32), {"normalization": mode, "y_mean": mean, "y_std": std}
    raise ValueError(f"Unsupported normalization: {mode}")


def _task_from_raster(spec: dict[str, Any], base_dir: Path) -> Task:
    path = _resolve(base_dir, spec["raster_path"])
    assert path is not None
    task_type = str(spec.get("task_type", "regression")).lower()
    if task_type not in {"regression", "distribution"}:
        raise ValueError("Raster source supports regression and distribution tasks.")
    boundary_path = _resolve(base_dir, spec.get("boundary_path"))
    sample_stride = int(spec.get("sample_stride", 1))
    min_value = spec.get("min_value", 0.0)
    drop_zeros = bool(spec.get("drop_zeros", False))

    with rasterio.open(path) as src:
        nodata = src.nodata
        if task_type == "regression":
            arr = src.read(int(spec.get("band", 1)))
            mask = np.isfinite(arr)
            if nodata is not None:
                mask &= arr != nodata
            if min_value is not None:
                mask &= arr >= float(min_value)
            if drop_zeros:
                mask &= arr != 0
        else:
            band_ids = spec.get("bands") or list(range(1, src.count + 1))
            arr = src.read([int(b) for b in band_ids]).astype(np.float32, copy=False)
            mask = np.all(np.isfinite(arr), axis=0)
            if nodata is not None and np.isfinite(nodata):
                mask &= np.all(arr != nodata, axis=0)
            sums = arr.sum(axis=0)
            mask &= np.isfinite(sums) & (sums > 0)
            tolerance = spec.get("distribution_sum_tolerance")
            if tolerance is not None:
                mask &= np.abs(sums - 1.0) <= float(tolerance)

        if boundary_path is not None:
            boundary = gpd.read_file(boundary_path).to_crs(src.crs)
            inside = ~geometry_mask([boundary.geometry.union_all()], transform=src.transform, invert=False, out_shape=mask.shape)
            mask &= inside

        rows, cols = np.where(mask)
        if sample_stride > 1:
            keep = (rows % sample_stride == 0) & (cols % sample_stride == 0)
            rows, cols = rows[keep], cols[keep]

        if task_type == "regression":
            values = arr[rows, cols].astype(np.float32)
            y_values, norm_meta = _normalize_regression(values, str(spec.get("normalization", "zscore")))
            y = y_values.reshape(-1, 1)
            label_cols = ["label"]
        else:
            values = arr[:, rows, cols].T.astype(np.float32, copy=False)
            if bool(spec.get("renormalize_distribution", True)):
                values = values / np.clip(values.sum(axis=1, keepdims=True), 1e-12, None)
            y = values
            norm_meta = {"normalization": "none"}
            label_cols = [f"label_{i}" for i in range(y.shape[1])]

        xs, ys = transform_xy(src.transform, rows, cols, offset="center")
        sample_ids = [f"{spec['task_id']}:r{int(r)}:c{int(c)}" for r, c in zip(rows, cols)]
        data = {"sample_id": sample_ids, "row": rows, "col": cols, "x": xs, "y": ys}
        for i, col in enumerate(label_cols):
            data[col] = y[:, i] if y.ndim == 2 else y
        geo = gpd.GeoDataFrame(data, geometry=[Point(x, y0) for x, y0 in zip(xs, ys)], crs=src.crs)

    meta = {**spec, **norm_meta, "raster_path": str(path)}
    return Task(str(spec["task_id"]), task_type, geo.reset_index(drop=True), y, label_cols, meta)
