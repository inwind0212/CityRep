from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import Resampling, reproject
from pyproj import Transformer
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from .embeddings import PointEmbedding, RasterEmbedding, RegionEmbedding
from .io import write_json
from .tasks import Task


@dataclass
class AlignedEmbedding:
    X: np.ndarray
    sample_ids: list[str]
    report: dict[str, Any]

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "X.npy", self.X.astype(np.float32, copy=False))
        pd.DataFrame({"sample_id": self.sample_ids}).to_parquet(out / "sample_index.parquet", index=False)
        write_json(out / "alignment_report.json", self.report)


def align_embedding(
    task: Task,
    embedding: RasterEmbedding | RegionEmbedding | PointEmbedding,
    *,
    method: str = "auto",
    normalize: bool = True,
    fill_missing: dict[str, Any] | None = None,
) -> AlignedEmbedding:
    if isinstance(embedding, RasterEmbedding):
        return _align_raster(task, embedding, method=method, normalize=normalize, fill_missing=fill_missing)
    if isinstance(embedding, RegionEmbedding):
        return _align_regions(task, embedding, method=method, normalize=normalize, fill_missing=fill_missing)
    if isinstance(embedding, PointEmbedding):
        return _align_points(task, embedding, method=method, normalize=normalize, fill_missing=fill_missing)
    raise TypeError(f"Unsupported embedding source: {type(embedding)!r}")


def _l2_normalize(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.clip(norm, eps, None)).astype(np.float32)


def _grid_matches(task: Task, src: rasterio.DatasetReader) -> bool:
    label_path = _label_raster_path(task)
    if label_path is None:
        return False
    try:
        with rasterio.open(label_path) as label:
            return _same_grid(label, src)
    except Exception:
        return False


def _label_raster_path(task: Task) -> str | None:
    meta = task.meta
    label_path = meta.get("raster_path") or meta.get("labels_path")
    return None if label_path is None else str(label_path)


def _same_grid(a: rasterio.DatasetReader, b: rasterio.DatasetReader) -> bool:
    return a.width == b.width and a.height == b.height and a.crs == b.crs and tuple(a.transform) == tuple(b.transform)


def _pixel_area(dataset: rasterio.DatasetReader) -> float:
    transform = dataset.transform
    return abs(transform.a * transform.e - transform.b * transform.d)


def _can_average_to_label_grid(task: Task, src: rasterio.DatasetReader) -> bool:
    if task.task_type not in {"regression", "distribution"}:
        return False
    if not {"row", "col"}.issubset(task.samples.columns):
        return False
    label_path = _label_raster_path(task)
    if label_path is None or src.crs is None:
        return False
    try:
        with rasterio.open(label_path) as label:
            return label.crs is not None and not _same_grid(label, src)
    except Exception:
        return False


def _align_raster_cell(task: Task, src: rasterio.DatasetReader) -> np.ndarray:
    rows = task.samples["row"].to_numpy(dtype=np.int64)
    cols = task.samples["col"].to_numpy(dtype=np.int64)
    if rows.size and (rows.min() < 0 or rows.max() >= src.height or cols.min() < 0 or cols.max() >= src.width):
        raise ValueError("Task row/col exceed embedding raster dimensions.")
    arr = src.read().astype(np.float32, copy=False)
    return arr[:, rows, cols].T


def _align_raster_area_average(task: Task, src: rasterio.DatasetReader, label_path: str) -> np.ndarray:
    rows = task.samples["row"].to_numpy(dtype=np.int64)
    cols = task.samples["col"].to_numpy(dtype=np.int64)
    with rasterio.open(label_path) as label:
        if label.crs is None or src.crs is None:
            raise ValueError("CRS is required for raster area-average alignment.")
        if rows.size and (rows.min() < 0 or rows.max() >= label.height or cols.min() < 0 or cols.max() >= label.width):
            raise ValueError("Task row/col exceed label raster dimensions.")

        X = np.empty((len(rows), src.count), dtype=np.float32)
        band_iter = tqdm(
            range(1, src.count + 1),
            desc=f"area-average {Path(src.name).stem} -> {Path(label_path).parent.name}",
            unit="band",
            leave=False,
            dynamic_ncols=True,
        )
        for band_idx in band_iter:
            target = np.full((label.height, label.width), np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, band_idx),
                destination=target,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=label.transform,
                dst_crs=label.crs,
                dst_nodata=np.nan,
                resampling=Resampling.average,
            )
            X[:, band_idx - 1] = target[rows, cols]
    return X


def _sample_coordinates(task: Task) -> np.ndarray:
    if {"row", "col"}.issubset(task.samples.columns):
        return task.samples[["row", "col"]].to_numpy(dtype=np.float64)
    if task.samples.geometry is None:
        raise ValueError("Task samples need row/col or geometry for spatial missing-value fill.")
    return np.column_stack(
        [
            task.samples.geometry.x.to_numpy(dtype=np.float64),
            task.samples.geometry.y.to_numpy(dtype=np.float64),
        ]
    )


def _spatial_fill_missing(
    X: np.ndarray,
    missing: np.ndarray,
    coords: np.ndarray,
    fill_missing: dict[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    cfg = dict(fill_missing or {})
    method = str(cfg.get("method", "none")).lower()
    stats: dict[str, Any] = {
        "method": method,
        "n_missing_before_fill": int(missing.sum()),
        "n_valid_before_fill": int((~missing).sum()),
        "n_filled": 0,
        "n_missing_after_fill": int(missing.sum()),
    }
    if method in {"none", "zero", ""} or not missing.any():
        return X, stats
    if method not in {"nearest", "idw"}:
        raise ValueError(f"Unsupported fill_missing method: {method!r}")

    valid = ~missing
    if not valid.any():
        return X, stats

    valid_coords = coords[valid]
    missing_coords = coords[missing]
    tree = cKDTree(valid_coords)
    max_distance = cfg.get("max_distance")
    distance_upper_bound = float(max_distance) if max_distance is not None else np.inf

    out = X.copy()
    missing_indices = np.flatnonzero(missing)
    if method == "nearest":
        dist, idx = tree.query(missing_coords, k=1, distance_upper_bound=distance_upper_bound)
        can_fill = np.isfinite(dist) & (idx < valid_coords.shape[0])
        source_indices = np.flatnonzero(valid)[idx[can_fill]]
        out[missing_indices[can_fill]] = X[source_indices]
    else:
        k = min(int(cfg.get("k", 4)), valid_coords.shape[0])
        power = float(cfg.get("power", 2.0))
        eps = float(cfg.get("eps", 1e-12))
        dist, idx = tree.query(missing_coords, k=k, distance_upper_bound=distance_upper_bound)
        if k == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        usable = np.isfinite(dist) & (idx < valid_coords.shape[0])
        weights = np.where(usable, 1.0 / np.power(dist + eps, power), 0.0)
        weight_sum = weights.sum(axis=1)
        can_fill = weight_sum > 0
        valid_rows = np.flatnonzero(valid)
        gathered = X[valid_rows[np.clip(idx[can_fill], 0, len(valid_rows) - 1)]]
        out[missing_indices[can_fill]] = (gathered * weights[can_fill, :, None]).sum(axis=1) / weight_sum[can_fill, None]

    filled = np.linalg.norm(out[missing_indices], axis=1) > 1e-8
    stats["n_filled"] = int(filled.sum())
    stats["n_missing_after_fill"] = int(stats["n_missing_before_fill"] - stats["n_filled"])
    if method == "idw":
        stats["k"] = int(cfg.get("k", 4))
        stats["power"] = float(cfg.get("power", 2.0))
    if max_distance is not None:
        stats["max_distance"] = float(max_distance)
    return out, stats


def _align_raster(
    task: Task,
    embedding: RasterEmbedding,
    *,
    method: str,
    normalize: bool,
    fill_missing: dict[str, Any] | None,
) -> AlignedEmbedding:
    sample_ids = task.samples["sample_id"].astype(str).tolist()
    with rasterio.open(embedding.path) as src:
        label_path = _label_raster_path(task)
        use_cell = method == "raster_cell" or (
            method == "auto" and {"row", "col"}.issubset(task.samples.columns) and _grid_matches(task, src)
        )
        use_average = method in {"raster_area_average", "raster_average", "area_average"} or (
            method == "auto" and _can_average_to_label_grid(task, src)
        )
        if use_cell:
            X = _align_raster_cell(task, src)
            aligner = "raster_cell"
        elif use_average:
            if label_path is None:
                raise ValueError("A label raster is required for raster area-average alignment.")
            X = _align_raster_area_average(task, src, label_path)
            aligner = "raster_area_average"
        else:
            if task.samples.crs is None or src.crs is None:
                raise ValueError("CRS is required for coordinate raster sampling.")
            transformer = Transformer.from_crs(task.samples.crs, src.crs, always_xy=True)
            xs = task.samples.geometry.x.to_numpy(dtype=np.float64)
            ys = task.samples.geometry.y.to_numpy(dtype=np.float64)
            x2, y2 = transformer.transform(xs, ys)
            sample_xy = np.column_stack([x2, y2]).astype(np.float64, copy=False)
            rows, cols = rowcol(src.transform, sample_xy[:, 0], sample_xy[:, 1], op=np.floor)
            rows = np.asarray(rows, dtype=np.int64)
            cols = np.asarray(cols, dtype=np.int64)
            block_h, block_w = src.block_shapes[0] if src.block_shapes else (1, 1)
            order = np.lexsort((cols // block_w, rows // block_h))
            sampled = np.vstack([v for v in src.sample(sample_xy[order])]).astype(np.float32, copy=False)
            X = np.empty_like(sampled)
            X[order] = sampled
            aligner = "raster_sample"
    X = X.astype(np.float32, copy=False)
    missing_before_fill = ~np.isfinite(X).all(axis=1) | (np.linalg.norm(np.nan_to_num(X, nan=0.0), axis=1) <= 1e-8)
    X = np.nan_to_num(X, nan=0.0).astype(np.float32)
    X, fill_report = _spatial_fill_missing(X, missing_before_fill, _sample_coordinates(task), fill_missing)
    if normalize:
        X = _l2_normalize(X)
    missing_rate = float(np.mean(np.linalg.norm(X, axis=1) <= 1e-8)) if len(X) else 0.0
    report = {
        "task_id": task.task_id,
        "embedding": embedding.metadata(),
        "aligner": aligner,
        "label_raster_path": _label_raster_path(task),
        "n_task_samples": task.n_samples,
        "n_aligned_samples": int(X.shape[0]),
        "embedding_dim": int(X.shape[1]),
        "missing_rate": missing_rate,
        "missing_rate_before_fill": float(np.mean(missing_before_fill)) if len(X) else 0.0,
        "fill_missing": fill_report,
        "l2_normalized": bool(normalize),
    }
    return AlignedEmbedding(X=X, sample_ids=sample_ids, report=report)


def _h3_latlng_to_cell(lat: float, lon: float, res: int) -> str:
    import h3

    if hasattr(h3, "latlng_to_cell"):
        return str(h3.latlng_to_cell(lat, lon, res))
    return str(h3.geo_to_h3(lat, lon, res))


def _align_regions(
    task: Task,
    embedding: RegionEmbedding,
    *,
    method: str,
    normalize: bool,
    fill_missing: dict[str, Any] | None,
) -> AlignedEmbedding:
    del method
    table = embedding.table()
    sample_ids = task.samples["sample_id"].astype(str).tolist()
    if embedding.task_region_id_col and embedding.task_region_id_col in task.samples.columns:
        ids = task.samples[embedding.task_region_id_col].astype(str).to_numpy()
        aligner = "region_lookup"
    elif embedding.region_type.lower() == "h3":
        if embedding.h3_resolution is None:
            raise ValueError("h3 region embeddings require h3_resolution.")
        if task.samples.crs is None:
            raise ValueError("Task CRS is required for H3 lookup.")
        transformer = Transformer.from_crs(task.samples.crs, "EPSG:4326", always_xy=True)
        xs = task.samples.geometry.x.to_numpy(dtype=np.float64)
        ys = task.samples.geometry.y.to_numpy(dtype=np.float64)
        lon, lat = transformer.transform(xs, ys)
        ids = np.asarray([_h3_latlng_to_cell(float(la), float(lo), int(embedding.h3_resolution)) for lo, la in zip(lon, lat)])
        aligner = "h3_centroid_lookup"
    else:
        raise ValueError("Region embeddings require task_region_id_col, or region_type='h3' with h3_resolution.")

    X = np.zeros((len(ids), table.shape[1]), dtype=np.float32)
    index = pd.Index(table.index.astype(str))
    loc = index.get_indexer(ids)
    hit = loc >= 0
    if hit.any():
        X[hit] = table.to_numpy(dtype=np.float32, copy=False)[loc[hit]]
    missing_before_fill = ~hit | (np.linalg.norm(np.nan_to_num(X, nan=0.0), axis=1) <= 1e-8)
    X = np.nan_to_num(X, nan=0.0).astype(np.float32)
    X, fill_report = _spatial_fill_missing(X, missing_before_fill, _sample_coordinates(task), fill_missing)
    if normalize:
        X = _l2_normalize(X)
    missing_rate = float(np.mean(np.linalg.norm(X, axis=1) <= 1e-8)) if len(X) else 0.0
    report = {
        "task_id": task.task_id,
        "embedding": embedding.metadata(),
        "aligner": aligner,
        "n_task_samples": task.n_samples,
        "n_aligned_samples": int(X.shape[0]),
        "embedding_dim": int(X.shape[1]),
        "hit_rate": float(1.0 - missing_rate),
        "hit_rate_before_fill": float(hit.mean()) if len(hit) else 0.0,
        "missing_rate": missing_rate,
        "missing_rate_before_fill": float(np.mean(missing_before_fill)) if len(X) else 0.0,
        "fill_missing": fill_report,
        "l2_normalized": bool(normalize),
    }
    return AlignedEmbedding(X=X, sample_ids=sample_ids, report=report)


def _embedding_columns_for_point_table(table: pd.DataFrame, embedding: PointEmbedding) -> list[str]:
    if embedding.embedding_cols is not None:
        return list(embedding.embedding_cols)
    reserved = {embedding.x_col, embedding.y_col}
    if embedding.point_id_col is not None:
        reserved.add(embedding.point_id_col)
    return [c for c in table.columns if c not in reserved and pd.api.types.is_numeric_dtype(table[c])]


def _task_xy_in_crs(task: Task, dst_crs: str) -> np.ndarray:
    if task.samples.crs is None:
        raise ValueError("Task CRS is required for point-table alignment.")
    transformer = Transformer.from_crs(task.samples.crs, dst_crs, always_xy=True)
    xs = task.samples.geometry.x.to_numpy(dtype=np.float64)
    ys = task.samples.geometry.y.to_numpy(dtype=np.float64)
    x2, y2 = transformer.transform(xs, ys)
    return np.column_stack([x2, y2]).astype(np.float64, copy=False)


def _align_points(
    task: Task,
    embedding: PointEmbedding,
    *,
    method: str,
    normalize: bool,
    fill_missing: dict[str, Any] | None,
) -> AlignedEmbedding:
    del method
    table = embedding.table()
    cols = _embedding_columns_for_point_table(table, embedding)
    if not cols:
        raise ValueError(f"No embedding columns found in {embedding.path}")

    sample_ids = task.samples["sample_id"].astype(str).tolist()
    X = np.zeros((len(task.samples), len(cols)), dtype=np.float32)
    hit = np.zeros(len(task.samples), dtype=bool)
    aligner = "point_nearest"
    distance_stats: dict[str, Any] = {}

    task_id_col = embedding.task_point_id_col or "sample_id"
    if embedding.point_id_col is not None and task_id_col in task.samples.columns:
        ids = task.samples[task_id_col].astype(str).to_numpy()
        source = table[[embedding.point_id_col] + cols].copy()
        source[embedding.point_id_col] = source[embedding.point_id_col].astype(str)
        if not source[embedding.point_id_col].is_unique:
            source = source.drop_duplicates(subset=[embedding.point_id_col], keep="first")
        index = pd.Index(source[embedding.point_id_col].to_numpy())
        loc = index.get_indexer(ids)
        hit = loc >= 0
        if hit.any():
            X[hit] = source[cols].to_numpy(dtype=np.float32, copy=False)[loc[hit]]
        aligner = "point_id_lookup"
    else:
        if len(table) == 0:
            raise ValueError(f"Point embedding table is empty: {embedding.path}")
        task_xy = _task_xy_in_crs(task, embedding.crs)
        point_xy = table[[embedding.x_col, embedding.y_col]].to_numpy(dtype=np.float64)
        tree = cKDTree(point_xy)
        max_distance = float(embedding.max_distance) if embedding.max_distance is not None else np.inf
        dist, idx = tree.query(task_xy, k=1, distance_upper_bound=max_distance)
        hit = np.isfinite(dist) & (idx < len(table))
        if hit.any():
            X[hit] = table.iloc[idx[hit]][cols].to_numpy(dtype=np.float32)
        finite_dist = dist[np.isfinite(dist)]
        distance_stats = {
            "distance_crs": embedding.crs,
            "max_distance": None if embedding.max_distance is None else float(embedding.max_distance),
            "nearest_distance_mean": float(finite_dist.mean()) if len(finite_dist) else None,
            "nearest_distance_p95": float(np.quantile(finite_dist, 0.95)) if len(finite_dist) else None,
        }

    missing_before_fill = ~hit | (np.linalg.norm(np.nan_to_num(X, nan=0.0), axis=1) <= 1e-8)
    X = np.nan_to_num(X, nan=0.0).astype(np.float32)
    X, fill_report = _spatial_fill_missing(X, missing_before_fill, _sample_coordinates(task), fill_missing)
    if normalize:
        X = _l2_normalize(X)
    missing_rate = float(np.mean(np.linalg.norm(X, axis=1) <= 1e-8)) if len(X) else 0.0
    report = {
        "task_id": task.task_id,
        "embedding": embedding.metadata(),
        "aligner": aligner,
        "n_task_samples": task.n_samples,
        "n_aligned_samples": int(X.shape[0]),
        "embedding_dim": int(X.shape[1]),
        "hit_rate": float(1.0 - missing_rate),
        "hit_rate_before_fill": float(hit.mean()) if len(hit) else 0.0,
        "missing_rate": missing_rate,
        "missing_rate_before_fill": float(np.mean(missing_before_fill)) if len(X) else 0.0,
        "fill_missing": fill_report,
        "l2_normalized": bool(normalize),
        **distance_stats,
    }
    return AlignedEmbedding(X=X, sample_ids=sample_ids, report=report)
