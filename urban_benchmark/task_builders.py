from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.mask import mask as raster_mask
from rasterio.transform import xy as transform_xy


DEFAULT_CLASS_NAMES = [
    "Residential",
    "Mixed Use",
    "Commercial",
    "Industrial",
    "Transportation",
    "Green / Recreation",
    "Institutional / Civic",
    "Utilities",
    "Water",
    "Agriculture / Rural",
    "Vacant / Reserve",
    "Other",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def relative_to_data(path: Path, data_root: Path) -> str:
    return str(path.resolve().relative_to(data_root.resolve()))


def load_boundary(path: Path, dst_crs: Any) -> gpd.GeoDataFrame:
    boundary = gpd.read_file(path)
    if boundary.empty:
        raise ValueError(f"Boundary file has no features: {path}")
    return boundary.to_crs(dst_crs)


def raster_to_samples(
    labels_path: Path,
    task_id: str,
    task_type: str,
    label_cols: list[str] | None = None,
    min_value: float | None = None,
    drop_zeros: bool = False,
    boundary_path: Path | None = None,
    sample_stride: int = 1,
) -> pd.DataFrame:
    with rasterio.open(labels_path) as src:
        nodata = src.nodata
        if task_type == "regression":
            arr = src.read(1)
            valid = np.isfinite(arr)
            if nodata is not None and np.isfinite(nodata):
                valid &= arr != nodata
            if min_value is not None:
                valid &= arr >= float(min_value)
            if drop_zeros:
                valid &= arr != 0
            labels = arr
            cols_out = ["label"]
        elif task_type == "distribution":
            arr = src.read().astype(np.float32, copy=False)
            valid = np.all(np.isfinite(arr), axis=0)
            if nodata is not None and np.isfinite(nodata):
                valid &= np.all(arr != nodata, axis=0)
            sums = arr.sum(axis=0)
            valid &= np.isfinite(sums) & (sums > 0)
            labels = arr / np.clip(sums, 1e-12, None)
            cols_out = label_cols or [f"label_{i}" for i in range(labels.shape[0])]
            if len(cols_out) != labels.shape[0]:
                raise ValueError(f"Expected {labels.shape[0]} label columns, got {len(cols_out)}")
        else:
            raise ValueError("Raster conversion supports regression and distribution tasks.")

        if boundary_path is not None:
            boundary = load_boundary(boundary_path, src.crs)
            inside = ~geometry_mask(
                [boundary.geometry.union_all()],
                transform=src.transform,
                invert=False,
                out_shape=valid.shape,
            )
            valid &= inside

        rows, cols = np.where(valid)
        if sample_stride > 1:
            keep = (rows % sample_stride == 0) & (cols % sample_stride == 0)
            rows, cols = rows[keep], cols[keep]
        xs, ys = transform_xy(src.transform, rows, cols, offset="center")
        out: dict[str, Any] = {
            "sample_id": [f"{task_id}:r{int(r)}:c{int(c)}" for r, c in zip(rows, cols)],
            "row": rows.astype(np.int32),
            "col": cols.astype(np.int32),
            "x": np.asarray(xs, dtype=np.float64),
            "y": np.asarray(ys, dtype=np.float64),
        }
        if task_type == "regression":
            out["label"] = labels[rows, cols].astype(np.float32)
        else:
            values = labels[:, rows, cols].T.astype(np.float32, copy=False)
            for i, col in enumerate(cols_out):
                out[col] = values[:, i]
        return pd.DataFrame(out)


def registry_spec_for_samples(
    *,
    data_root: Path,
    task_id: str,
    city: str,
    task: str,
    task_type: str,
    out_dir: Path,
    label_cols: list[str],
    label_col: str,
    normalization: str,
    source: str,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "task_id": task_id,
        "task_type": task_type,
        "source_type": "samples",
        "samples_path": relative_to_data(out_dir / "samples.parquet", data_root),
        "sample_id_col": "sample_id",
        "crs": "EPSG:4326",
        "task_meta_path": relative_to_data(out_dir / "task.json", data_root),
        "source": source,
        "city": city,
        "task": task,
        "split_method": "spatial_block",
        "n_blocks": 10,
    }
    if (out_dir / "labels.tif").exists():
        spec["labels_path"] = relative_to_data(out_dir / "labels.tif", data_root)
    if task_type == "distribution":
        spec["label_cols"] = label_cols
        spec["renormalize_distribution"] = True
    elif task_type == "classification":
        spec["label_col"] = label_col
        spec["label_id_col"] = label_col
        spec["class_names"] = class_names or DEFAULT_CLASS_NAMES
    else:
        spec["label_col"] = label_col
        spec["normalization"] = normalization
    return spec


def build_raster_task(
    *,
    data_root: Path,
    city: str,
    task: str,
    year: str,
    raw_raster: Path,
    task_type: str,
    boundary: Path | None = None,
    clip: bool = True,
    source: str = "",
    normalization: str = "zscore",
    label_cols: list[str] | None = None,
    min_value: float | None = None,
    drop_zeros: bool = False,
    sample_stride: int = 1,
) -> tuple[str, dict[str, Any]]:
    task_id = f"{city}.{task}.{year}"
    out_dir = data_root / "tasks" / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.tif"

    if boundary is not None and clip:
        with rasterio.open(raw_raster) as src:
            boundary_gdf = load_boundary(boundary, src.crs)
            data, transform = raster_mask(src, boundary_gdf.geometry, crop=True, nodata=src.nodata)
            profile = src.profile.copy()
            profile.update(height=data.shape[1], width=data.shape[2], transform=transform)
            with rasterio.open(labels_path, "w", **profile) as dst:
                dst.write(data)
    else:
        shutil.copy2(raw_raster, labels_path)

    samples = raster_to_samples(
        labels_path=labels_path,
        task_id=task_id,
        task_type=task_type,
        label_cols=label_cols,
        min_value=min_value,
        drop_zeros=drop_zeros,
        boundary_path=boundary if boundary is not None and not clip else None,
        sample_stride=sample_stride,
    )
    samples.to_parquet(out_dir / "samples.parquet", index=False)
    label_columns = label_cols or (["label"] if task_type == "regression" else [c for c in samples.columns if c.startswith("label_")])

    meta = {
        "task_id": task_id,
        "city": city,
        "task": task,
        "year": year,
        "task_type": task_type,
        "source": source,
        "n_samples": int(len(samples)),
        "crs": "EPSG:4326",
        "processing": [
            "Input raster was provided by the user or downloaded outside the benchmark.",
            "Raster was clipped/masked to the supplied boundary when requested.",
            "Finite non-nodata cells were exported as benchmark point samples.",
        ],
    }
    if task_type == "distribution":
        meta["label_cols"] = label_columns
    write_json(out_dir / "task.json", meta)

    spec = registry_spec_for_samples(
        data_root=data_root,
        task_id=task_id,
        city=city,
        task=task,
        task_type=task_type,
        out_dir=out_dir,
        label_cols=label_columns,
        label_col="label",
        normalization=normalization,
        source=source,
    )
    return task_id, spec


def read_vector_or_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".gpkg", ".geojson", ".json", ".shp"}:
        gdf = gpd.read_file(path)
        if gdf.crs is not None:
            gdf = gdf.to_crs("EPSG:4326")
        if "x" not in gdf.columns or "y" not in gdf.columns:
            geom = gdf.geometry if gdf.geometry.geom_type.isin(["Point"]).all() else gdf.geometry.centroid
            gdf["x"] = geom.x
            gdf["y"] = geom.y
        return pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported sample input: {path}")


def build_samples_task(
    *,
    data_root: Path,
    city: str,
    task: str,
    year: str,
    raw_samples: Path,
    task_type: str,
    source: str = "",
    x_col: str = "x",
    y_col: str = "y",
    sample_id_col: str = "sample_id",
    label_col: str = "label",
    label_id_col: str | None = None,
    label_cols: list[str] | None = None,
    normalization: str = "zscore",
    class_names: list[str] | None = None,
    mapping_csv: Path | None = None,
    mapping_source_col: str = "source_code",
    mapping_target_col: str = "label_id",
) -> tuple[str, dict[str, Any]]:
    task_id = f"{city}.{task}.{year}"
    out_dir = data_root / "tasks" / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    df = read_vector_or_table(raw_samples)
    if x_col != "x":
        df["x"] = df[x_col]
    if y_col != "y":
        df["y"] = df[y_col]
    if sample_id_col not in df.columns:
        df["sample_id"] = [f"{task_id}_{i:06d}" for i in range(len(df))]
    elif sample_id_col != "sample_id":
        df["sample_id"] = df[sample_id_col].astype(str)

    if task_type == "classification":
        if label_id_col and label_id_col in df.columns:
            df["label_id"] = pd.to_numeric(df[label_id_col], errors="raise").astype(int)
        elif mapping_csv is not None:
            mapping = pd.read_csv(mapping_csv)
            if mapping_source_col not in mapping.columns or mapping_target_col not in mapping.columns:
                raise KeyError(f"Mapping CSV must contain {mapping_source_col!r} and {mapping_target_col!r}")
            lut = dict(zip(mapping[mapping_source_col].astype(str), mapping[mapping_target_col]))
            df["label_id"] = df[label_col].astype(str).map(lut)
            if df["label_id"].isna().any():
                missing = sorted(df.loc[df["label_id"].isna(), label_col].astype(str).unique())[:20]
                raise ValueError(f"Unmapped label values: {missing}")
            df["label_id"] = pd.to_numeric(df["label_id"], errors="raise").astype(int)
        else:
            codes, uniques = pd.factorize(df[label_col].astype(str), sort=True)
            df["label_id"] = codes.astype(int)
            if class_names is None:
                class_names = [str(v) for v in uniques]
        keep = ["sample_id", "x", "y", label_col, "label_id"]
        out = df[keep].rename(columns={label_col: "label"})
        registry_label_col = "label_id"
        registry_label_cols = ["label_id"]
    else:
        registry_label_cols = label_cols or [label_col]
        out = df[["sample_id", "x", "y", *registry_label_cols]].copy()
        registry_label_col = registry_label_cols[0]
        class_names = None

    out.to_parquet(out_dir / "samples.parquet", index=False)
    meta = {
        "task_id": task_id,
        "city": city,
        "task": task,
        "year": year,
        "task_type": task_type,
        "source": source,
        "n_samples": int(len(out)),
        "crs": "EPSG:4326",
        "processing": [
            "Input samples were provided by the user or downloaded outside the benchmark.",
            "Coordinates and labels were converted to the benchmark sample schema.",
        ],
    }
    if class_names is not None:
        meta["class_names"] = class_names
    if task_type == "distribution":
        meta["label_cols"] = registry_label_cols
    write_json(out_dir / "task.json", meta)

    spec = registry_spec_for_samples(
        data_root=data_root,
        task_id=task_id,
        city=city,
        task=task,
        task_type=task_type,
        out_dir=out_dir,
        label_cols=registry_label_cols,
        label_col=registry_label_col,
        normalization=normalization,
        source=source,
        class_names=class_names,
    )
    return task_id, spec
