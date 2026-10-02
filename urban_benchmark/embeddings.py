from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import rasterio

from .io import read_table


@dataclass
class RasterEmbedding:
    name: str
    path: str
    source_type: str = "raster"

    def metadata(self) -> dict[str, Any]:
        with rasterio.open(self.path) as src:
            return {
                "name": self.name,
                "source_type": self.source_type,
                "path": str(Path(self.path).expanduser().absolute()),
                "dim": int(src.count),
                "crs": None if src.crs is None else src.crs.to_string(),
                "height": int(src.height),
                "width": int(src.width),
                "transform": tuple(src.transform),
                "nodata": src.nodata,
            }


@dataclass
class RegionEmbedding:
    name: str
    path: str
    region_id_col: str
    embedding_cols: list[str] | None = None
    region_type: str = "custom"
    task_region_id_col: str | None = None
    h3_resolution: int | None = None
    source_type: str = "region_table"

    def table(self) -> pd.DataFrame:
        df = read_table(self.path)
        if self.region_id_col not in df.columns:
            raise KeyError(f"Missing region id column '{self.region_id_col}' in {self.path}")
        cols = self.embedding_cols
        if cols is None:
            cols = [c for c in df.columns if c != self.region_id_col and pd.api.types.is_numeric_dtype(df[c])]
        if not cols:
            raise ValueError(f"No embedding columns found in {self.path}")
        out = df[[self.region_id_col] + list(cols)].copy()
        out[self.region_id_col] = out[self.region_id_col].astype(str)
        out[list(cols)] = out[list(cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return out.set_index(self.region_id_col)

    def metadata(self) -> dict[str, Any]:
        table = self.table()
        return {
            "name": self.name,
            "source_type": self.source_type,
            "path": str(Path(self.path).expanduser().absolute()),
            "region_id_col": self.region_id_col,
            "region_type": self.region_type,
            "h3_resolution": self.h3_resolution,
            "dim": int(table.shape[1]),
            "n_regions": int(table.shape[0]),
        }


@dataclass
class PointEmbedding:
    name: str
    path: str
    x_col: str = "x"
    y_col: str = "y"
    crs: str = "EPSG:4326"
    embedding_cols: list[str] | None = None
    point_id_col: str | None = None
    task_point_id_col: str | None = None
    max_distance: float | None = None
    source_type: str = "point_table"

    def table(self) -> pd.DataFrame:
        df = read_table(self.path)
        missing = [c for c in [self.x_col, self.y_col] if c not in df.columns]
        if missing:
            raise KeyError(f"Missing coordinate columns in {self.path}: {missing}")
        cols = self.embedding_cols
        reserved = {self.x_col, self.y_col}
        if self.point_id_col is not None:
            reserved.add(self.point_id_col)
        if cols is None:
            cols = [c for c in df.columns if c not in reserved and pd.api.types.is_numeric_dtype(df[c])]
        if not cols:
            raise ValueError(f"No embedding columns found in {self.path}")
        keep_cols = [self.x_col, self.y_col] + ([self.point_id_col] if self.point_id_col is not None else []) + list(cols)
        out = df[keep_cols].copy()
        out[self.x_col] = pd.to_numeric(out[self.x_col], errors="coerce")
        out[self.y_col] = pd.to_numeric(out[self.y_col], errors="coerce")
        out = out.dropna(subset=[self.x_col, self.y_col]).copy()
        out[list(cols)] = out[list(cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if self.point_id_col is not None:
            out[self.point_id_col] = out[self.point_id_col].astype(str)
        return out.reset_index(drop=True)

    def metadata(self) -> dict[str, Any]:
        table = self.table()
        cols = self.embedding_cols
        if cols is None:
            reserved = {self.x_col, self.y_col}
            if self.point_id_col is not None:
                reserved.add(self.point_id_col)
            cols = [c for c in table.columns if c not in reserved and pd.api.types.is_numeric_dtype(table[c])]
        return {
            "name": self.name,
            "source_type": self.source_type,
            "path": str(Path(self.path).expanduser().absolute()),
            "x_col": self.x_col,
            "y_col": self.y_col,
            "crs": self.crs,
            "point_id_col": self.point_id_col,
            "task_point_id_col": self.task_point_id_col,
            "max_distance": self.max_distance,
            "dim": int(len(cols)),
            "n_points": int(table.shape[0]),
        }


class EmbeddingSource:
    @staticmethod
    def from_raster(path: str | Path, name: str = "embedding") -> RasterEmbedding:
        return RasterEmbedding(name=name, path=str(path))

    @staticmethod
    def from_regions(
        path: str | Path,
        *,
        region_id_col: str,
        name: str = "embedding",
        embedding_cols: list[str] | None = None,
        region_type: str = "custom",
        task_region_id_col: str | None = None,
        h3_resolution: int | None = None,
    ) -> RegionEmbedding:
        return RegionEmbedding(
            name=name,
            path=str(path),
            region_id_col=region_id_col,
            embedding_cols=embedding_cols,
            region_type=region_type,
            task_region_id_col=task_region_id_col,
            h3_resolution=h3_resolution,
        )

    @staticmethod
    def from_points(
        path: str | Path,
        *,
        name: str = "embedding",
        x_col: str = "x",
        y_col: str = "y",
        crs: str = "EPSG:4326",
        embedding_cols: list[str] | None = None,
        point_id_col: str | None = None,
        task_point_id_col: str | None = None,
        max_distance: float | None = None,
    ) -> PointEmbedding:
        return PointEmbedding(
            name=name,
            path=str(path),
            x_col=x_col,
            y_col=y_col,
            crs=crs,
            embedding_cols=embedding_cols,
            point_id_col=point_id_col,
            task_point_id_col=task_point_id_col,
            max_distance=max_distance,
        )

    @staticmethod
    def from_entities(
        path: str | Path,
        *,
        name: str = "embedding",
        x_col: str = "x",
        y_col: str = "y",
        crs: str = "EPSG:4326",
        embedding_cols: list[str] | None = None,
        entity_id_col: str | None = None,
        task_entity_id_col: str | None = None,
        max_distance: float | None = None,
    ) -> PointEmbedding:
        """Entity embeddings are point-table embeddings with public naming."""
        return EmbeddingSource.from_points(
            path,
            name=name,
            x_col=x_col,
            y_col=y_col,
            crs=crs,
            embedding_cols=embedding_cols,
            point_id_col=entity_id_col,
            task_point_id_col=task_entity_id_col,
            max_distance=max_distance,
        )

    @staticmethod
    def from_config(cfg: dict[str, Any]) -> RasterEmbedding | RegionEmbedding | PointEmbedding:
        typ = str(cfg.get("type", cfg.get("source_type", "raster"))).lower()
        if typ == "raster":
            return EmbeddingSource.from_raster(cfg["path"], name=cfg.get("name", "embedding"))
        if typ in {"region", "region_table", "regions"}:
            return EmbeddingSource.from_regions(
                cfg["path"],
                region_id_col=cfg["region_id_col"],
                name=cfg.get("name", "embedding"),
                embedding_cols=cfg.get("embedding_cols"),
                region_type=cfg.get("region_type", "custom"),
                task_region_id_col=cfg.get("task_region_id_col"),
                h3_resolution=cfg.get("h3_resolution"),
            )
        if typ in {"point", "points", "point_table", "entity", "entities", "entity_table"}:
            return EmbeddingSource.from_points(
                cfg["path"],
                name=cfg.get("name", "embedding"),
                x_col=cfg.get("x_col", "x"),
                y_col=cfg.get("y_col", "y"),
                crs=cfg.get("crs", "EPSG:4326"),
                embedding_cols=cfg.get("embedding_cols"),
                point_id_col=cfg.get("point_id_col", cfg.get("entity_id_col")),
                task_point_id_col=cfg.get("task_point_id_col", cfg.get("task_entity_id_col")),
                max_distance=cfg.get("max_distance"),
            )
        raise ValueError(f"Unsupported embedding type: {typ}")
