from __future__ import annotations

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from urban_benchmark.align import align_embedding
from urban_benchmark.embeddings import RasterEmbedding
from urban_benchmark.tasks import Task


def _write_raster(path, array, transform):
    array = np.asarray(array, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=array.shape[-1],
        height=array.shape[-2],
        count=1 if array.ndim == 2 else array.shape[0],
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        if array.ndim == 2:
            dst.write(array, 1)
        else:
            dst.write(array)


def _task(task_type, label_path):
    df = gpd.GeoDataFrame(
        {
            "sample_id": ["a", "b"],
            "row": [0, 1],
            "col": [0, 1],
            "x": [0.5, 1.5],
            "y": [1.5, 0.5],
        },
        geometry=[Point(0.5, 1.5), Point(1.5, 0.5)],
        crs="EPSG:4326",
    )
    y = np.asarray([0.0, 1.0], dtype=np.float32).reshape(-1, 1)
    if task_type == "classification":
        y = np.asarray([0, 1], dtype=np.int64)
    return Task(
        task_id=f"demo.{task_type}",
        task_type=task_type,
        samples=df,
        y=y,
        label_columns=["label"],
        meta={"labels_path": str(label_path)},
    )


def test_auto_area_averages_finer_embedding_to_coarse_regression_label_grid(tmp_path):
    label_path = tmp_path / "label.tif"
    embedding_path = tmp_path / "embedding.tif"
    _write_raster(label_path, [[1, 1], [1, 1]], from_origin(0, 2, 1, 1))
    _write_raster(
        embedding_path,
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        from_origin(0, 2, 0.5, 0.5),
    )

    aligned = align_embedding(_task("regression", label_path), RasterEmbedding("demo", str(embedding_path)), normalize=False)

    assert aligned.report["aligner"] == "raster_area_average"
    np.testing.assert_allclose(aligned.X[:, 0], [3.5, 13.5], rtol=1e-6)


def test_auto_uses_cell_lookup_for_matching_grids(tmp_path):
    label_path = tmp_path / "label.tif"
    embedding_path = tmp_path / "embedding.tif"
    transform = from_origin(0, 2, 1, 1)
    _write_raster(label_path, [[1, 1], [1, 1]], transform)
    _write_raster(embedding_path, [[10, 20], [30, 40]], transform)

    aligned = align_embedding(_task("regression", label_path), RasterEmbedding("demo", str(embedding_path)), normalize=False)

    assert aligned.report["aligner"] == "raster_cell"
    np.testing.assert_allclose(aligned.X[:, 0], [10, 40], rtol=1e-6)



def test_auto_area_averages_when_label_grid_is_finer_than_embedding_grid(tmp_path):
    label_path = tmp_path / "label_fine.tif"
    embedding_path = tmp_path / "embedding_coarse.tif"
    _write_raster(
        label_path,
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        from_origin(0, 2, 0.5, 0.5),
    )
    _write_raster(embedding_path, [[10, 20], [30, 40]], from_origin(0, 2, 1, 1))
    df = gpd.GeoDataFrame(
        {
            "sample_id": ["a", "b"],
            "row": [0, 3],
            "col": [0, 3],
            "x": [0.25, 1.75],
            "y": [1.75, 0.25],
        },
        geometry=[Point(0.25, 1.75), Point(1.75, 0.25)],
        crs="EPSG:4326",
    )
    task = Task(
        task_id="demo.regression.fine",
        task_type="regression",
        samples=df,
        y=np.asarray([0.0, 1.0], dtype=np.float32).reshape(-1, 1),
        label_columns=["label"],
        meta={"labels_path": str(label_path)},
    )

    aligned = align_embedding(task, RasterEmbedding("demo", str(embedding_path)), normalize=False)

    assert aligned.report["aligner"] == "raster_area_average"
    np.testing.assert_allclose(aligned.X[:, 0], [10, 40], rtol=1e-6)

def test_auto_keeps_classification_raster_mismatch_as_coordinate_sampling(tmp_path):
    label_path = tmp_path / "label.tif"
    embedding_path = tmp_path / "embedding.tif"
    _write_raster(label_path, [[1, 1], [1, 1]], from_origin(0, 2, 1, 1))
    _write_raster(
        embedding_path,
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        from_origin(0, 2, 0.5, 0.5),
    )

    aligned = align_embedding(_task("classification", label_path), RasterEmbedding("demo", str(embedding_path)), normalize=False)

    assert aligned.report["aligner"] == "raster_sample"
    np.testing.assert_allclose(aligned.X[:, 0], [6, 16], rtol=1e-6)
