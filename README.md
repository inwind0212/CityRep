# CityRep Embedding Evaluation

This is the evaluation-only release package for **CITYREP: A Unified Benchmark
for Urban Representations Across Cities, Tasks, and Modalities**.

Paper: https://arxiv.org/abs/2605.26036  
arXiv:2605.26036, submitted May 25, 2026  
Release version: v0.1.0

CityRep evaluates urban representations with spatially structured splits across
multiple cities and downstream tasks. This package keeps the paper benchmark
task data and evaluation runner. It is meant for users
who want to plug in a new embedding.

## Package Contents

- `urban_benchmark/`: evaluation code and CLI.
- `data/tasks.json`: the 8-city x 8-task task registry.
- `data/tasks/`: processed downstream task payloads.
- `configs/release/protocols.json`: fixed evaluation protocols.
- `scripts/check_task_data.py`: local task-data integrity check.

## Benchmark Scope

Cities:

```text
cape_town, jakarta, london, mumbai, nairobi, new_york, singapore, sydney
```

Tasks:

```text
landuse, road_density, population, age_distribution, gdp, nightlight, pm25, lst_day_mean
```

Primary metrics:

- `landuse`: `F1_macro`, higher is better.
- `age_distribution`: `KL`, lower is better.
- All other tasks: `R2`, higher is better.

## Setup

```bash
cd /path/to/cityrep_embedding_eval
conda env create -f environment.yml
conda activate cityrep-embedding-eval
pip install -e .
python scripts/check_task_data.py
```

If you already have a compatible Python environment:

```bash
cd /path/to/cityrep_embedding_eval
pip install -e .
python scripts/check_task_data.py
```

Use `--device cpu` if CUDA is not available.

## Evaluate GeoTIFF Embeddings

The recommended release workflow is one georeferenced GeoTIFF per city:

```text
/path/to/my_embeddings/
  cape_town.tif
  jakarta.tif
  london.tif
  mumbai.tif
  nairobi.tif
  new_york.tif
  singapore.tif
  sydney.tif
```

Each `.tif` must have a valid CRS and transform. Multiple bands are treated as
embedding dimensions.

Run all CityRep cities and tasks:

```bash
python -m urban_benchmark run-model \
  --model my_tif_model \
  --model-label "My TIF Model" \
  --cities all \
  --tasks all \
  --embedding-type raster \
  --embedding-dir /path/to/my_embeddings \
  --embedding-pattern "{city}.tif" \
  --eval spatial \
  --device cuda:0 \
  --output results/my_tif_model_spatial
```

If files are nested by city, for example
`/path/to/my_embeddings/london/london.tif`, use:

```bash
--embedding-pattern "{city}/{city}.tif"
```

## Paper-Compatible Alignment

For raster embeddings, the default alignment follows the paper benchmark
protocol:

- Regression and distribution tasks use area-averaged embeddings on the target
  task grid when the embedding grid differs from the label grid.
- Same-grid rasters use row/column lookup.
- Point-sample tasks such as `landuse` sample the raster at task coordinates.

The main protocol is:

```bash
--eval spatial
```

This uses 10 x 10 spatial blocks, 5 seeds, and a fixed MLP downstream head.
For a random-split diagnostic, use `--eval random`.

## Results

Runs show progress bars for task execution, raster aggregation, seed loops, and
training epochs.

The main outputs are:

```text
results/my_tif_model_spatial/summary.csv
results/my_tif_model_spatial/main_table_avg_cstd_paper.csv
results/my_tif_model_spatial/failures.csv
results/my_tif_model_spatial/manifest.csv
```

- `summary.csv`: one row per city-task evaluation.
- `main_table_avg_cstd_paper.csv`: task-level aggregate table.
- `failures.csv`: failed runs, if any.
- `manifest.csv`: embedding files used by the run.

The run directory also stores aligned arrays, per-seed checkpoints,
predictions, and split metadata. These files can be large for full benchmark
runs; keep the summary CSVs separately if you only need final metrics.

## Run A Subset

Example: London and New York on population, GDP, and nightlight:

```bash
python -m urban_benchmark run-model \
  --model my_tif_model \
  --cities london,new_york \
  --tasks population,gdp,nightlight \
  --embedding-type raster \
  --embedding-dir /path/to/my_embeddings \
  --embedding-pattern "{city}.tif" \
  --eval spatial \
  --device cuda:0 \
  --output results/my_tif_model_subset
```

## Advanced Table Inputs

The paper benchmark also uses table-based alignment for some baselines. These
paths are kept for compatibility, but they are not the recommended interface
for new city-scale GeoTIFF embeddings.

- H3 region tables: task sample points are converted to H3 cells and looked up
  in the region table.
- Task-specific entity tables: embeddings are looked up by `sample_id`.

## Citation

If you use CityRep in your research, please cite:

```bibtex
@misc{liu2026cityrep,
  title={CITYREP: A Unified Benchmark for Urban Representations Across Cities, Tasks, and Modalities},
  author={Junyuan Liu and Xinglei Wang and Zichao Zeng and Jiazhuang Feng and Quan Qin and Ilya Ilyankou and Guangsheng Dong and Tao Cheng},
  year={2026},
  eprint={2605.26036},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2605.26036}
}
```
