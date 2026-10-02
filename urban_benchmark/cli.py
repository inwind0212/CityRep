from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from .align import align_embedding
from .city_sources import CityTaskConfig, build_city_tasks
from .embeddings import EmbeddingSource
from .evaluation import evaluate as evaluate_task
from .io import read_json
from .paths import DEFAULT_PROTOCOL_REGISTRY, DEFAULT_TASK_REGISTRY, PACKAGE_ROOT
from .protocols import load_protocol
from .task_builders import build_raster_task, build_samples_task
from .tasks import load_task, load_task_specs
from ._version import __version__


DEFAULT_MANIFEST = PACKAGE_ROOT / "data" / "registry" / "embedding_manifest.csv"
DEFAULT_MAIN_RESULT = PACKAGE_ROOT / "results" / "main_eval"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="urban-benchmark")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--task-registry", default=str(DEFAULT_TASK_REGISTRY))
    parser.add_argument("--protocol-registry", default=str(DEFAULT_PROTOCOL_REGISTRY))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-tasks")
    sub.add_parser("list-protocols")

    prepare_task = sub.add_parser("prepare-task", help="Build and register one city-task from a local raster or sample file.")
    prepare_task.add_argument("--task-registry", default=str(DEFAULT_TASK_REGISTRY))
    _add_prepare_task_args(prepare_task)

    extend_city = sub.add_parser("extend-city", help="User-facing alias for preparing one new city's downstream task data.")
    extend_city.add_argument("--task-registry", default=str(DEFAULT_TASK_REGISTRY))
    extend_city.add_argument("--city", required=True)
    extend_city.add_argument("--boundary", type=Path)
    extend_city.add_argument("--tasks", default="all", help="Comma-separated tasks to build; default all non-landuse standard tasks.")
    extend_city.add_argument("--task-manifest", help="Optional CSV/JSON rows describing downloadable or local downstream task inputs.")
    extend_city.add_argument("--data-root", default=str(PACKAGE_ROOT / "data"))
    extend_city.add_argument(
        "--raw-root",
        default=os.environ.get("CITYREP_RAW_ROOT", str(PACKAGE_ROOT / "external_raw" / "global")),
        help="Root containing reusable global raw data. Can also be set with CITYREP_RAW_ROOT.",
    )
    extend_city.add_argument("--prepared-raw-root", default=str(PACKAGE_ROOT / "external_raw" / "city_tasks"), help="Where city-clipped task rasters are written.")
    extend_city.add_argument(
        "--cache-root",
        default=os.environ.get("CITYREP_CACHE_ROOT", str(PACKAGE_ROOT / ".cache" / "city_tasks")),
        help="Large temporary/cache directory. Can also be set with CITYREP_CACHE_ROOT.",
    )
    extend_city.add_argument("--no-clip", action="store_true", help="Do not crop rasters to --boundary.")
    extend_city.add_argument("--allow-download-commands", action="store_true", help="Allow manifest rows to run download_command before preprocessing.")
    extend_city.add_argument("--allow-downloads", action="store_true", help="Allow built-in city task sources to download missing public inputs.")
    extend_city.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)

    register_task = sub.add_parser("register-task", help="Upsert a task specification into a task registry.")
    register_task.add_argument("--task-registry", default=str(DEFAULT_TASK_REGISTRY))
    register_task.add_argument("--out-registry")
    register_task.add_argument("--spec-json")
    register_task.add_argument("--task-id")
    register_task.add_argument("--city")
    register_task.add_argument("--task")
    register_task.add_argument("--task-type", choices=["regression", "classification", "distribution"])
    register_task.add_argument("--source-type", choices=["samples", "raster"], default="samples")
    register_task.add_argument("--samples-path")
    register_task.add_argument("--raster-path")
    register_task.add_argument("--label-col")
    register_task.add_argument("--label-cols")
    register_task.add_argument("--crs", default="EPSG:4326")
    register_task.add_argument("--normalization", choices=["none", "zscore", "log1p_zscore"])

    register_embedding = sub.add_parser("register-embedding", help="Append model embedding rows to a manifest.")
    register_embedding.add_argument("--task-registry", default=str(DEFAULT_TASK_REGISTRY))
    _add_manifest_target_args(register_embedding)
    _add_embedding_registration_args(register_embedding)

    validate = sub.add_parser("validate-embedding")
    _add_embedding_args(validate)
    validate.add_argument("--task", required=True)
    validate.add_argument("--no-normalize", action="store_true")

    align = sub.add_parser("align")
    _add_embedding_args(align)
    align.add_argument("--task", required=True)
    align.add_argument("--out", required=True)
    align.add_argument("--method", default="auto")
    align.add_argument("--no-normalize", action="store_true")

    evaluate = sub.add_parser("evaluate", help="Run registered model-task evaluations from an embedding manifest.")
    evaluate.add_argument("--task-registry", default=str(DEFAULT_TASK_REGISTRY))
    evaluate.add_argument("--protocol-registry", default=str(DEFAULT_PROTOCOL_REGISTRY))
    evaluate.add_argument("--embedding-manifest", default=str(DEFAULT_MANIFEST))
    evaluate.add_argument("--protocol", default="block10_5seed_mlp1024")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--out-root", default=str(DEFAULT_MAIN_RESULT))
    evaluate.add_argument("--models", nargs="*")
    evaluate.add_argument("--cities", nargs="*")
    evaluate.add_argument("--tasks", nargs="*")
    evaluate.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    evaluate.add_argument("--max-runs", type=int)
    evaluate.add_argument("--dry-run", action="store_true")

    evaluate_model = sub.add_parser("evaluate-model", help="Evaluate one external model without editing the release manifest.")
    evaluate_model.add_argument("--task-registry", default=str(DEFAULT_TASK_REGISTRY))
    evaluate_model.add_argument("--protocol-registry", default=str(DEFAULT_PROTOCOL_REGISTRY))
    _add_embedding_registration_args(evaluate_model)
    evaluate_model.add_argument("--protocol", default="block10_5seed_mlp1024")
    evaluate_model.add_argument("--split", choices=["spatial", "random"], default=None)
    evaluate_model.add_argument("--seeds", default=None)
    evaluate_model.add_argument("--device", default="cpu")
    evaluate_model.add_argument("--output", "--out-root", dest="out_root", required=True)
    evaluate_model.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    evaluate_model.add_argument("--dry-run", action="store_true")

    run_model = sub.add_parser("run-model", help="Simple one-command interface for evaluating a user-provided embedding directory.")
    run_model.add_argument("--task-registry", default=str(DEFAULT_TASK_REGISTRY))
    run_model.add_argument("--protocol-registry", default=str(DEFAULT_PROTOCOL_REGISTRY))
    run_model.add_argument("--tasks", required=True, help="Comma-separated task names or task ids.")
    run_model.add_argument("--embedding-type", choices=["raster", "region", "entity"], required=True)
    run_model.add_argument("--embedding-dir", required=True)
    run_model.add_argument("--eval", "--split", dest="split", choices=["spatial", "random"], default="spatial")
    run_model.add_argument("--output", "--out-root", dest="out_root", required=True)
    run_model.add_argument("--city", "--cities", dest="cities", default="all", help="Comma-separated cities; default all registered cities.")
    run_model.add_argument("--model", default="user_model")
    run_model.add_argument("--model-label")
    run_model.add_argument("--device", default="cpu")
    run_model.add_argument("--embedding-pattern", help="Optional pattern, e.g. '{city}/{task}.tif' or '{city}.parquet'.")
    run_model.add_argument("--region-id-col")
    run_model.add_argument("--region-type", default="h3")
    run_model.add_argument("--task-region-id-col")
    run_model.add_argument("--h3-resolution", type=int)
    run_model.add_argument("--x-col", default="x")
    run_model.add_argument("--y-col", default="y")
    run_model.add_argument("--crs", default="EPSG:4326")
    run_model.add_argument("--entity-id-col")
    run_model.add_argument("--task-entity-id-col")
    run_model.add_argument("--max-distance", type=float)
    run_model.add_argument("--embedding-cols", nargs="*")
    run_model.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    run_model.add_argument("--no-summarize", action="store_true")
    run_model.add_argument("--dry-run", action="store_true")

    summarize = sub.add_parser("summarize", help="Build compact CSV summaries from run_summary.json files.")
    summarize.add_argument("--result-root", default=str(DEFAULT_MAIN_RESULT))
    summarize.add_argument("--make-main-table", action=argparse.BooleanOptionalAction, default=True)

    audit = sub.add_parser("audit", help="Audit task registry, embedding manifest, and result completeness.")
    audit.add_argument("--task-registry", default=str(DEFAULT_TASK_REGISTRY))
    audit.add_argument("--embedding-manifest", default=str(DEFAULT_MANIFEST))
    audit.add_argument("--result-root", default=str(DEFAULT_MAIN_RESULT))
    audit.add_argument("--out", default=str(PACKAGE_ROOT / "results" / "release_audit.json"))
    audit.add_argument("--allow-missing-artifacts", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "list-tasks":
        _list_tasks(args.task_registry)
    elif args.command == "list-protocols":
        _list_protocols(args.protocol_registry)
    elif args.command == "prepare-task":
        _prepare_task(args)
    elif args.command == "extend-city":
        _prepare_city(args)
    elif args.command == "register-task":
        _register_task(args)
    elif args.command == "register-embedding":
        rows = _build_embedding_rows(args)
        _append_manifest(rows, Path(args.out_manifest))
    elif args.command == "validate-embedding":
        task = load_task(args.task, args.task_registry)
        emb = _embedding_from_args(args)
        aligned = align_embedding(task, emb, normalize=not args.no_normalize)
        print(json.dumps(aligned.report, indent=2))
    elif args.command == "align":
        task = load_task(args.task, args.task_registry)
        emb = _embedding_from_args(args)
        aligned = align_embedding(task, emb, method=args.method, normalize=not args.no_normalize)
        aligned.save(args.out)
        print(json.dumps(aligned.report, indent=2))
    elif args.command == "evaluate":
        _run_manifest_evaluation(args, Path(args.embedding_manifest), Path(args.out_root))
    elif args.command == "evaluate-model":
        _evaluate_single_model(args)
    elif args.command == "run-model":
        _run_model_simple(args)
    elif args.command == "summarize":
        _summarize(Path(args.result_root), make_main_table=bool(args.make_main_table))
    elif args.command == "audit":
        _audit(args)


def _add_manifest_target_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-manifest", default=str(DEFAULT_MANIFEST))


def _add_embedding_registration_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label")
    parser.add_argument("--embedding-type", choices=["raster", "region", "entity"], required=True)
    parser.add_argument("--embedding-path")
    parser.add_argument("--embedding-dir")
    parser.add_argument("--embedding-pattern")
    parser.add_argument("--cities", nargs="*", default=["all"])
    parser.add_argument("--tasks", nargs="*", default=["all"])
    parser.add_argument("--task-id")
    parser.add_argument("--region-id-col")
    parser.add_argument("--region-type", default="custom")
    parser.add_argument("--task-region-id-col")
    parser.add_argument("--h3-resolution", type=int)
    parser.add_argument("--x-col", default="x")
    parser.add_argument("--y-col", default="y")
    parser.add_argument("--crs", default="EPSG:4326")
    parser.add_argument("--entity-id-col")
    parser.add_argument("--task-entity-id-col")
    parser.add_argument("--max-distance", type=float)
    parser.add_argument("--embedding-cols", nargs="*")
    parser.add_argument("--alignment-policy", default="standard benchmark alignment")
    parser.add_argument("--available", action=argparse.BooleanOptionalAction, default=True)


def _add_embedding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding-type", choices=["raster", "region", "entity"], required=True)
    parser.add_argument("--embedding-path", required=True)
    parser.add_argument("--embedding-name", default="embedding")
    parser.add_argument("--region-id-col")
    parser.add_argument("--region-type", default="custom")
    parser.add_argument("--task-region-id-col")
    parser.add_argument("--h3-resolution", type=int)
    parser.add_argument("--x-col", default="x")
    parser.add_argument("--y-col", default="y")
    parser.add_argument("--crs", default="EPSG:4326")
    parser.add_argument("--entity-id-col")
    parser.add_argument("--task-entity-id-col")
    parser.add_argument("--max-distance", type=float)
    parser.add_argument("--embedding-cols", nargs="*")


def _add_prepare_task_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--city", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--year", required=True)
    parser.add_argument("--task-type", choices=["regression", "classification", "distribution"], required=True)
    parser.add_argument("--source-kind", choices=["raster", "samples"], required=True)
    parser.add_argument("--raw-path", required=True, help="Local raw raster/sample file. The benchmark does not download raw data here.")
    parser.add_argument("--boundary", type=Path)
    parser.add_argument("--data-root", default=str(PACKAGE_ROOT / "data"))
    parser.add_argument("--no-clip", action="store_true", help="Do not crop rasters to --boundary.")
    parser.add_argument("--source", default="")
    parser.add_argument("--normalization", default="zscore")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--label-id-col")
    parser.add_argument("--label-cols")
    parser.add_argument("--class-names")
    parser.add_argument("--x-col", default="x")
    parser.add_argument("--y-col", default="y")
    parser.add_argument("--sample-id-col", default="sample_id")
    parser.add_argument("--mapping-csv", type=Path)
    parser.add_argument("--mapping-source-col", default="source_code")
    parser.add_argument("--mapping-target-col", default="label_id")
    parser.add_argument("--min-value", type=float)
    parser.add_argument("--drop-zeros", action="store_true")
    parser.add_argument("--sample-stride", type=int, default=1)


def _embedding_from_args(args: argparse.Namespace):
    if args.embedding_type == "raster":
        return EmbeddingSource.from_raster(args.embedding_path, name=getattr(args, "embedding_name", "embedding"))
    if args.embedding_type == "region":
        if not args.region_id_col:
            raise SystemExit("--region-id-col is required for region embeddings")
        return EmbeddingSource.from_regions(
            args.embedding_path,
            name=getattr(args, "embedding_name", "embedding"),
            region_id_col=args.region_id_col,
            embedding_cols=args.embedding_cols,
            region_type=args.region_type,
            task_region_id_col=args.task_region_id_col,
            h3_resolution=args.h3_resolution,
        )
    return EmbeddingSource.from_entities(
        args.embedding_path,
        name=getattr(args, "embedding_name", "embedding"),
        x_col=args.x_col,
        y_col=args.y_col,
        crs=args.crs,
        embedding_cols=args.embedding_cols,
        entity_id_col=args.entity_id_col,
        task_entity_id_col=args.task_entity_id_col,
        max_distance=args.max_distance,
    )


def _list_tasks(registry_path: str) -> None:
    for task_id in sorted(load_task_specs(registry_path)):
        print(task_id)


def _list_protocols(registry_path: str) -> None:
    registry = read_json(registry_path)
    protocols = registry.get("protocols", registry)
    if isinstance(protocols, list):
        ids = [p["protocol_id"] for p in protocols]
    else:
        ids = protocols.keys()
    for protocol_id in sorted(ids):
        print(protocol_id)


def _register_task(args: argparse.Namespace) -> None:
    registry_path = Path(args.out_registry or args.task_registry)
    specs = load_task_specs(registry_path) if registry_path.exists() else {}
    if args.spec_json:
        spec = json.loads(Path(args.spec_json).read_text())
    else:
        if not args.task_id or not args.task_type:
            raise SystemExit("register-task requires --spec-json or --task-id with --task-type")
        spec = {
            "task_id": args.task_id,
            "task_type": args.task_type,
            "source_type": args.source_type,
            "crs": args.crs,
        }
        if args.city:
            spec["city"] = args.city
        if args.task:
            spec["task"] = args.task
        if args.source_type == "samples":
            if not args.samples_path:
                raise SystemExit("--samples-path is required for samples tasks")
            spec["samples_path"] = args.samples_path
        else:
            if not args.raster_path:
                raise SystemExit("--raster-path is required for raster tasks")
            spec["raster_path"] = args.raster_path
        if args.label_cols:
            spec["label_cols"] = _parse_csv_list(args.label_cols)
        elif args.label_col:
            spec["label_col"] = args.label_col
        if args.normalization:
            spec["normalization"] = args.normalization
    specs[str(spec["task_id"])] = spec
    _write_task_registry(registry_path, specs)
    print(f"[register-task] {spec['task_id']} -> {registry_path}")


def _prepare_task(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).resolve()
    boundary = Path(args.boundary).resolve() if args.boundary else None
    raw_path = Path(args.raw_path).resolve()
    task_id, spec = _build_prepared_task_from_values(
        data_root=data_root,
        city=str(args.city),
        task=str(args.task),
        year=str(args.year),
        task_type=str(args.task_type),
        source_kind=str(args.source_kind),
        raw_path=raw_path,
        boundary=boundary,
        clip=not bool(args.no_clip),
        source=str(args.source or ""),
        normalization=str(args.normalization or "zscore"),
        label_col=str(args.label_col or "label"),
        label_id_col=args.label_id_col,
        label_cols=_parse_optional_csv(args.label_cols),
        class_names=_parse_optional_csv(args.class_names),
        x_col=str(args.x_col or "x"),
        y_col=str(args.y_col or "y"),
        sample_id_col=str(args.sample_id_col or "sample_id"),
        mapping_csv=Path(args.mapping_csv).resolve() if args.mapping_csv else None,
        mapping_source_col=str(args.mapping_source_col or "source_code"),
        mapping_target_col=str(args.mapping_target_col or "label_id"),
        min_value=args.min_value,
        drop_zeros=bool(args.drop_zeros),
        sample_stride=int(args.sample_stride or 1),
    )
    _upsert_task_spec(Path(args.task_registry), spec, data_root=data_root)
    print(f"[prepare-task] {task_id} -> {data_root / 'tasks' / task_id}")


def _prepare_city(args: argparse.Namespace) -> None:
    command_label = "extend-city"
    data_root = Path(args.data_root).resolve()
    boundary = Path(args.boundary).resolve() if args.boundary else None
    if not args.task_manifest:
        if boundary is None:
            raise SystemExit("extend-city without --task-manifest requires --boundary")
        all_built: list[str] = []
        for city in _expand_arg_list(args.city):
            city_boundary = Path(str(boundary).format(city=city)).resolve()
            config = CityTaskConfig(
                city=city,
                boundary=city_boundary,
                data_root=data_root,
                registry_path=Path(args.task_registry).resolve(),
                raw_root=Path(args.raw_root).resolve(),
                prepared_raw_root=Path(args.prepared_raw_root).resolve(),
                cache_root=Path(args.cache_root).resolve(),
                allow_downloads=bool(args.allow_downloads or args.allow_download_commands),
                skip_existing=bool(args.skip_existing),
            )
            built = build_city_tasks(config, _expand_arg_list(args.tasks))
            print(f"[{command_label}] city={city} built={len(built)} -> {Path(args.task_registry)}")
            for task_id in built:
                print(f"  {task_id}")
            all_built.extend(built)
        print(f"[{command_label}] total_built_or_existing={len(all_built)}")
        return
    rows = _read_prepare_manifest(Path(args.task_manifest))
    built: list[str] = []
    for row in rows:
        task = str(_row_get(row, "task", required=True))
        year = str(_row_get(row, "year", required=True))
        task_type = str(_row_get(row, "task_type", required=True))
        source_kind = str(_row_get(row, "source_kind", default=_row_get(row, "source_type", default=""))).lower()
        raw_value = _row_get(row, "raw_path", default=None) or _row_get(row, "path", default=None)
        raw_value = raw_value or _row_get(row, "raw_raster", default=None) or _row_get(row, "raw_samples", default=None)
        if not raw_value:
            raise SystemExit(f"{command_label} row for task={task!r} is missing raw_path/path")
        raw_path = Path(str(raw_value).format(city=args.city)).expanduser().resolve()
        _maybe_run_download_command(args, row, raw_path)
        row_boundary = _row_get(row, "boundary", default=None)
        task_boundary = Path(str(row_boundary).format(city=args.city)).expanduser().resolve() if row_boundary else boundary
        task_id, spec = _build_prepared_task_from_values(
            data_root=data_root,
            city=str(args.city),
            task=task,
            year=year,
            task_type=task_type,
            source_kind=source_kind,
            raw_path=raw_path,
            boundary=task_boundary,
            clip=not bool(args.no_clip),
            source=str(_row_get(row, "source", default="") or ""),
            normalization=str(_row_get(row, "normalization", default="zscore") or "zscore"),
            label_col=str(_row_get(row, "label_col", default="label") or "label"),
            label_id_col=_row_get(row, "label_id_col", default=None),
            label_cols=_parse_optional_csv(_row_get(row, "label_cols", default=None)),
            class_names=_parse_optional_csv(_row_get(row, "class_names", default=None)),
            x_col=str(_row_get(row, "x_col", default="x") or "x"),
            y_col=str(_row_get(row, "y_col", default="y") or "y"),
            sample_id_col=str(_row_get(row, "sample_id_col", default="sample_id") or "sample_id"),
            mapping_csv=_optional_path(_row_get(row, "mapping_csv", default=None), city=str(args.city)),
            mapping_source_col=str(_row_get(row, "mapping_source_col", default="source_code") or "source_code"),
            mapping_target_col=str(_row_get(row, "mapping_target_col", default="label_id") or "label_id"),
            min_value=_optional_float(_row_get(row, "min_value", default=None)),
            drop_zeros=_truthy(_row_get(row, "drop_zeros", default=False)),
            sample_stride=int(_row_get(row, "sample_stride", default=1) or 1),
        )
        _upsert_task_spec(Path(args.task_registry), spec, data_root=data_root)
        built.append(task_id)
    print(f"[{command_label}] city={args.city} tasks={len(built)} -> {Path(args.task_registry)}")
    for task_id in built:
        print(f"  {task_id}")


def _maybe_run_download_command(args: argparse.Namespace, row: dict[str, Any], raw_path: Path) -> None:
    command = _row_get(row, "download_command", default=None)
    if not command or str(command).strip() == "" or raw_path.exists():
        return
    if not getattr(args, "allow_download_commands", False):
        raise SystemExit(
            f"Raw input is missing and the manifest row has download_command, but --allow-download-commands was not set: {raw_path}"
        )
    context = {
        "city": str(args.city),
        "raw_path": str(raw_path),
        "raw_dir": str(raw_path.parent),
        "data_root": str(Path(args.data_root).resolve()),
        "boundary": "" if getattr(args, "boundary", None) is None else str(Path(args.boundary).resolve()),
    }
    cmd = str(command).format(**context)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=PACKAGE_ROOT)
    if not raw_path.exists():
        raise FileNotFoundError(f"download_command completed but raw_path was not created: {raw_path}")


def _build_prepared_task_from_values(
    *,
    data_root: Path,
    city: str,
    task: str,
    year: str,
    task_type: str,
    source_kind: str,
    raw_path: Path,
    boundary: Path | None,
    clip: bool,
    source: str,
    normalization: str,
    label_col: str,
    label_id_col: str | None,
    label_cols: list[str] | None,
    class_names: list[str] | None,
    x_col: str,
    y_col: str,
    sample_id_col: str,
    mapping_csv: Path | None,
    mapping_source_col: str,
    mapping_target_col: str,
    min_value: float | None,
    drop_zeros: bool,
    sample_stride: int,
) -> tuple[str, dict[str, Any]]:
    if source_kind == "raster":
        return build_raster_task(
            data_root=data_root,
            city=city,
            task=task,
            year=year,
            raw_raster=raw_path,
            task_type=task_type,
            boundary=boundary,
            clip=clip,
            source=source,
            normalization=normalization,
            label_cols=label_cols,
            min_value=min_value,
            drop_zeros=drop_zeros,
            sample_stride=sample_stride,
        )
    if source_kind in {"samples", "sample"}:
        return build_samples_task(
            data_root=data_root,
            city=city,
            task=task,
            year=year,
            raw_samples=raw_path,
            task_type=task_type,
            source=source,
            x_col=x_col,
            y_col=y_col,
            sample_id_col=sample_id_col,
            label_col=label_col,
            label_id_col=label_id_col,
            label_cols=label_cols,
            normalization=normalization,
            class_names=class_names,
            mapping_csv=mapping_csv,
            mapping_source_col=mapping_source_col,
            mapping_target_col=mapping_target_col,
        )
    raise SystemExit(f"Unsupported source_kind: {source_kind!r}; use raster or samples")


def _upsert_task_spec(registry_path: Path, spec: dict[str, Any], *, data_root: Path) -> None:
    specs = load_task_specs(registry_path) if registry_path.exists() else {}
    specs[str(spec["task_id"])] = _spec_for_registry(spec, registry_path=registry_path, data_root=data_root)
    _write_task_registry(registry_path, specs)


def _spec_for_registry(spec: dict[str, Any], *, registry_path: Path, data_root: Path) -> dict[str, Any]:
    out = dict(spec)
    if registry_path.suffix.lower() != ".csv":
        return out
    base = registry_path.resolve().parent
    for key in ["samples_path", "labels_path", "task_meta_path"]:
        if key in out and out[key]:
            p = data_root / str(out[key])
            out[key] = str(p.resolve().relative_to(base))
    return out


def _read_prepare_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path).where(pd.notna, None).to_dict(orient="records")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("tasks", payload) if isinstance(payload, dict) else payload
    if isinstance(rows, dict):
        rows = list(rows.values())
    return [dict(row) for row in rows]


def _row_get(row: dict[str, Any], key: str, default: Any = None, *, required: bool = False) -> Any:
    value = row.get(key, default)
    if required and (value is None or str(value) == ""):
        raise SystemExit(f"extend-city manifest row is missing required column: {key}")
    return value


def _parse_optional_csv(value: Any) -> list[str] | None:
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, list):
        return [str(v) for v in value]
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    except json.JSONDecodeError:
        pass
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _optional_path(value: Any, *, city: str) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value).format(city=city)).expanduser().resolve()


def _write_task_registry(path: Path, specs: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        rows = []
        for spec in specs.values():
            row = dict(spec)
            for key, value in list(row.items()):
                if isinstance(value, (list, dict)):
                    row[key] = json.dumps(value)
            rows.append(row)
        pd.DataFrame(rows).sort_values("task_id").to_csv(path, index=False)
    else:
        path.write_text(json.dumps({"tasks": specs}, indent=2), encoding="utf-8")


def _build_embedding_rows(args: argparse.Namespace) -> pd.DataFrame:
    tasks = load_task_specs(args.task_registry)
    candidates = []
    if args.task_id:
        if args.task_id not in tasks:
            raise SystemExit(f"Task id not found: {args.task_id}")
        candidates = [tasks[args.task_id]]
    else:
        wanted_cities = set(_expand_arg_list(args.cities))
        wanted_tasks = set(_expand_arg_list(args.tasks))
        for spec in tasks.values():
            city = str(spec.get("city") or str(spec["task_id"]).split(".")[0])
            task = str(spec.get("task") or str(spec["task_id"]).split(".")[1])
            if ("all" in wanted_cities or city in wanted_cities) and ("all" in wanted_tasks or task in wanted_tasks):
                candidates.append(spec)
    rows: list[dict[str, Any]] = []
    for spec in sorted(candidates, key=lambda s: str(s["task_id"])):
        task_id = str(spec["task_id"])
        city = str(spec.get("city") or task_id.split(".")[0])
        task = str(spec.get("task") or task_id.split(".")[1])
        path = _embedding_path_for(args, city=city, task=task, task_id=task_id)
        config = _embedding_config(args, path)
        rows.append(
            {
                "model": args.model,
                "model_label": args.model_label or args.model,
                "city": city,
                "task": task,
                "task_id": task_id,
                "embedding_path": path,
                "embedding_source_type": args.embedding_type,
                "embedding_config": json.dumps(config, sort_keys=True),
                "source_role": f"{args.embedding_type}_embedding",
                "alignment_policy": args.alignment_policy,
                "available": bool(args.available),
                "path_exists": Path(path).is_file(),
                "mentions_population": "population" in str(path).lower(),
                "artifact_path": path,
            }
        )
    if not rows:
        raise SystemExit("No task rows matched --cities/--tasks/--task-id")
    return pd.DataFrame(rows)


def _embedding_path_for(args: argparse.Namespace, *, city: str, task: str, task_id: str) -> str:
    if args.embedding_path:
        return args.embedding_path
    if not args.embedding_dir or not args.embedding_pattern:
        raise SystemExit("Use either --embedding-path or --embedding-dir with --embedding-pattern")
    return str(Path(args.embedding_dir) / args.embedding_pattern.format(city=city, task=task, task_id=task_id, model=args.model))


def _embedding_config(args: argparse.Namespace, path: str) -> dict[str, Any]:
    cfg: dict[str, Any] = {"type": args.embedding_type, "name": args.model_label or args.model, "path": path}
    if args.embedding_type == "region":
        if not args.region_id_col:
            raise SystemExit("--region-id-col is required for region embeddings")
        cfg.update(
            {
                "region_id_col": args.region_id_col,
                "region_type": args.region_type,
                "task_region_id_col": args.task_region_id_col,
                "h3_resolution": args.h3_resolution,
            }
        )
    if args.embedding_type == "entity":
        cfg.update(
            {
                "x_col": args.x_col,
                "y_col": args.y_col,
                "crs": args.crs,
                "entity_id_col": args.entity_id_col,
                "task_entity_id_col": args.task_entity_id_col,
                "max_distance": args.max_distance,
            }
        )
    if args.embedding_cols:
        cfg["embedding_cols"] = list(args.embedding_cols)
    return {k: v for k, v in cfg.items() if v is not None}


def _append_manifest(rows: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        old = pd.read_csv(out_path)
        key_cols = ["model", "city", "task_id"]
        merged = old.merge(rows[key_cols], on=key_cols, how="left", indicator=True)
        old = old[merged["_merge"].eq("left_only").to_numpy()]
        rows = pd.concat([old, rows], ignore_index=True)
    rows.to_csv(out_path, index=False)
    print(f"[register-embedding] rows={len(rows)} -> {out_path}")


def _run_manifest_evaluation(args: argparse.Namespace, manifest_path: Path, out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_path, args)
    manifest.drop(columns=["embedding_config_resolved"]).to_csv(out_root / "manifest.csv", index=False)
    print(f"[manifest] rows={len(manifest)} path_exists={int(manifest['path_exists'].sum())} -> {out_root / 'manifest.csv'}")
    if getattr(args, "dry_run", False):
        print(f"[dry-run] existing={int(manifest['path_exists'].sum())} missing={int((~manifest['path_exists']).sum())}")
        return

    protocol = load_protocol(args.protocol, args.protocol_registry)
    protocol.setdefault("predictor", {})
    protocol["predictor"]["device"] = args.device

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    completed = 0
    task_iter = tqdm(list(manifest.itertuples(index=False)), desc="tasks", unit="task", dynamic_ncols=True)
    for row in task_iter:
        task_iter.set_postfix(city=str(row.city), task=str(row.task))
        if not bool(row.path_exists):
            failures.append(_failure_payload(row, "FileNotFoundError", "embedding path does not exist"))
            continue
        run_dir = out_root / str(row.model) / str(row.city) / str(row.task_id) / str(args.protocol)
        summary_path = run_dir / "run_summary.json"
        if args.skip_existing and summary_path.is_file():
            results.append(_result_payload(row, json.loads(summary_path.read_text()), args.protocol))
            print(f"[skip] {row.model} {row.task_id}")
            continue
        max_runs = getattr(args, "max_runs", None)
        if max_runs is not None and completed >= max_runs:
            break
        print(f"[run] {row.model} {row.task_id} -> {run_dir}")
        try:
            task = load_task(str(row.task_id), registry_path=args.task_registry)
            embedding = EmbeddingSource.from_config(row.embedding_config_resolved)
            result = evaluate_task(task, embedding, protocol=protocol, out_dir=run_dir)
            results.append(_result_payload(row, {"metrics": result.metrics, "metrics_std": result.metrics_std, "task_type": task.task_type, "n_samples": task.n_samples, "save_dir": result.save_dir}, args.protocol))
            pd.DataFrame(results).to_csv(out_root / "results_partial.csv", index=False)
        except Exception as exc:
            failures.append(_failure_payload(row, repr(exc), traceback.format_exc()))
            pd.DataFrame(failures).to_csv(out_root / "failures.csv", index=False)
            print(f"[fail] {row.model} {row.task_id}: {exc!r}")
        completed += 1
    pd.DataFrame(results).to_csv(out_root / "results.csv", index=False)
    pd.DataFrame(failures).to_csv(out_root / "failures.csv", index=False)
    print(f"[done] results={len(results)} failures={len(failures)}")


def _evaluate_single_model(args: argparse.Namespace) -> None:
    protocol = args.protocol
    if args.split == "random":
        protocol = "random_5seed_mlp1024"
    elif args.split == "spatial":
        protocol = "block10_5seed_mlp1024"
    rows = _build_embedding_rows(args)
    with tempfile.TemporaryDirectory(prefix="urban_benchmark_manifest_") as tmp:
        manifest = Path(tmp) / "manifest.csv"
        rows.to_csv(manifest, index=False)
        args.embedding_manifest = str(manifest)
        args.protocol = protocol
        if args.seeds:
            print("[warn] --seeds is accepted for CLI compatibility; use a custom protocol registry to change seed values.")
        _run_manifest_evaluation(args, manifest, Path(args.out_root))


def _run_model_simple(args: argparse.Namespace) -> None:
    protocol = "random_5seed_mlp1024" if args.split == "random" else "block10_5seed_mlp1024"
    rows = _build_simple_model_rows(args)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = out_root / "embedding_manifest.csv"
    rows.to_csv(manifest, index=False)
    args.embedding_manifest = str(manifest)
    args.protocol = protocol
    _run_manifest_evaluation(args, manifest, out_root)
    if not args.no_summarize and not args.dry_run:
        _summarize(out_root, make_main_table=True)


def _build_simple_model_rows(args: argparse.Namespace) -> pd.DataFrame:
    tasks = load_task_specs(args.task_registry)
    wanted_tasks = set(_expand_arg_list(args.tasks))
    wanted_cities = set(_expand_arg_list(args.cities))
    embedding_root = Path(args.embedding_dir).expanduser()
    rows: list[dict[str, Any]] = []
    for spec in sorted(tasks.values(), key=lambda s: str(s["task_id"])):
        task_id = str(spec["task_id"])
        city = str(spec.get("city") or task_id.split(".")[0])
        task = str(spec.get("task") or task_id.split(".")[1])
        if "all" not in wanted_cities and city not in wanted_cities:
            continue
        if "all" not in wanted_tasks and task not in wanted_tasks and task_id not in wanted_tasks:
            continue
        path = _resolve_simple_embedding_path(
            embedding_root=embedding_root,
            pattern=args.embedding_pattern,
            city=city,
            task=task,
            task_id=task_id,
            embedding_type=args.embedding_type,
        )
        config = _simple_embedding_config(args, path)
        rows.append(
            {
                "model": args.model,
                "model_label": args.model_label or args.model,
                "city": city,
                "task": task,
                "task_id": task_id,
                "embedding_path": str(path),
                "embedding_source_type": args.embedding_type,
                "embedding_config": json.dumps(config, sort_keys=True),
                "source_role": f"{args.embedding_type}_embedding",
                "alignment_policy": "standard benchmark alignment",
                "available": True,
                "path_exists": bool(path.is_file()),
                "mentions_population": "population" in str(path).lower(),
                "artifact_path": str(path),
            }
        )
    if not rows:
        raise SystemExit("No task rows matched. Check --tasks, --cities, and --task-registry.")
    missing = [r for r in rows if not r["path_exists"]]
    if missing:
        preview = "\n".join(f"  {r['city']} {r['task']}: {r['embedding_path']}" for r in missing[:12])
        print(f"[warn] missing embedding files for {len(missing)}/{len(rows)} rows:\n{preview}")
    return pd.DataFrame(rows)


def _resolve_simple_embedding_path(
    *,
    embedding_root: Path,
    pattern: str | None,
    city: str,
    task: str,
    task_id: str,
    embedding_type: str,
) -> Path:
    if pattern:
        return embedding_root / pattern.format(city=city, task=task, task_id=task_id)
    if embedding_type == "raster":
        candidates = [
            embedding_root / city / f"{task}.tif",
            embedding_root / city / f"{task_id}.tif",
            embedding_root / city / f"{city}_{task}.tif",
            embedding_root / f"{city}_{task}.tif",
            embedding_root / f"{task_id}.tif",
            embedding_root / city / f"{city}.tif",
            embedding_root / f"{city}.tif",
        ]
    else:
        candidates = []
        for suffix in ["parquet", "csv", "feather"]:
            candidates.extend(
                [
                    embedding_root / city / f"{task}.{suffix}",
                    embedding_root / city / f"{task_id}.{suffix}",
                    embedding_root / city / f"{city}_{task}.{suffix}",
                    embedding_root / f"{city}_{task}.{suffix}",
                    embedding_root / f"{task_id}.{suffix}",
                    embedding_root / city / f"{city}.{suffix}",
                    embedding_root / f"{city}.{suffix}",
                ]
            )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _simple_embedding_config(args: argparse.Namespace, path: Path) -> dict[str, Any]:
    cfg: dict[str, Any] = {"type": args.embedding_type, "name": args.model_label or args.model, "path": str(path)}
    if args.embedding_type == "region":
        if not args.region_id_col:
            raise SystemExit("run-model with --embedding-type region requires --region-id-col, usually --region-id-col h3_id")
        cfg.update(
            {
                "region_id_col": args.region_id_col,
                "region_type": args.region_type,
                "task_region_id_col": args.task_region_id_col,
                "h3_resolution": args.h3_resolution,
            }
        )
    if args.embedding_type == "entity":
        cfg.update(
            {
                "x_col": args.x_col,
                "y_col": args.y_col,
                "crs": args.crs,
                "entity_id_col": args.entity_id_col,
                "task_entity_id_col": args.task_entity_id_col,
                "max_distance": args.max_distance,
            }
        )
    if args.embedding_cols:
        cfg["embedding_cols"] = list(args.embedding_cols)
    return {k: v for k, v in cfg.items() if v is not None}


def _load_manifest(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {"model", "city", "task", "task_id", "embedding_config", "available"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise KeyError(f"Embedding manifest is missing columns: {missing}")
    models = _optional_filter_set(getattr(args, "models", None))
    cities = _optional_filter_set(getattr(args, "cities", None))
    tasks = _optional_filter_set(getattr(args, "tasks", None))
    if models:
        manifest = manifest[manifest["model"].astype(str).isin(models)]
    if cities and "all" not in cities:
        manifest = manifest[manifest["city"].astype(str).isin(cities)]
    if tasks and "all" not in tasks:
        manifest = manifest[manifest["task"].astype(str).isin(tasks)]
    available_mask = manifest["available"].map(_truthy).astype(bool)
    manifest = manifest.loc[available_mask].copy()
    manifest["embedding_config_resolved"] = manifest["embedding_config"].map(lambda v: _resolve_embedding_config(_parse_config(v), path))
    manifest["path_exists"] = manifest["embedding_config_resolved"].map(lambda c: Path(c["path"]).is_file())
    return manifest.sort_values(["model", "city", "task"]).reset_index(drop=True)


def _resolve_embedding_config(config: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    out = dict(config)
    path = out.get("path")
    if path is not None:
        p = Path(str(path))
        if not p.is_absolute():
            root_candidate = PACKAGE_ROOT / p
            manifest_candidate = manifest_path.resolve().parent / p
            p = root_candidate if root_candidate.exists() else manifest_candidate
        out["path"] = str(p)
    return out


def _summarize(result_root: Path, *, make_main_table: bool) -> None:
    rows = []
    manifest_path = result_root / "manifest.csv"
    allowed: set[tuple[str, str, str]] | None = None
    if manifest_path.is_file():
        manifest = pd.read_csv(manifest_path)
        allowed = set(manifest.loc[manifest["available"].map(_truthy), ["model", "city", "task_id"]].itertuples(index=False, name=None))
    for path in sorted(result_root.glob("*/*/*/*/run_summary.json")):
        summary = json.loads(path.read_text())
        model, city, task_id, protocol = path.relative_to(result_root).parts[:4]
        if allowed is not None and (model, city, task_id) not in allowed:
            continue
        row = {
            "model": model,
            "model_label": _model_label_from_manifest(manifest_path, model),
            "city": city,
            "task_id": task_id,
            "task": task_id.split(".")[1],
            "protocol": protocol,
            "task_type": summary.get("task_type"),
            "n_samples": summary.get("n_samples"),
        }
        row.update(summary.get("metrics", {}))
        row.update({f"{k}_std": v for k, v in summary.get("metrics_std", {}).items()})
        row["save_dir"] = str(path.parent)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty and (result_root / "summary.csv").is_file():
        df = pd.read_csv(result_root / "summary.csv")
        print(f"[summary] using existing compact summary -> {result_root / 'summary.csv'}")
    else:
        result_root.mkdir(parents=True, exist_ok=True)
        df.to_csv(result_root / "summary.csv", index=False)
        print(f"[summary] {len(df)} rows -> {result_root / 'summary.csv'}")
    result_root.mkdir(parents=True, exist_ok=True)
    if make_main_table and not df.empty:
        _write_main_table(df, result_root)


def _write_main_table(summary: pd.DataFrame, result_root: Path) -> None:
    table_dir = result_root / "paper_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    model_order = [
        "sphere2vec_fixed",
        "place2vec",
        "space2vec",
        "calliper",
        "cityfm",
        "urban2vec",
        "muse",
        "satclip",
        "tessera",
        "alphaearth",
        "aether",
    ]
    task_specs = [
        ("landuse", "LUC", "F1_macro"),
        ("road_density", "RDE", "R2"),
        ("population", "POP", "R2"),
        ("age_distribution", "AGE", "KL"),
        ("gdp", "GDP", "R2"),
        ("nightlight", "NTL", "R2"),
        ("pm25", "PM25", "R2"),
        ("lst_day_mean", "LST", "R2"),
    ]
    age_excluded = {"mumbai", "nairobi", "jakarta", "cape_town"}
    rows = []
    for task, abbr, metric in task_specs:
        sub = summary[summary["task"].eq(task)].copy()
        if sub.empty or metric not in sub.columns:
            continue
        if task == "age_distribution":
            sub = sub[~sub["city"].isin(age_excluded)]
        sub["metric"] = metric
        sub["score"] = pd.to_numeric(sub[metric], errors="coerce")
        rows.append(sub.dropna(subset=["score"]))
    city_scores = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    main = (
        city_scores.groupby(["model", "model_label", "task", "metric"], observed=False)
        .agg(avg=("score", "mean"), c_std=("score", lambda s: s.std(ddof=0)), n_cities=("city", "nunique"))
        .reset_index()
    )
    main["model"] = pd.Categorical(main["model"], model_order, ordered=True)
    main.to_csv(table_dir / "main_table_avg_cstd_long.csv", index=False)
    paper_rows = []
    for model in model_order:
        sub_model = main[main["model"].astype(str).eq(model)]
        if sub_model.empty:
            continue
        row = {"model": model, "model_label": sub_model["model_label"].iloc[0]}
        for task, abbr, metric in task_specs:
            sub = sub_model[sub_model["task"].eq(task)]
            row[f"{abbr} Avg ({metric})"] = pd.NA if sub.empty else round(float(sub["avg"].iloc[0]), 3)
            row[f"{abbr} C-Std."] = pd.NA if sub.empty else round(float(sub["c_std"].iloc[0]), 3)
        paper_rows.append(row)
    paper = pd.DataFrame(paper_rows)
    paper.to_csv(result_root / "main_table_avg_cstd_paper.csv", index=False)
    print(f"[main-table] {result_root / 'main_table_avg_cstd_paper.csv'}")


def _audit(args: argparse.Namespace) -> None:
    tasks = load_task_specs(args.task_registry)
    manifest = pd.read_csv(args.embedding_manifest)
    manifest["available_bool"] = manifest["available"].map(_truthy).astype(bool)
    available = manifest.loc[manifest["available_bool"]].copy()
    available["config"] = available["embedding_config"].map(_parse_config)
    available["path"] = available["config"].map(lambda c: c.get("path"))
    available["resolved_path"] = available["config"].map(lambda c: _resolve_embedding_config(c, Path(args.embedding_manifest)).get("path"))
    available["path_exists"] = available["resolved_path"].map(lambda p: Path(str(p)).is_file())
    available["mentions_population"] = available[["path", "source_role", "alignment_policy"]].astype(str).apply(
        lambda s: s.str.contains("population", case=False, na=False)
    ).any(axis=1)
    result_root = Path(args.result_root)
    summary_csv = result_root / "summary.csv"
    failure_csv = result_root / "failures.csv"
    failures = 0
    if failure_csv.is_file() and failure_csv.stat().st_size:
        try:
            failures = len(pd.read_csv(failure_csv))
        except pd.errors.EmptyDataError:
            failures = 0
    sparse_lu_population = available[available["task"].isin(["nightlight", "landuse"]) & available["mentions_population"]]
    payload = {
        "task_count": len(tasks),
        "embedding_rows": int(len(manifest)),
        "embedding_rows_available": int(len(available)),
        "embedding_paths_existing": int(available["path_exists"].sum()),
        "missing_artifact_rows": int((~available["path_exists"]).sum()),
        "missing_artifacts_allowed": bool(args.allow_missing_artifacts),
        "model_count": int(available["model"].nunique()),
        "city_count": int(pd.Series([str(s.get("city") or k.split(".")[0]) for k, s in tasks.items()]).nunique()),
        "nightlight_landuse_population_mentions": int(len(sparse_lu_population)),
        "compact_summary_rows": int(len(pd.read_csv(summary_csv))) if summary_csv.is_file() else 0,
        "failure_rows": int(failures),
        "run_summary_count": len(list(result_root.glob("*/*/*/*/run_summary.json"))),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    hard_fail = payload["nightlight_landuse_population_mentions"] > 0 or payload["failure_rows"] > 0
    if hard_fail or (payload["missing_artifact_rows"] > 0 and not args.allow_missing_artifacts):
        raise SystemExit(1)


def _parse_config(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return ast.literal_eval(str(value))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _optional_filter_set(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    return set(_expand_arg_list(values))


def _expand_arg_list(values: list[str] | str | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    out: list[str] = []
    for value in values:
        out.extend([v.strip() for v in str(value).split(",") if v.strip()])
    return out or ["all"]


def _parse_csv_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _result_payload(row: Any, summary: dict[str, Any], protocol: str) -> dict[str, Any]:
    payload = {
        "model": row.model,
        "model_label": getattr(row, "model_label", row.model),
        "city": row.city,
        "task": row.task,
        "task_id": row.task_id,
        "protocol": protocol,
        "task_type": summary.get("task_type"),
        "n_samples": summary.get("n_samples"),
    }
    payload.update(summary.get("metrics", {}))
    payload.update({f"{k}_std": v for k, v in summary.get("metrics_std", {}).items()})
    payload["save_dir"] = summary.get("save_dir")
    return payload


def _failure_payload(row: Any, error: str, detail: str) -> dict[str, Any]:
    return {
        "model": row.model,
        "city": row.city,
        "task": row.task,
        "task_id": row.task_id,
        "embedding_config": getattr(row, "embedding_config", None),
        "error": error,
        "detail": detail,
    }


def _model_label_from_manifest(manifest_path: Path, model: str) -> str:
    if manifest_path.is_file():
        manifest = pd.read_csv(manifest_path)
        sub = manifest[manifest["model"].astype(str).eq(model)]
        if not sub.empty and "model_label" in sub.columns:
            return str(sub["model_label"].iloc[0])
    return model


if __name__ == "__main__":
    main(sys.argv[1:])
