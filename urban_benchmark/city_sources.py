from __future__ import annotations

import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.mask import mask as raster_mask
from rasterio.transform import from_origin, rowcol

from .task_builders import build_raster_task


TASK_ORDER = ["population", "road_density", "age_distribution", "gdp", "nightlight", "pm25", "lst_day_mean"]

CITY_ISO3: dict[str, tuple[str, str]] = {
    "paris": ("FRA", "fra"),
    "tokyo": ("JPN", "jpn"),
    "mexico_city": ("MEX", "mex"),
    "sao_paulo": ("BRA", "bra"),
    "los_angeles": ("USA", "usa"),
    "toronto": ("CAN", "can"),
    "berlin": ("DEU", "deu"),
    "madrid": ("ESP", "esp"),
    "istanbul": ("TUR", "tur"),
    "beijing": ("CHN", "chn"),
    "johannesburg": ("ZAF", "zaf"),
    "dubai": ("ARE", "are"),
    "bangkok": ("THA", "tha"),
    "seoul": ("KOR", "kor"),
    "shanghai": ("CHN", "chn"),
    "buenos_aires": ("ARG", "arg"),
    "santiago": ("CHL", "chl"),
    "cairo": ("EGY", "egy"),
}

AGE_GROUPS = ["00", "01", "05", "10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60", "65", "70", "75", "80", "85", "90"]
AGE_BINS: dict[str, list[str]] = {
    "age_00_04": ["00", "01"],
    "age_05_14": ["05", "10"],
    "age_15_24": ["15", "20"],
    "age_25_34": ["25", "30"],
    "age_35_44": ["35", "40"],
    "age_45_54": ["45", "50"],
    "age_55_64": ["55", "60"],
    "age_65_74": ["65", "70"],
    "age_75_84": ["75", "80"],
    "age_85_plus": ["85", "90"],
}

DRIVABLE_HIGHWAYS = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
    "living_street",
    "service",
}


@dataclass(frozen=True)
class CityTaskConfig:
    city: str
    boundary: Path
    data_root: Path
    registry_path: Path
    raw_root: Path
    prepared_raw_root: Path
    cache_root: Path
    allow_downloads: bool = False
    skip_existing: bool = True


TASK_META = {
    "population": {
        "year": "2024",
        "task_type": "regression",
        "normalization": "zscore",
        "source": "WorldPop Global_2015_2030 R2024B 2024 constrained population count",
        "drop_zeros": True,
    },
    "gdp": {
        "year": "2024",
        "task_type": "regression",
        "normalization": "zscore",
        "source": "Kummu et al. gridded GDP total, 2024 band",
        "drop_zeros": True,
    },
    "nightlight": {
        "year": "2024",
        "task_type": "regression",
        "normalization": "zscore",
        "source": "VIIRS VNL V2.2 annual nighttime lights average_masked radiance, 2024",
        "drop_zeros": False,
    },
    "pm25": {
        "year": "2022",
        "task_type": "regression",
        "normalization": "zscore",
        "source": "SEDAC/CIESIN Global Annual PM2.5 Grids, V5.GL.04, 2022",
        "drop_zeros": False,
    },
    "road_density": {
        "year": "2026",
        "task_type": "regression",
        "normalization": "zscore",
        "source": "OpenStreetMap drivable highway ways; road length density on WorldPop 2024 grid",
        "drop_zeros": False,
    },
    "age_distribution": {
        "year": "2024",
        "task_type": "distribution",
        "normalization": "none",
        "source": "WorldPop AgeSex_structures Global_2015_2030 R2024A 2024 constrained 100m aggregated to 10 age-bin distributions",
        "drop_zeros": False,
    },
    "lst_day_mean": {
        "year": "2024",
        "task_type": "regression",
        "normalization": "zscore",
        "source": "MODIS/Terra MOD11A2.061 8-day daytime land surface temperature annual mean, 2024",
        "drop_zeros": False,
    },
}


def build_city_tasks(config: CityTaskConfig, tasks: list[str]) -> list[str]:
    task_names = TASK_ORDER if "all" in tasks else tasks
    built: list[str] = []
    for task in task_names:
        if task == "landuse":
            continue
        task_id = f"{config.city}.{task}.{TASK_META[task]['year']}"
        if config.skip_existing and task_id in _read_registry(config.registry_path).get("tasks", {}):
            print(f"[skip-existing] {task_id}")
            built.append(task_id)
            continue
        print(f"[build-city-task] {task_id}", flush=True)
        raster_path, label_cols = _materialize_task_raster(config, task)
        built.append(_register_raster(config, task, raster_path, label_cols=label_cols))
    return built


def _read_registry(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"tasks": {}}


def _write_registry(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _upsert_task(path: Path, task_id: str, spec: dict[str, Any]) -> None:
    payload = _read_registry(path)
    payload.setdefault("tasks", {})[task_id] = spec
    _write_registry(path, payload)


def _boundary(path: Path, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Boundary file has no features: {path}")
    return gdf.to_crs(crs)


def _register_raster(config: CityTaskConfig, task: str, raster_path: Path, *, label_cols: list[str] | None = None) -> str:
    meta = TASK_META[task]
    task_id, spec = build_raster_task(
        data_root=config.data_root,
        city=config.city,
        task=task,
        year=str(meta["year"]),
        raw_raster=raster_path,
        task_type=str(meta["task_type"]),
        boundary=config.boundary,
        clip=False,
        source=str(meta["source"]),
        normalization=str(meta["normalization"]),
        label_cols=label_cols,
        drop_zeros=bool(meta["drop_zeros"]),
    )
    _upsert_task(config.registry_path, task_id, spec)
    return task_id


def _materialize_task_raster(config: CityTaskConfig, task: str) -> tuple[Path, list[str] | None]:
    if task == "population":
        return _materialize_population(config), None
    if task == "gdp":
        return _materialize_gdp(config), None
    if task == "pm25":
        return _materialize_pm25(config), None
    if task == "nightlight":
        return _materialize_nightlight(config), None
    if task == "road_density":
        return _materialize_road_density(config), None
    if task == "age_distribution":
        return _materialize_age_distribution(config), list(AGE_BINS.keys())
    if task == "lst_day_mean":
        return _materialize_lst(config), None
    raise ValueError(f"Unsupported auto city task: {task}")


def _worldpop_population_url(city: str) -> str:
    iso_upper, iso_lower = CITY_ISO3[city]
    filename = f"{iso_lower}_pop_2024_CN_100m_R2024B_v1.tif"
    return f"https://data.worldpop.org/GIS/Population/Global_2015_2030/R2024B/2024/{iso_upper}/v1/100m/constrained/{filename}"


def _worldpop_age_url(city: str, sex: str, age: str) -> str:
    iso_upper, iso_lower = CITY_ISO3[city]
    filename = f"{iso_lower}_{sex}_{age}_2024_CN_100m_R2024A_v1.tif"
    return f"https://data.worldpop.org/GIS/AgeSex_structures/Global_2015_2030/R2024A/2024/{iso_upper}/v1/100m/constrained/{filename}"


def _download(url: str, out: Path, allow_downloads: bool) -> None:
    if out.exists():
        return
    if not allow_downloads:
        raise FileNotFoundError(f"Missing {out}. Re-run with --allow-downloads to fetch {url}")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess_cmd = ["curl", "-L", "--fail", "--retry", "3", "-o", str(out), url]
    import subprocess

    print("[download]", " ".join(subprocess_cmd), flush=True)
    subprocess.run(subprocess_cmd, check=True)


def _materialize_population(config: CityTaskConfig) -> Path:
    _, iso_lower = CITY_ISO3[config.city]
    filename = f"{iso_lower}_pop_2024_CN_100m_R2024B_v1.tif"
    candidates = [
        config.raw_root / "worldpop" / filename,
        config.raw_root / "worldpop" / config.city / filename,
        config.raw_root / "raw" / "worldpop" / config.city / filename,
        config.registry_path.parents[1] / "external_raw" / "worldpop" / filename,
    ]
    raw = next((p for p in candidates if p.exists()), candidates[0])
    _download(_worldpop_population_url(config.city), raw, config.allow_downloads)
    out = config.prepared_raw_root / "worldpop" / config.city / "worldpop_2024.tif"
    return _clip_raster(raw, out, config.boundary)


def _materialize_gdp(config: CityTaskConfig) -> Path:
    raw = config.raw_root / "gdp" / "global" / "rast_gdpTot_1990_2024_30arcsec.tif"
    if not raw.exists():
        raw = config.raw_root / "gdp" / "rast_gdpTot_1990_2024_30arcsec.tif"
    if not raw.exists():
        raise FileNotFoundError(raw)
    out = config.prepared_raw_root / "gdp" / config.city / "gdp_total_2024_30arcsec.tif"
    return _clip_raster(raw, out, config.boundary)


def _materialize_nightlight(config: CityTaskConfig) -> Path:
    candidates = [
        config.raw_root / "nighttime_lights" / "eog_vnl_v22" / config.city / "vnl_v22_2024_average_masked-cf_cvg.tif",
        config.raw_root / "nightlight" / config.city / "vnl_v22_2024_average_masked-cf_cvg.tif",
        config.raw_root / "nightlight" / f"{config.city}_nightlight_2024.tif",
        config.registry_path.parents[1] / "external_raw" / "nighttime_lights" / "eog_vnl_v22" / config.city / "vnl_v22_2024_average_masked-cf_cvg.tif",
    ]
    raw = next((p for p in candidates if p.exists()), candidates[0])
    if not raw.exists():
        raise FileNotFoundError(
            f"Missing nightlight raster for {config.city}: {raw}. "
            "Provide a city-clipped VIIRS VNL V2.2 annual average_masked raster "
            "or use an extend-city manifest row with a download_command."
        )
    out = config.prepared_raw_root / "nighttime_lights" / config.city / "vnl_v22_2024_average_masked.tif"
    return _clip_raster(raw, out, config.boundary)


def _clip_raster(raw: Path, out: Path, boundary_path: Path) -> Path:
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(raw) as src:
        boundary = _boundary(boundary_path, src.crs)
        data, transform = raster_mask(src, boundary.geometry, crop=True, nodata=src.nodata)
        profile = src.profile.copy()
        profile.update(height=data.shape[1], width=data.shape[2], transform=transform, compress="lzw", BIGTIFF="YES")
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(data)
    return out


def _materialize_pm25(config: CityTaskConfig) -> Path:
    out = config.prepared_raw_root / "pm25" / config.city / "pm25_2022.tif"
    if out.exists():
        return out
    raw = config.raw_root / "pm25" / "satpm_v5_gl_04" / "2022" / "V5GL04.HybridPM25.Global.202201-202212.nc"
    if not raw.exists():
        raise FileNotFoundError(raw)
    boundary = _boundary(config.boundary, "EPSG:4326")
    minx, miny, maxx, maxy = boundary.total_bounds
    with xr.open_dataset(raw) as ds:
        da = ds["GWRPM25"].sel(lon=slice(minx - 0.05, maxx + 0.05), lat=slice(miny - 0.05, maxy + 0.05))
        vals = da.values.astype("float32")
        lats = da["lat"].values
        lons = da["lon"].values
    if vals.size == 0:
        raise ValueError(f"PM2.5 crop is empty for {config.city}")
    if lats[0] < lats[-1]:
        vals = vals[::-1, :]
        lats = lats[::-1]
    xres = float(abs(np.nanmedian(np.diff(lons))))
    yres = float(abs(np.nanmedian(np.diff(lats))))
    transform = from_origin(float(lons.min() - xres / 2), float(lats.max() + yres / 2), xres, yres)
    mask = geometry_mask(boundary.geometry, out_shape=vals.shape, transform=transform, invert=True)
    vals = np.where(np.isfinite(vals) & (vals < 1e20) & mask, vals, np.nan).astype("float32")
    profile = {"driver": "GTiff", "height": vals.shape[0], "width": vals.shape[1], "count": 1, "dtype": "float32", "crs": "EPSG:4326", "transform": transform, "nodata": np.nan, "compress": "lzw"}
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(vals, 1)
    return out


def _iter_lines(geom):
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "LineString":
        yield geom
    elif geom.geom_type == "MultiLineString":
        yield from geom.geoms
    elif geom.geom_type == "GeometryCollection":
        for part in geom.geoms:
            yield from _iter_lines(part)


def _materialize_road_density(config: CityTaskConfig) -> Path:
    out = config.prepared_raw_root / "road_density" / config.city / "road_density_2026.tif"
    if out.exists():
        return out
    import osmnx as ox

    population_grid = config.data_root / "tasks" / f"{config.city}.population.2024" / "labels.tif"
    if not population_grid.exists():
        population_grid = _materialize_population(config)
    boundary = _boundary(config.boundary, "EPSG:4326")
    ox.settings.cache_folder = str(config.cache_root / "osmnx")
    ox.settings.use_cache = True
    print(f"[osm] {config.city} highway ways", flush=True)
    roads = ox.features_from_polygon(boundary.geometry.union_all(), tags={"highway": True})
    if "highway" not in roads.columns:
        raise ValueError(f"No OSM highway column for {config.city}")
    roads = roads[roads["highway"].apply(_is_drivable)].copy()
    if roads.empty:
        raise ValueError(f"No drivable OSM roads for {config.city}")
    roads = roads.reset_index(drop=False).set_geometry("geometry").set_crs("EPSG:4326", allow_override=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    roads[["highway", "geometry"]].to_file(out.parent / "osm_drivable_roads.gpkg", driver="GPKG")
    metric_crs = roads.estimate_utm_crs() or "EPSG:3857"
    roads_m = roads.to_crs(metric_crs)
    to_wgs84 = Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True).transform
    with rasterio.open(population_grid) as target:
        length_m = np.zeros((target.height, target.width), dtype="float32")
        area_km2 = np.zeros_like(length_m, dtype="float32")
        for r in range(target.height):
            y0 = target.xy(r, 0, offset="ul")[1]
            y1 = target.xy(r, 0, offset="ll")[1]
            lat_mid = np.deg2rad((y0 + y1) / 2.0)
            dx = abs(target.transform.a) * 111_320.0 * max(np.cos(lat_mid), 1e-6)
            dy = abs(target.transform.e) * 111_320.0
            area_km2[r, :] = (dx * dy) / 1_000_000.0
        for geom in roads_m.geometry:
            for line in _iter_lines(geom):
                line_len = float(line.length)
                if line_len <= 0:
                    continue
                n = max(1, int(np.ceil(line_len / 25.0)))
                seg_len = line_len / n
                for i in range(n):
                    pt = line.interpolate((i + 0.5) / n, normalized=True)
                    lon, lat = to_wgs84(pt.x, pt.y)
                    r, c = rowcol(target.transform, lon, lat)
                    if 0 <= r < length_m.shape[0] and 0 <= c < length_m.shape[1]:
                        length_m[r, c] += seg_len
        density = length_m / np.clip(area_km2, 1e-9, None)
        profile = target.profile.copy()
        profile.update(count=1, dtype="float32", nodata=0.0, compress="lzw", BIGTIFF="YES")
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(density.astype("float32"), 1)
    return out


def _is_drivable(value: Any) -> bool:
    values = value if isinstance(value, list) else [value]
    return any(str(v) in DRIVABLE_HIGHWAYS for v in values)


def _crop_remote_worldpop_age(url: str, boundary: gpd.GeoDataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    errors = []
    for source in [url, f"/vsicurl/{url}"]:
        try:
            with rasterio.open(source) as src:
                crop, transform = raster_mask(src, boundary.to_crs(src.crs).geometry, crop=True, filled=True, nodata=0.0)
                arr = crop[0].astype("float32")
                if src.nodata is not None and np.isfinite(src.nodata):
                    arr = np.where(arr == src.nodata, 0.0, arr)
                profile = src.profile.copy()
                profile.update(height=arr.shape[0], width=arr.shape[1], count=1, dtype="float32", transform=transform, nodata=0.0, compress="lzw", BIGTIFF="YES")
                return np.where(np.isfinite(arr), arr, 0.0).astype("float32"), profile
        except Exception as exc:
            errors.append(repr(exc))
    raise RuntimeError(f"Could not open WorldPop age raster {url}: {'; '.join(errors)}")


def _materialize_age_distribution(config: CityTaskConfig) -> Path:
    out = config.prepared_raw_root / "worldpop_age_distribution" / config.city / "age_distribution_2024_10bin.tif"
    if out.exists():
        return out
    boundary = _boundary(config.boundary, "EPSG:4326")
    age_arrays: dict[str, np.ndarray] = {}
    profile: dict[str, Any] | None = None
    for sex in ["f", "m"]:
        for age in AGE_GROUPS:
            url = _worldpop_age_url(config.city, sex, age)
            print(f"[age] {config.city} {sex}_{age}", flush=True)
            arr, arr_profile = _crop_remote_worldpop_age(url, boundary)
            profile = profile or arr_profile
            age_arrays.setdefault(age, np.zeros_like(arr, dtype="float32"))
            age_arrays[age] += arr
    data = np.stack([sum((age_arrays[a] for a in ages), np.zeros_like(next(iter(age_arrays.values())), dtype="float32")) for ages in AGE_BINS.values()])
    assert profile is not None
    profile.update(count=data.shape[0], dtype="float32", nodata=0.0, compress="lzw", BIGTIFF="YES")
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(data.astype("float32"))
        for i, name in enumerate(AGE_BINS.keys(), start=1):
            dst.set_band_description(i, name)
    out.with_suffix(".meta.json").write_text(json.dumps({"city": config.city, "age_bins": AGE_BINS}, indent=2), encoding="utf-8")
    return out


def _materialize_lst(config: CityTaskConfig) -> Path:
    out = config.prepared_raw_root / "modis_lst" / config.city / "lst_day_mean_2024.tif"
    if out.exists():
        return out
    legacy = config.raw_root.parent / "modis_lst" / config.city / "lst_day_mean_2024.tif"
    if legacy.exists():
        return _clip_raster(legacy, out, config.boundary)
    if not config.allow_downloads:
        raise FileNotFoundError(
            "MODIS LST source is not available locally. Put a city-ready raster "
            f"at {legacy}, or re-run with --allow-downloads to export from Earth Engine."
        )
    return _download_lst_from_earth_engine(config, out)


def _download_lst_from_earth_engine(config: CityTaskConfig, out: Path) -> Path:
    import ee

    ee.Initialize()
    boundary = _boundary(config.boundary, "EPSG:4326")
    geom_json = boundary.geometry.union_all().__geo_interface__
    region = ee.Geometry(geom_json)
    image = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterDate("2024-01-01", "2025-01-01")
        .select("LST_Day_1km")
        .mean()
        .multiply(0.02)
        .subtract(273.15)
        .rename("lst_day_mean_celsius")
        .clip(region)
    )
    url = image.getDownloadURL({"region": region, "scale": 1000, "format": "GEO_TIFF", "filePerBand": False})
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=config.cache_root) as tmp:
        tmp_path = Path(tmp)
        dl = tmp_path / "lst_download"
        print(f"[lst] Earth Engine export {config.city}", flush=True)
        urlretrieve(url, dl)
        if zipfile.is_zipfile(dl):
            with zipfile.ZipFile(dl) as zf:
                members = [m for m in zf.namelist() if m.lower().endswith((".tif", ".tiff"))]
                if not members:
                    raise RuntimeError(f"Earth Engine LST zip had no GeoTIFF: {zf.namelist()}")
                zf.extract(members[0], tmp_path)
                tif = tmp_path / members[0]
                return _clip_raster(tif, out, config.boundary)
        return _clip_raster(dl, out, config.boundary)
