#!/usr/bin/env python3
"""
Предсказание границ полей: обученная RF-модель (без длинной цепочки фильтров).

Постобработка только техническая: вода/nodata, мин. площадь 2 га, сегментация, форма.

Запуск: python3 train_field_model.py   # сначала обучить
        python3 predict_fields.py
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape
from skimage.segmentation import felzenszwalb

from features import FEATURE_COLS, S2_BAND_INDEX, build_raster_feature_stack
from labels import is_white_field

BASE_DIR = Path(__file__).resolve().parent
S2_RASTER = BASE_DIR / "Sentinel-2L2A.tiff"
S1_RASTER = BASE_DIR / "Sentinel-1.tiff"
MODEL_PATH = BASE_DIR / "field_rf_model.joblib"
META_PATH = BASE_DIR / "field_model_meta.json"
OUTPUT_MASK = BASE_DIR / "field_mask.tif"
OUTPUT_GPKG = BASE_DIR / "fields.gpkg"

MIN_AREA_M2 = 20_000.0
MAX_AREA_HA = 500.0
PROBA_THRESHOLD = 0.5
ROW_BLOCK = 256
FELZ_SCALE = 70
FELZ_MIN_SIZE = 200
MIN_COMPACTNESS = 0.18
MAX_ASPECT_RATIO = 7.0
SIMPLIFY_TOL_M = 8.0


def valid_pixel_mask(red: np.ndarray, nir: np.ndarray, data_mask: np.ndarray | None, scl: np.ndarray) -> np.ndarray:
    m = (red > 0) & (nir > 0)
    if data_mask is not None:
        m &= data_mask == 65535
    m &= scl != 6
    m &= scl != 2
    return m


def predict_proba_raster(model, height: int, width: int, s2_arrays: dict, vv: np.ndarray, vh: np.ndarray, valid: np.ndarray) -> np.ndarray:
    proba = np.zeros((height, width), dtype=np.float32)
    for row0 in range(0, height, ROW_BLOCK):
        row1 = min(row0 + ROW_BLOCK, height)
        block_valid = valid[row0:row1]
        if not block_valid.any():
            continue
        s2_block = {k: arr[row0:row1] for k, arr in s2_arrays.items()}
        X = build_raster_feature_stack(s2_block, vv[row0:row1], vh[row0:row1])
        p = model.predict_proba(X[block_valid.reshape(-1)])[:, 1]
        block = np.zeros(block_valid.shape, dtype=np.float32)
        block.reshape(-1)[block_valid.reshape(-1)] = p
        proba[row0:row1] = block
    return proba


def pixel_area_m2(transform) -> float:
    from rasterio.transform import xy

    _, lat = xy(transform, 0, 0, offset="center")
    lat_rad = np.deg2rad(lat)
    return abs(transform.a) * 111_320 * np.cos(lat_rad) * abs(transform.e) * 110_540


def build_parcel_mask(field_binary: np.ndarray, ndvi: np.ndarray, transform) -> np.ndarray:
    px_area = pixel_area_m2(transform)
    min_pixels = max(30, int(MIN_AREA_M2 / px_area))

    struct = np.ones((3, 3), dtype=bool)
    binary = ndimage.binary_opening(field_binary, structure=struct)
    binary = ndimage.binary_closing(binary, structure=struct)

    labeled, n = ndimage.label(binary)
    if n > 0:
        counts = np.bincount(labeled.ravel())
        remove = counts < min_pixels
        remove[0] = False
        binary = binary & ~remove[labeled]

    if not binary.any():
        return np.zeros_like(ndvi, dtype=np.uint16)

    seg_in = ndvi.copy()
    seg_in[~binary] = 0
    labels = felzenszwalb(seg_in, scale=FELZ_SCALE, sigma=0.8, min_size=FELZ_MIN_SIZE)
    labels[~binary] = 0

    parcel = np.zeros_like(labels, dtype=np.uint16)
    pid = 1
    max_px = int(MAX_AREA_HA * 10_000 / px_area)
    for sid in np.unique(labels):
        if sid == 0:
            continue
        m = labels == sid
        if m.sum() < min_pixels or m.sum() > max_px:
            continue
        parcel[m] = pid
        pid += 1
    return parcel


def filter_shapes(gdf_m: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    keep = []
    for geom in gdf_m.geometry:
        if geom is None or geom.is_empty:
            keep.append(False)
            continue
        a, p = geom.area, geom.length
        comp = 4 * np.pi * a / (p * p + 1e-6)
        b = geom.bounds
        ar = max((b[2] - b[0]) / max(b[3] - b[1], 1e-6), (b[3] - b[1]) / max(b[2] - b[0], 1e-6))
        keep.append(comp >= MIN_COMPACTNESS and ar <= MAX_ASPECT_RATIO)
    return gdf_m[np.array(keep)].copy()


def vectorize(parcel: np.ndarray, transform, crs) -> gpd.GeoDataFrame:
    feats = []
    for geom_dict, val in shapes(parcel.astype(np.int32), mask=parcel > 0, transform=transform):
        if int(val) == 0:
            continue
        g = shape(geom_dict)
        if g.is_valid and not g.is_empty:
            feats.append({"geometry": g})
    if not feats:
        return gpd.GeoDataFrame(columns=["field_id"], geometry=[], crs=crs)

    gdf = gpd.GeoDataFrame(feats, crs=crs or CRS.from_epsg(4326))
    gm = gdf.to_crs(epsg=3857)
    gm["area_m2"] = gm.geometry.area
    gm = gm[gm["area_m2"] >= MIN_AREA_M2].copy()
    gm = filter_shapes(gm)
    gm["geometry"] = gm.geometry.simplify(SIMPLIFY_TOL_M, preserve_topology=True)
    gm["area_ha"] = gm["area_m2"] / 10_000
    gm = gm.sort_values("area_ha", ascending=False).reset_index(drop=True)
    gm["field_id"] = np.arange(1, len(gm) + 1)
    return gm.to_crs(crs or CRS.from_epsg(4326))


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Сначала: python3 train_field_model.py")

    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    thresh = float(meta.get("proba_threshold", PROBA_THRESHOLD))
    print("Модель:", meta.get("note", meta.get("labeling", "")))
    print(f"Порог вероятности: {thresh}")

    model = joblib.load(MODEL_PATH)

    with rasterio.open(S2_RASTER) as s2, rasterio.open(S1_RASTER) as s1:
        if s2.shape != s1.shape:
            raise ValueError("S2 и S1 — разный размер")

        h, w = s2.shape
        crs = s2.crs or CRS.from_epsg(4326)
        transform = s2.transform

        s2_arrays = {c: s2.read(S2_BAND_INDEX[c]).astype(np.float32) for c in FEATURE_COLS if c.startswith("S2_")}
        vv = s1.read(1).astype(np.float32)
        vh = s1.read(2).astype(np.float32)
        vv[vv <= 0] = 0
        vh[vh <= 0] = 0

        red, nir = s2_arrays["S2_B04"], s2_arrays["S2_B08"]
        dm = s2.read(17)
        scl = s2.read(16)
        valid = valid_pixel_mask(red, nir, dm, scl)

        ndvi = np.zeros_like(red)
        d = red + nir
        m = d > 0
        ndvi[m] = (nir[m] - red[m]) / d[m]

        print(f"Валидных пикселей: {valid.sum():,} ({100*valid.mean():.1f}%)")
        print("Предсказание модели...")
        proba = predict_proba_raster(model, h, w, s2_arrays, vv, vh, valid)

        swir = s2_arrays["S2_B11"]
        ndbi = np.zeros_like(red)
        d2 = nir + swir
        m2 = d2 > 0
        ndbi[m2] = (swir[m2] - nir[m2]) / d2[m2]

        field_bin = (proba >= thresh) & valid & ~is_white_field(ndvi, ndbi)
        print(f"Пикселей «поле» по модели: {field_bin.sum():,} ({100*field_bin.sum()/valid.sum():.1f}% от valid)")

        print("Сегментация участков...")
        parcel = build_parcel_mask(field_bin, ndvi, transform)
        print(f"   участков: {parcel.max()}")

        profile = s2.profile.copy()
        profile.update(dtype=rasterio.uint16, count=1, nodata=0)
        with rasterio.open(OUTPUT_MASK, "w", **profile) as dst:
            dst.write(parcel, 1)

        gdf = vectorize(parcel, transform, crs)
        if gdf.empty:
            print("Полигоны не найдены.")
            return

        gdf.to_file(OUTPUT_GPKG, layer="fields", driver="GPKG")
        print(f"Маска: {OUTPUT_MASK}")
        print(f"Поля:  {OUTPUT_GPKG} — {len(gdf)} полигонов, {gdf['area_ha'].sum():.0f} га")


if __name__ == "__main__":
    main()
