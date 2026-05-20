"""Общие признаки для обучения (CSV) и предсказания (растры)."""
from __future__ import annotations

import numpy as np

# Спектральные каналы Sentinel-2 (как в dataset.csv)
S2_FEATURE_COLS = [f"S2_B{band:02d}" for band in [1, 2, 3, 4, 5, 6, 7, 8]] + ["S2_B8A", "S2_B09", "S2_B11", "S2_B12"]

# SAR на дату снимка (совпадает с колонками в dataset.csv)
S1_VV_COL = "S1_VV_20250610"
S1_VH_COL = "S1_VH_20250610"
S1_FEATURE_COLS = [S1_VV_COL, S1_VH_COL]

DERIVED_COLS = ["NDVI", "NDMI", "NDBI", "VH_over_VV"]

FEATURE_COLS = S2_FEATURE_COLS + S1_FEATURE_COLS + DERIVED_COLS

# Индексы полос в Sentinel-2L2A.tiff (1-based в rasterio.read)
S2_BAND_INDEX = {
    "S2_B01": 1,
    "S2_B02": 2,
    "S2_B03": 3,
    "S2_B04": 4,
    "S2_B05": 5,
    "S2_B06": 6,
    "S2_B07": 7,
    "S2_B08": 8,
    "S2_B8A": 9,
    "S2_B09": 10,
    "S2_B11": 11,
    "S2_B12": 12,
}

LABEL_TO_FIELD = {
    "vegetation": 1,   # посевы / растительность внутри полей
    "bare_soil": 0,    # открытая почва, дороги, межи
}


def add_derived_features(df_or_dict) -> None:
    """Добавляет NDVI, NDMI, VH/VV к pandas DataFrame in-place."""
    red = df_or_dict["S2_B04"].astype(np.float32)
    nir = df_or_dict["S2_B08"].astype(np.float32)
    swir = df_or_dict["S2_B11"].astype(np.float32)
    vv = df_or_dict[S1_VV_COL].astype(np.float32)
    vh = df_or_dict[S1_VH_COL].astype(np.float32)

    denom = nir + red
    df_or_dict["NDVI"] = np.where(denom > 0, (nir - red) / denom, 0.0)

    denom_m = nir + swir
    df_or_dict["NDMI"] = np.where(denom_m > 0, (nir - swir) / denom_m, 0.0)

    df_or_dict["NDBI"] = np.where(denom_m > 0, (swir - nir) / denom_m, 0.0)

    df_or_dict["VH_over_VV"] = np.where(vv > 0, vh / vv, 0.0)


def build_raster_feature_stack(s2_bands: dict[str, np.ndarray], vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """
    Собирает матрицу признаков shape (n_pixels, n_features) для валидных пикселей.
    s2_bands: имя колонки -> 2D array
    """
    red = s2_bands["S2_B04"].astype(np.float32)
    nir = s2_bands["S2_B08"].astype(np.float32)
    swir = s2_bands["S2_B11"].astype(np.float32)

    ndvi = np.zeros_like(red, dtype=np.float32)
    ndmi = np.zeros_like(red, dtype=np.float32)
    d = nir + red
    m = d > 0
    ndvi[m] = (nir[m] - red[m]) / d[m]
    d2 = nir + swir
    m2 = d2 > 0
    ndmi[m2] = (nir[m2] - swir[m2]) / d2[m2]

    ndbi = np.zeros_like(red, dtype=np.float32)
    ndbi[m2] = (swir[m2] - nir[m2]) / d2[m2]

    ratio = np.zeros_like(red, dtype=np.float32)
    m3 = vv > 0
    ratio[m3] = vh[m3] / vv[m3]

    planes = []
    for col in S2_FEATURE_COLS:
        planes.append(s2_bands[col].reshape(-1))
    planes.append(vv.reshape(-1))
    planes.append(vh.reshape(-1))
    planes.append(ndvi.reshape(-1))
    planes.append(ndmi.reshape(-1))
    planes.append(ndbi.reshape(-1))
    planes.append(ratio.reshape(-1))
    return np.column_stack(planes).astype(np.float32)
