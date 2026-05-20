"""
Разметка «сельхозполе / не поле» для обучения.
Правила согласованы с тем, что видно на True Color (зелёные, розовые, фиолетовые = поле;
лес, город, белое = не поле).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- пороги (единый источник для обучения) ---
FOREST_NDVI = 0.72
FOREST_NDMI = 0.08
NDVI_AGRI_MAX = 0.71

PINK_NDVI = (0.14, 0.54)
PINK_NDBI = (0.17, 0.43)
PURPLE_NDVI = (0.10, 0.48)
PURPLE_NDBI_MAX = 0.17

WHITE_NDVI_MAX = 0.20
WHITE_NDBI_MIN = 0.14

URBAN_RULES = ((0.07, 0.24), (0.28, 0.36))  # (ndbi_min, ndvi_max), ...


def compute_ndbi(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    d = nir + swir
    return np.where(d > 0, (swir - nir) / d, 0.0).astype(np.float32)


def is_forest(ndvi: np.ndarray, ndmi: np.ndarray) -> np.ndarray:
    return ((ndvi >= FOREST_NDVI) & (ndmi >= FOREST_NDMI)) | (ndvi >= 0.76)


def is_urban(ndbi: np.ndarray, ndvi: np.ndarray) -> np.ndarray:
    u = np.zeros_like(ndvi, dtype=bool)
    for ndbi_min, ndvi_max in URBAN_RULES:
        u |= (ndbi >= ndbi_min) & (ndvi < ndvi_max)
    return u


def is_white_field(ndvi: np.ndarray, ndbi: np.ndarray) -> np.ndarray:
    """Белые области — не поле."""
    return (ndvi <= WHITE_NDVI_MAX) & (ndbi >= WHITE_NDBI_MIN)


def is_pink_field(ndvi: np.ndarray, ndbi: np.ndarray) -> np.ndarray:
    return (
        (ndvi >= PINK_NDVI[0])
        & (ndvi <= PINK_NDVI[1])
        & (ndbi >= PINK_NDBI[0])
        & (ndbi <= PINK_NDBI[1])
    )


def is_purple_field(ndvi: np.ndarray, ndbi: np.ndarray) -> np.ndarray:
    return (
        (ndvi >= PURPLE_NDVI[0])
        & (ndvi <= PURPLE_NDVI[1])
        & (ndbi < PURPLE_NDBI_MAX)
        & (ndbi < PINK_NDBI[0])
    )


def assign_field_labels_df(df: pd.DataFrame) -> np.ndarray:
    """
    Векторная разметка для DataFrame с колонками NDVI, NDMI, S2_B04, S2_B08, S2_B11, class.
    Возвращает is_field: 1 = сельхозполе, 0 = не поле.
    """
    ndvi = df["NDVI"].to_numpy(dtype=np.float32)
    ndmi = df["NDMI"].to_numpy(dtype=np.float32)
    nir = df["S2_B08"].to_numpy(dtype=np.float32)
    swir = df["S2_B11"].to_numpy(dtype=np.float32)
    ndbi = compute_ndbi(nir, swir)
    orig = df["class"].to_numpy()

    forest = is_forest(ndvi, ndmi)
    urban = is_urban(ndbi, ndvi)
    white = is_white_field(ndvi, ndbi)

    pink = is_pink_field(ndvi, ndbi)
    purple = is_purple_field(ndvi, ndbi)

    green_veg = (orig == "vegetation") & (ndvi >= 0.08) & (ndvi <= NDVI_AGRI_MAX)
    green_bare = (orig == "bare_soil") & (ndvi >= 0.08) & (ndvi <= 0.58) & (ndbi < PURPLE_NDBI_MAX)

    field = (pink | purple | green_veg | green_bare) & ~forest & ~urban & ~white
    return field.astype(np.int8)


def assign_field_labels_arrays(
    ndvi: np.ndarray,
    ndmi: np.ndarray,
    ndbi: np.ndarray,
    *,
    is_vegetation: np.ndarray | None = None,
) -> np.ndarray:
    """Разметка для растров (is_vegetation ≈ SCL==4, опционально)."""
    if is_vegetation is None:
        is_vegetation = ndvi >= 0.08

    forest = is_forest(ndvi, ndmi)
    urban = is_urban(ndbi, ndvi)
    white = is_white_field(ndvi, ndbi)
    pink = is_pink_field(ndvi, ndbi)
    purple = is_purple_field(ndvi, ndbi)
    green = is_vegetation & (ndvi >= 0.08) & (ndvi <= NDVI_AGRI_MAX)

    field = (pink | purple | green) & ~forest & ~urban & ~white
    return field.astype(np.uint8)
