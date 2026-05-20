#!/usr/bin/env python3
"""
Обучение Random Forest: целевой класс «сельхозполе» (не сырой vegetation/bare_soil).

Разметка строится по спектральным правилам в labels.py (зелёные, розовые,
фиолетовые = поле; лес, город, белое = не поле), затем модель учится
обобщать на новых пикселях.

Запуск: python3 train_field_model.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from features import FEATURE_COLS, S1_FEATURE_COLS, S2_FEATURE_COLS, add_derived_features
from labels import assign_field_labels_df

BASE_DIR = Path(__file__).resolve().parent
DATASET_CSV = BASE_DIR / "dataset.csv"
MODEL_PATH = BASE_DIR / "field_rf_model.joblib"
META_PATH = BASE_DIR / "field_model_meta.json"

MAX_ROWS = 2_000_000
CHUNK_SIZE = 250_000
TEST_SIZE = 0.2
RANDOM_STATE = 42
PROBA_THRESHOLD = 0.5


def load_labeled_sample() -> pd.DataFrame:
    usecols = ["class"] + S2_FEATURE_COLS + S1_FEATURE_COLS
    parts: list[pd.DataFrame] = []
    rows_read = 0

    print(f"Чтение до {MAX_ROWS:,} строк из {DATASET_CSV.name}...")
    for chunk in pd.read_csv(DATASET_CSV, usecols=usecols, chunksize=CHUNK_SIZE):
        chunk = chunk[chunk["class"].isin(["vegetation", "bare_soil"])]
        if chunk.empty:
            continue
        add_derived_features(chunk)
        chunk["is_field"] = assign_field_labels_df(chunk)
        parts.append(chunk)
        rows_read += len(chunk)
        if rows_read % 500_000 == 0 or rows_read >= MAX_ROWS:
            print(f"  строк: {rows_read:,}...")
        if rows_read >= MAX_ROWS:
            break

    df = pd.concat(parts, ignore_index=True)
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=RANDOM_STATE)

    n_field = int(df["is_field"].sum())
    n_non = len(df) - n_field
    print(f"Разметка: поле={n_field:,} ({100*n_field/len(df):.1f}%), не поле={n_non:,}")

    # Баланс 50/50 для обучения
    field_df = df[df["is_field"] == 1]
    non_df = df[df["is_field"] == 0]
    n_each = min(len(field_df), len(non_df), MAX_ROWS // 2)
    balanced = pd.concat(
        [
            field_df.sample(n_each, random_state=RANDOM_STATE),
            non_df.sample(n_each, random_state=RANDOM_STATE),
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=RANDOM_STATE)

    print(f"Выборка для RF: {len(balanced):,} (поле {n_each:,} / не поле {n_each:,})")
    return balanced


def main() -> None:
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"Нет файла {DATASET_CSV}")

    df = load_labeled_sample()
    y = df["is_field"].astype(np.int8)
    X = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("\nОбучение Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=28,
        min_samples_leaf=4,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== Отчёт (порог 0.5) ===")
    print(classification_report(y_test, y_pred, target_names=["not_field", "field"]))

    y_pred_t = (y_proba >= PROBA_THRESHOLD).astype(int)
    print(f"=== С порогом proba>={PROBA_THRESHOLD} ===")
    print(classification_report(y_test, y_pred_t, target_names=["not_field", "field"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_t))

    imp = sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1])[:12]
    print("\nТоп признаков:")
    for name, val in imp:
        print(f"  {name}: {val:.4f}")

    joblib.dump(model, MODEL_PATH)
    meta = {
        "feature_cols": FEATURE_COLS,
        "target": "is_field",
        "target_names": {"0": "not_field", "1": "field"},
        "labeling": "labels.assign_field_labels_df (green, pink, purple field; exclude forest, urban, white)",
        "proba_threshold": PROBA_THRESHOLD,
        "train_rows": int(len(df)),
        "test_accuracy": float((y_pred_t == y_test).mean()),
        "note": "Модель обучена на разметке «сельхозполе», не на сыром vegetation/bare_soil",
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nСохранено: {MODEL_PATH}")
    print(f"Метаданные: {META_PATH}")
    print("\nДалее: python3 predict_fields.py")


if __name__ == "__main__":
    main()
