# Выделение границ полей (вариант 1, путь 2)

## Идея

1. **Разметка** (`labels.py`) — по спектру назначаем `is_field`: зелёные, розовые, фиолетовые поля = 1; лес, город, белое = 0.
2. **Обучение** (`train_field_model.py`) — Random Forest учится на `dataset.csv` с этой разметкой.
3. **Предсказание** (`predict_fields.py`) — модель сама решает по пикселю; только мин. площадь 2 га и сегментация.

## Запуск

```bash
pip install -r requirements.txt

# 1. Обучение (~1–2 мин)
python3 train_field_model.py

# 2. Карта полей
python3 predict_fields.py
```

Результат: `fields.gpkg`, `field_mask.tif` — открыть в QGIS с `TRUE_COLOR.tif`.

## Файлы

| Файл | Роль |
|------|------|
| `labels.py` | Правила разметки для обучения |
| `features.py` | Признаки S2, S1, NDVI, NDMI, NDBI |
| `field_rf_model.joblib` | Обученная модель |
| `field_model_meta.json` | Метрики и описание |

## Переобучение

После изменения порогов в `labels.py` снова запустите `train_field_model.py`, затем `predict_fields.py`.
