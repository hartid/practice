import pandas as pd
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
from sklearn.ensemble import RandomForestClassifier
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.filters import sobel
import os

print("=== СТАРТ ИНТЕЛЛЕКТУАЛЬНОГО РАЗДЕЛЕНИЯ ПОЛЕЙ (ВЕРСИЯ 4.0) ===")

csv_path = "dataset.csv"
tiff_path = "Sentinel-2L2A.tiff"
output_gpkg = "field_boundaries.gpkg"

if not os.path.exists(csv_path) or not os.path.exists(tiff_path):
    raise FileNotFoundError("Убедись, что файлы dataset.csv и Sentinel-2L2A.tiff находятся в папке скрипта!")

# 1. ОБУЧЕНИЕ МОДЕЛИ
print("Шаг 1: Обучение базового классификатора на сигнатурах...")
df = pd.read_csv(csv_path, nrows=300000)
features = ['S2_B02', 'S2_B03', 'S2_B04', 'S2_B08']
df = df.dropna(subset=features + ['class'])

X_train = df[features].values
y_train = df['class'].values

rf = RandomForestClassifier(n_estimators=40, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print("Модель натренирована.")

# 2. ЧТЕНИЕ КАНАЛОВ
print("\nШаг 2: Чтение спектральных каналов...")
with rasterio.open(tiff_path) as src:
    transform = src.transform
    crs = src.crs
    img_shape = (src.height, src.width)
    
    b02 = src.read(2).astype(np.float32)
    b03 = src.read(3).astype(np.float32)
    b04 = src.read(4).astype(np.float32)
    b08 = src.read(8).astype(np.float32)

# Подготовка матрицы для классификации
X_img = np.stack([b02.ravel(), b03.ravel(), b04.ravel(), b08.ravel()], axis=1)
X_img = np.nan_to_num(X_img, nan=0.0)

# 3. КЛАССИФИКАЦИЯ
print("\nШаг 3: Первичная пиксельная классификация...")
preds = rf.predict(X_img)

unique_classes = np.unique(preds)
class_to_id = {cls: idx for idx, cls in enumerate(unique_classes)}
id_to_class = {idx: cls for cls, idx in class_to_id.items()}

preds_id = np.zeros(preds.shape, dtype=np.int32)
for cls_name, cls_id in class_to_id.items():
    preds_id[preds == cls_name] = cls_id
preds_id = preds_id.reshape(img_shape)

# 4. ВЫДЕЛЕНИЕ ГРАНИЦ И РАЗРЕЗАНИЕ БЛОКОВ (МЕТОД СОБЕЛЯ)
print("\nШаг 4: Поиск резких переходов и разрезание смежных полей...")
# Нормализуем инфракрасный канал для корректного поиска перепадов яркости
b08_norm = (b08 - np.min(b08)) / (np.max(b08) - np.min(b08) + 1e-5)
# Фильтр Собеля находит дороги, межи и лесополосы
edges = sobel(b08_norm)

# Выделяем топ-12% самых резких границ (линии дорог и контуры разделения)
edge_threshold = np.percentile(edges, 88)
edge_mask = edges > edge_threshold

clean_preds_id = np.full_like(preds_id, -1)

for cls_name, cls_id in class_to_id.items():
    if cls_name in ['vegetation', 'bare_soil']:
        mask = (preds_id == cls_id)
        
        # КРИТИЧЕСКИЙ ШАГ: Стираем пиксели дорог, превращая монолит в изолированные поля
        mask[edge_mask] = False
        
        # Морфологическая очистка изолированных объектов
        mask = remove_small_objects(mask, min_size=200)  # убираем точечный шум
        mask = remove_small_holes(mask, area_threshold=200)  # латаем дыры внутри полей
        
        clean_preds_id[mask] = cls_id

# 5. ВЕКТОРИЗАЦИЯ ОДНОРОДНЫХ ПОЛЕЙ
print("\nШаг 5: Векторизация разделенных контуров полей...")
generator = shapes(clean_preds_id, mask=(clean_preds_id != -1), transform=transform)

records = []
for geometry, value in generator:
    val_int = int(value)
    if val_int in id_to_class:
        poly = shape(geometry)
        records.append({
            "geometry": poly,
            "land_cover": id_to_class[val_int]
        })

print(f"Успешно выделено уникальных изолированных полей: {len(records)}")

# СОХРАНЕНИЕ С КОРРЕСПОНДИРУЮЩЕЙ СИСТЕМОЙ КООРДИНАТ
if len(records) > 0:
    gdf = gpd.GeoDataFrame(records, columns=["geometry", "land_cover"], geometry="geometry")
    
    # Явно передаем CRS, чтобы убрать Warning в pyogrio на твоем Mac
    if crs is not None:
        gdf.crs = crs
    else:
        # Если в TIFF нет CRS, ставим стандартную географическую WGS84 (совпадает с lat/lon из CSV)
        gdf.crs = "EPSG:4326"
        
    gdf.to_file(output_gpkg, layer="detected_fields", driver="GPKG")
    print(f"\n=== КУРСОВАЯ СПАСЕНА! Файл {output_gpkg} успешно обновлен ===")
else:
    print("\nОшибка сегментации. Проверь структуру входных матриц.")