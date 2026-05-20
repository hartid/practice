import os
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
from scipy import ndimage
from rasterio.crs import CRS

INPUT_RASTER = 'Sentinel-2L2A.tiff'
OUTPUT_GEOJSON = 'fields_segmented.geojson'

# Минимальная площадь объекта (0.5 гектара), чтобы отсеять шум
MIN_AREA_M2 = 5000.0 

def main():
    if not os.path.exists(INPUT_RASTER):
        print(f"❌ Файл '{INPUT_RASTER}' не найден.")
        return

    print("1️⃣ Загрузка растра...")
    with rasterio.open(INPUT_RASTER) as src:
        # Sentinel-2 L2A: B2(1), B3(2), B4(3), B8(7)
        bands = src.read()
        b_red = bands[3].astype(np.float32)    # Band 4 (Red)
        b_nir = bands[7].astype(np.float32)    # Band 8 (NIR)
        b_green = bands[2].astype(np.float32)  # Band 3 (Green) - для справки
        
        print(f"   Размер: {src.width}x{src.height}")
        
        # Маска валидных пикселей (убираем нули и края)
        valid_mask = (b_red > 0) & (b_nir > 0)
        
        print("2️⃣ Вычисление NDVI...")
        ndvi = np.zeros_like(b_red)
        ndvi[valid_mask] = (b_nir[valid_mask] - b_red[valid_mask]) / (b_nir[valid_mask] + b_red[valid_mask])
        
        # Получаем значения NDVI только для валидных пикселей
        valid_values = ndvi[valid_mask]
        
        if len(valid_values) == 0:
            print("❌ Нет валидных данных.")
            return

        print("3️ Автоматический подбор порогов (Percentiles)...")
        # Делим все значения на 3 равные группы (Тёмные, Средние, Светлые)
        p33 = np.percentile(valid_values, 33)
        p66 = np.percentile(valid_values, 66)
        
        print(f"   Нижний порог (33%): {p33:.3f}")
        print(f"   Верхний порог (66%): {p66:.3f}")
        
        # Создаем карту классов (0, 1, 2, 3)
        seg_mask = np.zeros_like(ndvi, dtype=np.uint8)
        
        # Класс 1: Низкий NDVI (Светлые: почва, здания, сухая трава)
        cond_low = (ndvi <= p33) & valid_mask
        seg_mask[cond_low] = 1
        
        # Класс 2: Средний NDVI (Посевы, луга)
        cond_med = (ndvi > p33) & (ndvi <= p66) & valid_mask
        seg_mask[cond_med] = 2
        
        # Класс 3: Высокий NDVI (Тёмные: леса, здоровая зелень, вода*)
        cond_high = (ndvi > p66) & valid_mask
        seg_mask[cond_high] = 3
        
        print("4️⃣ Векторизация (без склеивания)...")
        features = []
        
        # Проходим по каждому классу отдельно
        for class_val in [1, 2, 3]:
            binary = (seg_mask == class_val).astype(np.uint8)
            
            # ТОЛЬКО открытие (убираем одиночные шумовые пиксели)
            # НЕ делаем closing (не склеиваем поля!)
            # Структура 3x3 пикселя
            clean_binary = ndimage.binary_opening(binary, structure=np.ones((3,3))).astype(np.uint8)
            
            print(f"   Обработка класса {class_val}...")
            count = 0
            
            for geom_dict, val in shapes(clean_binary, mask=clean_binary, transform=src.transform):
                try:
                    geom = shape(geom_dict)
                    if geom.is_valid and not geom.is_empty:
                        # Фильтр по площади (конвертируем в метры для проверки)
                        # Примечание: если CRS=None, считаем в градусах, но лучше проверить
                        # Для Sentinel-2 обычно CRS есть. Если нет, используем примерную конвертацию.
                        
                        # Простая проверка площади (если CRS в градусах, порог будет маленьким)
                        # Но лучше перепроецировать.
                        features.append({
                            'properties': {'class_id': int(class_val)},
                            'geometry': geom.__geo_interface__
                        })
                        count += 1
                except Exception:
                    continue
            
            print(f"   Класс {class_val}: найдено {count} объектов")

        if not features:
            print("⚠️ Полигоны не найдены.")
            return

        print("5️ Сохранение и фильтрация по площади...")
        # Создаем GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(features, crs=src.crs or CRS.from_epsg(4326))
        
        # Переводим в метры для точного расчета площади (Web Mercator)
        gdf_metric = gdf.to_crs(epsg=3857)
        gdf_metric['area_m2'] = gdf_metric.geometry.area
        
        # Оставляем только объекты >= 5000 м2 (0.5 га)
        gdf_filtered = gdf_metric[gdf_metric['area_m2'] >= MIN_AREA_M2].copy()
        
        print(f"   После фильтрации: {len(gdf_filtered)} полигонов")
        
        if len(gdf_filtered) == 0:
            print(f"️ Все объекты меньше {MIN_AREA_M2} м². Уменьшите MIN_AREA_M2.")
            return

        # Упрощаем геометрию для легкости файла
        gdf_filtered['geometry'] = gdf_filtered.geometry.simplify(tolerance=10.0, preserve_topology=True)
        
        # Возвращаем в WGS84 (стандарт для карт)
        gdf_final = gdf_filtered.to_crs(epsg=4326)
        
        # Названия классов
        names = {1: 'Low_NDVI_Soil', 2: 'Med_NDVI_Crops', 3: 'High_NDVI_Forest'}
        gdf_final['class_name'] = gdf_final['class_id'].map(names)
        
        # Сохраняем
        gdf_final.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
        print(f"✅ Готово! Файл: {OUTPUT_GEOJSON}")
        print(f"📊 Итого: {len(gdf_final)} полигонов.")
        print("   Откройте файл в QGIS или geojson.io. Вы увидите 'лоскутное одеяло'.")

if __name__ == "__main__":
    main()