import os
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops
from rasterio.crs import CRS
import warnings
warnings.filterwarnings('ignore')

# ================= НАСТРОЙКИ =================
S2_RASTER = 'Sentinel-2L2A.tiff'   # Оптика (обязательно)
S1_RASTER = 'Sentinel-1.tiff'        # SAR (опционально)
OUTPUT_GEOJSON = 'fields_with_sar.geojson'

MIN_AREA_M2 = 5000.0  # Мин. площадь объекта (0.5 га)
GLCM_DIST = 2         # Дистанция для текстур
GLCM_LEVELS = 8       # Уровни квантизации для GLCM
# =============================================

def load_and_preprocess_sar(raster_path, target_shape, target_transform):
    """Загрузка и предобработка Sentinel-1: конвертация в dB, ресемплинг"""
    print(f"   📡 Загрузка SAR: {raster_path}")
    
    with rasterio.open(raster_path) as src:
        # Sentinel-1 обычно: band 1 = VV, band 2 = VH
        if src.count < 2:
            print("   ⚠️ Sentinel-1 должен иметь минимум 2 банды (VV, VH)")
            return None, None
        
        # Читаем и конвертируем в dB: 10*log10(linear)
        vv_linear = src.read(1).astype(np.float32)
        vh_linear = src.read(2).astype(np.float32)
        
        # Замена нулей/отрицательных на NaN
        vv_linear[vv_linear <= 0] = np.nan
        vh_linear[vh_linear <= 0] = np.nan
        
        # Конвертация в децибелы
        vv_db = 10 * np.log10(vv_linear)
        vh_db = 10 * np.log10(vh_linear)
        
        # Ресемплинг под размер Sentinel-2 (если нужно)
        if src.shape != target_shape:
            from rasterio.enums import Resampling
            vv_db = rasterio.warp.reproject(
                source=vv_db,
                destination=np.zeros(target_shape),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=src.crs,  # Предполагаем одинаковую CRS
                resampling=Resampling.bilinear
            )[0]
            vh_db = rasterio.warp.reproject(
                source=vh_db,
                destination=np.zeros(target_shape),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=src.crs,
                resampling=Resampling.bilinear
            )[0]
        
        # Индекс поляризации (разность в dB = отношение в линейной шкале)
        ratio = vh_db - vv_db
        
        return vv_db, vh_db, ratio


def extract_sar_texture(band_db, distances=[2], levels=8):
    """GLCM текстуры для SAR банды (исправлено квантование)"""
    # 1. Нормализация и квантование до [0, levels-1]
    band_min, band_max = np.nanpercentile(band_db, [2, 98])
    band_norm = np.clip(
        (band_db - band_min) / (band_max - band_min + 1e-10) * (levels - 1), 
        0, levels - 1
    ).astype(np.uint8)
    
    # 2. Заполняем нули (бывшие NaN/края) средним значением, чтобы GLCM не падал
    mean_val = band_norm[band_norm > 0].mean() if (band_norm > 0).any() else 0
    band_norm[band_norm == 0] = int(mean_val)
    
    # 3. Вычисляем GLCM
    glcm = graycomatrix(
        band_norm, 
        distances=distances, 
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=levels,
        symmetric=True, 
        normed=True
    )
    
    # 4. Извлекаем свойства
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    return {
        'sar_contrast': contrast,
        'sar_homogeneity': homogeneity,
        'sar_energy': energy,
        'sar_correlation': correlation
    }

def main():
    if not os.path.exists(S2_RASTER):
        print(f"❌ Файл '{S2_RASTER}' не найден.")
        return

    print("1️⃣ Загрузка Sentinel-2...")
    with rasterio.open(S2_RASTER) as src_s2:
        bands_s2 = src_s2.read()
        transform_s2 = src_s2.transform
        shape_s2 = src_s2.shape
        crs_s2 = src_s2.crs or CRS.from_epsg(4326)
        
        # Каналы Sentinel-2
        b_red = bands_s2[3].astype(np.float32)    # Band 4
        b_nir = bands_s2[7].astype(np.float32)    # Band 8
        b_green = bands_s2[2].astype(np.float32)  # Band 3
        
        print(f"   Размер: {shape_s2[1]}x{shape_s2[0]} | CRS: {crs_s2}")
        
        # Маска валидных пикселей
        valid_mask = (b_red > 0) & (b_nir > 0)
        
        print("2️⃣ Вычисление оптических признаков...")
        ndvi = np.zeros_like(b_red)
        ndvi[valid_mask] = (b_nir[valid_mask] - b_red[valid_mask]) / (b_nir[valid_mask] + b_red[valid_mask] + 1e-10)
        
        # Загружаем Sentinel-1, если есть
        sar_features = {}
        if os.path.exists(S1_RASTER):
            print("3️⃣ Обработка Sentinel-1 (SAR)...")
            vv_db, vh_db, ratio = load_and_preprocess_sar(S1_RASTER, shape_s2, transform_s2)
            
            if vv_db is not None:
                # SAR признаки для каждого пикселя
                sar_features['vv_db'] = vv_db
                sar_features['vh_db'] = vh_db
                sar_features['vh_vv_ratio'] = ratio
                
                # Текстуры SAR (на основе VH, так как он лучше для растительности)
                print("   🌀 Вычисление SAR текстур (это может занять время)...")
                sar_textures = extract_sar_texture(vh_db, distances=GLCM_DIST, levels=GLCM_LEVELS)
                
                # Создаём растровые слои текстур (одинаковые для всех пикселей — упрощение)
                # Для полноценной версии нужно считать текстуру в скользящем окне
                for name, value in sar_textures.items():
                    sar_features[name] = np.full_like(b_red, value, dtype=np.float32)
                
                print(f"   ✅ SAR признаки добавлены: {list(sar_features.keys())}")
        else:
            print(f"   ⚠️ Файл '{S1_RASTER}' не найден, работаем только с оптикой")
        
        # === СЕГМЕНТАЦИЯ ===
        print("4️⃣ Сегментация с использованием всех признаков...")
        
        # Формируем многомерный "признаковый куб"
        # Для простоты используем комбинацию порогов по разным признакам
        seg_mask = np.zeros_like(ndvi, dtype=np.uint8)
        
        # Пороги (можно настроить)
        NDVI_LOW, NDVI_HIGH = np.percentile(ndvi[valid_mask], [33, 66])
        
        if sar_features:
            # Если есть SAR, добавляем условия на VH и ratio
            VH_MED = np.nanmedian(sar_features['vh_db'][valid_mask])
            RATIO_MED = np.nanmedian(sar_features['vh_vv_ratio'][valid_mask])
            
            # Класс 1: Низкая растительность + низкий VH (почва/застройка)
            cond1 = (ndvi <= NDVI_LOW) & (sar_features['vh_db'] <= VH_MED) & valid_mask
            seg_mask[cond1] = 1
            
            # Класс 2: Средняя растительность (посевы)
            cond2 = (ndvi > NDVI_LOW) & (ndvi <= NDVI_HIGH) & valid_mask
            seg_mask[cond2] = 2
            
            # Класс 3: Высокая растительность + высокий VH/VV (лес/густая трава)
            cond3 = (ndvi > NDVI_HIGH) & (sar_features['vh_vv_ratio'] > RATIO_MED) & valid_mask
            seg_mask[cond3] = 3
            
            # Если не попало ни в один класс — по умолчанию класс 2
            seg_mask[(seg_mask == 0) & valid_mask] = 2
        else:
            # Только оптика (старая логика)
            seg_mask[(ndvi <= NDVI_LOW) & valid_mask] = 1
            seg_mask[(ndvi > NDVI_LOW) & (ndvi <= NDVI_HIGH) & valid_mask] = 2
            seg_mask[(ndvi > NDVI_HIGH) & valid_mask] = 3
        
        print("5️⃣ Векторизация...")
        features = []
        
        for class_val in [1, 2, 3]:
            binary = (seg_mask == class_val).astype(np.uint8)
            clean_binary = ndimage.binary_opening(binary, structure=np.ones((3,3))).astype(np.uint8)
            
            print(f"   Класс {class_val}...")
            count = 0
            
            for geom_dict, val in shapes(clean_binary, mask=clean_binary, transform=transform_s2):
                try:
                    geom = shape(geom_dict)
                    if geom.is_valid and not geom.is_empty:
                        features.append({
                            'properties': {'class_id': int(class_val)},
                            'geometry': geom.__geo_interface__
                        })
                        count += 1
                except:
                    continue
            print(f"   → {count} полигонов")
        
        if not features:
            print("⚠️ Полигоны не найдены.")
            return
        
        print("6️⃣ Фильтрация и сохранение...")
        gdf = gpd.GeoDataFrame.from_features(features, crs=crs_s2)
        
        # Площадь в метрах
        gdf_metric = gdf.to_crs(epsg=3857)
        gdf_metric['area_m2'] = gdf_metric.geometry.area
        gdf_filtered = gdf_metric[gdf_metric['area_m2'] >= MIN_AREA_M2].copy()
        
        # Упрощение
        gdf_filtered['geometry'] = gdf_filtered.geometry.simplify(tolerance=10.0, preserve_topology=True)
        
        # Возврат в WGS84
        gdf_final = gdf_filtered.to_crs(epsg=4326)
        
        # Названия классов
        names = {1: 'Soil_Bare', 2: 'Crops_Medium', 3: 'Forest_Dense'}
        gdf_final['class_name'] = gdf_final['class_id'].map(names)
        
        # Добавляем информацию об использовании SAR
        gdf_final.attrs['sar_used'] = os.path.exists(S1_RASTER)
        gdf_final.attrs['features'] = ['NDVI', 'VV_dB', 'VH_dB', 'VH/VV_ratio'] if sar_features else ['NDVI']
        
        gdf_final.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
        print(f"✅ Готово! Файл: {OUTPUT_GEOJSON}")
        print(f"📊 Итого: {len(gdf_final)} полигонов")
        print(f"📡 SAR использован: {os.path.exists(S1_RASTER)}")

if __name__ == "__main__":
    main()