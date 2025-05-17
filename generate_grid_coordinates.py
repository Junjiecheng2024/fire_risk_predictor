import numpy as np
import pandas as pd

# 区域边界（经度范围、纬度范围）
lon_min, lon_max = -8.5, -7.5
lat_min, lat_max = 39.5, 40.5

# 每多少米一个点（1000 米）
spacing_m = 1000

# 地球半径（用于粗略计算经纬度间隔）
earth_radius = 6371000  # meters

# 1° 纬度 ≈ 111 km，计算出对应的经纬度间隔（粗略换算）
lat_spacing_deg = spacing_m / 111000
lon_spacing_deg = spacing_m / (111000 * np.cos(np.deg2rad((lat_min + lat_max) / 2)))

# 构建网格坐标点
lat_points = np.arange(lat_min, lat_max, lat_spacing_deg)
lon_points = np.arange(lon_min, lon_max, lon_spacing_deg)

# 生成经纬度网格点
grid = pd.DataFrame([
    {'lat': lat, 'lon': lon}
    for lat in lat_points
    for lon in lon_points
])

# 保存为 CSV
grid.to_csv("data/grid_coordinates.csv", index=False)
print(f"✅ 已生成 grid_coordinates.csv，共 {len(grid)} 个点")
