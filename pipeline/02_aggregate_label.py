import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# === Parameters ===
LABEL_CSV = "../data/fire_archive_M-C61_614567.csv"
OUT_CSV = "../output/grid_time_labels.csv"

# Read grid data
df_grid = pd.read_csv("../data/fire_features_4000m_lonlat.csv")

required_cols = {'grid_id', 'longitude', 'latitude'}
if not required_cols.issubset(df_grid.columns):
    raise ValueError(f"df_grid must contain columns: {required_cols}. Current columns: {df_grid.columns.tolist()}")

# Read fire hotspot label data
df_label = pd.read_csv(LABEL_CSV)

# Extract fire hotspot longitude and latitude as arrays
label_lon = df_label['longitude'].values
label_lat = df_label['latitude'].values

# Build KDTree to accelerate matching
tree = cKDTree(df_grid[['longitude', 'latitude']].values)

# Set matching radius (in degrees, adjust as needed. 0.05 roughly equals 5km)
radius = 0.05

# Find the nearest grid for each hotspot
dist, idx = tree.query(np.stack([label_lon, label_lat], axis=1), distance_upper_bound=radius)

# Assign matched grid_id
label_grid_ids = df_grid['grid_id'].values[idx]
df_label['grid_id'] = label_grid_ids

# Filter out unmatched points (distance greater than radius will be set to len(df_grid))
mask = idx < len(df_grid)
df_label = df_label[mask]

# Output: number of fire hotspots per grid per day
df_label['date'] = pd.to_datetime(df_label['acq_date'])  # acq_date is the date field
df_label['date'] = df_label['date'].dt.strftime('%Y-%m-%d')

group = df_label.groupby(['grid_id', 'date']).size().reset_index(name='fire_count')
group.to_csv(OUT_CSV, index=False)
print(f"Aggregated fire hotspot counts saved to: {OUT_CSV}, total {len(group)} records")
