import pandas as pd
import numpy as np
import os

# 1. Read basic grid data (automatically add grid_id and save a new CSV)
df_grid = pd.read_csv("../data/fire_features_4000m.csv")

if "grid_id" not in df_grid.columns:
    df_grid['grid_id'] = np.arange(len(df_grid))
    df_grid.to_csv("../data/fire_features_4000m_with_id.csv", index=False)
    print("Saved fire_features_4000m_with_id.csv with added grid_id")
else:
    print("fire_features_4000m.csv already contains grid_id field")

# 2. Set the time range
dates = pd.date_range("2024-01-01", "2024-12-31", freq='D')
df_time = pd.DataFrame({'date': dates})

# 3. Cartesian product to obtain all grid-time combinations
df_grid_time = df_grid[['grid_id']].assign(key=1).merge(df_time.assign(key=1), on='key').drop('key', axis=1)
os.makedirs("../output", exist_ok=True)
df_grid_time.to_csv("../output/grid_time_table.csv", index=False)
print("Output grid_time_table.csv:", df_grid_time.shape)
