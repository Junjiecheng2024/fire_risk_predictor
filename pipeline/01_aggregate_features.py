import pandas as pd

# Read grid-time combinations and feature data with grid_id
df_grid_time = pd.read_csv("../output/grid_time_table.csv")
df_feat = pd.read_csv("../data/fire_features_4000m_with_id.csv")

# Feature columns (modify according to actual column names!)
feature_cols = ['NDVI', 'NDWI', 'dist_to_city', 'elevation', 'landcover', 'slope']
use_cols = ['grid_id'] + feature_cols

# Merge (left join to retain all grid-time records)
df = df_grid_time.merge(df_feat[use_cols], on='grid_id', how='left')

df.to_csv("../output/grid_time_features.csv", index=False)
print("Output grid_time_features.csv:", df.shape)
