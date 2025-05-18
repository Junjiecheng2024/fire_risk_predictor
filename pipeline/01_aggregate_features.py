import pandas as pd

# 读取带 grid_id 的格点特征数据
df_grid_time = pd.read_csv("../output/grid_time_table.csv")
df_feat = pd.read_csv("../data/fire_features_4000m_with_id.csv")

# 特征列（根据你的实际列名修改！）
feature_cols = ['NDVI', 'NDWI', 'dist_to_city', 'elevation', 'landcover', 'slope']
use_cols = ['grid_id'] + feature_cols

# 合并（left join，所有格点时间表都保留）
df = df_grid_time.merge(df_feat[use_cols], on='grid_id', how='left')

df.to_csv("../output/grid_time_features.csv", index=False)
print("输出 grid_time_features.csv:", df.shape)
