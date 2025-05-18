import pandas as pd

# Read feature file
df_feat = pd.read_csv("../output/grid_time_features.csv")

# Read label file
df_label = pd.read_csv("../output/grid_time_labels.csv")

# Merge data based on grid_id and date
df = df_feat.merge(df_label, on=["grid_id", "date"], how="left")

# Replace NaN with 0 and convert to integer
df["fire_count"] = df["fire_count"].fillna(0).astype(int)

# Save result
df.to_csv("../output/grid_time_features_labels.csv", index=False)
print("Output grid_time_features_labels.csv:", df.shape)
