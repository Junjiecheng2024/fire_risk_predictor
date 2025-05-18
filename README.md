# 🔥 Fire Risk Predictor

This project is a **spatio-temporal fire risk prediction pipeline**, which uses Earth observation data (vegetation indices, weather reanalysis, topography) to train a **Poisson regression model** and produce **interactive fire risk heatmaps**.

## 🚀 Features

- Poisson regression for rare event modeling
- Uses NDVI, NDWI, ERA5, and static features
- Handles data from GEE, Copernicus CDS, and FIRMS
- Interactive heatmap visualization with Folium
- Scalable to large datasets

---

## 🧱 Project Structure

```bash
fire-risk-predictor/
├── pipeline/                     # Model training pipeline
│   ├── 00_make_grid_time_table.py
│   ├── 01_aggregate_features.py
│   ├── 02_aggregate_label.py
│   ├── 03_merge_features_labels.py
│   ├── 10_train_poisson.py
│
├── predictor/                   # Inference & visualization
│   ├── 20_predict_and_plot.py
│   ├── generate_fire_risk_heatmap.py
│
├── data/                        # Raw GEE, ERA5, FIRMS data
├── output/                      # Processed features + predictions
├── model/                       # Saved Poisson model
├── requirements.txt
└── README.md
