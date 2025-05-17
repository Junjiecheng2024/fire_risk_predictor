import folium
import pandas as pd

df = pd.read_csv("../predict/data/prediction_today.csv")
m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()], zoom_start=8)

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=6,
        color=None,
        fill=True,
        fill_color="red" if row["fire_risk"] > 0.7 else "orange" if row["fire_risk"] > 0.4 else "green",
        fill_opacity=0.6,
        popup=f"Fire Risk: {row['fire_risk']:.2f}"
    ).add_to(m)

m.save("fire_risk_map.html")
print("✅ 热力图已保存为 fire_risk_map.html")
