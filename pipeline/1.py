import pandas as pd
import json

infile = "../data/fire_features_4000m_with_id.csv"
outfile = "../data/fire_features_4000m_lonlat.csv"

df = pd.read_csv(infile)

if 'longitude' not in df.columns or 'latitude' not in df.columns:
    def extract_lonlat(geo_str):
        try:
            # 修复嵌套双引号格式，然后用 json.loads 解析
            geo_str = geo_str.replace('""', '"').strip('"')
            geo = json.loads(geo_str)
            return pd.Series({
                'longitude': geo['coordinates'][0],
                'latitude': geo['coordinates'][1]
            })
        except Exception as e:
            print("❌ geo 字符串解析失败:", geo_str)
            raise e

    df[['longitude', 'latitude']] = df['.geo'].apply(extract_lonlat)
    print("✅ 已添加 longitude / latitude 列")

df.to_csv(outfile, index=False)
print("✅ 新文件已保存：", outfile)
