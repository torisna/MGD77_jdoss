import pandas as pd
import numpy as np
import folium
import branca.colormap as cm

def read_mgd77_kaiho(input_file):
    """
    Read MGD77 formatted data (Kaiho version) and return as a DataFrame.
    
    Parameters:
    input_file (str): Path to the data file.
    
    Returns:
    pd.DataFrame: Processed data as a DataFrame.
    """
    # ファイルを読み込み
    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # 修正後の固定列幅
    column_widths = [1, 7, 4, 4, 2, 2, 2, 5, 8, 9, 1, 
                     6, 6, 2, 1, 6, 6, 6, 1, 5, 6, 7, 
                     6, 6, 5, 5, 1]  

    # 修正後の英語カラム名
    column_names = [
        "Record_Type", "Survey_ID", "Time_Zone_Correction", "Year", "Month", "Day", "Hour", "Minute",
        "Latitude", "Longitude", "Position_Type_Code", "Depth_Travel_Time",
        "Depth_Corrected", "Depth_Correction_Code", "Depth_Type_Code", "Total_Magnetic_Field_Sensor1",
        "Total_Magnetic_Field_Sensor2", "Magnetic_Anomaly", "Magnetic_Anomaly_Sensor",
        "Geomagnetic_Diurnal_Correction", "Magnetic_Sensor_Depth_Altitude", "Observed_Gravity", 
        "Eotvos_Correction", "Free_Air_Anomaly", "Seismic_Survey_Line_Number", 
        "Seismic_Survey_Shot_Point", "Position_Accuracy_Code"
    ]

    # 全データをリストに格納
    data = []
    for line in lines:
        columns = []
        start = 0
        for width in column_widths:
            columns.append(line[start:start + width].strip())  # 余計な空白を削除
            start += width
        data.append(columns)

    # データフレームを作成
    df = pd.DataFrame(data, columns=column_names)

    # 数値として扱えるものは変換（Survey_ID 以外）
    numeric_cols = df.columns.difference(["Survey_ID"])  # Survey_ID 以外を数値化
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")  # 数値変換

    # データ型の変換
    df["Survey_ID"] = df["Survey_ID"].astype(str)  # 文字列として扱う
    df["Minute"] = df["Minute"] / 1000  # 分単位に変換
    df["Latitude"] = df["Latitude"] / 100000  # 緯度を度単位に変換
    df["Longitude"] = df["Longitude"] / 100000  # 経度を度単位に変換
    df["Total_Magnetic_Field_Sensor1"] = df["Total_Magnetic_Field_Sensor1"] / 10  # 経度を度単位に変換
    df["Total_Magnetic_Field_Sensor2"] = df["Total_Magnetic_Field_Sensor2"] / 10  # 経度を度単位に変換
    df["Magnetic_Anomaly"] = df["Magnetic_Anomaly"] / 10  # 経度を度単位に変換
    df["Observed_Gravity"] = df["Observed_Gravity"] / 10  # 経度を度単位に変換
    df["Eotvos_Correction"] = df["Eotvos_Correction"] / 10  # 経度を度単位に変換
    df["Free_Air_Anomaly"] = df["Free_Air_Anomaly"] / 10  # 経度を度単位に変換

    return df


def export_mgd77_kaiho(df, infile):
    """
    Export MGD77 data to two CSV files:
    - Raw data file: `{infile}_raw.csv`
    - Cleaned data file: `{infile}_cleaned.csv`
      (Only specific columns are processed: max value used for zero detection, original values retained)
    
    Parameters:
    df (pd.DataFrame): The DataFrame to export.
    infile (str): The input file name (used to generate output file names).
    
    Returns:
    pd.DataFrame: The cleaned DataFrame (df_cleaned).
    """
    # 出力ファイル名の定義
    output_raw = f"{infile}_raw.csv"
    output_cleaned = f"{infile}_cleaned.csv"

    # そのままのデータをCSV出力
    df.to_csv(output_raw, index=False)
    print(f"Raw data saved as: {output_raw}")

    # NaNを適用する対象カラム（指定の7カラムのみ）
    target_cols = [
        "Day", "Hour", "Minute", "Total_Magnetic_Field_Sensor1", 
        "Magnetic_Anomaly", "Observed_Gravity", "Free_Air_Anomaly"
    ]

    # クリーンデータ用にコピー
    df_cleaned = df.copy()

    # 各カラムの最大値を取得（判定用）
    max_values = df_cleaned[target_cols].max()

    # 元の値を保持しつつ、最大値を引いた結果が0になった場所だけ NaN にする
    for col in target_cols:
        mask = df_cleaned[col] - max_values[col] == 0  # 0 になった場所を特定
        df_cleaned.loc[mask, col] = np.nan  # そこだけ NaN に置換

    # クリーンデータをCSV出力
    df_cleaned.to_csv(output_cleaned, index=False)
    print(f"Cleaned data saved as: {output_cleaned}")

    return df_cleaned  # df_cleaned を返す


def plot_anomaly_maps(df_cleaned, infile):
    """
    Generate two folium scatter maps:
    - `Free_Air_Anomaly` values visualized with a color scale.
    - `Magnetic_Anomaly` values visualized with a color scale.
    
    Maps are saved as `{infile}_free_air_anomaly_map.html` and `{infile}_magnetic_anomaly_map.html`.

    Parameters:
    df_cleaned (pd.DataFrame): The cleaned DataFrame with NaN values.
    infile (str): The input file name (used to generate output file names).
    """

    # Free_Air_Anomaly のデータがある行を抽出
    free_air_df = df_cleaned.dropna(subset=["Free_Air_Anomaly"])
    
    # Magnetic_Anomaly のデータがある行を抽出
    magnetic_df = df_cleaned.dropna(subset=["Magnetic_Anomaly"])

    # 地図の初期中心点（データがある最初の地点）
    if not free_air_df.empty:
        center_lat, center_lon = free_air_df.iloc[0][["Latitude", "Longitude"]]
    elif not magnetic_df.empty:
        center_lat, center_lon = magnetic_df.iloc[0][["Latitude", "Longitude"]]
    else:
        print("No valid latitude/longitude data available.")
        return

    # カラーマップの設定（青→緑→赤）
    free_air_colormap = cm.linear.YlGnBu_09.scale(
        free_air_df["Free_Air_Anomaly"].min(), free_air_df["Free_Air_Anomaly"].max()
    )

    magnetic_colormap = cm.linear.PuRd_09.scale(
        magnetic_df["Magnetic_Anomaly"].min(), magnetic_df["Magnetic_Anomaly"].max()
    )

    # Free_Air_Anomaly のマップ作成
    free_air_map = folium.Map(location=[center_lat, center_lon], zoom_start=5)

    for _, row in free_air_df.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=4,
            color=free_air_colormap(row["Free_Air_Anomaly"]),
            fill=True,
            fill_color=free_air_colormap(row["Free_Air_Anomaly"]),
            fill_opacity=0.8,
            popup=f"Free Air Anomaly: {row['Free_Air_Anomaly']}"
        ).add_to(free_air_map)

    # カラーバーを追加
    free_air_colormap.caption = "Free Air Anomaly"
    free_air_map.add_child(free_air_colormap)

    # Magnetic_Anomaly のマップ作成
    magnetic_map = folium.Map(location=[center_lat, center_lon], zoom_start=5)

    for _, row in magnetic_df.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=4,
            color=magnetic_colormap(row["Magnetic_Anomaly"]),
            fill=True,
            fill_color=magnetic_colormap(row["Magnetic_Anomaly"]),
            fill_opacity=0.8,
            popup=f"Magnetic Anomaly: {row['Magnetic_Anomaly']}"
        ).add_to(magnetic_map)

    # カラーバーを追加
    magnetic_colormap.caption = "Magnetic Anomaly"
    magnetic_map.add_child(magnetic_colormap)

    # ファイル保存
    free_air_map_file = f"{infile}_free_air_anomaly_map.html"
    magnetic_map_file = f"{infile}_magnetic_anomaly_map.html"

    free_air_map.save(free_air_map_file)
    magnetic_map.save(magnetic_map_file)

    print(f"Free Air Anomaly map saved as: {free_air_map_file}")
    print(f"Magnetic Anomaly map saved as: {magnetic_map_file}")


def main(input_file):
    # MGD77データを読み込む
    df = read_mgd77_kaiho(input_file)
    
    # データをCSVファイルにエクスポート
    df_cleaned = export_mgd77_kaiho(df, input_file)

    # 地図の作成 & 出力
    plot_anomaly_maps(df_cleaned, input_file)


# 使用例
if __name__ == "__main__":
    infile = "HM7503-data"  # 入力ファイル名
    main(infile)
