import pandas as pd
from data import load_meta_by_json, process_df
video_df = pd.read_csv("camera_view.csv")

meta_data = load_meta_by_json()
video_df = video_df.replace("Batter, 3B side", "Batter")
video_df = video_df.replace("Pitcher, 3B side", "Pitcher")
video_df = process_df(video_df, meta_data)
