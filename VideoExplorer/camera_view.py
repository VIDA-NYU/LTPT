import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot
from streamlit_canvas import component_func as st_canvas
from streamlit_vega_lite import altair_component
from canvas_interaction import plot_canvas, build_metric_func, get_fake_data, calculate_metrics, MetricManager, metric_manager
from data import load_meta_by_json as load_meta, meta_data
from charts import make_histogram, process_hist_event, make_parallel_distribution
##############
# Setting up data and paths
img_path = Path("../../../baseball-analysis/videos/video_images")
video_df = pd.read_csv("camera_view.csv")
camera_views = video_df["true_camera_view"].unique()
def divide_chunks(l, n):
    # divide list l in chunks of size n
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

def plot_image_grid(df, n_columns):
    n = df.shape[0]
    for idxs_chunk in divide_chunks(range(n), n_columns):
        cols = st.beta_columns(3)
        for col_idx, img_idx in enumerate(idxs_chunk):
            path = img_path / df["file"].iloc[img_idx]
            pred_class = df["pred_camera_view"].iloc[img_idx]
            img = pyplot.imread(path)
            cols[col_idx].write("**Pred**: "+ pred_class)
            cols[col_idx].write("**Pred**: " + df['file'].iloc[img_idx])
            cols[col_idx].image(img, width=200)


valid_files = []
for file_id in meta_data:
    meta = meta_data[file_id]
    if "pose_data" in meta:
        valid_files.append(file_id)
valid_files = list(map(lambda x: x + ".jpg", valid_files))
video_df = video_df[video_df['file'].isin(valid_files)]

##############
# Streamlit Dashboard
def main():
    st.title("Video Tool - Camera Position Explorer")
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True) # radio button hack

    # clear_button = st.button("clear")
    left_column, right_column = st.beta_columns(2)
    with right_column:
        finish_button = st.button("finish drawing")
    with left_column:

        canvas_result = plot_canvas("create")
    if finish_button:
        metric_manager.add_metric(canvas_result)
    # if clear_button:
    #     plot_canvas("redraw")
    filtered = []
    use_metric_filter = False

    true_camera_view = st.selectbox("Select camera position", camera_views)
    correct_pred_option = st.radio(
        "Show Predictions",
        ("Correct", "Incorrect"))
    filtered_df = video_df[video_df["true_camera_view"] == true_camera_view]

    if correct_pred_option == "Correct":
        filtered_df = filtered_df[filtered_df["true_camera_view"] == filtered_df["pred_camera_view"]]
    else:
        filtered_df = filtered_df[filtered_df["true_camera_view"] != filtered_df["pred_camera_view"]]
    if canvas_result and "metric" in canvas_result:
        # parallel_config = make_parallel_distribution([])
        # parallel_event = altair_component(parallel_config)
        with right_column:
            metric_values = pd.DataFrame(valid_files)
            metric_values.columns = ['file']
            for i, metric_def in enumerate(metric_manager.metrics[:1]):
                values = calculate_metrics(metric_def, filtered_df, meta_data)
                config = make_histogram(values)
                hist_event = altair_component(altair_chart=config)
                if "name" in hist_event:
                    filtered = process_hist_event(hist_event, values)
                    filtered = filtered['file'].tolist()
                    filtered = list(map(lambda x: x + ".jpg", filtered))
                    use_metric_filter = True
    if use_metric_filter:
        filtered_df = filtered_df[filtered_df['file'].isin(filtered)]
    filtered_df = filtered_df.iloc[:3*10]
    plot_image_grid(filtered_df, 3)


##############
# Main
if __name__ == "__main__":
    main()