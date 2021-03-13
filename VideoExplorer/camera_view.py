import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot
from streamlit_canvas import component_func as st_canvas
from canvas_interaction import plot_canvas, build_metric_func, get_fake_data
from data import load_meta_by_json as load_meta
from charts import make_histogram
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
            cols[col_idx].write("*Metric*: " + "0")
            cols[col_idx].image(img, width=200)


meta_data = load_meta()

data = pd.DataFrame(np.random.normal(42, 10, (200, 1)), columns=["x"])


##############
# Streamlit Dashboard
def main():
    st.title("Video Tool - Camera Position Explorer")
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True) # radio button hack
    # canvas_result = plot_canvas()
    my_expander = st.beta_expander("Draw Metrics", expanded=True)
    left_column, right_column = st.beta_columns(2)
    with left_column:
        canvas_result = plot_canvas()

    if canvas_result and "metric" in canvas_result:
        f = build_metric_func(canvas_result)
        values = []
        for row in video_df.iterrows():
            file_id = row[1]['file'][:-4]
            meta = meta_data[file_id]
            if "pose_data" in meta:
                metric_value = f(meta['pose_data'])
                values.append([file_id, metric_value])
        values = pd.DataFrame(values)
        values.columns = ['file', "x"]
        # hist_values = np.histogram(values, bins="auto")
        # fig, ax = pyplot.subplots()
        # ax.hist(values, bins=20)
        # st.pyplot(fig)
        # st.bar_chart(hist_values[0])
        st.write(values)
        with right_column:
            if len(values) > 0:
                hist_event = make_histogram(data)
                st.write(hist_event)

    true_camera_view = st.selectbox("Select camera position", camera_views)
    correct_pred_option = st.radio(
        "Show Predictions",
        ("Correct", "Incorrect"))
    filtered_df = video_df[video_df["true_camera_view"] == true_camera_view]
    if correct_pred_option == "Correct":
        filtered_df = filtered_df[filtered_df["true_camera_view"] == filtered_df["pred_camera_view"]]
    else:
        filtered_df = filtered_df[filtered_df["true_camera_view"] != filtered_df["pred_camera_view"]]
    filtered_df = filtered_df.iloc[:3*10]
    plot_image_grid(filtered_df, 3)


##############
# Main
if __name__ == "__main__":
    main()