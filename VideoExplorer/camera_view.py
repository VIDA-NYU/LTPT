import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot
from streamlit_canvas import component_func as st_canvas
from streamlit_vega_lite import altair_component
from canvas_interaction import plot_canvas, build_metric_func, get_fake_data, calculate_metrics, MetricManager, metric_manager
from data import load_meta_by_json as load_meta
from charts import make_histogram, process_hist_event, make_parallel_distribution
from streamlit_parallel_coordinates import component_func as st_parcoords
from streamlit_image_grid import component_func as st_imagegrid
from prepare import meta_data, video_df
##############
# Setting up data and paths
img_path = Path("../../videos/video_images")
camera_views = video_df["true_camera_view"].unique()
actions = list(filter(lambda x: x != "Not Detected", video_df['action'].unique()))
# camera_views = camera_views[1:3]
st.set_page_config(layout="wide")
# st.write(meta_data)
def divide_chunks(l, n):
    # divide list l in chunks of size n
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

def plot_image_grid(df, n_columns, cols):
    n = df.shape[0]
    for idxs_chunk in divide_chunks(range(n), n_columns):
        # cols = st.beta_columns(3)
        for col_idx, img_idx in enumerate(idxs_chunk):
            path = img_path / df["file"].iloc[img_idx]
            pred_class = df["pred_camera_view"].iloc[img_idx]
            img = pyplot.imread(path)
            # cols[col_idx].write("**Pred**: "+ pred_class)
            cols[col_idx].write("**File**: " + df['file'].iloc[img_idx])
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
    # st.markdown(
    #     f'''
    #                <style>
    #                 .e1fqkh3o0 {{
    #                     width: 1200px;
    #                 }}
    #                    .sidebar .e1fqkh3o0  .sidebar-content {{
    #                        width: 1200px;
    #                    }}
    #                </style>
    #            ''',
    #     unsafe_allow_html=True
    # )
    st.title("Video Tool - Camera Position Explorer")
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True) # radio button hack

    selection_columns = st.beta_columns(2)
    with selection_columns[0]:
        true_camera_view = st.selectbox("Select camera position", camera_views)
    with selection_columns[1]:
        true_action = st.selectbox("Select the player action", actions)


    # correct_pred_option = st.radio(
    #     "Show Predictions",
    #     ("Correct", "Incorrect"))
    # clear_button = st.button("clear")
    correct_pred_option = "Correct"
    columns = st.beta_columns((4, 3))
    image_columns = columns[1:]
    left_column = columns[0]
    right_column = columns[1]
    with left_column:
        canvas_result = plot_canvas("create")

    # if clear_button:
    #     plot_canvas("redraw")
    filtered = []
    use_metric_filter = False


    filtered_df = video_df[video_df["true_camera_view"] == true_camera_view]
    filtered_df = filtered_df[filtered_df['action'] == true_action]
    if correct_pred_option == "Correct":
        filtered_df = filtered_df[filtered_df["true_camera_view"] == filtered_df["pred_camera_view"]]
    else:
        filtered_df = filtered_df[filtered_df["true_camera_view"] != filtered_df["pred_camera_view"]]
    json_df = []
    columns = []
    if canvas_result and "metrics" in canvas_result:
        metric_manager.metrics = list(filter(lambda x:x['visibility'], canvas_result['metrics']))
        if len(metric_manager.metrics) > 1 or True:
            with left_column:
                metrics_df = metric_manager.build_data(filtered_df, meta_data)


                for col in list(metrics_df.columns):
                    if col == "file_id":
                        continue
                    else:
                        metric_type = metric_manager.get_metric_type(col)
                        metric = metric_manager.get_metrc(col)
                        columns.append({
                            "key": col,
                            "type": metric_type if metric_type == "Angle" else "Number",
                            "name": metric['name']
                        })
                for row in metrics_df.iterrows():
                    json_df.append(dict(row[1]))
                parcoords_result = st_parcoords(data=json_df, columns=columns)
                if parcoords_result and "filtered" in parcoords_result:
                    filtered_images = list(map(lambda x: x + ".jpg", parcoords_result['filtered']))
                    # st.write(filtered_images)
                    filtered_df = filtered_df[filtered_df['file'].isin(filtered_images)]
                    use_metric_filter = False
                # parallel_config, source = make_parallel_distribution(metrics_df)
                # parallel_event = altair_component(parallel_config)
        # with right_column:
        #     metric_values = pd.DataFrame(valid_files)
        #     metric_values.columns = ['file']
        #     for i, metric_def in enumerate(metric_manager.metrics[:1]):
        #         values = calculate_metrics(metric_def, filtered_df, meta_data)
        #         config = make_histogram(values)
        #         hist_event = altair_component(altair_chart=config)
        #         if "name" in hist_event:
        #             filtered = process_hist_event(hist_event, values)
        #             filtered = filtered['file'].tolist()
        #             filtered = list(map(lambda x: x + ".jpg", filtered))
        #             use_metric_filter = True
    # st.write(filtered_df)
    if use_metric_filter:
        filtered_df = filtered_df[filtered_df['file'].isin(filtered)]

    image_json = []
    for row in filtered_df.iterrows():
        obj = {
            "file_id": row[1]['play'] + "_" + row[1]['view'],
            "col": row[1]['col'],
            "game": row[1]['game'],
            "play": row[1]['play'],
            "view": row[1]['view']
        }
        image_json.append(obj)

    with right_column:
        st_imagegrid(images=image_json, data=json_df, columns=columns)
    filtered_df = filtered_df.iloc[:3*10]

    # plot_image_grid(filtered_df, 3, image_columns)


##############
# Main
if __name__ == "__main__":
    main()