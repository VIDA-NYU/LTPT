import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot
from streamlit_canvas import component_func as st_canvas

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
            cols[col_idx].image(img, width=200)


##############
# Streamlit Dashboard
def main(): 
    st.title("Video Tool - Camera Position Explorer")
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True) # radio button hack
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
    st_canvas()


##############
# Main
if __name__ == "__main__":
    main()