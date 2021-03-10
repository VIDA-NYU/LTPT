import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot
import pymongo
import json


##############
# Setting up data and paths
with open("config.json", "r") as f:
    config = json.load(f)
    conn_path = config["mongodb_url"]
    img_path = Path(config["img_path"])

##############
# MongoDB connector
try:
    client = pymongo.MongoClient(conn_path)
    db = client.ltpt
    camera_views = db["videos"].distinct("predicted_view")
    games = db["videos"].distinct("game")
except Exception as e:
    st.write("# MongoDB Connection Error");
    st.write(str(e))

##############
# Mappings
map_letter_view = {"A": "Behind Pitcher", "B": "Batter", "C": "Pitcher", "D": "Behind Home"}
map_view_letter = {"Behind Pitcher": "A", "Batter": "B", "Pitcher": "C", "Behind Home": "D"}

def query_db(true_view, correct, game_id):
    true_letter = map_view_letter[true_view]
    if correct:
        return db["videos"].find({"predicted_view": true_view, "view": true_letter, "game":game_id})
    else:
        return db["videos"].find({"predicted_view": {"$ne": true_view}, "view": true_letter, "game":game_id})

##############
# Image Grid

def plot_image_grid(query, n_columns):
    col_idx = 0
    cols = st.beta_columns(n_columns)
    for document in query:
        pred_class = document["predicted_view"]
        true_class = map_letter_view[document["view"]]
        image_name = str(document["_id"]) + "_" + document["view"] + ".png"
        image_path = img_path / image_name
        img = pyplot.imread(image_path)
        cols[col_idx].write("**Pred**: "+ pred_class)
        cols[col_idx].image(img, width=200)
        col_idx+=1
        if col_idx == n_columns:
            col_idx = 0
            cols = st.beta_columns(n_columns)

##############
# Streamlit Dashboard
def main(): 
    st.title("Video Tool - Camera Position Explorer")
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True) # radio button hack
    game_id = st.selectbox("Select game", games)
    true_camera_view = st.selectbox("Select camera position", camera_views)
    correct_pred_option = st.radio(
        "Show Predictions",
        ("Correct", "Incorrect"))
    correct =  correct_pred_option == "Correct"
    query = query_db(true_camera_view, correct, game_id).limit(3*5) # 5 rows
    plot_image_grid(query, 3)


##############
# Main
if __name__ == "__main__":
    main()