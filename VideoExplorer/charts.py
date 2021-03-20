import altair as alt
import streamlit as st
import pandas as pd
import numpy as np
from vega_datasets import data

from streamlit_vega_lite import vega_lite_component, altair_component

hist_data = pd.DataFrame(np.random.normal(42, 10, (200, 1)), columns=["x"])

# @st.cache
def make_histogram(data):
    brushed = alt.selection_interval(encodings=["x"], name="brushed")
    config = (
        alt.Chart(data)
        .mark_bar()
        .encode(alt.X("x:Q", bin=True), y="count()")
        .add_selection(brushed)
    )
    return config
    # return event_dict


def process_hist_event(event, values):
    brush_range = event['x']
    filtered_values = values[(values['x'] > brush_range[0]) & (values['x'] < brush_range[1])]
    return filtered_values


def make_parallel_distribution(datum):
    # datum = data.iris()
    metric_names = list(filter(lambda x: x != "file_id", datum.columns))
    brushed = alt.selection_interval(encodings=["y"], name="brushed")
    config = (alt.Chart(datum)
              .transform_window(
                    index='count()'
                )
              .transform_fold(
                    metric_names
                ).mark_line().encode(
                    x='key:N',
                    y='value:Q',
                    color='species:N',
                    detail='index:N',
                    opacity=alt.value(0.5)
                ).properties(width=500)).add_selection(brushed)
    return config, datum

