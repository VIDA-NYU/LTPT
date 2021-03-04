import os
import streamlit.components.v1 as components
import pandas as pd
import streamlit as st
from .utils.constants import kps_source


_RELEASE = False  # on packaging, pass this to True

if not _RELEASE:
    _component_func = components.declare_component(
        name="st-canvas",
        url="http://localhost:3000/"
    )

else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component("st-canvas", path=build_dir)

component_func = _component_func

def format_line(line):
    def format_value(key, value):
        joint_names = list(kps_source.keys())
        if key == 'src' or key == 'dest':
            return joint_names[int(value)]
        else:
            return value
    keys = ['src', 'dest', 'color']
    row = list(map(lambda key: format_value(key, line[key]), keys))
    return row

def format_metric(metric, data):
    lines = list(map(lambda x: data[int(x)][0] + " to " + data[int(x)][1], metric['lines']))
    return metric['type'] + " of " + " and ".join(lines)

def use_component(component):
    mc_result = component()
    if mc_result is not None:
        rows = list(map(format_line, mc_result['data']))
        if "metric" in mc_result:
            metrc_str = format_metric(mc_result['metric'], rows)
            st.write(metrc_str)
        st.dataframe(pd.DataFrame(rows, columns=['joint A', 'joint B', 'color']))

# use_component(_component_func)



