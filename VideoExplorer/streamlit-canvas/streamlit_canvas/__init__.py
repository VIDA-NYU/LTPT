import streamlit.components.v1 as components
import pandas as pd
import streamlit as st


mc_component = components.declare_component(
    name="streamlit-canvas",
    url="http://localhost:3000/"
)
def format_line(line):
    def format_value(key, value):
        joint_names = [
            "head", "left-hand", 'right-hand', 'left-foot', 'right-foot', 'up-spine', 'down-spine'
        ]
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
mc_result = mc_component()
if mc_result is not None:
    rows = list(map(format_line, mc_result['data']))
    if "metric" in mc_result:
        metrc_str = format_metric(mc_result['metric'], rows)
        st.write(metrc_str)
    st.dataframe(pd.DataFrame(rows, columns=['joint A', 'joint B', 'color']))
