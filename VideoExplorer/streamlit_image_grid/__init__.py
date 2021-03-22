import os
import streamlit.components.v1 as components
import pandas as pd
import streamlit as st


_RELEASE = False  # on packaging, pass this to True

if not _RELEASE:
    _component_func = components.declare_component(
        name="st-canvas",
        url="http://localhost:3002/"
    )

else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component("st-canvas", path=build_dir)

component_func = _component_func


# use_component(_component_func)



