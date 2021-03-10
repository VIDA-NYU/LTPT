import time
import numpy as np
from pathlib import Path
import streamlit as st
from matplotlib import pyplot
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

# st.beta_set_page_config(layout="wide")
img_path = Path("../baseball_field")

def quantize(img, n=16):
    h, w, d = img.shape
    assert(d == 3)
    img = img.reshape(-1, 3)

    samples = shuffle(img, n_samples=3000, random_state=0)
    kmeans = KMeans(n_clusters=n, random_state=0).fit(samples)
    colors, labels = kmeans.cluster_centers_, kmeans.predict(img)
    # print(labels.shape, colors)

    img2 = np.zeros_like(img)
    img2[:] = colors[labels]

    return img2.reshape((h, w, d)), colors, labels


def main():
    st.title("Image Segmentation Tool")
    img_filename = st.selectbox("Select image", [img_path / "baseball_field_example_1.jpg",
                                                 img_path / "baseball_field_example_2.jpg"])
    n = st.slider('Number of Colors', min_value=1, max_value=64, value=8)
    mode = st.radio("Segmentation Mode", ("Colors", "Connected Components"))

    img = pyplot.imread(img_filename)
    quanized, colors, labels = quantize(img, n)
    st.image(quanized)

    sz = 50
    cimg = np.zeros((sz, n * sz, 3), dtype=np.uint8)
    for i in range(n):
        cimg[:, i*sz:(i+1)*sz] = colors[i, :]
    st.image(cimg)

    color = st.beta_color_picker('Pick A Color', '#00f900')
    # st.write(x, 'squared is', x * x)
    # progress_bar = st.sidebar.progress(0)
    # status_text = st.sidebar.empty()
    # last_rows = np.random.randn(1, 1)
    # chart = st.line_chart(last_rows)
    #
    # for i in range(1, 101):
    #     new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    #     status_text.text("%i%% Complete" % i)
    #     chart.add_rows(new_rows)
    #     progress_bar.progress(i)
    #     last_rows = new_rows
    #     time.sleep(0.05)
    #
    # progress_bar.empty()
    #
    # # Streamlit widgets automatically run the script from top to bottom. Since
    # # this button is not connected to any other logic, it just causes a plain
    # # rerun.
    # st.button("Re-run")

if __name__ == "__main__":
    main()
