from streamlit_canvas import component_func as st_canvas, use_component
# from streamlit_canvas import component_func as st
import streamlit as st
import numpy as np
import pandas as pd
kps_source = {
    'Nose': 0,
    'RShoulder': 1,
    'RElbow': 2,
    'RWrist': 3,
    'LShoulder': 4,
    'LElbow': 5,
    'LWrist': 6,
    'RHip': 7,
    'RKnee': 8,
    'RAnkle': 9,
    'LHip': 10,
    'LKnee': 11,
    'LAnkle': 12,
    'REye': 13,
    'LEye': 14,
    'REar': 15,
    'LEar': 16,
    # 'Background': 18,
}

def plot_canvas(op):
    return st_canvas(operation=op)

def extract_vector(line, data):

    src_pos = data[kps_source[line['src']]]
    dest_pos = data[kps_source[line['dest']]]
    src_pos = np.array(src_pos)
    dest_pos = np.array(dest_pos)
    v = dest_pos - src_pos
    return v


def calculate_vector_length(v):
    return np.sqrt(np.sum(np.power(v, 2)))


def calculate_angle(v0, v1):
    l0 = calculate_vector_length(v0)
    l1 = calculate_vector_length(v1)
    cos = np.sum(v0 * v1) / (l0 * l1)
    angle = np.arccos(cos)
    return angle


def format_lines(line0, line1):
    apex = None
    mutual_vertex_num = 0
    keys = ['src', 'dest']
    line0_vertexes = list(map(lambda x: line0[x], keys))
    assert line0['src'] != line0['dest']
    assert line1['src'] != line1['dest']
    for key in ['src', 'dest']:
        if line1[key] in line0_vertexes:
            apex = line1[key]
            mutual_vertex_num += 1
    assert mutual_vertex_num == 1
    _line0 = {
        "src": apex,
        "dest": line0[list(filter(lambda x: line0[x]!=apex, keys))[0]]
    }
    _line1 = {
        "src": apex,
        "dest": line1[list(filter(lambda x: line1[x] != apex, keys))[0]]
    }
    return _line0, _line1


@st.cache
def build_metric_func(metric):
    if not metric:
        return lambda x: 0


    def _f(data):
        if metric['type'] == "Angle":
            vectoddrs = []
            selected_lines = metric['lines']
            # for line_idx in metric['lines']:
            #     line = lines[int(line_idx)]
            #     selected_lines.append(line)
            line0, line1 = format_lines(*selected_lines)
            v0 = extract_vector(line0, data)
            v1 = extract_vector(line1, data)

            angle = calculate_angle(v0, v1)
            angle = radian_to_angle(angle)
            return angle
        else:
            line = metric['lines'][0]
            # line = lines[int(line_idx)]
            vector = extract_vector(line, data)
            return calculate_vector_length(vector) * 100
    return _f

def calculate_metrics(metric_def, video_df, meta_data):
    f = build_metric_func(metric_def)
    values = []
    for row in video_df.iterrows():
        file_id = row[1]['file'][:-4]
        meta = meta_data[file_id]
        if "pose_data" in meta:
            metric_value = f(meta['pose_data'])
            values.append([file_id, metric_value])
    values = pd.DataFrame(values)
    values.columns = ['file', "x"]
    return values

def get_fake_data():
    data = [[0.041666666666666664, 0.1956521739130435], [0.018518518518518517, 0.21195652173913043], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.032407407407407406, 0.21739130434782608], [0.027777777777777776, 0.30434782608695654], [0.037037037037037035, 0.3858695652173913], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.018518518518518517, 0.43478260869565216], [0.018518518518518517, 0.5597826086956522], [0.018518518518518517, 0.6956521739130435], [0.037037037037037035, 0.18478260869565216], [0.0, 0.0], [0.023148148148148147, 0.1793478260869565], [0.0, 0.0]]
    data = [[0.4215269386768341, 0.37059587240219116], [0.4177720844745636, 0.3659610450267792], [0.4193824529647827, 0.36473599076271057], [0.3782474100589752, 0.37624943256378174], [0.42872631549835205, 0.3747600018978119], [0.47433868050575256, 0.43500542640686035], [0.4608614146709442, 0.42827171087265015], [0.4767116606235504, 0.5121890902519226], [0.3557441234588623, 0.38959959149360657], [0.46645060181617737, 0.550093948841095], [0.48679256439208984, 0.41668176651000977], [0.4579068422317505, 0.6090853810310364], [0.4066760540008545, 0.6132463812828064], [0.511817216873169, 0.6708146333694458], [0.35595887899398804, 0.742112398147583], [0.5375047326087952, 0.8074959516525269], [0.2646635174751282, 0.7936386466026306]]
    data = [[0.7807246446609497, 0.38548755645751953], [0.7839173674583435, 0.3798346221446991], [0.7781311869621277, 0.3797035813331604], [0.7875562310218811, 0.3851783871650696], [0.5992623567581177, 0.35279813408851624], [0.6062614917755127, 0.4293033480644226], [0.5445217490196228, 0.41809654235839844], [0.45325201749801636, 0.4919627606868744], [0.5524997115135193, 0.4428524971008301], [0.45269104838371277, 0.49260079860687256], [0.5503361225128174, 0.4350198805332184], [0.5884430408477783, 0.6227784752845764], [0.5249643325805664, 0.6208804845809937], [0.6418211460113525, 0.7336814403533936], [0.4793747067451477, 0.7242373824119568], [0.7246833443641663, 0.8094434142112732], [0.44495224952697754, 0.8766688108444214]]
    data = [[0.7807246446609497, 0.38548755645751953], [0.5445217490196228, 0.41809654235839844], [0.5524997115135193, 0.4428524971008301], [0.5503361225128174, 0.4350198805332184], [0.6062614917755127, 0.4293033480644226], [0.45325201749801636, 0.4919627606868744], [0.45269104838371277, 0.49260079860687256], [0.5249643325805664, 0.6208804845809937], [0.4793747067451477, 0.7242373824119568], [0.44495224952697754, 0.8766688108444214], [0.5884430408477783, 0.6227784752845764], [0.6418211460113525, 0.7336814403533936], [0.7246833443641663, 0.8094434142112732], [0.7781311869621277, 0.3797035813331604], [0.7839173674583435, 0.3798346221446991], [0.5992623567581177, 0.35279813408851624], [0.7875562310218811, 0.3851783871650696]]
    return data


def radian_to_angle(radian):
    return 180 * radian / np.pi





if __name__ == '__main__':
    fake_data = get_fake_data()
    sample_result = {
        "metric": {
            "type": "angle",
            "lines": [
                0, 1
            ]
        },
        "data": [
            {
                "src": "RShoulder",
                "dest": "RHip"
            },
            {
                "dest": "RKnee",
                "src": "RHip",
            }
        ]
    }
    f = build_metric_func(sample_result)
    value = f(fake_data)


class MetricManager:
    def __init__(self):
        self.metrics = []

    def add_metric(self, metric):
        self.metrics.append(metric)

    def build_data(self, df, meta_data):
        result_df = pd.DataFrame()
        first = True
        for i, metric in enumerate(self.metrics):
            metric_df = calculate_metrics(metric, df, meta_data)
            metric_df.columns = ['file_id', metric['id']]
            if first:
                result_df = metric_df
                first = False
            else:
                result_df = pd.concat([result_df, metric_df[metric['id']]], axis=1)
        metric_ids = list(map(lambda x: x['id'], self.metrics))
        result_df.columns = ["file_id", *metric_ids]
        return result_df

    def get_metric_type(self, metric_id):
        metric = list(filter(lambda x: x['id']==metric_id, self.metrics))[0]
        return metric['type']

    def get_metrc(self, metric_id):
        metric = list(filter(lambda x: x['id'] == metric_id, self.metrics))[0]
        return metric
metric_manager = MetricManager()