from streamlit_canvas import component_func as st_canvas, use_component
import numpy as np
kps_source={
    'Nose': 0,
    'Neck': 1,
    'RShoulder': 2,
    'RElbow':3,
    'RWrist':4,
    'LShoulder':5,
    'LElbow':6,
    'LWrist':7,
    'RHip':8,
    'RKnee': 9,
    'RAnkle':10,
    'LHip':11,
    'LKnee':12,
    'LAnkle':13,
    'REye': 14,
    'LEye': 15,
    'REar': 16,
    'LEar' :17,
    # 'Background': 18,
}

def plot_canvas():
    return st_canvas()

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



def build_metric_func(result):
    if not result or 'metric' not in result:
        return lambda x: 0
    metric = result['metric']
    lines = result['data']

    def _f(data):
        if metric['type'] == "angle":
            vectoddrs = []
            selected_lines = []
            for line_idx in metric['lines']:
                line = lines[int(line_idx)]
                selected_lines.append(line)
            line0, line1 = format_lines(*selected_lines)
            print(line0)
            print(line1)
            v0 = extract_vector(line0, data)
            v1 = extract_vector(line1, data)
            angle = calculate_angle(v0, v1)
            angle = radian_to_angle(angle)
            return angle
        else:
            line_idx = metric['lines'][0]
            line = lines[line_idx]
            vector = extract_vector(line, data)
            return calculate_vector_length(vector)
    return _f


def get_fake_data():
    data = [[0.041666666666666664, 0.1956521739130435], [0.018518518518518517, 0.21195652173913043], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.032407407407407406, 0.21739130434782608], [0.027777777777777776, 0.30434782608695654], [0.037037037037037035, 0.3858695652173913], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.018518518518518517, 0.43478260869565216], [0.018518518518518517, 0.5597826086956522], [0.018518518518518517, 0.6956521739130435], [0.037037037037037035, 0.18478260869565216], [0.0, 0.0], [0.023148148148148147, 0.1793478260869565], [0.0, 0.0]]
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
                "src": "LShoulder",
                "dest": "LHip"
            },
            {
                "dest": "LShoulder",
                "src": "RShoulder",
            }
        ]
    }
    f = build_metric_func(sample_result)
    value = f(fake_data)
    print(value)
