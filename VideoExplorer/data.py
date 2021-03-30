from pymongo import MongoClient
import json
import pandas as pd
# import streamlit as st
# from memory_profiler import profile


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
pose_kps = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}
pose_kps = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}
def convert_keypoint_name(original):
    if original[0] == "L":
        return "left_" + original[1:].lower()
    elif original[0] == "R":
        return "right_" + original[1:].lower()
    else:
        return original.lower()


def save_remote_db_to_local():
    client = MongoClient("mongodb+srv://guande:guandemongo@ltpt.qsvio.mongodb.net")
    local_client = MongoClient()
    db = client.ltpt
    local_db = local_client.ltpt
    collection_names = ['videos', 'poses', 'actions']
    for collection_name in collection_names:
        local_db[collection_name].remove({})
        collection = db[collection_name].find({})
        local_db[collection_name].insert_many(collection)
        print("saved:", collection_name)
# @st.cache
def load_meta_by_json():
    with open("meta.json", "r") as fp:
        meta_data = json.load(fp)
        return meta_data

def load_meta_from_mongo():
    client = MongoClient("mongodb+srv://guande:guandemongo@ltpt.qsvio.mongodb.net")
    data = _load_meta(client)
    return data

def load_meta():
    # client = MongoClient("mongodb+srv://guande:guandemongo@ltpt.qsvio.mongodb.net")
    client = MongoClient()
    return _load_meta(client)

# @profile
def _load_meta(client):

    print("db connected")
    db = client.ltpt
    data = {}
    pose_ids = []
    selected_video_ids = []
    files_with_pose = []
    count = 0
    videos = list(db['videos'].find({}))
    print("videos fetched")
    file_count = 0
    for doc in videos:
        # file_id = doc['filepath'].split("/")[-1][:-4]
        # file_id = file_id[:-1] + "_" +file_id[-1]
        file_id = str(doc['_id'])
        # if doc['predicted_view'] != "Batter":
        #     continue
        obj = {
            "file_id": str(doc['_id']),
            "view": doc['view'],
            "play": doc['play'],
            "game": doc['game'],
            "col": doc['col'],
            "predicted_view": doc['predicted_view'],
        }
        if "calibration" in doc:
            obj['calibration'] = doc['calibration']
        data[file_id] = obj
        if "poses" in doc:
            pose_ids.append(doc['pose_id'])
            files_with_pose.append(file_id)
            selected_video_ids.append(doc['_id'])
        count += 1
    print("videos processed", len(selected_video_ids))

    pose_index = {}
    # for i, pose_doc in enumerate(pose_docs):
    #     pose_index[str(pose_doc['_id'])] = i

    action_docs = list(db['actions'].find({
        "$or": [
            {
                "actions.swing.video_id": {
                    "$in": selected_video_ids
                }
            },
            {
                "actions.release.video_id": {
                    "$in": selected_video_ids
                }
            }
        ]

    }))
    print("actions fetched", len(action_docs))

    count = 0
    # valid_games = ["181101_AFL-PEORIA_AFL-SCOTTSDALE__0"]
    pose_info = {}
    pose_ids = []
    for i, action_doc in enumerate(action_docs):
        actions = extract_action_data(action_doc)
        for action in actions:
            pose_id = action['pose_id']
            if pose_id in pose_info:
                continue
            pose_info[pose_id] = {
                "frame": action['frame'],
                "action": action['action']
            }
            pose_ids.append(action['pose_bid'])
            # pose_doc = pose_docs[pose_index[action['pose_id']]]
            # file_doc = data[action['file_id']]
            # if file_doc['game'] not in valid_games:
            #     continue
            # if "frame" in pose_doc:
            #     count += 1
            #     continue
            # print(action['frame'])
            # pose_doc['frame'] = action['frame']
            # pose_doc['action'] = action['action']
            # print(len(file_doc['pose_data']))
            # print(file_doc['frame'])
            # file_doc['pose_frame_data'] = file_doc['pose_data'][file_doc['frame']]
            # print(file_doc)
    pose_docs = db['poses'].find({
        "_id": {
            "$in": pose_ids
        }
    })
    print("pose fetched", len(pose_docs))
    print("duplicated records:", count)
    filtered_data = {}
    i = 0
    for pose_doc in pose_docs:
    # for pose_id in pose_index:
    #     pos_doc = pose_index[pose_id]
        pose_data = extract_pose_data(pose_doc)
        # file_doc = data[files_with_pose[i]]
        file_doc = data[str(pose_doc['video_id'])]
        # print(pos_doc['filepath'])
        if str(pose_doc['_id']) in pose_info:
            # print(file_doc['frame'])
            # print(len(pose_data[0][0]))
            # print(len(pose_data[0][1]))
            # print(pose_doc['_id'])
            # print(file_doc)
            # print(pose_doc['frame'])

            pose_item = pose_info[str(pose_doc['_id'])]
            frame = pose_item["frame"]
            if frame >= len(pose_data[0][0]):
                continue

            file_doc['pose_data'] = list(map( lambda x: list(map(lambda y: y[frame], x)), pose_data))
            # filtered_data[file_doc['file_id']] = file_doc
            file_id = "video-" + str(file_count)
            file_id = str(pose_doc['video_id'])
            file_doc["file_id"] = file_id
            file_doc['action'] = pose_item['action']
            file_count += 1
            filtered_data[file_id] = file_doc
        else:
            pass
            # print(file_doc['view'])
            # print("fu")
        i += 1
    client.close()
    print("processed", len(list(filtered_data.keys())))
    return filtered_data

def extract_action_data(action_doc):
    result = []
    for action_name in action_doc['actions']:
        action_data = action_doc['actions'][action_name]
        frame = action_data['frame']
        # file_id = action_doc['play'] + "_" + action_data['view']
        data = {
            "file_id": str(action_data['video_id']),
            "pose_id": str(action_data['pose_id']),
            "pose_bid": action_data['pose_id'],
            "frame": frame,
            "view": action_data['view'],
            "action": action_name
        }
        result.append(data)
    return result
def extract_pose_data(pose_doc):
    doc = pose_doc
    pose_data = []
    for kp_key in kps_source:
        kp_key = convert_keypoint_name(kp_key)
        coordinate_keys = ["x", "y"]
        kp_data = []
        for coordinate_key in coordinate_keys:
            key = kp_key + "_" + coordinate_key
            frames = doc['poses']['Agent1'][key]
            kp_data.append(frames)
        pose_data.append(kp_data)
    return pose_data

def generate_df_by_meta(meta_data):
    views = []
    plays = []
    games = []
    cols = []
    file_ids = []

    values = []
    true_camera_views = []
    view_names = {
        "A": "Behind Pitcher",
        "B": "Batter, 3B side",
        "C": "Pitcher, 3B side",
        "D": "Behind Home"
    }
    for file_id in meta_data:
        doc = meta_data[file_id]
        value = meta_data[file_id]['action'][0].upper() + meta_data[file_id]['action'][1:]
        view = meta_data[file_id]['view']
        game = meta_data[file_id]['game']
        play = doc['play']
        col = doc['col']
        predicted_view = doc['predicted_view']
        if predicted_view == "Pitcher" and "calibration" not in doc:
            continue
        file_ids.append(file_id)
        values.append(value)
        views.append(view)
        games.append(game)
        plays.append(play)
        cols.append(col)
        true_camera_views.append(predicted_view)
    data = {
        "file": file_ids,
        "true_camera_view": true_camera_views,
        "pred_camera_view": true_camera_views,
        "col": cols,
        "game": games,
        "play": plays,
        "view": views,
        "action": values
    }
    df = pd.DataFrame(data=data)
    # df = pd.DataFrame([file_ids, cols, games, plays, views])
    return df

def process_df(oldcsv, meta_data):
    values = []
    valid_files = list(meta_data.keys())
    valid_files = list(map(lambda x: x+".jpg", valid_files))
    oldcsv = oldcsv[oldcsv['file'].isin(valid_files)]
    view_names = {
        "A": "Behind Pitcher",
        "B": "Batter, 3B side",
        "C": "Pitcher, 3B side",
        "D": "Behind Home"
    }
    views = []
    plays = []
    games = []
    cols = []
    true_camera_views = []
    for row in oldcsv.iterrows():
        file_id = row[1]['file'][:-4]
        if file_id in meta_data:
            doc = meta_data[file_id]
            value = meta_data[file_id]['action'][0].upper() + meta_data[file_id]['action'][1:]
            view = meta_data[file_id]['view']
            game = meta_data[file_id]['game']
            play = doc['play']
            col = doc['col']

            if view == "C" and "calibration" not in doc:
                continue
            values.append(value)
            views.append(view)
            games.append(game)
            plays.append(play)
            cols.append(col)
            true_camera_views.append(view_names[view])
        else:
            value = "Not Detected"
    oldcsv['action'] = values
    oldcsv['view'] = views
    oldcsv['game'] = games
    oldcsv['col'] = cols
    oldcsv['play'] = plays
    return oldcsv
if __name__ == '__main__':
    load_meta_from_mongo()
    # process_csv()
    # save_remote_db_to_local()
    # data = load_meta()
    # with open("meta.json", "w") as fp:
    #     json.dump(data, fp)
    # load_meta()
# meta_data = load_meta_by_json()