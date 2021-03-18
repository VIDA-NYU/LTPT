from pymongo import MongoClient
import json
# import streamlit as st

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
    collection_names = [ 'poses', 'actions']
    for collection_name in collection_names:
        collection = db[collection_name].find({})
        local_db[collection_name].insert_many(collection)
        print("saved:", collection_name)
# @st.cache
def load_meta_by_json():
    with open("./meta.json", "r") as fp:
        meta_data = json.load(fp)
        return meta_data
def load_meta():
    # client = MongoClient("mongodb+srv://guande:guandemongo@ltpt.qsvio.mongodb.net")
    client = MongoClient()

    print("db connected")
    db = client.ltpt
    data = {}
    pose_ids = []
    selected_video_ids = []
    files_with_pose = []
    count = 0
    videos = list(db['videos'].find({}))
    print("videos fetched")
    for doc in videos:
        file_id = doc['filepath'].split("/")[-1][:-4]
        file_id = file_id[:-1] + "_" +file_id[-1]
        obj = {
            "file_id": file_id
        }
        data[file_id] = obj
        if "poses" in doc:
            pose_ids.append(doc['pose_id'])
            files_with_pose.append(file_id)
            selected_video_ids.append(doc['_id'])
        count += 1
    print("videos processed")
    pose_docs = list(db['poses'].find({
        "_id": {
            "$in": pose_ids
        }
    }))
    print("pose fetched")

    action_docs = list(db['actions'].find({
        "$or": [
            {
                "actions.release.video_id": {
                    "$in": selected_video_ids
                }
            },
            {
                "actions.swing.video_id": {
                    "$in": selected_video_ids
                }
            }
        ]

    }))
    print("actions fetched")
    for i, action_doc in enumerate(action_docs):
        actions = extract_action_data(action_doc)
        for action in actions:
            file_doc = data[action['file_id']]
            file_doc['frame'] = action['frame']
            file_doc['action'] = action['action']
            # print(len(file_doc['pose_data']))
            # print(file_doc['frame'])
            # file_doc['pose_frame_data'] = file_doc['pose_data'][file_doc['frame']]
            # print(file_doc)
    filtered_data = {}
    for i, pos_doc in enumerate(pose_docs):
        pose_data = extract_pose_data(pos_doc)
        file_doc = data[files_with_pose[i]]
        if "frame" in file_doc:
            file_doc['pose_data'] = list(map( lambda x: list(map(lambda y: y[file_doc['frame']], x)), pose_data))
            filtered_data[file_doc['file_id']] = file_doc
    client.close()
    return filtered_data

def extract_action_data(action_doc):
    result = []
    for action_name in action_doc['actions']:
        action_data = action_doc['actions'][action_name]
        frame = action_data['frame']
        file_id = action_doc['play'] + "_" + action_data['view']
        data = {
            "file_id": file_id,
            "frame": frame,
            "action": action_name
        }
        result.append(data)
    return result
def extract_pose_data(pose_doc):
    doc = pose_doc
    pose_data = []
    for kp_key in kps_source:
        kp_key = convert_keypoint_name(kp_key)
        print(kp_key)
        coordinate_keys = ["x", "y"]
        kp_data = []
        for coordinate_key in coordinate_keys:
            key = kp_key + "_" + coordinate_key
            frames = doc['poses']['Agent1'][key]
            kp_data.append(frames)
        pose_data.append(kp_data)
    return pose_data
if __name__ == '__main__':
    # save_remote_db_to_local()
    data = load_meta()
    with open("meta.json", "w") as fp:
        json.dump(data, fp)
    load_meta()
meta_data = load_meta_by_json()
