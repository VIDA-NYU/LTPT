from pymongo import MongoClient
import json
import streamlit as st

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
meta_data = {}
with open("./meta.json", "r") as fp:
     meta_data = json.load(fp)
def load_meta_by_json():
    return meta_data
def load_meta():
    client = MongoClient("mongodb+srv://guande:guandemongo@ltpt.qsvio.mongodb.net")
    db = client.ltpt
    data = {}
    pose_ids = []
    files_with_pose = []
    count = 0
    for doc in db['videos'].find({}):
        file_id = doc['filepath'].split("/")[-1][:-4]
        file_id = file_id[:-1] + "_" +file_id[-1]
        obj = {
            "file_id": file_id
        }
        data[file_id] = obj
        if "poses" in doc:
            pose_ids.append(doc['pose_id'])
            files_with_pose.append(file_id)
        count += 1

    pose_docs = list(db['poses'].find({
        "_id": {
            "$in": pose_ids
        }
    }))
    for i, pos_doc in enumerate(pose_docs):
        pose_data = extract_pose_data(pos_doc)
        file_doc = data[files_with_pose[i]]
        file_doc['pose_data'] = pose_data
    client.close()
    return data

def extract_pose_data(pose_doc):
    doc = pose_doc
    pose_data = []
    for kp_key in pose_kps:
        coordinate_keys = ["x", "y"]
        kp_data = []
        for coordinate_key in coordinate_keys:
            key = kp_key + "_" + coordinate_key
            frames = doc['poses']['Agent1'][key]
            kp_data.append(frames[len(frames) // 2])
        pose_data.append(kp_data)
    return pose_data
if __name__ == '__main__':
    data = load_meta()
    with open("meta.json", "w") as fp:
        json.dump(data, fp)
    # load_meta()