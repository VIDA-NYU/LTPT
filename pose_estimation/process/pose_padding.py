import os
import numpy as np

import cv2
from tqdm import tqdm

import math
import csv
import pandas as pd
import ast

import matplotlib.pyplot as plt

videoname = "3370C" #"3367C","3370B","3387B","3390B"
nhumans=1 # C:1
padding=0 # 5 is safer, 0 is more accurate

resolution = "432x368"
iw, ih = 432, 368

print("video",videoname)
print("padding",padding)

# kps_source={
#     'Nose': 0,
#     'Neck': 1,
#     'RShoulder': 2,
#     'RElbow':3,
#     'RWrist':4,
#     'LShoulder':5,
#     'LElbow':6,
#     'LWrist':7,
#     'RHip':8,
#     'RKnee': 9,
#     'RAnkle':10,
#     'LHip':11,
#     'LKnee':12,
#     'LAnkle':13,
#     'REye': 14,
#     'LEye': 15,
#     'REar': 16,
#     'LEar' :17,
#     'Background': 18,
# }


kps_filename = os.path.join("/media/felicia/Data/sportsvideos/results", "{:s}_skeleton_players_pad{:02d}.npy".format(videoname, padding))
kps_dict = np.load(kps_filename, allow_pickle=True).item()

rgbimgs= kps_dict["images"]   # w*h, rgb
humans_video=  kps_dict["humans"]

full_kps=18

nframes=len(humans_video)

kps_all=[]
for i in range(nframes):
    frame_humans=[]
    for j in range(len(humans_video[i])):
        kps=humans_video[i][j].body_parts.keys()
        print(i,j,len(list(kps)))
        single_human=[]
        for p in kps:
            x,y=humans_video[i][j].body_parts[p].x,humans_video[i][j].body_parts[p].y
            single_human.append((p,[x,y]))
        frame_humans.append(single_human)
    kps_all.append(frame_humans)


kps_sorted=[]
for i in range(nframes):
    if len(kps_all[i])==1:
        kps_sorted.append(kps_all[i])
        continue

    cxs=[]
    for j in range(len(kps_all[i])):
        cx=np.mean([coor[1][0] for coor in kps_all[i][j]]) #j-th human:cx
        cxs.append(cx)
    lr_order=np.argsort(cxs)
    frame_humans=[]
    for idx in lr_order:
        frame_humans.append(kps_all[i][idx])
    kps_sorted.append(frame_humans)


kps_valid=[]
for i in range(nframes):
    if len(kps_sorted[i])<=nhumans:
        kps_valid.append(kps_sorted[i])
        continue
    ## sort in area
    areas=[]
    centers=[]
    for j in range(len(kps_sorted[i])):
        center=np.mean([coor[1] for coor in kps_sorted[i][j]],axis=0)
        centers.append(center)

        l=np.min([coor[1][0] for coor in kps_sorted[i][j]])
        r=np.max([coor[1][0] for coor in kps_sorted[i][j]])
        t=np.min([coor[1][1] for coor in kps_sorted[i][j]])
        b=np.max([coor[1][1] for coor in kps_sorted[i][j]])
        area=(r-l)*(b-t)
        areas.append(area)
    area_order=np.argsort(areas)[::-1] # max to min

    centers=np.array(centers)

    ## sort in number of kps
    nkps=[len(coors) for coors in kps_sorted[i]]
    ct_order=np.argsort(nkps)[::-1] # max to min

    # main_idx=ct_order[:2]
    # query_idx=ct_order[2:]

    main_idx=area_order[:nhumans]
    query_idx=area_order[nhumans:]

    # double check
    # bd_main=main_idx[1]
    # bd_query=query_idx[0]
    # if nkps[bd_main]-nkps[bd_query]<=5 and areas[bd_main]<areas[bd_query]: ## ori:<3
    #     main_idx[1],query_idx[0]=query_idx[0],main_idx[1]

    # combine humans
    for idx in query_idx:
        single_ct=centers[idx]
        dist=np.linalg.norm(centers[main_idx]-single_ct,axis=1)
        dist_order=main_idx[np.argmin(dist)]

        # key_query=set([coor[0] for coor in kps_sorted[i][idx]])
        # key_main=set([coor[0] for coor in kps_sorted[i][dist_order]])
        key_query={}
        for coor in kps_sorted[i][idx]:
            key_query[coor[0]]=coor[1]
        key_main={}
        for coor in kps_sorted[i][dist_order]:
            key_main[coor[0]]=coor[1]

        set_query=set(key_query.keys())
        set_main=set(key_main.keys())

        if len(set_main.intersection(set_query))==0:
            kps_sorted[i][dist_order]+=kps_sorted[i][idx]
        else:
            same=True
            for k in set_main.intersection(set_query):
                if key_query[k]!=key_main[k]:
                    same=False
            if same:
                for k in set_query.difference(set_main):
                    kps_sorted[i][dist_order] += [key_query[k]]
    frame_humans=[]
    for idx in main_idx:
        frame_humans.append(kps_sorted[i][idx])
    kps_valid.append(frame_humans)

kps_valid_sorted = [] # left to right
for i in range(nframes):
    if len(kps_valid[i]) == 1:
        kps_valid_sorted.append(kps_valid[i])
        continue

    cxs = []
    for j in range(len(kps_valid[i])):
        cx = np.mean([coor[1][0] for coor in kps_valid[i][j]])  # j-th human:cx
        cxs.append(cx)
    lr_order = np.argsort(cxs)
    frame_humans = []
    for idx in lr_order:
        frame_humans.append(kps_valid[i][idx])
    kps_valid_sorted.append(frame_humans)

# new_dict = {
#     "humans": kps_valid_sorted,
# }
#
# output_filename = os.path.join("/media/felicia/Data/sportsvideos/results", '{}_skeleton_players_valid_pad{:02d}.npy'.format(videoname, padding))
# with open(output_filename, 'wb') as file:
#     np.save(file, new_dict)

## plot keypoints
# kps_filename = os.path.join("/media/felicia/Data/sportsvideos/results", "{:s}_skeleton_players_valid.npy".format(videoname, padding))
# kps_valid = np.load(kps_filename, allow_pickle=True).item()
#
# kps_valid_sorted=  kps_valid["humans"]

# iw, ih = 432, 368
#
# for i in tqdm(range(nframes)):
#     fig, ax = plt.subplots()
#     for j in range(len(kps_valid_sorted[i])):
#         for p ,coor in enumerate(kps_valid_sorted[i][j]):
#             ax.scatter(coor[1][0]*iw, coor[1][1]*ih)
#             ax.annotate(coor[0], (coor[1][0]*iw, coor[1][1]*ih))
#     plt.xlim([0, iw])
#     plt.ylim([0, ih])
#     plt.gca().invert_yaxis()
#     plt.savefig("/media/felicia/Data/sportsvideos/vis_/{}_skeleton_scatter_pad{:02d}_{:03d}".format(videoname,padding,i), doi=100)
#     plt.close()


# iw, ih = 432, 368
#
# for i in tqdm(range(nframes)):
#     fig, ax = plt.subplots()
#     for j in range(len(kps_all[i])):
#         for p ,coor in enumerate(kps_all[i][j]):
#             ax.scatter(coor[1][0]*iw, coor[1][1]*ih)
#             ax.annotate(coor[0], (coor[1][0]*iw, coor[1][1]*ih))
#     plt.xlim([0, iw])
#     plt.ylim([0, ih])
#     plt.gca().invert_yaxis()
#     plt.savefig("/media/felicia/Data/sportsvideos/vis_/{}_skeleton_original_scatter_{:03d}".format(videoname,i), doi=100)
#     plt.close()


kps_csv=[]
kps_columns=[x for x in range(nhumans)]
for i in range(nframes):
    frame_humans=[]
    for j in range(nhumans):
        if j>=len(kps_valid_sorted[i]):
            single_human=[[0,0] for x in range(full_kps)]
        else:
            single_human=np.zeros((full_kps,2))
            for _,coor in enumerate(kps_valid_sorted[i][j]):
                single_human[coor[0]]=coor[1]
            single_human=single_human.tolist()
        frame_humans.append(single_human)
    kps_csv.append(frame_humans)


poses = pd.DataFrame(kps_csv, columns=kps_columns)
poses.to_csv("/media/felicia/Data/sportsvideos/results/{}_pose_pad{:02d}.csv".format(videoname,padding))

## load csv
# poses_=pd.read_csv("/media/felicia/Data/sportsvideos/results/{}_pose.csv".format(videoname),header=0,index_col=0)
#
# col0=poses_['0']
# a_list = ast.literal_eval(col0[0])

"""
Order: left to right (need to sort)

Issues:
1. single human is divided into two parts
2. player in background
3. 


solution:
0. sort cx, cy: cx in ascending order

1. sort in len ; area=(l-r)*(t-b) !!!
2. shortest: if all kps are not in its nearest neightbor(two main skeletons), combine else skip
"""