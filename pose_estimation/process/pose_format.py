import pandas as pd
import ast
from tqdm import tqdm

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


## body parts of keypoints
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


## load data from csv
videoname = "3370C" #"3367C","3370B","3387B","3390B"
padding=0
pose_df=pd.read_csv("/media/felicia/Data/sportsvideos/results/{}_pose_pad{:02d}.csv".format(videoname,padding),header=0,index_col=0)
# pose_df=pd.read_csv("{}_pose.csv".format(videoname),header=0,index_col=0)
nhumans=1

print("video",videoname)
print("padding",padding)
print("nhumans",nhumans)


kps_csv=[]
for i in range(len(pose_df)):
    p0=ast.literal_eval(pose_df.iloc[i][0])

    kps_csv.append([p0])

    # p1=ast.literal_eval(pose_df.iloc[i][1])
    # kps_csv.append([p0,p1])



## Scatter plot of keypoints
iw, ih = 432, 368 # image size for openpose
nframes=len(pose_df)
full_kps=18
for i in tqdm(range(nframes)):
    fig, ax = plt.subplots()
    for j in range(nhumans):
        for p ,coor in enumerate(kps_csv[i][j]):
            if coor==[0,0]:
                continue
            x,y=coor
            ax.scatter(x*iw, y*ih)
            ax.annotate(p, (x*iw, y*ih))
    plt.xlim([0, iw])
    plt.ylim([0, ih])
    plt.gca().invert_yaxis()
    plt.savefig("/media/felicia/Data/sportsvideos/vis/{}_skeleton_scatter_pad{:02d}_{:03d}".format(videoname,padding,i), doi=100)
    # plt.show()
    plt.close()