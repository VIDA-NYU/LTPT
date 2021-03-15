import os
import numpy as np

import cv2
from tqdm import tqdm


import math
import time

from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator_sports import TfPoseEstimator


model="mobilenet_thin" # "mobilenet_v2_large", "cmu"

# videodir = "/media/felicia/Data/sportsvideos/AFL_Video/181101_AFL-PEORIA_AFL-SCOTTSDALE__0/"
videodir = "/run/user/1000/gvfs/smb-share:server=storage.rcs.nyu.edu,share=sportsvideos/TexasRangers/AFL Video/181101_AFL-PEORIA_AFL-SCOTTSDALE__0/"

videoname = "3390C" #"3367C","3370B","3387B","3390B"
videopath = os.path.join(videodir, videoname + ".mp4")

resolution="432x368"
show_process=False
resize_out_ratio= 4.0# default:4.0
showBG=True
w, h = model_wh(resolution)


cap = cv2.VideoCapture(videopath)
images_all=[]
count=0
stride=1
while True:
    ret_val, image = cap.read()
    if not ret_val:
        break
    if count%stride==0:
        img=cv2.resize(image, (w,h), interpolation=cv2.INTER_CUBIC)
        images_all.append(img)
    count+=1
images_all=np.stack(images_all,axis=0)
nframes=len(images_all)


e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

batch=100
nbatch=math.ceil(nframes/batch)

print("video",videoname)
print("image size",images_all.shape)
print("number of frames", nframes)
print("len_num_shards",nbatch)

start_time=time.time()

for t in tqdm(range(nbatch)):
    st=t*batch
    ed=t*batch+batch if t*batch+batch <nframes else nframes
    images=images_all[st:ed]
    peaks,heatMat_up,pafMat_up = e.inference(images, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)

    # numpy_dict={
    #     "images":images,
    #     "peak":peaks,
    #     "heatmat":heatMat_up,
    #     "pafmat":pafMat_up
    # }
    #
    # output_filename = os.path.join("/media/felicia/Data/sportsvideos/results","{:s}_heatmap_{:02d}.npy".format(videoname,t))

    # print(t,output_filename)
    #
    # with open(output_filename,'wb') as file:
    #   np.save(file,numpy_dict)

end_time=time.time()

print("inference time",end_time-start_time)

hours, rem = divmod(end_time-start_time, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

### load from numpy
# def draw_humans(npimg, humans, imgcopy=False):
#     if imgcopy:
#         npimg = np.copy(npimg)
#     image_h, image_w = npimg.shape[:2]
#     centers = {}
#     for human in humans:
#         # draw point
#         for i in range(common.CocoPart.Background.value):
#             if i not in human.body_parts.keys():
#                 continue
#
#             body_part = human.body_parts[i]
#             center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
#             centers[i] = center
#             cv2.circle(npimg, center, 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)
#
#         # draw line
#         for pair_order, pair in enumerate(common.CocoPairsRender):
#             if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
#                 continue
#             cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)
#     return npimg



# nbatch=3
# batch=100


# cv2_images=[]
# humans_video=[]
# heatmap_img=[]
# for t in tqdm(range(nbatch)):
#     numpy_path = os.path.join("results","{:s}_heatmap_{:02d}.npy".format(videoname,t))
#     numpy_dict=np.load(numpy_path,allow_pickle=True).item()

#     images= numpy_dict["images"]
#     peaks=numpy_dict["peak"]
#     heatMat_up=numpy_dict["heatmat"]
#     pafMat_up=numpy_dict["pafmat"]

#     heatmap_max=np.amax(heatMat_up[:,:, :, :-1], axis=3)

#     st=t*batch
#     ed=st+len(images)

#     cv2_images.append(images)

#     for i in tqdm(range(len(images))):
#         tmp1 = cv2.cvtColor(heatmap_max[i], cv2.COLOR_GRAY2RGB)
#         tmp1 = cv2.resize(tmp1, (w, h))
#         tmp1 = tmp1 - tmp1.min() # Now between 0 and 8674
#         tmp1 = tmp1 / tmp1.max() * 255
#         tmp1 = np.uint8(tmp1)

#         tmp1 = cv2.addWeighted(images[i],0.5,tmp1,0.5,0,dtype=cv2.CV_8UC1)
#         cv2.imwrite("vis/{:s}_heatmat_{:03d}.png".format(videoname,st+i),tmp1)
#         heatmap_img.append(tmp1)

#         humans = PoseEstimator.estimate_paf(peaks[i], heatMat_up[i], pafMat_up[i])
#         humans_video.append(humans)


# cv2_images=np.concatenate(cv2_images,axis=0)


# drawn_humans=[]
# nframes=len(cv2_images)
# for i in tqdm(range(nframes)):
#     img=cv2_images[i]
#     humans=humans_video[i]
#     img = draw_humans(img, humans, imgcopy=False)
#     cv2.imwrite("vis/{:s}_skeleton_{:03d}.png".format(videoname,i),img)
#     drawn_humans.append(img)