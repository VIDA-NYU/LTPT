import os
import numpy as np

import cv2
from tqdm import tqdm

import matplotlib.pyplot as plt

import math
import collections
# import copy

from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator_sports import TfPoseEstimator, PoseEstimator

from tf_pose import common

from process.utils import label_map_util
from process.utils import visualization_utils as vis_util

videoname = "3390C" #"3367B", "3370B","3387B","3390B"

resolution = "432x368"
iw, ih = 432, 368

## load cv2 images and pose estimation results

len_num_shards = 3
batch = 100

print("video",videoname)
print("len_num_shards",len_num_shards)

cv2_images = []
peak_np = []
heatmat_np = []
pafmat_np = []

for t in tqdm(range(len_num_shards)):
    numpy_path = os.path.join("/media/felicia/Data/sportsvideos/results", "{:s}_heatmap_{:02d}.npy".format(videoname, t))
    numpy_dict = np.load(numpy_path, allow_pickle=True).item()

    images = numpy_dict["images"]  # cv2
    peaks = numpy_dict["peak"]
    heatMat_up = numpy_dict["heatmat"]
    pafMat_up = numpy_dict["pafmat"]

    st = t * batch
    ed = st + len(images)

    cv2_images.append(images)
    peak_np.append(peaks)
    heatmat_np.append(heatMat_up)
    pafmat_np.append(pafMat_up)

cv2_images = np.concatenate(cv2_images, axis=0)
peak_np = np.concatenate(peak_np, axis=0)
heatmat_np = np.concatenate(heatmat_np, axis=0)
pafmat_np = np.concatenate(pafmat_np, axis=0)

## load bbox

# bbox_filename = os.path.join("/media/felicia/Data/sportsvideos/results", '{:s}_bbox_players.npy'.format(videoname))
# bbox_dict = np.load(bbox_filename, allow_pickle=True).item()
#
# players_boxes = bbox_dict["boxes"]
# players_scores = bbox_dict["scores"]
# players_classes = bbox_dict["classes"]


## generate bbox mask

def visualize_ordered_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=False,
        max_boxes_to_draw=20,
        min_score_thresh=0.0,
        agnostic_mode=False,
        line_thickness=4,
        groundtruth_box_visualization_color='black',
        skip_scores=False,
        skip_labels=False):
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    categorical_colors = ['darkorchid', 'mediumorchid', 'violet', 'plum', 'mediumpurple',
                          'royalblue', 'deepskyblue', 'darkturquoise', 'paleturquoise', 'mediumspringgreen',
                          'lightseagreen', 'seagreen', 'olivedrab', 'darkkhaki', 'gold',
                          'moccasin', 'orange', 'darkorange', 'coral', 'orangered']  # 20

    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    if not max_boxes_to_draw:
        max_boxes_to_draw = len(boxes)
    for i in range(min(max_boxes_to_draw, len(boxes))):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i])
            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
            else:
                display_str = ''
            if not skip_labels:
                if not agnostic_mode:
                    if classes[i] in category_index.keys():
                        class_name = category_index[classes[i]]['name']
                    else:
                        class_name = 'N/A'
                    display_str = str(class_name)
            if not skip_scores:
                if not display_str:
                    display_str = '{}%'.format(int(100 * scores[i]))
                else:
                    display_str = '{}: {}%'.format(display_str, int(100 * scores[i]))

            box_to_display_str_map[box].append(display_str)
            if agnostic_mode:
                box_to_color_map[box] = 'DarkOrange'
            else:
                box_to_color_map[box] = categorical_colors[i]

    # Draw all boxes onto image.
    for box, color in box_to_color_map.items():
        left, right, top, bottom = box
        ymin, xmin, ymax, xmax = top, left, bottom, right
        vis_util.draw_bounding_box_on_image_array(
            image,
            ymin,
            xmin,
            ymax,
            xmax,
            color=color,
            thickness=line_thickness,
            display_str_list=box_to_display_str_map[box],
            use_normalized_coordinates=use_normalized_coordinates)

    return image


data_label = "mscoco_label_map.pbtxt"
PATH_TO_LABELS = os.path.join("data", data_label)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
IMAGE_SIZE = (12, 8)

oh, ow = peak_np.shape[1:3]

# nframes = len(cv2_images)
#
# for padding in [0,5,10,15]:
#     new_boxes = []
#     for i in tqdm(range(nframes)):
#         bgimg = cv2.cvtColor(cv2_images[i], cv2.COLOR_BGR2RGB)
#         tmp_box = []
#         for j in range(len(players_boxes[i])):
#             left, right, top, bottom = players_boxes[i][j]
#             gl, gr, gt, gb = left * ow, right * ow, top * oh, bottom * oh
#             gl, gr, gt, gb = max(gl - padding, 0), min(gr + padding, ow), max(gt - padding, 0), min(gb + padding, oh)
#             tmp_box.append([gl / ow, gr / ow, gt / oh, gb / oh])
#
#         new_boxes.append(tmp_box)
#
#         # visualize_ordered_boxes_and_labels_on_image_array(
#         #     bgimg,
#         #     tmp_box,
#         #     players_classes[i],
#         #     players_scores[i],
#         #     category_index,
#         #     use_normalized_coordinates=True,
#         #     max_boxes_to_draw=2,  # view B:2, view C"3
#         #     min_score_thresh=0.3,
#         #     line_thickness=1,
#         #     )
#
#         # fig, ax = plt.subplots(figsize=IMAGE_SIZE)
#         # ax.imshow(bgimg)
#         #
#         # output_filename = os.path.join("/media/felicia/Data/sportsvideos/vis", "{:s}_bbox_players_pad{:02d}_{:03d}.png".format(videoname,padding,i))
#         # plt.savefig(output_filename, doi=100)
#         # plt.close()
#
#     new_dict = {
#         "images": bbox_dict["images"],  # w*h, rgb
#         "boxes": new_boxes,
#         "scores": players_scores,
#         "classes": players_classes
#     }
#
#     output_filename = os.path.join("/media/felicia/Data/sportsvideos/results", '{}_bbox_players_pad{:02d}.npy'.format(videoname, padding))
#     with open(output_filename, 'wb') as file:
#         np.save(file, new_dict)

## compute human ,draw and save
padding = 15
bbox_filename = os.path.join("/media/felicia/Data/sportsvideos/results", '{:s}_bbox_players_pad{:02d}.npy'.format(videoname, padding))
bbox_dict = np.load(bbox_filename, allow_pickle=True).item()

players_boxes = bbox_dict["boxes"]
players_scores = bbox_dict["scores"]
players_classes = bbox_dict["classes"]

print("padding",padding)

nframes=len(players_boxes)

obj_mask = np.zeros((nframes, oh, ow))
for i in tqdm(range(nframes)):
    for j in range(len(players_boxes[i])):
        left, right, top, bottom = players_boxes[i][j]
        gl, gr, gt, gb = math.floor(left * ow), math.ceil(right * ow), math.floor(top * oh), math.ceil(bottom * oh)
        obj_mask[i,gt:gb,gl:gr].fill(1)

idx_mask=np.where(obj_mask<1)
peak_np[idx_mask]=0
heatmat_np[idx_mask]=0
pafmat_np[idx_mask]=0


def draw_humans(npimg, humans, imgcopy=False):
    if imgcopy:
        npimg = np.copy(npimg)
    image_h, image_w = npimg.shape[:2]
    centers = {}
    for human in humans:
        # draw point
        for i in range(common.CocoPart.Background.value):
            if i not in human.body_parts.keys():
                continue

            body_part = human.body_parts[i]
            center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
            centers[i] = center
            cv2.circle(npimg, center, 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)

        # draw line
        for pair_order, pair in enumerate(common.CocoPairsRender):
            if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                continue
            cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)
    return npimg


heatmap_max=np.amax(heatmat_np[:,:, :, :-1], axis=3)
humans_video = []
for i in tqdm(range(nframes)):
    humans = PoseEstimator.estimate_paf(peak_np[i], heatmat_np[i], pafmat_np[i])
    humans_video.append(humans)

    img=cv2_images[i]
    skeleton = draw_humans(img, humans, imgcopy=False)
    # cv2.imwrite("/media/felicia/Data/sportsvideos/vis/{:s}_skeleton_player_pad{:02d}_{:03d}.png".format(videoname,padding,i),skeleton)

    heat = cv2.cvtColor(heatmap_max[i], cv2.COLOR_GRAY2RGB)
    heat = cv2.resize(heat, (iw, ih))
    heat = heat - heat.min() # Now between 0 and 8674
    heat = heat / heat.max() * 255
    heat = np.uint8(heat)

    heat = cv2.addWeighted(img,0.5,heat,0.5,0,dtype=cv2.CV_8UC1)
    cv2.imwrite("/media/felicia/Data/sportsvideos/vis/{:s}_heatmat_player_pad{:02d}_{:03d}.png".format(videoname,padding,i),heat)


new_dict = {
    "images": bbox_dict["images"],  # w*h, rgb
    "humans": humans_video,
}

output_filename = os.path.join("/media/felicia/Data/sportsvideos/results", '{}_skeleton_players_pad{:02d}.npy'.format(videoname, padding))
with open(output_filename, 'wb') as file:
    np.save(file, new_dict)