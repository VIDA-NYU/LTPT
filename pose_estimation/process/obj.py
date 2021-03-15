import numpy as np
import os
import sys
import cv2
import copy
import collections

import tensorflow as tf
from distutils.version import StrictVersion

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


from tqdm import tqdm

import matplotlib.pyplot as plt


def run_inference_for_minibatch_images(images, graph):
    """
    images: numpy.array- B*H *W *C
    """
    with graph.as_default():
        with tf.compat.v1.Session() as sess:
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name)
            image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: images})
    return output_dict


def valid_boxes_on_minibatch_images(boxes, classes, scores, max_boxes_to_draw, min_score_thresh, selected_class=None):
    valid_output = {}
    b = boxes.shape[0]

    minibatch_boxes = []
    minibatch_scores = []
    minibatch_classes = []
    for i in range(b):
        single_boxes = []
        single_scores = []
        single_classes = []
        for j in range(min(max_boxes_to_draw, boxes.shape[1])):
            if scores[i][j] > min_score_thresh and (not selected_class or classes[i][j] == selected_class):
                ymin, xmin, ymax, xmax = boxes[i][j]
                # (left, right, top, bottom) = (xmin * WIDTH, xmax * WIDTH, ymin * HEIGHT, ymax * HEIGHT)
                (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
                single_boxes.append([left, right, top, bottom])
                single_scores.append(scores[i][j])
                single_classes.append(classes[i][j])
        minibatch_boxes.append(single_boxes)
        minibatch_scores.append(single_scores)
        minibatch_classes.append(single_classes)
    return minibatch_boxes, minibatch_scores, minibatch_classes




gfile = tf.io.gfile

max_boxes = 20  # default: 10
min_score = .3


# videodir = "/media/felicia/Data/sportsvideos/AFL_Video/181101_AFL-PEORIA_AFL-SCOTTSDALE__0/"
videodir = "/run/user/1000/gvfs/smb-share:server=storage.rcs.nyu.edu,share=sportsvideos/TexasRangers/AFL Video/181101_AFL-PEORIA_AFL-SCOTTSDALE__0/"
videoname = "3390C" #"3367C","3370B","3387B","3390B"
videopath = os.path.join(videodir, videoname + ".mp4")

print("video",videoname)


w, h = 432, 368

cap = cv2.VideoCapture(videopath)
images_all = []
bgimages_all=[]
count = 0
stride = 1
while True:
    ret_val, image = cap.read()
    if not ret_val:
        break
    if count % stride == 0:
        img = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
        bgimg=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images_all.append(img)
        bgimages_all.append(bgimg)
    count += 1
images_all = np.stack(images_all, axis=0)
bgimages_all = np.stack(bgimages_all, axis=0)

nframes = len(images_all)


# What model to download.
MODEL_NAME = 'ssd_inception_v2_coco_2018_01_28'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = '/home/felicia/research/models-1.13.0-app/research/object_detection_app/pretrained/' + MODEL_NAME + '/frozen_inference_graph.pb'

data_label='mscoco_label_map.pbtxt'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join("data", data_label)

BATCH = 20

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


## Size, in inches, of the output images.

valid_boxes = []
valid_scores = []
valid_classes = []

nbatch = nframes // BATCH + (1 if nframes % BATCH > 0 else 0)

for j in tqdm(range(nbatch)):
    idx_st = j * BATCH
    idx_ed = (j + 1) * BATCH if (j + 1) * BATCH < nframes else nframes
    image_batch = bgimages_all[idx_st:idx_ed]  # B*H*W*C

    output_dict = run_inference_for_minibatch_images(image_batch, detection_graph)

    minibatch_boxes, minibatch_scores, minibatch_classes = valid_boxes_on_minibatch_images(
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        max_boxes,
        min_score,
        selected_class=1
    )

    valid_boxes += minibatch_boxes
    valid_scores += minibatch_scores
    valid_classes += minibatch_classes

valid_output = {
    'images':bgimages_all, # w*h
    'boxes': valid_boxes,
    'scores': valid_scores,
    'classes': valid_classes
}

output_filename = os.path.join("/media/felicia/Data/sportsvideos/results",'%s_bbox.npy' % (videoname))

with open(output_filename, 'wb') as file:
    np.save(file, valid_output)


#### load numpy

def visualize_ordered_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=0,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color='black',
    skip_scores=False,
    skip_labels=False):

    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    categorical_colors= ['darkorchid','mediumorchid','violet','plum','mediumpurple',
        'royalblue','deepskyblue','darkturquoise','paleturquoise','mediumspringgreen',
        'lightseagreen','seagreen','olivedrab','darkkhaki','gold',
        'moccasin','orange','darkorange','coral','orangered'] #20

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
                    display_str = '{}%'.format(int(100*scores[i]))
                else:
                    display_str = '{}: {}%'.format(display_str, int(100*scores[i]))

            box_to_display_str_map[box].append(display_str)
            if agnostic_mode:
                box_to_color_map[box] = 'DarkOrange'
            else:
                box_to_color_map[box] = categorical_colors[i]

    # Draw all boxes onto image.
    for box, color in box_to_color_map.items():
        left, right, top, bottom=box
        ymin, xmin, ymax, xmax = top,left, bottom,right
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


## load bbox
# numpy_path = os.path.join("/media/felicia/Data/sportsvideos/results","{:s}_bbox.npy".format(videoname))
# numpy_dict=np.load(numpy_path,allow_pickle=True).item()
#
# bgimages_all=numpy_dict["images"]
# valid_boxes= numpy_dict["boxes"]
# valid_scores=numpy_dict["scores"]
# valid_classes=numpy_dict["classes"]
# nframes=len(bgimages_all)


## visualize original bbox
# IMAGE_SIZE = (12, 8)
# for j in tqdm(range(nframes)):
#     image_bbox=copy.deepcopy(bgimages_all[j])
#
#     visualize_ordered_boxes_and_labels_on_image_array(
#         image_bbox,
#         valid_boxes[j],
#         valid_classes[j],
#         valid_scores[j],
#         category_index,
#         use_normalized_coordinates=True,
#         max_boxes_to_draw=20, # next time: change to 20
#         min_score_thresh=0.3,
#         line_thickness=1
#         )
#
#     fig,ax=plt.subplots(figsize=IMAGE_SIZE)
#     ax.imshow(image_bbox)
#
#     output_filename = os.path.join("/media/felicia/Data/sportsvideos/vis", "{:s}_bbox_{:03d}.png".format(videoname,j))
#     plt.savefig(output_filename,doi=100)
#     plt.close()



### filter bbox

def filter_boxes_on_minibatch_images(boxes, classes, scores, max_boxes_to_draw=2, min_score_thresh=0.3):
    player_boxes = []
    player_scores = []
    player_classes = []
    candi=[]
    for j in range(len(boxes)):
        if scores[j] > min_score_thresh:
            left, right, top, bottom = boxes[j]
            area=(right-left)*(bottom-top)
            if 0<area<0.25:
                candi.append((j,area))
    candi=sorted(candi,key=lambda x:x[1],reverse=True)
    for t in range(min(max_boxes_to_draw,len(candi))):
        idx,_=candi[t]
        left, right, top, bottom = boxes[idx]
        player_boxes.append([left, right, top, bottom])
        player_scores.append(scores[idx])
        player_classes.append(classes[idx])

    return player_boxes, player_scores, player_classes


players_boxes = []
players_scores = []
players_classes = []

for j in tqdm(range(nframes)):

    player_boxes, player_scores, player_classes = filter_boxes_on_minibatch_images(
        valid_boxes[j],
        valid_classes[j],
        valid_scores[j],
        max_boxes_to_draw=1 # default:2
    )

    players_boxes.append(player_boxes)
    players_scores.append(player_scores)
    players_classes.append(player_classes)


valid_output = {
    'images':bgimages_all, # w*h
    'boxes': players_boxes,
    'scores': players_scores,
    'classes': players_classes
}

output_filename = os.path.join("/media/felicia/Data/sportsvideos/results",'%s_bbox_players.npy' % (videoname))

with open(output_filename, 'wb') as file:
    np.save(file, valid_output)


IMAGE_SIZE = (12, 8)

for j in tqdm(range(nframes)):
    image_bbox=copy.deepcopy(bgimages_all[j])

    visualize_ordered_boxes_and_labels_on_image_array(
        image_bbox,
        players_boxes[j],
        players_classes[j],
        players_scores[j],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=20, # next time: change to 20
        min_score_thresh=0.3,
        line_thickness=1,
        )

    fig,ax=plt.subplots(figsize=IMAGE_SIZE)
    ax.imshow(image_bbox)

    output_filename = os.path.join("/media/felicia/Data/sportsvideos/vis", "{:s}_bbox_players_{:03d}.png".format(videoname,j))
    plt.savefig(output_filename,doi=100)
    plt.close()