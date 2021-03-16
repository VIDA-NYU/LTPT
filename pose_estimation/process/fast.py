import os
import numpy as np

import matplotlib.pyplot as plt

import collections
import copy


from process.utils import label_map_util
from process.utils import visualization_utils as vis_util


data_label='mscoco_label_map.pbtxt'
PATH_TO_LABELS = os.path.join("data", data_label)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
resolution = "432x368"
iw, ih = 432, 368

## mislabled
"""
video: 3387C--[308,310]
"""
videoname = "3387C"  # "3367B", "3370B","3387B","3390B"
err_frames = [308, 310]

print("video", videoname)

bbox_filename = os.path.join("/media/felicia/Data/sportsvideos/results", '{:s}_bbox_players.npy'.format(videoname))
bbox_dict = np.load(bbox_filename, allow_pickle=True).item()

bgimages_all = bbox_dict["images"]
players_boxes = bbox_dict["boxes"]
players_scores = bbox_dict["scores"]
players_classes = bbox_dict["classes"]

new_boxes = players_boxes
new_scores = players_scores
new_classes = players_classes

for f in err_frames:
    # print(new_scores[f-1])
    new_boxes[f][0]=np.mean([new_boxes[f-1][0],new_boxes[f+1][0]],axis=0).tolist()
    # print(new_scores[f])
    new_scores[f][0]=np.mean([new_scores[f-1][0],new_scores[f+1][0]],axis=0).tolist()
    new_classes[f]=new_classes[f-1]

valid_output = {
    'images':bgimages_all, # w*h
    'boxes': new_boxes,
    'scores': new_scores,
    'classes': new_classes
}

output_filename = os.path.join("/media/felicia/Data/sportsvideos/results",'%s_bbox_players.npy' % (videoname))

with open(output_filename, 'wb') as file:
    np.save(file, valid_output)



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

IMAGE_SIZE = (12, 8)
for f in err_frames:
    image_bbox=copy.deepcopy(bgimages_all[f])

    visualize_ordered_boxes_and_labels_on_image_array(
        image_bbox,
        new_boxes[f],
        new_classes[f],
        new_scores[f],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=20, # next time: change to 20
        min_score_thresh=0.3,
        line_thickness=1,
        )

    fig,ax=plt.subplots(figsize=IMAGE_SIZE)
    ax.imshow(image_bbox)

    output_filename = os.path.join("/media/felicia/Data/sportsvideos/vis", "{:s}_bbox_players_{:03d}.png".format(videoname,f))
    plt.savefig(output_filename,doi=100)
    plt.close()
