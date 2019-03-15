import os
import numpy as np


def IoU(random_box, target_boxes):
    try:
       random_box_area = (random_box[3] - random_box[1] + 1) * (random_box[2] - random_box[0] + 1)
       areas = (target_boxes[:, 3] - target_boxes[:, 1] + 1) * (target_boxes[:, 2] - target_boxes[:, 0] + 1)

       overlap_x1 = np.maximum(random_box[0], target_boxes[:, 0])
       overlap_y1 = np.maximum(random_box[1], target_boxes[:, 1])
       overlap_x2 = np.minimum(random_box[2], target_boxes[:, 2])
       overlap_y2 = np.minimum(random_box[3], target_boxes[:, 3])
       print (random_box, target_boxes)
       print (overlap_x1, overlap_y1, overlap_x2, overlap_y2)

       overlap_height = np.maximum(0, overlap_x2 - overlap_x1 + 1)
       overlap_width = np.maximum(0, overlap_y2 - overlap_y1 + 1)

       overlap_area = overlap_height * overlap_width
       iou = np.true_divide(overlap_area, (random_box_area + areas - overlap_area))
       return iou
    except:
       return 0
