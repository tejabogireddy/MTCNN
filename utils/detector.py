import os
import cv2
import torch
import numpy as np
from utils.nms import nms
from models.pnet import PNet
from models.rnet import RNet
from models.onet import ONet
import torchvision.transforms as transforms
from torch.autograd.variable import Variable

transform = transforms.ToTensor()


def network_loader(p_model_path=None, r_model_path=None, o_model_path=None):
    p_network, r_network, o_network = None, None, None
    if p_model_path is not None:
        p_network = PNet()
        p_network.load_state_dict(torch.load(p_model_path))
        p_network.eval()

    if r_model_path is not None:
        r_network = RNet()
        r_network.load_state_dict(torch.load(r_model_path))
        r_network.eval()

    if o_model_path is not None:
        o_network = ONet()
        o_network.load_state_dict(torch.load(o_model_path))
        o_network.eval()

    return p_network, r_network, o_network


class MTCNNDetector(object):
    def __init__(self, p_network=None, r_network=None, o_network=None,
                 min_face_size=12, stride=2, threshold=[0.6, 0.7, 0.7], scale_factor=0.709):
        self.p_network = p_network
        self.r_network = r_network
        self.o_network = o_network
        self.min_face_size = min_face_size
        self.stride = stride
        self.threshold = threshold
        self.scale_factor = scale_factor

    def resize_image(self, image, scale):
        height, width, channels = image.shape
        new_height = int(height * scale)
        new_width = int(width * scale)
        new_dim = (new_width, new_height)
        image_resized = cv2.resize(image, new_dim, interpolation=cv2.INTER_LINEAR)
        return image_resized

    def convert_to_tensor(self, image):
        image = image.astype(np.float)
        return transform(image)

    def generate_bounding_boxes(self, detection, bbox, scale, threshold):
        stride = 2
        cellsize = 12
        threshold_index = np.where(detection > threshold)

        if threshold_index[0].size == 0:
            return np.array([])

        dx1, dy1, dx2, dy2 = [bbox[0, threshold_index[0], threshold_index[1], i] for i in range(4)]
        bbox = np.array([dx1, dy1, dx2, dy2])
        score = detection[threshold_index[0], threshold_index[1], 0]

        boundingbox = np.vstack([np.round((stride * threshold_index[1]) / scale),
                                 np.round((stride * threshold_index[0]) / scale),
                                 np.round((stride * threshold_index[1] + cellsize) / scale),
                                 np.round((stride * threshold_index[0] + cellsize) / scale),
                                 score,
                                 bbox,
                                 ])

        return boundingbox.T

    def square_bbox(self, bbox): 
        square_bbox = bbox.copy()

        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        
        l = np.maximum(h,w)
        
        square_bbox[:, 0] = bbox[:, 0] + w*0.5 - l*0.5
        square_bbox[:, 1] = bbox[:, 1] + h*0.5 - l*0.5

        square_bbox[:, 2] = square_bbox[:, 0] + l - 1
        square_bbox[:, 3] = square_bbox[:, 1] + l - 1
        
        return square_bbox

    def pad(self, bboxes, w, h):
        tmpw = (bboxes[:, 2] - bboxes[:, 0] + 1).astype(np.int32)
        tmph = (bboxes[:, 3] - bboxes[:, 1] + 1).astype(np.int32)
        numbox = bboxes.shape[0]

        dx = np.zeros((numbox, ))
        dy = np.zeros((numbox, ))
        edx, edy  = tmpw.copy()-1, tmph.copy()-1

        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        tmp_index = np.where(ex > w-1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h-1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list

    def detect_pnetwork(self, image_input):
        image_height, image_width, image_channels = image_input.shape
        net_size = 12
        final_boxes = []

        current_scale = float(net_size)/self.min_face_size
        image_resized = self.resize_image(image_input, current_scale)
        current_height, current_weight, _ = image_resized.shape

        while min(current_height, current_weight) > net_size:
            image_list = []
            image_resized_tensor = self.convert_to_tensor(image_resized)
            image_list.append(image_resized_tensor)
            image_list = torch.stack(image_list)
            image_list = Variable(image_list)

            detection, bbox = self.p_network(image_list.float())

            detection = np.transpose(detection.data.numpy(), (0, 2, 3, 1))
            bbox = np.transpose(bbox.data.numpy(), (0, 2, 3, 1))

            boxes = self.generate_bounding_boxes(detection[0, :, :], bbox,
                                                 current_scale, self.threshold[0])
            current_scale *= self.scale_factor
            image_resized =  self.resize_image(image_input, current_scale)
            current_height, current_weight, _ = image_resized.shape

            if (boxes.size) == 0:
                continue

            keep = nms(boxes[:, :5], 0.5, 'Union')
            boxes = boxes[keep]
            final_boxes.append(boxes)

        if len(final_boxes) == 0:
            return None, None

        final_boxes = np.vstack(final_boxes)

        keep = nms(final_boxes[:, 0:5], 0.7, 'Union')
        final_boxes = final_boxes[keep]

        bw = final_boxes[:, 2] - final_boxes[:, 0] + 1
        bh = final_boxes[:, 3] - final_boxes[:, 1] + 1

        boxes = np.vstack([final_boxes[:, 0],
                           final_boxes[:, 1],
                           final_boxes[:, 2],
                           final_boxes[:, 3],
                           final_boxes[:, 4],
                           ])

        boxes = boxes.T

        align_topx = final_boxes[:, 0] + final_boxes[:, 5] * bw
        align_topy = final_boxes[:, 1] + final_boxes[:, 6] * bh
        align_bottomx = final_boxes[:, 2] + final_boxes[:, 7] * bw
        align_bottomy = final_boxes[:, 3] + final_boxes[:, 8] * bh

        boxes_align = np.vstack([align_topx,
                                 align_topy,
                                 align_bottomx,
                                 align_bottomy,
                                 final_boxes[:, 4],
                                 ])
        boxes_align = boxes_align.T
        return boxes, boxes_align
    
    def detect_rnetwork(self, image_input, dets):
        image_height, image_width, image_channels = image_input.shape

        if dets is None:
           return None, None

        dets = self.square_bbox(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, image_width, image_height)
        num_boxes = dets.shape[0]

        cropped_ims_tensors = []
        for i in range(num_boxes):
            try:
               tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
               tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = image_input[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
               crop_im = cv2.resize(tmp, (24, 24))
               crop_im_tensor = self.convert_to_tensor(crop_im)
               cropped_ims_tensors.append(crop_im_tensor)
            except:
               continue

        feed_imgs = Variable(torch.stack(cropped_ims_tensors))

        detection, bbox = self.r_network(feed_imgs.float())

        detection = detection.data.numpy()
        bbox = bbox.data.numpy()

        keep_inds = np.where(detection > self.threshold[1])[0]

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            detection = detection[keep_inds]
            bbox = bbox[keep_inds]
        else:
            return None, None

        keep = nms(boxes, 0.7)

        if len(keep) == 0:
            return None, None

        keep_detection = detection[keep]
        keep_boxes = boxes[keep]
        keep_bbox = bbox[keep]


        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1


        boxes = np.vstack([keep_boxes[:, 0],
                           keep_boxes[:, 1],
                           keep_boxes[:, 2],
                           keep_boxes[:, 3],
                           keep_detection[:, 0]
                            ])

        align_topx = keep_boxes[:, 0] + keep_bbox[:, 0] * bw
        align_topy = keep_boxes[:, 1] + keep_bbox[:, 1] * bh
        align_bottomx = keep_boxes[:, 2] + keep_bbox[:, 2] * bw
        align_bottomy = keep_boxes[:, 3] + keep_bbox[:, 3] * bh

        boxes_align = np.vstack([align_topx,
                                 align_topy,
                                 align_bottomx,
                                 align_bottomy,
                                 keep_detection[:, 0]])


        boxes = boxes.T
        boxes_align = boxes_align.T

        return boxes, boxes_align

    def detect_onetwork(self, image_input, dets):
        image_height, image_width, image_channels = image_input.shape

        if dets is None:
           return None, None

        dets = self.square_bbox(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, image_width, image_height)
        num_boxes = dets.shape[0]

        cropped_ims_tensors = []
        for i in range(num_boxes):
            try:
               tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
               tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = image_input[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
               crop_im = cv2.resize(tmp, (48, 48))
               crop_im_tensor = self.convert_to_tensor(crop_im)
               cropped_ims_tensors.append(crop_im_tensor)
            except:
               continue

        try:
           feed_imgs = Variable(torch.stack(cropped_ims_tensors))
        except:
           return None, None

        detection, bbox = self.o_network(feed_imgs.float())

        detection = detection.data.numpy()
        bbox = bbox.data.numpy()

        keep_inds = np.where(detection > self.threshold[2])[0]

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            detection = detection[keep_inds]
            bbox = bbox[keep_inds]
        else:
            return None, None

        keep = nms(boxes, 0.7, mode="Minimum")

        if len(keep) == 0:
            return None, None

        keep_detection = detection[keep]
        keep_boxes = boxes[keep]
        keep_bbox = bbox[keep]


        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1

        align_topx = keep_boxes[:, 0] + keep_bbox[:, 0] * bw
        align_topy = keep_boxes[:, 1] + keep_bbox[:, 1] * bh
        align_bottomx = keep_boxes[:, 2] + keep_bbox[:, 2] * bw
        align_bottomy = keep_boxes[:, 3] + keep_bbox[:, 3] * bh

        boxes = np.vstack([align_topx,
                           align_topy,
                           align_bottomx,
                           align_bottomy,
                           keep_detection[:, 0]
                           ])
        boxes_align = boxes.T

        return boxes_align

    def detect_face(self,img):
        boxes_align = np.array([])

        if self.p_network:
            boxes, boxes_align = self.detect_pnetwork(img)
            if boxes_align is None:
                return np.array([]), np.array([])

        if self.r_network:
            boxes, boxes_align = self.detect_rnetwork(img, boxes_align)
            if boxes_align is None:
                return np.array([]), np.array([])

        if self.o_network:
            boxes_align = self.detect_onetwork(img, boxes_align)
            if boxes_align is None:
                return np.array([])

        return boxes_align
