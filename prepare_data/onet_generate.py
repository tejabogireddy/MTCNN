import os
import cv2
import time
import pickle as cPickle
import numpy as np
from utils.iou import IoU
from utils.convert_to_square import convert_to_square
from utils.data_loader import DataLoader
from utils.image_loader import TestLoader
from utils.detector import network_loader, MTCNNDetector

p_model_path = './model_store/pnet_epoch_9.pt'
r_model_path = './model_store/rnet_epoch_9.pt'
annotation_file = "../Datasets/WIDER_train/wider_origin_anno.txt"
onet_directory = './prepare_data/Data/ONet/'
positives_directory = './prepare_data/Data/ONet/Positive'
negative_directory = './prepare_data/Data/ONet/Negative'
partial_directory = './prepare_data/Data/ONet/Partial'

if not os.path.exists(positives_directory):
    os.mkdir(positives_directory)
if not os.path.exists(negative_directory):
    os.mkdir(negative_directory)
if not os.path.exists(partial_directory):
    os.mkdir(partial_directory)

p_network, r_network, _ = network_loader(p_model_path=p_model_path, r_model_path=r_model_path)
mtcnn_detector = MTCNNDetector(p_network=p_network, r_network=r_network, min_face_size=12)

dataloader = DataLoader('../Datasets/WIDER_train/wider_origin_anno.txt', mode='Test')
data = dataloader.load_data()
test_data = TestLoader(data, 1, False)

batch_index = 0
final_boxes = []

def generate_sample_data_onet(annotation_file, pnet_file):
    image_size = 48
    image_path_list, bbox_list = [], []

    with open(annotation_file, 'r') as f:
        annotations = f.readlines()

    training_images = len(annotations)

    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        image_path = '../Datasets/WIDER_train/images/'\
                     + annotation[0]
        boxes = list(map(float, annotation[1:]))
        boxes = np.array(boxes, dtype=np.int32).reshape(-1, 4)
        image_path_list.append(image_path)
        bbox_list.append(boxes)

    if not os.path.exists(onet_directory):
        os.makedirs(onet_directory)

    f1 = open(os.path.join(onet_directory, 'positive.txt'), 'w')
    f2 = open(os.path.join(onet_directory, 'negative.txt'), 'w')
    f3 = open(os.path.join(onet_directory, 'partial.txt'), 'w')

    load_onet_detections = open('./model_store/detections_onet.pkl', 'rb')
    onet_boxes = cPickle.load(load_onet_detections)

    positives_samples, negative_samples, partial_samples, count = 0, 0, 0, 0
    for image_idx, onet_box, bbox in zip(image_path_list, onet_boxes, bbox_list):
        if onet_box.shape[0] == 0:
            continue

        image = cv2.imread(image_idx)
        print (((count)/training_images) * 100, negative_samples)
        count = count + 1
        onet_box = convert_to_square(onet_box)
        onet_box[:, 0:4] = np.round(onet_box[:, 0:4])

        for box in onet_box:
            x_left, y_top, x_right, y_bottom = box[0:4].astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            if width < 20 or x_left < 0 or y_top < 0 or x_right > image.shape[1] - 1 or y_bottom > image.shape[0] - 1:
                continue

            Iou = IoU(box, bbox)
            cropped_im = image[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size),
                                    interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                image_file = os.path.join(negative_directory, "%s.jpg" % negative_samples)
                f2.write(image_file + ' 0\n')
                cv2.imwrite(image_file, resized_im)
                negative_samples = negative_samples + 1
            else:
                idx = np.argmax(Iou)
                assigned_gt = bbox[idx]
                x1, y1, x2, y2 = assigned_gt

                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                if np.max(Iou) >= 0.65:
                    image_file = os.path.join(positives_directory, "%s.jpg" % positives_samples)
                    f1.write(image_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(image_file, resized_im)
                    positives_samples = positives_samples + 1

                elif np.max(Iou) >= 0.4:
                    image_file = os.path.join(partial_directory, "%s.jpg" % partial_samples)
                    f3.write(image_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(image_file, resized_im)
                    partial_samples = partial_samples + 1
    f1.close()
    f2.close()
    f3.close()

counter = 0
for databatch in test_data:
    print ((counter/len(data))*100)
    counter = counter + 1

    im_input = databatch
    p_boxes, p_boxes_align = mtcnn_detector.detect_pnetwork(image_input=im_input)
    boxes, boxes_align = mtcnn_detector.detect_rnetwork(image_input=im_input, dets=p_boxes_align)

    if boxes_align is None:
        final_boxes.append(np.array([]))
        batch_index = batch_index + 1
        continue

    final_boxes.append(boxes_align)
    batch_index = batch_index + 1

save_file = os.path.join('./model_store/detections_onet.pkl')
with open(save_file, 'wb') as f:
    cPickle.dump(final_boxes, f, cPickle.HIGHEST_PROTOCOL)

#save_file = os.path.join('./model_store/detections_onet.pkl')
generate_sample_data_onet(annotation_file, save_file)
