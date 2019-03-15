import os
import cv2
import numpy as np
from utils.iou import IoU

annotation_file = "../Datasets/WIDER_train/wider_origin_anno.txt"
image_directory = "../Datasets/WIDER_train/images"
positives_directory = "./prepare_data/Data/PNet/Positives"
negative_directory = "./prepare_data/Data/PNet/Negatives"
partial_directory = "./prepare_data/Data/PNet/Partial"

f1 = open(os.path.join("./prepare_data/Data/PNet", 'positive.txt'), 'w')
f2 = open(os.path.join("./prepare_data/Data/PNet", 'negative.txt'), 'w')
f3 = open(os.path.join("./prepare_data/Data/PNet", 'partial.txt'), 'w')

if not os.path.exists(positives_directory):
    os.mkdir(positives_directory)
if not os.path.exists(negative_directory):
    os.mkdir(negative_directory)
if not os.path.exists(partial_directory):
    os.mkdir(partial_directory)

with open(annotation_file, 'r') as f:
    annotations = f.readlines()

negative_image_name, positive_image_name, partial_image_name, total_index = 0, 0, 0, 0


for annotation in annotations:
    image_path = annotation.strip().split(' ')[0]
    bbox = list(annotation.strip().split(' ')[1:])
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)

    image = cv2.imread(os.path.join(image_directory, image_path))
    image_height, image_width, image_channels = image.shape
    total_index = total_index + 1

    negative_samples, positive_samples, partial_samples = 0, 0, 0

    while negative_samples < 50:
        random_crop_size = np.random.randint(12, min(image_width, image_height) / 2)
        coordinate_x = np.random.randint(0, image_width - random_crop_size)
        coordinate_y = np.random.randint(0, image_height - random_crop_size)

        random_crop_box = np.array([coordinate_x, coordinate_y,
                                    coordinate_x + random_crop_size, coordinate_y + random_crop_size])
        iou = IoU(random_crop_box, boxes)
        cropped_im = image[coordinate_y: coordinate_y + random_crop_size,
                     coordinate_x: coordinate_x + random_crop_size, :]
        resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

        if np.max(iou) < 0.30:
            save_file = os.path.join(negative_directory, "%s.jpg" % negative_image_name)
            f2.write(negative_directory + "/" + "%s.jpg" % negative_image_name + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            negative_image_name += 1
            negative_samples += 1

    for box in boxes:
        box_width = int(box[2] - box[0] + 1)
        box_height = int(box[3] - box[1] + 1)

        if max(box_width, box_height) < 40 or box[0] < 0 or box[1] < 0:
            continue

        for sample in range(20):
            crop_size = np.random.randint(int(min(box_width, box_height) * 0.8),
                                          np.ceil(1.25 * max(box_width, box_height)))

            delta_x = np.random.randint(-box_width * 0.2, box_width * 0.2)
            delta_y = np.random.randint(-box_height * 0.2, box_height * 0.2)

            crop_box_x1 = int(max(box[0] + box_width/2 - crop_size/2 + delta_x, 0))
            crop_box_y1 = int(max(box[1] + box_height/2 - crop_size/2 + delta_y, 0))
            crop_box_x2 = int(crop_box_x1 + crop_size)
            crop_box_y2 = int(crop_box_y1 + crop_size)

            if crop_box_x2 > image_width or crop_box_y2 > image_height:
                continue

            crop_box = np.array([crop_box_x1, crop_box_y1, crop_box_x2, crop_box_y2])

            offset_x1 = (box[0] - crop_box_x1) / float(crop_size)
            offset_y1 = (box[1] - crop_box_y1) / float(crop_size)
            offset_x2 = (box[2] - crop_box_x2) / float(crop_size)
            offset_y2 = (box[3] - crop_box_y2) / float(crop_size)

            cropped_im = image[crop_box_y1:crop_box_y2, crop_box_x1:crop_box_x2, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            new_box = box.reshape(1, -1)
            iou = IoU(crop_box, new_box)

            if iou >= 0.65:
                save_file = os.path.join(positives_directory, "%s.jpg" % positive_image_name)
                f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                positive_image_name += 1
            elif iou >= 0.40:
                save_file = os.path.join(partial_directory, "%s.jpg" % partial_image_name)
                f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                partial_image_name += 1


        print("%s images done, positive: %s partial: %s negative: %s" % (total_index,
                                                                positive_image_name,
                                                                partial_image_name,
                                                                negative_image_name))

f1.close()
f2.close()
f3.close()
