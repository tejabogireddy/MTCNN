import os
import cv2
import numpy as np


class ImageLoader:
    def __init__(self, data, image_size, batch_size, shuffle):
        self.data = data
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.current_image = 0
        self.number_of_sample = len(data)
        self.index = np.arange(self.number_of_sample)
        self.num_classes = 2

        self.batch = None
        self.im_input = None
        self.label = None

        self.label_names = ['label', 'bbox']
        self.reset()
        self.get_batch()

    def reset(self):
        self.current_image = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.current_image + self.batch_size <= self.number_of_sample

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.current_image += self.batch_size
            return self.im_input, self.label
        else:
            raise StopIteration

    def getindex(self):
        return self.current_image / self.batch_size

    def getpad(self):
        if self.current_image + self.batch_size > self.number_of_sample:
            return self.current_image + self.batch_size - self.number_of_sample
        else:
            return 0

    def get_batch(self):
        current_from = self.current_image
        current_to = min(current_from + self.batch_size, self.number_of_sample)
        data = [self.data[self.index[i]] for i in range(current_from, current_to)]
        im_input, label = get_minibatch(data)
        self.im_input = im_input['data']
        self.label = [label[name] for name in self.label_names]


class TestLoader:
    def __init__(self, data, batch_size, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.number_of_samples = len(data)
        self.index = np.arange(self.number_of_samples)
        self.current_image = 0
        self.im_input = None
        self.label = None

        self.reset()
        self.get_batch()

    def reset(self):
        self.current_image = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.current_image + self.batch_size <= self.number_of_samples

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.current_image += self.batch_size
            return self.im_input
        else:
            raise StopIteration

    def getindex(self):
        return self.current_image / self.batch_size

    def getpad(self):
        if self.current_image + self.batch_size > self.number_of_samples:
            return self.current_image + self.batch_size - self.number_of_samples
        else:
            return 0

    def get_batch(self):
        cur_from = self.current_image
        cur_to = min(cur_from + self.batch_size, self.number_of_samples)
        data = [self.data[self.index[i]] for i in range(cur_from, cur_to)]
        im_input = get_testbatch(data)
        self.im_input = im_input['data']


def get_testbatch(data):
    assert len(data) == 1, "Single batch only"
    im = cv2.imread(data[0]['image_path'])
    image = {'data': im}
    return image


def get_minibatch(data):
    num_images = len(data)
    processed_ims = list()
    cls_label = list()
    bbox_reg_target = list()
    landmark_reg_target = list()

    for i in range(num_images):
        im = cv2.imread(data[i]['image_path'])

        cls = data[i]['label']
        bbox_target = data[i]['bbox']

        processed_ims.append(im)
        cls_label.append(cls)
        bbox_reg_target.append(bbox_target)

    im_array = np.asarray(processed_ims)
    label_array = np.array(cls_label)
    bbox_target_array = np.vstack(bbox_reg_target)

    im_input = {'data': im_array}
    label = {'label': label_array,
             'bbox': bbox_target_array}

    return im_input, label

