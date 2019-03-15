import os
import numpy as np

class DataLoader(object):
    def __init__(self, combined_data_file_name, mode='Train'):
        self.combined_data_file_name = combined_data_file_name
        self.mode = mode

    def load_data(self):
        with open(self.combined_data_file_name, 'r') as f:
            annotations = f.readlines()

        data = []
        for sample in range(len(annotations)):
            data_dict = {}
            annotation = annotations[sample].strip().split(' ')
            if self.mode == 'Test':
                data_dict['image_path'] = '../Datasets/WIDER_train/images/' + annotation[0]
            else:
                data_dict['image_path'] = annotation[0]
                data_dict['label'] = int(annotation[1])
                data_dict['bbox'] = np.zeros((4,))
                if len(annotation[2:]) == 4:
                    data_dict['bbox'] = np.array(annotation[2:6]).astype(float)
            data.append(data_dict)
        return data
