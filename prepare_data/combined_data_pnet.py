import os
import numpy as np

with open('./prepare_data/Data/PNet/positive.txt', 'r') as f:
    positive = f.readlines()

with open('./prepare_data/Data/PNet/negative.txt', 'r') as f:
    negative = f.readlines()

with open('./prepare_data/Data/PNet/partial.txt', 'r') as f:
    partial = f.readlines()

#with open('./prepare_data/Data/PNet/landmark_12_aug.txt', 'r') as f:
#   landmark = f.readlines()

destination_directory = "./prepare_data/Combined_Data/PNet"

if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

with open(os.path.join("./prepare_data/Combined_Data/PNet/train_pnet_combined.txt"), 'w') as f:

    neg_keep = np.random.choice(len(negative), size=len(negative), replace=True)
    pos_keep = np.random.choice(len(positive), size=len(positive), replace=True)
    part_keep = np.random.choice(len(partial), size=len(partial), replace=True)

    for index in pos_keep:
        f.write(positive[index])
    for index in neg_keep:
        f.write(negative[index])
    for index in part_keep:
        f.write(partial[index])

f.close()

