import os
import numpy as np

with open('./prepare_data/Data/RNet/positive.txt', 'r') as f:
    positive = f.readlines()

with open('./prepare_data/Data/RNet/negative.txt', 'r') as f:
    negative = f.readlines()

with open('./prepare_data/Data/RNet/partial.txt', 'r') as f:
    partial = f.readlines()


destination_directory = "./prepare_data/Combined_Data/RNet"

if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

with open(os.path.join("./prepare_data/Combined_Data/RNet/train_rnet_combined_new.txt"), 'w') as f:

    neg_keep = np.random.choice(len(negative), size=400000, replace=True)
    pos_keep = np.random.choice(len(positive), size=len(positive), replace=True)
    part_keep = np.random.choice(len(partial), size=len(partial), replace=True)

    for index in pos_keep:
        f.write(positive[index])
    for index in neg_keep:
        f.write(negative[index])
    for index in part_keep:
        f.write(partial[index])

f.close()
