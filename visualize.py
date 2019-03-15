from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import random
import pylab
import numpy as np
import cv2

img = cv2.imread("./test.jpg")
im_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dets = np.load('./bbox.npy')

figure = pylab.figure()
pylab.imshow(im_array)
figure.suptitle('DFace Detector', fontsize=20)

print (dets)

for i in range(dets.shape[0]):
    bbox = dets[i, :4]
    rect = pylab.Rectangle((bbox[0], bbox[1]),
                             bbox[2] - bbox[0],
                             bbox[3] - bbox[1], fill=False,
                             edgecolor='yellow', linewidth=0.9)
    pylab.gca().add_patch(rect)

pylab.show()
