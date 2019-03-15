import cv2
import numpy as np
from utils.detector import network_loader, MTCNNDetector



if __name__ == '__main__':
    pnet, rnet, onet = network_loader(p_model_path="./model_store/pnet_epoch_9.pt",
                                      r_model_path="./model_store/rnet_epoch_6.pt",
                                      o_model_path="./model_store/onet_epoch_3.pt")
    mtcnn_detector = MTCNNDetector(p_network=pnet, r_network=rnet, o_network=onet, min_face_size=24)
    img = cv2.imread("./test.jpg")
    img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxs = mtcnn_detector.detect_face(img)
    print (bboxs)
    np.save('./bbox.npy', bboxs)
