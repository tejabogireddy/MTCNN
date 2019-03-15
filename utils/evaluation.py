import cv2
import numpy as np
from utils.detector import network_loader, MTCNNDetector
import scipy.io
from utils.iou import IoU


path = '../Datasets/WIDER_val/images/'

def evaluate(mtcnn_detector, facebox_list, event_list, file_list, iou_threshold):
	data_total_iou = 0
	data_total_precision = 0
	count = 0

	for i in range(len(event_list)):
		folder_name = event_list[i][0][0]
		for j in range(len(file_list[i][0])):
			bboxs = facebox_list[i][0][j][0]
			total_gt_face = len(bboxs)
			image_name = file_list[i][0][j][0][0]
			image_path = path + folder_name + '/' + image_name + '.jpg'
			img = cv2.imread(image_path)
			img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			predicted_bboxs = mtcnn_detector.detect_face(img)
			total_iou, tp = 0, 0
			pred_dict = dict()
			for bbox in bboxs:
				max_iou_per_gt = 0
				k = 0
				count = count + 1
				bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
				for predicted_bbox in predicted_bboxs:
					try:
						if k not in pred_dict.keys():
							pred_dict[k] = 0
						predicted_bbox = np.asarray(predicted_bbox)
						pred_bbox = [[predicted_bbox[0], predicted_bbox[1], predicted_bbox[2], predicted_bbox[3]]]
						pred_bbox = np.asarray(pred_bbox)
						iou = IoU(bbox, pred_bbox)
						print (iou)
					except:
						continue
					if iou > max_iou_per_gt:
						max_iou_per_gt = iou
						print (max_iou_per_gt)
					if iou > pred_dict[k]:
						pred_dict[k] = iou
						print (pred_dict[k])
					k = k + 1
				total_iou = total_iou + max_iou_per_gt
			
			if total_gt_face != 0:
				if len(pred_dict.keys()) > 0:
					for l in pred_dict:
						if pred_dict[l] >= 0.5:
							tp += 1
					precision = float(tp) / float(total_gt_face)
				else:
					precision = 0

				image_average_iou = total_iou / total_gt_face
				image_average_precision = precision

				data_total_iou += image_average_iou
				data_total_precision += image_average_precision

	print (data_total_iou, data_total_precision)
	result = dict()
	result['average_iou'] = float(data_total_iou) / float(count)
	result['mean_average_precision'] = float(data_total_precision) / float(count)
	return result


if __name__ == "__main__":
	pnet, rnet, onet = network_loader(p_model_path="./model_store/pnet_epoch_9.pt",
                                          r_model_path="./model_store/rnet_epoch_9.pt",
                                          o_model_path="./model_store/onet_epoch_3.pt")
	mtcnn_detector = MTCNNDetector(p_network=pnet, r_network=rnet, o_network=onet, min_face_size=24)

	iou_threshold = 0.6

	mat = scipy.io.loadmat('../Datasets/eval_tools/ground_truth/wider_face_val.mat')
	hard_mat = scipy.io.loadmat('../Datasets/eval_tools/ground_truth/wider_hard_val.mat')
	medium_mat = scipy.io.loadmat('../Datasets/eval_tools/ground_truth/wider_medium_val.mat')
	easy_mat = scipy.io.loadmat('../Datasets/eval_tools/ground_truth/wider_easy_val.mat')

	facebox_list = mat['face_bbx_list']
	event_list = mat['event_list']
	file_list = mat['file_list']

	result = evaluate(mtcnn_detector, facebox_list, event_list, file_list, iou_threshold)
	print (result)
